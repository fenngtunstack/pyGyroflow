"""Focal length smoothing for zoom lenses.

Port of Gyroflow's ``smoothing/focal_length.rs``.

A zoom lens changes focal length while the clip runs, and cameras report that
length coarsely — often in steps of a whole millimetre. Two problems follow,
and the two functions here solve one each:

* **Quantization stairs.** The step changes are real discontinuities in the
  metadata. :func:`smooth_focal_lengths_gaussian` is the short-kernel pass that
  irons them out before anything else looks at the curve.
* **Jitter versus intent.** A hand tremor nudging the zoom ring and a
  deliberate zoom move look the same to a low-pass filter. The adaptive filter
  uses the *relative* rate of change to tell them apart: heavy smoothing while
  the length is steady, light smoothing once it is clearly moving, so a real
  zoom is tracked without lag and a tremor is flattened.

Both work on ``list[float | None]`` and preserve ``None`` positions — a frame
whose focal length the camera did not report stays unreported rather than
being invented from its neighbours.
"""

from __future__ import annotations

import math

__all__ = [
    "smooth_focal_lengths_gaussian",
    "smooth_focal_lengths_adaptive",
]


def smooth_focal_lengths_gaussian(
    focal_lengths: list[float | None],
    strength: float,
    window_size: int,
) -> list[float | None]:
    """Short-kernel Gaussian blur over the focal length curve.

    The dequantization pass: camera-quantized focal length values produce
    visible stairs, and the compensation ratio's denominator is this curve, so
    stairs here become stairs in the sampling position. A blur narrow enough to
    leave a real zoom move intact is enough to flatten the steps.

    Args:
        focal_lengths: Per-frame focal length in mm; ``None`` where unknown.
        strength: How far to move each sample toward the blurred value, in
            ``[0, 1]``. It also widens the kernel (``sigma`` grows with it).
            ``<= 0`` returns a copy unchanged.
        window_size: Kernel width in frames, forced odd. Callers pass roughly
            half a second of frames.

    Returns:
        Smoothed curve, same length as the input.

    The edge handling clamps indices to the ends (``.max(0).min(n-1)``), and a
    sample only averages over neighbours that exist — ``weight_sum`` counts
    only the ``Some`` ones, so a gap in the middle of a window does not drag
    the result toward zero.
    """
    if not focal_lengths or strength <= 0.0:
        return list(focal_lengths)

    if window_size % 2 == 0:
        window_size += 1
    half_window = window_size // 2

    sigma = (window_size / 6.0) * (1.0 + strength * 2.0)
    kernel: list[float] = []
    kernel_sum = 0.0
    for i in range(window_size):
        x = float(i - half_window)
        weight = math.exp(-x * x / (2.0 * sigma * sigma))
        kernel.append(weight)
        kernel_sum += weight
    kernel = [weight / kernel_sum for weight in kernel]

    count = len(focal_lengths)
    smoothed: list[float | None] = []
    for i in range(count):
        original = focal_lengths[i]
        if original is None:
            smoothed.append(None)
            continue

        weighted_sum = 0.0
        weight_sum = 0.0
        for j in range(window_size):
            index = min(max(i + j - half_window, 0), count - 1)
            value = focal_lengths[index]
            if value is not None:
                weighted_sum += value * kernel[j]
                weight_sum += kernel[j]

        if weight_sum > 0.0:
            # `original` is not None (checked above), so upstream's fallback
            # branch here — pushing the bare average — is unreachable.
            blurred = weighted_sum / weight_sum
            smoothed.append(original * (1.0 - strength) + blurred * strength)
        else:
            smoothed.append(original)

    return smoothed


def smooth_focal_lengths_adaptive(
    focal_lengths: list[float | None],
    fps: float,
    max_smoothness_time: float,
    min_smoothness_time: float,
    max_velocity: float,
) -> list[float | None]:
    """Velocity-adaptive exponential smoothing of the focal length.

    Same idea as :mod:`pygyroflow.smoothing.default_algo`: at low velocity use
    a long time constant (heavy smoothing, kills jitter), at high velocity use
    a short one (light smoothing, tracks the real zoom without lag). Two passes
    — forward then backward — cancel the phase shift, so the output stays
    aligned in time with the input.

    Args:
        focal_lengths: Per-frame focal length in mm; ``None`` where unknown.
        fps: Frame rate, for turning frame indices into seconds.
        max_smoothness_time: Time constant (seconds) at low velocity.
        min_smoothness_time: Time constant (seconds) at high velocity.
        max_velocity: Relative velocity (1/s) at which the filter is fully
            switched to ``min_smoothness_time``.

    Returns:
        Smoothed curve, same length as the input.

    Velocity is measured *relatively* — ``|Δfl| * fps / fl`` — so the threshold
    means the same thing across lenses: a 1 mm step on an 18 mm lens is a
    bigger event than 1 mm on a 200 mm lens.
    """
    count = len(focal_lengths)
    if count < 2 or fps <= 0.0:
        return list(focal_lengths)

    dt = 1.0 / fps
    alpha_max = 1.0 - math.exp(-dt / max(max_smoothness_time, 1e-3))
    alpha_min = 1.0 - math.exp(-dt / max(min_smoothness_time, 1e-3))

    velocity = [0.0] * count
    for i in range(1, count):
        previous = focal_lengths[i - 1]
        current = focal_lengths[i]
        if previous is not None and current is not None and previous > 0.0:
            velocity[i] = abs((current - previous) * fps / previous)
    velocity[0] = velocity[1]

    # Smooth the velocity signal itself, so a single noisy sample does not flip
    # the alpha for one frame.
    for i in range(1, count):
        velocity[i] = velocity[i - 1] * (1.0 - alpha_min) + velocity[i] * alpha_min
    for i in range(count - 2, -1, -1):
        velocity[i] = velocity[i + 1] * (1.0 - alpha_min) + velocity[i] * alpha_min

    def alpha_at(i: int) -> float:
        ratio = min(velocity[i] / max_velocity, 1.0) if max_velocity > 1e-6 else 1.0
        return alpha_max * (1.0 - ratio) + alpha_min * ratio

    # Seed the filter at the first frame that has a value.
    start = next(
        ((i, value) for i, value in enumerate(focal_lengths) if value is not None),
        None,
    )
    if start is None:
        return list(focal_lengths)
    start_index, seed = start

    smoothed: list[float | None] = [None] * count
    state = seed
    for i in range(start_index, count):
        value = focal_lengths[i]
        if value is not None:
            alpha = alpha_at(i)
            state = state * (1.0 - alpha) + value * alpha
        # A gap holds the last state; the backward pass picks it back up.
        smoothed[i] = state

    # Backward pass, seeded from the last forward state.
    state = smoothed[count - 1] if smoothed[count - 1] is not None else seed
    for i in range(count - 1, start_index - 1, -1):
        value = smoothed[i]
        if value is not None:
            alpha = alpha_at(i)
            state = state * (1.0 - alpha) + value * alpha
            smoothed[i] = state

    return smoothed
