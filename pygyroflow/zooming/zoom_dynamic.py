"""Dynamic zoom smoothing algorithms.

Port of Gyroflow's zoom_dynamic.rs. Provides two methods for smoothing
per-frame FOV values over time to prevent visible zoom pulsing:

  1. Gaussian filter: sliding window minimum + Gaussian convolution
  2. Envelope follower: bidirectional exponential smoothing

Both methods accept a time window (in seconds) that controls smoothness.
"""

from __future__ import annotations

import math
from enum import IntEnum

from pygyroflow.stabilization.compute_params import ComputeParams


class ZoomMethod(IntEnum):
    """Dynamic zoom smoothing method."""

    GaussianFilter = 0
    EnvelopeFollower = 1


def compute(
    compute_params: ComputeParams,
    fov_values: list[float],
    timestamps: list[tuple[int, float]],
    method: ZoomMethod,
) -> tuple[list[float], list[float]]:
    """Apply dynamic zoom smoothing to FOV values.

    Args:
        compute_params: Stabilization parameters.
        fov_values: Per-frame raw FOV scale factors.
        timestamps: List of (frame_index, timestamp_ms).
        method: Smoothing algorithm to use.

    Returns:
        (smoothed_fovs, minimal_fovs) where minimal_fovs is the unsmoothed
        input and smoothed_fovs is the time-filtered result.
    """
    window = compute_params.adaptive_zoom_window
    fov_minimal = list(fov_values)

    if method == ZoomMethod.GaussianFilter:
        frames = _get_frames_per_window(compute_params)

        fov_padded = _pad_edge(fov_values, (frames // 2, frames // 2))
        fov_min = _min_rolling(fov_padded, frames)
        fov_min_padded = _pad_edge(fov_min, (frames // 2, frames // 2))

        gaussian = _gaussian_window_normalized(frames, frames / 6.0)
        fov_values = _convolve(fov_min_padded, gaussian)

    elif method == ZoomMethod.EnvelopeFollower:
        fps = compute_params.scaled_fps if compute_params.scaled_fps > 0 else 30.0
        first_pass_alpha = 1.0 - math.exp(-(1.0 / fps) / window)
        second_pass_alpha = 1.0 - math.exp(-(1.0 / fps) / 0.2)

        fov_values = _envelope_follower(fov_values, first_pass_alpha)
        fov_values = _envelope_follower(fov_values, second_pass_alpha)

    return fov_values, fov_minimal


def _get_frames_per_window(compute_params: ComputeParams) -> int:
    """Convert time window to frame count (guaranteed odd)."""
    fps = compute_params.scaled_fps if compute_params.scaled_fps > 0 else 30.0
    frames = int(compute_params.adaptive_zoom_window * fps)
    if frames % 2 == 0:
        frames += 1
    return max(frames, 3)


def _min_rolling(a: list[float], window: int) -> list[float]:
    """Sliding window minimum filter.

    For each window position, takes the minimum value. This ensures the
    smoothed FOV never undershoots the minimum needed for any frame.
    """
    result = []
    for i in range(len(a) - window + 1):
        result.append(min(a[i:i + window]))
    return result


def _convolve(v: list[float], kernel: list[float]) -> list[float]:
    """1D convolution of signal with filter kernel."""
    result = []
    for i in range(len(v) - len(kernel) + 1):
        s = sum(x * y for x, y in zip(v[i:i + len(kernel)], kernel))
        result.append(s)
    return result


def _gaussian_window(width: int, std: float) -> list[float]:
    """Generate an unnormalized Gaussian window function."""
    half = width // 2
    sig2 = 2.0 * std * std
    return [math.exp(-(x * x) / sig2) for x in range(-half, half + 1)]


def _gaussian_window_normalized(m: int, std: float) -> list[float]:
    """Generate a normalized Gaussian window (sum = 1)."""
    w = _gaussian_window(m, std)
    total = sum(w)
    return [x / total for x in w]


def _pad_edge(arr: list[float], pad_to: tuple[int, int]) -> list[float]:
    """Pad array edges with first/last values for boundary continuity."""
    if not arr:
        return [0.0] * (pad_to[0] + pad_to[1])

    first = arr[0]
    last = arr[-1]

    new_arr = [first] * pad_to[0] + list(arr) + [last] * pad_to[1]
    return new_arr


def _envelope_follower(a: list[float], alpha: float) -> list[float]:
    """Bidirectional exponential envelope follower.

    Two-pass smoothing:
      1. Reverse scan: q = min(x, x*alpha + q*(1-alpha))
      2. Forward scan: same formula on the reversed result

    This creates a smooth envelope that never undershoots the input,
    with attack/release behavior controlled by alpha.

    Args:
        a: Input FOV values.
        alpha: Smoothing coefficient (0 < alpha <= 1).
            Larger = faster response, more abrupt transitions.

    Returns:
        Smoothed FOV values.
    """
    if not a:
        return []

    # Reverse pass
    q = a[-1]
    smoothed_rev = []
    for x in reversed(a):
        q = min(x, x * alpha + q * (1.0 - alpha))
        smoothed_rev.append(q)

    # Forward pass
    q = smoothed_rev[-1]
    smoothed = []
    for x in reversed(smoothed_rev):
        q = min(x, x * alpha + q * (1.0 - alpha))
        smoothed.append(q)

    return smoothed
