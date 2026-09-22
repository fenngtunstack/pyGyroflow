"""Dynamic zoom smoothing algorithms.

Port of Gyroflow's zoom_dynamic.rs. Provides two methods for smoothing
per-frame FOV values over time to prevent visible zoom pulsing:

  1. Gaussian filter: sliding window minimum + Gaussian convolution
  2. Envelope follower: bidirectional exponential smoothing

Both methods accept a time window (in seconds) that controls smoothness.
The window itself can be keyframed (``ZoomingSpeed``), and video speed can
scale it per frame (``video_speed_affects_zooming``) — then every window
operation runs per-timestamp with its own size (``zoom_dynamic.rs:24-69``),
instead of one static window for the whole clip.
"""

from __future__ import annotations

import logging
import math
from enum import IntEnum

from pygyroflow.stabilization.compute_params import ComputeParams
from pygyroflow.keyframes.types import KeyframeType

logger = logging.getLogger(__name__)


class _DataPerTimestamp:
    """Per-frame window parameters (``zoom_dynamic.rs:12-18``)."""

    __slots__ = ("fps", "window", "frames", "half_frames", "gaussian_window")

    def __init__(self, fps: float, window: float, frames: int) -> None:
        self.fps = fps
        self.window = window
        self.frames = frames
        self.half_frames = frames // 2
        self.gaussian_window = _gaussian_window_normalized(
            frames, float(frames) / 6.0
        )


class ZoomMethod(IntEnum):
    """Dynamic zoom smoothing method."""

    GaussianFilter = 0
    EnvelopeFollower = 1

    @classmethod
    def from_index(cls, value: int) -> "ZoomMethod":
        """``zooming/mod.rs:20-27``: an unknown method index logs an error
        and falls back to GaussianFilter instead of failing the render —
        a project written by a newer Gyroflow must still load."""
        try:
            return cls(value)
        except ValueError:
            logger.error("Invalid zooming method: %s", value)
            return cls.GaussianFilter


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
    keyframes = compute_params.keyframes

    speed_keyframed = keyframes.is_keyframed(KeyframeType.ZoomingSpeed)
    speed_affects = (
        compute_params.video_speed_affects_zooming
        and (
            compute_params.video_speed != 1.0
            or keyframes.is_keyframed(KeyframeType.VideoSpeed)
        )
    )

    if speed_keyframed or speed_affects:
        # Keyframed window (zoom_dynamic.rs:24-69): every timestamp gets its
        # own window — ZoomingSpeed keyframes scale it, video speed scales it
        # further when enabled. The frame COUNT still comes from the static
        # adaptive_zoom_window (upstream quirk: get_frames_per_window reads
        # the parameter, not the keyframed value).
        fps = compute_params.scaled_fps if compute_params.scaled_fps > 0 else 30.0
        max_window = 0
        data_per_timestamp: list[_DataPerTimestamp] = []
        for _frame, ts in timestamps:
            per_ts_window = keyframes.value_at_video_timestamp(
                KeyframeType.ZoomingSpeed, ts
            )
            if per_ts_window is None:
                per_ts_window = window
            if compute_params.video_speed_affects_zooming:
                vid_speed = keyframes.value_at_video_timestamp(
                    KeyframeType.VideoSpeed, ts
                )
                vid_speed = (
                    compute_params.video_speed
                    if vid_speed is None
                    else vid_speed
                )
                per_ts_window *= abs(vid_speed)
            frames = _get_frames_per_window(compute_params)
            max_window = max(frames, max_window)
            data_per_timestamp.append(
                _DataPerTimestamp(fps=fps, window=per_ts_window, frames=frames)
            )

        if method == ZoomMethod.GaussianFilter:
            max_window_half = max_window // 2
            fov_values_pad = _pad_edge(fov_values, (max_window_half, max_window_half))
            fov_min = _min_rolling_dynamic(
                fov_values_pad, max_window_half, data_per_timestamp
            )
            fov_min_pad = _pad_edge(fov_min, (max_window_half, max_window_half))
            fov_values = _convolve_dynamic(
                fov_min_pad, max_window_half, data_per_timestamp
            )
        else:
            second_pass_alpha = 1.0 - math.exp(-(1.0 / fps) / 0.2)
            fov_values = _envelope_follower(fov_values, data_per_timestamp, None)
            fov_values = _envelope_follower(
                fov_values, data_per_timestamp, second_pass_alpha
            )
        return fov_values, fov_minimal

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

        fov_values = _envelope_follower(fov_values, [], first_pass_alpha)
        fov_values = _envelope_follower(fov_values, [], second_pass_alpha)

    return fov_values, fov_minimal


def _get_frames_per_window(compute_params: ComputeParams) -> int:
    """Convert time window to frame count (guaranteed odd).

    No lower clamp: upstream (zoom_dynamic.rs) allows a window of one frame,
    and clamping to 3 silently widens any window below 3/fps — a sub-frame
    window is a legitimate way to say "barely smooth the FOV".
    """
    fps = compute_params.scaled_fps if compute_params.scaled_fps > 0 else 30.0
    frames = int(compute_params.adaptive_zoom_window * fps)
    if frames % 2 == 0:
        frames += 1
    return max(frames, 1)


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


def _min_rolling_dynamic(
    a: list[float], max_window_half: int, data_per_timestamp: list[_DataPerTimestamp]
) -> list[float]:
    """Sliding minimum with a per-timestamp window (``zoom_dynamic.rs:129-145``).

    Timestamp *di* owns the window centered on it: the slice starts at
    ``di + (max_window_half - half_frames)``, so smaller windows read less
    context than larger ones.
    """
    ret: list[float] = []
    for di, data in enumerate(data_per_timestamp):
        i = di + (max_window_half - data.half_frames)
        if 0 <= i and i + data.frames <= len(a):
            ret.append(min(a[i:i + data.frames]))
        else:
            logger.error(
                "min_rolling_dynamic: window out of range i=%s len(a)=%s frames=%s",
                i, len(a), data.frames,
            )
    return ret


def _convolve_dynamic(
    a: list[float], max_window_half: int, data_per_timestamp: list[_DataPerTimestamp]
) -> list[float]:
    """Convolution with a per-timestamp Gaussian (``zoom_dynamic.rs:147-167``)."""
    ret: list[float] = []
    for di, data in enumerate(data_per_timestamp):
        i = di + (max_window_half - data.half_frames)
        if 0 <= i and i + data.frames <= len(a):
            window = a[i:i + data.frames]
            if len(window) == len(data.gaussian_window):
                ret.append(
                    sum(x * y for x, y in zip(window, data.gaussian_window))
                )
            else:
                logger.error(
                    "convolve_dynamic: window %s vs filter %s",
                    len(window), len(data.gaussian_window),
                )
        else:
            logger.error(
                "convolve_dynamic: window out of range i=%s len(a)=%s frames=%s",
                i, len(a), data.frames,
            )
    return ret


def _envelope_follower(
    a: list[float],
    data_per_timestamp: list[_DataPerTimestamp],
    alpha: float | None,
) -> list[float]:
    """Bidirectional exponential envelope follower.

    Two-pass smoothing:
      1. Reverse scan: q = min(x, x*alpha + q*(1-alpha))
      2. Forward scan: same formula on the reversed result

    This creates a smooth envelope that never undershoots the input, with
    attack/release behavior controlled by alpha. A concrete ``alpha``
    applies to every sample (the static path); ``None`` derives one per
    sample from that timestamp's keyframed window
    (``zoom_dynamic.rs:169-189``).
    """
    if not a:
        return []

    if alpha is not None:
        alphas = [alpha] * len(a)
    else:
        alphas = [
            1.0 - math.exp(-(1.0 / d.fps) / d.window) for d in data_per_timestamp
        ]

    # Reverse pass. Upstream pairs a[i] with alphas[i] then iterates from the
    # end (`.zip(&alphas).rev()`).
    q = a[-1]
    smoothed_rev = []
    for x, coeff in zip(reversed(a), reversed(alphas)):
        q = min(x, x * coeff + q * (1.0 - coeff))
        smoothed_rev.append(q)

    # Forward pass. Here upstream walks smoothed_rev backwards but pairs it
    # with alphas *forwards* (`.iter().rev().zip(&alphas)`) — sample a[k]
    # uses alphas[k] in both passes.
    q = smoothed_rev[-1]
    smoothed = []
    for x, coeff in zip(reversed(smoothed_rev), alphas):
        q = min(x, x * coeff + q * (1.0 - coeff))
        smoothed.append(q)

    return smoothed
