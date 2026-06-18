"""Plain 3D smoothing -- first-order exponential filter with fixed time constant.

Port of Gyroflow's core/smoothing/plain.rs.

Uses a bidirectional (forward + backward) pass with SLERP to achieve
zero-phase-delay filtering. The time constant controls smoothness:
larger values = more smoothing, smaller values = more responsive.
"""

from __future__ import annotations

import copy
import math
from typing import Any

from pygyroflow.keyframes import KeyframeManager, KeyframeType
from pygyroflow.smoothing.base import SmoothingAlgorithm
from pygyroflow.smoothing.trim import get_trimmed_quats
from pygyroflow.types.quaternion import Quat64
from pygyroflow.types.time_types import TimeQuat


class PlainSmoothing(SmoothingAlgorithm):
    """Simple first-order exponential smoothing with bidirectional passes.

    Parameters:
        time_constant: Smoothing time constant in seconds (default 0.25).
            Larger = smoother, smaller = more responsive.
        trim_range_only: Only smooth within trim ranges (default True).
    """

    def __init__(self) -> None:
        self.time_constant: float = 0.25
        self.trim_range_only: bool = True

    def get_name(self) -> str:
        return "Plain 3D"

    def get_parameters_json(self) -> list[dict]:
        return [
            {
                "name": "time_constant",
                "description": "Smoothness",
                "type": "SliderWithField",
                "from": 0.01,
                "to": 10.0,
                "value": self.time_constant,
                "default": 0.25,
                "unit": "s",
                "keyframe": "SmoothingParamTimeConstant",
            },
            {
                "name": "trim_range_only",
                "description": "Only within trim range",
                "advanced": True,
                "type": "CheckBox",
                "default": self.trim_range_only,
                "value": 1.0 if self.trim_range_only else 0.0,
            },
        ]

    def set_parameter(self, name: str, val: float) -> None:
        if name == "time_constant":
            self.time_constant = val
        elif name == "trim_range_only":
            self.trim_range_only = val > 0.1

    def get_parameter(self, name: str) -> float:
        if name == "time_constant":
            return self.time_constant
        if name == "trim_range_only":
            return 1.0 if self.trim_range_only else 0.0
        return 0.0

    def get_checksum(self) -> int:
        return hash(self.time_constant)

    def smooth(
        self,
        quats: TimeQuat,
        duration_ms: float,
        compute_params: Any,
    ) -> TimeQuat:
        if not quats or duration_ms <= 0.0:
            return dict(quats)

        keyframes: KeyframeManager = compute_params.keyframes

        # Compute sample rate (samples per second)
        sample_rate: float = len(quats) / (duration_ms / 1000.0)

        # alpha = 1 - exp(-dt / tau) where dt = 1/sample_rate
        def get_alpha(time_constant: float) -> float:
            return 1.0 - math.exp(-(1.0 / sample_rate) / time_constant)

        alpha = 1.0
        if self.time_constant > 0.0:
            alpha = get_alpha(self.time_constant)

        # Get trimmed quats (SLERP-fill outside trim ranges)
        trimmed = get_trimmed_quats(
            quats,
            compute_params.scaled_duration_ms,
            self.trim_range_only,
            getattr(compute_params, "trim_ranges", []),
        )

        # Per-timestamp alpha if keyframed or video speed affects smoothing
        alpha_per_timestamp: dict[int, float] = {}
        has_keyframes = (
            keyframes.is_keyframed(KeyframeType.SmoothingParamTimeConstant)
            or (
                getattr(compute_params, "video_speed_affects_smoothing", False)
                and (
                    getattr(compute_params, "video_speed", 1.0) != 1.0
                    or keyframes.is_keyframed(KeyframeType.VideoSpeed)
                )
            )
        )
        if has_keyframes:
            for ts in trimmed:
                timestamp_ms = ts / 1000.0
                val = keyframes.value_at_gyro_timestamp(
                    KeyframeType.SmoothingParamTimeConstant, timestamp_ms
                )
                if val is None:
                    val = self.time_constant
                if getattr(compute_params, "video_speed_affects_smoothing", False):
                    vid_speed = keyframes.value_at_gyro_timestamp(
                        KeyframeType.VideoSpeed, timestamp_ms
                    )
                    if vid_speed is None:
                        vid_speed = abs(getattr(compute_params, "video_speed", 1.0))
                    else:
                        vid_speed = abs(vid_speed)
                    val *= vid_speed
                alpha_per_timestamp[ts] = get_alpha(val)

        # ========== FOV limit scaler computation ==========
        scalers: dict[int, float] = {}
        fov_limit_per_frame = getattr(compute_params, "smoothing_fov_limit_per_frame", {})
        scaled_fps = getattr(compute_params, "scaled_fps", 30.0)

        for ts in trimmed:
            scale = 1.0
            frame = int(ts / 1000.0 * scaled_fps)
            if frame in fov_limit_per_frame:
                scale *= fov_limit_per_frame[frame]
            scalers[ts] = scale

        # Bidirectional smooth on scalers
        if scalers:
            ts_sorted = sorted(scalers.keys())
            # Forward
            prev_scaler = scalers[ts_sorted[0]]
            for i in range(1, len(ts_sorted)):
                ts = ts_sorted[i]
                a = alpha_per_timestamp.get(ts, alpha)
                scalers[ts] = prev_scaler * (1.0 - a) + scalers[ts] * a
                prev_scaler = scalers[ts]
            # Backward
            for i in range(len(ts_sorted) - 2, -1, -1):
                ts = ts_sorted[i]
                a = alpha_per_timestamp.get(ts, alpha)
                scalers[ts] = prev_scaler * (1.0 - a) + scalers[ts] * a
                prev_scaler = scalers[ts]

        # ========== Forward pass ==========
        ts_sorted = sorted(trimmed.keys())
        q = trimmed[ts_sorted[0]]
        smoothed1: TimeQuat = {}
        for ts in ts_sorted:
            x = trimmed[ts]
            a = alpha_per_timestamp.get(ts, alpha)
            if ts in scalers:
                a /= scalers[ts]
            q = q.slerp(x, a)
            smoothed1[ts] = q

        # ========== Backward pass ==========
        q = smoothed1[ts_sorted[-1]]
        result: TimeQuat = {}
        for ts in reversed(ts_sorted):
            x = smoothed1[ts]
            a = alpha_per_timestamp.get(ts, alpha)
            if ts in scalers:
                a /= scalers[ts]
            q = q.slerp(x, a)
            result[ts] = q

        return result
