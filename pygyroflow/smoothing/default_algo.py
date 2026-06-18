"""Default adaptive smoothing algorithm -- velocity-aware bidirectional smoothing.

Port of Gyroflow's core/smoothing/default_algo.rs (782 lines).

This is the primary smoothing algorithm. It adapts smoothing strength based on
rotation velocity: slow motion gets strong smoothing (1s time constant),
fast motion gets light smoothing (0.1s time constant). An optional second pass
uses velocity * distance weighting for additional adaptive refinement.

Algorithm flow:
1. Compute per-frame angular velocity
2. Bidirectional smooth velocity (zero-phase filter)
3. Normalize velocity by max_velocity * smoothness * fov_ratio
4. First pass: bidirectional smoothing with velocity-adaptive alpha
5. (Optional) Second pass: compute distance, normalize, smooth distance,
   then bidirectional smoothing with velocity*distance product weighting
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pygyroflow.keyframes import KeyframeManager, KeyframeType
from pygyroflow.smoothing.base import SmoothingAlgorithm
from pygyroflow.smoothing.trim import get_trimmed_quats
from pygyroflow.types.quaternion import Quat64
from pygyroflow.types.time_types import TimeQuat

# Maximum rotation velocity threshold (degrees/second)
MAX_VELOCITY: float = 500.0

# FOV reference value (degrees diagonal)
FOV_REFERENCE: float = 120.0

# Radians to degrees conversion
RAD_TO_DEG: float = 180.0 / math.pi


class DefaultAlgo(SmoothingAlgorithm):
    """Adaptive velocity-sensing smoothing algorithm.

    Parameters:
        smoothness: Overall smoothing strength (0.001-1.0, default 0.5).
        smoothness_pitch/yaw/roll: Per-axis overrides (used when per_axis=True).
        per_axis: Enable per-axis smoothing (default False).
        second_pass: Enable second smoothing pass (default True).
        trim_range_only: Only smooth within trim ranges (default True).
        max_smoothness: Time constant at low velocity (seconds, default 1.0).
        alpha_0_1s: Time constant at high velocity (seconds, default 0.1).
    """

    def __init__(self) -> None:
        self.smoothness: float = 0.5
        self.smoothness_pitch: float = 0.5
        self.smoothness_yaw: float = 0.5
        self.smoothness_roll: float = 0.5
        self.per_axis: bool = False
        self.second_pass: bool = True
        self.trim_range_only: bool = True
        self.max_smoothness: float = 1.0
        self.alpha_0_1s: float = 0.1

    # ------------------------------------------------------------------
    # SmoothingAlgorithm interface
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        return "Default"

    def get_parameters_json(self) -> list[dict]:
        return [
            {
                "name": "smoothness",
                "description": "Smoothness",
                "type": "SliderWithField",
                "from": 0.001,
                "to": 1.0,
                "value": self.smoothness,
                "default": 0.5,
                "unit": "",
                "precision": 3,
                "keyframe": "SmoothingParamSmoothness",
            },
            {
                "name": "smoothness_pitch",
                "description": "Pitch smoothness",
                "type": "SliderWithField",
                "from": 0.001,
                "to": 1.0,
                "value": self.smoothness_pitch,
                "default": 0.5,
                "unit": "",
                "precision": 3,
                "keyframe": "SmoothingParamPitch",
            },
            {
                "name": "smoothness_yaw",
                "description": "Yaw smoothness",
                "type": "SliderWithField",
                "from": 0.001,
                "to": 1.0,
                "value": self.smoothness_yaw,
                "default": 0.5,
                "unit": "",
                "precision": 3,
                "keyframe": "SmoothingParamYaw",
            },
            {
                "name": "smoothness_roll",
                "description": "Roll smoothness",
                "type": "SliderWithField",
                "from": 0.001,
                "to": 1.0,
                "value": self.smoothness_roll,
                "default": 0.5,
                "unit": "",
                "precision": 3,
                "keyframe": "SmoothingParamRoll",
            },
            {
                "name": "per_axis",
                "description": "Per axis",
                "advanced": True,
                "type": "CheckBox",
                "default": self.per_axis,
                "value": 1.0 if self.per_axis else 0.0,
            },
            {
                "name": "trim_range_only",
                "description": "Only within trim range",
                "advanced": True,
                "type": "CheckBox",
                "default": self.trim_range_only,
                "value": 1.0 if self.trim_range_only else 0.0,
            },
            {
                "name": "max_smoothness",
                "description": "Max smoothness",
                "advanced": True,
                "type": "SliderWithField",
                "from": 0.1,
                "to": 5.0,
                "value": self.max_smoothness,
                "default": 1.0,
                "precision": 3,
                "unit": "s",
                "keyframe": "SmoothingParamTimeConstant",
            },
            {
                "name": "alpha_0_1s",
                "description": "Max smoothness at high velocity",
                "advanced": True,
                "type": "SliderWithField",
                "from": 0.01,
                "to": 1.0,
                "value": self.alpha_0_1s,
                "default": 0.1,
                "precision": 3,
                "unit": "s",
                "keyframe": "SmoothingParamTimeConstant2",
            },
        ]

    def set_parameter(self, name: str, val: float) -> None:
        if name == "smoothness":
            self.smoothness = val
        elif name == "smoothness_pitch":
            self.smoothness_pitch = val
        elif name == "smoothness_yaw":
            self.smoothness_yaw = val
        elif name == "smoothness_roll":
            self.smoothness_roll = val
        elif name == "per_axis":
            self.per_axis = val > 0.1
        elif name == "trim_range_only":
            self.trim_range_only = val > 0.1
        elif name == "max_smoothness":
            self.max_smoothness = val
        elif name == "alpha_0_1s":
            self.alpha_0_1s = val

    def get_parameter(self, name: str) -> float:
        if name == "smoothness":
            return self.smoothness
        if name == "smoothness_pitch":
            return self.smoothness_pitch
        if name == "smoothness_yaw":
            return self.smoothness_yaw
        if name == "smoothness_roll":
            return self.smoothness_roll
        if name == "per_axis":
            return 1.0 if self.per_axis else 0.0
        if name == "trim_range_only":
            return 1.0 if self.trim_range_only else 0.0
        if name == "max_smoothness":
            return self.max_smoothness
        if name == "alpha_0_1s":
            return self.alpha_0_1s
        return 0.0

    def get_checksum(self) -> int:
        return hash((
            self.smoothness,
            self.smoothness_pitch,
            self.smoothness_yaw,
            self.smoothness_roll,
            self.max_smoothness,
            self.alpha_0_1s,
            self.per_axis,
            self.second_pass,
        ))

    # ------------------------------------------------------------------
    # Core smoothing
    # ------------------------------------------------------------------

    def smooth(
        self,
        quats: TimeQuat,
        duration_ms: float,
        compute_params: Any,
    ) -> TimeQuat:
        if not quats or duration_ms <= 0.0:
            return dict(quats)

        sample_rate: float = len(quats) / (duration_ms / 1000.0)
        rad_to_deg_per_sec: float = sample_rate * RAD_TO_DEG

        # Lambda: alpha = 1 - exp(-dt / tau)
        def get_alpha(time_constant: float) -> float:
            return 1.0 - math.exp(-(1.0 / sample_rate) / time_constant)

        keyframes: KeyframeManager = compute_params.keyframes

        # Get trimmed quats
        trimmed = get_trimmed_quats(
            quats,
            getattr(compute_params, "scaled_duration_ms", duration_ms),
            self.trim_range_only,
            getattr(compute_params, "trim_ranges", []),
        )

        # Lambda: get keyframed parameter per timestamp
        def get_keyframed_param(
            typ: KeyframeType,
            default: float,
            cb,  # Callable[[float], float]
        ) -> dict[int, float]:
            ret: dict[int, float] = {}
            should_compute = (
                keyframes.is_keyframed(typ)
                or (
                    getattr(compute_params, "video_speed_affects_smoothing", False)
                    and (
                        getattr(compute_params, "video_speed", 1.0) != 1.0
                        or keyframes.is_keyframed(KeyframeType.VideoSpeed)
                    )
                )
            )
            if not should_compute:
                return ret
            for ts in trimmed:
                timestamp_ms = ts / 1000.0
                val = keyframes.value_at_gyro_timestamp(typ, timestamp_ms)
                if val is None:
                    val = default
                if getattr(compute_params, "video_speed_affects_smoothing", False):
                    vid_speed = keyframes.value_at_gyro_timestamp(
                        KeyframeType.VideoSpeed, timestamp_ms
                    )
                    if vid_speed is None:
                        vid_speed = abs(getattr(compute_params, "video_speed", 1.0))
                    else:
                        vid_speed = abs(vid_speed)
                    if typ in (
                        KeyframeType.SmoothingParamTimeConstant,
                        KeyframeType.SmoothingParamTimeConstant2,
                    ):
                        val *= 1.0 + ((vid_speed - 1.0) / 2.0)
                    else:
                        val *= vid_speed
                ret[ts] = cb(val)
            return ret

        def noop(v: float) -> float:
            return v

        # Get keyframed parameters
        alpha_smoothness_per_ts = get_keyframed_param(
            KeyframeType.SmoothingParamTimeConstant, self.max_smoothness, get_alpha
        )
        alpha_0_1s_per_ts = get_keyframed_param(
            KeyframeType.SmoothingParamTimeConstant2, self.alpha_0_1s, get_alpha
        )
        smoothness_per_ts = get_keyframed_param(
            KeyframeType.SmoothingParamSmoothness, self.smoothness, noop
        )
        smoothness_pitch_per_ts = get_keyframed_param(
            KeyframeType.SmoothingParamPitch, self.smoothness_pitch, noop
        )
        smoothness_yaw_per_ts = get_keyframed_param(
            KeyframeType.SmoothingParamYaw, self.smoothness_yaw, noop
        )
        smoothness_roll_per_ts = get_keyframed_param(
            KeyframeType.SmoothingParamRoll, self.smoothness_roll, noop
        )

        alpha_smoothness = get_alpha(self.max_smoothness)
        alpha_0_1s = get_alpha(self.alpha_0_1s)

        # Sorted timestamps for ordered iteration
        ts_sorted = sorted(trimmed.keys())

        # ========== Step 1: Compute velocity ==========
        velocity: dict[int, list[float]] = {}

        first_ts = ts_sorted[0]
        velocity[first_ts] = [0.0, 0.0, 0.0]

        prev_quat = trimmed[first_ts]
        for ts in ts_sorted[1:]:
            quat = trimmed[ts]
            dist = prev_quat.inverse() * quat
            if self.per_axis:
                euler = dist.euler_angles()
                velocity[ts] = [
                    abs(euler[0]) * rad_to_deg_per_sec,  # Pitch
                    abs(euler[1]) * rad_to_deg_per_sec,  # Yaw
                    abs(euler[2]) * rad_to_deg_per_sec,  # Roll
                ]
            else:
                angle = dist.angle()
                deg_per_sec = angle * rad_to_deg_per_sec
                velocity[ts] = [deg_per_sec, deg_per_sec, deg_per_sec]
            prev_quat = quat

        # ========== Step 2: Smooth velocity (bidirectional) ==========
        # Forward pass
        prev_vel = velocity[ts_sorted[0]]
        for ts in ts_sorted[1:]:
            vel = velocity[ts]
            vel[0] = prev_vel[0] * (1.0 - alpha_0_1s) + vel[0] * alpha_0_1s
            vel[1] = prev_vel[1] * (1.0 - alpha_0_1s) + vel[1] * alpha_0_1s
            vel[2] = prev_vel[2] * (1.0 - alpha_0_1s) + vel[2] * alpha_0_1s
            prev_vel = vel

        # Backward pass
        prev_vel = velocity[ts_sorted[-1]]
        for ts in reversed(ts_sorted[:-1]):
            vel = velocity[ts]
            vel[0] = prev_vel[0] * (1.0 - alpha_0_1s) + vel[0] * alpha_0_1s
            vel[1] = prev_vel[1] * (1.0 - alpha_0_1s) + vel[1] * alpha_0_1s
            vel[2] = prev_vel[2] * (1.0 - alpha_0_1s) + vel[2] * alpha_0_1s
            prev_vel = vel

        # ========== Steps 3-4: Normalize velocity and first smoothing pass ==========
        fov_limit_per_frame = getattr(compute_params, "smoothing_fov_limit_per_frame", {})
        scaled_fps = getattr(compute_params, "scaled_fps", 30.0)
        camera_diagonal_fovs = getattr(compute_params, "camera_diagonal_fovs", [120.0])

        for ts in ts_sorted:
            vel = velocity[ts]
            sp = smoothness_pitch_per_ts.get(ts, self.smoothness_pitch)
            sy = smoothness_yaw_per_ts.get(ts, self.smoothness_yaw)
            sr = smoothness_roll_per_ts.get(ts, self.smoothness_roll)
            s = smoothness_per_ts.get(ts, self.smoothness)

            frame = int(ts / 1000.0 * scaled_fps)
            if len(camera_diagonal_fovs) == 1:
                fov_ratio = camera_diagonal_fovs[0] / FOV_REFERENCE
            else:
                fov_ratio = (
                    camera_diagonal_fovs[frame] / FOV_REFERENCE
                    if frame < len(camera_diagonal_fovs)
                    else 1.0
                )

            if frame in fov_limit_per_frame:
                fov_ratio *= fov_limit_per_frame[frame]

            max_vel = [MAX_VELOCITY, MAX_VELOCITY, MAX_VELOCITY]
            if self.per_axis:
                max_vel[0] *= sp * fov_ratio
                max_vel[1] *= sy * fov_ratio
                max_vel[2] *= sr * fov_ratio
            else:
                max_vel[0] *= s * fov_ratio

            if self.second_pass:
                max_vel[0] *= 0.5
                if self.per_axis:
                    max_vel[1] *= 0.5
                    max_vel[2] *= 0.5

            vel[0] /= max_vel[0]
            if self.per_axis:
                vel[1] /= max_vel[1]
                vel[2] /= max_vel[2]

        # ========== First smoothing pass: forward ==========
        q = trimmed[ts_sorted[0]]
        smoothed1: TimeQuat = {}
        for ts in ts_sorted:
            x = trimmed[ts]
            ratio = velocity[ts]
            a_s = alpha_smoothness_per_ts.get(ts, alpha_smoothness)
            a_0 = alpha_0_1s_per_ts.get(ts, alpha_0_1s)

            if self.per_axis:
                pitch_factor = a_s * (1.0 - ratio[0]) + a_0 * ratio[0]
                yaw_factor = a_s * (1.0 - ratio[1]) + a_0 * ratio[1]
                roll_factor = a_s * (1.0 - ratio[2]) + a_0 * ratio[2]

                euler_rot = (q.inverse() * x).euler_angles()
                quat_rot = Quat64.from_euler_angles(
                    euler_rot[0] * min(pitch_factor, 1.0),
                    euler_rot[1] * min(yaw_factor, 1.0),
                    euler_rot[2] * min(roll_factor, 1.0),
                )
                q = q * quat_rot
            else:
                val = a_s * (1.0 - ratio[0]) + a_0 * ratio[0]
                q = q.slerp(x, min(val, 1.0))

            smoothed1[ts] = q

        # ========== First smoothing pass: backward ==========
        q = smoothed1[ts_sorted[-1]]
        smoothed2: TimeQuat = {}
        for ts in reversed(ts_sorted):
            x = smoothed1[ts]
            a_s = alpha_smoothness_per_ts.get(ts, alpha_smoothness)
            a_0 = alpha_0_1s_per_ts.get(ts, alpha_0_1s)
            ratio = velocity[ts]

            if self.per_axis:
                pitch_factor = a_s * (1.0 - ratio[0]) + a_0 * ratio[0]
                yaw_factor = a_s * (1.0 - ratio[1]) + a_0 * ratio[1]
                roll_factor = a_s * (1.0 - ratio[2]) + a_0 * ratio[2]

                euler_rot = (q.inverse() * x).euler_angles()
                quat_rot = Quat64.from_euler_angles(
                    euler_rot[0] * min(pitch_factor, 1.0),
                    euler_rot[1] * min(yaw_factor, 1.0),
                    euler_rot[2] * min(roll_factor, 1.0),
                )
                q = q * quat_rot
            else:
                val = a_s * (1.0 - ratio[0]) + a_0 * ratio[0]
                q = q.slerp(x, min(val, 1.0))

            smoothed2[ts] = q

        if not self.second_pass:
            return smoothed2

        # ========== Second pass: compute distance ==========
        distance: dict[int, list[float]] = {}
        max_distance = [0.0, 0.0, 0.0]

        for ts in ts_sorted:
            quat = smoothed2[ts]
            orig = trimmed[ts]
            dist = orig.inverse() * quat

            if self.per_axis:
                euler = dist.euler_angles()
                d = [abs(euler[0]), abs(euler[1]), abs(euler[2])]
                distance[ts] = d
                if d[0] > max_distance[0]:
                    max_distance[0] = d[0]
                if d[1] > max_distance[1]:
                    max_distance[1] = d[1]
                if d[2] > max_distance[2]:
                    max_distance[2] = d[2]
            else:
                angle = dist.angle()
                distance[ts] = [angle, 0.0, 0.0]
                if angle > max_distance[0]:
                    max_distance[0] = angle

        # Normalize distance, discard under 0.5
        for ts in ts_sorted:
            d = distance[ts]
            d[0] /= max_distance[0] if max_distance[0] > 0 else 1.0
            if d[0] < 0.5:
                d[0] = 0.0
            if self.per_axis:
                d[1] /= max_distance[1] if max_distance[1] > 0 else 1.0
                if d[1] < 0.5:
                    d[1] = 0.0
                d[2] /= max_distance[2] if max_distance[2] > 0 else 1.0
                if d[2] < 0.5:
                    d[2] = 0.0

        # Smooth distance (bidirectional)
        prev_dist = distance[ts_sorted[0]]
        for ts in ts_sorted[1:]:
            d = distance[ts]
            d[0] = prev_dist[0] * (1.0 - alpha_0_1s) + d[0] * alpha_0_1s
            d[1] = prev_dist[1] * (1.0 - alpha_0_1s) + d[1] * alpha_0_1s
            d[2] = prev_dist[2] * (1.0 - alpha_0_1s) + d[2] * alpha_0_1s
            prev_dist = list(d)  # Copy

        prev_dist = distance[ts_sorted[-1]]
        for ts in reversed(ts_sorted[:-1]):
            d = distance[ts]
            d[0] = prev_dist[0] * (1.0 - alpha_0_1s) + d[0] * alpha_0_1s
            d[1] = prev_dist[1] * (1.0 - alpha_0_1s) + d[1] * alpha_0_1s
            d[2] = prev_dist[2] * (1.0 - alpha_0_1s) + d[2] * alpha_0_1s
            prev_dist = list(d)

        # Recompute max distance
        max_distance = [0.0, 0.0, 0.0]
        for ts in ts_sorted:
            d = distance[ts]
            if d[0] > max_distance[0]:
                max_distance[0] = d[0]
            if self.per_axis:
                if d[1] > max_distance[1]:
                    max_distance[1] = d[1]
                if d[2] > max_distance[2]:
                    max_distance[2] = d[2]

        # Normalize distance and remap to [0.5, 1.0]
        for ts in ts_sorted:
            d = distance[ts]
            d[0] /= max_distance[0] if max_distance[0] > 0 else 1.0
            d[0] = (d[0] + 1.0) / 2.0
            if self.per_axis:
                d[1] /= max_distance[1] if max_distance[1] > 0 else 1.0
                d[1] = (d[1] + 1.0) / 2.0
                d[2] /= max_distance[2] if max_distance[2] > 0 else 1.0
                d[2] = (d[2] + 1.0) / 2.0

        # ========== Second pass: forward ==========
        q = smoothed2[ts_sorted[0]]
        smoothed3: TimeQuat = {}
        for ts in ts_sorted:
            x = smoothed2[ts]
            a_s = alpha_smoothness_per_ts.get(ts, alpha_smoothness)
            a_0 = alpha_0_1s_per_ts.get(ts, alpha_0_1s)
            vel_ratio = velocity[ts]
            dist_ratio = distance[ts]

            if self.per_axis:
                pitch_factor = a_s * (1.0 - vel_ratio[0] * dist_ratio[0]) + a_0 * vel_ratio[0] * dist_ratio[0]
                yaw_factor = a_s * (1.0 - vel_ratio[1] * dist_ratio[1]) + a_0 * vel_ratio[1] * dist_ratio[1]
                roll_factor = a_s * (1.0 - vel_ratio[2] * dist_ratio[2]) + a_0 * vel_ratio[2] * dist_ratio[2]

                euler_rot = (q.inverse() * x).euler_angles()
                quat_rot = Quat64.from_euler_angles(
                    euler_rot[0] * min(pitch_factor, 1.0),
                    euler_rot[1] * min(yaw_factor, 1.0),
                    euler_rot[2] * min(roll_factor, 1.0),
                )
                q = q * quat_rot
            else:
                val = a_s * (1.0 - vel_ratio[0] * dist_ratio[0]) + a_0 * vel_ratio[0] * dist_ratio[0]
                q = q.slerp(x, min(val, 1.0))

            smoothed3[ts] = q

        # ========== Second pass: backward ==========
        q = smoothed3[ts_sorted[-1]]
        result: TimeQuat = {}
        for ts in reversed(ts_sorted):
            x = smoothed3[ts]
            a_s = alpha_smoothness_per_ts.get(ts, alpha_smoothness)
            a_0 = alpha_0_1s_per_ts.get(ts, alpha_0_1s)
            vel_ratio = velocity[ts]
            dist_ratio = distance[ts]

            if self.per_axis:
                pitch_factor = a_s * (1.0 - vel_ratio[0] * dist_ratio[0]) + a_0 * vel_ratio[0] * dist_ratio[0]
                yaw_factor = a_s * (1.0 - vel_ratio[1] * dist_ratio[1]) + a_0 * vel_ratio[1] * dist_ratio[1]
                roll_factor = a_s * (1.0 - vel_ratio[2] * dist_ratio[2]) + a_0 * vel_ratio[2] * dist_ratio[2]

                euler_rot = (q.inverse() * x).euler_angles()
                quat_rot = Quat64.from_euler_angles(
                    euler_rot[0] * min(pitch_factor, 1.0),
                    euler_rot[1] * min(yaw_factor, 1.0),
                    euler_rot[2] * min(roll_factor, 1.0),
                )
                q = q * quat_rot
            else:
                val = a_s * (1.0 - vel_ratio[0] * dist_ratio[0]) + a_0 * vel_ratio[0] * dist_ratio[0]
                q = q.slerp(x, min(val, 1.0))

            result[ts] = q

        return result
