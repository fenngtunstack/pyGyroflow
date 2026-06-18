"""Fixed camera smoothing -- locks camera to a fixed orientation.

Port of Gyroflow's core/smoothing/fixed.rs.

Replaces all quaternions with a fixed quaternion built from user-specified
roll/pitch/yaw angles. Supports keyframe animation of each axis.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pygyroflow.smoothing.base import SmoothingAlgorithm
from pygyroflow.types.quaternion import Quat64
from pygyroflow.types.time_types import TimeQuat


def _quat_for_rpy(roll_deg: float, pitch_deg: float, yaw_deg: float) -> Quat64:
    """Convert roll/pitch/yaw (degrees) to a quaternion.

    Matches the Rust implementation's rotation order:
    Z(yaw) * X(pitch) * Y(roll+90) * correction_matrix

    The correction matrix (Z90 * Y90) converts from standard 3D coordinates
    to Gyroflow's camera coordinate system.
    """
    DEG2RAD = math.pi / 180.0

    # Build individual rotation matrices
    # X rotation: pitch
    cos_x = math.cos(pitch_deg * DEG2RAD)
    sin_x = math.sin(pitch_deg * DEG2RAD)
    rot_x = np.array([
        [1, 0, 0],
        [0, cos_x, -sin_x],
        [0, sin_x, cos_x],
    ], dtype=np.float64)

    # Y rotation: roll + 90 deg offset (gyro forward vs camera forward alignment)
    cos_y = math.cos((roll_deg + 90.0) * DEG2RAD)
    sin_y = math.sin((roll_deg + 90.0) * DEG2RAD)
    rot_y = np.array([
        [cos_y, 0, sin_y],
        [0, 1, 0],
        [-sin_y, 0, cos_y],
    ], dtype=np.float64)

    # Z rotation: yaw
    cos_z = math.cos(yaw_deg * DEG2RAD)
    sin_z = math.sin(yaw_deg * DEG2RAD)
    rot_z = np.array([
        [cos_z, -sin_z, 0],
        [sin_z, cos_z, 0],
        [0, 0, 1],
    ], dtype=np.float64)

    # Correction: Z90 * Y90
    cos_90 = math.cos(90.0 * DEG2RAD)
    sin_90 = math.sin(90.0 * DEG2RAD)
    corr_z = np.array([
        [cos_90, -sin_90, 0],
        [sin_90, cos_90, 0],
        [0, 0, 1],
    ], dtype=np.float64)
    corr_y = np.array([
        [cos_90, 0, sin_90],
        [0, 1, 0],
        [-sin_90, 0, cos_90],
    ], dtype=np.float64)
    correction = corr_z @ corr_y

    # Combined: Z(yaw) * X(pitch) * Y(roll+90) * correction
    combined = rot_z @ rot_x @ rot_y @ correction
    return Quat64.from_rotation_matrix(combined)


class FixedSmoothing(SmoothingAlgorithm):
    """Fixed camera smoothing: locks camera to user-specified roll/pitch/yaw.

    All quaternions are replaced with the same fixed quaternion.
    Supports keyframe animation of roll, pitch, yaw independently.
    """

    def __init__(self) -> None:
        self.roll: float = 0.0
        self.pitch: float = 0.0
        self.yaw: float = 0.0

    def get_name(self) -> str:
        return "Fixed camera"

    def get_parameters_json(self) -> list[dict]:
        return [
            {
                "name": "roll",
                "description": "Roll angle",
                "type": "SliderWithField",
                "from": -180,
                "to": 180,
                "value": self.roll,
                "default": 0,
                "unit": "°",
                "keyframe": "SmoothingParamRoll",
            },
            {
                "name": "pitch",
                "description": "Pitch angle",
                "type": "SliderWithField",
                "from": -90,
                "to": 90,
                "value": self.pitch,
                "default": 0,
                "unit": "°",
                "keyframe": "SmoothingParamPitch",
            },
            {
                "name": "yaw",
                "description": "Yaw angle",
                "type": "SliderWithField",
                "from": -180,
                "to": 180,
                "value": self.yaw,
                "default": 0,
                "unit": "°",
                "keyframe": "SmoothingParamYaw",
            },
        ]

    def set_parameter(self, name: str, val: float) -> None:
        if name == "roll":
            self.roll = val
        elif name == "pitch":
            self.pitch = val
        elif name == "yaw":
            self.yaw = val

    def get_parameter(self, name: str) -> float:
        if name == "roll":
            return self.roll
        if name == "pitch":
            return self.pitch
        if name == "yaw":
            return self.yaw
        return 0.0

    def get_checksum(self) -> int:
        return hash((self.roll, self.pitch, self.yaw))

    def smooth(
        self,
        quats: TimeQuat,
        duration_ms: float,
        compute_params: Any,
    ) -> TimeQuat:
        if not quats or duration_ms <= 0.0:
            return dict(quats)

        keyframes = compute_params.keyframes

        # Compute fixed quaternion (used when no keyframes)
        fixed_quat = _quat_for_rpy(self.roll, self.pitch, self.yaw)

        # Check if any axis has keyframes
        from pygyroflow.keyframes import KeyframeType

        is_keyframed = (
            keyframes.is_keyframed(KeyframeType.SmoothingParamRoll)
            or keyframes.is_keyframed(KeyframeType.SmoothingParamPitch)
            or keyframes.is_keyframed(KeyframeType.SmoothingParamYaw)
        )

        result: TimeQuat = {}
        for ts in quats:
            if is_keyframed:
                timestamp_ms = ts / 1000.0
                r = keyframes.value_at_gyro_timestamp(
                    KeyframeType.SmoothingParamRoll, timestamp_ms
                )
                if r is None:
                    r = self.roll
                p = keyframes.value_at_gyro_timestamp(
                    KeyframeType.SmoothingParamPitch, timestamp_ms
                )
                if p is None:
                    p = self.pitch
                y = keyframes.value_at_gyro_timestamp(
                    KeyframeType.SmoothingParamYaw, timestamp_ms
                )
                if y is None:
                    y = self.yaw
                result[ts] = _quat_for_rpy(r, p, y)
            else:
                result[ts] = fixed_quat

        return result
