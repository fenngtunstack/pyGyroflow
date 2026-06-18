"""Simple gyroscope-only integrator — dead reckoning without drift correction.

Port of Gyroflow's SimpleGyroIntegrator (Rust).
"""

from __future__ import annotations

import math

import numpy as np

from pygyroflow.types.quaternion import Quat64
from pygyroflow.types.time_types import TimeIMU, TimeQuat
from pygyroflow.imu_integration.base import GyroIntegrator

DEG2RAD: float = math.pi / 180.0


def coordinate_transform(v: np.ndarray) -> np.ndarray:
    """IMU coordinate frame to camera frame: (x,y,z) -> (-y, x, z)."""
    return np.array([-v[1], v[0], v[2]], dtype=np.float64)


class SimpleGyroIntegrator(GyroIntegrator):
    """Pure gyroscope integration. No accelerometer correction.

    Algorithm per sample:
        1. omega = coordinate_transform(gyro) * DEG2RAD
        2. dt = (timestamp_ms - prev_timestamp_ms) / 1000
        3. delta_q = from_scaled_axis(omega * dt)
        4. orientation = orientation * delta_q

    Initial orientation is a pi/2 rotation around X axis.
    """

    def integrate(self, imu_data: list[TimeIMU], duration_ms: float) -> TimeQuat:
        if not imu_data:
            return {}

        quats: TimeQuat = {}
        orientation = Quat64.from_euler_angles(math.pi / 2.0, 0.0, 0.0)

        sample_time_ms = duration_ms / len(imu_data)
        prev_time = imu_data[0].timestamp_ms - sample_time_ms

        for sample in imu_data:
            if sample.gyro is None:
                continue

            # Coordinate transform + degrees to radians
            omega = coordinate_transform(sample.gyro) * DEG2RAD

            # Time step in seconds
            dt = (sample.timestamp_ms - prev_time) / 1000.0
            if dt <= 0.0:
                dt = sample_time_ms / 1000.0

            # Incremental rotation quaternion from scaled axis
            delta_q = Quat64.from_scaled_axis(omega * dt)

            # Update orientation: right-multiply delta
            orientation = orientation * delta_q

            # Store with microsecond timestamp
            quats[int(sample.timestamp_ms * 1000.0)] = orientation

            prev_time = sample.timestamp_ms

        return quats
