"""Complementary filter integrator — exact port of Rust golden generator.

This matches the integrate_complementary function in the Rust golden generator,
which implements Gyroflow's complementary_v2 algorithm (simplified version).
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


class ComplementaryIntegrator(GyroIntegrator):
    """Complementary filter — exact port of Rust golden generator.

    Algorithm per sample:
        1. Gyro prediction: delta_q = from_scaled_axis(omega * dt); orientation *= delta_q
        2. Accel correction (in world frame):
           - Normalize accel, rotate to world frame
           - cross = acc_world x [0,0,1]
           - angle = |cross|
           - if angle > 1e-10: correction = from_scaled_axis(w * cross/|cross|)
             where w = high_alpha*angle (during settle) or alpha*angle (after)
           - orientation = correction * orientation

    Initial orientation: from_euler_angles(pi/2, 0, 0).
    """

    def integrate(self, imu_data: list[TimeIMU], duration_ms: float) -> TimeQuat:
        if not imu_data:
            return {}

        quats: TimeQuat = {}
        orientation = Quat64.from_euler_angles(math.pi / 2.0, 0.0, 0.0)

        sample_time_ms = duration_ms / len(imu_data)
        prev_time = imu_data[0].timestamp_ms - sample_time_ms

        # Complementary filter gains — match Rust exactly
        alpha = 0.02  # accelerometer weight
        settle_time_ms = min(duration_ms * 0.05, 2000.0)
        high_alpha = 0.1

        for sample in imu_data:
            if sample.gyro is None:
                continue

            # Base angular velocity: transform + deg->rad
            g = sample.gyro
            omega = coordinate_transform(g) * DEG2RAD

            dt = (sample.timestamp_ms - prev_time) / 1000.0

            # --- Gyro prediction step ---
            delta_q = Quat64.from_scaled_axis(omega * dt)
            orientation = orientation * delta_q

            # --- Accelerometer correction ---
            if sample.accl is not None:
                a = sample.accl.copy().astype(np.float64)
                if abs(a[0]) == 0.0 and abs(a[1]) == 0.0 and abs(a[2]) == 0.0:
                    a[0] += 0.0000001
                acc = coordinate_transform(a)

                # try_normalize(0.0)
                acc_norm_val = np.linalg.norm(acc)
                if acc_norm_val > 1e-10:
                    acc_normalized = acc / acc_norm_val

                    # Gravity in world frame should be (0, 0, 1)
                    gravity_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                    acc_world = orientation.to_rotation_matrix() @ acc_normalized
                    cross = np.cross(acc_world, gravity_world)
                    angle = np.linalg.norm(cross)

                    if angle > 1e-10:
                        axis = cross / angle
                        # Rust: if v.timestamp_ms < settle_time_ms
                        w = high_alpha * angle if sample.timestamp_ms < settle_time_ms else alpha * angle
                        correction = Quat64.from_scaled_axis(w * axis)
                        orientation = correction * orientation

            quats[int(sample.timestamp_ms * 1000.0)] = orientation
            prev_time = sample.timestamp_ms

        return quats
