"""Simple gyroscope + accelerometer integrator.

Exact port of Gyroflow's SimpleGyroAccelIntegrator (Rust golden generator).

Adds gravity-based tilt correction on top of pure gyro integration.
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


class SimpleGyroAccelIntegrator(GyroIntegrator):
    """Gyroscope integration with simple accelerometer gravity correction.

    Exact port of the Rust golden generator's integrate_simple_gyro_accel.

    In the Rust code, try_normalize(0.0) normalizes the accel vector,
    then the (0.9..1.1) check on the normalized norm always passes for
    any non-zero input (normalized vector has norm 1.0). So correction
    is applied whenever accel is non-zero.
    """

    def integrate(self, imu_data: list[TimeIMU], duration_ms: float) -> TimeQuat:
        if not imu_data:
            return {}

        quats: TimeQuat = {}
        orientation = Quat64.from_euler_angles(math.pi / 2.0, 0.0, 0.0)

        sample_time_ms = duration_ms / len(imu_data)
        prev_time = imu_data[0].timestamp_ms - sample_time_ms
        start_time = prev_time

        for sample in imu_data:
            if sample.gyro is None:
                continue

            # Base angular velocity: transform + deg->rad
            g = sample.gyro
            omega = coordinate_transform(g) * DEG2RAD

            # Accelerometer correction
            if sample.accl is not None:
                a = sample.accl
                acc_transformed = coordinate_transform(a)
                # Rust: try_normalize(0.0).unwrap_or_default()
                # If non-zero: returns normalized vector (norm=1.0)
                # If zero: returns zero vector
                acc_norm = np.linalg.norm(acc_transformed)
                if acc_norm > 1e-10:
                    acc = acc_transformed / acc_norm
                    # Rust: g_norm = acc.norm() → always 1.0 for normalized
                    # Rust: (0.9..1.1).contains(&g_norm) → always true
                    # So we always apply correction when accel is non-zero.

                    # acc_world_vec = orientation * acc
                    acc_world = orientation.to_rotation_matrix() @ acc

                    # correction_world = acc_world_vec.cross(&Vector3::new(0.0, 0.0, 1.0))
                    gravity_ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                    correction_world = np.cross(acc_world, gravity_ref)

                    # weight = if v.timestamp_ms - start_time < 15000.0 { 10.0 } else { 0.6 }
                    weight = 10.0 if (sample.timestamp_ms - start_time) < 15000.0 else 0.6

                    # correction_body = weight * (orientation.conjugate() * correction_world)
                    correction_body = weight * (orientation.inverse().to_rotation_matrix() @ correction_world)

                    # omega += correction_body
                    omega = omega + correction_body

            # Time step in seconds
            dt = (sample.timestamp_ms - prev_time) / 1000.0
            if dt <= 0.0:
                dt = sample_time_ms / 1000.0

            delta_q = Quat64.from_scaled_axis(omega * dt)
            orientation = orientation * delta_q

            quats[int(sample.timestamp_ms * 1000.0)] = orientation
            prev_time = sample.timestamp_ms

        return quats
