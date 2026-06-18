"""Mahony filter integrator — exact port of Rust ahrs crate v0.7 Mahony::update_imu.

The algorithm is a line-by-line port of:
  /ahrs-0.7.0/src/mahony.rs  →  Mahony::update_imu()
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


def _hamilton_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions [w, x, y, z]."""
    a_w, a_x, a_y, a_z = a[0], a[1], a[2], a[3]
    b_w, b_x, b_y, b_z = b[0], b[1], b[2], b[3]
    return np.array([
        a_w * b_w - a_x * b_x - a_y * b_y - a_z * b_z,
        a_w * b_x + a_x * b_w + a_y * b_z - a_z * b_y,
        a_w * b_y - a_x * b_z + a_y * b_w + a_z * b_x,
        a_w * b_z + a_x * b_y - a_y * b_x + a_z * b_w,
    ])


def _mahony_update_imu(
    q: np.ndarray,
    gyro: np.ndarray,
    accel_raw: np.ndarray,
    sample_period: float,
    kp: float,
    ki: float,
    e_int: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact port of ahrs::Mahony::update_imu.

    Args:
        q: Current quaternion [w, x, y, z].
        gyro: Gyroscope in rad/s, shape (3,).
        accel_raw: Accelerometer (NOT pre-normalized), shape (3,).
        sample_period: dt in seconds.
        kp: Proportional gain.
        ki: Integral gain.
        e_int: Running integral error, shape (3,).

    Returns:
        (updated quaternion [w,x,y,z], updated e_int)
    """
    w, x, y, z = q[0], q[1], q[2], q[3]

    # Normalize accelerometer measurement
    a_norm = np.linalg.norm(accel_raw)
    if a_norm < 1e-10:
        # ahrs returns Err on zero accel — in golden gen this is skipped
        # Fallback: pure gyro integration
        gyro_quat = np.array([0.0, gyro[0], gyro[1], gyro[2]])
        q_dot = _hamilton_product(q, gyro_quat) * 0.5
        q_new = q + q_dot * sample_period
        norm = np.linalg.norm(q_new)
        if norm > 1e-15:
            q_new = q_new / norm
        return q_new, e_int
    accel = accel_raw / a_norm

    # Estimated gravity direction from current quaternion
    # Exact port of ahrs-0.7 mahony.rs update_imu.
    # IMPORTANT: nalgebra stores quaternions internally as [i, j, k, w] = [x, y, z, w].
    # When the ahrs crate indexes with q[0],q[1],q[2],q[3], it gets [x, y, z, w].
    # Rust: v = Vector3::new(
    #     two*( q[0]*q[2] - q[3]*q[1] ),
    #     two*( q[3]*q[0] + q[1]*q[2] ),
    #     q[3]*q[3] - q[0]*q[0] - q[1]*q[1] + q[2]*q[2]
    # );
    # With nalgebra indexing: q[0]=x, q[1]=y, q[2]=z, q[3]=w:
    v = np.array([
        2.0 * (x * z - w * y),
        2.0 * (w * x + y * z),
        w * w - x * x - y * y + z * z,
    ])

    # Error is cross product between measured and estimated gravity direction
    # e = accel.cross(&v)
    e = np.cross(accel, v)

    # Integrate error (e_int += e * sample_period)
    e_int = e_int + e * sample_period

    # Apply feedback terms: gyro = gyroscope + e * kp + e_int * ki
    gyro_corrected = gyro + e * kp + e_int * ki

    # Compute rate of change of quaternion:
    # qDot = q * Quaternion::from_parts(zero, gyro) * half
    gyro_quat = np.array([0.0, gyro_corrected[0], gyro_corrected[1], gyro_corrected[2]])
    q_dot = _hamilton_product(q, gyro_quat) * 0.5

    # Integrate to yield quaternion: q + qDot * sample_period
    q_new = q + q_dot * sample_period

    # Normalize
    norm = np.linalg.norm(q_new)
    if norm > 1e-15:
        q_new = q_new / norm

    return q_new, e_int


class MahonyIntegrator(GyroIntegrator):
    """Mahony complementary filter with PI controller.

    Exact port of the Rust ahrs crate v0.7 Mahony filter.

    Parameters:
        kp: Proportional gain (default 0.5).
        ki: Integral gain (default 0.0).

    Initial orientation: from_euler_angles(pi/2, 0, 0).
    """

    def __init__(self, kp: float = 0.5, ki: float = 0.0) -> None:
        self.kp = kp
        self.ki = ki

    def integrate(self, imu_data: list[TimeIMU], duration_ms: float) -> TimeQuat:
        if not imu_data:
            return {}

        quats: TimeQuat = {}

        # Initial orientation: pi/2 around X
        init_quat = Quat64.from_euler_angles(math.pi / 2.0, 0.0, 0.0)
        q = init_quat.quaternion()  # [w, x, y, z]

        # Match Rust exactly:
        #   sample_time_s = duration_ms / 1000.0 / samples.len()
        #   prev_time = samples[0].timestamp_ms - sample_time_s
        # NOTE: Rust mixes ms and s here (subtracts seconds from milliseconds).
        # We replicate this exactly.
        sample_time_s = duration_ms / 1000.0 / len(imu_data)
        prev_time = imu_data[0].timestamp_ms - sample_time_s

        e_int = np.zeros(3, dtype=np.float64)

        for sample in imu_data:
            if sample.gyro is None:
                continue

            # Coordinate transform + deg->rad
            gyro = coordinate_transform(sample.gyro) * DEG2RAD

            # Accelerometer: transform, avoid zero vector
            if sample.accl is not None:
                a = sample.accl.copy().astype(np.float64)
                if abs(a[0]) == 0.0 and abs(a[1]) == 0.0 and abs(a[2]) == 0.0:
                    a[0] += 0.0000001
                accl = coordinate_transform(a)
            else:
                accl = np.array([1e-7, 0.0, 0.0], dtype=np.float64)

            # Dynamic sample period — match Rust exactly:
            #   *ahrs.sample_period_mut() = (v.timestamp_ms - prev_time) / 1000.0
            dt = (sample.timestamp_ms - prev_time) / 1000.0

            q, e_int = _mahony_update_imu(q, gyro, accl, dt, self.kp, self.ki, e_int)

            quats[int(sample.timestamp_ms * 1000.0)] = Quat64.from_quaternion(q)
            prev_time = sample.timestamp_ms

        return quats
