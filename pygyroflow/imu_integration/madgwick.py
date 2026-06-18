"""Madgwick filter integrator — exact port of Rust ahrs crate v0.7 Madgwick::update_imu.

The algorithm is a line-by-line port of:
  /ahrs-0.7.0/src/madgwick.rs  →  Madgwick::update_imu()
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


def _madgwick_update_gyro(
    q: np.ndarray,
    gyro: np.ndarray,
    sample_period: float,
) -> np.ndarray:
    """Pure gyro update — port of ahrs::Madgwick::update_gyro."""
    gyro_quat = np.array([0.0, gyro[0], gyro[1], gyro[2]])
    q_dot = _hamilton_product(q, gyro_quat) * 0.5
    q_new = q + q_dot * sample_period
    norm = np.linalg.norm(q_new)
    if norm > 1e-15:
        q_new = q_new / norm
    return q_new


def _madgwick_update_imu(
    q: np.ndarray,
    gyro: np.ndarray,
    accel_raw: np.ndarray,
    sample_period: float,
    beta: float,
) -> np.ndarray:
    """Exact port of ahrs::Madgwick::update_imu.

    Args:
        q: Current quaternion [w, x, y, z].
        gyro: Gyroscope in rad/s, shape (3,).
        accel_raw: Accelerometer (NOT pre-normalized), shape (3,).
        sample_period: dt in seconds.
        beta: Filter gain.

    Returns:
        Updated quaternion [w, x, y, z].
    """
    w, x, y, z = q[0], q[1], q[2], q[3]

    # Normalize accelerometer measurement
    a_norm = np.linalg.norm(accel_raw)
    if a_norm < 1e-10:
        return _madgwick_update_gyro(q, gyro, sample_period)
    accel = accel_raw / a_norm

    # Gradient descent algorithm corrective step
    # Exact port of ahrs-0.7 madgwick.rs update_imu.
    # IMPORTANT: nalgebra stores quaternions internally as [i, j, k, w] = [x, y, z, w].
    # When the ahrs crate indexes with q[0],q[1],q[2],q[3], it gets [x, y, z, w].
    # Rust: F = Vector4::new(
    #     two*(       q[0]*q[2] - q[3]*q[1]) - accel[0],
    #     two*(       q[3]*q[0] + q[1]*q[2]) - accel[1],
    #     two*(half - q[0]*q[0] - q[1]*q[1]) - accel[2],
    #     zero
    # );
    # With nalgebra indexing: q[0]=x, q[1]=y, q[2]=z, q[3]=w:
    F = np.array([
        2.0 * (x * z - w * y) - accel[0],
        2.0 * (w * x + y * z) - accel[1],
        2.0 * (0.5 - x * x - y * y) - accel[2],
        0.0,
    ])

    # Jacobian transpose * F
    # Exact port. With nalgebra indexing q[0]=x, q[1]=y, q[2]=z, q[3]=w:
    #   J_t = Matrix4::new(
    #       -two*q[1], two*q[0],       zero, zero,    = [-2y, 2x, 0, 0]
    #        two*q[2], two*q[3], -four*q[0], zero,    = [2z, 2w, -4x, 0]
    #       -two*q[3], two*q[2], -four*q[1], zero,    = [-2w, 2y, -4y, 0]
    #        two*q[0], two*q[1],       zero, zero     = [2x, 2y, 0, 0]
    #   );
    # step = J_t * F (computed with nalgebra indexing q[0]=x, q[1]=y, q[2]=z, q[3]=w)
    # step_nalgebra = [grad_wrt_x, grad_wrt_y, grad_wrt_z, grad_wrt_w]
    step_nalgebra = np.array([
        -2.0 * y * F[0] + 2.0 * x * F[1],
        2.0 * z * F[0] + 2.0 * w * F[1] - 4.0 * x * F[2],
        -2.0 * w * F[0] + 2.0 * z * F[1] - 4.0 * y * F[2],
        2.0 * x * F[0] + 2.0 * y * F[1],
    ])

    # Try to normalize step, falling back to gyro update if not possible
    step_norm = np.linalg.norm(step_nalgebra)
    if step_norm < 1e-15:
        return _madgwick_update_gyro(q, gyro, sample_period)
    step_nalgebra = step_nalgebra / step_norm

    # Rust does: Quaternion::new(step[0], step[1], step[2], step[3])
    # Quaternion::new(w, i, j, k) stores as [i, j, k, w]
    # step_nalgebra = [s0, s1, s2, s3] where s0 was computed as the x-index term
    # Quaternion::new(s0, s1, s2, s3) => w=s0, x=s1, y=s2, z=s3
    # In our [w,x,y,z] Python convention: step_quat = [s0, s1, s2, s3]
    step_quat = step_nalgebra  # Same array, but now interpreted as [w,x,y,z]

    # Compute rate of change of quaternion:
    #   qDot = (q * Quaternion::from_parts(zero, *gyroscope)) * half
    #        - Quaternion::new(step[0], step[1], step[2], step[3]) * self.beta;
    gyro_quat = np.array([0.0, gyro[0], gyro[1], gyro[2]])
    q_dot = _hamilton_product(q, gyro_quat) * 0.5 - step_quat * beta

    # Integrate to yield quaternion: q + qDot * sample_period
    q_new = q + q_dot * sample_period

    # Normalize
    norm = np.linalg.norm(q_new)
    if norm > 1e-15:
        q_new = q_new / norm

    return q_new


class MadgwickIntegrator(GyroIntegrator):
    """Madgwick gradient-descent filter.

    Exact port of the Rust ahrs crate v0.7 Madgwick filter.

    Parameters:
        beta: Filter gain (default 0.02).

    Initial orientation: from_euler_angles(pi/2, 0, 0).
    """

    def __init__(self, beta: float = 0.02) -> None:
        self.beta = beta

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
        sample_time_s = duration_ms / 1000.0 / len(imu_data)
        prev_time = imu_data[0].timestamp_ms - sample_time_s

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

            q = _madgwick_update_imu(q, gyro, accl, dt, self.beta)

            quats[int(sample.timestamp_ms * 1000.0)] = Quat64.from_quaternion(q)
            prev_time = sample.timestamp_ms

        return quats
