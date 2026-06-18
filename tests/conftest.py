"""Shared fixtures for pygyroflow test suite."""

import pytest
import numpy as np
from pygyroflow.types.quaternion import Quat64
from pygyroflow.types.time_types import TimeIMU, TimeQuat


@pytest.fixture
def identity_quat():
    return Quat64.identity()


@pytest.fixture
def sample_imu_data():
    """Generate synthetic IMU data: constant rotation around Z axis at 10 deg/s."""
    data = []
    for i in range(1000):
        t_ms = i * 5.0  # 200Hz sampling
        gyro = np.array([0.0, 0.0, 10.0])  # 10 deg/s around Z
        accl = np.array([0.0, 0.0, 9.8])    # gravity
        data.append(TimeIMU(timestamp_ms=t_ms, gyro=gyro, accl=accl, magn=None))
    return data


@pytest.fixture
def sample_quaternions():
    """Generate a sequence of quaternions rotating around Z."""
    quats = {}
    for i in range(100):
        angle = i * 0.01  # radians
        quats[i * 10000] = Quat64.from_euler_angles(0.0, 0.0, angle)
    return quats


@pytest.fixture
def sample_compute_params(sample_quaternions):
    from pygyroflow.stabilization import ComputeParams
    return ComputeParams(
        width=1920, height=1080, output_width=1920, output_height=1080,
        frame_count=100, scaled_fps=100.0, scaled_duration_ms=1000.0,
        quaternions=sample_quaternions, fovs=[1.0]*100, fov_scale=1.0,
        camera_matrix=np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]),
    )


def assert_quats_close(actual: TimeQuat, expected: TimeQuat, tol: float = 1e-6):
    """Compare two TimeQuat maps, handling quaternion double-cover."""
    assert set(actual.keys()) == set(expected.keys()), \
        f"Timestamp mismatch: extra={set(actual.keys()) - set(expected.keys())}, missing={set(expected.keys()) - set(actual.keys())}"
    for ts in actual:
        q1 = actual[ts].quaternion()
        q2 = expected[ts].quaternion()
        if np.dot(q1, q2) < 0:
            q2 = -q2
        np.testing.assert_allclose(q1, q2, atol=tol, err_msg=f"Quaternion mismatch at t={ts}")
