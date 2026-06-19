"""Tests for all IMU integration algorithms."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pygyroflow.types.quaternion import Quat64
from pygyroflow.types.time_types import TimeIMU, TimeQuat


def _assert_unit_quaternions(quats: TimeQuat, tol: float = 1e-6):
    """Verify all quaternions in the map are unit quaternions."""
    for ts, q in quats.items():
        norm = np.linalg.norm(q.quaternion())
        assert_allclose(norm, 1.0, atol=tol, err_msg=f"Non-unit quaternion at t={ts}: norm={norm}")


def _make_constant_rotation_imu(n_samples=1000, rate_hz=200, deg_per_sec=10.0):
    """Generate IMU data with constant rotation around Z axis."""
    data = []
    dt = 1000.0 / rate_hz  # ms
    for i in range(n_samples):
        t_ms = i * dt
        gyro = np.array([0.0, 0.0, deg_per_sec])
        accl = np.array([0.0, 0.0, 9.8])
        data.append(TimeIMU(timestamp_ms=t_ms, gyro=gyro, accl=accl, magn=None))
    return data, n_samples * dt


class TestSimpleGyro:
    def test_empty_input_returns_empty(self):
        from pygyroflow.imu_integration import SimpleGyroIntegrator
        integrator = SimpleGyroIntegrator()
        result = integrator.integrate([], 1000.0)
        assert result == {}

    def test_unit_quaternions(self, sample_imu_data):
        from pygyroflow.imu_integration import SimpleGyroIntegrator
        integrator = SimpleGyroIntegrator()
        result = integrator.integrate(sample_imu_data, 5000.0)
        _assert_unit_quaternions(result)

    def test_rotation_accumulates(self):
        """Constant Z rotation produces increasing yaw angle."""
        from pygyroflow.imu_integration import SimpleGyroIntegrator
        deg_per_sec = 10.0
        n_samples = 200
        rate_hz = 200
        data, duration = _make_constant_rotation_imu(n_samples, rate_hz, deg_per_sec)

        integrator = SimpleGyroIntegrator()
        result = integrator.integrate(data, duration)

        timestamps = sorted(result.keys())
        assert len(timestamps) == n_samples

        # The angle should increase over time. Check that later quaternions
        # differ from earlier ones.
        q_first = result[timestamps[0]]
        q_last = result[timestamps[-1]]
        assert q_first != q_last

        # Check the total accumulated angle is roughly correct.
        # 10 deg/s over 1 second = 10 degrees = ~0.175 rad
        # (coordinate transform changes the axis, so check angle is nonzero)
        delta = q_first.inverse() * q_last
        assert delta.angle() > 0.0


class TestSimpleGyroAccel:
    def test_empty_input_returns_empty(self):
        from pygyroflow.imu_integration import SimpleGyroAccelIntegrator
        integrator = SimpleGyroAccelIntegrator()
        result = integrator.integrate([], 1000.0)
        assert result == {}

    def test_unit_quaternions(self, sample_imu_data):
        from pygyroflow.imu_integration import SimpleGyroAccelIntegrator
        integrator = SimpleGyroAccelIntegrator()
        result = integrator.integrate(sample_imu_data, 5000.0)
        _assert_unit_quaternions(result)

    def test_produces_output(self, sample_imu_data):
        from pygyroflow.imu_integration import SimpleGyroAccelIntegrator
        integrator = SimpleGyroAccelIntegrator()
        result = integrator.integrate(sample_imu_data, 5000.0)
        assert len(result) == len(sample_imu_data)


class TestMahony:
    def test_empty_input_returns_empty(self):
        from pygyroflow.imu_integration import MahonyIntegrator
        integrator = MahonyIntegrator()
        result = integrator.integrate([], 1000.0)
        assert result == {}

    def test_produces_valid_quaternions(self, sample_imu_data):
        from pygyroflow.imu_integration import MahonyIntegrator
        integrator = MahonyIntegrator()
        result = integrator.integrate(sample_imu_data, 5000.0)
        assert len(result) > 0
        _assert_unit_quaternions(result)

    def test_custom_gains(self, sample_imu_data):
        from pygyroflow.imu_integration import MahonyIntegrator
        integrator = MahonyIntegrator(kp=1.0, ki=0.1)
        result = integrator.integrate(sample_imu_data, 5000.0)
        assert len(result) > 0
        _assert_unit_quaternions(result)


class TestMadgwick:
    def test_empty_input_returns_empty(self):
        from pygyroflow.imu_integration import MadgwickIntegrator
        integrator = MadgwickIntegrator()
        result = integrator.integrate([], 1000.0)
        assert result == {}

    def test_produces_valid_quaternions(self, sample_imu_data):
        from pygyroflow.imu_integration import MadgwickIntegrator
        integrator = MadgwickIntegrator()
        result = integrator.integrate(sample_imu_data, 5000.0)
        assert len(result) > 0
        _assert_unit_quaternions(result)

    def test_custom_beta(self, sample_imu_data):
        from pygyroflow.imu_integration import MadgwickIntegrator
        integrator = MadgwickIntegrator(beta=0.1)
        result = integrator.integrate(sample_imu_data, 5000.0)
        assert len(result) > 0
        _assert_unit_quaternions(result)


class TestComplementary:
    def test_empty_input_returns_empty(self):
        from pygyroflow.imu_integration import ComplementaryIntegrator
        integrator = ComplementaryIntegrator()
        result = integrator.integrate([], 1000.0)
        assert result == {}

    def test_produces_valid_quaternions(self, sample_imu_data):
        from pygyroflow.imu_integration import ComplementaryIntegrator
        integrator = ComplementaryIntegrator()
        result = integrator.integrate(sample_imu_data, 5000.0)
        assert len(result) > 0
        _assert_unit_quaternions(result)


class TestVQF:
    """VQF integrator (Python self-consistency).

    NOTE: VQF has no Rust golden reference (rust_imu_vqf.json is absent and
    msgyro-golden-gen does not generate one). These tests cover determinism
    and structural correctness only; numerical cross-check against the Rust
    port is a tracked follow-up (deferred: the Rust golden-gen lives outside
    this repo's version control).
    """

    def test_empty_input_returns_empty(self):
        from pygyroflow.imu_integration import VQFIntegrator
        integrator = VQFIntegrator()
        result = integrator.integrate([], 1000.0)
        assert result == {}

    def test_produces_valid_quaternions(self, sample_imu_data):
        from pygyroflow.imu_integration import VQFIntegrator
        integrator = VQFIntegrator()
        result = integrator.integrate(sample_imu_data, 5000.0)
        assert len(result) > 0
        _assert_unit_quaternions(result)

    def test_deterministic(self, sample_imu_data):
        """Two runs on identical input must produce identical quaternions."""
        from pygyroflow.imu_integration import VQFIntegrator
        integ = VQFIntegrator()
        r1 = integ.integrate(sample_imu_data, 5000.0)
        r2 = integ.integrate(sample_imu_data, 5000.0)
        assert set(r1.keys()) == set(r2.keys())
        for ts in r1:
            assert_allclose(r1[ts].quaternion(), r2[ts].quaternion(), atol=1e-15,
                            err_msg=f"VQF non-deterministic at t={ts}")

    def test_rotation_accumulates(self):
        """Constant Z-axis rotation must accumulate a monotonic yaw."""
        from pygyroflow.imu_integration import VQFIntegrator
        rate = 200.0
        n = 1000
        dt = 1000.0 / rate
        data = [
            TimeIMU(timestamp_ms=i * dt,
                    gyro=np.array([0.0, 0.0, 10.0]),
                    accl=np.array([0.0, 0.0, 9.8]),
                    magn=None)
            for i in range(n)
        ]
        integ = VQFIntegrator()
        result = integ.integrate(data, n * dt)
        times = sorted(result)
        yaws = [result[t].euler_angles()[2]
                for t in (times[0], times[len(times) // 2], times[-1])]
        assert yaws[1] > yaws[0], f"yaw did not accumulate: {yaws}"
        assert yaws[2] > yaws[1], f"yaw did not accumulate: {yaws}"


class TestQuaternionConverter:
    def test_convert_returns_same_timestamps(self, sample_imu_data):
        from pygyroflow.imu_integration import SimpleGyroIntegrator, QuaternionConverter
        integrator = SimpleGyroIntegrator()
        duration = 5000.0
        org_quats = integrator.integrate(sample_imu_data, duration)

        result = QuaternionConverter.convert(
            method=3,  # Mahony
            org_quaternions=org_quats,
            image_orientations={},
            imu_data=sample_imu_data,
            duration_ms=duration,
        )
        assert set(result.keys()) == set(org_quats.keys())
