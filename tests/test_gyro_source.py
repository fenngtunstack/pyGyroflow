"""Tests for GyroSource and IMUTransforms."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pygyroflow.types.quaternion import Quat64
from pygyroflow.types.time_types import TimeIMU, TimeQuat
from pygyroflow.gyro_source import GyroSource, IMUTransforms


class TestGyroSourceInit:
    def test_empty_initialization(self):
        gs = GyroSource()
        assert gs.duration_ms == 0.0
        assert len(gs.quaternions) == 0
        assert len(gs.smoothed_quaternions) == 0
        assert len(gs.raw_imu) == 0
        assert gs.integration_method == 2  # VQF default

    def test_clear(self):
        gs = GyroSource()
        gs.quaternions = {0: Quat64.identity()}
        gs.raw_imu = [TimeIMU(timestamp_ms=0, gyro=np.zeros(3))]
        gs.clear()
        assert len(gs.quaternions) == 0
        assert len(gs.raw_imu) == 0


class TestGyroSourceIntegration:
    def test_integrate_simple_gyro(self, sample_imu_data):
        gs = GyroSource()
        gs.duration_ms = 5000.0
        gs.integration_method = 3  # SimpleGyro
        gs.raw_imu = sample_imu_data
        gs.integrate()

        assert len(gs.quaternions) > 0
        # All quaternions should be unit quaternions
        for ts, q in gs.quaternions.items():
            assert_allclose(np.linalg.norm(q.quaternion()), 1.0, atol=1e-6)

    def test_integrate_mahony(self, sample_imu_data):
        gs = GyroSource()
        gs.duration_ms = 5000.0
        gs.integration_method = 5  # Mahony
        gs.raw_imu = sample_imu_data
        gs.integrate()

        assert len(gs.quaternions) > 0


class TestGyroSourceGetQuatAtTimestamp:
    def test_returns_identity_when_empty(self):
        gs = GyroSource()
        q = gs.get_quat_at_timestamp(100.0)
        assert q == Quat64.identity()

    def test_returns_interpolated_quaternion(self, sample_imu_data):
        gs = GyroSource()
        gs.duration_ms = 5000.0
        gs.integration_method = 3
        gs.raw_imu = sample_imu_data
        gs.integrate()

        # Request a quaternion at an intermediate timestamp
        ts_ms = 1250.0  # Somewhere in the middle
        q = gs.get_quat_at_timestamp(ts_ms)
        assert_allclose(np.linalg.norm(q.quaternion()), 1.0, atol=1e-6)

    def test_exact_timestamp(self, sample_imu_data):
        gs = GyroSource()
        gs.duration_ms = 5000.0
        gs.integration_method = 3
        gs.raw_imu = sample_imu_data
        gs.integrate()

        # Pick an exact timestamp that should exist
        ts_us = sorted(gs.quaternions.keys())[50]
        ts_ms = ts_us / 1000.0
        q = gs.get_quat_at_timestamp(ts_ms)
        # Should get the exact stored quaternion
        assert q == gs.quaternions[ts_us]


class TestGyroSourceOffsets:
    def test_no_offset(self):
        gs = GyroSource()
        assert gs.offset_at_video_timestamp(100.0) == 0.0

    def test_set_and_get_offset(self):
        gs = GyroSource()
        gs.set_offset(0, 10.0)
        offset = gs.offset_at_video_timestamp(0.0)
        assert_allclose(offset, 10.0, atol=0.1)

    def test_clear_offsets(self):
        gs = GyroSource()
        gs.set_offset(0, 10.0)
        gs.clear_offsets()
        assert gs.offset_at_video_timestamp(0.0) == 0.0


class TestIMUTransforms:
    def test_default_no_transform(self):
        transforms = IMUTransforms()
        assert not transforms.has_any()

    def test_orientation_transform(self):
        transforms = IMUTransforms(imu_orientation="xYZ")
        assert transforms.has_any()

        v = np.array([1.0, 2.0, 3.0])
        result = transforms.transform(v.copy(), is_acc=False)
        # x -> -X, Y -> Y, Z -> Z
        assert_allclose(result, [-1.0, 2.0, 3.0], atol=1e-10)

    def test_identity_orientation_no_transform(self):
        transforms = IMUTransforms(imu_orientation="XYZ")
        v = np.array([1.0, 2.0, 3.0])
        result = transforms.transform(v.copy(), is_acc=False)
        assert_allclose(result, [1.0, 2.0, 3.0], atol=1e-10)

    def test_bias_transform(self):
        transforms = IMUTransforms(gyro_bias=[1.0, -1.0, 0.5])
        assert transforms.has_any()

        v = np.array([0.0, 0.0, 0.0])
        result = transforms.transform(v.copy(), is_acc=False)
        assert_allclose(result, [1.0, -1.0, 0.5], atol=1e-10)

    def test_rotation_transform(self):
        transforms = IMUTransforms()
        transforms.set_imu_rotation(0.0, 0.0, 90.0)  # 90 yaw
        assert transforms.has_any()

        v = np.array([1.0, 0.0, 0.0])
        result = transforms.transform(v.copy(), is_acc=False)
        # 90 deg yaw: X -> Y
        assert_allclose(abs(result[0]), 0.0, atol=1e-6)
        assert result[1] > 0.5  # Y component should be large
