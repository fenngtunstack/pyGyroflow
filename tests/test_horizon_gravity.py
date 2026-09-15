# -*- coding: utf-8 -*-
"""Tests for gravity-vector horizon mode wiring (roadmap P4-4)."""

from __future__ import annotations

import numpy as np
import pytest

from pygyroflow.manager import StabilizationManager
from pygyroflow.types.quaternion import Quat64
from pygyroflow.types.time_types import TimeIMU


def _roll_quat(deg: float) -> Quat64:
    r = np.deg2rad(deg)
    return Quat64.from_quaternion(np.array([np.cos(r / 2), 0.0, 0.0, np.sin(r / 2)]))


def _mgr_with(roll_deg: float, with_accl: bool):
    mgr = StabilizationManager()
    n = 100
    mgr.params.fps = 30.0
    mgr.params.frame_count = 30
    mgr.params.duration_ms = 1000.0
    mgr.params.size = (640, 480)
    q = _roll_quat(roll_deg)
    mgr.gyro.quaternions = {i * 10_000: q for i in range(n)}
    if with_accl:
        # camera rolled by `roll_deg` -> accelerometer "up" vector rotated
        # the same amount about the roll axis
        r = np.deg2rad(roll_deg)
        up = np.array([0.0, np.sin(r), np.cos(r)])  # sensor frame reading
        mgr.gyro.raw_imu = [TimeIMU(timestamp_ms=i * 10.0, gyro=None, accl=9.81 * up) for i in range(n)]
    return mgr


class TestGravityHorizon:
    def test_flag_off_quaternion_mode(self):
        mgr = _mgr_with(10.0, with_accl=True)
        mgr.gyro.use_gravity_vectors = False
        mgr.smoothing.horizon_lock.set_horizon(100.0, 0.0, False, 0.0)
        mgr.recompute_smoothing()
        assert len(mgr.gyro.smoothed_quaternions) == 100  # ran, no crash

    def test_gravity_mode_with_accl(self):
        mgr = _mgr_with(10.0, with_accl=True)
        mgr.gyro.set_use_gravity_vectors(True)
        mgr.smoothing.horizon_lock.set_horizon(100.0, 0.0, False, 0.0)
        mgr.recompute_smoothing()  # gravity branch must execute without error
        assert len(mgr.gyro.smoothed_quaternions) == 100

    def test_gravity_requested_but_no_accl_falls_back(self):
        mgr = _mgr_with(10.0, with_accl=False)
        mgr.gyro.set_use_gravity_vectors(True)
        mgr.smoothing.horizon_lock.set_horizon(100.0, 0.0, False, 0.0)
        mgr.recompute_smoothing()  # falls back to quaternion mode
        assert len(mgr.gyro.smoothed_quaternions) == 100

    def test_gravity_corrects_roll_more_than_no_data(self):
        # quaternion mode derives roll from the (single constant) orientation
        # only; gravity mode has an independent roll reference — the corrected
        # output must differ
        def mid_corr(with_grav: bool) -> float:
            mgr = _mgr_with(12.0, with_accl=True)
            if with_grav:
                mgr.gyro.set_use_gravity_vectors(True)
            mgr.smoothing.horizon_lock.set_horizon(100.0, 0.0, False, 0.0)
            mgr.recompute_smoothing()
            sq = mgr.gyro.smoothed_quaternions
            corr = list(sq.values())[len(sq) // 2]
            w, x, y, z = corr.quaternion()
            return abs(float(np.degrees(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))))

        assert mid_corr(True) != pytest.approx(mid_corr(False))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
