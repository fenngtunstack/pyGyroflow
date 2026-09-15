# -*- coding: utf-8 -*-
"""Tests for horizon lock wiring (smoothing post-processor + CLI flag)."""

from __future__ import annotations

import numpy as np
import pytest

from pygyroflow.manager import StabilizationManager
from pygyroflow.types.quaternion import Quat64


def _roll_quat(deg: float) -> Quat64:
    r = np.deg2rad(deg)
    return Quat64.from_quaternion(np.array([np.cos(r / 2), 0.0, 0.0, np.sin(r / 2)]))


class TestHorizonLockWiring:
    def _mgr_with_rolled_gyro(self):
        mgr = StabilizationManager()
        # constant 10 deg roll over 1 s at 100 Hz
        n = 100
        mgr.params.fps = 30.0
        mgr.params.frame_count = 30
        mgr.params.duration_ms = 1000.0
        mgr.params.size = (640, 480)
        q = _roll_quat(10.0)
        mgr.gyro.quaternions = {i * 10_000: q for i in range(n)}
        return mgr

    def test_off_by_default_no_change(self):
        mgr = self._mgr_with_rolled_gyro()
        mgr.recompute_smoothing()
        # with lock off the correction quats exist and are non-trivial
        assert len(mgr.gyro.smoothed_quaternions) == 100

    def test_lock_flattens_constant_roll(self):
        mgr = self._mgr_with_rolled_gyro()
        mgr.smoothing.horizon_lock.set_horizon(lock_percent=100.0, roll=0.0,
                                               lock_pitch=False, pitch=0.0)
        mgr.recompute_smoothing()
        sq = mgr.gyro.smoothed_quaternions
        assert len(sq) == 100
        # correction = sm^-1 * org; horizon-locked sm removes the constant
        # roll, so the correction now CONTAINS that roll (frame gets rotated
        # back to level). Extract the roll component of the correction:
        rolls = []
        for ts, corr in list(sq.items())[::10]:
            w, x, y, z = corr.quaternion()
            # roll from quaternion (ZYX convention, yaw/pitch ~0 here)
            roll = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            rolls.append(np.degrees(roll))
        rolls = np.array(rolls)
        # a 10 deg constant roll should be fully removed by the lock ->
        # the correction carries ~-10..10 deg of roll energy, definitely NOT
        # the near-zero it carries when the lock is off
        assert np.abs(rolls).max() > 5.0

    def test_partial_lock_between_off_and_full(self):
        mgr50 = self._mgr_with_rolled_gyro()
        mgr50.smoothing.horizon_lock.set_horizon(lock_percent=50.0, roll=0.0,
                                                 lock_pitch=False, pitch=0.0)
        mgr50.recompute_smoothing()

        mgr100 = self._mgr_with_rolled_gyro()
        mgr100.smoothing.horizon_lock.set_horizon(lock_percent=100.0, roll=0.0,
                                                  lock_pitch=False, pitch=0.0)
        mgr100.recompute_smoothing()

        def mid_roll(m):
            corr = list(m.gyro.smoothed_quaternions.values())[len(m.gyro.smoothed_quaternions) // 2]
            w, x, y, z = corr.quaternion()
            return abs(np.degrees(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))))

        assert mid_roll(mgr50) < mid_roll(mgr100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
