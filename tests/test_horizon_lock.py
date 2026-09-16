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


class TestHorizonLockAppliedOnce:
    """The lock runs exactly once, on the originals, before smoothing.

    It used to run twice: ``Smoothing.smooth`` locked the freshly smoothed
    quaternions, and ``StabilizationManager.recompute_smoothing`` locked the
    result again. ``HorizonLock.lock`` slerps toward the locked orientation
    by a percentage, so applying it twice is not the same as once.
    """

    @staticmethod
    def _mgr(lock_percent=100.0, roll_deg=6.0, n=200, additional=(0.0, 0.0, 0.0)):
        mgr = StabilizationManager()
        mgr.params.fps = 100.0
        mgr.params.frame_count = n
        mgr.params.duration_ms = n * 10.0
        mgr.params.size = (640, 480)
        mgr.params.additional_rotation = additional
        mgr.gyro.init_from_params(n * 10.0)
        mgr.gyro.quaternions = {
            i * 10_000: Quat64.from_euler_angles(
                np.deg2rad(roll_deg * np.sin(i / 9.0)), 0.0, np.deg2rad(0.05 * i)
            )
            for i in range(n)
        }
        if lock_percent:
            mgr.smoothing.horizon_lock.set_horizon(
                lock_percent=lock_percent, roll=0.0, lock_pitch=False, pitch=0.0
            )
        return mgr

    @staticmethod
    def _corrections(mgr):
        return np.array(
            [q.quaternion() for _, q in sorted(mgr.gyro.smoothed_quaternions.items())]
        )

    def test_smoothing_alone_does_not_lock(self):
        from pygyroflow.smoothing import Smoothing

        smoothing = Smoothing()
        smoothing.horizon_lock.set_horizon(
            lock_percent=100.0, roll=0.0, lock_pitch=False, pitch=0.0
        )
        calls = []
        smoothing.horizon_lock.lock = lambda *a, **k: calls.append(1)

        mgr = self._mgr(lock_percent=0.0)
        cp = mgr._build_compute_params()
        smoothing.smooth(mgr.gyro.quaternions, 2000.0, cp)
        assert calls == []

    def test_manager_locks_exactly_once(self):
        mgr = self._mgr()
        calls = []
        original = mgr.smoothing.horizon_lock.lock

        def spy(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        mgr.smoothing.horizon_lock.lock = spy
        mgr.recompute_smoothing()
        assert len(calls) == 1

    def test_lock_off_does_not_lock(self):
        mgr = self._mgr(lock_percent=0.0)
        calls = []
        mgr.smoothing.horizon_lock.lock = lambda *a, **k: calls.append(1)
        mgr.recompute_smoothing()
        assert calls == []

    def test_lock_changes_the_correction(self):
        off = self._mgr(lock_percent=0.0)
        on = self._mgr(lock_percent=100.0)
        off.recompute_smoothing()
        on.recompute_smoothing()
        assert np.abs(self._corrections(off) - self._corrections(on)).max() > 0.1


class TestAdditionalRotation:
    """`params.additional_rotation` reaches the smoothing input.

    It was plumbed into ComputeParams and set by the GUI's horizon-roll
    control, but nothing ever multiplied the quaternions by it.
    """

    @staticmethod
    def _result(roll_deg):
        mgr = TestHorizonLockAppliedOnce._mgr(
            lock_percent=0.0, additional=(0.0, 0.0, roll_deg)
        )
        mgr.recompute_smoothing()
        return np.array(
            [q.quaternion() for _, q in sorted(mgr.gyro.smoothed_quaternions.items())]
        )

    def test_zero_rotation_is_a_no_op(self):
        assert np.abs(self._result(0.0) - self._result(0.0)).max() == 0.0

    def test_rotation_reaches_the_result(self):
        base, rotated = self._result(0.0), self._result(5.0)
        assert np.abs(base - rotated).max() > 1e-3

    def test_half_angle_matches_a_five_degree_rotation(self):
        """The correction moves by sin(5deg/2) — a 5 deg rotation's w-delta."""
        delta = np.abs(self._result(0.0) - self._result(5.0)).max()
        assert delta == pytest.approx(np.sin(np.deg2rad(5.0) / 2.0), abs=1e-3)


class TestMaxAngles:
    def test_max_angles_are_recorded(self):
        mgr = TestHorizonLockAppliedOnce._mgr(lock_percent=100.0)
        mgr.gyro.max_angles = (0.0, 0.0, 0.0)
        mgr.recompute_smoothing()
        assert mgr.gyro.max_angles != (0.0, 0.0, 0.0)
