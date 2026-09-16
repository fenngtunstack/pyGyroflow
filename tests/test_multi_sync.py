# -*- coding: utf-8 -*-
"""Tests for multi-point sync refinement (manager.synchronize wiring).

The orchestration itself needs real footage; these tests pin the outlier
rejection logic and the eligibility guards, which are the parts that can
silently degrade stabilization quality when wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from pygyroflow.manager import StabilizationManager


class TestValidSyncPoints:
    def test_global_point_always_present(self):
        out = StabilizationManager._valid_sync_points({}, global_offset=8.0)
        assert out == {0: 8.0}

    def test_inliers_kept_outliers_dropped(self):
        pts = {
            5_000_000: 9.5,     # +1.5 from global -> keep
            10_000_000: 55.0,   # +47 from global -> drop (mislock, not drift)
            15_000_000: -12.0,  # -20 from global -> keep
            20_000_000: -45.0,  # -53 -> drop
        }
        out = StabilizationManager._valid_sync_points(pts, global_offset=8.0)
        assert 0 in out and 5_000_000 in out and 15_000_000 in out
        assert 10_000_000 not in out and 20_000_000 not in out

    def test_non_finite_dropped(self):
        pts = {5_000_000: float("nan"), 6_000_000: float("inf"), 7_000_000: 10.0}
        out = StabilizationManager._valid_sync_points(pts, global_offset=8.0)
        assert set(out) == {0, 7_000_000}

    def test_single_point_map_falls_back_to_global_only(self):
        # caller enables multi-point only when len >= 2; a lone inlier is
        # worthless for a drift curve and must not linger in the map
        pts = {5_000_000: 9.0}
        out = StabilizationManager._valid_sync_points(pts, global_offset=8.0)
        assert len(out) == 2  # {0: 8.0, 5_000_000: 9.0} — caller gates on >=2


class TestRefineEligibility:
    def test_short_clip_skipped(self):
        mgr = StabilizationManager()
        mgr.params.duration_ms = 9_000  # < 12 s
        assert mgr._refine_sync_points(frames=[], gyro_data=[], global_offset=0.0,
                                       quaternions=None, frame_readout_time_ms=0.0) == {}

    def test_too_few_frames_skipped(self):
        mgr = StabilizationManager()
        mgr.params.duration_ms = 30_000
        assert mgr._refine_sync_points(frames=[(0, np.zeros((4, 4), np.uint8))] * 30,
                                       gyro_data=[], global_offset=0.0,
                                       quaternions=None, frame_readout_time_ms=0.0) == {}


class TestDriftSignificance:
    def test_clean_ramp_accepted(self):
        pts = {0: 40.0, 4_000_000: 45.0, 8_000_000: 50.0, 12_000_000: 55.0}
        assert StabilizationManager._drift_significant(pts) is True

    def test_flat_with_one_stray_rejected(self):
        # Hero6-style: constant ~41 with one 19 ms excursion -> fake ramp
        pts = {0: 41.0, 11_283_424: 46.8, 18_928_089: 42.2, 25_526_642: 39.2, 30_515_792: 19.0}
        assert StabilizationManager._drift_significant(pts) is False

    def test_flat_clean_rejected(self):
        pts = {0: 41.0, 5_000_000: 42.0, 10_000_000: 40.5, 15_000_000: 41.5}
        assert StabilizationManager._drift_significant(pts) is False

    def test_two_points_rejected(self):
        pts = {0: 40.0, 10_000_000: 60.0}
        assert StabilizationManager._drift_significant(pts) is False


class TestPiecewiseLookup:
    def test_multi_point_offsets_interpolate(self):
        # the GyroSource lookup the render path uses: piecewise-linear
        # between sync points, clamped at the ends
        mgr = StabilizationManager()
        mgr.gyro.set_offsets({0: 10.0, 10_000_000: 20.0})  # us keys, ms values
        at_start = mgr.gyro.offset_at_video_timestamp(0.0)
        at_mid = mgr.gyro.offset_at_video_timestamp(5000.0)   # ms
        at_end = mgr.gyro.offset_at_video_timestamp(60000.0)
        assert at_start == pytest.approx(10.0, abs=0.05)  # 1us clamp nudge at keys[0]
        # interpolation runs on the ADJUSTED grid (key = ts + offset, upstream
        # semantics), so mid is 15.0 minus the key-shift skew (~0.015 ms here)
        assert at_mid == pytest.approx(15.0, abs=0.05)
        assert at_end == pytest.approx(20.0, abs=0.05)  # clamped past last point


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

class TestRsSyncOffsetArithmetic:
    """The rolling-shutter offset is delay plus a readout correction.

    Upstream (rs_sync.rs::full_sync) computes ``-delay - readout/2`` and
    rejects anything more than 90% of the search radius away from the initial
    guess. The readout half was missing here, which biases every synced
    offset by a constant — tens of milliseconds on a slow sensor.
    """

    class _Frame:
        def __init__(self, frame_no, timestamp_us, n=20):
            self.frame_no = frame_no
            self.timestamp_us = timestamp_us
            self.prev_points = np.zeros((n, 2), np.float32)
            self.curr_points = np.ones((n, 2), np.float32)

    @staticmethod
    def _proc(delay_ms, *, cost=1.0, neighbour_cost=100.0):
        """An AutosyncProcess whose RS search returns *delay_ms* verbatim."""
        from pygyroflow.synchronization import autosync as mod
        from pygyroflow.synchronization.find_offset import rs_sync as rs_mod

        class _StubRs:
            def __init__(self, *a, **k):
                pass

            def add_track_from_frames(self, *a, **k):
                pass

            def full_sync(self, **kwargs):
                self.kwargs = kwargs
                return cost, delay_ms

            def _compute_cost(self, *a, **k):
                return neighbour_cost

        proc = mod.AutosyncProcess(fps=30.0)
        proc._pose_estimator = type(
            "PE",
            (),
            {"get_frame_results": lambda self: {
                i: TestRsSyncOffsetArithmetic._Frame(i, i * 33_000)
                for i in range(6)
            }},
        )()
        rs_mod.RollingShutterSync = _StubRs
        return proc

    def _frames(self, n=6):
        return [(i * 33_000, np.zeros((64, 64), np.uint8)) for i in range(n)]

    def test_readout_half_is_subtracted(self):
        proc = self._proc(delay_ms=10.0)
        offset = proc._rs_sync_offset(
            self._frames(), {}, frame_readout_time_ms=14.0, search_range_ms=500.0
        )
        assert offset == pytest.approx(-10.0 - 7.0)

    def test_global_shutter_is_plain_negation(self):
        proc = self._proc(delay_ms=10.0)
        offset = proc._rs_sync_offset(
            self._frames(), {}, frame_readout_time_ms=0.0, search_range_ms=500.0
        )
        assert offset == pytest.approx(-10.0)

    def test_offset_near_the_search_limit_is_rejected(self):
        """|delay - initial| >= 90% of the radius is the window edge, not a match."""
        proc = self._proc(delay_ms=240.0)  # radius 250 -> limit 225
        assert proc._rs_sync_offset(
            self._frames(), {}, frame_readout_time_ms=0.0, search_range_ms=500.0
        ) is None

    def test_offset_inside_the_limit_is_accepted(self):
        proc = self._proc(delay_ms=200.0)  # < 225
        assert proc._rs_sync_offset(
            self._frames(), {}, frame_readout_time_ms=0.0, search_range_ms=500.0
        ) == pytest.approx(-200.0)

    def test_initial_offset_shifts_the_acceptance_window(self):
        """The guard measures distance from the initial guess, not from zero."""
        proc = self._proc(delay_ms=300.0)
        # initial_offset 100 -> initial_delay -100 -> |300 - (-100)| = 400 > 225
        assert proc._rs_sync_offset(
            self._frames(), {}, frame_readout_time_ms=0.0,
            search_range_ms=500.0, initial_offset_ms=100.0,
        ) is None
        # initial_offset -300 -> initial_delay 300 -> distance 0
        assert proc._rs_sync_offset(
            self._frames(), {}, frame_readout_time_ms=0.0,
            search_range_ms=500.0, initial_offset_ms=-300.0,
        ) == pytest.approx(-300.0)
