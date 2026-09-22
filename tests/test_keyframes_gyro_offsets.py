"""Keyframe lookups on the gyro timeline (gap item D-10).

``value_at_gyro_timestamp`` used to delegate straight to the video-time
lookup — its own docstring admitted the offset was "not yet implemented".
Upstream (``keyframes.rs:204-208``) adds the sync offset at the query
instant first, so once an auto-sync offset is known, keyframed values
read on the gyro timeline move with the footage. Every smoothing-path
keyframe lookup (video speed, horizon lock) goes through this method.
"""

from __future__ import annotations

import pytest

from pygyroflow.keyframes.manager import KeyframeManager
from pygyroflow.keyframes.types import Easing, KeyframeType
from pygyroflow.util import offset_at_timestamp


def _manager_with_centers() -> KeyframeManager:
    """ZoomingCenterX: 1.0 at t=1000 ms, 2.0 at t=2000 ms (linear between)."""
    mgr = KeyframeManager()
    for ts_us, value in ((1_000_000, 1.0), (2_000_000, 2.0)):
        mgr.set_keyframe(
            KeyframeType.ZoomingCenterX, ts_us, value, easing=Easing.NoEasing
        )
    return mgr


class TestTheGyroTimelineLookup:
    def test_without_offsets_it_is_the_video_lookup(self):
        mgr = _manager_with_centers()
        assert mgr.value_at_gyro_timestamp(
            KeyframeType.ZoomingCenterX, 1500.0
        ) == pytest.approx(1.5)
        assert mgr.value_at_video_timestamp(
            KeyframeType.ZoomingCenterX, 1500.0
        ) == pytest.approx(1.5)

    def test_the_offset_moves_the_query_onto_the_video_timeline(self):
        """Gyro 250 ms behind the video: a gyro-time query of 1000 ms must
        read the video curve at 1250 ms (``keyframes.rs:206``)."""
        mgr = _manager_with_centers()
        mgr.update_gyro({0: 250.0})
        assert mgr.value_at_gyro_timestamp(
            KeyframeType.ZoomingCenterX, 1000.0
        ) == pytest.approx(1.25)

    def test_a_single_offset_applies_everywhere(self):
        """Upstream's single-key path returns that offset for every query —
        one sync point means one constant shift. Video time 1000 + 300 =
        1300 ms -> a third of the way from 1.0 to 2.0."""
        mgr = _manager_with_centers()
        mgr.update_gyro({5_000_000: 300.0})
        assert mgr.value_at_gyro_timestamp(
            KeyframeType.ZoomingCenterX, 1000.0
        ) == pytest.approx(1.3)

    def test_update_gyro_replaces_the_whole_curve(self):
        mgr = _manager_with_centers()
        mgr.update_gyro({0: 100.0})
        mgr.update_gyro({0: 500.0, 10_000_000: 500.0})
        assert mgr.gyro_offsets == {0: 500.0, 10_000_000: 500.0}


class TestLifecycle:
    def test_clear_drops_the_offsets_too(self):
        """Upstream's ``clear`` is ``*self = Self::new()``: the mirrored
        offsets go with everything else (D-19). A lookup afterwards raises
        no stale-offset effects — there is simply nothing left."""
        mgr = _manager_with_centers()
        mgr.update_gyro({0: 100.0})
        mgr.clear()
        assert mgr.gyro_offsets == {}
        # Fresh keyframes on a cleared manager read on the video timeline.
        mgr.set_keyframe(
            KeyframeType.ZoomingCenterX, 1_000_000, 3.0, easing=Easing.NoEasing
        )
        assert mgr.value_at_gyro_timestamp(
            KeyframeType.ZoomingCenterX, 1000.0
        ) == pytest.approx(3.0)

    def test_serialize_roundtrips_the_offsets(self):
        mgr = _manager_with_centers()
        mgr.update_gyro({1_000_000: 12.5})
        data = mgr.serialize()
        assert data["gyro_offsets"] == {"1000000": 12.5}

        fresh = KeyframeManager()
        fresh.deserialize(data)
        assert fresh.gyro_offsets == {1_000_000: 12.5}
        # Video time 1000 + 12.5 = 1012.5 ms -> 1.25% of the way to 2.0.
        assert fresh.value_at_gyro_timestamp(
            KeyframeType.ZoomingCenterX, 1000.0
        ) == pytest.approx(1.0125)

    def test_deserialize_without_offsets_starts_clean(self):
        mgr = KeyframeManager()
        mgr.deserialize({"ZoomingCenterX": {"1000000": {"id": 1, "value": 1.0,
                                                        "easing": 0}}})
        assert mgr.gyro_offsets == {}


class TestTheSharedHelper:
    def test_util_matches_the_gyro_source_path(self):
        """One implementation in ``util``, with the gyro source delegating —
        the two call sites must agree by construction."""
        from pygyroflow.gyro_source.source import GyroSource

        offsets = {1_000_000: 10.0, 3_000_000: 30.0}
        for ts in (-500.0, 1000.0, 1500.0, 2500.0, 5000.0):
            assert offset_at_timestamp(offsets, ts) == (
                GyroSource.offset_at_timestamp(offsets, ts)
            )

    def test_interpolation_clamps_at_the_ends(self):
        offsets = {1_000_000: 10.0, 3_000_000: 30.0}
        # Before the first key: clamped to the first offset, not extrapolated.
        assert offset_at_timestamp(offsets, -1000.0) == pytest.approx(10.0)
        assert offset_at_timestamp(offsets, 1000.0) == pytest.approx(10.0)
        assert offset_at_timestamp(offsets, 2000.0) == pytest.approx(20.0)
        assert offset_at_timestamp(offsets, 3000.0) == pytest.approx(30.0)
        # After the last key: clamped, not runaway.
        assert offset_at_timestamp(offsets, 9000.0) == pytest.approx(30.0)
        assert offset_at_timestamp({}, 1000.0) == 0.0


class TestTheManagerWiring:
    def test_synchronize_wrapper_mirrors_into_keyframes(self):
        """``StabilizationManager.set_gyro_offset`` must do the gyro mutation
        *and* the mirror — that mirror is what makes the fix reach the
        smoothing path."""
        from pygyroflow.manager import StabilizationManager

        mgr = StabilizationManager()
        mgr.set_gyro_offset(0, 42.0)
        assert mgr.keyframes.gyro_offsets == {0: 42.0}
        mgr.remove_gyro_offset(0)
        assert mgr.keyframes.gyro_offsets == {}
        mgr.set_gyro_offset(0, 7.0)
        mgr.clear_gyro_offsets()
        assert mgr.keyframes.gyro_offsets == {}

    def test_keyframes_shared_with_compute_params_see_the_mirror(self):
        """The smoothing path reads the keyframe manager handed to
        ComputeParams — the same object the manager mirrors into."""
        from pygyroflow.manager import StabilizationManager

        mgr = StabilizationManager()
        mgr.set_gyro_offset(0, 42.0)
        cp = mgr._build_compute_params()
        assert cp.keyframes is mgr.keyframes
        assert mgr.keyframes.gyro_offsets == {0: 42.0}
        assert cp.keyframes.gyro_offsets == {0: 42.0}
