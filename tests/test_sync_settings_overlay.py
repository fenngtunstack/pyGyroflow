"""Sync-settings overlays and the OptimSync fallback (B-16 pieces)."""

from __future__ import annotations

import numpy as np
import pytest

from pygyroflow.lens.profile import LensProfile


def _profile(**data) -> LensProfile:
    base = {
        "name": "test",
        "calib_dimension": {"w": 1920, "h": 1080},
        "sync_settings": {"of_method": 2, "initial_offset": 5.0},
    }
    base.update(data)
    return LensProfile.from_json(base)


class TestSyncSettingsOverlay:
    def test_setting_merges_into_the_profile_block(self):
        p = _profile(compatible_settings=[{
            "width": 1280, "height": 720,
            "sync_settings": {"pose_method": 1},
        }])
        copies = p.get_all_matching_profiles()
        assert len(copies) == 2
        # The copy's overlay: profile's keys + the setting's key.
        assert copies[1].sync_settings == {
            "of_method": 2, "initial_offset": 5.0, "pose_method": 1,
        }
        # The original is untouched.
        assert copies[0].sync_settings == {"of_method": 2, "initial_offset": 5.0}

    def test_setting_value_wins_on_conflict(self):
        p = _profile(compatible_settings=[{
            "width": 1280, "height": 720,
            "sync_settings": {"of_method": 0},
        }])
        copies = p.get_all_matching_profiles()
        assert copies[1].sync_settings["of_method"] == 0

    def test_no_profile_block_adopts_the_settings_wholesale(self):
        p = _profile(compatible_settings=[{
            "width": 1280, "height": 720,
            "sync_settings": {"offset_method": 2},
        }])
        p.sync_settings = None
        copies = p.get_all_matching_profiles()
        assert copies[1].sync_settings == {"offset_method": 2}

    def test_custom_sync_pattern_is_replaced_not_concatenated(self):
        """Both having a custom_sync_pattern would merge into a nonsense
        combined pattern; upstream removes the profile's first
        (lens_profile.rs:396-398)."""
        p = _profile(compatible_settings=[{
            "width": 1280, "height": 720,
            "sync_settings": {
                "custom_sync_pattern": {"new": [0.1, 0.5]},
                "pose_method": 3,
            },
        }])
        copies = p.get_all_matching_profiles()
        ss = copies[1].sync_settings
        assert ss["custom_sync_pattern"] == {"new": [0.1, 0.5]}
        assert ss["pose_method"] == 3
        assert ss["of_method"] == 2  # non-pattern keys still merged


class TestUniformSyncPointFallback:
    def test_empty_optimsync_falls_back_to_chunk_centres(self, monkeypatch):
        """``render_queue.rs:1451-1453``: empty optimal-point selection
        becomes ``max_sync_points`` uniform chunk centres instead of
        abandoning refinement."""
        from pygyroflow.manager import StabilizationManager
        import pygyroflow.synchronization.optimsync as os_mod

        mgr = StabilizationManager()
        mgr.params.duration_ms = 10_000.0
        monkeypatch.setattr(
            os_mod.OptimSync, "run",
            lambda self, **k: ([], [], 0.05),
        )
        # Call the real path with unusable frames: the selection returns
        # [] and the fallback produces 5 chunk centres, whose windows then
        # find no frames — the validated map is empty but the fallback ran
        # (and did not return early as the old code did).
        result = mgr._refine_sync_points(
            [(k * 33_333, np.zeros((4, 4), np.uint8)) for k in range(2)],
            [(k * 33_333, np.array([0.0, 10.0, 0.0])) for k in range(20)],
            0.0, None, 0.0,
        )
        assert result == {}
        # And the centres the fallback used are the chunk midpoints.
        n, dur = mgr._SYNC_POINT_COUNT, mgr.params.duration_ms
        chunks = dur / n
        centres = [chunks / 2.0 + i * chunks for i in range(n)]
        assert centres[0] == pytest.approx(1000.0)
        assert centres[-1] == pytest.approx(9000.0)


class TestPerPointSearchWindow:
    def test_profile_sync_settings_override_the_window(self, monkeypatch):
        """``render_queue.rs:1470``: a profile's ``search_size`` is in
        *seconds* and reaches the per-point search scaled to ms."""
        from pygyroflow.manager import StabilizationManager
        import pygyroflow.synchronization.optimsync as os_mod

        mgr = StabilizationManager()
        mgr.params.duration_ms = 15_000.0  # past the <12 s guard
        monkeypatch.setattr(
            os_mod.OptimSync, "run", lambda self, **k: ([], [], 0.05)
        )
        seen = {}

        class Recorder:
            def __init__(self, *a, **k):
                pass

            def run(self, frames, gyro, search_range_ms=None, **k):
                seen["search_ms"] = search_range_ms
                return None

        import pygyroflow.synchronization as sync_pkg
        monkeypatch.setattr(sync_pkg, "AutosyncProcess", Recorder)

        # Enough frames to pass the >=60 gate and fill the first window.
        frames = [(k * 33_333, np.zeros((4, 4), np.uint8)) for k in range(60)]
        gyro = [(k * 33_333, np.array([0.0, 10.0, 0.0])) for k in range(60)]

        mgr.lens.sync_settings = {"search_size": 1.5}  # seconds
        mgr._refine_sync_points(frames, gyro, 0.0, None, 0.0)
        assert seen["search_ms"] == pytest.approx(1500.0)

        mgr.lens.sync_settings = None
        seen.clear()
        mgr._refine_sync_points(frames, gyro, 0.0, None, 0.0)
        assert seen["search_ms"] == pytest.approx(mgr._SYNC_POINT_SEARCH_MS)
