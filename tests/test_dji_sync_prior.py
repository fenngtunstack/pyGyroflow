# -*- coding: utf-8 -*-
"""Tests for the DJI sync prior (two-stage narrow-then-wide search)."""

from __future__ import annotations

import pytest

from pygyroflow.manager import StabilizationManager


class TestDjiPriorConstants:
    def test_prior_values(self):
        # measured on Osmo Nano (autosync lock 7.92 ms) and Avata
        # (offset-sweep optimum +8 ms)
        assert StabilizationManager._DJI_SYNC_PRIOR_MS == 8.0
        assert StabilizationManager._DJI_SYNC_PRIOR_WINDOW_MS == 40.0


class TestDjiPriorGate:
    def test_dji_source_triggers_prior(self, monkeypatch):
        mgr = StabilizationManager()
        mgr.gyro.file_metadata.detected_source = "DJI FC8183"
        calls: list[tuple[float, float]] = []

        class _FakeProc:
            def __init__(self, **kw):
                pass

            def run(self, frames, gyro_data, search_range_ms, sample_count,
                    progress_callback, quaternions, frame_readout_time_ms,
                    initial_offset_ms=0.0):
                calls.append((search_range_ms, initial_offset_ms))
                return 9.5

        import pygyroflow.manager as mgr_mod
        monkeypatch.setattr(mgr_mod, "AutosyncProcess", _FakeProc, raising=False)
        # manager imports AutosyncProcess inside synchronize(); patch the
        # module it imports from
        import pygyroflow.synchronization as sync_mod
        monkeypatch.setattr(sync_mod, "AutosyncProcess", _FakeProc, raising=False)

        import numpy as _np
        _gray = _np.zeros((360, 480), dtype=_np.uint8)
        mgr._extract_gray_frames = lambda path, n: [(i * 33333, _gray) for i in range(20)]
        mgr._gyro_angular_velocity = lambda: [(i * 1000, [0.1, 0.0, 0.0]) for i in range(50)]
        mgr.params.size = (1920, 1080)
        mgr.gyro.quaternions = {i * 1000: object() for i in range(50)}
        mgr.lens.get_camera_matrix = lambda size=None: None

        off = mgr.synchronize(input_path="fake.mp4")
        assert off == pytest.approx(9.5)
        # narrow window around the prior, exactly one call (success)
        assert calls == [(40.0, 8.0)]

    def test_non_dji_source_no_prior(self, monkeypatch):
        mgr = StabilizationManager()
        mgr.gyro.file_metadata.detected_source = "GoPro HERO6 Black"
        calls: list[tuple[float, float]] = []

        class _FakeProc:
            def __init__(self, **kw):
                pass

            def run(self, frames, gyro_data, search_range_ms, sample_count,
                    progress_callback, quaternions, frame_readout_time_ms,
                    initial_offset_ms=0.0):
                calls.append((search_range_ms, initial_offset_ms))
                return 41.0

        import pygyroflow.synchronization as sync_mod
        monkeypatch.setattr(sync_mod, "AutosyncProcess", _FakeProc, raising=False)

        import numpy as _np
        _gray = _np.zeros((360, 480), dtype=_np.uint8)
        mgr._extract_gray_frames = lambda path, n: [(i * 33333, _gray) for i in range(20)]
        mgr._gyro_angular_velocity = lambda: [(i * 1000, [0.1, 0.0, 0.0]) for i in range(50)]
        mgr.params.size = (1920, 1080)
        mgr.gyro.quaternions = {i * 1000: object() for i in range(50)}
        mgr.lens.get_camera_matrix = lambda size=None: None

        off = mgr.synchronize(input_path="fake.mp4")
        assert off == pytest.approx(41.0)
        # full window, zero prior
        assert calls[-1] == (500.0, 0.0)

    def test_narrow_failure_falls_back_to_wide(self, monkeypatch):
        mgr = StabilizationManager()
        mgr.gyro.file_metadata.detected_source = "DJI FC8183"
        calls: list[tuple[float, float]] = []

        class _FakeProc:
            def __init__(self, **kw):
                pass

            def run(self, frames, gyro_data, search_range_ms, sample_count,
                    progress_callback, quaternions, frame_readout_time_ms,
                    initial_offset_ms=0.0):
                calls.append((search_range_ms, initial_offset_ms))
                return None if search_range_ms < 100 else 12.0

        import pygyroflow.synchronization as sync_mod
        monkeypatch.setattr(sync_mod, "AutosyncProcess", _FakeProc, raising=False)

        import numpy as _np
        _gray = _np.zeros((360, 480), dtype=_np.uint8)
        mgr._extract_gray_frames = lambda path, n: [(i * 33333, _gray) for i in range(20)]
        mgr._gyro_angular_velocity = lambda: [(i * 1000, [0.1, 0.0, 0.0]) for i in range(50)]
        mgr.params.size = (1920, 1080)
        mgr.gyro.quaternions = {i * 1000: object() for i in range(50)}
        mgr.lens.get_camera_matrix = lambda size=None: None

        off = mgr.synchronize(input_path="fake.mp4")
        assert off == pytest.approx(12.0)
        assert calls == [(40.0, 8.0), (500.0, 0.0)]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
