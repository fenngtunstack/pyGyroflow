"""Tests for adaptive zoom FOV computation."""

import numpy as np
import pytest

from pygyroflow.types.quaternion import Quat64
from pygyroflow.stabilization import ComputeParams
from pygyroflow.zooming import calculate_fovs, ZoomMethod


def _make_test_params():
    """Create params with no rotation (identity quaternions)."""
    quats = {}
    smoothed = {}
    n = 50
    for i in range(n):
        ts = i * 20000  # 20ms apart = 50fps
        q = Quat64.identity()
        quats[ts] = q
        smoothed[ts] = q

    return ComputeParams(
        width=1920, height=1080, output_width=1920, output_height=1080,
        frame_count=n, scaled_fps=50.0, scaled_duration_ms=1000.0,
        quaternions=quats, smoothed_quaternions=smoothed,
        fovs=[], fov_scale=1.0,
        camera_matrix=np.array([
            [1000.0, 0.0, 960.0],
            [0.0, 1000.0, 540.0],
            [0.0, 0.0, 1.0],
        ]),
    )


class TestCalculateFovs:
    def test_empty_timestamps(self):
        params = _make_test_params()
        fovs, minimal = calculate_fovs(params, [])
        assert fovs == []
        assert minimal == []

    def test_disabled_zoom(self):
        params = _make_test_params()
        params.adaptive_zoom_window = 0.0  # Disabled

        timestamps = [(i, i * 20.0) for i in range(50)]
        fovs, minimal = calculate_fovs(params, timestamps)

        assert len(fovs) == 50
        assert all(f == 1.0 for f in fovs)

    def test_static_zoom(self):
        params = _make_test_params()
        params.adaptive_zoom_window = -1.0  # Static

        timestamps = [(i, i * 20.0) for i in range(50)]
        fovs, minimal = calculate_fovs(params, timestamps)

        assert len(fovs) == 50
        # All FOVs should be the same value in static mode
        assert len(set(fovs)) == 1

    def test_dynamic_zoom_envelope(self):
        params = _make_test_params()
        params.adaptive_zoom_window = 2.0

        timestamps = [(i, i * 20.0) for i in range(50)]
        fovs, minimal = calculate_fovs(params, timestamps, ZoomMethod.EnvelopeFollower)

        assert len(fovs) == 50
        # FOVs should be positive
        assert all(f > 0.0 for f in fovs)

    def test_dynamic_zoom_gaussian(self):
        params = _make_test_params()
        params.adaptive_zoom_window = 2.0

        timestamps = [(i, i * 20.0) for i in range(50)]
        fovs, minimal = calculate_fovs(params, timestamps, ZoomMethod.GaussianFilter)

        assert len(fovs) == 50
        assert all(f > 0.0 for f in fovs)

class TestFovEstimatorParams:
    """The FOV estimator's working params carry the caller's keyframes.

    Upstream clones the whole ComputeParams (zooming/mod.rs), so the keyframe
    manager and the sync-offset map come with it. The copy built here used to
    let both fall back to their defaults — an empty KeyframeManager — which
    meant every keyframed zooming parameter was silently ignored.
    """

    @staticmethod
    def _captured(compute_params, timestamps):
        from pygyroflow import zooming as zooming_mod

        captured = {}
        real = zooming_mod.FovIterative

        class _Spy(real):  # type: ignore[misc, valid-type]
            def __init__(self, params, org_output_size):
                captured["params"] = params
                super().__init__(params, org_output_size)

        zooming_mod.FovIterative = _Spy
        try:
            calculate_fovs(compute_params, timestamps)
        finally:
            zooming_mod.FovIterative = real
        return captured["params"]

    def _timestamps(self, n=50):
        return [(i, i * 20.0) for i in range(n)]

    def test_keyframes_are_forwarded(self):
        from pygyroflow.keyframes import KeyframeManager, KeyframeType

        params = _make_test_params()
        params.keyframes = KeyframeManager()
        params.keyframes.set_keyframe(KeyframeType.ZoomingCenterX, 0, 0.25)

        forwarded = self._captured(params, self._timestamps())
        assert forwarded.keyframes is params.keyframes
        assert forwarded.keyframes.value_at_video_timestamp(
            KeyframeType.ZoomingCenterX, 0.0
        ) == pytest.approx(0.25)

    def test_sync_offsets_are_forwarded(self):
        params = _make_test_params()
        params.sync_offsets_adjusted = {0: 12.0, 1_000_000: 12.0}
        forwarded = self._captured(params, self._timestamps())
        assert forwarded.sync_offsets_adjusted == {0: 12.0, 1_000_000: 12.0}

    def test_forwarded_params_do_not_alias_the_originals(self):
        """The working copy resets fov state; the caller's must stay intact."""
        params = _make_test_params()
        params.fovs = [1.5] * 50
        forwarded = self._captured(params, self._timestamps())
        assert forwarded.fovs == []
        assert params.fovs == [1.5] * 50


class TestFramesPerWindow:
    """A sub-frame adaptive-zoom window stays a sub-frame window."""

    def test_no_lower_clamp(self):
        from pygyroflow.zooming.zoom_dynamic import _get_frames_per_window

        params = _make_test_params()
        params.scaled_fps = 100.0
        params.adaptive_zoom_window = 0.01  # 1 frame at 100 fps
        assert _get_frames_per_window(params) == 1

    def test_window_is_forced_odd(self):
        from pygyroflow.zooming.zoom_dynamic import _get_frames_per_window

        params = _make_test_params()
        params.scaled_fps = 100.0
        params.adaptive_zoom_window = 0.04  # 4 frames -> 5
        assert _get_frames_per_window(params) == 5
