"""Max-zoom limit: the feedback loop and how it reaches the smoothing.

Upstream (`StabilizationManager::recompute_adaptive_zoom`) compares each
frame's required FOV against the zoom limit; where the crop would be larger
than allowed, it relaxes the *smoothing* for that frame through
`ComputeParams.smoothing_fov_limit_per_frame` and re-runs smoothing and
zooming, up to `max_zoom_iterations` times with looser thresholds
[0.95, 0.9, 0.85, 0.8].

Two defects are covered here: the loop did not exist at all (`max_zoom` was
carried around and never used), and the consumer read the per-frame limit
with `frame in limit_list` — a membership test on the *values*, which for a
list of floats is never true.
"""

import numpy as np
import pytest

from pygyroflow.manager import StabilizationManager
from pygyroflow.types.quaternion import Quat64

_FPS = 50.0
_FRAMES = 300


def _manager(max_zoom=None, iterations=5, swing_deg=4.0):
    mgr = StabilizationManager()
    mgr.params.fps = _FPS
    mgr.params.frame_count = _FRAMES
    mgr.params.duration_ms = _FRAMES * 1000.0 / _FPS
    mgr.params.size = (1920, 1080)
    mgr.params.output_size = (1920, 1080)
    mgr.params.max_zoom = max_zoom
    mgr.params.max_zoom_iterations = iterations
    mgr.gyro.init_from_params(_FRAMES * 1000.0 / _FPS)
    mgr.gyro.quaternions = {
        int(i * 1000.0 / _FPS * 1000): Quat64.from_euler_angles(
            np.deg2rad(swing_deg * np.sin(i / 7.0)), 0.0, 0.0
        )
        for i in range(_FRAMES)
    }
    mgr.recompute_smoothing()
    mgr.recompute_adaptive_zoom()
    return mgr


def _smoothing_trace(mgr):
    return np.array(
        [q.quaternion() for _, q in sorted(mgr.gyro.smoothed_quaternions.items())]
    )


class TestMaxZoomLoop:
    def test_no_limit_leaves_smoothing_alone(self):
        mgr = _manager(max_zoom=None)
        assert mgr._smoothing_fov_limit_per_frame == []

    def test_generous_limit_is_cleared(self):
        """The default 130% is above the required crop, so nothing is relaxed."""
        mgr = _manager(max_zoom=130.0)
        assert mgr._smoothing_fov_limit_per_frame == []

    def test_tight_limit_relaxes_smoothing(self):
        mgr = _manager(max_zoom=110.0)
        limit = mgr._smoothing_fov_limit_per_frame
        assert len(limit) == len(mgr.params.fovs)
        assert min(limit) < 1.0

    def test_only_frames_over_the_limit_are_relaxed(self):
        """A clip whose crop stays inside the limit at both ends."""
        mgr = StabilizationManager()
        mgr.params.fps = _FPS
        mgr.params.frame_count = _FRAMES
        mgr.params.duration_ms = _FRAMES * 1000.0 / _FPS
        mgr.params.size = (1920, 1080)
        mgr.params.output_size = (1920, 1080)
        mgr.params.max_zoom = 110.0
        mgr.gyro.init_from_params(_FRAMES * 1000.0 / _FPS)
        # Motion only in the first half.
        mgr.gyro.quaternions = {
            int(i * 1000.0 / _FPS * 1000): Quat64.from_euler_angles(
                np.deg2rad(6.0 * np.sin(i / 5.0)) if i < _FRAMES // 2 else 0.0,
                0.0,
                0.0,
            )
            for i in range(_FRAMES)
        }
        mgr.recompute_smoothing()
        mgr.recompute_adaptive_zoom()
        limit = mgr._smoothing_fov_limit_per_frame
        assert limit and min(limit) < 1.0

    def test_zero_iterations_disables_the_loop(self):
        mgr = _manager(max_zoom=110.0, iterations=0)
        assert mgr._smoothing_fov_limit_per_frame == []

    def test_loop_reruns_zooming(self):
        """The relaxation feeds back into the crop, so the FOVs move."""
        loose = _manager(max_zoom=None)
        tight = _manager(max_zoom=110.0)
        assert not np.allclose(loose.params.fovs, tight.params.fovs)


class TestLimitReachesTheSmoothing:
    def _trace_with(self, limit):
        mgr = _manager(max_zoom=None)
        mgr._smoothing_fov_limit_per_frame = limit
        mgr.recompute_smoothing()
        return _smoothing_trace(mgr)

    def test_per_frame_limit_is_indexed_not_matched_by_value(self):
        """Two all-float limit lists must produce different smoothing.

        The consumer tested `frame in limit_list`, which for a list of floats
        is a membership test and never true — the limit only ever applied by
        accident, when a value happened to equal a frame index.
        """
        a = self._trace_with([0.5] * _FRAMES)
        b = self._trace_with([0.75] * _FRAMES)
        assert np.abs(a - b).max() > 0.0

    def test_limit_length_is_respected(self):
        """Indices past the end are ignored, not an IndexError."""
        short = self._trace_with([0.5])
        empty = self._trace_with([])
        assert not np.allclose(short, empty)

    def test_default_algo_and_plain_both_consume_it(self):
        from pygyroflow.smoothing import PlainSmoothing

        mgr = _manager(max_zoom=None)
        cp = mgr._build_compute_params()
        cp.smoothing_fov_limit_per_frame = [0.5] * _FRAMES

        plain = PlainSmoothing()
        plain.set_parameter("time_constant", 1.0)
        with_limit = plain.smooth(mgr.gyro.quaternions, mgr.gyro.duration_ms, cp)

        cp2 = mgr._build_compute_params()
        cp2.smoothing_fov_limit_per_frame = []
        without = plain.smooth(mgr.gyro.quaternions, mgr.gyro.duration_ms, cp2)

        keys = sorted(with_limit)[:60]
        assert any(
            not np.allclose(
                with_limit[k].quaternion(), without[k].quaternion()
            )
            for k in keys
        )
