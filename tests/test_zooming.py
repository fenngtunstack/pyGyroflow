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
