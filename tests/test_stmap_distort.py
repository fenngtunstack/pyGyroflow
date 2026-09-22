"""The STMap distort map through the real model (C-13 / G-06).

The distort map used to be the inverse of ``matrices[0]``'s 3×3 — no lens,
no per-row rolling shutter, no FOV scaling. Upstream (``stmap.rs:104-119``)
runs every input pixel through ``at_timestamp_for_points(use_fovs = True)``
+ ``undistort_points`` at full correction; the port now feeds the whole
grid through the points family in one call. These tests pin the properties
that separate the real model from the old matrix inverse.
"""

from __future__ import annotations

import ctypes
import math

import numpy as np
import pytest

from pygyroflow.stabilization.compute_params import ComputeParams
from pygyroflow.stmap.exporter import STMapExporter
from pygyroflow.types.quaternion import Quat64
from scipy.spatial.transform import Rotation

W, H = 96, 64
K = np.array([[48.0, 0.0, W / 2], [0.0, 48.0, H / 2], [0.0, 0.0, 1.0]])
FISHEYE = [-0.2, 0.05, 0.0, 0.0]


def _params(**overrides) -> ComputeParams:
    values = dict(
        width=W, height=H, output_width=W, output_height=H,
        camera_matrix=K.copy(),
        distortion_coeffs=list(FISHEYE) + [0.0] * 8,
        distortion_model_name="opencv_fisheye",
        quaternions={0: Quat64(Rotation.identity())},
        smoothed_quaternions={0: Quat64(Rotation.identity())},
    )
    values.update(overrides)
    return ComputeParams(**values)


class TestTheDistortMapUsesTheModel:
    def test_nonzero_distortion_changes_the_map(self):
        """The old matrix inverse could not see the lens at all: with the
        identity rotation, matrices[0] is ~identity and the map was ~the
        identity grid. Through the real model a fisheye bends it."""
        exporter = STMapExporter(_params())
        distorted = exporter.compute_distort_map(0.0, 0)

        flat = _params(distortion_coeffs=[0.0] * 12)
        exporter_flat = STMapExporter(flat)
        undistorted = exporter_flat.compute_distort_map(0.0, 0)

        delta = np.abs(distorted - undistorted)
        # The centre is untouched by a radial model; the corners move by
        # pixels, not rounding.
        assert delta[2:-2, 2:-2].max() > 1.0

    def test_identity_scene_maps_pixels_near_themselves(self):
        """No rotation, no distortion: every pixel maps to (x, y) in the
        stabilized space (the stabilizing rotation is the identity)."""
        exporter = STMapExporter(_params(distortion_coeffs=[0.0] * 12))
        mapped = exporter.compute_distort_map(0.0, 0)
        ys, xs = np.mgrid[0:H, 0:W]
        assert np.abs(mapped[..., 0] - xs).max() < 1e-6
        assert np.abs(mapped[..., 1] - ys).max() < 1e-6

    def test_the_principal_point_survives_the_fisheye(self):
        """Radial distortion leaves the optical centre fixed: the map at
        (cx, cy) is (cx, cy) (any FOV scaling is centred there too)."""
        exporter = STMapExporter(_params())
        mapped = exporter.compute_distort_map(0.0, 0)
        cx, cy = W / 2, H / 2
        assert mapped[int(cy), int(cx), 0] == pytest.approx(cx, abs=1e-6)
        assert mapped[int(cy), int(cx), 1] == pytest.approx(cy, abs=1e-6)

    def test_rolling_shutter_varies_the_map_by_row(self):
        """With a readout time and a rotating camera, different rows get
        different stabilizing rotations — the map's row dependence is not
        the identity grid the matrix inverse produced."""
        quats = {}
        n = 60
        for k in range(n):
            t = k / 200.0
            ang = math.radians(10.0 * t)
            quats[int(round(t * 1e6))] = Quat64(
                Rotation.from_euler("y", ang)
            )
        params = _params(
            quaternions=quats,
            smoothed_quaternions=dict(quats),
            frame_readout_time=20.0,
        )
        exporter = STMapExporter(params)
        mapped = exporter.compute_distort_map(timestamp_ms=100.0, frame=3)

        # Column-centre pixels in two distant rows map to different y's
        # than a static model would give (which is exactly y).
        col = W // 2
        centre_dev = abs(mapped[:, col, 1] - np.arange(H)).max()
        assert centre_dev > 0.05

    def test_failure_points_are_zeroed_not_sentinel(self):
        """Points the model cannot invert come back as the (-1e6, -1e6)
        sentinel; the export path must not leak it into the map."""
        exporter = STMapExporter(_params())
        mapped = exporter.compute_distort_map(0.0, 0)
        from pygyroflow.stabilization.cpu_undistort import _POINT_FAILURE

        assert not np.isclose(mapped, _POINT_FAILURE).any()
        assert np.isfinite(mapped).all()


class TestShapeAndOverrides:
    def test_dimensions_override(self):
        exporter = STMapExporter(_params())
        mapped = exporter.compute_distort_map(0.0, 0, width=32, height=24)
        assert mapped.shape == (24, 32, 2)
        assert mapped.dtype == np.float32
