"""The rolling-shutter readout estimator (``for_rs``, B-02 remainder).

Upstream's ``estimate_rolling_shutter`` autosync mode drives the ``for_rs``
branch of ``visual_features::find_offsets``: instead of shifting the gyro
timeline, each candidate is a readout *time* — the per-point RS model in
the undistortion reads ``params.frame_readout_time``, so the candidate
swaps that field and measures the same point-pair distance at zero offset.

The scene reuses ``test_visual_features_offset``'s proven-aligned
conventions (the quaternion stream and the scene both use ``_omega``
directly as the pan angle; the offset landscape there hits exact zero),
with each point's exposure skewed by ``R·(row/H − ½)`` — the port's
readout window is centred on the frame timestamp, and the vertical readout
direction makes the row the pixel's y, which a Y-axis pan leaves fixed in
both frames.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pygyroflow.synchronization.find_offset.visual_features import (
    estimate_rolling_shutter,
)
from pygyroflow.stabilization.compute_params import ComputeParams
from pygyroflow.types.quaternion import Quat64

W = H = 1080
K = np.array([[540.0, 0.0, W / 2], [0.0, 540.0, H / 2], [0.0, 0.0, 1.0]])
FPS = 30.0
READOUT_MS = 15.0  # the truth to recover


# Pan about an axis tilted 30 degrees from Y toward X: features then move
# vertically between frames, so their exposure ROWS differ — without that
# (a pure Y pan leaves y fixed) the row skew applies identically to both
# frames of a pair and cancels in the pair distance to first order. The
# readout is only observable through cross-row motion.
_AXIS = np.array([math.sin(math.radians(30.0)), math.cos(math.radians(30.0)), 0.0])


def _omega(t: float) -> float:
    return 25.0 * math.sin(2.0 * math.pi * t / 1.5)


def _params() -> ComputeParams:
    params = ComputeParams(
        width=W, height=H, output_width=W, output_height=H,
        camera_matrix=K.copy(), scaled_fps=FPS,
        distortion_coeffs=[0.0] * 12,
        distortion_model_name="opencv_fisheye",
        frame_readout_time=0.0,  # the estimator searches this
    )
    params.quaternions = {
        int(k / FPS * 1e6): Quat64(Rotation.from_rotvec(
            _AXIS * math.radians(_omega(k / FPS))))
        for k in range(20)
    }
    params.smoothed_quaternions = dict(params.quaternions)
    return params


def _scene(readout_ms: float = READOUT_MS, n_pairs: int = 10,
           n_points: int = 16, seed: int = 1):
    """Frame pairs with a per-row exposure skew of *readout_ms*."""
    rng = np.random.default_rng(seed)
    world = np.column_stack([
        rng.uniform(-0.8, 0.8, n_points),
        rng.uniform(-0.3, 0.3, n_points),  # spread rows across the frame
        np.ones(n_points),
    ])
    world /= np.linalg.norm(world, axis=1, keepdims=True)

    def project(t, ray):
        rot = Rotation.from_rotvec(_AXIS * math.radians(_omega(t)))
        r = rot.inv().apply(ray)
        return (K @ (r / r[2]))[:2]

    pairs = []
    for k in range(1, n_pairs + 1):
        t0 = k / FPS
        pts1 = []
        pts2 = []
        for i in range(n_points):
            # Each frame's exposure skew uses that frame's OWN unskewed row
            # (the model reads each point's row from the point itself).
            base1 = project(t0, world[i])
            base2 = project(t0 + 1.0 / FPS, world[i])
            pts1.append(tuple(project(
                t0 + readout_ms / 1000.0 * (base1[1] / H - 0.5), world[i])))
            pts2.append(tuple(project(
                t0 + 1.0 / FPS + readout_ms / 1000.0 * (base2[1] / H - 0.5),
                world[i])))
        pairs.append((
            (int(t0 * 1e6), pts1),
            (int((t0 + 1.0 / FPS) * 1e6), pts2),
        ))
    return pairs


class TestEstimateRollingShutter:
    def test_finds_a_known_minimum(self, monkeypatch):
        """Mechanics: with a distance landscape whose minimum sits at a
        known readout (here 12.34 ms), the coarse ±(1000/fps) sweep plus
        the 0.01 ms refinement lands on it."""
        import pygyroflow.synchronization.find_offset.visual_features as vf

        def fake_distance(pairs, params, offs, w, h):
            return (params.frame_readout_time - 12.34) ** 2

        monkeypatch.setattr(vf, "_total_distance", fake_distance)
        result = estimate_rolling_shutter(
            [((0, [(1.0, 1.0)]), (33_333, [(1.0, 1.0)]))], _params(), FPS,
        )
        assert result is not None
        assert result[0] == pytest.approx(12.34, abs=0.02)

    def test_the_sweep_bounds_match_upstream(self, monkeypatch):
        """``-steps..steps`` with ``steps = (1000/fps) as isize`` — the
        coarse grid spans the full frame interval, endpoints included."""
        import pygyroflow.synchronization.find_offset.visual_features as vf

        seen = []

        def fake_distance(pairs, params, offs, w, h):
            seen.append(params.frame_readout_time)
            return abs(params.frame_readout_time)

        monkeypatch.setattr(vf, "_total_distance", fake_distance)
        estimate_rolling_shutter(
            [((0, [(1.0, 1.0)]), (33_333, [(1.0, 1.0)]))], _params(), FPS,
        )
        coarse = [v for v in seen if float(v).is_integer()]
        # Rust's `-steps..steps` is half-open, like Python's range: the
        # coarse grid spans [-33, +32] for a 30 fps clip.
        assert min(coarse) == -33.0 and max(coarse) == 32.0
        assert 0.0 in coarse

    def test_no_pairs_is_none(self):
        assert estimate_rolling_shutter([], _params(), FPS) is None

    def test_the_search_swaps_frame_readout_time_only(self, monkeypatch):
        """The candidate reaches the undistortion as ``frame_readout_time``
        on a private copy — the caller's params are untouched, the offset
        is zero, and the copy keeps the lens and gyro streams."""
        import importlib

        # The stabilization package shadows its submodule attribute with
        # the *function* cpu_undistort; resolve the module explicitly.
        cu = importlib.import_module("pygyroflow.stabilization.cpu_undistort")

        seen = {}
        real = cu.undistort_points_with_rolling_shutter

        def spy(pts, ts_ms, frame, params, amount, use_fovs):
            seen.setdefault("params", params)
            return real(pts, ts_ms, frame, params, amount, use_fovs)

        monkeypatch.setattr(cu, "undistort_points_with_rolling_shutter", spy)
        pairs = [((0, [(100.0, 100.0), (500.0, 500.0)]),
                  (33_333, [(102.0, 100.0), (502.0, 500.0)]))]
        params = _params()
        params.frame_readout_time = 99.0  # would poison the search if used
        estimate_rolling_shutter(pairs, params, FPS)
        assert seen["params"].frame_readout_time != 99.0  # a candidate, not the input
        assert params.frame_readout_time == 99.0  # caller untouched
