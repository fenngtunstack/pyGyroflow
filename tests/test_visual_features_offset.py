"""The point-pair offset search (upstream offset method 1, ``visual_features``).

The port used to run a 1-D angular-velocity cross-correlation under this
name; upstream's method 1 maps every point pair through the lens *and the
gyro orientation at (frame_time − offs)* and minimizes the squared distances
over candidate offsets. The faithful port is tested here against an
independently constructed synthetic scene:

* the gyro stream is built straight from scipy ``Rotation`` (not the port's
  integrators),
* the footage is rendered with ``cv2.fisheye.distortPoints`` (not the port's
  distortion model),
* the footage's motion is shifted by a known ``delta`` against the gyro.

The metric degenerates on constant angular velocity — every offset then
makes the mapped pairs coincide, because the *inter-frame* rotation is
uniform — so the synthetic motion is sinusoidal, like real footage. The
measured convention: footage whose motion lags the gyro by ``delta`` yields
``offset = −delta``, which is exactly ``visual = gyro + offset`` (the
file-time of the motion is ``gyro_time − offset``).
"""

from __future__ import annotations

import dataclasses
import math

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pygyroflow.synchronization.autosync import AutosyncProcess
from pygyroflow.synchronization.find_offset import visual_features as vf_mod
from pygyroflow.synchronization.find_offset.visual_features import (
    find_offset_visual_features,
    find_offset_visual_features_correlation_fallback,
)
from pygyroflow.stabilization.compute_params import ComputeParams
from pygyroflow.types.quaternion import Quat64

W = H = 1080
K = np.array([[540.0, 0.0, W / 2], [0.0, 540.0, H / 2], [0.0, 0.0, 1.0]])
D = [-0.05, 0.02, 0.0, 0.0]
FPS = 30.0


def _omega(t: float) -> float:
    return 25.0 * math.sin(2.0 * math.pi * t / 1.5)


def _quat(t: float) -> Quat64:
    return Quat64(Rotation.from_euler("y", np.radians(_omega(t))))


def _scene(n_points=16, n_pairs=16, delta_s=0.060, seed=1):
    """Frame pairs of a camera panning with ``_omega``, delayed by ``delta_s``."""
    rng = np.random.default_rng(seed)
    rays = np.column_stack(
        [rng.uniform(-0.8, 0.8, n_points), rng.uniform(-0.8, 0.8, n_points),
         np.ones(n_points)]
    )
    d = np.array(D, dtype=np.float64).reshape(1, 4)

    def project(rot):
        rm = rays @ rot.T
        rn = rm[:, :2] / rm[:, 2:3]
        pts = cv2.fisheye.distortPoints(
            rn.reshape(-1, 1, 2).astype(np.float32), K, d
        ).reshape(-1, 2)
        return [tuple(map(float, p)) for p in pts]

    pairs = []
    for k in range(1, n_pairs + 1):
        t1 = k / FPS + delta_s
        t2 = (k + 1) / FPS + delta_s
        p1 = project(Rotation.from_euler("y", np.radians(_omega(t1))).as_matrix())
        p2 = project(Rotation.from_euler("y", np.radians(_omega(t2))).as_matrix())
        pairs.append((
            (int(k / FPS * 1e6), p1),
            (int((k + 1) / FPS * 1e6), p2),
        ))
    return pairs


def _params(n=32, **overrides) -> ComputeParams:
    params = ComputeParams(
        width=W, height=H, output_width=W, output_height=H,
        camera_matrix=K.copy(), scaled_fps=FPS,
        distortion_coeffs=list(D) + [0.0] * 8,
        distortion_model_name="opencv_fisheye",
    )
    # The gyro stream lives on the *unshifted* frame timeline; the footage
    # is the one that moved by delta_s.
    params.quaternions = {
        int(k / FPS * 1e6): _quat(k / FPS) for k in range(n + 2)
    }
    params.smoothed_quaternions = dict(params.quaternions)
    return params


class TestTheSearchFindsTheOffset:
    def test_footage_late_by_60ms_yields_minus_60(self):
        """Sign and localization: footage whose motion lags the gyro stream
        by 60 ms gives offset = −60 ms (``visual = gyro + offset``).

        The window is seeded near the truth on purpose: on short synthetic
        footage the cost landscape has side lobes (the inter-frame rotation
        is only ~1°, so its curvature signal is weak), and an unseeded
        global sweep can land in one. The production caller passes the
        prior offset the same way.
        """
        pairs = _scene(n_points=25, n_pairs=29, delta_s=0.060)
        params = _params()
        result = find_offset_visual_features(
            pairs, params, search_size_ms=60.0, initial_offset_ms=-60.0,
        )
        assert result is not None
        offset_ms, cost = result
        # 4 ms of slack: float32 pixel quantization and the quaternion
        # keyframe grid discretize the landscape.
        assert offset_ms == pytest.approx(-60.0, abs=5.0)
        # And it is a real minimum: the window edge costs measurably more.
        assert cost < vf_mod._total_distance(
            pairs, params, offset_ms + 25.0, float(W), float(H)
        )

    def test_the_window_edge_guard_is_vacuous_and_stays_so(self):
        """Upstream rejects offsets within the outer 10 % of the window
        (``|lowest − initial| ≥ 0.9·search_size``) — but the coarse grid only
        ever reaches ±0.5·search_size, so the guard can never fire. It is
        dead code upstream, and the port keeps it faithfully: any minimum,
        however bad, is returned."""
        pairs = _scene(n_points=25, n_pairs=29, delta_s=0.150)
        params = _params()
        result = find_offset_visual_features(
            pairs, params, search_size_ms=60.0, initial_offset_ms=-120.0,
        )
        assert result is not None  # the vacuous guard does not reject

    def test_no_pairs_is_no_result(self):
        assert find_offset_visual_features([], _params()) is None

    def test_progress_is_reported_monotonically(self):
        pairs = _scene(n_points=8, n_pairs=6, delta_s=0.0)
        params = _params()
        seen = []
        find_offset_visual_features(
            pairs, params, search_size_ms=40.0,
            progress_callback=seen.append,
        )
        assert seen == sorted(seen)
        assert seen[-1] == pytest.approx(1.0)


class TestTheDistanceFunction:
    """``calculate_distance`` (``visual_features.rs:49-83``), quirks included.

    The gyro-rotating undistortion itself is stubbed: these tests pin the
    pure post-processing (bounds, truncation, the 10 % drop) on known mapped
    coordinates, which a live lens would smear.
    """

    @pytest.fixture(autouse=True)
    def _stub_undistortion(self, monkeypatch):
        """Identity map: the mapped points equal the stored ones."""
        import importlib

        cu = importlib.import_module("pygyroflow.stabilization.cpu_undistort")
        monkeypatch.setattr(
            cu, "undistort_points_with_rolling_shutter",
            lambda pts, ts_ms, frame, params, amount, use_fovs: [
                tuple(map(float, p)) for p in pts
            ],
        )

    def test_u64_truncation_of_the_squared_distance(self):
        """``dist as u64`` truncates: dx = 3.5 -> 12.25 counted as 12."""
        params = _params()
        # Two points (keep = int(2 * 0.9) = 1), each with dx = 3.5.
        pts1 = [(100.0, 100.0), (200.0, 100.0)]
        pts2 = [(103.5, 100.0), (203.5, 100.0)]
        pairs = [((0, pts1), (33_333, pts2))]
        assert vf_mod._total_distance(pairs, params, 0.0, W, H) == pytest.approx(12.0)

    def test_strict_bounds_drop_edge_points(self):
        params = _params()
        # A point exactly ON the x=0 boundary is not counted (strict >).
        # 3 points -> 2 distances -> keep = int(2 * 0.9) = 1.
        pairs = [((0, [(0.0, 100.0), (200.0, 100.0), (300.0, 100.0)]),
                  (33_333, [(0.0, 100.0), (210.0, 100.0), (310.0, 100.0)]))]
        assert vf_mod._total_distance(pairs, params, 0.0, W, H) == pytest.approx(100.0)

    def test_the_longest_tenth_is_dropped(self):
        params = _params()
        # 10 points: 9 with distance 100, 1 outlier with distance 1000^2.
        pts1 = [(100.0 + 40.0 * i, 100.0) for i in range(10)]
        pts2 = [(x[0] + 10.0, x[1]) for x in pts1]
        pts2[0] = (pts1[0][0] + 500.0, 100.0)  # outlier, still inside the frame
        pairs = [((0, pts1), (33_333, pts2))]
        # keep = int(10 * 0.9) = 9 -> after sorting, the outlier is dropped.
        assert vf_mod._total_distance(pairs, params, 0.0, W, H) == pytest.approx(
            9 * 100.0
        )

    def test_a_single_surviving_point_contributes_nothing(self):
        """``(len * 0.9) as usize`` truncates: one point -> keep = 0."""
        params = _params()
        pairs = [((0, [(500.0, 500.0)]), (33_333, [(510.0, 500.0)]))]
        assert vf_mod._total_distance(pairs, params, 0.0, W, H) == 0.0


class TestTheOffsetSnapshot:
    def test_existing_sync_offsets_do_not_bias_the_sweep(self):
        """Upstream clears the gyro offsets on its private clone; the port
        blanks ``sync_offsets_adjusted``. A caller that synced before must
        get the same answer as a fresh one."""
        pairs = _scene(delta_s=0.060)
        params = _params()
        dirty = dataclasses.replace(params, sync_offsets_adjusted={123: 45.0})
        clean = find_offset_visual_features(pairs, params, search_size_ms=200.0)
        dirty_result = find_offset_visual_features(pairs, dirty, search_size_ms=200.0)
        assert clean is not None and dirty_result is not None
        assert clean[0] == pytest.approx(dirty_result[0])

    def test_the_callers_params_are_not_mutated(self):
        pairs = _scene(delta_s=0.0)
        params = _params()
        params.sync_offsets_adjusted = {7: 3.0}
        find_offset_visual_features(pairs, params, search_size_ms=40.0)
        assert params.sync_offsets_adjusted == {7: 3.0}


class TestTheAutosyncIntegration:
    def test_without_compute_params_there_is_no_pairs_search(self):
        proc = AutosyncProcess(camera_matrix=K, fps=FPS)
        assert proc._visual_features_offset(
            search_range_ms=200.0, initial_offset_ms=0.0
        ) is None

    def test_the_pairs_search_is_reached_and_returns_the_offset(self, monkeypatch):
        proc = AutosyncProcess(camera_matrix=K, fps=FPS, compute_params=_params())
        seen = {}

        def fake(pairs, params, **kwargs):
            seen["n_pairs"] = len(pairs)
            seen["kwargs"] = kwargs
            return (-42.0, 7.0)

        monkeypatch.setattr(vf_mod, "find_offset_visual_features", fake)

        pts = np.array([[100.0, 100.0], [800.0, 500.0]])
        from pygyroflow.synchronization.pose_estimator import FrameResult

        est = proc.pose_estimator
        for k in range(4):
            est._frames[k * 33_333] = FrameResult(
                timestamp_us=k * 33_333, frame_no=k,
                prev_points=pts + k, curr_points=pts + k + 1.0,
            )
        est._frames[4 * 33_333] = FrameResult(timestamp_us=4 * 33_333, frame_no=4)
        offset = proc._visual_features_offset(
            search_range_ms=200.0, initial_offset_ms=0.0
        )
        assert offset == pytest.approx(-42.0)
        assert seen["n_pairs"] == 4
        assert seen["kwargs"]["search_size_ms"] == pytest.approx(200.0)


class TestTheCorrelationFallback:
    def test_it_is_still_exported_and_finds_a_shift(self):
        ts = [int(k / FPS * 1e6) for k in range(40)]
        omega = [_omega(k / FPS) for k in range(40)]
        visual = [(t, np.array([0.0, w, 0.0])) for t, w in zip(ts, omega)]
        # Gyro delayed by 50 ms (records the motion 50 ms late).
        gyro = [
            (int((k / FPS + 0.050) * 1e6), np.array([0.0, w, 0.0]))
            for k, w in enumerate(omega)
        ]
        offset = find_offset_visual_features_correlation_fallback(
            visual, gyro, search_range_ms=500.0
        )
        # visual = gyro + offset with the gyro 50 ms late -> offset = -50 ms.
        assert offset == pytest.approx(-50.0, abs=5.0)
