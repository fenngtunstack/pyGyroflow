"""The pose-estimation dispatch (``estimate_rotation``) and its estimators.

The dispatch used to be wrong in three ways, all fixed here and pinned:

* method 0 — the *default* — ran the eight-point RANSAC helper instead of
  what upstream calls ``PoseFindEssentialMat``; ``estimate_essential_matrix``
  was unreachable;
* an unknown method index fell back to the essential path, where upstream
  falls back to **Almeida**;
* none of the OpenCV estimators undistorted their points, so every sync run
  on a real lens estimated rotation from distorted pixels. Upstream runs
  ``undistort_points_for_optical_flow`` on each point set at its own frame's
  timestamp before all three OpenCV estimators.

The rotation-recovery tests synthesize correspondences through
``cv2.fisheye`` (an independent implementation of the same distortion
equations, not the port's own model code), so agreement cannot come from a
shared bug.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

import pygyroflow.synchronization.estimate_pose as pkg
import pygyroflow.synchronization.estimate_pose.almeida as almeida_mod
import pygyroflow.synchronization.estimate_pose.homography as homography_mod
from pygyroflow.stabilization.compute_params import ComputeParams

W, H = 1920, 1080
K = np.array([[900.0, 0.0, 960.0], [0.0, 900.0, 540.0], [0.0, 0.0, 1.0]])
FISHEYE = [-0.05, 0.02, 0.0, 0.0]


def _params(coeffs=FISHEYE, **overrides) -> ComputeParams:
    values = dict(
        width=W,
        height=H,
        output_width=W,
        output_height=H,
        camera_matrix=K.copy(),
        distortion_coeffs=list(coeffs) + [0.0] * (12 - len(coeffs)),
        distortion_model_name="opencv_fisheye",
    )
    values.update(overrides)
    return ComputeParams(**values)


def _rot(*, rx=0.0, ry=0.0, rz=0.0) -> np.ndarray:
    """Rotation matrix from Euler angles (applied x, then y, then z)."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _angle_between(Ra, Rb) -> float:
    """Rotation angle between two matrices, in degrees."""
    R = Ra.T @ Rb
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))))


def _synth_fisheye_pair(R_true, coeffs=FISHEYE, n=80, seed=3,
                        translation=(0.05, -0.02, 0.05)):
    """Pixel correspondences related by ``R_true`` through the fisheye.

    Frame-1 pixels are drawn away from the edges, unprojected with cv2's own
    fisheye model, rotated (plus a small camera translation — a *pure*
    rotation makes the triangulated cheirality of upstream's method 0
    degenerate, on real footage just as here), and reprojected — so frame 2
    is exactly what the moved camera with intrinsics K and distortion
    ``coeffs`` sees.
    """
    rng = np.random.default_rng(seed)
    d = np.array(coeffs[:4], dtype=np.float64).reshape(1, 4)
    kf = K.astype(np.float64)
    pix1 = np.column_stack([
        rng.uniform(W * 0.08, W * 0.92, n),
        rng.uniform(H * 0.08, H * 0.92, n),
    ]).astype(np.float32)
    rays1 = cv2.fisheye.undistortPoints(
        pix1.reshape(-1, 1, 2), kf, d
    ).reshape(-1, 2)
    moved = R_true @ np.column_stack([rays1, np.ones(len(rays1))]).T
    moved = moved + np.asarray(translation, dtype=np.float64).reshape(3, 1)
    rays2 = (moved / moved[2:3, :]).T[:, :2]  # projective divide by z
    pix2 = cv2.fisheye.distortPoints(
        rays2.reshape(-1, 1, 2).astype(np.float32), kf, d
    ).reshape(-1, 2)
    return pix1.astype(np.float64), pix2.astype(np.float64)


# ---------------------------------------------------------------------------
# The dispatch table
# ---------------------------------------------------------------------------


class TestTheDispatch:
    def test_method_0_is_the_essential_mat_estimator(self, monkeypatch):
        seen = {}

        def spy(p1, p2, k):
            seen["k"] = np.asarray(k).copy()
            return np.full((3, 3), 7.0)

        monkeypatch.setattr(pkg, "estimate_pose_find_essential_mat", spy)
        out = pkg.estimate_rotation(
            np.zeros((10, 2)), np.zeros((10, 2)), K, method=0, params=_params()
        )
        assert out[0, 0] == 7.0  # the spy's matrix came back
        assert np.allclose(seen["k"], np.eye(3))  # normalized points, identity K

    def test_unknown_method_falls_back_to_almeida(self, monkeypatch, caplog):
        called = []
        real = almeida_mod.estimate_pose_almeida

        def spy(*args, **kwargs):
            called.append(1)
            return real(*args, **kwargs)

        monkeypatch.setattr(almeida_mod, "estimate_pose_almeida", spy)
        with caplog.at_level("ERROR", logger=pkg.__name__):
            pts = np.array([[100.0, 100.0], [400.0, 150.0], [300.0, 500.0],
                            [800.0, 700.0]], dtype=np.float32)
            pkg.estimate_rotation(pts, pts + 2.0, K, method=7)
        assert called == [1]
        assert any("Unknown pose method 7" in r.message for r in caplog.records)

    def test_method_2_is_the_eight_point_stand_in(self, monkeypatch):
        seen = {}

        def spy(p1, p2, k):
            seen["k"] = np.asarray(k).copy()
            return np.eye(3), np.zeros(3)

        monkeypatch.setattr(pkg, "estimate_pose_eight_point", spy)
        out = pkg.estimate_rotation(
            np.zeros((10, 2)), np.zeros((10, 2)), K, method=2
        )
        assert np.allclose(out, np.eye(3))
        assert np.allclose(seen["k"], K)  # pinhole fallback passes the real K

    def test_method_3_is_homography(self, monkeypatch):
        marker = np.full((3, 3), 5.0)
        monkeypatch.setattr(pkg, "estimate_pose_find_homography", lambda p1, p2: marker)
        out = pkg.estimate_rotation(
            np.zeros((10, 2)), np.zeros((10, 2)), K, method=3
        )
        assert np.allclose(out, marker)


class TestEachSetIsUndistortedAtItsOwnTimestamp:
    """Upstream undistorts pts1 and pts2 at ``timestamp_us`` and
    ``next_timestamp_us`` — on a zoom lens those are different lenses."""

    def test_the_two_timestamps_reach_the_undistortion(self, monkeypatch):
        seen = []
        real = pkg.undistort_points_for_optical_flow

        def spy(pts, ts_us, params, dims):
            seen.append((ts_us, tuple(dims)))
            return real(pts, ts_us, params, dims)

        monkeypatch.setattr(pkg, "undistort_points_for_optical_flow", spy)
        pts = np.array([[100.0, 100.0], [900.0, 500.0], [1800.0, 900.0]] * 4,
                       dtype=np.float64)
        pkg.estimate_rotation(
            pts, pts, K, method=0, size_wh=(960.0, 540.0),
            params=_params(), timestamp_ms=1000.0, next_timestamp_ms=1033.0,
        )
        assert [ts for ts, _ in seen] == [1_000_000, 1_033_000]
        assert all(dims == (960.0, 540.0) for _, dims in seen)

    def test_size_wh_defaults_to_the_params_dimensions(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            pkg, "undistort_points_for_optical_flow",
            lambda pts, ts_us, params, dims: seen.append(tuple(dims)) or [(0.0, 0.0)],
        )
        pkg.estimate_rotation(
            np.zeros((12, 2)), np.zeros((12, 2)), K, method=0, params=_params()
        )
        assert seen == [(W, H), (W, H)]


# ---------------------------------------------------------------------------
# find_essential_mat (method 0)
# ---------------------------------------------------------------------------


class TestFindEssentialMat:
    def test_it_pins_upstreams_opencv_arguments(self, monkeypatch):
        """``find_essential_mat.rs:37,42``: LMEDS on an identity K, threshold
        1e-5, 4000 iterations, triangulated recoverPose at distance 1e5."""
        import pygyroflow.synchronization.estimate_pose.find_essential_mat as mod

        calls = []
        eye = np.eye(3)
        pts = np.random.default_rng(0).uniform(-0.5, 0.5, (40, 2))

        def fake_find(p1, p2, k, method, prob, thr, iters):
            calls.append(("find", k.copy(), method, prob, thr, iters))
            return np.eye(3), np.ones((40, 1), np.uint8)

        def fake_recover(e, p1, p2, k, distance):
            calls.append(("recover", k.copy(), distance))
            return 12, eye.copy(), np.zeros(3), np.ones((40, 1), np.uint8)

        monkeypatch.setattr(cv2, "findEssentialMat", fake_find)
        monkeypatch.setattr(cv2, "recoverPose", fake_recover)
        out = mod.estimate_pose_find_essential_mat(pts, pts, eye)
        assert out is not None
        assert calls[0][0] == "find"
        assert np.allclose(calls[0][1], eye)
        assert calls[0][2] == cv2.LMEDS
        assert calls[0][3:] == (0.999, 1e-5, 4000)
        assert calls[1][0] == "recover"
        assert np.allclose(calls[1][1], eye)
        assert calls[1][2] == 100000.0

    def test_below_ten_inliers_is_no_model(self, monkeypatch):
        import pygyroflow.synchronization.estimate_pose.find_essential_mat as mod

        monkeypatch.setattr(
            cv2, "findEssentialMat",
            lambda *a: (np.eye(3), np.ones((40, 1), np.uint8)),
        )
        monkeypatch.setattr(
            cv2, "recoverPose", lambda *a: (9, np.eye(3), np.zeros(3), None)
        )
        pts = np.random.default_rng(1).uniform(-0.5, 0.5, (40, 2))
        assert mod.estimate_pose_find_essential_mat(pts, pts, np.eye(3)) is None

    def test_it_recovers_a_real_rotation_through_a_real_lens(self):
        """End to end: distorted pixels in, rotation out, within half a
        degree. This is the path the default pose method never had."""
        R_true = _rot(rx=math.radians(0.8), ry=math.radians(-0.5),
                      rz=math.radians(0.3))
        pix1, pix2 = _synth_fisheye_pair(R_true)
        R = pkg.estimate_rotation(
            pix1, pix2, K, method=0, size_wh=(W, H), params=_params(),
        )
        assert R is not None
        assert _angle_between(R, R_true) < 0.5

    def test_the_lens_actually_matters(self):
        """The regression: distorted pixels through the pinhole path land far
        off the true rotation; the undistorted path does not."""
        R_true = _rot(rx=math.radians(0.8), ry=math.radians(-0.5))
        pix1, pix2 = _synth_fisheye_pair(R_true)
        got_lens = pkg.estimate_rotation(
            pix1, pix2, K, method=0, size_wh=(W, H), params=_params()
        )
        # Old behaviour: straight onto the distorted pixels with K.
        E, mask = cv2.findEssentialMat(
            pix1.astype(np.float32), pix2.astype(np.float32), K, cv2.LMEDS
        )
        _, R_pinhole, _, _ = cv2.recoverPose(
            E, pix1.astype(np.float32), pix2.astype(np.float32), K, mask=mask
        )
        assert _angle_between(got_lens, R_true) < 0.5
        assert _angle_between(R_pinhole, R_true) > 5.0 * _angle_between(
            got_lens, R_true
        )

    def test_nan_pairs_are_dropped_not_fatal(self):
        """A failed undistortion comes back NaN; the C++ call shrugs it off,
        the Python binding would raise — so the pair is dropped instead."""
        R_true = _rot(ry=math.radians(0.5))
        pix1, pix2 = _synth_fisheye_pair(R_true)
        pix1[:3] = np.nan
        pix2[:3] = np.nan
        R = pkg.estimate_rotation(
            pix1, pix2, K, method=0, size_wh=(W, H), params=_params()
        )
        assert R is not None
        assert _angle_between(R, R_true) < 0.5

    def test_all_nan_is_no_pose(self):
        nan = np.full((20, 2), np.nan)
        assert pkg.estimate_pose_find_essential_mat(nan, nan, np.eye(3)) is None

    def test_without_params_the_real_k_is_passed(self, monkeypatch):
        seen = {}

        def spy(p1, p2, k):
            seen["k"] = np.asarray(k).copy()
            return np.eye(3)

        monkeypatch.setattr(pkg, "estimate_pose_find_essential_mat", spy)
        pkg.estimate_rotation(np.zeros((10, 2)), np.zeros((10, 2)), K, method=0)
        assert np.allclose(seen["k"], K)


# ---------------------------------------------------------------------------
# eight_point (method 2) — the documented ARRSAC stand-in
# ---------------------------------------------------------------------------


class TestMethodTwoStandIn:
    def test_points_go_back_through_pixels(self, monkeypatch):
        """The stand-in needs a K, so the undistorted points are put back into
        pixel units of the flow size and the K at pts1's timestamp is used."""
        seen = {}

        def spy(p1, p2, k):
            seen["p1_scale"] = float(np.abs(p1).max())
            seen["k"] = np.asarray(k).copy()
            return np.eye(3), np.zeros(3)

        monkeypatch.setattr(pkg, "estimate_pose_eight_point", spy)
        pix = np.array([[100.0, 100.0], [960.0, 540.0], [1800.0, 900.0]] * 5,
                       dtype=np.float64)
        pkg.estimate_rotation(
            pix, pix, K, method=2, size_wh=(960.0, 540.0), params=_params()
        )
        # Half-size flow space: K scaled by 0.5, points back in pixel scale.
        assert np.allclose(seen["k"], K * 0.5)
        assert seen["p1_scale"] > 100.0

    def test_it_still_recovers_a_rotation_with_a_lens(self):
        """With a strong lens the undistorted stand-in lands on the true
        rotation while distorted pixels through the old path visibly miss.

        The bound is loose relative to method 0's on purpose: the stand-in
        inherits classic ``recoverPose``'s sign-test, which occasionally takes
        the wrong fold of the two-fold solution — the exact weakness ARRSAC
        does not have. Seed and coefficients are pinned so the fold is
        resolved and the assertion measures the wiring, not the coin flip.
        """
        coeffs = [-0.15, 0.05, 0.0, 0.0]
        R_true = _rot(rx=math.radians(3.0), ry=math.radians(-2.0))
        pix1, pix2 = _synth_fisheye_pair(R_true, coeffs=coeffs)
        R = pkg.estimate_rotation(
            pix1, pix2, K, method=2, size_wh=(W, H), params=_params(coeffs=coeffs),
        )
        assert R is not None
        assert _angle_between(R, R_true) < 0.5

        E, mask = cv2.findEssentialMat(
            pix1.astype(np.float32), pix2.astype(np.float32), K,
            cv2.RANSAC, 0.999, 1.0,
        )
        _, R_pinhole, _, _ = cv2.recoverPose(
            E, pix1.astype(np.float32), pix2.astype(np.float32), K, mask=mask
        )
        assert _angle_between(R_pinhole, R_true) > 2.0


# ---------------------------------------------------------------------------
# find_homography (method 3)
# ---------------------------------------------------------------------------


class TestFindHomography:
    def test_it_pins_upstreams_find_homography_arguments(self, monkeypatch):
        """``find_homography.rs:38``: RANSAC, threshold 0.001 on normalized
        points, 2000 iterations, confidence 0.999."""
        seen = {}
        real = cv2.findHomography

        def spy(p1, p2, **kwargs):
            seen.update(kwargs)
            return real(p1, p2, **kwargs)

        monkeypatch.setattr(homography_mod.cv2, "findHomography", spy)
        pts = np.random.default_rng(2).uniform(-0.5, 0.5, (40, 2))
        homography_mod.estimate_pose_find_homography(pts, pts + 0.01)
        assert seen["method"] == cv2.RANSAC
        assert seen["ransacReprojThreshold"] == 0.001
        assert seen["maxIters"] == 2000
        assert seen["confidence"] == 0.999

    def test_decomposition_runs_against_an_identity_k(self, monkeypatch):
        """Normalized points: the homography lives in identity-K space."""
        seen = {}
        real = homography_mod.cv2.decomposeHomographyMat

        def spy(H, k):
            seen["k"] = np.asarray(k).copy()
            return real(H, k)

        monkeypatch.setattr(homography_mod.cv2, "decomposeHomographyMat", spy)
        pts = np.random.default_rng(2).uniform(-0.5, 0.5, (40, 2))
        homography_mod.estimate_pose_find_homography(pts, pts + 0.01)
        assert np.allclose(seen["k"], np.eye(3))

    @pytest.mark.parametrize("order", ["small-first", "small-last"])
    def test_the_pick_is_the_smallest_translation_squared(self, monkeypatch, order):
        """The upstream fold is easy to misread as "largest |t| wins"; its
        guard keeps the stored solution when the candidate's ``t·t`` is
        bigger, so the *smallest* translation norm wins. Pinned in both
        argument orders."""
        R_a = _rot(rz=0.1)
        R_b = _rot(rx=0.1)
        t_big = np.array([[2.0], [0.0], [0.0]])
        t_small = np.array([[0.3], [0.0], [0.0]])
        pairs = [(R_a, t_big), (R_b, t_small)]
        if order == "small-first":
            pairs = pairs[::-1]

        def fake_decompose(H, k):
            Rs = [np.ascontiguousarray(r, dtype=np.float64) for r, _ in pairs]
            Ts = [np.ascontiguousarray(t, dtype=np.float64) for _, t in pairs]
            return len(pairs), Rs, Ts, [np.array([0.0, 0.0, 1.0])] * len(pairs)

        monkeypatch.setattr(
            homography_mod.cv2, "decomposeHomographyMat", fake_decompose
        )
        pts = np.random.default_rng(2).uniform(-0.5, 0.5, (40, 2))
        got = homography_mod.estimate_pose_find_homography(pts, pts + 0.01)
        assert np.allclose(got, R_b)

    def test_few_points_and_nan_are_handled(self, monkeypatch):
        monkeypatch.setattr(
            homography_mod.cv2, "findHomography",
            lambda *a, **k: pytest.fail("should not run on insufficient input"),
        )
        assert homography_mod.estimate_pose_find_homography(
            np.zeros((3, 2)), np.zeros((3, 2))
        ) is None
        nan = np.full((20, 2), np.nan)
        assert homography_mod.estimate_pose_find_homography(nan, nan) is None


# ---------------------------------------------------------------------------
# The estimator plumbing (PoseEstimator -> estimate_rotation)
# ---------------------------------------------------------------------------


class _StubDetector:
    def __init__(self, pts):
        self._pts = pts

    def detect_and_track(self, _prev, _curr):
        return self._pts.copy(), self._pts.copy() + 0.5


class TestThePoseEstimatorPlumbing:
    def test_process_all_passes_frame_size_and_both_timestamps(self, monkeypatch):
        import pygyroflow.synchronization.pose_estimator as pe_mod

        h, w = 540, 960
        pts = np.random.default_rng(4).uniform(0, 1, (40, 2)) * [w, h]
        seen = {}
        real = pe_mod.estimate_rotation

        def spy(prev, curr, k, **kwargs):
            seen.update(kwargs)
            return real(prev, curr, k, **kwargs)

        monkeypatch.setattr(pe_mod, "estimate_rotation", spy)

        est = pe_mod.PoseEstimator()
        est._detector = _StubDetector(pts)
        monkeypatch.setattr(est, "_get_detector", lambda: est._detector)
        est.set_camera_matrix(K)
        params = _params()
        est.set_compute_params(params)
        est.feed_frame(0, 1_000_000, np.zeros((h, w), np.uint8))
        est.feed_frame(1, 1_033_333, np.zeros((h, w), np.uint8))
        est.process_all()

        assert seen["size_wh"] == (w, h)
        assert seen["params"] is params
        assert seen["timestamp_ms"] == pytest.approx(1000.0)
        assert seen["next_timestamp_ms"] == pytest.approx(1033.333)

    def test_process_frame_pair_takes_the_next_timestamp(self, monkeypatch):
        import pygyroflow.synchronization.pose_estimator as pe_mod

        pts = np.random.default_rng(5).uniform(0, 1, (40, 2)) * [192, 108]
        seen = {}
        real = pe_mod.estimate_rotation

        def spy(prev, curr, k, **kwargs):
            seen.update(kwargs)
            return real(prev, curr, k, **kwargs)

        monkeypatch.setattr(pe_mod, "estimate_rotation", spy)

        est = pe_mod.PoseEstimator()
        est._detector = _StubDetector(pts)
        monkeypatch.setattr(pe_mod, "create_detector", lambda *_: est._detector)
        est.set_camera_matrix(K)
        est.process_frame_pair(
            np.zeros((108, 192), np.uint8), np.zeros((108, 192), np.uint8),
            500_000, next_timestamp_us=533_333,
        )
        assert seen["size_wh"] == (192, 108)
        assert seen["next_timestamp_ms"] == pytest.approx(533.333)
