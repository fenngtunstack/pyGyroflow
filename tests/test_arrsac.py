"""ARRSAC + eight-point: the real pose method 2 (``eight_point.rs``).

The scene is built from first principles: 3-D world points, a known
camera pose (rotation + translation), unit rays as the camera sees them.
The estimator must recover the rotation exactly on clean data and survive
outliers — the whole point of a sample-consensus layer.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pygyroflow.synchronization.estimate_pose.arrsac import (
    Arrsac,
    EightPointEstimator,
    estimate_pose_arrsac,
)


def _rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _angle_between(ra: np.ndarray, rb: np.ndarray) -> float:
    r = ra.T @ rb
    return math.degrees(
        math.acos(max(-1.0, min(1.0, (np.trace(r) - 1.0) / 2.0)))
    )


def _scene(n: int = 120, rot: float = 0.02, seed: int = 7):
    rng = np.random.default_rng(seed)
    r_true = _rot_y(rot)
    t_true = np.array([0.1, -0.05, 0.02])
    world = np.column_stack([
        rng.uniform(-1, 1, n), rng.uniform(-1, 1, n), rng.uniform(2, 8, n),
    ])
    rays_a = world / np.linalg.norm(world, axis=1, keepdims=True)
    moved = (r_true @ world.T).T + t_true
    rays_b = moved / np.linalg.norm(moved, axis=1, keepdims=True)
    return rays_a, rays_b, r_true


class TestTheEightPointEstimator:
    def test_a_minimal_sample_yields_four_poses(self):
        rays_a, rays_b, _ = _scene()
        est = EightPointEstimator()
        models = est.estimate(list(range(8)), rays_a, rays_b)
        assert len(models) == 4

    def test_one_pose_matches_the_true_rotation(self):
        rays_a, rays_b, r_true = _scene()
        est = EightPointEstimator()
        models = est.estimate(list(range(8)), rays_a, rays_b)
        assert any(_angle_between(m.rotation, r_true) < 1e-3 for m in models)

    def test_the_correct_pose_has_tiny_residuals(self):
        """The residual discriminates: the correct pose sits ~1e-12, the
        opposite-translation one ~2.0 — the SPRT thresholds (1e-10) are
        calibrated for exactly this scale."""
        rays_a, rays_b, r_true = _scene()
        est = EightPointEstimator()
        models = est.estimate(list(range(8)), rays_a, rays_b)
        best = min(models, key=lambda m: _angle_between(m.rotation, r_true))
        worst = max(models, key=lambda m: _angle_between(m.rotation, r_true))
        resids = est.residuals_batch(best, rays_a[:30], rays_b[:30])
        bad = est.residuals_batch(worst, rays_a[:30], rays_b[:30])
        assert resids.max() < 1e-9
        assert bad.min() > 0.5

    def test_batch_and_scalar_residuals_agree(self):
        rays_a, rays_b, _ = _scene()
        est = EightPointEstimator()
        models = est.estimate(list(range(8)), rays_a, rays_b)
        for m in models:
            for i in (0, 17, 42):
                single = est.residual(m, rays_a[i], rays_b[i])
                batch = float(est.residuals_batch(m, rays_a[i:i+1], rays_b[i:i+1])[0])
                assert single == pytest.approx(batch, abs=1e-12)


class TestArrsac:
    def test_recovers_the_true_rotation(self):
        rays_a, rays_b, r_true = _scene()
        arrsac = Arrsac(1e-10, np.random.default_rng(0))
        result = arrsac.model(EightPointEstimator(), rays_a, rays_b)
        assert result is not None
        pose, inliers = result
        assert _angle_between(pose.rotation, r_true) < 1e-4
        assert len(inliers) == len(rays_a)  # clean data: everything inliers

    def test_survives_thirty_percent_outliers(self):
        rays_a, rays_b, r_true = _scene(seed=3)
        rng = np.random.default_rng(11)
        bad = rng.choice(len(rays_b), 36, replace=False)
        rays_b[bad] = rng.normal(size=(36, 3))
        rays_b[bad] /= np.linalg.norm(rays_b[bad], axis=1, keepdims=True)
        arrsac = Arrsac(1e-10, np.random.default_rng(0))
        result = arrsac.model(EightPointEstimator(), rays_a, rays_b)
        assert result is not None
        pose, inliers = result
        assert _angle_between(pose.rotation, r_true) < 1e-4
        assert len(inliers) >= len(rays_a) - 40

    def test_too_few_points_is_none(self):
        rays_a = np.zeros((7, 3))
        arrsac = Arrsac(1e-10, np.random.default_rng(0))
        assert arrsac.model(EightPointEstimator(), rays_a, rays_a) is None

    def test_identical_rays_do_not_crash(self):
        """Identical a/b rays: the two triangulation constraints coincide,
        every pose fits exactly (residual ~0 along the shared ray), so
        ARRSAC returns some model with all inliers — degenerate but
        well-defined behaviour, not a failure."""
        rays = np.tile([0.1, 0.2, 1.0], (60, 1))
        arrsac = Arrsac(1e-10, np.random.default_rng(0))
        result = arrsac.model(EightPointEstimator(), rays, rays)
        assert result is not None
        pose, inliers = result
        assert len(inliers) == 60


class TestEstimatePoseArrsac:
    def test_clean_scene_exact(self):
        rays_a, rays_b, r_true = _scene()
        r = estimate_pose_arrsac(rays_a, rays_b)
        assert r is not None
        assert _angle_between(r, r_true) < 1e-3

    def test_thresholds_are_tried_in_order(self):
        """Upstream tries 1e-10, then 1e-8, then 1e-6 — noisy data that
        fails the first must still be found on a later threshold."""
        rays_a, rays_b, r_true = _scene(seed=5, n=100)
        rng = np.random.default_rng(2)
        noise = rng.normal(scale=2e-6, size=rays_b.shape)
        rays_b = rays_b + noise
        rays_b /= np.linalg.norm(rays_b, axis=1, keepdims=True)
        r = estimate_pose_arrsac(rays_a, rays_b)
        assert r is not None
        assert _angle_between(r, r_true) < 0.05

    def test_under_eight_points_is_none(self):
        assert estimate_pose_arrsac(np.zeros((5, 3)), np.zeros((5, 3))) is None
