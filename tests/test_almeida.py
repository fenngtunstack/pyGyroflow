# -*- coding: utf-8 -*-
"""Tests for the Almeida rotation estimator (port parity + accuracy)."""

from __future__ import annotations

import numpy as np
import pytest

from pygyroflow.synchronization.estimate_pose import estimate_rotation
from pygyroflow.synchronization.estimate_pose.almeida import (
    estimate_pose_almeida,
    solve_ypr_given,
)


def _K(w=1920.0, h=1080.0):
    return np.array([[w * 1.1, 0, w / 2], [0, w * 1.1, h / 2], [0, 0, 1]], dtype=np.float64)


def _rot_xyz(rx, ry, rz):
    from pygyroflow.synchronization.estimate_pose.almeida import _rot_x, _rot_y, _rot_z
    return _rot_x(rx) @ _rot_y(ry) @ _rot_z(rz)


def _projected_pair(R, K, n=400, seed=3):
    """Synthetic prev/curr pixel pairs produced by rotating the camera by R."""
    rng = np.random.default_rng(seed)
    w, h = 2 * K[0, 2], 2 * K[1, 2]
    # spread over the full frame: a narrow central patch makes pitch flow
    # degenerate into pure translation (unidentifiable for a rotation-only model)
    pts = np.stack([rng.uniform(0.08 * w, 0.92 * w, n),
                    rng.uniform(0.08 * h, 0.92 * h, n)], axis=-1)
    xn = (pts[:, 0] - K[0, 2]) / K[0, 0]
    yn = (pts[:, 1] - K[1, 2]) / K[1, 1]
    # camera rotates by R -> points move by R^-1 on the unit sphere
    v = np.stack([xn, yn, np.ones(n)], axis=-1) @ np.linalg.inv(R).T
    px = v[:, 0] / v[:, 2] * K[0, 0] + K[0, 2]
    py = v[:, 1] / v[:, 2] * K[1, 1] + K[1, 2]
    return pts.astype(np.float32), np.stack([px, py], axis=-1).astype(np.float32)


def _rot_angle_deg(R) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


class TestAlmeida:
    def test_recovers_known_rotation(self):
        R_true = _rot_xyz(np.deg2rad(-0.3), np.deg2rad(0.5), np.deg2rad(0.2))
        K = _K()
        prev, curr = _projected_pair(R_true, K)
        R_est = estimate_pose_almeida(prev, curr, K, (1920.0, 1080.0), use_ransac=False)
        delta = R_est @ np.linalg.inv(R_true)
        assert _rot_angle_deg(delta) < 0.02  # sub-0.02 deg recovery

    def test_ransac_rejects_outliers(self):
        R_true = _rot_xyz(np.deg2rad(0.2), np.deg2rad(-0.4), np.deg2rad(0.1))
        K = _K()
        prev, curr = _projected_pair(R_true, K, n=300, seed=5)
        # corrupt 20% of the correspondences with garbage motion
        rng = np.random.default_rng(9)
        bad = rng.choice(len(prev), size=60, replace=False)
        curr = curr.copy()
        curr[bad] += rng.uniform(-40, 40, size=(60, 2)).astype(np.float32)
        R_est = estimate_pose_almeida(prev, curr, K, (1920.0, 1080.0), use_ransac=True)
        delta = R_est @ np.linalg.inv(R_true)
        assert _rot_angle_deg(delta) < 0.1  # robust despite 20% outliers

    def test_dispatch_method_1(self):
        R_true = _rot_xyz(0.0, np.deg2rad(0.6), 0.0)
        K = _K()
        prev, curr = _projected_pair(R_true, K, n=200, seed=7)
        R_est = estimate_rotation(prev, curr, K, method=1, size_wh=(1920.0, 1080.0))
        assert R_est is not None
        delta = R_est @ np.linalg.inv(R_true)
        assert _rot_angle_deg(delta) < 0.1

    def test_identity_flow_identity_rotation(self):
        K = _K()
        rng = np.random.default_rng(2)
        pts = np.stack([rng.uniform(400, 1500, 150), rng.uniform(300, 800, 150)], axis=-1).astype(np.float32)
        R = solve_ypr_given(
            pts / np.float32([1920.0, 1080.0]),
            np.zeros_like(pts, dtype=np.float32),
            __import__("pygyroflow.synchronization.estimate_pose.almeida", fromlist=["_CameraK"])._CameraK(K, (1920.0, 1080.0)),
        )
        assert _rot_angle_deg(R) < 0.01

    def test_too_few_points_returns_none(self):
        K = _K()
        pts = np.zeros((2, 2), dtype=np.float32)
        assert estimate_pose_almeida(pts, pts, K, (1920.0, 1080.0)) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
