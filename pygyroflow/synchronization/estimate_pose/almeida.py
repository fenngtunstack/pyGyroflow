# -*- coding: utf-8 -*-
"""Almeida optical-flow rotation estimator.

Port of Gyroflow's ``estimate_pose/almeida.rs``, itself from
https://github.com/h33p/ofps (Almeida et al., "Robust Estimation of
Camera Motion Using Optical Flow Models"). Iterative least-squares
solve for yaw/pitch/roll over the whole flow field, wrapped in RANSAC.

Divergence from upstream (documented): our ``estimate_rotation`` API
carries only the camera matrix, so the Camera helper evaluates the
delta terms without lens distortion. Upstream disables lens correction
for this estimator anyway (``init`` sets ``lens_correction_amount = 0``);
the remaining raw-distortion term is second-order for sync estimation.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

_EPS = 0.001 * math.pi / 180.0  # 0.001 deg, rad
_ALPHA = 0.5
_LIMIT = math.ceil(15.0 / _ALPHA)


class _CameraK:
    """K-only stand-in for upstream's Camera (point_angle / delta)."""

    def __init__(self, K: npt.NDArray[np.float64], size_wh: tuple[float, float]) -> None:
        self.K = np.asarray(K, dtype=np.float64)
        self.fx = float(K[0, 0])
        self.fy = float(K[1, 1])
        self.cx = float(K[0, 2])
        self.cy = float(K[1, 2])
        self.w, self.h = size_wh

    def delta(self, coords: npt.NDArray[np.float32], rot3: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
        """Displacement (in normalized 0..1 image units) of each point under a
        rotation, viewed through the same K.

        Upstream pipes points through undistort_points with the rotation and
        the same K back; without distortion that reduces to: normalize -> R ->
        project. coords are (N, 2) in 0..1 units.
        """
        coords64 = coords.astype(np.float64)
        # upstream multiplies 0..1 coords back to pixels for the lens
        # pipeline, then divides again on the way out
        px = coords64[:, 0] * self.w
        py = coords64[:, 1] * self.h
        xn = (px - self.cx) / self.fx
        yn = (py - self.cy) / self.fy
        # rotate (column vector convention)
        rx = rot3[0, 0] * xn + rot3[0, 1] * yn + rot3[0, 2]
        ry = rot3[1, 0] * xn + rot3[1, 1] * yn + rot3[1, 2]
        rz = rot3[2, 0] * xn + rot3[2, 1] * yn + rot3[2, 2]
        # project back through K with unit depth
        qx = rx / rz * self.fx + self.cx
        qy = ry / rz * self.fy + self.cy
        out = np.empty_like(coords64)
        out[:, 0] = (qx - px) / self.w
        out[:, 1] = (qy - py) / self.h
        return out.astype(np.float32)

    def roll(self, coords, eps):
        return self.delta(coords, _rot_y(eps))

    def pitch(self, coords, eps):
        return self.delta(coords, _rot_x(eps))

    def yaw(self, coords, eps):
        return self.delta(coords, _rot_z(-eps))


def _rot_x(a: float) -> npt.NDArray[np.float64]:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(a: float) -> npt.NDArray[np.float64]:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_z(a: float) -> npt.NDArray[np.float64]:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def solve_ypr_given(
    pts: npt.NDArray[np.float32],
    motion: npt.NDArray[np.float32],
    cam: _CameraK,
) -> npt.NDArray[np.float64]:
    """Iterative least-squares YPR solve. Returns the 3x3 point-rotation
    (caller applies the inverse for the camera rotation, as upstream)."""
    rotation = np.eye(3)
    n = len(pts)
    for i in range(_LIMIT):
        alpha = 1.0 if i == _LIMIT - 1 else _ALPHA

        d = cam.delta(pts, rotation)                      # (N,2)
        v0 = motion - d                                   # residual motion
        v1 = cam.roll(pts, _EPS)
        v2 = cam.pitch(pts, _EPS)
        v3 = cam.yaw(pts, _EPS)

        A = np.empty((3, 3), dtype=np.float64)
        b = np.empty(3, dtype=np.float64)
        rows = (v1, v2, v3)
        for r in range(3):
            for c in range(3):
                A[r, c] = np.sum(rows[r] * rows[c])
            b[r] = np.sum(rows[r] * v0)

        try:
            model = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            model = np.zeros(3)

        model = model * _EPS * alpha

        # upstream order: pitch * roll * yaw, yaw negated
        rot = _rot_x(model[1]) @ _rot_y(model[0]) @ _rot_z(-model[2])
        rotation = rotation @ rot

    return rotation


def solve_ypr_ransac(
    pts: npt.NDArray[np.float32],
    motion: npt.NDArray[np.float32],
    cam: _CameraK,
    num_iters: int = 200,
    inlier_angle_deg: float = 0.05,
    num_samples: int = 1000,
    seed: int = 12345,
) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    target = math.radians(inlier_angle_deg)
    n = len(pts)
    if n < 3:
        return solve_ypr_given(pts, motion, cam)

    m = min(num_samples, n)
    best_mask = np.zeros(n, dtype=bool)

    for _ in range(num_iters):
        idx3 = rng.choice(n, size=3, replace=False)
        # solve_ypr_given returns the POINT rotation; upstream's `fit` is the
        # camera rotation (their solver already inverted), so their
        # `fit.inverse()` corresponds to our `fit` used directly here
        fit = solve_ypr_given(pts[idx3], motion[idx3], cam)

        idx = rng.choice(n, size=m, replace=False)
        d = cam.delta(pts[idx], fit)
        resid = motion[idx] - d

        # angular residual: |vec * cos(angle)| <= target  (upstream check)
        moved = pts[idx] + d
        xn = (moved[:, 0] * cam.w - cam.cx) / cam.fx
        yn = (moved[:, 1] * cam.h - cam.cy) / cam.fy
        cosx = np.cos(np.arctan(xn))
        cosy = np.cos(np.arctan(yn))
        mag2 = np.sum((resid * np.stack([cosx, cosy], axis=-1)) ** 2, axis=-1)
        mask = mag2 <= target * target

        if mask.sum() > best_mask.sum():
            best_mask = np.zeros(n, dtype=bool)
            best_mask[idx] = mask

    if best_mask.sum() >= 3:
        return solve_ypr_given(pts[best_mask], motion[best_mask], cam)
    return np.eye(3)


def estimate_pose_almeida(
    prev_pts: npt.NDArray[np.float32],
    curr_pts: npt.NDArray[np.float32],
    camera_matrix: npt.NDArray[np.float64],
    size_wh: tuple[float, float],
    use_ransac: bool = True,
) -> npt.NDArray[np.float64] | None:
    """3x3 camera rotation between two frames from the flow field.

    Points arrive in pixel units; upstream normalizes by the frame size and
    the Camera works in those units, so K is rescaled accordingly.
    """
    if len(prev_pts) < 3 or prev_pts.shape != curr_pts.shape:
        return None
    w, h = size_wh
    cam = _CameraK(camera_matrix, size_wh)
    pts = prev_pts.astype(np.float32) / np.float32([w, h])
    motion = ((curr_pts - prev_pts) / np.float32([w, h])).astype(np.float32)

    if use_ransac:
        rot = solve_ypr_ransac(pts, motion, cam)
    else:
        rot = solve_ypr_given(pts, motion, cam)
    # we estimated how POINTS rotate; the camera rotation is the inverse
    return np.linalg.inv(rot)
