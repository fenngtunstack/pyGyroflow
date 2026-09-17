# -*- coding: utf-8 -*-
"""Almeida optical-flow rotation estimator.

Port of Gyroflow's ``estimate_pose/almeida.rs``, itself from
https://github.com/h33p/ofps (Almeida et al., "Robust Estimation of
Camera Motion Using Optical Flow Models"). Iterative least-squares
solve for yaw/pitch/roll over the whole flow field, wrapped in RANSAC.

The ``Camera`` helper in upstream holds a whole ``ComputeParams``, and its
``delta`` runs every point through ``undistort_points`` — normalization, then
the lens model's *inverse*, then the rotation. The port used to normalize and
rotate only, on the argument that upstream "disables lens correction for this
estimator anyway". That argument is wrong twice over:

* ``PoseAlmeida::init`` sets ``compute_params.lens_correction_amount = 0.0``,
  but ``delta`` calls ``undistort_points`` with a *literal* ``1.0`` for that
  argument. The field it sets is not the one that decides the branch, so the
  blend is skipped and the distortion is applied in full.
* Even if the blend did run, ``lens_correction_amount = 0`` means "keep the
  original look", i.e. re-apply the forward distortion afterwards — not "skip
  the lens".

The measured cost of the omission, at 900 px focal length with a fisheye
k1 = -0.15 and a 0.5 degree perturbation: 80% relative error in pitch, 130% in
roll. Yaw is unaffected because it is a rotation about the optical axis and the
radial distortion is symmetric about it. The YPR solve fits all three at once,
so the pitch/roll terms were being fitted against a model that did not contain
them.

``params`` is optional here so the direct-API callers keep working, but with
``params=None`` the estimator silently reverts to the old distortion-free
model. The manager path supplies it; see ``PoseEstimator.set_compute_params``.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

_EPS = 0.001 * math.pi / 180.0  # 0.001 deg, rad
_ALPHA = 0.5
_LIMIT = math.ceil(15.0 / _ALPHA)

# ``undistort_points``' return for a point the model cannot invert. Upstream's
# `delta` inherits it and feeds it into the solve; reproducing it keeps a
# non-converged point behaving the same on both sides.
_POINT_FAILURE = -1000000.0


class _CameraK:
    """Stand-in for upstream's ``Camera`` (point_angle / delta)."""

    def __init__(
        self,
        K: npt.NDArray[np.float64],
        size_wh: tuple[float, float],
        params=None,
        timestamp_ms: float = 0.0,
    ) -> None:
        self.K = np.asarray(K, dtype=np.float64)
        self.w, self.h = size_wh
        self._model = None
        self._kernel_params = None

        if params is None:
            matrix = self.K
        else:
            # The intrinsics and the coefficients are the per-timestamp ones,
            # exactly as upstream's `get_lens_data_at_timestamp` supplies them.
            from pygyroflow.stabilization.cpu_undistort import _points_kernel_params
            from pygyroflow.stabilization.distortion_models import from_name
            from pygyroflow.stabilization.frame_transform import (
                _get_lens_data_at_timestamp,
            )

            matrix, coeffs, *_ = _get_lens_data_at_timestamp(params, timestamp_ms, False)
            matrix = np.asarray(matrix, dtype=np.float64)
            self._model = from_name(params.distortion_model_name)
            self._kernel_params = _points_kernel_params(
                matrix, coeffs, params, float(params.light_refraction_coefficient)
            )

        self.fx = float(matrix[0, 0])
        self.fy = float(matrix[1, 1])
        self.cx = float(matrix[0, 2])
        self.cy = float(matrix[1, 2])

    # -- the lens, when one is known -------------------------------------

    def _undistort(self, coords: npt.NDArray[np.float64]):
        """Normalise, then invert the lens — the head of ``undistort_points``.

        Vectorized where the scalar family is not, because the YPR solver calls
        this on the whole field on every iteration (and RANSAC on top of that).
        The arithmetic is the scalar path's; `test_almeida_lens.py` checks the
        two against each other so they cannot drift.

        Non-converged points become upstream's sentinel rather than NaN, and
        they *skip* the refraction step — the scalar loop `continue`s past it.
        """
        px = coords[:, 0] * self.w
        py = coords[:, 1] * self.h
        pw_x = (px - self.cx) / self.fx
        pw_y = (py - self.cy) / self.fy

        ux, uy = self._model.undistort_points(pw_x, pw_y, self._kernel_params)
        bad = np.isnan(ux) | np.isnan(uy)

        refraction = self._kernel_params.light_refraction_coefficient
        if refraction != 1.0 and refraction > 0.0:
            safe_x = np.where(bad, 0.0, ux)
            safe_y = np.where(bad, 0.0, uy)
            r = np.sqrt(safe_x * safe_x + safe_y * safe_y)
            with np.errstate(divide="ignore", invalid="ignore"):
                sin_theta_d = (r / np.sqrt(1.0 + r * r)) / refraction
                r_d = sin_theta_d / np.sqrt(
                    np.maximum(1.0 - sin_theta_d * sin_theta_d, 1e-12)
                )
                factor = np.where(r > 0.0, r_d / r, 1.0)
            ux = np.where(bad, ux, ux * factor)
            uy = np.where(bad, uy, uy * factor)

        return (
            np.where(bad, _POINT_FAILURE, ux),
            np.where(bad, _POINT_FAILURE, uy),
        )

    def cos_angle(self, coords: npt.NDArray[np.float64]):
        """``cos`` of each point's incidence angle (upstream's ``point_angle``).

        Pixel -> centred -> normalised -> ``atan``, per axis. Used by the RANSAC
        residual to convert a displacement into an angle, and it reads the same
        (per-timestamp, when there is one) intrinsics that ``delta`` does.
        """
        px = coords[:, 0] * self.w
        py = coords[:, 1] * self.h
        xn = (px - self.cx) / self.fx
        yn = (py - self.cy) / self.fy
        return np.cos(np.arctan(xn)), np.cos(np.arctan(yn))

    def delta(self, coords: npt.NDArray[np.float32], rot3: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
        """Displacement (in normalized 0..1 image units) of each point under a
        rotation, viewed through the same K.

        coords are (N, 2) in 0..1 units. With a lens, the point is put through
        the model's inverse first — so the coordinate that gets rotated is a
        *ray*, not a distorted pixel.
        """
        coords64 = coords.astype(np.float64)
        if self._model is None:
            xn = (coords64[:, 0] * self.w - self.cx) / self.fx
            yn = (coords64[:, 1] * self.h - self.cy) / self.fy
        else:
            xn, yn = self._undistort(coords64)

        rx = rot3[0, 0] * xn + rot3[0, 1] * yn + rot3[0, 2]
        ry = rot3[1, 0] * xn + rot3[1, 1] * yn + rot3[1, 2]
        rz = rot3[2, 0] * xn + rot3[2, 1] * yn + rot3[2, 2]
        # project back through K with unit depth
        qx = rx / rz * self.fx + self.cx
        qy = ry / rz * self.fy + self.cy
        out = np.empty_like(coords64)
        out[:, 0] = (qx - coords64[:, 0] * self.w) / self.w
        out[:, 1] = (qy - coords64[:, 1] * self.h) / self.h
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
        cosx, cosy = cam.cos_angle(pts[idx] + d)
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
    params=None,
    timestamp_ms: float = 0.0,
) -> npt.NDArray[np.float64] | None:
    """3x3 camera rotation between two frames from the flow field.

    Points arrive in pixel units; upstream normalizes by the frame size and
    the Camera works in those units, so K is rescaled accordingly.

    ``params`` is the ``ComputeParams`` the lens comes from — upstream's
    ``Camera`` holds one. Without it the estimator runs on a pinhole camera:
    see the module docstring for what that costs.
    """
    if len(prev_pts) < 3 or prev_pts.shape != curr_pts.shape:
        return None
    w, h = size_wh
    cam = _CameraK(camera_matrix, size_wh, params=params, timestamp_ms=timestamp_ms)
    pts = prev_pts.astype(np.float32) / np.float32([w, h])
    motion = ((curr_pts - prev_pts) / np.float32([w, h])).astype(np.float32)

    if use_ransac:
        rot = solve_ypr_ransac(pts, motion, cam)
    else:
        rot = solve_ypr_given(pts, motion, cam)
    # we estimated how POINTS rotate; the camera rotation is the inverse
    return np.linalg.inv(rot)
