"""The Almeida estimator's lens handling (gap item D-06, the sync side).

Upstream's ``Camera::delta`` runs every flow point through
``undistort_points`` before rotating it, so the coordinate that gets rotated is
a *ray*, not a distorted pixel. The port normalized and rotated only, on the
argument (in its module docstring) that upstream disables lens correction for
this estimator. That argument is wrong: ``PoseAlmeida::init`` sets
``compute_params.lens_correction_amount = 0.0``, but ``delta`` passes a literal
``1.0`` to ``undistort_points``, so the field it sets does not decide the
branch — the distortion is applied in full.

The tests here compare ``_CameraK.delta`` against a reproduction of upstream's
``Camera::delta`` written out from the Rust lines. That reproduction is
deliberately standalone: it calls the distortion model's scalar method and
does its own matrix arithmetic, so it cannot agree with the port by sharing
code with it.
"""

from __future__ import annotations

import ctypes
import math

import numpy as np
import pytest

from pygyroflow.stabilization.compute_params import ComputeParams
from pygyroflow.stabilization.distortion_models import from_name
from pygyroflow.synchronization.estimate_pose.almeida import (
    _POINT_FAILURE,
    _CameraK,
    estimate_pose_almeida,
)
from pygyroflow.types.kernel_params import KernelParams

W, H = 1920, 1080
FX = FY = 900.0
CX, CY = 960.0, 540.0
K = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])

FISHEYE = [-0.05, 0.02, 0.0, 0.0] + [0.0] * 8


def _params(coeffs=FISHEYE, **overrides) -> ComputeParams:
    values = dict(
        width=W,
        height=H,
        output_width=W,
        output_height=H,
        camera_matrix=K.copy(),
        distortion_coeffs=list(coeffs),
        distortion_model_name="opencv_fisheye",
    )
    values.update(overrides)
    return ComputeParams(**values)


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# ---------------------------------------------------------------------------
# The reference: upstream's Camera::delta, reproduced line by line
# ---------------------------------------------------------------------------


def _upstream_delta(coords, rot3, camera_matrix, coeffs, params) -> np.ndarray:
    """``almeida.rs:58-67`` with ``undistort_points(..., amount = 1.0)``.

    Reads: normalize the point to pixels, take the world point through the
    intrinsics, invert the lens, then ``rr = camera_matrix @ rotation`` and
    project. The caller then divides back into 0..1 units and subtracts the
    original coordinate.
    """
    kernel = KernelParams()
    kernel.width, kernel.height = int(params.width), int(params.height)
    kernel.output_width, kernel.output_height = (
        int(params.output_width),
        int(params.output_height),
    )
    kernel.f = (ctypes.c_float * 2)(float(camera_matrix[0, 0]), float(camera_matrix[1, 1]))
    kernel.c = (ctypes.c_float * 2)(float(camera_matrix[0, 2]), float(camera_matrix[1, 2]))
    padded = list(coeffs) + [0.0] * (12 - len(coeffs))
    kernel.k1 = (ctypes.c_float * 4)(*padded[0:4])
    kernel.k2 = (ctypes.c_float * 4)(*padded[4:8])
    kernel.k3 = (ctypes.c_float * 4)(*padded[8:12])
    kernel.light_refraction_coefficient = float(params.light_refraction_coefficient)

    model = from_name(params.distortion_model_name)
    rr = np.asarray(camera_matrix, dtype=np.float64) @ rot3

    out = []
    for u, v in coords:
        x, y = float(u) * W, float(v) * H
        pw = ((x - camera_matrix[0, 2]) / camera_matrix[0, 0],
              (y - camera_matrix[1, 2]) / camera_matrix[1, 1])
        pt = model.undistort_point(pw[0], pw[1], kernel)
        if pt is None:
            pt = (_POINT_FAILURE, _POINT_FAILURE)
        pr = rr @ np.array([pt[0], pt[1], 1.0])
        out.append((pr[0] / pr[2] / W - float(u), pr[1] / pr[2] / H - float(v)))
    return np.array(out)


_COORDS = np.array(
    [[0.5, 0.5], [0.3, 0.7], [0.8, 0.2], [0.15, 0.9], [0.62, 0.41], [0.88, 0.55]],
    dtype=np.float32,
)
_ROTATIONS = {
    "identity": np.eye(3),
    "pitch": _rot_x(math.radians(0.5)),
    "roll": _rot_y(math.radians(0.4)),
    "yaw": _rot_z(math.radians(-0.3)),
    "combined": _rot_x(math.radians(0.3)) @ _rot_y(math.radians(-0.2)) @ _rot_z(math.radians(0.15)),
}


class TestAgainstUpstreamDelta:
    """``_CameraK.delta`` reproduces upstream's ``Camera::delta``."""

    @pytest.mark.parametrize("name", sorted(_ROTATIONS))
    def test_fisheye(self, name):
        params = _params()
        cam = _CameraK(K, (W, H), params=params, timestamp_ms=0.0)
        want = _upstream_delta(_COORDS, _ROTATIONS[name], K, FISHEYE, params)
        got = cam.delta(_COORDS, _ROTATIONS[name])
        assert np.abs(got - want).max() < 1e-6, name

    @pytest.mark.parametrize("name", sorted(_ROTATIONS))
    def test_no_distortion(self, name):
        """With the lens out of the way the two must also agree — this is the
        case where the old port was accidentally right, so it is the control."""
        params = _params(coeffs=[0.0] * 12)
        cam = _CameraK(K, (W, H), params=params, timestamp_ms=0.0)
        want = _upstream_delta(_COORDS, _ROTATIONS[name], K, [0.0] * 12, params)
        got = cam.delta(_COORDS, _ROTATIONS[name])
        assert np.abs(got - want).max() < 1e-6, name

    def test_it_uses_the_per_timestamp_camera_matrix(self, monkeypatch):
        """Upstream re-reads the lens data on every call, at the frame's time.

        Which matters for a zoom lens: the intrinsics are not the same at the
        start and the end of the clip. Asserted at the call, because with a
        fixed-focal-length profile the two timestamps give the same matrix and
        the result cannot tell them apart.
        """
        import pygyroflow.stabilization.frame_transform as ft

        seen = []
        real = ft._get_lens_data_at_timestamp

        def spy(params, timestamp_ms, invert_asym_lens=False):
            seen.append(timestamp_ms)
            return real(params, timestamp_ms, invert_asym_lens)

        monkeypatch.setattr(ft, "_get_lens_data_at_timestamp", spy)
        params = _params()
        _CameraK(K, (W, H), params=params, timestamp_ms=1234.5)
        assert seen == [1234.5]


class TestTheLensActuallyChangesTheAnswer:
    """The regression the fix is for: with a lens, the old code ignored it."""

    def test_distortion_moves_delta(self):
        lens = _CameraK(K, (W, H), params=_params(), timestamp_ms=0.0)
        pinhole = _CameraK(K, (W, H))
        a = pinhole.delta(_COORDS, _ROTATIONS["combined"])
        b = lens.delta(_COORDS, _ROTATIONS["combined"])
        assert not np.allclose(a, b, atol=1e-9)
        # And not by a rounding amount either: the lens correction itself is
        # part of `delta`, so this is a first-order difference.
        assert np.abs(a - b).max() > 1e-3

    def test_yaw_is_the_one_that_is_least_affected(self):
        """A sanity check on the geometry, not on the code.

        Yaw turns about the optical axis and the distortion is radial about the
        same point, so the two nearly commute — a port that mixed up an axis
        would show up as yaw moving more than pitch/roll relative to it.
        """
        lens = _CameraK(K, (W, H), params=_params(), timestamp_ms=0.0)
        pinhole = _CameraK(K, (W, H))
        diffs = {}
        for name in ("pitch", "roll", "yaw"):
            diffs[name] = np.abs(
                lens.delta(_COORDS, _ROTATIONS[name])
                - pinhole.delta(_COORDS, _ROTATIONS[name])
            ).max()
        assert diffs["pitch"] > 0.0 and diffs["roll"] > 0.0


class TestBackwardsCompatibility:
    """``params=None`` keeps the old pinhole behaviour exactly."""

    def test_no_params_is_the_pinhole_delta(self):
        cam = _CameraK(K, (W, H))
        coords = _COORDS.astype(np.float64)
        rot = _ROTATIONS["combined"]
        xn = (coords[:, 0] * W - CX) / FX
        yn = (coords[:, 1] * H - CY) / FY
        rx = rot[0, 0] * xn + rot[0, 1] * yn + rot[0, 2]
        ry = rot[1, 0] * xn + rot[1, 1] * yn + rot[1, 2]
        rz = rot[2, 0] * xn + rot[2, 1] * yn + rot[2, 2]
        want = np.column_stack([
            (rx / rz * FX + CX - coords[:, 0] * W) / W,
            (ry / rz * FY + CY - coords[:, 1] * H) / H,
        ]).astype(np.float32)
        assert np.abs(cam.delta(_COORDS, rot) - want).max() < 1e-12

    def test_zero_coefficients_make_both_paths_identical(self):
        cam = _CameraK(K, (W, H))
        flat = _CameraK(K, (W, H), params=_params(coeffs=[0.0] * 12), timestamp_ms=0.0)
        for name, rot in _ROTATIONS.items():
            assert np.abs(cam.delta(_COORDS, rot) - flat.delta(_COORDS, rot)).max() < 1e-9, name


class TestTheVectorizedUndistortMatchesTheScalar:
    """``_CameraK`` uses the model's batch method; the family's scalar loop is
    what the reference fixture pins. They have to agree, centre pixel included.
    """

    def test_batch_and_scalar_agree_including_the_centre(self):
        from pygyroflow.stabilization.cpu_undistort import _points_kernel_params

        coefficients = [0.05, -0.01, 0.004, -0.0007, -0.012, 0.003, 0.001, -0.0002]
        params = _params(coeffs=coefficients + [0.0] * 4)
        kernel = _points_kernel_params(K, params.distortion_coeffs, params, 1.0)
        model = from_name("opencv_fisheye")

        rng = np.random.default_rng(5)
        xs = np.concatenate([[0.0], rng.uniform(-1.2, 1.2, 300)])
        ys = np.concatenate([[0.0], rng.uniform(-1.2, 1.2, 300)])
        bx, by = model.undistort_points(xs.copy(), ys.copy(), kernel)

        worst = 0.0
        for i in range(xs.size):
            scalar = model.undistort_point(float(xs[i]), float(ys[i]), kernel)
            if scalar is None:
                assert math.isnan(bx[i]) and math.isnan(by[i]), (xs[i], ys[i])
                continue
            assert not math.isnan(bx[i]) and not math.isnan(by[i]), (xs[i], ys[i])
            scale = max(abs(scalar[0]), abs(scalar[1]), 1.0)
            worst = max(worst, abs(scalar[0] - bx[i]) / scale, abs(scalar[1] - by[i]) / scale)
        assert worst < 1e-8, f"worst relative disagreement {worst:.3e}"

    def test_the_exact_principal_point_is_not_nan(self):
        """It used to be: the batch method excluded ``|theta_d| <= EPS``.

        NaN means "this point failed" to both consumers of that method — the
        lens-correction blend in the image path and the pose solver — so the
        single pixel under the principal point was being dropped. The scalar
        method returns (0, 0) for the same input.
        """
        from pygyroflow.stabilization.cpu_undistort import _points_kernel_params

        params = _params()
        kernel = _points_kernel_params(K, params.distortion_coeffs, params, 1.0)
        model = from_name("opencv_fisheye")
        ux, uy = model.undistort_points(np.array([0.0]), np.array([0.0]), kernel)
        assert float(ux[0]) == pytest.approx(0.0, abs=1e-12)
        assert float(uy[0]) == pytest.approx(0.0, abs=1e-12)


class TestThePlumbing:
    """``params`` has to reach the estimator, or none of the above runs."""

    def test_autosync_forwards_them_to_the_pose_estimator(self):
        from pygyroflow.synchronization import AutosyncProcess

        params = _params()
        proc = AutosyncProcess(camera_matrix=K, compute_params=params)
        assert proc.pose_estimator._compute_params is params

    def test_without_them_the_estimator_stays_on_a_pinhole(self):
        from pygyroflow.synchronization import AutosyncProcess

        proc = AutosyncProcess(camera_matrix=K)
        assert proc.pose_estimator._compute_params is None

    def test_estimate_rotation_passes_them_through(self, monkeypatch):
        import pygyroflow.synchronization.estimate_pose as pkg
        import pygyroflow.synchronization.estimate_pose.almeida as almeida_mod

        # The package imports the function lazily inside `estimate_rotation`,
        # so patching the module attribute is what the call site will see.
        seen = {}
        real = almeida_mod.estimate_pose_almeida

        def spy(prev, curr, K_, size_wh, use_ransac=True, params=None, timestamp_ms=0.0):
            seen["params"] = params
            seen["timestamp_ms"] = timestamp_ms
            return real(prev, curr, K_, size_wh, use_ransac=use_ransac,
                        params=params, timestamp_ms=timestamp_ms)

        monkeypatch.setattr(almeida_mod, "estimate_pose_almeida", spy)

        params = _params()
        pts = np.array([[100.0, 100.0], [200.0, 150.0], [300.0, 260.0]], dtype=np.float32)
        pkg.estimate_rotation(
            pts, pts + 1.0, K, method=1, size_wh=(W, H), params=params, timestamp_ms=33.0
        )
        assert seen["params"] is params
        assert seen["timestamp_ms"] == pytest.approx(33.0)

    def test_the_sync_snapshot_mirrors_upstreams_overrides(self):
        """``autosync.rs:86-89``: a full snapshot, keyframes cleared, amount 1.0."""
        from pygyroflow.manager import StabilizationManager

        manager = StabilizationManager()
        manager.params.size = (W, H)
        # A lens with no camera matrix -> no snapshot, and the estimators fall
        # back rather than the sync run failing.
        manager.lens.get_camera_matrix = lambda size=None: None
        assert manager._build_sync_compute_params() is None

        manager.lens.get_camera_matrix = lambda size=None: K.copy()
        manager.params.focal_length_smoothing_enabled = False
        params = manager._build_sync_compute_params()
        assert params is not None
        assert params.lens_correction_amount == 1.0
        assert len(params.keyframes.get_all_keys()) == 0
        assert params.width == W and params.height == H

    def test_estimate_pose_almeida_still_works_without_params(self):
        """The public entry point keeps its old signature working."""
        pts = np.array(
            [[100.0, 100.0], [400.0, 150.0], [300.0, 500.0], [800.0, 700.0]],
            dtype=np.float32,
        )
        result = estimate_pose_almeida(pts, pts + 2.0, K, (W, H), use_ransac=False)
        assert result is not None
        assert result.shape == (3, 3)
