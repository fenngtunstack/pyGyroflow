# -*- coding: utf-8 -*-
"""Parity tests: cv2 C-accelerated paths vs the reference numpy formulas.

The cv2 route is only allowed to replace the numpy evaluation when the two
agree to tight tolerances across the coefficient space real footage uses
(weak/typical/negative/strong fisheye), including edge behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from pygyroflow.stabilization.distortion_models.opencv_fisheye import OpenCVFisheyeModel
from pygyroflow.types.kernel_params import KernelParams

cv2 = pytest.importorskip("cv2")


def _kp(coeffs):
    kp = KernelParams()
    for i, c in enumerate(coeffs):
        kp.k1[i] = c
    return kp


COEFF_SETS = [
    (0.12, 0.03, -0.01, 0.002),
    (-0.3, 0.08, 0.0, 0.0),
    (0.5, 0.1, 0.0, 0.0),
    (0.05, 0.0, 0.0, 0.0),
]

RNG = np.random.default_rng(11)


class TestFisheyeDistortParity:
    @pytest.mark.parametrize("coeffs", COEFF_SETS)
    def test_matches_reference_numpy(self, coeffs, monkeypatch):
        """cv2 path vs the retained numpy formula (forced via _cv2=None)."""
        import pygyroflow.stabilization.distortion_models.opencv_fisheye as mod

        model = OpenCVFisheyeModel()
        kp = _kp(coeffs)
        n = 5000
        x = RNG.uniform(-1.3, 1.3, n)
        y = RNG.uniform(-1.3, 1.3, n)
        z = np.ones(n)

        cx, cy = model.distort_points(x, y, z, kp)

        monkeypatch.setattr(mod, "_cv2", None)  # force the numpy branch
        nx, ny = model.distort_points(x, y, z, kp)

        assert np.nanmax(np.hypot(cx - nx, cy - ny)) < 1e-7

    def test_image_center_exact(self):
        # r == 0 must not produce NaN (cv2 handles the 0/0 scale)
        model = OpenCVFisheyeModel()
        kp = _kp(COEFF_SETS[0])
        x = np.array([0.0, 0.0, 1e-12])
        y = np.array([0.0, 1e-12, 0.0])
        fx, fy = model.distort_points(x, y, np.ones(3), kp)
        assert np.all(np.isfinite(fx)) and np.all(np.isfinite(fy))
        assert abs(fx[0]) < 1e-9 and abs(fy[0]) < 1e-9

    def test_zero_coeffs_shortcut(self):
        model = OpenCVFisheyeModel()
        kp = _kp((0.0, 0.0, 0.0, 0.0))
        x = np.array([0.3, -0.7])
        y = np.array([0.5, 0.1])
        fx, fy = model.distort_points(x, y, np.array([2.0, 2.0]), kp)
        np.testing.assert_allclose(fx, [0.15, -0.35])
        np.testing.assert_allclose(fy, [0.25, 0.05])

    @pytest.mark.parametrize("coeffs", COEFF_SETS)
    def test_roundtrip_with_numpy_inverse(self, coeffs):
        """distort (cv2) then undistort (numpy Newton) returns the input."""
        model = OpenCVFisheyeModel()
        kp = _kp(coeffs)
        n = 3000
        x = RNG.uniform(-1.0, 1.0, n)
        y = RNG.uniform(-1.0, 1.0, n)
        fx, fy = model.distort_points(x, y, np.ones(n), kp)
        ux, uy = model.undistort_points(fx, fy, kp)
        ok = np.isfinite(ux) & np.isfinite(uy)
        assert ok.mean() > 0.95  # extreme corners may legitimately NaN out
        assert np.max(np.hypot(ux[ok] - x[ok], uy[ok] - y[ok])) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
