# -*- coding: utf-8 -*-
"""Tests for cpu_undistort interpolation support (upstream index semantics).

0 = Bilinear, 1 = Bicubic, 2 = Lanczos4 (upstream default), 3-6 = EWA
(fallback to Lanczos4). The bilinear path keeps BORDER_CONSTANT semantics;
wider kernels replicate the edge and paint invalid / out-of-frame pixels
with the background colour afterwards.
"""

from __future__ import annotations

import numpy as np
import pytest

from pygyroflow.stabilization import FrameTransform
from pygyroflow.stabilization.cpu_undistort import cpu_undistort
from pygyroflow.types.kernel_params import KernelParams


def _identity_transform(w: int = 64, h: int = 48, r_limit: float = 0.0) -> FrameTransform:
    """Global-shutter identity transform with zero distortion."""
    kp = KernelParams()
    kp.width = w
    kp.height = h
    kp.output_width = w
    kp.output_height = h
    kp.matrix_count = 1
    # u = xd*fx + cx with zero distortion => identity needs f=(1,1), c=(0,0)
    kp.f[0] = 1.0
    kp.f[1] = 1.0
    kp.c[0] = 0.0
    kp.c[1] = 0.0
    kp.fov = 1.0
    kp.r_limit = r_limit
    kp.background[0] = 0.0
    kp.background[1] = 255.0
    kp.background[2] = 0.0
    kp.background[3] = 1.0

    matrices = np.zeros((1, 14), dtype=np.float32)
    matrices[0, 0] = 1.0  # row-major identity 3x3
    matrices[0, 4] = 1.0
    matrices[0, 8] = 1.0
    return FrameTransform(matrices=matrices, kernel_params=kp, fov=1.0)


def _textured_frame(w: int = 64, h: int = 48) -> np.ndarray:
    """High-frequency pattern so interpolation kernels measurably differ."""
    ys, xs = np.mgrid[0:h, 0:w]
    img = (127 * (1 + np.sin(xs * 0.9) * np.cos(ys * 0.8))).astype(np.uint8)
    return np.stack([img] * 3, axis=-1)


class TestInterpolation:
    def test_identity_transform_preserves_frame_for_all_modes(self):
        frame = _textured_frame()
        ft = _identity_transform()
        for interp in (0, 1, 2):
            out = cpu_undistort(frame, ft, interpolation=interp)
            assert out.shape == frame.shape
            # Integer-centre coordinates reproduce the source pixels for
            # any symmetric kernel.
            np.testing.assert_allclose(out, frame, atol=1)

    def test_lanczos4_differs_from_bilinear_on_shifted_sampling(self):
        # A half-pixel translation forces true resampling; the kernels
        # must disagree somewhere.
        frame = _textured_frame()
        ft = _identity_transform()
        ft.kernel_params.translation2d[0] = 0.5
        ft.kernel_params.translation2d[1] = 0.5
        bilinear = cpu_undistort(frame, ft, interpolation=0)
        lanczos = cpu_undistort(frame, ft, interpolation=2)
        assert bilinear.shape == lanczos.shape == frame.shape
        assert np.mean(np.abs(bilinear.astype(int) - lanczos.astype(int)) > 0) > 0.0

    def test_default_is_lanczos4(self):
        frame = _textured_frame()
        ft = _identity_transform()
        default = cpu_undistort(frame, ft)
        lanczos = cpu_undistort(frame, ft, interpolation=2)
        np.testing.assert_array_equal(default, lanczos)

    def test_ewa_indices_fall_back_to_lanczos4(self):
        frame = _textured_frame()
        ft = _identity_transform()
        ref = cpu_undistort(frame, ft, interpolation=2)
        for idx in (3, 4, 5, 6):
            np.testing.assert_array_equal(cpu_undistort(frame, ft, interpolation=idx), ref)

    def test_invalid_pixels_get_background_in_wide_kernel_path(self):
        # r_limit so small everything is invalid -> both paths must paint
        # the background colour (validates the REPLICATE + post-paint path).
        frame = _textured_frame()
        ft = _identity_transform(r_limit=0.01)
        ft.kernel_params.translation2d[0] = 0.5  # even the origin goes invalid
        for interp in (0, 2):
            out = cpu_undistort(frame, ft, interpolation=interp)
            # background (0, 255, 0) for uint8
            assert out[:, :, 0].max() == 0
            assert out[:, :, 1].min() == 255
            assert out[:, :, 2].max() == 0

    def test_unknown_index_falls_back_to_lanczos4(self):
        frame = _textured_frame()
        ft = _identity_transform()
        ref = cpu_undistort(frame, ft, interpolation=2)
        np.testing.assert_array_equal(cpu_undistort(frame, ft, interpolation=99), ref)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
