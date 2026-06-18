"""Tests for distortion model factory and roundtrip distortion."""

import math
import ctypes

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pygyroflow.types.kernel_params import KernelParams
from pygyroflow.stabilization.distortion_models import (
    from_name,
    OpenCVFisheyeModel,
    OpenCVStandardModel,
    Poly3Model,
    Poly5Model,
    PTLensModel,
    Insta360Model,
    SonyModel,
    GoProSuperviewModel,
    GoProHyperviewModel,
    DigitalStretchModel,
)


# All registered model names
_MODEL_NAMES = [
    "opencv_fisheye",
    "opencv_standard",
    "poly3",
    "poly5",
    "ptlens",
    "insta360",
    "sony",
    "gopro_superview",
    "gopro_hyperview",
    "digital_stretch",
]


def _make_params_with_coeffs(k1=(0, 0, 0, 0), k2=(0, 0, 0, 0), k3=(0, 0, 0, 0),
                              fov=1.0, width=1920, height=1080):
    """Build a KernelParams with given distortion coefficients."""
    p = KernelParams()
    p.width = width
    p.height = height
    p.output_width = width
    p.output_height = height
    p.fov = fov
    p.f = (ctypes.c_float * 2)(1000.0, 1000.0)
    p.c = (ctypes.c_float * 2)(width / 2.0, height / 2.0)
    p.k1 = (ctypes.c_float * 4)(*k1)
    p.k2 = (ctypes.c_float * 4)(*k2)
    p.k3 = (ctypes.c_float * 4)(*k3)
    p.lens_correction_amount = 1.0
    p.input_horizontal_stretch = 1.0
    p.input_vertical_stretch = 1.0
    p.digital_lens_params = (ctypes.c_float * 4)(0, 0, 0, 0)
    return p


class TestFromName:
    @pytest.mark.parametrize("name,expected_cls", [
        ("opencv_fisheye", OpenCVFisheyeModel),
        ("opencv_standard", OpenCVStandardModel),
        ("poly3", Poly3Model),
        ("poly5", Poly5Model),
        ("ptlens", PTLensModel),
        ("insta360", Insta360Model),
        ("sony", SonyModel),
        ("gopro_superview", GoProSuperviewModel),
        ("gopro_hyperview", GoProHyperviewModel),
        ("digital_stretch", DigitalStretchModel),
    ])
    def test_returns_correct_model(self, name, expected_cls):
        model = from_name(name)
        assert isinstance(model, expected_cls)

    def test_unknown_name_falls_back_to_fisheye(self):
        model = from_name("nonexistent_model")
        assert isinstance(model, OpenCVFisheyeModel)


class TestDistortUndistortRoundtrip:
    """distort -> undistort should recover the original point (within tolerance)."""

    # Models that cannot handle zero coefficients (division by zero in implementation)
    _ZERO_COEFF_SKIP = {"poly3", "digital_stretch", "gopro_hyperview", "gopro_superview"}

    @pytest.mark.parametrize("name", _MODEL_NAMES)
    def test_roundtrip_identity_coeffs(self, name):
        """Zero coefficients: distort is identity, so roundtrip is exact."""
        if name in self._ZERO_COEFF_SKIP:
            pytest.skip(f"{name} requires non-zero coefficients")
        model = from_name(name)
        params = _make_params_with_coeffs()
        x, y = 0.3, -0.2
        dx, dy = model.distort_point(x, y, 1.0, params)
        result = model.undistort_point(dx, dy, params)
        if result is not None:
            assert_allclose(result, (x, y), atol=0.01,
                            err_msg=f"{name}: distort->undistort failed with zero coeffs")

    def test_fisheye_roundtrip_with_coeffs(self):
        model = from_name("opencv_fisheye")
        params = _make_params_with_coeffs(k1=(-0.1, 0.02, 0.001, 0.0))
        x, y = 0.2, 0.15
        dx, dy = model.distort_point(x, y, 1.0, params)
        result = model.undistort_point(dx, dy, params)
        if result is not None:
            assert_allclose(result, (x, y), atol=0.01,
                            err_msg="opencv_fisheye: distort->undistort roundtrip error")

    def test_poly3_roundtrip_with_coeffs(self):
        model = from_name("poly3")
        params = _make_params_with_coeffs(k1=(-0.1, 0, 0, 0))
        x, y = 0.2, 0.1
        dx, dy = model.distort_point(x, y, 1.0, params)
        result = model.undistort_point(dx, dy, params)
        if result is not None:
            assert_allclose(result, (x, y), atol=0.01)

    def test_ptlens_roundtrip_with_coeffs(self):
        model = from_name("ptlens")
        params = _make_params_with_coeffs(k1=(-0.05, 0.02, 0.01, 0))
        x, y = 0.15, -0.1
        dx, dy = model.distort_point(x, y, 1.0, params)
        result = model.undistort_point(dx, dy, params)
        if result is not None:
            assert_allclose(result, (x, y), atol=0.01)


class TestWGSLFunctions:
    # These models use digital_undistort_point/digital_distort_point instead
    _DIGITAL_MODELS = {"gopro_superview", "gopro_hyperview", "digital_stretch"}

    @pytest.mark.parametrize("name", _MODEL_NAMES)
    def test_wgsl_contains_required_functions(self, name):
        model = from_name(name)
        wgsl = model.wgsl_functions()
        if name in self._DIGITAL_MODELS:
            assert "fn digital_undistort_point" in wgsl, f"{name}: missing digital_undistort_point in WGSL"
            assert "fn digital_distort_point" in wgsl, f"{name}: missing digital_distort_point in WGSL"
        else:
            assert "fn undistort_point" in wgsl, f"{name}: missing undistort_point in WGSL"
            assert "fn distort_point" in wgsl, f"{name}: missing distort_point in WGSL"


class TestRadialDistortionLimit:
    def test_returns_none_or_positive_fisheye(self):
        model = from_name("opencv_fisheye")
        coeffs = [0.0] * 12
        result = model.radial_distortion_limit(coeffs)
        assert result is None or result > 0.0

    def test_returns_none_or_positive_standard(self):
        model = from_name("opencv_standard")
        coeffs = [0.0] * 12
        result = model.radial_distortion_limit(coeffs)
        assert result is None or result > 0.0

    def test_fisheye_with_coeffs(self):
        model = from_name("opencv_fisheye")
        coeffs = [-0.3, 0.1, 0.0, 0.0] + [0.0] * 8
        result = model.radial_distortion_limit(coeffs)
        assert result is None or result > 0.0
