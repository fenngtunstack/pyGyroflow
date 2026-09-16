"""GoPro6 Superview digital lens.

``gopro6_superview`` was missing from the model registry entirely, while two
other places in this port already knew the name: ``_DISTORTION_MODEL_IDS`` in
``lens/profile.py`` maps it, and ``get_all_matching_profiles`` lists it
alongside ``gopro_superview`` in the calibration-dimension rescale. Upstream
has a whole file for it (``distortion_models/gopro6_superview.rs``) with its
own polynomial and its own compiled fragment shader.

The consequence of the gap was silent: ``from_name`` falls back to
``OpenCVFisheye`` for an unknown name — the same fallback upstream uses — so a
lens profile carrying ``digital_lens: "gopro6_superview"`` rendered through a
fisheye model instead, with no error and no warning. Two visibly different
digital lenses collapsing into a third thing.

The expected values in ``tests/golden/gopro6_superview.json`` are upstream Rust
output (see ``generate_gopro6_superview_reference.py``), not this port's. The
comparison is relative and loose for a documented reason: upstream computes
this model in ``f32`` and this port in ``f64``, and the inversion is a
fixed-point iteration whose error is amplified where it converges slowly.
"""

from __future__ import annotations

import ctypes
import json
import pathlib

import numpy as np
import pytest

from pygyroflow.stabilization.distortion_models import (
    GoPro6SuperviewModel,
    GoProSuperviewModel,
    from_name,
)
from pygyroflow.types.kernel_params import KernelParams

_FIXTURE = pathlib.Path(__file__).parent / "golden" / "gopro6_superview.json"

with open(_FIXTURE, encoding="utf-8") as _handle:
    _DOC = json.load(_handle)
_CASES = _DOC["cases"]

# Measured worst disagreement with the fixture: 1.05e-7 (undistort) and
# 1.14e-6 (distort). Almost all of the distort figure comes from the two
# probes in the region where the iteration is not a contraction — there the
# f32 iteration amplifies the f32/f64 input difference by an order of
# magnitude. The per-value bound is set an order above that; the aggregate
# bound below it is what actually catches drift.
_PER_VALUE_REL = 1e-5
_AGGREGATE_REL = 2e-6


def _params(case) -> KernelParams:
    params = KernelParams()
    params.width = case["width"]
    params.height = case["height"]
    params.output_width = case["width"]
    params.output_height = case["height"]
    return params


def _model():
    return GoPro6SuperviewModel()


class TestAgainstUpstreamRust:
    """The fixture is upstream's own output, run verbatim."""

    @pytest.mark.parametrize("case", _CASES, ids=lambda c: c["name"])
    def test_undistorted_matches_the_rust(self, case):
        model = _model()
        params = _params(case)
        for probe, want in zip(case["probes"], case["undistorted"]):
            got = model.undistort_point(probe[0], probe[1], params)
            assert got == pytest.approx(tuple(want), rel=_PER_VALUE_REL), (
                f"{case['name']} probe {probe}"
            )

    @pytest.mark.parametrize("case", _CASES, ids=lambda c: c["name"])
    def test_distorted_matches_the_rust(self, case):
        model = _model()
        params = _params(case)
        for probe, want in zip(case["probes"], case["distorted"]):
            got = model.distort_point(probe[0], probe[1], 1.0, params)
            assert got == pytest.approx(tuple(want), rel=_PER_VALUE_REL), (
                f"{case['name']} probe {probe}"
            )

    def test_the_agreement_is_tight_not_merely_within_tolerance(self):
        """Loose per-value bounds must not hide a widening gap.

        The f32/f64 difference is a fixed, known quantity here; if a change to
        the port (or a regenerated fixture from a different Rust build) moves
        it, this is what notices.
        """
        model = _model()
        worst = 0.0
        for case in _CASES:
            params = _params(case)
            for probe, want_u, want_d in zip(
                case["probes"], case["undistorted"], case["distorted"]
            ):
                got_u = model.undistort_point(probe[0], probe[1], params)
                got_d = model.distort_point(probe[0], probe[1], 1.0, params)
                for got, want in ((got_u, want_u), (got_d, want_d)):
                    for index in range(2):
                        scale = max(abs(want[index]), 1.0)
                        worst = max(worst, abs(got[index] - want[index]) / scale)
        assert worst < _AGGREGATE_REL, f"worst relative disagreement {worst:.3e}"


class TestTheModelItself:
    def test_the_registry_resolves_this_name(self):
        """The bug this file exists for: the name used to fall back silently."""
        model = from_name("gopro6_superview")
        assert isinstance(model, GoPro6SuperviewModel)
        assert model.id() == "gopro6_superview"

    def test_it_is_not_the_other_superview(self):
        """Two different polynomials — collapsing them is the whole failure mode.

        Both are selectable digital lenses upstream and they produce different
        pixels; a port that aliased one to the other would look plausible.
        """
        params = KernelParams()
        params.width = params.output_width = 1920
        params.height = params.output_height = 1080
        probe = (1440.0, 540.0)
        six = GoPro6SuperviewModel().undistort_point(*probe, params)
        five = GoProSuperviewModel().undistort_point(*probe, params)
        assert six != pytest.approx(five, rel=1e-4)

    def test_the_centre_is_a_fixed_point(self):
        params = KernelParams()
        params.width = params.output_width = 1920
        params.height = params.output_height = 1080
        model = _model()
        assert model.undistort_point(960.0, 540.0, params) == pytest.approx(
            (960.0, 540.0)
        )
        assert model.distort_point(960.0, 540.0, 1.0, params) == pytest.approx(
            (960.0, 540.0)
        )

    def test_the_stretch_pulls_the_frame_inward(self):
        """Superview maps a wider FOV into the same frame: pixels move inward."""
        params = KernelParams()
        params.width = params.output_width = 1920
        params.height = params.output_height = 1080
        model = _model()
        x, _ = model.undistort_point(1700.0, 540.0, params)
        assert abs(x - 960.0) < abs(1700.0 - 960.0)

    def test_radial_distortion_limit_is_not_applicable(self):
        assert _model().radial_distortion_limit([0.0] * 12) is None

    def test_the_wgsl_is_the_digital_variant(self):
        wgsl = _model().wgsl_functions()
        assert "fn digital_undistort_point" in wgsl
        assert "fn digital_distort_point" in wgsl
        # The polynomial's own constants, so a copy-paste of the sibling model
        # would fail here.
        assert "0.48" in wgsl
        assert "0.157895" in wgsl
        assert "1.2100393" not in wgsl


class TestVectorizedPath:
    """``distort_points`` is what the render path calls, not the scalar loop."""

    def test_it_agrees_with_the_scalar_loop(self):
        params = KernelParams()
        params.width = params.output_width = 1920
        params.height = params.output_height = 1080
        model = _model()
        xs = np.array([960.0, 1200.0, 1440.0, 700.0, 1700.0])
        ys = np.array([540.0, 540.0, 540.0, 800.0, 500.0])

        vx, vy = model.distort_points(xs, ys, np.ones_like(xs), params)
        for index in range(xs.size):
            sx, sy = model.distort_point(
                float(xs[index]), float(ys[index]), 1.0, params
            )
            assert vx[index] == pytest.approx(sx, rel=1e-12)
            assert vy[index] == pytest.approx(sy, rel=1e-12)

    def test_it_broadcasts_over_a_grid(self):
        params = KernelParams()
        params.width = params.output_width = 1920
        params.height = params.output_height = 1080
        xs, ys = np.meshgrid(
            np.linspace(200.0, 1700.0, 5), np.linspace(100.0, 1000.0, 4)
        )
        vx, vy = _model().distort_points(xs, ys, np.ones_like(xs), params)
        assert vx.shape == xs.shape and vy.shape == ys.shape
        assert np.isfinite(vx).all() and np.isfinite(vy).all()


class TestTheInversionStepCap:
    """Upstream's loop stops at 12 steps; in part of the frame that is early.

    The substitution is not a contraction everywhere, so the number of steps
    needed to reach the 1e-6 epsilon ranges from one (at the centre, which is
    the exact fixed point) to more than twenty. Where it exceeds twelve the
    returned pre-image is upstream's *unconverged* answer, and the fixture
    records it — a port that iterated until it converged, or that added a
    divergence guard, would disagree with the reference.
    """

    def _steps_to_converge(self, pixel, cap=64):
        """Steps the same iteration needs to reach the loop's 1e-6 epsilon."""
        from pygyroflow.stabilization.distortion_models.gopro6_superview import (
            _superview,
        )

        target = (pixel[0] / 1920.0 - 0.5, pixel[1] / 1080.0 - 0.5)
        px, py = target
        for step in range(1, cap + 1):
            dp_x, dp_y = _superview((px, py))
            if abs(dp_x - target[0]) < 1e-6 and abs(dp_y - target[1]) < 1e-6:
                return step
            px -= dp_x - target[0]
            py -= dp_y - target[1]
        return cap

    def _steps_the_port_takes(self, pixel):
        """How many times ``distort_point`` evaluates the polynomial."""
        import pygyroflow.stabilization.distortion_models.gopro6_superview as module

        calls = []
        original = module._superview

        def counting(uv):
            calls.append(uv)
            return original(uv)

        module._superview = counting
        try:
            params = KernelParams()
            params.width = params.output_width = 1920
            params.height = params.output_height = 1080
            _model().distort_point(pixel[0], pixel[1], 1.0, params)
        finally:
            module._superview = original
        return len(calls)

    def test_the_loop_stops_as_soon_as_it_converges(self):
        assert self._steps_to_converge((960.0, 540.0)) == 1
        assert self._steps_the_port_takes((960.0, 540.0)) == 1

    def test_a_mid_frame_point_converges_inside_the_cap(self):
        needed = self._steps_to_converge((700.0, 900.0))
        assert needed == self._steps_the_port_takes((700.0, 900.0))
        assert needed <= 12

    def test_the_cap_binds_where_convergence_is_slow(self):
        """This probe needs 22 steps; upstream spends 12 and stops."""
        assert self._steps_to_converge((100.0, 900.0)) > 12
        assert self._steps_the_port_takes((100.0, 900.0)) == 12

    def test_the_capped_answer_is_the_one_in_the_fixture(self):
        """The fixture's value for that probe is the *unconverged* one.

        Asserting this keeps the truncation from being quietly "fixed": the
        fixture is upstream's output, so matching it means matching the cap.
        """
        case = next(c for c in _CASES if c["name"] == "1080p_frame")
        index = case["probes"].index([100.0, 900.0])
        params = _params(case)
        got = _model().distort_point(100.0, 900.0, 1.0, params)
        assert got == pytest.approx(tuple(case["distorted"][index]), rel=1e-5)
        assert got[0] < 0.0  # past the left edge of the frame


class TestThePreImageCanLeaveTheFrame:
    """The map pulls the frame's edge inward, so inverting pushes past it."""

    def test_a_target_near_the_edge_inverts_to_a_point_beyond_it(self):
        params = KernelParams()
        params.width = params.output_width = 1920
        params.height = params.output_height = 1080
        model = _model()
        point = (200.0, 200.0)
        # The x map compresses: 200 px in the undistorted frame came from a
        # point at about -57 px in the distorted one, i.e. outside the sensor.
        got = model.distort_point(point[0], point[1], 1.0, params)
        assert got[0] < 0.0
        # And undistorting it comes back, which is what makes it a pre-image
        # rather than a garbage answer. Not exact: the 12-step cap leaves a
        # 7.9e-6 residual here, which the forward map turns into 0.015 px.
        back = model.undistort_point(got[0], got[1], params)
        assert back == pytest.approx(point, abs=0.05)


class TestCoefficientsAreIrrelevant:
    """A digital lens runs after the optical model; it reads no coefficients."""

    def test_changing_the_coefficients_changes_nothing(self):
        model = _model()
        point = (1440.0, 540.0)
        results = []
        for k1 in ((0.0, 0.0, 0.0, 0.0), (-0.3, 0.1, 0.0, 0.0)):
            params = KernelParams()
            params.width = params.output_width = 1920
            params.height = params.output_height = 1080
            params.k1 = (ctypes.c_float * 4)(*k1)
            results.append(model.undistort_point(point[0], point[1], params))
        assert results[0] == pytest.approx(results[1])
