# -*- coding: utf-8 -*-
"""Tests for EWA (Elliptical Weighted Average) resampling.

The reference here is a scalar, line-by-line transcription of upstream's
``sample_input_at`` EWA branch (``cpu_undistort.rs``): the vectorized
implementation is compared against it tap by tap, which checks the ellipse,
the bounding box, the CubicBC kernel and the accumulation independently of
how the vectorized version arranges the loops.

Invariants that catch whole classes of weighting bugs are checked separately:
a constant image must survive any Jacobian exactly (normalisation), off-frame
taps must take the background colour, and the four filters must match the
(B, C) parameters upstream uses.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pygyroflow.stabilization.ewa import (
    EWA_FILTERS,
    EWA_NAMES,
    MAX_RADIUS,
    affine_bbox,
    bc2,
    clamped_ellipse,
    cubic_bc_coeffs,
    ewa_sample,
    map_jacobian,
)

IDENTITY_JAC = (1.0, 0.0, 0.0, 1.0)


# --------------------------------------------------------------------------- #
#  Scalar reference — transcribed from the Rust, not shared with the port      #
# --------------------------------------------------------------------------- #


def _affine_bbox_scalar(jac):
    jx, jy, jz, jw = jac
    return (
        2.0 * max(abs(jx + jy), abs(jx - jy), 1.0),
        2.0 * max(abs(jz + jw), abs(jz - jw), 1.0),
    )


def _clamped_ellipse_scalar(jac):
    jx, jy, jz, jw = jac
    f0 = abs(jx * jw - jy * jz)
    f = max(f0 * f0, 0.1)
    a = (jz * jz + jw * jw) / f
    b = -2.0 * (jx * jz + jy * jw) / f
    c = (jx * jx + jy * jy) / f

    vx, vy = c - a, -b
    lv = math.hypot(vx, vy)
    v0 = vx / lv if lv > 0.01 else 1.0
    cc = math.sqrt(max(1.0 + v0, 0.0) / 2.0)
    s = math.sqrt(max(1.0 - v0, 0.0) / 2.0)

    a0 = a * cc * cc - b * cc * s + c * s * s
    c0 = a * s * s + b * cc * s + c * cc * cc
    bt1 = b * (cc * cc - s * s)
    bt2 = 2.0 * (a - c) * cc * s
    b0, b0_alt = bt1 + bt2, bt1 - bt2
    if abs(b0) > abs(b0_alt):
        s = -s
        b0 = b0_alt
    a0, c0 = min(a0, 1.0), min(c0, 1.0)
    sn = -s
    return (
        a0 * cc * cc - b0 * cc * sn + c0 * sn * sn,
        2.0 * a0 * cc * sn + b0 * cc * cc - b0 * sn * sn - 2.0 * c0 * cc * sn,
        a0 * sn * sn + b0 * cc * sn + c0 * cc * cc,
    )


def _bc2_scalar(x, p, q):
    x = abs(x)
    x2 = x * x
    if x < 1.0:
        return p[0] + p[1] * x + p[2] * x2 + p[3] * x2 * x
    if x < 2.0:
        return q[0] + q[1] * x + q[2] * x2 + q[3] * x2 * x
    return 0.0


def scalar_ewa(frame, map_x, map_y, jac, valid, interpolation, border):
    """Upstream's EWA branch, written out as plain loops."""
    p, q = cubic_bc_coeffs(*EWA_FILTERS[interpolation])
    p, q = p.tolist(), q.tolist()
    height, width = map_x.shape
    channels = frame.shape[2] if frame.ndim == 3 else 1
    border = np.broadcast_to(np.asarray(border, dtype=np.float64).reshape(-1), (channels,))

    out = np.zeros((height, width, channels), dtype=np.float64)
    for y in range(height):
        for x in range(width):
            if not valid[y, x]:
                out[y, x] = border
                continue

            uv = (map_x[y, x], map_y[y, x])
            J = tuple(comp[y, x] for comp in jac)
            tx, ty = _affine_bbox_scalar(J)
            ea, eb, ec = _clamped_ellipse_scalar(J)

            total = np.zeros(channels, dtype=np.float64)
            weight_sum = 0.0
            for iy in range(int(math.floor(uv[1] - ty)), int(math.ceil(uv[1] + ty)) + 1):
                in_fy = iy - uv[1]
                in_fy2 = in_fy * eb
                in_fy3 = in_fy * in_fy * ec
                for ix in range(int(math.floor(uv[0] - tx)), int(math.ceil(uv[0] + tx)) + 1):
                    in_fx = ix - uv[0]
                    dr = in_fx * in_fx * ea + in_fx * in_fy2 + in_fy3
                    k = _bc2_scalar(math.sqrt(dr), p, q)
                    if k == 0.0:
                        continue
                    if 0 <= iy < height and 0 <= ix < width:
                        px = frame[iy, ix]
                    else:
                        px = border
                    total += k * px
                    weight_sum += k

            out[y, x] = total / weight_sum if weight_sum != 0.0 else border
    return out[:, :, 0] if frame.ndim == 2 else out


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #


def make_frame(h=12, w=16, channels=3, seed=4):
    rng = np.random.default_rng(seed)
    shape = (h, w, channels) if channels > 1 else (h, w)
    return rng.integers(0, 256, shape, dtype=np.uint8)


def make_jac(h, w, kind="identity"):
    if kind == "identity":
        return (np.full((h, w), 1.0), np.zeros((h, w)), np.zeros((h, w)), np.full((h, w), 1.0))
    if kind == "shrink_x":  # minification along x only -> anisotropic ellipse
        return (np.full((h, w), 2.0), np.zeros((h, w)), np.zeros((h, w)), np.full((h, w), 1.0))
    if kind == "rotate":
        rng = np.random.default_rng(2)
        theta = rng.uniform(-1.0, 1.0, (h, w))
        c, s = np.cos(theta), np.sin(theta)
        return (c, -s, s, c)
    raise ValueError(kind)


# --------------------------------------------------------------------------- #
#  Kernel and geometry                                                         #
# --------------------------------------------------------------------------- #


class TestKernels:
    def test_filter_table_matches_upstream(self):
        assert EWA_FILTERS == {
            3: (0.2620145, 0.3689927),
            4: (0.3782157, 0.3108921),
            5: (0.3333333, 0.3333333),
            6: (0.0000000, 0.5000000),
        }
        assert EWA_NAMES[3] == "RobidouxSharp"
        assert EWA_NAMES[6] == "CatmullRom"

    @pytest.mark.parametrize("interp", sorted(EWA_FILTERS))
    def test_kernel_support_ends_at_two(self, interp):
        p, q = cubic_bc_coeffs(*EWA_FILTERS[interp])
        v = bc2(np.array([2.0, 2.5, 8.0]), p, q)
        assert np.allclose(v, 0.0, atol=1e-12)

    def test_only_catmull_rom_is_interpolating(self):
        """|x| = 1 lands on a lattice point: Catmull-Rom (B=0) is zero there,
        the other three are approximating filters and are not."""
        catmull_p, catmull_q = cubic_bc_coeffs(*EWA_FILTERS[6])
        assert float(bc2(np.array([1.0]), catmull_p, catmull_q)[0]) == pytest.approx(0.0, abs=1e-12)
        for interp in (3, 4, 5):
            p, q = cubic_bc_coeffs(*EWA_FILTERS[interp])
            assert float(bc2(np.array([1.0]), p, q)[0]) != pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("interp", sorted(EWA_FILTERS))
    def test_kernel_peak_matches_the_b_parameter(self, interp):
        """bc2(0) = (6 - 2B)/6 — 1.0 only for Catmull-Rom."""
        b_param, _c_param = EWA_FILTERS[interp]
        p, q = cubic_bc_coeffs(b_param, _c_param)
        assert float(bc2(np.array([0.0]), p, q)[0]) == pytest.approx((6.0 - 2.0 * b_param) / 6.0)

    def test_catmull_rom_peak_is_one(self):
        p, q = cubic_bc_coeffs(*EWA_FILTERS[6])
        assert float(bc2(np.array([0.0]), p, q)[0]) == pytest.approx(1.0)

    def test_catmull_rom_has_negative_lobes(self):
        p, q = cubic_bc_coeffs(*EWA_FILTERS[6])
        assert float(bc2(np.array([math.sqrt(2.0)]), p, q)[0]) < 0.0

    def test_bc2_matches_the_scalar_transcription(self):
        p, q = cubic_bc_coeffs(*EWA_FILTERS[3])
        xs = np.linspace(0.0, 2.5, 41)
        got = bc2(xs, p, q)
        want = np.array([_bc2_scalar(float(x), p.tolist(), q.tolist()) for x in xs])
        assert np.allclose(got, want, atol=1e-12)


class TestEllipse:
    def test_identity_jacobian_gives_a_unit_circle(self):
        jac = tuple(np.array([v]) for v in IDENTITY_JAC)
        a, b, c = clamped_ellipse(jac)
        assert (float(a[0]), float(b[0]), float(c[0])) == pytest.approx((1.0, 0.0, 1.0))

    def test_identity_bbox_is_the_two_pixel_support(self):
        jac = tuple(np.array([v]) for v in IDENTITY_JAC)
        assert [float(np.ravel(x)[0]) for x in affine_bbox(jac)] == pytest.approx([2.0, 2.0])

    def test_minification_widens_the_box(self):
        jac = tuple(np.array([v]) for v in (2.0, 0.0, 0.0, 1.0))
        bx, by = affine_bbox(jac)
        assert float(bx[0]) == pytest.approx(4.0)  # 2x compression -> twice the reach
        assert float(by[0]) == pytest.approx(2.0)

    def test_rotation_is_covered_in_both_axes(self):
        theta = 0.4
        jac = tuple(np.array([v]) for v in (math.cos(theta), -math.sin(theta),
                                            math.sin(theta), math.cos(theta)))
        bx, by = affine_bbox(jac)
        # a rotated unit circle needs at least sqrt(2)*2 of reach on both axes
        assert float(bx[0]) >= 2.0
        assert float(by[0]) >= 2.0

    def test_ellipse_matches_the_scalar_transcription(self):
        jac_arrays = make_jac(4, 5, "rotate")
        a, b, c = clamped_ellipse(jac_arrays)
        for y in range(4):
            for x in range(5):
                want = _clamped_ellipse_scalar(tuple(comp[y, x] for comp in jac_arrays))
                assert (float(a[y, x]), float(b[y, x]), float(c[y, x])) == pytest.approx(want, abs=1e-12)

    def test_degenerate_jacobian_is_clamped_not_exploded(self):
        # A flat region (zero Jacobian) must not produce an unbounded ellipse.
        jac = tuple(np.zeros((2, 2)) for _ in range(4))
        a, b, c = clamped_ellipse(jac)
        assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.all(np.isfinite(c))
        bx, by = affine_bbox(jac)
        assert float(bx.max()) == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
#  Parity with the scalar reference                                            #
# --------------------------------------------------------------------------- #


class TestScalarParity:
    @pytest.mark.parametrize("interp", sorted(EWA_FILTERS))
    @pytest.mark.parametrize("kind", ["identity", "shrink_x", "rotate"])
    def test_vectorized_matches_scalar(self, interp, kind):
        h, w = 12, 16
        frame = make_frame(h, w)
        # a smooth, mildly sub-pixel map keeps the identity-ish cases honest
        ys, xs = np.mgrid[0:h, 0:w]
        map_x = xs + 0.25 * np.sin(ys * 0.3)
        map_y = ys + 0.25 * np.cos(xs * 0.4)
        jac = make_jac(h, w, kind)
        valid = np.ones((h, w), dtype=bool)
        border = (3.0, 7.0, 11.0)

        got = ewa_sample(frame, map_x, map_y, jac, valid, interp, border)
        want = scalar_ewa(frame, map_x, map_y, jac, valid, interp, border)

        assert np.allclose(got.astype(np.float64), np.round(np.clip(want, 0, 255)),
                           atol=1.0, rtol=0)

    @pytest.mark.parametrize("interp", sorted(EWA_FILTERS))
    def test_vectorized_matches_scalar_with_float_input(self, interp):
        h, w = 10, 12
        rng = np.random.default_rng(8)
        frame = rng.random((h, w, 1)).astype(np.float32)
        ys, xs = np.mgrid[0:h, 0:w]
        map_x = xs + 0.3
        map_y = ys - 0.2
        jac = make_jac(h, w, "identity")
        border = [0.0]

        got = ewa_sample(frame, map_x, map_y, jac, np.ones((h, w), bool), interp, border)
        want = scalar_ewa(frame, map_x, map_y, jac, np.ones((h, w), bool), interp, border)

        assert np.allclose(got, want, atol=1e-5)


# --------------------------------------------------------------------------- #
#  Invariants                                                                  #
# --------------------------------------------------------------------------- #


class TestInvariants:
    @pytest.mark.parametrize("interp", sorted(EWA_FILTERS))
    @pytest.mark.parametrize("kind", ["identity", "shrink_x", "rotate"])
    def test_constant_image_is_preserved(self, interp, kind):
        """The weights must normalise: any Jacobian, any filter.  The border
        carries the same value so that normalisation is isolated from the
        deliberate background bleed at the frame edge."""
        h, w = 12, 16
        frame = np.full((h, w, 3), 77, np.uint8)
        ys, xs = np.mgrid[0:h, 0:w]
        jac = make_jac(h, w, kind)

        out = ewa_sample(frame, xs.astype(np.float64), ys.astype(np.float64),
                         jac, np.ones((h, w), bool), interp, (77.0, 77.0, 77.0))

        assert np.array_equal(out, frame)

    def test_off_frame_taps_take_the_border(self):
        """A map pointing outside the frame must yield the background."""
        h, w = 8, 8
        frame = np.full((h, w, 1), 200, np.uint8)
        map_x = np.full((h, w), -50.0)
        map_y = np.full((h, w), -50.0)
        jac = make_jac(h, w, "identity")

        out = ewa_sample(frame, map_x, map_y, jac, np.ones((h, w), bool), 5, (11.0,))

        assert np.all(out == 11)

    def test_invalid_pixels_take_the_border(self):
        h, w = 8, 8
        frame = np.full((h, w, 1), 200, np.uint8)
        ys, xs = np.mgrid[0:h, 0:w]
        valid = np.ones((h, w), bool)
        valid[2:5, 2:5] = False

        out = ewa_sample(frame, xs.astype(np.float64), ys.astype(np.float64),
                         make_jac(h, w, "identity"), valid, 5, (33.0,))

        assert np.all(out[2:5, 2:5] == 33)
        assert np.all(out[~valid] == 33)

    def test_background_bleeds_into_the_frame_edge(self):
        """Upstream substitutes *bg for off-frame taps rather than
        renormalising, so a dark border tints the outermost pixels."""
        h, w = 10, 10
        frame = np.full((h, w, 1), 200, np.uint8)
        ys, xs = np.mgrid[0:h, 0:w]

        out = ewa_sample(frame, xs.astype(np.float64), ys.astype(np.float64),
                         make_jac(h, w, "identity"), np.ones((h, w), bool), 5, (0.0,))

        assert out[0, 0] < 200          # pulled toward the background
        assert out[5, 5] == 200         # interior untouched

    def test_output_never_wraps_around_the_dtype(self):
        """Weighted sums can overshoot with border bleed; uint8 must clamp,
        not wrap (that turned edges into speckle)."""
        h, w = 10, 10
        frame = np.full((h, w, 3), 254, np.uint8)
        ys, xs = np.mgrid[0:h, 0:w]

        out = ewa_sample(frame, xs.astype(np.float64), ys.astype(np.float64),
                         make_jac(h, w, "identity"), np.ones((h, w), bool), 3, (255.0, 255.0, 255.0))

        assert out.dtype == np.uint8
        assert out.min() >= 200 and out.max() <= 255

    def test_radius_cap_is_reported_by_the_box(self):
        jac = tuple(np.array([v]) for v in (40.0, 0.0, 0.0, 40.0))
        bx, by = affine_bbox(jac)
        assert float(bx[0]) > MAX_RADIUS  # the cap is what bounds the loop

    def test_rejects_a_non_ewa_index(self):
        with pytest.raises(ValueError):
            ewa_sample(np.zeros((4, 4, 1), np.uint8), np.zeros((4, 4)), np.zeros((4, 4)),
                       make_jac(4, 4), np.ones((4, 4), bool), 2, (0.0,))


# --------------------------------------------------------------------------- #
#  Map Jacobian                                                                #
# --------------------------------------------------------------------------- #


class TestMapJacobian:
    def test_identity_map_has_identity_jacobian(self):
        ys, xs = np.mgrid[0:8, 0:10]
        jac = map_jacobian(xs.astype(float), ys.astype(float))
        assert np.allclose(jac[0], 1.0)     # du/dx
        assert np.allclose(jac[1], 0.0)
        assert np.allclose(jac[2], 0.0)
        assert np.allclose(jac[3], 1.0)     # dv/dy

    def test_scaled_map_reports_the_scale(self):
        ys, xs = np.mgrid[0:8, 0:10]
        jac = map_jacobian(2.0 * xs.astype(float), 0.5 * ys.astype(float))
        assert np.allclose(jac[0], 2.0)
        assert np.allclose(jac[3], 0.5)

    def test_invalid_pixels_do_not_inflate_the_radius(self):
        """Placeholder coordinates at invalid pixels would otherwise show up
        as enormous derivatives and blow the sampling box up frame-wide."""
        ys, xs = np.mgrid[0:8, 0:10]
        map_x = xs.astype(float)
        map_y = ys.astype(float)
        valid = np.ones((8, 10), bool)
        valid[3:5, 3:5] = False
        map_x[~valid] = -1e5
        map_y[~valid] = -1e5

        jac = map_jacobian(map_x, map_y, valid)
        bx, by = affine_bbox(jac)

        assert float(bx.max()) == pytest.approx(2.0)
        assert float(by.max()) == pytest.approx(2.0)

    def test_halo_around_invalid_pixels_is_neutralised(self):
        ys, xs = np.mgrid[0:8, 0:10]
        map_x = xs.astype(float)
        map_y = ys.astype(float)
        valid = np.ones((8, 10), bool)
        valid[2, 2] = False
        map_x[~valid] = -1e5

        jac = map_jacobian(map_x, map_y, valid)

        assert jac[0][1, 1] == pytest.approx(1.0)   # neighbour, not 1e5
        assert jac[0][2, 2] == pytest.approx(1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
