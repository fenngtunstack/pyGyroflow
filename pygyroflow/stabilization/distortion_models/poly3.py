"""Poly3 distortion model (single-coefficient cubic polynomial).

Ported from Gyroflow's poly3.rs and the corresponding WGSL shader.

Forward:  r_d = k1 * r_u^3 + r_u  =  r_u * (k1 * r_u^2 + 1)
Inverse:  Newton-Raphson on f(r_u) = r_u^3 + r_u/k1 - r_d/k1 = 0

Coefficient layout (KernelParams):
  k1[0] = k1
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .base import DistortionModelBase

if TYPE_CHECKING:
    from pygyroflow.types.kernel_params import KernelParams

_HALF_PI = math.pi / 2.0
_NEWTON_EPS = 1e-5


class Poly3Model(DistortionModelBase):
    """Poly3 single-coefficient cubic radial distortion model."""

    # -- undistort -------------------------------------------------------

    def undistort_point(
        self, x: float, y: float, params: "KernelParams"
    ) -> tuple[float, float] | None:
        k1 = float(params.k1[0])
        if k1 == 0.0:
            # No distortion: identity (mirrors the fisheye zero-coeffs guard)
            return (x, y)
        inv_k1 = 1.0 / k1

        rd = math.sqrt(x * x + y * y)
        if rd == 0.0:
            return None

        rd_div_k1 = rd * inv_k1

        # Newton-Raphson on r_u^3 + r_u/k1 - r_d/k1 = 0
        ru = rd
        for i in range(10):
            fru = ru * ru * ru + ru * inv_k1 - rd_div_k1
            if -_NEWTON_EPS <= fru < _NEWTON_EPS:
                break
            if i > 5:
                return None
            ru = ru - (fru / (3.0 * ru * ru + inv_k1))

        if ru < 0.0:
            return None

        ru = ru / rd
        return (x * ru, y * ru)

    # -- distort ---------------------------------------------------------

    def distort_point(
        self, x: float, y: float, z: float, params: "KernelParams"
    ) -> tuple[float, float]:
        k1 = float(params.k1[0])
        x = x / z
        y = y / z
        # r_d = r_u * (k1 * r_u^2 + 1)  =>  scale = k1 * r^2 + 1
        poly2 = k1 * (x * x + y * y) + 1.0
        return (x * poly2, y * poly2)

    def distort_points(self, xs, ys, zs, params):
        """Vectorized forward poly3 distortion."""
        k1 = float(params.k1[0])
        x = xs / zs
        y = ys / zs
        # r_d = r_u * (k1 * r_u^2 + 1)  =>  scale = k1 * r^2 + 1
        poly2 = k1 * (x * x + y * y) + 1.0
        return x * poly2, y * poly2

    # -- radial distortion limit -----------------------------------------

    def radial_distortion_limit(self, coeffs: list[float]) -> float | None:
        if len(coeffs) < 1:
            return None
        k1 = coeffs[0]
        inv_k1 = 1.0 / k1
        low = 0.0
        high = _HALF_PI
        tolerance = 1e-4

        while high - low > tolerance:
            mid = (low + high) / 2.0
            ru = mid
            deriv = 3.0 * ru * ru + inv_k1
            if deriv > 0.0:
                low = mid
            else:
                high = mid

        theta_max = (low + high) / 2.0
        if abs(theta_max - _HALF_PI) > 0.001:
            return math.tan(theta_max)
        return None

    # -- WGSL shader -----------------------------------------------------

    def wgsl_functions(self) -> str:
        return (
            """
fn undistort_point(pos: vec2<f32>) -> vec2<f32> {
    let NEWTON_EPS = 0.00001;

    let inv_k1 = (1.0 / params.k1.x);

    let rd = length(pos);
    if (rd == 0.0) { return vec2<f32>(0.0, 0.0); }

    let rd_div_k1 = rd * inv_k1;

    // Use Newton's method to avoid dealing with complex numbers.
    // When carefully tuned this works almost as fast as Cardano's method (and we don't use complex numbers in it, which is required for a full solution!)
    //
    // Original function: Rd = k1_ * Ru^3 + Ru
    // Target function:   k1_ * Ru^3 + Ru - Rd = 0
    // Divide by k1_:     Ru^3 + Ru/k1_ - Rd/k1_ = 0
    // Derivative:        3 * Ru^2 + 1/k1_
    var ru = rd;
    for (var i: i32 = 0; i < 10; i = i + 1) {
        let fru = ru * ru * ru + ru * inv_k1 - rd_div_k1;
        if (fru >= -NEWTON_EPS && fru < NEWTON_EPS) {
            break;
        }
        if (i > 5) {
            // Does not converge, no real solution in this area?
            return vec2<f32>(0.0, 0.0);
        }

        ru = ru - (fru / (3.0 * ru * ru + inv_k1));
    }
    if (ru < 0.0) {
        return vec2<f32>(0.0, 0.0);
    }

    ru = ru / rd;

    return pos * ru;
}

fn distort_point(x: f32, y: f32, z: f32) -> vec2<f32> {
    let pos = vec2<f32>(x, y) / z;
    let poly2 = params.k1.x * (pos.x * pos.x + pos.y * pos.y) + 1.0;
    return pos * poly2;
}
"""
        )

    def id(self) -> str:
        return "poly3"
