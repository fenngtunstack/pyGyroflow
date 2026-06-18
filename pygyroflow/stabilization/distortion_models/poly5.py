"""Poly5 distortion model (two-coefficient quintic polynomial).

Ported from Gyroflow's poly5.rs and the corresponding WGSL shader.

Forward:  r_d = r_u * (1 + k1 * r_u^2 + k2 * r_u^4)
Inverse:  Newton-Raphson on f(r_u) = r_u*(1 + k1*r_u^2 + k2*r_u^4) - r_d = 0

Coefficient layout (KernelParams):
  k1[0] = k1
  k1[1] = k2
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .base import DistortionModelBase

if TYPE_CHECKING:
    from pygyroflow.types.kernel_params import KernelParams

_HALF_PI = math.pi / 2.0
_NEWTON_EPS = 1e-5


class Poly5Model(DistortionModelBase):
    """Poly5 two-coefficient quintic radial distortion model."""

    # -- undistort -------------------------------------------------------

    def undistort_point(
        self, x: float, y: float, params: "KernelParams"
    ) -> tuple[float, float] | None:
        k1 = float(params.k1[0])
        k2 = float(params.k1[1])

        rd = math.sqrt(x * x + y * y)
        if rd == 0.0:
            return None

        ru = rd
        for i in range(10):
            ru2 = ru * ru
            fru = ru * (1.0 + k1 * ru2 + k2 * ru2 * ru2) - rd
            if -_NEWTON_EPS <= fru < _NEWTON_EPS:
                break
            if i > 5:
                return None
            ru = ru - (fru / (1.0 + 3.0 * k1 * ru2 + 5.0 * k2 * ru2 * ru2))

        if ru < 0.0:
            return None

        ru = ru / rd
        return (x * ru, y * ru)

    # -- distort ---------------------------------------------------------

    def distort_point(
        self, x: float, y: float, z: float, params: "KernelParams"
    ) -> tuple[float, float]:
        k1 = float(params.k1[0])
        k2 = float(params.k1[1])
        x = x / z
        y = y / z
        ru2 = x * x + y * y
        poly4 = 1.0 + k1 * ru2 + k2 * ru2 * ru2
        return (x * poly4, y * poly4)

    # -- radial distortion limit -----------------------------------------

    def radial_distortion_limit(self, coeffs: list[float]) -> float | None:
        if len(coeffs) < 2:
            return None
        k1, k2 = coeffs[0], coeffs[1]
        low = 0.0
        high = _HALF_PI
        tolerance = 1e-4

        while high - low > tolerance:
            mid = (low + high) / 2.0
            ru2 = mid * mid
            deriv = 1.0 + 3.0 * k1 * ru2 + 5.0 * k2 * ru2 * ru2
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

    let rd = length(pos);
    if (rd == 0.0) { return vec2<f32>(0.0, 0.0); }

    var ru = rd;
    for (var i: i32 = 0; i < 10; i = i + 1) {
        let ru2 = ru * ru;
        let fru = ru * (1.0 + params.k1.x * ru2 + params.k1.y * ru2 * ru2) - rd;
        if (fru >= -NEWTON_EPS && fru < NEWTON_EPS) {
            break;
        }
        if (i > 5) {
            // Does not converge, no real solution in this area?
            return vec2<f32>(0.0, 0.0);
        }

        ru = ru - (fru / (1.0 + 3.0 * params.k1.x * ru2 + 5.0 * params.k1.y * ru2 * ru2));
    }
    if (ru < 0.0) {
        return vec2<f32>(0.0, 0.0);
    }

    ru = ru / rd;

    return pos * ru;
}

fn distort_point(x: f32, y: f32, z: f32) -> vec2<f32> {
    let pos = vec2<f32>(x, y) / z;
    let ru2 = (pos.x * pos.x + pos.y * pos.y);
    let poly4 = 1.0 + params.k1.x * ru2 + params.k1.y * ru2 * ru2;
    return pos * poly4;
}
"""
        )

    def id(self) -> str:
        return "poly5"
