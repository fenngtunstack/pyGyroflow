"""PTLens distortion model (three-coefficient polynomial).

Ported from Gyroflow's ptlens.rs and the corresponding WGSL shader.

Forward:  r_d = r_u * (a * r_u^3 + b * r_u^2 + c * r_u + 1)
Inverse:  Newton-Raphson on f(r_u) = r_u*(a*r_u^3 + b*r_u^2 + c*r_u + 1) - r_d = 0

Coefficient layout (KernelParams):
  k1[0] = a  (cubic)
  k1[1] = b  (quadratic)
  k1[2] = c  (linear)
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


class PTLensModel(DistortionModelBase):
    """PTLens three-coefficient radial distortion model."""

    # -- undistort -------------------------------------------------------

    def undistort_point(
        self, x: float, y: float, params: "KernelParams"
    ) -> tuple[float, float] | None:
        a = float(params.k1[0])
        b = float(params.k1[1])
        c = float(params.k1[2])

        rd = math.sqrt(x * x + y * y)
        if rd == 0.0:
            return None

        ru = rd
        for i in range(10):
            fru = ru * (a * ru * ru * ru + b * ru * ru + c * ru + 1.0) - rd
            if -_NEWTON_EPS <= fru < _NEWTON_EPS:
                break
            if i > 5:
                return None
            ru = ru - (
                fru / (4.0 * a * ru * ru * ru + 3.0 * b * ru * ru + 2.0 * c * ru + 1.0)
            )

        if ru < 0.0:
            return None

        ru = ru / rd
        return (x * ru, y * ru)

    # -- distort ---------------------------------------------------------

    def distort_point(
        self, x: float, y: float, z: float, params: "KernelParams"
    ) -> tuple[float, float]:
        a = float(params.k1[0])
        b = float(params.k1[1])
        c = float(params.k1[2])
        x = x / z
        y = y / z
        ru2 = x * x + y * y
        r = math.sqrt(ru2)
        poly3 = a * ru2 * r + b * ru2 + c * r + 1.0
        return (x * poly3, y * poly3)

    def distort_points(self, xs, ys, zs, params):
        """Vectorized forward PTLens distortion."""
        a = float(params.k1[0])
        b = float(params.k1[1])
        c = float(params.k1[2])
        x = xs / zs
        y = ys / zs
        ru2 = x * x + y * y
        r = np.sqrt(ru2)
        poly3 = a * ru2 * r + b * ru2 + c * r + 1.0
        return x * poly3, y * poly3

    # -- radial distortion limit -----------------------------------------

    def radial_distortion_limit(self, coeffs: list[float]) -> float | None:
        if len(coeffs) < 3:
            return None
        a, b, c = coeffs[0], coeffs[1], coeffs[2]
        low = 0.0
        high = _HALF_PI
        tolerance = 1e-4

        while high - low > tolerance:
            mid = (low + high) / 2.0
            ru = mid
            deriv = 4.0 * a * ru * ru * ru + 3.0 * b * ru * ru + 2.0 * c * ru + 1.0
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
        let fru = ru * (params.k1.x * ru * ru * ru + params.k1.y * ru * ru + params.k1.z * ru + 1.0) - rd;
        if (fru >= -NEWTON_EPS && fru < NEWTON_EPS) {
            break;
        }
        if (i > 5) {
            // Does not converge, no real solution in this area?
            return vec2<f32>(0.0, 0.0);
        }

        ru = ru - (fru / (4.0 * params.k1.x * ru * ru * ru + 3.0 * params.k1.y * ru * ru + 2.0 * params.k1.z * ru + 1.0));
    }
    if (ru < 0.0) {
        return vec2<f32>(0.0, 0.0);
    }

    ru = ru / rd;

    // Apply only requested amount
    ru = 1.0 + (ru - 1.0) * (1.0 - amount);

    return pos * ru;
}

fn distort_point(x: f32, y: f32, z: f32) -> vec2<f32> {
    let pos = vec2<f32>(x, y) / z;
    let ru2 = (pos.x * pos.x + pos.y * pos.y);
    let r = sqrt(ru2);
    let poly3 = params.k1.x * ru2 * r + params.k1.y * ru2 + params.k1.z * r + 1.0;
    return pos * poly3;
}
"""
        )

    def id(self) -> str:
        return "ptlens"
