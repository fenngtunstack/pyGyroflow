"""OpenCV Standard (Brown-Conrady) distortion model.

Ported from Gyroflow's opencv_standard.rs and the corresponding WGSL shader.
Standard pinhole camera with radial (k1-k6) and tangential (p1-p2) distortion,
plus thin-prism terms (s1-s4).  Up to 12 coefficients.

Coefficient layout (KernelParams):
  k1[0] = k1   (forward radial, r^2)
  k1[1] = k2   (forward radial, r^4)
  k1[2] = p1   (tangential)
  k1[3] = p2   (tangential)
  k2[0] = k3   (forward radial, r^6)
  k2[1] = k6   (inverse radial, r^2)
  k2[2] = k7   (inverse radial, r^4)
  k2[3] = k8   (inverse radial, r^6)
  k3[0] = s1   (thin prism, x, r^2)
  k3[1] = s2   (thin prism, x, r^4)
  k3[2] = s3   (thin prism, y, r^2)
  k3[3] = s4   (thin prism, y, r^4)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .base import DistortionModelBase

if TYPE_CHECKING:
    from pygyroflow.types.kernel_params import KernelParams

_HALF_PI = math.pi / 2.0


class OpenCVStandardModel(DistortionModelBase):
    """OpenCV Standard (Brown-Conrady) distortion model."""

    # -- undistort -------------------------------------------------------

    def undistort_point(
        self, x: float, y: float, params: "KernelParams"
    ) -> tuple[float, float] | None:
        # Unpack coefficients
        k1_0 = float(params.k1[0])  # k1 (radial r^2)
        k1_1 = float(params.k1[1])  # k2 (radial r^4)
        k1_2 = float(params.k1[2])  # p1 (tangential)
        k1_3 = float(params.k1[3])  # p2 (tangential)
        k2_0 = float(params.k2[0])  # k3 (radial r^6)
        k2_1 = float(params.k2[1])  # k6 (inv radial r^2)
        k2_2 = float(params.k2[2])  # k7 (inv radial r^4)
        k2_3 = float(params.k2[3])  # k8 (inv radial r^6)
        k3_0 = float(params.k3[0])  # s1 (thin prism x r^2)
        k3_1 = float(params.k3[1])  # s2 (thin prism x r^4)
        k3_2 = float(params.k3[2])  # s3 (thin prism y r^2)
        k3_3 = float(params.k3[3])  # s4 (thin prism y r^4)

        ux, uy = x, y
        x0, y0 = x, y

        for _ in range(20):
            r2 = ux * ux + uy * uy
            icdist = (1.0 + ((k2_3 * r2 + k2_2) * r2 + k2_1) * r2) / (
                1.0 + ((k2_0 * r2 + k1_1) * r2 + k1_0) * r2
            )
            if icdist < 0.0:
                return None

            delta_x = (
                2.0 * k1_2 * ux * uy
                + k1_3 * (r2 + 2.0 * ux * ux)
                + k3_0 * r2
                + k3_1 * r2 * r2
            )
            delta_y = (
                k1_2 * (r2 + 2.0 * uy * uy)
                + 2.0 * k1_3 * ux * uy
                + k3_2 * r2
                + k3_3 * r2 * r2
            )

            ux = (x0 - delta_x) * icdist
            uy = (y0 - delta_y) * icdist

        return (ux, uy)

    # -- distort ---------------------------------------------------------

    def distort_point(
        self, x: float, y: float, z: float, params: "KernelParams"
    ) -> tuple[float, float]:
        k1_0 = float(params.k1[0])
        k1_1 = float(params.k1[1])
        k1_2 = float(params.k1[2])
        k1_3 = float(params.k1[3])
        k2_0 = float(params.k2[0])
        k2_1 = float(params.k2[1])
        k2_2 = float(params.k2[2])
        k2_3 = float(params.k2[3])
        k3_0 = float(params.k3[0])
        k3_1 = float(params.k3[1])
        k3_2 = float(params.k3[2])
        k3_3 = float(params.k3[3])

        x = x / z
        y = y / z
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        a1 = 2.0 * x * y
        a2 = r2 + 2.0 * x * x
        a3 = r2 + 2.0 * y * y

        cdist = 1.0 + k1_0 * r2 + k1_1 * r4 + k2_0 * r6
        icdist2 = 1.0 / (1.0 + k2_1 * r2 + k2_2 * r4 + k2_3 * r6)

        xd0 = (
            x * cdist * icdist2 + k1_2 * a1 + k1_3 * a2 + k3_0 * r2 + k3_1 * r4
        )
        yd0 = (
            y * cdist * icdist2 + k1_2 * a3 + k1_3 * a1 + k3_2 * r2 + k3_3 * r4
        )
        return (xd0, yd0)

    def distort_points(self, xs, ys, zs, params):
        """Vectorized forward Brown-Conrady distortion."""
        k1_0 = float(params.k1[0]); k1_1 = float(params.k1[1])
        k1_2 = float(params.k1[2]); k1_3 = float(params.k1[3])
        k2_0 = float(params.k2[0]); k2_1 = float(params.k2[1])
        k2_2 = float(params.k2[2]); k2_3 = float(params.k2[3])
        k3_0 = float(params.k3[0]); k3_1 = float(params.k3[1])
        k3_2 = float(params.k3[2]); k3_3 = float(params.k3[3])

        x = xs / zs
        y = ys / zs
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        a1 = 2.0 * x * y
        a2 = r2 + 2.0 * x * x
        a3 = r2 + 2.0 * y * y

        cdist = 1.0 + k1_0 * r2 + k1_1 * r4 + k2_0 * r6
        icdist2 = 1.0 / (1.0 + k2_1 * r2 + k2_2 * r4 + k2_3 * r6)

        xd0 = (
            x * cdist * icdist2 + k1_2 * a1 + k1_3 * a2 + k3_0 * r2 + k3_1 * r4
        )
        yd0 = (
            y * cdist * icdist2 + k1_2 * a3 + k1_3 * a1 + k3_2 * r2 + k3_3 * r4
        )
        return xd0, yd0

    # -- radial distortion limit -----------------------------------------

    def radial_distortion_limit(self, coeffs: list[float]) -> float | None:
        if len(coeffs) < 8:
            return None
        k = coeffs[:8]
        low = 0.0
        high = _HALF_PI
        tolerance = 1e-4

        while high - low > tolerance:
            mid = (low + high) / 2.0
            r2 = mid * mid
            deriv = (1.0 + ((k[7] * r2 + k[6]) * r2 + k[5]) * r2) / (
                1.0 + ((k[4] * r2 + k[1]) * r2 + k[0]) * r2
            )
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
fn undistort_point(pos_param: vec2<f32>) -> vec2<f32> {
    var pos = pos_param;

    let start_pos = pos;

    // compensate distortion iteratively
    for (var i: i32 = 0; i < 20; i = i + 1) {
        let r2 = pos.x * pos.x + pos.y * pos.y;
        let icdist = (1.0 + ((params.k2.w * r2 + params.k2.z) * r2 + params.k2.y) * r2)/(1.0 + ((params.k2.x * r2 + params.k1.y) * r2 + params.k1.x) * r2);
        if (icdist < 0.0) {
            return vec2<f32>(0.0, 0.0);
        }
        let delta_x = 2.0 * params.k1.z * pos.x * pos.y + params.k1.w * (r2 + 2.0 * pos.x * pos.x)+ params.k3.x * r2 + params.k3.y * r2 * r2;
        let delta_y = params.k1.z * (r2 + 2.0 * pos.y * pos.y) + 2.0 * params.k1.w * pos.x * pos.y+ params.k3.z * r2 + params.k3.w * r2 * r2;
        pos = vec2<f32>(
            (start_pos.x - delta_x) * icdist,
            (start_pos.y - delta_y) * icdist
        );
    }

    return pos;
}

fn distort_point(x: f32, y: f32, z: f32) -> vec2<f32> {
    let pos = vec2<f32>(x, y) / z;
    let r2 = pos.x * pos.x + pos.y * pos.y;
    let r4 = r2 * r2;
    let r6 = r4 * r2;
    let a1 = 2.0 * pos.x * pos.y;
    let a2 = r2 + 2.0 * pos.x * pos.x;
    let a3 = r2 + 2.0 * pos.y * pos.y;
    let cdist = 1.0 + params.k1.x * r2 + params.k1.y * r4 + params.k2.x * r6;
    let icdist2 = 1.0 / (1.0 + params.k2.y * r2 + params.k2.z * r4 + params.k2.w * r6);

    return vec2<f32>(
        pos.x * cdist * icdist2 + params.k1.z * a1 + params.k1.w * a2 + params.k3.x * r2 + params.k3.y * r4,
        pos.y * cdist * icdist2 + params.k1.z * a3 + params.k1.w * a1 + params.k3.z * r2 + params.k3.w * r4
    );
}
"""
        )

    def id(self) -> str:
        return "opencv_standard"
