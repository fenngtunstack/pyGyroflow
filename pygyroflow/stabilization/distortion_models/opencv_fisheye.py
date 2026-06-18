"""OpenCV Fisheye (Kannala-Brandt) distortion model.

Ported from Gyroflow's opencv_fisheye.rs and the corresponding WGSL shader.
Uses 4 coefficients k1..k4, modelling distortion as a polynomial in the
incidence angle theta rather than the radius directly.

Forward:  theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
Inverse:  Newton-Raphson iteration to solve for theta given theta_d.

Coefficient layout (KernelParams):
  k1[0..3] = k1, k2, k3, k4
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .base import DistortionModelBase

if TYPE_CHECKING:
    from pygyroflow.types.kernel_params import KernelParams

_HALF_PI = math.pi / 2.0


def _get_k(params: "KernelParams") -> list[float]:
    """Extract the 4 fisheye coefficients from KernelParams.k1."""
    return [float(params.k1[i]) for i in range(4)]


class OpenCVFisheyeModel(DistortionModelBase):
    """OpenCV Fisheye (Kannala-Brandt) distortion model."""

    # -- undistort -------------------------------------------------------

    def undistort_point(
        self, x: float, y: float, params: "KernelParams"
    ) -> tuple[float, float] | None:
        k = _get_k(params)
        if k[0] == 0.0 and k[1] == 0.0 and k[2] == 0.0 and k[3] == 0.0:
            return (x, y)

        EPS = 1e-6

        theta_d = math.sqrt(x * x + y * y)
        # Clamp to [-pi, pi] for >180 deg FOV
        theta_d = max(-math.pi, min(math.pi, theta_d))

        converged = False
        theta = theta_d
        scale = 0.0

        if abs(theta_d) > EPS:
            theta = 0.0
            for _ in range(10):
                theta2 = theta * theta
                theta4 = theta2 * theta2
                theta6 = theta4 * theta2
                theta8 = theta6 * theta2

                k0_t2 = k[0] * theta2
                k1_t4 = k[1] * theta4
                k2_t6 = k[2] * theta6
                k3_t8 = k[3] * theta8

                theta_fix = (
                    theta * (1.0 + k0_t2 + k1_t4 + k2_t6 + k3_t8) - theta_d
                ) / (1.0 + 3.0 * k0_t2 + 5.0 * k1_t4 + 7.0 * k2_t6 + 9.0 * k3_t8)

                # Clamp step to prevent divergence
                theta_fix = max(-0.9, min(0.9, theta_fix))
                theta = theta - theta_fix

                if abs(theta_fix) < EPS:
                    converged = True
                    break

            scale = math.tan(theta) / theta_d
        else:
            converged = True

        theta_flipped = (theta_d < 0.0 and theta > 0.0) or (
            theta_d > 0.0 and theta < 0.0
        )

        if converged and not theta_flipped:
            return (x * scale, y * scale)
        return None

    # -- distort ---------------------------------------------------------

    def distort_point(
        self, x: float, y: float, z: float, params: "KernelParams"
    ) -> tuple[float, float]:
        k = _get_k(params)
        x = x / z
        y = y / z
        if k[0] == 0.0 and k[1] == 0.0 and k[2] == 0.0 and k[3] == 0.0:
            return (x, y)

        r = math.sqrt(x * x + y * y)
        theta = math.atan(r)
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4

        theta_d = theta * (
            1.0 + k[0] * theta2 + k[1] * theta4 + k[2] * theta6 + k[3] * theta8
        )
        scale = theta_d / r if r != 0.0 else 1.0
        return (x * scale, y * scale)

    # -- radial distortion limit -----------------------------------------

    def radial_distortion_limit(self, coeffs: list[float]) -> float | None:
        if len(coeffs) < 4:
            return None
        k = coeffs[:4]
        low = 0.0
        high = _HALF_PI
        tolerance = 1e-4

        while high - low > tolerance:
            mid = (low + high) / 2.0
            t2 = mid * mid
            t4 = t2 * t2
            t6 = t4 * t2
            t8 = t6 * t2
            deriv = (
                1.0
                + 3.0 * k[0] * t2
                + 5.0 * k[1] * t4
                + 7.0 * k[2] * t6
                + 9.0 * k[3] * t8
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
            # SPDX-SnippetBegin
            # SPDX-License-Identifier: GPL-3.0-or-later
            # SPDX-SnippetCopyrightText: 2022 Adrian <adrian.eddy at gmail>
            """
fn undistort_point(pos: vec2<f32>) -> vec2<f32> {
    if (params.k1.x == 0.0 && params.k1.y == 0.0 && params.k1.z == 0.0 && params.k1.w == 0.0) { return pos; }
    let theta_d = min(max(length(pos), -1.5707963267948966), 1.5707963267948966); // PI/2

    var converged = false;
    var theta = theta_d;

    var scale = 0.0;

    if (abs(theta_d) > 1e-6) {
        for (var i: i32 = 0; i < 10; i = i + 1) {
            let theta2 = theta*theta;
            let theta4 = theta2*theta2;
            let theta6 = theta4*theta2;
            let theta8 = theta6*theta2;
            let k0_theta2 = params.k1.x * theta2;
            let k1_theta4 = params.k1.y * theta4;
            let k2_theta6 = params.k1.z * theta6;
            let k3_theta8 = params.k1.w * theta8;
            // new_theta = theta - theta_fix, theta_fix = f0(theta) / f0'(theta)
            let theta_fix = (theta * (1.0 + k0_theta2 + k1_theta4 + k2_theta6 + k3_theta8) - theta_d)
                            /
                            (1.0 + 3.0 * k0_theta2 + 5.0 * k1_theta4 + 7.0 * k2_theta6 + 9.0 * k3_theta8);

            theta -= theta_fix;
            if (abs(theta_fix) < 1e-6) {
                converged = true;
                break;
            }
        }

        scale = tan(theta) / theta_d;
    } else {
        converged = true;
    }
    let theta_flipped = (theta_d < 0.0 && theta > 0.0) || (theta_d > 0.0 && theta < 0.0);

    if (converged && !theta_flipped) {
        return pos * scale;
    }
    return vec2<f32>(0.0, 0.0);
}

fn distort_point(x: f32, y: f32, z: f32) -> vec2<f32> {
    let pos = vec2<f32>(x, y) / z;
    if (params.k1.x == 0.0 && params.k1.y == 0.0 && params.k1.z == 0.0 && params.k1.w == 0.0) { return pos; }
    let r = length(pos);

    let theta = atan(r);
    let theta2 = theta*theta;
    let theta4 = theta2*theta2;
    let theta6 = theta4*theta2;
    let theta8 = theta4*theta4;

    let theta_d = theta * (1.0 + dot(params.k1, vec4<f32>(theta2, theta4, theta6, theta8)));

    var scale: f32 = 1.0;
    if (r != 0.0) {
        scale = theta_d / r;
    }
    return pos * scale;
}
"""
            # SPDX-SnippetEnd
        )

    def id(self) -> str:
        return "opencv_fisheye"
