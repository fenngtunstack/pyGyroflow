"""Sony distortion model (extended angle-based polynomial).

Ported from Gyroflow's sony.rs (CPU) and the corresponding WGSL shader
(sony.wgsl).  Uses a 6-coefficient angle polynomial with post-scale.

Algorithm
---------
Forward (distort_point):
  1. Perspective divide:  pos = (x/z, y/z)
  2. r = |pos|;  theta = atan(r)   (incident angle)
  3. theta_d = k0*theta + k1*theta^2 + ... + k5*theta^6
  4. scale = theta_d / r
  5. result = pos * scale * post_scale

Inverse (undistort_point):
  1. Undo post-scale.
  2. theta_d = |pos|
  3. Newton's method to solve:  theta*(k0 + k1*theta + ... + k5*theta^5) = theta_d
     f'(theta) = k0 + 2*k1*theta + 3*k2*theta^2 + ... + 6*k5*theta^5
  4. scale = tan(theta) / theta_d
  5. result = pos * scale

Coefficient layout (KernelParams k arrays):
  k1[0..3] = k0, k1, k2, k3   (polynomial coefficients 0-3)
  k2[0..1] = k4, k5           (polynomial coefficients 4-5)
  k2[2..3] = post_scale_x, post_scale_y
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .base import DistortionModelBase

if TYPE_CHECKING:
    from pygyroflow.types.kernel_params import KernelParams

_HALF_PI = math.pi / 2.0


class SonyModel(DistortionModelBase):
    """Sony extended angle-based polynomial distortion model."""

    # -- undistort -------------------------------------------------------

    def undistort_point(
        self, x: float, y: float, params: "KernelParams"
    ) -> tuple[float, float] | None:
        """Inverse via Newton's method (up to 10 iterations).

        Solves for theta in:  theta * P(theta) = theta_d,
        where P(theta) = k0 + k1*theta + k2*theta^2 + ... + k5*theta^5.
        Starting guess theta=0, iterates until |theta_fix| < 1e-6.

        Returns None if iteration does not converge or theta flips sign
        (physically impossible ray).
        """
        k0 = float(params.k1[0])
        k1 = float(params.k1[1])
        k2 = float(params.k1[2])
        k3 = float(params.k1[3])
        k4 = float(params.k2[0])
        k5 = float(params.k2[1])
        post_scale_x = float(params.k2[2])
        post_scale_y = float(params.k2[3])

        # Early exit: no distortion
        if k0 == 0.0 and k1 == 0.0 and k2 == 0.0 and k3 == 0.0:
            return (x, y)

        EPS = 1e-6

        # Undo post-scale to get back to sensor-space metres
        px = x / post_scale_x
        py = y / post_scale_y

        theta_d = math.sqrt(px * px + py * py)

        converged = False
        theta = theta_d
        scale = 0.0

        if abs(theta_d) > EPS:
            theta = 0.0  # initial guess
            for _ in range(10):
                theta2 = theta * theta
                theta3 = theta2 * theta
                theta4 = theta2 * theta2
                theta5 = theta2 * theta3

                k1_t1 = k1 * theta
                k2_t2 = k2 * theta2
                k3_t3 = k3 * theta3
                k4_t4 = k4 * theta4
                k5_t5 = k5 * theta5

                # Newton step:  theta_fix = f(theta) / f'(theta)
                # f(theta)  = theta * (k0 + k1*t + k2*t^2 + k3*t^3 + k4*t^4 + k5*t^5) - theta_d
                # f'(theta) = k0 + 2*k1*t + 3*k2*t^2 + 4*k3*t^3 + 5*k4*t^4 + 6*k5*t^5
                f_val = theta * (k0 + k1_t1 + k2_t2 + k3_t3 + k4_t4 + k5_t5) - theta_d
                f_deriv = k0 + 2.0 * k1_t1 + 3.0 * k2_t2 + 4.0 * k3_t3 + 5.0 * k4_t4 + 6.0 * k5_t5

                if abs(f_deriv) < 1e-12:
                    break  # derivative too small, cannot continue

                theta_fix = f_val / f_deriv
                theta = theta - theta_fix

                if abs(theta_fix) < EPS:
                    converged = True
                    break

            scale = math.tan(theta) / theta_d
        else:
            converged = True

        # Check for sign flip (physically impossible ray direction)
        theta_flipped = (theta_d < 0.0 and theta > 0.0) or (
            theta_d > 0.0 and theta < 0.0
        )

        if converged and not theta_flipped:
            return (px * scale, py * scale)
        return None

    # -- distort ---------------------------------------------------------

    def distort_point(
        self, x: float, y: float, z: float, params: "KernelParams"
    ) -> tuple[float, float]:
        """Forward: 3D ray -> distorted normalised 2D.

        Projects via perspective divide, computes incident angle theta,
        applies 6-coefficient angle polynomial, and post-scales.
        """
        k0 = float(params.k1[0])
        k1 = float(params.k1[1])
        k2 = float(params.k1[2])
        k3 = float(params.k1[3])
        k4 = float(params.k2[0])
        k5 = float(params.k2[1])
        post_scale_x = float(params.k2[2])
        post_scale_y = float(params.k2[3])

        # Perspective divide
        x = x / z
        y = y / z

        # Early exit: no distortion
        if k0 == 0.0 and k1 == 0.0 and k2 == 0.0 and k3 == 0.0:
            return (x, y)

        r = math.sqrt(x * x + y * y)
        theta = math.atan(r)

        theta2 = theta * theta
        theta3 = theta2 * theta
        theta4 = theta2 * theta2
        theta5 = theta2 * theta3
        theta6 = theta3 * theta3

        # Distorted angle:  theta_d = sum(k_i * theta^(i+1), i=0..5)
        theta_d = (
            theta * k0
            + theta2 * k1
            + theta3 * k2
            + theta4 * k3
            + theta5 * k4
            + theta6 * k5
        )

        scale = theta_d / r if r != 0.0 else 1.0

        return (x * scale * post_scale_x, y * scale * post_scale_y)

    # -- radial distortion limit -----------------------------------------

    def radial_distortion_limit(self, coeffs: list[float]) -> float | None:
        """Binary search on the distortion derivative d(theta_d)/d(theta).

        Finds the largest theta in [0, pi/2] where the derivative is positive
        (i.e. distortion is monotonically increasing).  Returns tan(theta_max)
        as the max valid normalised radius, or None if monotonic over the
        entire half-circle.
        """
        if len(coeffs) < 6:
            return None
        k = coeffs[:6]
        low = 0.0
        high = _HALF_PI
        tolerance = 1e-4

        while high - low > tolerance:
            mid = (low + high) / 2.0
            t = mid
            t2 = t * t
            t3 = t2 * t
            t4 = t2 * t2
            t5 = t2 * t3
            # Derivative of theta_d w.r.t. theta
            deriv = (
                k[0]
                + 2.0 * k[1] * t
                + 3.0 * k[2] * t2
                + 4.0 * k[3] * t3
                + 5.0 * k[4] * t4
                + 6.0 * k[5] * t5
            )
            if deriv > 0.0:
                low = mid
            else:
                high = mid

        theta_max = (low + high) / 2.0
        # If theta_max is close to pi/2 the model is valid everywhere
        if abs(theta_max - _HALF_PI) > 0.001:
            return math.tan(theta_max)
        return None

    # -- WGSL shader -----------------------------------------------------

    def wgsl_functions(self) -> str:
        """WGSL shader matching Gyroflow's sony.wgsl exactly."""
        return (
            """
fn undistort_point(pos_param: vec2<f32>) -> vec2<f32> {
    if (params.k1.x == 0.0 && params.k1.y == 0.0 && params.k1.z == 0.0 && params.k1.w == 0.0) { return pos_param; }

    let post_scale = vec2<f32>(params.k2.z, params.k2.w);
    var pos = pos_param / post_scale;

    // now pos is in meters from center of sensor

    let theta_d = length(pos);

    var converged = false;
    var theta = theta_d;

    var scale = 0.0;

    if (abs(theta_d) > 1e-6) {
        for (var i: i32 = 0; i < 10; i = i + 1) {
                let theta2 = theta*theta;
                let theta3 = theta2*theta;
                let theta4 = theta2*theta2;
                let theta5 = theta2*theta3;
                let k0  = params.k1.x;
                let k1_theta1 = params.k1.y * theta;
                let k2_theta2 = params.k1.z * theta2;
                let k3_theta3 = params.k1.w * theta3;
                let k4_theta4 = params.k2.x * theta4;
                let k5_theta5 = params.k2.y * theta5;
                // new_theta = theta - theta_fix, theta_fix = f0(theta) / f0'(theta)
                let theta_fix = (theta * (k0 + k1_theta1 + k2_theta2 + k3_theta3 + k4_theta4 + k5_theta5) - theta_d)
                                /
                                (k0 + 2.0 * k1_theta1 + 3.0 * k2_theta2 + 4.0 * k3_theta3 + 5.0 * k4_theta4 + 6.0 * k5_theta5);

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
    let theta3 = theta2*theta;
    let theta4 = theta2*theta2;
    let theta5 = theta2*theta3;
    let theta6 = theta3*theta3;

    let theta_d = theta  * params.k1.x
                + theta2 * params.k1.y
                + theta3 * params.k1.z
                + theta4 * params.k1.w
                + theta5 * params.k2.x
                + theta6 * params.k2.y;

    var scale: f32 = 1.0;
    if (r != 0.0) {
        scale = theta_d / r;
    }

    let post_scale = vec2<f32>(params.k2.z, params.k2.w);

    return pos * scale * post_scale;
}
"""
        )

    def id(self) -> str:
        return "sony"
