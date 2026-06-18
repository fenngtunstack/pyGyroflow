"""Insta360 distortion model (Unified Camera Model + Brown-Conrady).

Ported from Gyroflow's insta360.rs (CPU) and the corresponding WGSL shader
(installed at stabilization/distortion_models/insta360.wgsl).

Algorithm
---------
Forward (distort_point):
  1. Normalize 3D ray to unit length.
  2. UCM projection:  x_proj = (X/|P|) / (Z/|P| + xi)
  3. Brown-Conrady radial + tangential distortion:
     r^2 = x^2 + y^2
     x_d = x*(1 + k1*r^2 + k2*r^4 + k3*r^6) + 2*p1*x*y + p2*(r^2 + 2*x^2)
     y_d = y*(1 + k1*r^2 + k2*r^4 + k3*r^6) + 2*p2*x*y + p1*(r^2 + 2*y^2)

Inverse (undistort_point):
  Fixed-point iteration (up to 200 steps), starting from the distorted
  point as initial guess.  Converges to within 1e-6 tolerance.

Coefficient layout (KernelParams k arrays, mapped as in Gyroflow WGSL):
  k1[0] = k1  (radial r^2)      k1[1] = k2  (radial r^4)
  k1[2] = k3  (radial r^6)      k1[3] = p1  (tangential y-tilt)
  k2[0] = p2  (tangential x-tilt) k2[1] = xi (UCM mirror parameter)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .base import DistortionModelBase

if TYPE_CHECKING:
    from pygyroflow.types.kernel_params import KernelParams


class Insta360Model(DistortionModelBase):
    """Insta360 distortion model (UCM + Brown-Conrady)."""

    # -- undistort -------------------------------------------------------

    def undistort_point(
        self, x: float, y: float, params: "KernelParams"
    ) -> tuple[float, float] | None:
        """Inverse via fixed-point iteration (up to 200 steps).

        Starts from the distorted coordinate as initial guess and iteratively
        subtracts the forward-model error until convergence or max iterations.
        Always returns a result (matching Gyroflow behaviour).
        """
        px, py = x, y
        for _ in range(200):
            dp_x, dp_y = self.distort_point(px, py, 1.0, params)
            diff_x = dp_x - x
            diff_y = dp_y - y
            if abs(diff_x) < 1e-6 and abs(diff_y) < 1e-6:
                break
            px -= diff_x
            py -= diff_y
        return (px, py)

    # -- distort ---------------------------------------------------------

    def distort_point(
        self, x: float, y: float, z: float, params: "KernelParams"
    ) -> tuple[float, float]:
        """Forward: 3D ray -> distorted normalised 2D.

        Step 1 -- Unified Camera Model projection with mirror parameter xi.
        Step 2 -- Brown-Conrady radial (k1,k2,k3) + tangential (p1,p2).
        """
        k1 = float(params.k1[0])
        k2 = float(params.k1[1])
        k3 = float(params.k1[2])
        p1 = float(params.k1[3])
        p2 = float(params.k2[0])
        xi = float(params.k2[1])

        length = math.sqrt(x * x + y * y + z * z)

        # UCM projection:  normalise to unit sphere, then apply mirror param
        # When xi=0 this degenerates to standard pinhole (x/z, y/z).
        proj_x = (x / length) / ((z / length) + xi)
        proj_y = (y / length) / ((z / length) + xi)

        r2 = proj_x * proj_x + proj_y * proj_y
        r4 = r2 * r2
        r6 = r4 * r2

        # Radial factor common to both axes
        radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

        return (
            proj_x * radial + 2.0 * p1 * proj_x * proj_y + p2 * (r2 + 2.0 * proj_x * proj_x),
            proj_y * radial + 2.0 * p2 * proj_x * proj_y + p1 * (r2 + 2.0 * proj_y * proj_y),
        )

    # -- radial distortion limit -----------------------------------------

    def radial_distortion_limit(self, coeffs: list[float]) -> float | None:
        """UCM + Brown-Conrady has no closed-form derivative.

        The combined model (mirror projection + radial + tangential) is not
        easily expressible as a single-variable function of radius, so no
        analytical limit can be computed.  Returns None.
        """
        return None

    # -- WGSL shader -----------------------------------------------------

    def wgsl_functions(self) -> str:
        """WGSL shader matching Gyroflow's insta360.wgsl exactly."""
        return (
            """
fn distort_point(px: f32, py: f32, pz: f32) -> vec2<f32> {
    let k1 = params.k1.x;
    let k2 = params.k1.y;
    let k3 = params.k1.z;
    let p1 = params.k1.w;

    let p2 = params.k2.x;
    let xi = params.k2.y;

    var p = vec3<f32>(px, py, pz);
    p /= length(p);

    let x = p.x / (p.z + xi);
    let y = p.y / (p.z + xi);

    let r2 = x*x + y*y;
    let r4 = r2 * r2;
    let r6 = r4 * r2;

    return vec2<f32>(
        x * (1.0 + k1*r2 + k2*r4 + k3*r6) + 2.0*p1*x*y + p2*(r2 + 2.0*x*x),
        y * (1.0 + k1*r2 + k2*r4 + k3*r6) + 2.0*p2*x*y + p1*(r2 + 2.0*y*y)
    );
}

fn undistort_point(p: vec2<f32>) -> vec2<f32> {
    var pp = p;

    for (var i: i32 = 0; i < 200; i = i + 1) {
        let diff = distort_point(pp.x, pp.y, 1.0) - p;
        if (abs(diff.x) < 1e-6 && abs(diff.y) < 1e-6) {
            break;
        }
        pp -= diff;
    }

    return pp;
}
"""
        )

    def id(self) -> str:
        return "insta360"
