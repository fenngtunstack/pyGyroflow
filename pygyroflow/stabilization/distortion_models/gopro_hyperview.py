"""GoPro Hyperview digital lens distortion model.

Ported from Gyroflow's gopro_hyperview.rs (CPU) and the corresponding WGSL
shader (inline in the Rust source).

Applies a high-order nonlinear polynomial stretch that converts 8:7 sensor
output to 16:9 Hyperview output.  All coefficients are hardcoded (obtained
by reverse-engineering GoPro's processing).

Algorithm
---------
The core transform is _hyperview(uv) on normalised coords [-0.5, 0.5]:

  x' = x * P(x^2) + y^2 * (-0.1086027)
    where P(t) = 1.5805143
                + t*(-8.1668825
                + t*(74.5198746
                + t*(-451.5002441
                + t*(1551.2922363
                + t*(-2735.5422363 + t*1923.1572266)))))

  y' = y * (1.0238225 + y^2*(-0.1025671) + x^2*(-0.2639930 + x^2*0.2979266))

undistort (Hyperview -> Wide):
  1. Normalise pixel to [-0.5, 0.5]
  2. Apply _hyperview (forward transform maps Hyperview -> Wide)
  3. Divide x by 14/9 to undo the aspect-ratio stretch
  4. Back to pixels

distort (Wide -> Hyperview):
  1. Normalise pixel to [-0.5, 0.5]
  2. Multiply x by 14/9 for aspect-ratio stretch
  3. Fixed-point iteration (up to 20 steps) to invert _hyperview
  4. Back to pixels

NOTE: The 7th-order polynomial can cause the fixed-point iteration to
diverge at extreme image edges.  The implementation guards against this
by clamping the iteration variables.  Gyroflow's SPIR-V version uses a
different piecewise square-root approach that avoids iteration entirely;
the polynomial version here is the CPU reference.

Reference: https://github.com/gyroflow/gyroflow/issues/43
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .base import DistortionModelBase

if TYPE_CHECKING:
    from pygyroflow.types.kernel_params import KernelParams

# Aspect ratio stretch: 8:7 sensor -> 16:9 output
# (16/9) / (8/7) = 112/72 = 14/9
_ASPECT_RATIO = 14.0 / 9.0  # 1.555555555

# Maximum iterations for fixed-point inversion.
# More than GoPro Superview needs because the 7th-order polynomial
# has stronger nonlinearity.
_MAX_ITER = 20

# Guard: if iteration variables exceed this normalised range, reset.
_DIVERGENCE_LIMIT = 2.0


def _hyperview(uv: tuple[float, float]) -> tuple[float, float]:
    """Core Hyperview polynomial transform on normalised coords [-0.5, 0.5].

    Returns the mapped coordinates (not pixel coords).
    """
    x2 = uv[0] * uv[0]
    y2 = uv[1] * uv[1]
    return (
        uv[0]
        * (
            1.5805143
            + x2
            * (
                -8.1668825
                + x2
                * (
                    74.5198746
                    + x2
                    * (
                        -451.5002441
                        + x2
                        * (1551.2922363 + x2 * (-2735.5422363 + x2 * 1923.1572266))
                    )
                )
            )
        )
        + y2 * -0.1086027,
        uv[1] * (1.0238225 + y2 * -0.1025671 + x2 * (-0.2639930 + x2 * 0.2979266)),
    )


class GoProHyperviewModel(DistortionModelBase):
    """GoPro Hyperview digital lens distortion model."""

    # -- undistort -------------------------------------------------------

    def undistort_point(
        self, x: float, y: float, params: "KernelParams"
    ) -> tuple[float, float] | None:
        """Hyperview pixel -> Wide pixel.

        Applies the forward _hyperview() transform (which maps from the
        Hyperview space back to the linear Wide space), then undoes the
        8:7 -> 16:9 aspect ratio stretch.
        """
        out_w = float(params.output_width)
        out_h = float(params.output_height)

        nx = (x / out_w) - 0.5
        ny = (y / out_h) - 0.5

        # Forward transform: Hyperview -> Wide
        nx, ny = _hyperview((nx, ny))

        # Undo 8:7 -> 16:9 stretch
        nx = nx / _ASPECT_RATIO

        return ((nx + 0.5) * out_w, (ny + 0.5) * out_h)

    # -- distort ---------------------------------------------------------

    def distort_point(
        self, x: float, y: float, z: float, params: "KernelParams"
    ) -> tuple[float, float]:
        """Wide pixel -> Hyperview pixel.

        Applies the 8:7 -> 16:9 aspect ratio stretch, then uses fixed-point
        iteration to invert _hyperview().  The 7th-order polynomial can cause
        divergence at extreme image edges; iteration is guarded by clamping.
        """
        size_w = float(params.width)
        size_h = float(params.height)

        nx = (x / size_w) - 0.5
        ny = (y / size_h) - 0.5

        # Apply 8:7 -> 16:9 stretch
        nx = nx * _ASPECT_RATIO

        # Fixed-point iteration to invert _hyperview
        px, py = nx, ny
        for _ in range(_MAX_ITER):
            dp_x, dp_y = _hyperview((px, py))
            diff_x = dp_x - nx
            diff_y = dp_y - ny
            if abs(diff_x) < 1e-6 and abs(diff_y) < 1e-6:
                break
            px -= diff_x
            py -= diff_y
            # Guard against divergence: reset if overshooting
            if abs(px) > _DIVERGENCE_LIMIT or abs(py) > _DIVERGENCE_LIMIT:
                px = nx
                py = ny
                break

        return ((px + 0.5) * size_w, (py + 0.5) * size_h)

    # -- radial distortion limit -----------------------------------------

    def radial_distortion_limit(self, coeffs: list[float]) -> float | None:
        """Not applicable -- digital stretch, not radial distortion."""
        return None

    # -- WGSL shader -----------------------------------------------------

    def wgsl_functions(self) -> str:
        """WGSL shader matching Gyroflow's inline WGSL exactly."""
        return (
            """
fn hyperview(uv: vec2<f32>) -> vec2<f32> {
    let x2 = uv.x * uv.x;
    let y2 = uv.y * uv.y;
    return vec2<f32>(
        uv.x * (1.5805143 + x2 * (-8.1668825 + x2 * (74.5198746 + x2 * (-451.5002441 + x2 * (1551.2922363 + x2 * (-2735.5422363 + x2 * 1923.1572266))))) + y2 * -0.1086027),
        uv.y * (1.0238225 + y2 * -0.1025671 + x2 * (-0.2639930 + x2 * 0.2979266))
    );
}
fn digital_undistort_point(_uv: vec2<f32>) -> vec2<f32> {
    let out_c2 = vec2<f32>(f32(params.output_width), f32(params.output_height));
    var uv = _uv;
    uv = (uv / out_c2) - 0.5;

    uv = hyperview(uv);

    uv.x = uv.x / 1.555555555;
    uv = (uv + 0.5) * out_c2;

    return uv;
}
fn digital_distort_point(_uv: vec2<f32>) -> vec2<f32> {
    let size = vec2<f32>(f32(params.width), f32(params.height));
    var uv = _uv;
    uv = (uv / size) - 0.5;

    uv.x = uv.x * 1.555555555;

    var P = uv;
    for (var i: i32 = 0; i < 20; i = i + 1) {
        let diff = hyperview(P) - uv;
        if (abs(diff.x) < 1e-6 && abs(diff.y) < 1e-6) {
            break;
        }
        P -= diff;
    }

    uv = (P + 0.5) * size;

    return uv;
}
"""
        )

    def id(self) -> str:
        return "gopro_hyperview"
