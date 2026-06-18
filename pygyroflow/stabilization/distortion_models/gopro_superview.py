"""GoPro Superview digital lens distortion model.

Ported from Gyroflow's gopro_superview.rs (CPU) and the corresponding WGSL
shader (inline in the Rust source).

Applies a nonlinear polynomial stretch that converts 4:3 sensor output to
16:9 Superview output.  All coefficients are hardcoded (obtained by reverse-
engineering GoPro's processing).

Algorithm
---------
The core transform is _superview(uv) operating on normalised coords [-0.5, 0.5]:

  x' = x * (1.2100393 + x^2 * (-1.2758402 + x^2 * 1.7751845))
  y' = y * (0.9364505 + (0.4465308 - 0.7683315*y^2)*y^2
            + (-0.3574087 + 1.1584653*y^2 + 0.3529348*x^2)*x^2)

undistort (Superview -> Wide):
  1. Normalise pixel to [-0.5, 0.5]
  2. Apply _superview (forward transform maps Superview -> Wide)
  3. Divide x by 4/3 to undo the aspect-ratio stretch
  4. Back to pixels

distort (Wide -> Superview):
  1. Normalise pixel to [-0.5, 0.5]
  2. Multiply x by 4/3 for aspect-ratio stretch
  3. Fixed-point iteration (up to 12 steps) to invert _superview
  4. Back to pixels

NOTE: Gyroflow also has a separate SPIR-V implementation (stabilize_spirv)
that uses piecewise square-root functions instead of polynomials.  Both
produce similar results; the polynomial version here is the CPU reference.

Reference: https://github.com/gyroflow/gyroflow/issues/43
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .base import DistortionModelBase

if TYPE_CHECKING:
    from pygyroflow.types.kernel_params import KernelParams

# Aspect ratio stretch: 4:3 sensor -> 16:9 output
_ASPECT_RATIO = 4.0 / 3.0  # 1.333333333


def _superview(uv: tuple[float, float]) -> tuple[float, float]:
    """Core Superview polynomial transform on normalised coords [-0.5, 0.5].

    Returns the mapped coordinates (not pixel coords).
    """
    x2 = uv[0] * uv[0]
    y2 = uv[1] * uv[1]
    return (
        uv[0] * (1.2100393 + x2 * (-1.2758402 + x2 * 1.7751845)),
        uv[1]
        * (
            0.9364505
            + (0.4465308 - 0.7683315 * y2) * y2
            + (-0.3574087 + 1.1584653 * y2 + 0.3529348 * x2) * x2
        ),
    )


class GoProSuperviewModel(DistortionModelBase):
    """GoPro Superview digital lens distortion model."""

    # -- undistort -------------------------------------------------------

    def undistort_point(
        self, x: float, y: float, params: "KernelParams"
    ) -> tuple[float, float] | None:
        """Superview pixel -> Wide pixel.

        Applies the forward _superview() transform (which maps from the
        Superview space back to the linear Wide space), then undoes the
        4:3 -> 16:9 aspect ratio stretch.
        """
        out_w = float(params.output_width)
        out_h = float(params.output_height)

        # Normalise to [-0.5, 0.5]
        nx = (x / out_w) - 0.5
        ny = (y / out_h) - 0.5

        # Forward transform: Superview -> Wide
        nx, ny = _superview((nx, ny))

        # Undo 4:3 -> 16:9 stretch
        nx = nx / _ASPECT_RATIO

        # Back to pixels
        return ((nx + 0.5) * out_w, (ny + 0.5) * out_h)

    # -- distort ---------------------------------------------------------

    def distort_point(
        self, x: float, y: float, z: float, params: "KernelParams"
    ) -> tuple[float, float]:
        """Wide pixel -> Superview pixel.

        Applies the 4:3 -> 16:9 aspect ratio stretch, then uses fixed-point
        iteration (up to 12 steps) to invert _superview().  Guarded against
        divergence by clamping the iteration variables.
        """
        size_w = float(params.width)
        size_h = float(params.height)

        # Normalise to [-0.5, 0.5]
        nx = (x / size_w) - 0.5
        ny = (y / size_h) - 0.5

        # Apply 4:3 -> 16:9 stretch
        nx = nx * _ASPECT_RATIO

        # Fixed-point iteration to invert _superview
        px, py = nx, ny
        for _ in range(12):
            dp_x, dp_y = _superview((px, py))
            diff_x = dp_x - nx
            diff_y = dp_y - ny
            if abs(diff_x) < 1e-6 and abs(diff_y) < 1e-6:
                break
            px -= diff_x
            py -= diff_y
            # Guard against divergence
            if abs(px) > 2.0 or abs(py) > 2.0:
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
fn superview(uv: vec2<f32>) -> vec2<f32> {
    let x2 = uv.x * uv.x;
    let y2 = uv.y * uv.y;
    return vec2<f32>(
        uv.x * (1.2100393 + x2 * (-1.2758402 + x2 * 1.7751845)),
        uv.y * (0.9364505 + (0.4465308 - 0.7683315 * y2) * y2 + (-0.3574087 + 1.1584653 * y2 + 0.3529348 * x2) * x2)
    );
}
fn digital_undistort_point(_uv: vec2<f32>) -> vec2<f32> {
    let out_c2 = vec2<f32>(f32(params.output_width), f32(params.output_height));
    var uv = _uv;
    uv = (uv / out_c2) - 0.5;

    uv = superview(uv);

    uv.x = uv.x / 1.333333333;
    uv = (uv + 0.5) * out_c2;

    return uv;
}
fn digital_distort_point(_uv: vec2<f32>) -> vec2<f32> {
    let size = vec2<f32>(f32(params.width), f32(params.height));
    var uv = _uv;
    uv = (uv / size) - 0.5;

    uv.x = uv.x * 1.333333333;

    var P = uv;
    for (var i: i32 = 0; i < 12; i = i + 1) {
        let diff = superview(P) - uv;
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
        return "gopro_superview"
