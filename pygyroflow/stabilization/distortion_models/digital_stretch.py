"""Digital Stretch distortion model (simple anisotropic scaling).

Ported from Gyroflow's digital_stretch.rs (CPU) and the corresponding WGSL
shader (inline in the Rust source).

Applies independent linear scale factors in x and y directions.  This is the
simplest possible distortion model -- pure linear scaling with no nonlinearity.

Algorithm
---------
Forward (distort_point):
  x_d = x * stretch_x
  y_d = y * stretch_y

Inverse (undistort_point):
  x = x_d / stretch_x
  y = y_d / stretch_y

Coefficient layout (KernelParams):
  digital_lens_params[0] = stretch_x
  digital_lens_params[1] = stretch_y
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import DistortionModelBase

if TYPE_CHECKING:
    from pygyroflow.types.kernel_params import KernelParams


class DigitalStretchModel(DistortionModelBase):
    """Digital stretch (anisotropic scaling) distortion model."""

    # -- undistort -------------------------------------------------------

    def undistort_point(
        self, x: float, y: float, params: "KernelParams"
    ) -> tuple[float, float] | None:
        """Inverse: divide by each axis scale factor.

        Always returns a result (linear model, no convergence issues).
        """
        sx = float(params.digital_lens_params[0])
        sy = float(params.digital_lens_params[1])
        if sx == 0.0 or sy == 0.0:
            return None
        return (x / sx, y / sy)

    # -- distort ---------------------------------------------------------

    def distort_point(
        self, x: float, y: float, z: float, params: "KernelParams"
    ) -> tuple[float, float]:
        """Forward: multiply by each axis scale factor.

        z parameter is unused (digital lens, not optical projection).
        """
        sx = float(params.digital_lens_params[0])
        sy = float(params.digital_lens_params[1])
        return (x * sx, y * sy)

    # -- radial distortion limit -----------------------------------------

    def radial_distortion_limit(self, coeffs: list[float]) -> float | None:
        """Not applicable -- linear scaling, not radial distortion."""
        return None

    # -- WGSL shader -----------------------------------------------------

    def wgsl_functions(self) -> str:
        """WGSL shader matching Gyroflow's inline WGSL exactly."""
        return (
            """
fn digital_undistort_point(uv: vec2<f32>) -> vec2<f32> {
    uv.x = uv.x / params.digital_lens_params.x;
    uv.y = uv.y / params.digital_lens_params.y;
    return uv;
}
fn digital_distort_point(uv: vec2<f32>) -> vec2<f32> {
    uv.x = uv.x * params.digital_lens_params.x;
    uv.y = uv.y * params.digital_lens_params.y;
    return uv;
}
"""
        )

    def id(self) -> str:
        return "digital_stretch"
