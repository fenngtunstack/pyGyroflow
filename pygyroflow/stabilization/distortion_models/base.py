"""Base class for all lens distortion models.

Ported from Gyroflow's distortion_models module. Each model implements:
- undistort_point: distorted normalized (x,y) -> undistorted normalized (x,y)
- distort_point:   3D ray (x,y,z) -> distorted normalized 2D (x_d, y_d)
- radial_distortion_limit: max valid radius (binary search on derivative)
- wgsl_functions: WGSL shader code string for GPU-accelerated distortion
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pygyroflow.types.kernel_params import KernelParams


class DistortionModelBase(ABC):
    """Abstract base for lens distortion models."""

    @abstractmethod
    def undistort_point(
        self, x: float, y: float, params: "KernelParams"
    ) -> tuple[float, float] | None:
        """Undistort a normalized point (x, y) -> (x', y').

        Input and output are in normalized image coordinates
        (centered at optical center, scaled by focal length).

        Returns None if the iteration does not converge or the point
        is outside the valid distortion range.
        """
        ...

    @abstractmethod
    def distort_point(
        self, x: float, y: float, z: float, params: "KernelParams"
    ) -> tuple[float, float]:
        """Distort a 3D point (x, y, z) -> normalized 2D (x_d, y_d).

        First projects via perspective division (x/z, y/z), then
        applies the forward distortion model.
        """
        ...

    @abstractmethod
    def radial_distortion_limit(self, coeffs: list[float]) -> float | None:
        """Get the maximum radius where distortion is valid.

        Uses binary search on the distortion derivative to find where
        it becomes non-positive.  Returns None = no limit (monotonic
        over the entire half-circle).
        """
        ...

    @abstractmethod
    def wgsl_functions(self) -> str:
        """Return WGSL shader code string for this distortion model.

        Must define these WGSL functions:
        - fn distort_point(x: f32, y: f32, z: f32) -> vec2<f32>
        - fn undistort_point(uv: vec2<f32>) -> vec2<f32>
        """
        ...

    @abstractmethod
    def id(self) -> str:
        """Return the distortion model identifier string."""
        ...
