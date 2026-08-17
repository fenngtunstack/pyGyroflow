"""Base class for all lens distortion models.

Ported from Gyroflow's distortion_models module. Each model implements:
- undistort_point: distorted normalized (x,y) -> undistorted normalized (x,y)
- distort_point:   3D ray (x,y,z) -> distorted normalized 2D (x_d, y_d)
- radial_distortion_limit: max valid radius (binary search on derivative)
- wgsl_functions: WGSL shader code string for GPU-accelerated distortion

Vectorized batch variants (suffix ``_points``) operate on whole NumPy arrays
in one call; they are used by the CPU rendering path. Subclasses should
override them with closed-form NumPy implementations — the defaults here
fall back to the scalar per-point methods, which is correct but slow.
Points that fail to converge in ``undistort_points`` map to NaN so callers
can send them to the background fill.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

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

    # -- vectorized batch variants ----------------------------------------

    def distort_points(
        self,
        xs: NDArray[np.float64],
        ys: NDArray[np.float64],
        zs: NDArray[np.float64],
        params: "KernelParams",
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Vectorized forward distortion of many points at once.

        Args:
            xs, ys, zs: Arrays of 3D ray coordinates (zs used for the
                perspective divide; pass ones for pre-divided 2D input).
            params: Kernel parameters with this model's coefficients.

        Returns:
            (xd, yd) distorted normalized coordinates.

        Default implementation loops over the scalar ``distort_point``.
        Subclasses override with closed-form NumPy for performance.
        """
        xd = np.empty_like(xs)
        yd = np.empty_like(ys)
        flat_xs = xs.ravel()
        flat_ys = ys.ravel()
        flat_zs = zs.ravel()
        out_x = xd.ravel()
        out_y = yd.ravel()
        for i in range(flat_xs.size):
            out_x[i], out_y[i] = self.distort_point(
                float(flat_xs[i]), float(flat_ys[i]), float(flat_zs[i]), params
            )
        return xd, yd

    def undistort_points(
        self,
        xs: NDArray[np.float64],
        ys: NDArray[np.float64],
        params: "KernelParams",
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Vectorized inverse distortion of many points at once.

        Args:
            xs, ys: Arrays of distorted normalized coordinates.
            params: Kernel parameters with this model's coefficients.

        Returns:
            (xu, yu) undistorted normalized coordinates. Non-converged
            points are set to NaN (callers should treat them as invalid,
            mirroring the scalar ``undistort_point`` -> None contract).

        Default implementation loops over the scalar ``undistort_point``.
        Subclasses override with closed-form NumPy for performance.
        """
        xu = np.full_like(xs, np.nan)
        yu = np.full_like(ys, np.nan)
        flat_xs = xs.ravel()
        flat_ys = ys.ravel()
        out_x = xu.ravel()
        out_y = yu.ravel()
        for i in range(flat_xs.size):
            pt = self.undistort_point(float(flat_xs[i]), float(flat_ys[i]), params)
            if pt is not None:
                out_x[i], out_y[i] = pt
        return xu, yu
