"""GoPro6 Superview digital lens distortion model.

Ported from Gyroflow's ``gopro6_superview.rs`` (CPU) and the WGSL shader
inlined in that file.

This is GoPro's *second* Superview stretch and it is **not** the same function
as :mod:`gopro_superview` — it is a different polynomial, and it has no
4:3 -> 16:9 aspect-ratio step. The two produce visibly different output, which
is why Gyroflow ships them as two selectable digital lenses (and two compiled
fragment shaders).

Algorithm
---------
The core transform ``superview(uv)`` operates on normalised coords
``[-0.5, 0.5]`` and is applied in place, x before y::

    x *= (1.0 - 0.48 * |x|) * 0.943396 * (1.0 + 0.157895 * |x|)
    y *= 0.943396 * (1.0 + 0.060000 * |2y|)

Note that the second factor for x reads the *already updated* x, and that this
is a single expression split over two statements: ``x`` is multiplied twice.

undistort (Superview -> Wide):
  1. Normalise pixel to [-0.5, 0.5] against the *output* dimensions
  2. Apply ``superview``
  3. Back to pixels

distort (Wide -> Superview):
  1. Normalise pixel to [-0.5, 0.5] against the *input* dimensions
  2. Fixed-point iteration (up to 12 steps) to invert ``superview``
  3. Back to pixels

The iteration has no divergence guard here because upstream has none: the
substitution converges over the coordinate range a GoPro frame actually
covers, and where it would not, upstream's answer is the diverged one. The
sibling :mod:`gopro_superview` port carries a guard its upstream does not —
that is a pre-existing deviation, not something to copy into this model.

Reference: https://github.com/gyroflow/gyroflow/issues/43
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import DistortionModelBase

if TYPE_CHECKING:
    from pygyroflow.types.kernel_params import KernelParams


def _superview(uv: tuple[float, float]) -> tuple[float, float]:
    """Core GoPro6 Superview polynomial on normalised coords [-0.5, 0.5].

    Returns the mapped coordinates (not pixel coords). The x factors are two
    sequential multiplies, so the second one sees the first one's result.
    """
    x = uv[0]
    y = uv[1]
    x = x * (1.0 - 0.48 * abs(x))
    x = x * 0.943396 * (1.0 + 0.157895 * abs(x))
    y = y * 0.943396 * (1.0 + 0.060000 * abs(y * 2.0))
    return x, y


class GoPro6SuperviewModel(DistortionModelBase):
    """GoPro6 Superview digital lens distortion model."""

    # -- undistort -------------------------------------------------------

    def undistort_point(
        self, x: float, y: float, params: "KernelParams"
    ) -> tuple[float, float] | None:
        """Superview pixel -> Wide pixel.

        Never returns None: unlike the radial models there is no iteration
        here to fail, which is also why upstream declares this ``-> Option``
        and always returns ``Some``.
        """
        out_w = float(params.output_width)
        out_h = float(params.output_height)

        nx = (x / out_w) - 0.5
        ny = (y / out_h) - 0.5

        nx, ny = _superview((nx, ny))

        return ((nx + 0.5) * out_w, (ny + 0.5) * out_h)

    # -- distort ---------------------------------------------------------

    def distort_point(
        self, x: float, y: float, z: float, params: "KernelParams"
    ) -> tuple[float, float]:
        """Wide pixel -> Superview pixel, by fixed-point inversion.

        12 iterations of ``p -= superview(p) - target``, which converges to
        the pre-image of ``superview``. Stops early on a 1e-6 match.
        """
        size_w = float(params.width)
        size_h = float(params.height)

        nx = (x / size_w) - 0.5
        ny = (y / size_h) - 0.5

        px, py = nx, ny
        for _ in range(12):
            dp_x, dp_y = _superview((px, py))
            diff_x = dp_x - nx
            diff_y = dp_y - ny
            if abs(diff_x) < 1e-6 and abs(diff_y) < 1e-6:
                break
            px -= diff_x
            py -= diff_y

        return ((px + 0.5) * size_w, (py + 0.5) * size_h)

    def distort_points(self, xs, ys, zs, params):
        """Vectorized Wide -> Superview via masked fixed-point iteration.

        Same 12-step inversion as the scalar version, with converged points
        frozen out of further updates. Divergence is *not* guarded, matching
        the scalar path.
        """
        import numpy as np

        size_w = float(params.width)
        size_h = float(params.height)

        nx = (xs / size_w) - 0.5
        ny = (ys / size_h) - 0.5

        px = nx.copy()
        py = ny.copy()
        active = np.ones(nx.shape, dtype=bool)

        for _ in range(12):
            if not active.any():
                break
            dp_x = px * (1.0 - 0.48 * np.abs(px))
            dp_x = dp_x * 0.943396 * (1.0 + 0.157895 * np.abs(dp_x))
            dp_y = py * 0.943396 * (1.0 + 0.060000 * np.abs(py * 2.0))

            diff_x = dp_x - nx
            diff_y = dp_y - ny

            # Converged: freeze without applying the update.
            active &= ~((np.abs(diff_x) < 1e-6) & (np.abs(diff_y) < 1e-6))
            if not active.any():
                break

            px = np.where(active, px - diff_x, px)
            py = np.where(active, py - diff_y, py)

        return (px + 0.5) * size_w, (py + 0.5) * size_h

    # -- radial distortion limit -----------------------------------------

    def radial_distortion_limit(self, coeffs: list[float]) -> float | None:
        """Not applicable -- digital stretch, not radial distortion."""
        return None

    # -- WGSL shader -----------------------------------------------------

    def wgsl_functions(self) -> str:
        """WGSL shader matching Gyroflow's inline WGSL exactly."""
        return """
fn superview(_uv: vec2<f32>) -> vec2<f32> {
    var uv = _uv;
    uv.x *= 1.0 - 0.48 * abs(uv.x);
    uv.x *= 0.943396 * (1.0 + 0.157895 * abs(uv.x));
    uv.y *= 0.943396 * (1.0 + 0.060000 * abs(uv.y * 2.0));
    return uv;
}
fn digital_undistort_point(_uv: vec2<f32>) -> vec2<f32> {
    let out_c2 = vec2<f32>(f32(params.output_width), f32(params.output_height));
    var uv = _uv;
    uv = (uv / out_c2) - 0.5;

    uv = superview(uv);

    uv = (uv + 0.5) * out_c2;

    return uv;
}
fn digital_distort_point(_uv: vec2<f32>) -> vec2<f32> {
    let size = vec2<f32>(f32(params.width), f32(params.height));
    var uv = _uv;
    uv = (uv / size) - 0.5;

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

    def id(self) -> str:
        return "gopro6_superview"
