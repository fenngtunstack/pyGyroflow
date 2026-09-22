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
  3. Fixed-point iteration (at most 12 steps) to invert _hyperview
  4. Back to pixels

NOTE: that inversion is a substitution, and the 7th-order polynomial makes it
a contraction over only part of the frame. Where it diverges, upstream returns
whatever the 12th step produced — often NaN. `_MAX_ITER` below carries the
measurement. The GPU uses the same substitution.
Note also that the digital lens runs *after* the optical one, so this maps
camera pixels rather than rays: a point that leaves the frame here is not an
error.

Reference: https://github.com/gyroflow/gyroflow/issues/43
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .base import DistortionModelBase

if TYPE_CHECKING:
    from pygyroflow.types.kernel_params import KernelParams

# Aspect ratio stretch: 8:7 sensor -> 16:9 output
# (16/9) / (8/7) = 112/72 = 14/9
_ASPECT_RATIO = 14.0 / 9.0  # 1.555555555

# Fixed-point iteration count, from upstream (gopro_hyperview.rs:44 —
# `for _ in 0..12`) and the same number its WGSL uses.
#
# It is not enough. Measured over a whole 1920x1080 frame at 8 px spacing, the
# substitution converges for only about 47% of the points; the rest run all 12
# steps and stop wherever they got to, and about 9% come out NaN (the iterate
# overflows f32 and subtracting two infinities propagates). Upstream behaves
# the same way — this is not a porting artifact — and the reference fixture
# (tests/golden/undistort_points.json, case `digital_lens_hyperview`) records
# it.
#
# Reproducing that is deliberate. The port exists to match Gyroflow, and a
# convergent rewrite here (Newton, bisection, or simply more steps) would make
# every HyperView comparison disagree for a reason a reader could not see in
# this file. If the count is ever raised, do it as a *recorded* deviation with
# the fixture regenerated and the gap analysis updated — not as a quiet local
# improvement.
_MAX_ITER = 12


def _hyperview(uv: tuple[float, float]) -> tuple[float, float]:
    """Core Hyperview polynomial transform on normalised coords [-0.5, 0.5].

    Returns the mapped coordinates (not pixel coords).

    The ``float`` coercions are load-bearing: the inversion below feeds this
    values that overflow, and on a NumPy scalar that raises a RuntimeWarning on
    every step. As Python floats the same arithmetic returns ``inf`` and then
    ``nan`` silently, which is what the inversion wants — the NaN is a result
    here, not an accident. The coercion is exact (widening within the same
    format).
    """
    x = float(uv[0])
    y = float(uv[1])
    x2 = x * x
    y2 = y * y
    return (
        x
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
            # Inside the multiplication by x, not after it. Written where the
            # eye expects an additive term, this is `x * (... + y2*c)` and not
            # `x * (...) + y2*c`; the two differ by a factor of x, so they agree
            # on the vertical centreline and nowhere else. The port had it
            # outside until the reference fixture caught it — on the
            # x == width/2 column upstream returns the column unchanged and the
            # port drifted off it. The WGSL below carries it inside, which is
            # what gave the discrepancy away.
            + y2 * -0.1086027
        ),
        y * (1.0238225 + y2 * -0.1025671 + x2 * (-0.2639930 + x2 * 0.2979266)),
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

        Applies the 8:7 -> 16:9 aspect ratio stretch, then inverts
        ``_hyperview`` by substitution. No divergence guard: upstream has none,
        and the answer it produces where the substitution runs away (frequently
        NaN — see ``_MAX_ITER``) is the one the reference fixture pins.
        """
        size_w = float(params.width)
        size_h = float(params.height)

        # Python floats, not NumPy scalars: the subtraction below can be
        # inf - inf, which is a silent NaN for a float and a RuntimeWarning for
        # a NumPy scalar. See `_hyperview`.
        nx = (float(x) / size_w) - 0.5
        ny = (float(y) / size_h) - 0.5

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

        return ((px + 0.5) * size_w, (py + 0.5) * size_h)

    def distort_points(self, xs, ys, zs, params):
        """Vectorized Wide -> Hyperview via masked fixed-point iteration.

        The scalar loop with a convergence mask: a point whose difference
        falls under the epsilon is frozen so later steps cannot move it, which
        is exactly what the scalar ``break`` does for the whole array at once
        only if every point has converged. No divergence guard, matching the
        scalar version and upstream.

        Needs ``np.errstate``: the substitution overflows f32 on the way out of
        the frame, and the resulting inf-inf is the NaN upstream also
        produces. Warning about it once per call would drown the log.
        """
        size_w = float(params.width)
        size_h = float(params.height)

        # Normalise to [-0.5, 0.5] and apply 8:7 -> 16:9 stretch
        nx = (xs / size_w) - 0.5
        ny = (ys / size_h) - 0.5
        nx = nx * _ASPECT_RATIO

        px = nx.copy()
        py = ny.copy()
        active = np.ones(nx.shape, dtype=bool)

        with np.errstate(over="ignore", invalid="ignore"):
            for _ in range(_MAX_ITER):
                if not active.any():
                    break
                x2 = px * px
                y2 = py * py
                dp_x = px * (
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
                                + x2 * (1551.2922363 + x2 * (-2735.5422363 + x2 * 1923.1572266))
                            )
                        )
                    )
                    # Inside the multiplication — see `_hyperview`.
                    + y2 * -0.1086027
                )
                dp_y = py * (1.0238225 + y2 * -0.1025671 + x2 * (-0.2639930 + x2 * 0.2979266))
                diff_x = dp_x - nx
                diff_y = dp_y - ny

                # Converged: freeze without applying the update
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
    for (var i: i32 = 0; i < 12; i = i + 1) {
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

    def adjust_lens_profile(self, profile) -> None:
        """``gopro_hyperview.rs:57-63``: an 8:7 calibration was authored on
        the squeezed pixels — widen it back. ``lens_model`` is renamed
        unconditionally, aspect matching or not."""
        aspect = int(profile.calib_dimension["w"] / profile.calib_dimension["h"] * 100.0)
        if aspect == 114:  # It's 8:7
            profile.calib_dimension["w"] = round(
                profile.calib_dimension["w"] * 1.55555555555
            )
        profile.lens_model = "Hyperview"

    def id(self) -> str:
        return "gopro_hyperview"
