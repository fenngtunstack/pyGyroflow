"""Spline interpolation, ported from Gyroflow's ``gyro_source/splines.rs``.

Two independent pieces live here, and both are on the path from a camera's
stabilization metadata to the pixels:

* :class:`CatmullRom` — the IBIS/OIS displacement curves. A Sony file reports
  in-body and optical stabilization as a series of samples; the render has to
  ask for the displacement at an arbitrary sensor row.
* :class:`BivariateSpline` — the mesh correction. Sony's "focal plane
  distortion" and mesh tables are stored as natural cubic spline *coefficient
  blocks*, not as points, so interpolating them means re-running the same
  tridiagonal solve the writer did.

Neither is vectorised: both are called per point, per frame, and the original
is written as a scalar loop. Keeping the arithmetic identical — including the
order the terms are summed in — is what makes the port verifiable against the
Rust, so the expressions below are the Rust ones, re-parenthesised only where
Python needs it.
"""

from __future__ import annotations

import bisect
import math
from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T")

# From splines.rs: the largest mesh grid the buffer layout can hold.
MAX_GRID_SIZE = 9
MAX_BUFFER_SIZE = (
    9
    + MAX_GRID_SIZE * MAX_GRID_SIZE * 2
    + MAX_GRID_SIZE * MAX_GRID_SIZE * 4 * 2
    + 20  # focal plane data
)

__all__ = [
    "CatmullRom",
    "BivariateSpline",
    "MAX_GRID_SIZE",
    "MAX_BUFFER_SIZE",
    "as_catmull_rom",
    "interpolate_mesh",
]


# `Generic[T]` rather than PEP 695's `class CatmullRom[T]`: the package
# supports Python 3.11, where the new syntax is a SyntaxError.
class CatmullRom(Generic[T]):  # noqa: UP046
    """Catmull-Rom interpolation over ``(position, value)`` control points.

    ``value`` is generic: ``f64`` for a scalar curve, a 3-vector for the IBIS
    displacement. The arithmetic is expressed so it works for both — Python's
    ``*`` and ``+`` do the right thing for a float and for a numpy array, and
    :meth:`_catmull_rom` never relies on anything else.
    """

    __slots__ = ("points",)

    def __init__(self, points: list[tuple[float, T]] | None = None) -> None:
        self.points = list(points) if points else []

    def add_point(self, position: float, value: T) -> None:
        self.points.append((position, value))

    def __len__(self) -> int:
        return len(self.points)

    def __bool__(self) -> bool:
        return bool(self.points)

    def interpolate(self, t: float) -> T | None:
        """The value at *t*, or None outside the control points.

        None is a real answer, not an error: outside the curve there is no
        displacement to apply, and upstream's callers fall back to zero. The
        ends are also exclusive — ``search_lower_cp`` refuses the last point,
        because the interpolation needs a segment *after* the lower one.
        """
        if len(self.points) < 2:
            return None

        lower = self._search_lower_cp(t)
        if lower is None or lower + 1 >= len(self.points):
            return None

        lower_pos, lower_val = self.points[lower]
        next_pos, next_val = self.points[lower + 1]

        k = self._normalize(t, lower_pos, next_pos)

        # The neighbouring control points, reflected at the ends: a Catmull-Rom
        # segment needs two points either side of it, and the curve is only
        # defined between the first and last, so the reflection is what the
        # original does rather than extrapolating.
        if lower <= 0:
            lower2_val = lower_val * 2.0 - next_val
        else:
            lower2_val = self.points[lower - 1][1]

        if lower + 2 >= len(self.points):
            next2_val = next_val * 2.0 - lower_val
        else:
            next2_val = self.points[lower + 2][1]

        return self._catmull_rom(k, lower2_val, lower_val, next_val, next2_val)

    def _search_lower_cp(self, t: float) -> int | None:
        """Index of the control point just below *t*, or None.

        Mirrors ``Vec::binary_search_by``: an exact hit returns that index, and
        a miss returns the insertion point, which this then shifts down by one.
        A hit on the *last* point is refused — there is no segment after it.
        """
        points = self.points
        count = len(points)
        if count < 2 or math.isnan(t):
            return None

        positions = [p[0] for p in points]
        index = bisect.bisect_left(positions, t)

        exact = index < count and positions[index] == t
        if exact:
            if index == count - 1:
                return None
            return index

        if index >= count or index == 0:
            return None
        return index - 1

    @staticmethod
    def _normalize(value: float, start: float, end: float) -> float:
        return (value - start) / (end - start)

    @staticmethod
    def _catmull_rom(t: float, x: T, a: T, b: T, y: T) -> T:
        # Term order copied from the Rust: for floats the sum is not
        # associative, and the last mantissa bit is part of the test.
        return (
            ((((a * 3.0 - x) - b * 3.0) + y) * 0.5) * t * t * t
            + ((b - x) * 0.5) * t
            + a
            + (((b * 4.0 + a * -5.0 + x + x) - y) * 0.5) * t * t
        )


class BivariateSpline:
    """Natural cubic spline interpolation over a stored coefficient grid.

    The mesh a Sony file carries is not a table of points; it is the output of
    a natural cubic spline fit along one axis, and then another fit along the
    other. So interpolating means evaluating two cubic polynomials — no fit, no
    solve at render time.
    """

    __slots__ = ("grid_size",)

    def __init__(self, width: int, height: int) -> None:
        if width > MAX_GRID_SIZE or height > MAX_GRID_SIZE:
            raise ValueError(
                f"grid {width}x{height} exceeds the {MAX_GRID_SIZE}x{MAX_GRID_SIZE} "
                "mesh buffer"
            )
        self.grid_size = (width, height)

    @staticmethod
    def _cubic_spline_coefficients(
        mesh, step: int, offset: int, size: float, n: int,
        a: list[float], b: list[float], c: list[float], d: list[float],
        alpha: list[float], mu: list[float], z: list[float],
    ) -> None:
        """Solve the natural cubic spline tridiagonal system in place.

        ``a`` gets the sampled values; ``b``/``c``/``d`` the quadratic, linear
        and cubic coefficients. ``alpha``/``mu``/``z`` are the scratch vectors
        of the Thomas algorithm the original uses.
        """
        h = size / (n - 1)
        inv_h = 1.0 / h
        three_inv_h = 3.0 * inv_h
        h_over_3 = h / 3.0
        inv_3h = 1.0 / (3.0 * h)

        for i in range(n):
            a[i] = mesh[(i + offset) * step]
        for i in range(1, n - 1):
            alpha[i] = three_inv_h * (a[i + 1] - 2.0 * a[i] + a[i - 1])

        mu[0] = 0.0
        z[0] = 0.0
        for i in range(1, n - 1):
            mu[i] = 1.0 / (4.0 - mu[i - 1])
            z[i] = (alpha[i] * inv_h - z[i - 1]) * mu[i]

        # Natural end condition: zero second derivative at the far end.
        c[n - 1] = 0.0
        for j in range(n - 2, -1, -1):
            c[j] = z[j] - mu[j] * c[j + 1]
            b[j] = (a[j + 1] - a[j]) * inv_h - h_over_3 * (c[j + 1] + 2.0 * c[j])
            d[j] = (c[j + 1] - c[j]) * inv_3h

    @staticmethod
    def _cubic_spline_interpolate(
        a: list[float], b: list[float], c: list[float], d: list[float],
        n: int, x: float, size: float,
    ) -> float:
        index = min(n - 2, max(0, int((n - 1.0) * x / size)))
        dx = x - size * index / (n - 1.0)
        return a[index] + b[index] * dx + c[index] * dx * dx + d[index] * dx * dx * dx

    def interpolate(
        self, size_x: float, size_y: float, mesh, mesh_offset: int, x: float, y: float
    ) -> float:
        """Sample one component of the mesh at ``(x, y)``.

        ``mesh_offset`` picks the component, and the arithmetic below is the
        *layout* of the buffer — see the writer in ``gyro_source/sony.rs``::

            9 header words, then gs*gs*2 coefficients, then one block of
            gs*4 per grid row per component, and the focal-plane table last.

        Indexing it wrong reads a neighbouring component and returns a
        plausible number, which is why the port keeps upstream's formula
        rather than re-deriving an offset it understands.
        """
        grid_w, grid_h = self.grid_size
        a = [0.0] * MAX_GRID_SIZE
        b = [0.0] * MAX_GRID_SIZE
        c = [0.0] * MAX_GRID_SIZE
        d = [0.0] * MAX_GRID_SIZE
        alpha = [0.0] * (MAX_GRID_SIZE - 1)
        mu = [0.0] * MAX_GRID_SIZE
        z = [0.0] * MAX_GRID_SIZE
        intermediate_values = [0.0] * MAX_GRID_SIZE

        index = min(grid_w - 2, max(0, int((grid_w - 1.0) * x / size_x)))
        dx = x - size_x * index / (grid_w - 1.0)
        dx2 = dx * dx

        block = grid_h * 4
        offs = 9 + grid_h * grid_h * 2 + (block * grid_h * mesh_offset) + index

        for j in range(grid_h):
            base = offs + j * block
            intermediate_values[j] = (
                mesh[base + grid_h * 0]
                + mesh[base + grid_h * 1] * dx
                + mesh[base + grid_h * 2] * dx2
                + mesh[base + grid_h * 3] * dx2 * dx
            )

        self._cubic_spline_coefficients(
            intermediate_values, 1, 0, size_y, grid_h, a, b, c, d, alpha, mu, z
        )
        return self._cubic_spline_interpolate(a, b, c, d, grid_h, y, size_y)


def interpolate_mesh(x: float, y: float, size: tuple[float, float], mesh) -> tuple[float, float]:
    """Both components of a mesh correction at ``(x, y)``.

    Port of ``gyro_source/sony.rs::interpolate_mesh``. Header words 1 and 2 are
    the grid dimensions.
    """
    grid = BivariateSpline(int(mesh[1]), int(mesh[2]))
    return (
        grid.interpolate(size[0], size[1], mesh, 0, x, y),
        grid.interpolate(size[0], size[1], mesh, 1, x, y),
    )


def as_catmull_rom(value) -> CatmullRom | None:
    """Coerce a stored curve into a :class:`CatmullRom`.

    A curve reaches this port in three shapes and all three are real: an
    already-built ``CatmullRom``, the ``{"points": [[t, [x, y, z]], ...]}`` map
    a ``.gyroflow`` file decodes to, and the bare point list. The values are
    converted to numpy arrays because the interpolation multiplies them by
    scalars — a Python list would concatenate instead.

    Returns None for an empty or missing curve, which is what upstream's
    ``interpolate().unwrap_or_default()`` callers treat as zero displacement.
    """
    if value is None:
        return None
    if isinstance(value, CatmullRom):
        return value

    points = value.get("points") if isinstance(value, dict) else value
    if not points:
        return None

    spline: CatmullRom = CatmullRom()
    for position, vector in points:
        spline.add_point(float(position), np.asarray(vector, dtype=np.float64))
    return spline
