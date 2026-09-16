"""Spline interpolation (gap item A-07).

``gyro_source/splines.rs`` had no Python side at all, which left the IBIS/OIS
displacement curves and the mesh correction unreachable — and those are the
two things a Sony file reports that a plain lens profile does not.

The filters themselves are checked against the upstream Rust, run verbatim:
``tests/golden/splines.json`` holds its output for 17 cases. See
``tests/golden/generate_splines_reference.py`` for how it was produced.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from pygyroflow.gyro_source.splines import (  # noqa: E402
    MAX_BUFFER_SIZE,
    MAX_GRID_SIZE,
    BivariateSpline,
    CatmullRom,
    interpolate_mesh,
)

_FIXTURE = pathlib.Path(__file__).parent / "golden" / "splines.json"

with open(_FIXTURE, encoding="utf-8") as _handle:
    _DOC = json.load(_handle)
_CASES = _DOC["cases"]


def _run(case):
    payload = case["input"]
    if case["op"] == "catmull":
        spline: CatmullRom = CatmullRom()
        for position, value in payload["points"]:
            spline.add_point(position, value)
        return [spline.interpolate(t) for t in payload["probes"]]

    spline = BivariateSpline(payload["grid_w"], payload["grid_h"])
    return [
        spline.interpolate(
            payload["size_x"], payload["size_y"], payload["mesh"],
            payload["mesh_offset"], x, y,
        )
        for x, y in payload["probes"]
    ]


class TestAgainstUpstreamRust:
    @pytest.mark.parametrize("case", _CASES, ids=lambda c: c["name"])
    def test_matches_the_rust(self, case):
        expected = case["expected"]
        got = _run(case)
        assert len(got) == len(expected)
        for index, (value, want) in enumerate(zip(got, expected)):
            if want is None:
                assert value is None, f"{case['name']}[{index}]"
            else:
                assert value == pytest.approx(want, rel=1e-12, abs=1e-12), (
                    f"{case['name']}[{index}]: {value!r} vs {want!r}"
                )

    def test_bit_exact_here(self):
        """The port reproduces the *fixture's* build exactly.

        Not a claim that the two languages agree bit for bit in general: a
        recompilation of the same Rust shifts some of these values by one ULP
        (a debug and a release build disagree too), so the reference is a
        specific binary's output rather than the algorithm's canonical result.
        What this pins is that the port's arithmetic — term order included —
        is the arithmetic that binary performed. A platform where libm or the
        summation order differed would fail here, visibly, rather than
        quietly loosening the tolerance everywhere.

        Separate from the comparison above for exactly that reason.
        """
        for case in _CASES:
            for value, want in zip(_run(case), case["expected"]):
                if want is None:
                    assert value is None
                else:
                    assert value == want, case["name"]

    def test_the_fixture_is_not_self_generated(self):
        provenance = _DOC["_provenance"].lower()
        assert "rust" in provenance
        assert "not produced by the python port" in provenance
        # The build-dependence is part of the fixture's contract, so it is
        # asserted rather than left to a comment nobody reads.
        assert "not bit-reproducible" in provenance


class TestCatmullRom:
    """The behaviour the golden cases imply, stated as properties."""

    def test_needs_two_points(self):
        assert CatmullRom().interpolate(0.0) is None
        single = CatmullRom()
        single.add_point(0.0, 5.0)
        assert single.interpolate(0.0) is None

    def test_hits_the_control_points(self):
        spline = CatmullRom()
        for position, value in [(0.0, 1.0), (1.0, 2.0), (2.0, 0.5)]:
            spline.add_point(position, value)
        assert spline.interpolate(0.0) == pytest.approx(1.0)
        assert spline.interpolate(1.0) == pytest.approx(2.0)

    def test_the_last_point_has_no_segment(self):
        """The interpolation needs a lower point *and* one after it, so the
        final control point is a dead end — and so is anything past it."""
        spline = CatmullRom()
        for position, value in [(0.0, 1.0), (1.0, 2.0), (2.0, 0.5)]:
            spline.add_point(position, value)
        assert spline.interpolate(2.0) is None
        assert spline.interpolate(3.0) is None

    def test_before_the_first_point_is_out_of_range(self):
        spline = CatmullRom()
        for position, value in [(0.0, 1.0), (1.0, 2.0)]:
            spline.add_point(position, value)
        assert spline.interpolate(-0.5) is None

    def test_nan_is_refused(self):
        spline = CatmullRom()
        for position, value in [(0.0, 1.0), (1.0, 2.0)]:
            spline.add_point(position, value)
        assert spline.interpolate(float("nan")) is None

    def test_the_ends_reflect_rather_than_extrapolate(self):
        """A Catmull-Rom segment needs two points either side of it. At the
        ends the original mirrors the neighbour instead of extrapolating a
        trend, and the first segment is what shows it."""
        spline = CatmullRom()
        for position, value in [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]:
            spline.add_point(position, value)
        # Linear data with reflected ends stays linear: any extrapolating
        # tangent would bow the first segment.
        assert spline.interpolate(0.5) == pytest.approx(0.5)

    def test_vector_values_work_elementwise(self):
        """The IBIS curve stores a 3-vector per point, not a scalar."""
        spline: CatmullRom = CatmullRom()
        spline.add_point(0.0, np.array([0.0, 0.0, 0.0]))
        spline.add_point(1.0, np.array([1.0, 2.0, 3.0]))
        spline.add_point(2.0, np.array([2.0, 4.0, 6.0]))
        value = spline.interpolate(0.5)
        assert value == pytest.approx([0.5, 1.0, 1.5])

    def test_adding_a_point_extends_the_range(self):
        spline = CatmullRom()
        spline.add_point(0.0, 1.0)
        spline.add_point(1.0, 2.0)
        assert spline.interpolate(1.0) is None
        spline.add_point(2.0, 3.0)
        assert spline.interpolate(1.0) == pytest.approx(2.0)


class TestBivariateSpline:
    def test_an_oversized_grid_is_refused(self):
        BivariateSpline(MAX_GRID_SIZE, MAX_GRID_SIZE)
        with pytest.raises(ValueError):
            BivariateSpline(MAX_GRID_SIZE + 1, 2)

    def test_a_constant_mesh_interpolates_to_that_constant(self):
        """The sanity case the real numbers cannot give.

        A constant mesh means every coefficient block has its constant term
        set and the other three at zero. Building that buffer *is* the test:
        the index arithmetic below is what the layout claims, so getting the
        row/column strides wrong leaves zeros behind and the samples stop
        being constant.
        """
        grid = 3
        block = grid * 4
        base = 9 + grid * grid * 2
        mesh = [0.0] * MAX_BUFFER_SIZE
        mesh[1] = mesh[2] = float(grid)
        for component in (0, 1):
            for column in range(grid - 1):
                for row in range(grid):
                    mesh[base + block * grid * component + column + row * block] = 7.5
        spline = BivariateSpline(grid, grid)
        for x, y in [(0.0, 0.0), (5.0, 5.0), (100.0, 90.0)]:
            assert spline.interpolate(100.0, 100.0, mesh, 0, x, y) == pytest.approx(7.5)

    def test_the_two_components_read_different_blocks(self):
        grid = 3
        block = grid * 4
        mesh = [0.0] * MAX_BUFFER_SIZE
        mesh[1] = mesh[2] = float(grid)
        start = 9 + grid * grid * 2
        for index in range(start, start + block * grid):
            mesh[index] = 1.0
        for index in range(start + block * grid, start + block * grid * 2):
            mesh[index] = 9.0
        spline = BivariateSpline(grid, grid)
        assert spline.interpolate(10.0, 10.0, mesh, 0, 5.0, 5.0) == pytest.approx(1.0)
        assert spline.interpolate(10.0, 10.0, mesh, 1, 5.0, 5.0) == pytest.approx(9.0)

    def test_interpolate_mesh_reads_the_grid_from_the_header(self):
        """Header words 1 and 2 are the grid dimensions, and both components
        come back in one call."""
        grid = 3
        block = grid * 4
        mesh = [0.0] * MAX_BUFFER_SIZE
        mesh[1] = mesh[2] = float(grid)
        start = 9 + grid * grid * 2
        for index in range(start, start + block * grid):
            mesh[index] = 2.0
        for index in range(start + block * grid, start + block * grid * 2):
            mesh[index] = 4.0
        x, y = interpolate_mesh(5.0, 5.0, (10.0, 10.0), mesh)
        assert x == pytest.approx(2.0)
        assert y == pytest.approx(4.0)

    def test_the_buffer_constants_match_the_layout(self):
        assert MAX_BUFFER_SIZE == (
            9 + MAX_GRID_SIZE * MAX_GRID_SIZE * 2
            + MAX_GRID_SIZE * MAX_GRID_SIZE * 4 * 2 + 20
        )

    def test_samples_are_finite(self):
        """No NaN from the tridiagonal solve on a realistic grid."""
        grid = 9
        mesh = [
            math.sin(index * 0.37) * (1.0 + index % 7) * 0.25
            for index in range(MAX_BUFFER_SIZE)
        ]
        mesh[1] = mesh[2] = float(grid)
        spline = BivariateSpline(grid, grid)
        for x, y in [(0.0, 0.0), (960.0, 540.0), (1919.0, 1079.0)]:
            assert math.isfinite(spline.interpolate(1920.0, 1080.0, mesh, 0, x, y))
