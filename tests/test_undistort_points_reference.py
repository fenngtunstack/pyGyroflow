"""``undistort_points`` against upstream Rust (gap item D-06).

``tests/test_point_undistort.py`` checks the *structure* of the points family:
that each stage is wired in the right order, that the sentinel appears where it
should. It deliberately does not re-derive the distortion arithmetic. This file
does the other half — it runs the port against upstream's own
``cpu_undistort.rs::undistort_points``, compiled verbatim, on 28 cases /
644 coordinate values.

That comparison earned its keep. It found four defects; the structural tests
could not see three of them by construction, because each is either a crash in
a branch whose result was never compared to anything, or an ordering mistake
where both sides of a symmetric assertion share the error. They are pinned
individually in ``TestTheDeviationsTheReferenceFound`` and
``TestTheHyperviewInversion``.

The fixture's ``expected`` values are upstream's. ``null`` means upstream
returned NaN — JSON cannot hold a NaN, and a NaN is a *value* here, not a
missing one: two branches can produce it (a square root of a negative, and the
HyperView inversion running away), and both are upstream's answers.

Comparison is relative over a 1.0 floor. Upstream computes in f32 and the port
in f64, so the floor is what keeps a coordinate near the origin from demanding
meaningless precision; measured disagreement across the well-conditioned cases
is at most 3.7e-7, and the per-case bounds in the fixture are set well above it
rather than tuned to it. The one case that raises its bound says why.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from pygyroflow.keyframes.types import KeyframeType  # noqa: E402
from pygyroflow.stabilization.compute_params import ComputeParams  # noqa: E402
from pygyroflow.stabilization.cpu_undistort import undistort_points  # noqa: E402
from pygyroflow.stabilization.distortion_models import from_name  # noqa: E402

_FIXTURE = pathlib.Path(__file__).parent / "golden" / "undistort_points.json"

with open(_FIXTURE, encoding="utf-8") as _handle:
    _DOC = json.load(_handle)
_CASES = _DOC["cases"]
_BY_NAME = {case["name"]: case for case in _CASES}


class _ExactTimestampKeyframes:
    """Answers only at the exact timestamps the fixture lists.

    The same shim the Rust driver uses, for the same reason: interpolation is
    the keyframe manager's job (see D-10 in the gap analysis), not this
    function's. Both sides answer or decline identically, so a difference here
    can only come from the code under test.
    """

    def __init__(self, pairs: list) -> None:
        self._pairs = pairs

    def __bool__(self) -> bool:
        return True

    def value_at_video_timestamp(self, key: KeyframeType, timestamp_ms: float):
        del key  # only LightRefractionCoeff is reachable from undistort_points
        for timestamp, value in self._pairs:
            if abs(timestamp - timestamp_ms) < 1e-9:
                return value
        return None


class _Lens:
    """The two fields upstream reads off ``params.lens``."""

    def __init__(self, horizontal: float, vertical: float) -> None:
        self.input_horizontal_stretch = horizontal
        self.input_vertical_stretch = vertical


def _params(spec: dict) -> ComputeParams:
    params = ComputeParams(
        width=spec["width"],
        height=spec["height"],
        output_width=spec["output_width"],
        output_height=spec["output_height"],
        light_refraction_coefficient=spec["light_refraction_coefficient"],
        input_horizontal_stretch=spec["input_horizontal_stretch"],
        input_vertical_stretch=spec["input_vertical_stretch"],
        distortion_model_name=spec["distortion_model"],
        digital_lens=(
            from_name(spec["digital_lens"]) if spec["digital_lens"] else None
        ),
        keyframes=_ExactTimestampKeyframes(spec["light_refraction_keyframes"]),
    )
    # Upstream reads the stretch from `params.lens`, not from `ComputeParams`.
    # `_input_stretch` prefers `params.lens` when it is set, so setting it here
    # is what makes both sides read the same field.
    params.lens = _Lens(
        spec["input_horizontal_stretch"], spec["input_vertical_stretch"]
    )
    return params


def _run(case: dict) -> list[tuple[float, float]]:
    return undistort_points(
        [tuple(point) for point in case["distorted"]],
        np.array(case["camera_matrix"], dtype=np.float64),
        case["distortion_coeffs"],
        np.array(case["rotation"], dtype=np.float64),
        p=np.array(case["p"], dtype=np.float64) if case.get("p") else None,
        rot_per_point=(
            [np.array(m, dtype=np.float64) for m in case["rot_per_point"]]
            if case.get("rot_per_point")
            else None
        ),
        params=_params(case["params"]),
        lens_correction_amount=case.get("lens_correction_amount", 1.0),
        timestamp_ms=case.get("timestamp_ms", 0.0),
        shift_per_point=case.get("shift_per_point"),
        mesh=case.get("mesh"),
    )


class TestAgainstUpstreamRust:
    """Every fixture case, value by value."""

    @pytest.mark.parametrize("case", _CASES, ids=lambda c: c["name"])
    def test_matches_the_rust(self, case):
        got = _run(case)
        expected = case["expected"]
        assert len(got) == len(expected), case["name"]
        tolerance = case.get("tolerance", 1e-5)
        for index, (pair, want) in enumerate(zip(got, expected)):
            for axis in range(2):
                where = f"{case['name']}[{index}].{'xy'[axis]}"
                if want[axis] is None:
                    assert math.isnan(pair[axis]), f"{where}: {pair[axis]!r}, want NaN"
                    continue
                assert not math.isnan(pair[axis]), f"{where}: NaN, want {want[axis]!r}"
                scale = max(abs(want[axis]), 1.0)
                assert abs(pair[axis] - want[axis]) / scale <= tolerance, (
                    f"{where}: {pair[axis]!r} vs {want[axis]!r}"
                )

    def test_the_fixture_covers_the_whole_branch_tree(self):
        """A guard against the fixture quietly shrinking.

        Each name is a branch of the copied function; losing one would leave
        that branch unverified without failing anything.
        """
        required = {
            "identity_no_distortion": "the straight-through path",
            "fisheye_distortion": "a real optical model",
            "opencv_standard_distortion": "the model dispatch is not hardcoded",
            "sony_distortion": "a long Cartesian expression, as a model",
            "insta360_distortion": "the other long one",
            "p_is_premultiplied": "rr = p @ rotation",
            "rot_per_point_overrides": "the per-point rotation override",
            "light_refraction_from_params": "the refraction branch",
            "light_refraction_from_keyframe": "the keyframe override of it",
            "lens_correction_full": "amount == 1 skips the blend",
            "lens_correction_half": "the blend itself",
            "shift_per_point": "IBIS/OIS displacement",
            "mesh_focal_plane_only": "the focal-plane branch",
            "mesh_full": "the full-mesh branch",
            "digital_lens_superview": "the digital lens and its 0.91 factor",
            "digital_lens_hyperview": "the 0.81 factor and the runaway inverse",
            "digital_lens_hyperview_converging": "the same, where it converges",
            "non_converging_point_returns_sentinel": "the sentinel",
        }
        missing = set(required) - set(_BY_NAME)
        assert not missing, {name: required[name] for name in sorted(missing)}

    def test_the_fixture_is_upstreams_output_not_the_ports(self):
        with open(_FIXTURE, encoding="utf-8") as handle:
            doc = json.load(handle)
        assert "upstream" in doc["_provenance"]
        assert all(case["expected"] for case in doc["cases"]), (
            "a case has no expected values — rerun --fill"
        )


class TestTheDeviationsTheReferenceFound:
    """The three things the comparison caught. Each is pinned individually.

    These are not "regression tests for a fixed bug" in the usual sense: the
    first two are, but the third records a difference that is still there on
    purpose, so that whoever reads the fixture does not think the port simply
    matches upstream everywhere.
    """

    def test_the_ibis_rotation_uses_the_updated_x_for_y(self):
        """Upstream assigns x, then reads the new x when computing y.

        A simultaneous ``x, y = ...`` — the obvious Python spelling — leaves y
        on the old x and agrees with upstream only when sin(angle) is zero.
        The fixture's shifts all carry a non-zero angle, and getting this wrong
        was worth 1.5e-2 relative against a bound of 1e-5.
        """
        case = _BY_NAME["shift_per_point"]
        angles = [shift[2] for shift in case["shift_per_point"]]
        assert any(a != 0.0 for a in angles), "the case no longer exercises it"

        got = _run(case)
        want = case["expected"]
        # Index 0 carries angle 0.0 and must agree either way; index 1 carries
        # 0.002 and is where the ordering shows.
        assert got[0] == pytest.approx(tuple(want[0]), abs=1e-4)
        simultaneous = self._with_the_wrong_ordering(case)
        assert simultaneous[1] != pytest.approx(got[1], abs=1e-3)

    @staticmethod
    def _with_the_wrong_ordering(case):
        """Recompute the shift branch the way the simultaneous form would."""
        from pygyroflow.stabilization.cpu_undistort import (
            _points_kernel_params,
        )

        params = _params(case["params"])
        point = case["distorted"][1]
        shift = case["shift_per_point"][1]
        camera_matrix = np.array(case["camera_matrix"], dtype=np.float64)
        coeffs = case["distortion_coeffs"]
        rotation = np.array(case["rotation"], dtype=np.float64)

        c = (camera_matrix[0][2], camera_matrix[1][2])
        f = (camera_matrix[0][0], camera_matrix[1][1])
        kernel_params = _points_kernel_params(
            camera_matrix, coeffs, params, params.light_refraction_coefficient
        )
        model = from_name(params.distortion_model_name)

        x = float(point[0])
        y = float(point[1])
        cos_a = math.cos(shift[2])
        sin_a = math.sin(shift[2])
        x = x - c[0] - shift[3] + shift[0]
        y = y - c[1] - shift[4] + shift[1]
        # The wrong form: both read the pre-rotation x.
        x, y = cos_a * x - sin_a * y + c[0], sin_a * x + cos_a * y + c[1]
        pt = model.undistort_point((x - c[0]) / f[0], (y - c[1]) / f[1], kernel_params)
        projected = rotation @ np.array([pt[0], pt[1], 1.0])
        return (projected[0] / projected[2], projected[1] / projected[2])

    def test_a_negative_square_root_root_gives_nan_not_a_crash(self):
        """Rust's ``sqrt`` returns NaN outside its domain; Python's raises.

        With a refraction coefficient below 1, ``sin_theta_d`` exceeds 1 for
        points far enough from the centre and the expression under the root
        goes negative. Upstream makes those coordinates NaN and carries on —
        the callers drop NaN points because every comparison against NaN is
        false. ``math.sqrt`` would raise ``ValueError`` and abort a render
        upstream completes.
        """
        case = _BY_NAME["light_refraction_from_keyframe"]
        assert case["params"]["light_refraction_coefficient"] < 1.0 or any(
            value < 1.0 for _, value in case["params"]["light_refraction_keyframes"]
        )
        nan_expected = [
            index
            for index, pair in enumerate(case["expected"])
            if any(component is None for component in pair)
        ]
        assert nan_expected, "the fixture no longer produces a NaN"

        got = _run(case)  # would raise before the fix
        for index in nan_expected:
            assert any(math.isnan(component) for component in got[index])

class TestTheHyperviewInversion:
    """The digital lens whose inverse upstream does not really have.

    Two separate things are pinned here, and they were found separately:

    * The coupling term in the x polynomial was outside the multiplication by
      x in the port and inside it upstream. The reference fixture caught it.
    * The inversion itself is a substitution that converges over only part of
      the frame. Upstream caps it at 12 steps with no divergence guard and
      returns NaN where it runs away; the port now does the same, so the
      fixture's two HyperView cases agree instead of one being skipped.
    """

    def test_the_y_squared_term_is_inside_the_multiplication(self):
        """``x * (P + y2*c)``, not ``x * P + y2*c``.

        The two differ by a factor of x, so they agree on the *vertical*
        centreline and diverge away from it. Written as an additive term at the
        end of the expression it looks right and is wrong; upstream and the
        port's own WGSL text both have it inside, which is what exposed it.

        Tested at the point where the difference is largest and least
        ambiguous: x = width/2 makes the corrected x exactly 0 in the correct
        form, and non-zero in the wrong one.
        """
        from pygyroflow.stabilization.distortion_models import from_name
        from pygyroflow.types.kernel_params import KernelParams

        kernel_params = KernelParams()
        kernel_params.width = kernel_params.output_width = 1920
        kernel_params.height = kernel_params.output_height = 1080
        hyperview = from_name("gopro_hyperview")

        # (960, 200) is on the centreline and off the horizontal one, so y2 is
        # non-zero while x is not. With the term inside, the forward map sends
        # the column's x to exactly 0; with it outside, to -y2 * 0.1086027.
        for point in ((960.0, 200.0), (960.0, 980.0)):
            ux, _ = hyperview.undistort_point(point[0], point[1], kernel_params)
            assert ux == pytest.approx(960.0), f"{point} left the centreline"

    def test_the_fixture_pins_both_outcomes_of_the_inversion(self):
        """One case where it diverges, one where it converges.

        Without the converging case the HyperView polynomial would only be
        exercised by a wall of NaN — the failure path would be pinned and the
        arithmetic would not.
        """
        diverging = _BY_NAME["digital_lens_hyperview"]
        converging = _BY_NAME["digital_lens_hyperview_converging"]
        assert all(
            component is None for pair in diverging["expected"] for component in pair
        )
        assert all(
            component is not None
            for pair in converging["expected"]
            for component in pair
        )

    def test_the_port_returns_nan_where_upstream_does(self):
        got = _run(_BY_NAME["digital_lens_hyperview"])
        assert all(math.isnan(component) for pair in got for component in pair)

    def test_the_iteration_count_is_upstreams_and_the_shaders(self):
        """12, in the Python and in the WGSL text, as upstream has it.

        The port carried 20 in the Python *and* in the WGSL, while claiming the
        WGSL matched Gyroflow exactly. A different count changes where the
        iteration stops, so it changes the answer wherever it has not
        converged — which, measured below, is most of the frame.
        """
        from pygyroflow.stabilization.distortion_models import from_name, gopro_hyperview

        upstream = "for (var i: i32 = 0; i < 12; i = i + 1)"
        assert upstream in from_name("gopro_hyperview").wgsl_functions()
        assert gopro_hyperview._MAX_ITER == 12

    def test_the_inversion_converges_over_a_minority_of_the_frame(self):
        """The measurement behind reproducing upstream rather than fixing it.

        If upstream's inversion ever becomes something else — fewer NaN, a
        different cap — this is what notices, and the fixture has to be
        regenerated alongside it.
        """
        import numpy as np

        from pygyroflow.stabilization.distortion_models import from_name
        from pygyroflow.types.kernel_params import KernelParams

        kernel_params = KernelParams()
        kernel_params.width = kernel_params.output_width = 1920
        kernel_params.height = kernel_params.output_height = 1080
        hyperview = from_name("gopro_hyperview")

        xs, ys = np.meshgrid(
            np.arange(0.0, 1920.0, 16.0), np.arange(0.0, 1080.0, 16.0)
        )
        xs, ys = xs.ravel(), ys.ravel()
        with np.errstate(over="ignore", invalid="ignore"):
            dx, dy = hyperview.distort_points(xs, ys, np.ones_like(xs), kernel_params)

        nan_fraction = float(np.isnan(dx).mean())
        assert 0.03 < nan_fraction < 0.20, (
            f"NaN fraction moved to {nan_fraction:.1%}; upstream's behaviour "
            "may have changed and the fixture needs regenerating"
        )
        # And the points that are not NaN are mostly not the input either:
        # the substitution stops wherever it got to.
        moved = np.abs(dx[~np.isnan(dx)] - xs[~np.isnan(dx)]) > 1.0
        assert moved.mean() > 0.5

    def test_the_forward_map_still_round_trips_where_it_converges(self):
        """The check that would have passed while the coupling term was wrong.

        A round trip through ``distort_point`` lands back near the input for
        the points where the iteration converges — including on the centreline
        column, where the coupling-term bug lived. That is why the bug needed
        an external reference rather than a round-trip test.
        """
        from pygyroflow.stabilization.distortion_models import from_name
        from pygyroflow.types.kernel_params import KernelParams

        kernel_params = KernelParams()
        kernel_params.width = kernel_params.output_width = 1920
        kernel_params.height = kernel_params.output_height = 1080
        hyperview = from_name("gopro_hyperview")

        for point in ((960.0, 600.0), (1200.0, 540.0), (960.0, 980.0)):
            moved = hyperview.distort_point(point[0], point[1], 1.0, kernel_params)
            back = hyperview.undistort_point(moved[0], moved[1], kernel_params)
            assert back == pytest.approx(point, abs=0.05), point
