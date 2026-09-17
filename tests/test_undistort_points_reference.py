"""``undistort_points`` against upstream Rust (gap item D-06).

``tests/test_point_undistort.py`` checks the *structure* of the points family:
that each stage is wired in the right order, that the sentinel appears where it
should. It deliberately does not re-derive the distortion arithmetic. This file
does the other half — it runs the port against upstream's own
``cpu_undistort.rs::undistort_points``, compiled verbatim, on 26 cases /
304 coordinate values.

That comparison was worth doing. It is what found the three defects recorded in
``TestTheDeviationsTheReferenceFound`` below, two of which the structural tests
could not see by construction: one is a crash in a branch whose result the
structural tests never compared to anything, the other is an ordering mistake
that a symmetric test cannot detect because both sides of the assertion share
it.

The fixture's ``expected`` values are upstream's. ``null`` means upstream
returned NaN — JSON cannot hold a NaN, and a NaN is a *value* here, not a
missing one: the light-refraction branch divides by a coefficient and can push
the expression under a square root negative, which Rust's ``f64::sqrt`` turns
into NaN and which the port now matches.

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

# The one case where the port deliberately does not match upstream; see
# `TestTheDeviationsTheReferenceFound`. Excluded here by name rather than by
# loosening a tolerance, so the exclusion is visible.
_KNOWN_DEVIATION = "digital_lens_hyperview"


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
        if case["name"] == _KNOWN_DEVIATION:
            pytest.skip(
                "the port's HyperView inverse is guarded and upstream's is not; "
                "TestTheDeviationsTheReferenceFound pins the difference"
            )
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
            "digital_lens_hyperview": "the 0.81 factor and the guard",
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

    def test_the_digital_lens_inversion_is_guarded_where_upstream_is_not(self):
        """A deliberate, still-live deviation — recorded, not resolved.

        The port's GoPro HyperView ``distort_point`` has two additions its
        upstream does not: 20 fixed-point steps instead of 12, and a reset to
        the un-inverted input when the iterate leaves [-2, 2]. Upstream does
        neither, so on this case — the digital lens reached through
        ``lens_correction_amount < 1`` — it returns NaN for every point while
        the port returns finite coordinates.

        Both additions make the port more robust and both change output, so
        removing them is not a bug fix; it is a behaviour change for every
        HyperView clip, and it is a decision for whoever owns the port rather
        than a side effect of porting the points family.
        """
        case = _BY_NAME[_KNOWN_DEVIATION]
        assert all(
            component is None for pair in case["expected"] for component in pair
        ), "upstream no longer returns NaN here; the deviation is gone"

        got = _run(case)
        assert all(
            math.isfinite(component) for pair in got for component in pair
        ), "the port now returns NaN too — the guard was removed"

    def test_the_guard_returns_the_uninverted_input(self):
        """What the port's guard actually does, called directly.

        The reset sets the iterate back to the normalised, aspect-stretched
        input — the value the inversion started from — so the "answer" is the
        input passed through, not a pre-image of it. That is exactly why the
        deviation matters and why it is not obviously visible downstream.
        """
        from pygyroflow.stabilization.distortion_models import from_name
        from pygyroflow.types.kernel_params import KernelParams

        width, height = 1920, 1080
        kernel_params = KernelParams()
        kernel_params.width = width
        kernel_params.height = height
        kernel_params.output_width = width
        kernel_params.output_height = height

        hyperview = from_name("gopro_hyperview")
        aspect = 14.0 / 9.0
        # Chosen because the first fixed-point step leaves [-2, 2]: with the
        # coefficient set this polynomial grows fast at the frame edge.
        point = (1960.0, 540.0)
        result = hyperview.distort_point(point[0], point[1], 1.0, kernel_params)
        nx = (point[0] / width - 0.5) * aspect
        ny = point[1] / height - 0.5
        assert result == pytest.approx(
            ((nx + 0.5) * width, (ny + 0.5) * height), abs=1e-6
        )

    def test_the_guarded_output_is_not_a_real_inverse(self):
        """Why the deviation matters: the guarded point is not a pre-image.

        Feeding the port's answer back through the *forward* map does not
        return the input, so a caller that trusts it gets a plausible-looking
        wrong coordinate rather than an obviously missing pixel.
        """
        from pygyroflow.stabilization.distortion_models import from_name
        from pygyroflow.types.kernel_params import KernelParams

        width, height = 1920, 1080
        kernel_params = KernelParams()
        kernel_params.width = width
        kernel_params.height = height
        kernel_params.output_width = width
        kernel_params.output_height = height

        hyperview = from_name("gopro_hyperview")
        point = (1960.0, 540.0)
        guarded = hyperview.distort_point(point[0], point[1], 1.0, kernel_params)
        # The forward map is `undistort_point`'s inverse; going back through it
        # recovers the stretched input rather than the point we asked about.
        back = hyperview.undistort_point(guarded[0], guarded[1], kernel_params)
        assert back != pytest.approx(point, abs=1.0)
