"""Focal length smoothing for zoom lenses (gap item D-08).

Upstream's ``smoothing/focal_length.rs`` is a whole file this port did not
have: the two filters, the orchestration in ``lib.rs`` that maps the single UI
strength knob onto three filter dials, and the ``fov`` compensation in
``frame_transform.rs`` that turns a smoothed curve into a digital zoom.

The filters are checked against the upstream Rust implementation itself, in
``tests/golden/focal_length_smoothing.json`` — NOT against a second Python
transcription, which would only prove the two transcriptions agree. See
``tests/golden/generate_focal_length_reference.py`` for how the fixture was
produced.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from pygyroflow.gyro_source.file_metadata import LensParams  # noqa: E402
from pygyroflow.lens import LensProfile  # noqa: E402
from pygyroflow.manager import StabilizationManager  # noqa: E402
from pygyroflow.smoothing.focal_length import (  # noqa: E402
    smooth_focal_lengths_adaptive,
    smooth_focal_lengths_gaussian,
)
from pygyroflow.stabilization.compute_params import ComputeParams  # noqa: E402
from pygyroflow.stabilization.frame_transform import (  # noqa: E402
    FrameTransform,
    _focal_length_fov_compensation,
)
from pygyroflow.stabilization_params import StabilizationParams  # noqa: E402
from pygyroflow.types.quaternion import Quat64  # noqa: E402
from pygyroflow.util import ClosestMap  # noqa: E402
from pygyroflow.zooming import get_checksum  # noqa: E402

_GOLDEN = pathlib.Path(__file__).parent / "golden" / "focal_length_smoothing.json"

# Rust and Python may differ by an ULP in `exp`; measured 0 on this machine
# (see TestGoldenRustReference.test_bit_exact_here).
_TOL = 1e-12


def _load_golden():
    with open(_GOLDEN, encoding="utf-8") as handle:
        return json.load(handle)


def _run_case(case):
    values = list(case["input"])
    args = case["args"]
    if case["op"] == "gaussian":
        return smooth_focal_lengths_gaussian(
            values, args["strength"], args["window_size"]
        )
    return smooth_focal_lengths_adaptive(
        values,
        args["fps"],
        args["max_smoothness_time"],
        args["min_smoothness_time"],
        args["max_velocity"],
    )


_GOLDEN_DOC = _load_golden()
_GOLDEN_CASES = _GOLDEN_DOC["cases"]


class TestGoldenRustReference:
    """Every case, against the upstream Rust output."""

    @pytest.mark.parametrize("case", _GOLDEN_CASES, ids=lambda c: c["name"])
    def test_matches_upstream_rust(self, case):
        expected = case["expected"]
        got = _run_case(case)
        assert len(got) == len(expected), case["name"]
        for index, (g, e) in enumerate(zip(got, expected)):
            if e is None:
                assert g is None, f"{case['name']}[{index}]"
                continue
            assert g is not None, f"{case['name']}[{index}]"
            assert math.isclose(g, e, rel_tol=_TOL, abs_tol=_TOL), (
                f"{case['name']}[{index}]: {g!r} vs {e!r}"
            )

    def test_bit_exact_here(self):
        """On this machine the port reproduces the Rust bit for bit.

        Kept separate from the comparison above so an ULP of ``exp``
        difference on another platform shows up as this test failing rather
        than as a silently loosened tolerance everywhere.
        """
        exact = 0
        total = 0
        for case in _GOLDEN_CASES:
            for g, e in zip(_run_case(case), case["expected"]):
                if e is None:
                    continue
                total += 1
                if g == e:
                    exact += 1
        assert exact == total

    def test_the_fixture_is_not_self_generated(self):
        """Guard the provenance claim in the fixture itself."""
        provenance = _GOLDEN_DOC["_provenance"].lower()
        assert "rust" in provenance
        assert "not self-generated" in provenance


class TestGaussian:
    def test_empty_and_zero_strength_are_passthrough(self):
        assert smooth_focal_lengths_gaussian([], 1.0, 5) == []
        values = [18.0, None, 20.0]
        assert smooth_focal_lengths_gaussian(values, 0.0, 5) == values

    def test_strength_zero_does_not_alias_the_input(self):
        values = [18.0, 19.0]
        out = smooth_focal_lengths_gaussian(values, 0.0, 5)
        assert out is not values

    def test_none_positions_survive(self):
        values = [18.0, None, 19.0]
        out = smooth_focal_lengths_gaussian(values, 1.0, 5)
        assert out[1] is None

    def test_an_even_window_is_widened(self):
        """A Gaussian kernel needs a centre, so an even width becomes odd."""
        values = [18.0 + i for i in range(9)]
        assert smooth_focal_lengths_gaussian(values, 1.0, 4) == pytest.approx(
            smooth_focal_lengths_gaussian(values, 1.0, 5), abs=0.0
        )

    def test_full_strength_moves_every_sample_to_the_blurred_value(self):
        """At strength 1 the original contributes nothing.

        A single spike surrounded by a flat plateau is therefore averaged
        down rather than blended back up.
        """
        values = [20.0] * 9
        values[4] = 40.0
        out = smooth_focal_lengths_gaussian(values, 1.0, 5)
        assert out[4] < 30.0

    def test_half_strength_lands_between(self):
        values = [20.0] * 9
        values[4] = 40.0
        full = smooth_focal_lengths_gaussian(values, 1.0, 5)
        half = smooth_focal_lengths_gaussian(values, 0.5, 5)
        assert full[4] < half[4] < 40.0

    def test_stairs_are_flattened_but_the_ramp_survives(self):
        """The point of the pass: kill the steps, keep the shape."""
        values = [18.0] * 10 + [19.0] * 10
        out = smooth_focal_lengths_gaussian(values, 1.0, 11)
        assert all(a <= b for a, b in zip(out, out[1:]))
        # Strictly increasing through the transition — the plateau is gone.
        assert all(a < b for a, b in zip(out[5:15], out[6:16]))
        # And the interior never overshoots the step.
        assert all(18.0 <= value <= 19.0 for value in out)

    def test_the_ends_stay_on_their_plateau(self):
        """Edge clamping pulls the window inward rather than fabricating
        samples, so a frame deep inside a plateau is left exactly alone."""
        values = [18.0] * 10 + [19.0] * 10
        out = smooth_focal_lengths_gaussian(values, 1.0, 11)
        assert out[0] == 18.0
        assert out[-1] == 19.0

    def test_a_gap_does_not_drag_the_result_toward_zero(self):
        """Only neighbours that exist count toward the weight sum."""
        solid = [20.0] * 11
        holed = list(solid)
        holed[5] = None
        out = smooth_focal_lengths_gaussian(holed, 1.0, 5)
        assert out[5] is None
        # Neighbours of the hole ignore it, so they stay at the plateau.
        assert out[4] == pytest.approx(20.0)


class TestAdaptive:
    def test_short_inputs_are_passthrough(self):
        assert smooth_focal_lengths_adaptive([20.0], 30.0, 1.0, 0.1, 3.0) == [20.0]
        assert smooth_focal_lengths_adaptive([], 30.0, 1.0, 0.1, 3.0) == []

    def test_non_positive_fps_is_passthrough(self):
        values = [18.0, 19.0, 20.0]
        assert smooth_focal_lengths_adaptive(values, 0.0, 1.0, 0.1, 3.0) == values

    def test_all_none_stays_none(self):
        values = [None] * 6
        assert smooth_focal_lengths_adaptive(values, 30.0, 1.0, 0.1, 3.0) == values

    def test_leading_gap_before_the_first_value_stays_none(self):
        """The filter is seeded at the first known frame.

        Filling the frames before it would mean guessing an optical state
        that was never reported — and the forward pass cannot invent one.
        """
        values = [None, None, 30.0, 30.0, 30.0]
        out = smooth_focal_lengths_adaptive(values, 30.0, 1.0, 0.1, 3.0)
        assert out[0] is None
        assert out[1] is None
        assert all(value is not None for value in out[2:])

    def test_a_gap_after_the_seed_is_filled_by_the_passes(self):
        values = [20.0, 20.0, None, None, 20.0, 20.0]
        out = smooth_focal_lengths_adaptive(values, 30.0, 1.0, 0.1, 3.0)
        assert all(value is not None for value in out)

    def test_a_steady_length_stays_put(self):
        values = [24.0] * 20
        out = smooth_focal_lengths_adaptive(values, 30.0, 5.0, 0.1, 3.0)
        assert out == pytest.approx(values)

    def test_a_fast_zoom_is_tracked_closer_than_a_slow_one(self):
        """Velocity opens the filter: the same total move tracked with less
        lag when it happens quickly. That is the whole point of the adaptive
        time constant — a deliberate zoom must not be smoothed away."""
        slow = [18.0 + 42.0 * i / 119.0 for i in range(120)]
        fast = slow[:10] + [60.0] * 110
        args = (30.0, 30.0, 0.4, 8.0)
        slow_out = smooth_focal_lengths_adaptive(slow, *args)
        fast_out = smooth_focal_lengths_adaptive(fast, *args)
        # At the end of the move both have caught up; mid-move the fast one
        # has travelled further toward the target.
        assert abs(fast_out[20] - 60.0) < abs(slow_out[20] - 60.0)

    def test_the_backward_pass_cancels_the_lag(self):
        """Forward + backward, versus a one-sided exponential.

        A single forward pass holds on to what it has just seen, so after a
        step up it trails: nearly all of the ramp's "mass" sits to the right
        of centre. The backward pass pulls it back. Exact symmetry is not on
        offer — the velocity signal is seeded from one side (``velocity[0] =
        velocity[1]``) and smoothed with its own one-sided pair of passes —
        so this pins the *order of magnitude* of the correction.
        """
        values = [20.0] * 5 + [40.0] * 5 + [20.0] * 5
        out = smooth_focal_lengths_adaptive(values, 30.0, 30.0, 0.4, 8.0)
        base = 20.0
        left = sum(out[i] - base for i in range(7))
        right = sum(out[i] - base for i in range(8, 15))
        two_pass_imbalance = abs(left - right) / right

        dt = 1.0 / 30.0
        alpha = 1.0 - math.exp(-dt / 0.4)
        state = base
        one_sided = []
        for value in values:
            state = state * (1.0 - alpha) + value * alpha
            one_sided.append(state)
        one_sided_left = sum(one_sided[i] - base for i in range(7))
        one_sided_right = sum(one_sided[i] - base for i in range(8, 15))
        one_sided_imbalance = abs(one_sided_left - one_sided_right) / one_sided_right

        assert one_sided_imbalance > 0.5          # the problem
        assert two_pass_imbalance < 0.15          # and it is fixed
        assert two_pass_imbalance < one_sided_imbalance / 5

    def test_the_output_range_stays_inside_the_input_range(self):
        """An exponential blend of two values never leaves the pair's span,
        so no overshoot — a compensation built from it cannot ring."""
        values = [18.0] * 20 + [60.0] + [18.0] * 20
        out = smooth_focal_lengths_adaptive(values, 30.0, 1.0, 0.1, 0.5)
        assert min(out) >= 18.0
        assert max(out) <= 60.0


def _params_with_curve(curve, smoothed, enabled=True, frame_count=None, fovs=None):
    """A ComputeParams carrying an explicit focal length curve."""
    count = frame_count if frame_count is not None else len(curve)
    return ComputeParams(
        width=1920,
        height=1080,
        output_width=1920,
        output_height=1080,
        frame_count=count,
        scaled_fps=30.0,
        fovs=list(fovs if fovs is not None else []),
        focal_lengths=list(curve),
        smoothed_focal_lengths=list(smoothed),
        focal_length_smoothing_enabled=enabled,
    )


class TestFovCompensation:
    """`dequantized / smoothed`, with every bail-out pinned."""

    def test_disabled_is_one(self):
        params = _params_with_curve([18.0], [20.0], enabled=False)
        assert _focal_length_fov_compensation(params, 0) == 1.0

    def test_the_ratio_is_dequantized_over_smoothed(self):
        params = _params_with_curve([18.0, 21.0], [18.0, 20.0])
        assert _focal_length_fov_compensation(params, 1) == pytest.approx(1.05)

    def test_a_shorter_true_length_zooms_in(self):
        """Below 1 the factor crops: the compensated fov is smaller."""
        params = _params_with_curve([18.0], [20.0])
        assert _focal_length_fov_compensation(params, 0) == pytest.approx(0.9)

    def test_a_frame_past_the_end_is_one(self):
        params = _params_with_curve([18.0], [18.5])
        assert _focal_length_fov_compensation(params, 5) == 1.0

    def test_an_unknown_sample_on_either_side_is_one(self):
        assert _focal_length_fov_compensation(
            _params_with_curve([None], [18.5]), 0
        ) == 1.0
        assert _focal_length_fov_compensation(
            _params_with_curve([18.0], [None]), 0
        ) == 1.0

    def test_non_positive_lengths_are_one(self):
        assert _focal_length_fov_compensation(
            _params_with_curve([0.0], [18.0]), 0
        ) == 1.0
        assert _focal_length_fov_compensation(
            _params_with_curve([18.0], [0.0]), 0
        ) == 1.0

    def test_an_empty_curve_is_one(self):
        assert _focal_length_fov_compensation(
            _params_with_curve([], []), 0
        ) == 1.0


class _StubManager:
    """Just enough manager for the two methods under test.

    The real bodies are bound, not reimplemented, so the wiring between
    them — which is where the orchestration lives — is what runs.
    """

    extract_focal_lengths = staticmethod(StabilizationManager.extract_focal_lengths)
    _apply_focal_length_smoothing = StabilizationManager._apply_focal_length_smoothing

    def __init__(self, params):
        self.params = params


def _cp_with_lens_params(entries, frame_count=60, fps=30.0):
    """A ComputeParams whose lens metadata gives a focal length per frame."""
    cp = ComputeParams(
        width=1920,
        height=1080,
        output_width=1920,
        output_height=1080,
        frame_count=frame_count,
        scaled_fps=fps,
        lens_params=ClosestMap(entries),
    )
    return cp


def _zoom_entries(frame_count=60, fps=30.0, start=18.0, end=60.0, per=10):
    """Per-frame lens metadata: a quantization staircase from start to end.

    The steps are sized so the last one lands exactly on *end* — the curve
    is a staircase version of a linear zoom.
    """
    entries = {}
    span = end - start
    steps = max(1, (frame_count - 1) // per)
    for frame in range(frame_count):
        timestamp_us = round(frame * 1000.0 / fps * 1000.0)
        value = start + span * (frame // per) / steps
        entries[timestamp_us] = LensParams(focal_length=value)
    return entries


class TestExtractFocalLengths:
    def test_no_lens_params_yields_an_empty_list(self):
        cp = ComputeParams(frame_count=10, scaled_fps=30.0)
        assert StabilizationManager.extract_focal_lengths(cp) == []

    def test_one_entry_per_frame_in_order(self):
        cp = _cp_with_lens_params(_zoom_entries(), frame_count=60)
        curve = StabilizationManager.extract_focal_lengths(cp)
        assert len(curve) == 60
        assert curve[0] == pytest.approx(18.0)
        assert curve[-1] == pytest.approx(60.0)

    def test_frames_without_metadata_are_none_not_zero(self):
        """A gap must stay a gap: filling it with 0 would make the
        compensation ratio divide by zero-length optics."""
        entries = _zoom_entries(frame_count=30)
        cp = _cp_with_lens_params(entries, frame_count=200)
        curve = StabilizationManager.extract_focal_lengths(cp)
        assert len(curve) == 200
        assert curve[-1] is None

    def test_an_entry_without_a_focal_length_is_none(self):
        cp = _cp_with_lens_params(
            {0: LensParams(pixel_focal_length=1000.0)}, frame_count=1
        )
        assert StabilizationManager.extract_focal_lengths(cp) == [None]


class TestApplyFocalLengthSmoothing:
    def _manager(self, enabled, strength=0.5):
        return _StubManager(
            StabilizationParams(
                focal_length_smoothing_enabled=enabled,
                focal_length_smoothing_strength=strength,
            )
        )

    def test_disabled_clears_the_render_side_but_not_the_chart_side(self):
        """With smoothing off the compensation must be a clean 1.0, but the
        UI timeline still gets the raw curve when per-frame data exists."""
        cp = _cp_with_lens_params(_zoom_entries())
        manager = self._manager(enabled=False)
        manager._apply_focal_length_smoothing(cp)

        assert cp.focal_length_smoothing_enabled is False
        assert cp.focal_lengths == []
        assert cp.smoothed_focal_lengths == []
        assert len(manager.params.focal_lengths) == 60
        assert manager.params.smoothed_focal_lengths == []

    def test_enabled_fills_both_curves(self):
        cp = _cp_with_lens_params(_zoom_entries())
        manager = self._manager(enabled=True)
        manager._apply_focal_length_smoothing(cp)

        assert cp.focal_length_smoothing_enabled is True
        assert len(cp.focal_lengths) == 60
        assert len(cp.smoothed_focal_lengths) == 60
        # The chart side holds the RAW curve, which still has its stairs.
        raw = manager.params.focal_lengths
        assert len(set(raw)) > 1
        assert raw == pytest.approx([e.focal_length for e in
                                     _zoom_entries().values()])

    def test_the_dequantized_curve_is_smoother_than_the_raw_one(self):
        """That is the pass's whole job: the raw staircase has plateaus, the
        dequantized curve does not."""
        cp = _cp_with_lens_params(_zoom_entries())
        manager = self._manager(enabled=True)
        manager._apply_focal_length_smoothing(cp)

        raw = manager.params.focal_lengths
        dequantized = cp.focal_lengths
        flat_raw = sum(1 for a, b in zip(raw, raw[1:]) if a == b)
        flat_dequantized = sum(
            1 for a, b in zip(dequantized, dequantized[1:]) if a == b
        )
        assert flat_dequantized < flat_raw

    def test_no_per_frame_data_means_no_smoothing(self):
        """A fixed lens has nothing to smooth, and the flag has to say so."""
        cp = ComputeParams(frame_count=30, scaled_fps=30.0)
        manager = self._manager(enabled=True)
        manager._apply_focal_length_smoothing(cp)

        assert cp.focal_length_smoothing_enabled is False
        assert cp.focal_lengths == []

    def test_strength_is_passed_through_for_the_cache_key(self):
        cp = _cp_with_lens_params(_zoom_entries())
        cp.focal_length_smoothing_strength = 0.25
        manager = self._manager(enabled=True)
        manager._apply_focal_length_smoothing(cp)
        # The manager does not overwrite it; `_build_compute_params` copies it
        # from the stabilization params.
        assert cp.focal_length_smoothing_strength == 0.25

    def test_the_strength_knob_changes_the_smoothed_curve(self):
        curves = []
        for strength in (0.0, 0.5, 1.0):
            cp = _cp_with_lens_params(_zoom_entries())
            self._manager(enabled=True, strength=strength)._apply_focal_length_smoothing(cp)
            curves.append(cp.smoothed_focal_lengths)
        # Higher strength = heavier smoothing = the curve lags further behind
        # the raw staircase, so it sits lower after a rise.
        assert curves[0][-1] > curves[1][-1] > curves[2][-1]

    @pytest.mark.parametrize(
        "fps,expected_window",
        [(30.0, 15), (8.0, 5), (4.0, 5), (60.0, 30), (63.0, 32), (10.0, 5)],
    )
    def test_the_dequantize_window_follows_the_frame_rate(self, fps, expected_window):
        """Half a second of frames, floored at 5.

        Pinned by recomputing the Gaussian by hand with the expected width:
        the raw curve goes in, so a different window would land elsewhere.
        """
        cp = _cp_with_lens_params(_zoom_entries(fps=fps), fps=fps)
        self._manager(enabled=True)._apply_focal_length_smoothing(cp)
        assert cp.focal_lengths == pytest.approx(
            smooth_focal_lengths_gaussian(
                manager_curve(fps), 1.0, expected_window
            ),
            rel=1e-12,
        )


def manager_curve(fps):
    """The raw curve `_zoom_entries` produces at this frame rate."""
    return [value.focal_length for _, value in sorted(_zoom_entries(fps=fps).items())]


class TestZoomingChecksum:
    """The FOV cache key has to move with the smoothing controls.

    Otherwise re-rendering with a different strength reuses the previous
    fovs, and the render silently keeps the old zoom.
    """

    def test_enabling_changes_the_key(self):
        base = ComputeParams()
        base.focal_length_smoothing_enabled = False
        base.focal_length_smoothing_strength = 0.5
        other = ComputeParams()
        other.focal_length_smoothing_enabled = True
        other.focal_length_smoothing_strength = 0.5
        assert get_checksum(base) != get_checksum(other)

    def test_strength_changes_the_key(self):
        a = ComputeParams()
        a.focal_length_smoothing_strength = 0.2
        b = ComputeParams()
        b.focal_length_smoothing_strength = 0.8
        assert get_checksum(a) != get_checksum(b)

    def test_the_disable_case_is_still_covered(self):
        """Strength is copied onto the compute params whether or not
        smoothing is on, so it must reach the key in both states."""
        a = ComputeParams()
        a.focal_length_smoothing_enabled = False
        a.focal_length_smoothing_strength = 0.1
        b = ComputeParams()
        b.focal_length_smoothing_enabled = False
        b.focal_length_smoothing_strength = 0.9
        assert get_checksum(a) != get_checksum(b)

    def test_an_unrelated_field_does_not_change_the_key(self):
        a = ComputeParams()
        b = ComputeParams()
        b.video_speed = 2.0
        assert get_checksum(a) == get_checksum(b)


def _lens(fx=1000.0, fy=1000.0):
    return LensProfile(
        name="test",
        camera_matrix=[[fx, 0.0, 960.0], [0.0, fy, 540.0], [0.0, 0.0, 1.0]],
        distortion_coeffs=[0.0] * 4,
        calib_dimension={"w": 1920, "h": 1080},
        orig_dimension={"w": 1920, "h": 1080},
        distortion_model="opencv_fisheye",
        focal_length=18.0,
    )


class TestCompensationReachesTheRenderTransform:
    """End to end through `FrameTransform.at_timestamp`.

    With identity gyro and no rolling shutter the inverse transform's
    ``[0, 0]`` entry is exactly ``fov / camera_matrix[0, 0]``, so the ratio
    between a compensated and an uncompensated run is the compensation
    itself. That is what makes this a measurement of the render path rather
    than of the helper.
    """

    def _transform(self, params, frame=0):
        return FrameTransform.at_timestamp(params, frame * 1000.0 / 30.0, frame)

    def _params(self, enabled, curve, smoothed):
        identity = Quat64.identity()
        lens = _lens()
        return ComputeParams(
            width=1920,
            height=1080,
            output_width=1920,
            output_height=1080,
            frame_count=len(curve),
            scaled_fps=30.0,
            quaternions={0: identity, 2_000_000: identity},
            smoothed_quaternions={0: identity, 2_000_000: identity},
            camera_matrix=lens.get_camera_matrix((1920, 1080)),
            distortion_coeffs=lens.get_distortion_coeffs(),
            distortion_model_name="opencv_fisheye",
            lens=lens,
            focal_length=18.0,
            focal_lengths=list(curve),
            smoothed_focal_lengths=list(smoothed),
            focal_length_smoothing_enabled=enabled,
            frame_readout_time=0.0,
            fovs=[],
            minimal_fovs=[],
        )

    def test_the_inverse_transform_scales_with_the_compensation(self):
        curve = [18.0, 18.0, 18.0]
        smoothed = [18.0, 24.0, 18.0]
        plain = self._params(False, curve, smoothed)
        compensated = self._params(True, curve, smoothed)

        for frame in range(3):
            base = self._transform(plain, frame).matrices[0, 0]
            with_comp = self._transform(compensated, frame).matrices[0, 0]
            expected = _focal_length_fov_compensation(compensated, frame)
            assert with_comp / base == pytest.approx(expected, rel=1e-5)

    def test_a_longer_true_length_zooms_out(self):
        """Compensation above 1 makes new_k smaller, so the inverse
        transform gets bigger — the sampler reaches further out."""
        params_off = self._params(True, [18.0], [18.0])
        params_on = self._params(True, [18.0], [14.4])
        off = self._transform(params_off, 0).matrices[0, 0]
        on = self._transform(params_on, 0).matrices[0, 0]
        assert on > off

    def test_the_reported_focal_length_is_the_smoothed_one(self):
        """The "Focal length: X mm" readout should match what is rendered,
        not the raw metadata."""
        params = self._params(True, [18.0, 18.0], [18.0, 22.5])
        assert self._transform(params, 1).focal_length == pytest.approx(22.5)

    def test_the_reported_focal_length_falls_back_when_unknown(self):
        params = self._params(True, [18.0], [None])
        assert self._transform(params, 0).focal_length == pytest.approx(18.0)

    def test_without_smoothing_the_raw_length_is_reported(self):
        params = self._params(False, [18.0], [22.5])
        assert self._transform(params, 0).focal_length == pytest.approx(18.0)

    def test_scaled_k_keeps_the_raw_pixel_focal_length(self):
        """The compensation must not leak into the distortion forward
        projection — that is what makes it a digital zoom rather than a
        relabelling of the lens."""
        params = self._params(True, [18.0], [14.4])
        transform = self._transform(params, 0)
        assert transform.kernel_params.f[0] == pytest.approx(1000.0, rel=1e-6)
