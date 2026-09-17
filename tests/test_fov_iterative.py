"""The FOV polygon now goes through the real points family (gap item D-06).

``fov_iterative`` used to undistort its border polygon itself, in
``_undistort_points_simple``. That copy applied the rotation and the distortion
model but nothing else: no per-point IBIS/OIS displacement, no mesh or
focal-plane correction, no digital lens, and no ``lens_correction_amount < 1``
blend. It also rebuilt the intrinsics from ``params.camera_matrix`` by hand
rather than calling ``get_lens_data_at_timestamp``, so a zoom lens' per-frame
calibration never reached this path.

The polygon is the input to the crop decision, so all of that showed up as a
wrong crop. The tests below are written to catch exactly that: each piece of
data is fed in and the resulting FOV is required to move. Against the old
implementation they would all have been unchanged — which is the point, and is
why several of them assert a *specific* direction rather than just "different".
"""

from __future__ import annotations

import numpy as np
import pytest

from pygyroflow.stabilization import ComputeParams
from pygyroflow.types.quaternion import Quat64
from pygyroflow.zooming import calculate_fovs
from pygyroflow.zooming import fov_iterative as fov_mod

N = 50
FPS = 50.0


def _make(**overrides) -> ComputeParams:
    quats = {i * 20000: Quat64.identity() for i in range(N)}
    values = dict(
        width=1920,
        height=1080,
        output_width=1920,
        output_height=1080,
        frame_count=N,
        scaled_fps=FPS,
        scaled_duration_ms=1000.0,
        quaternions=dict(quats),
        smoothed_quaternions=dict(quats),
        fovs=[],
        fov_scale=1.0,
        camera_matrix=np.array(
            [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
        ),
        # Static zoom: one FOV for the whole clip, so a single number carries
        # the whole result and there is no smoothing to hide a difference.
        adaptive_zoom_window=-1.0,
    )
    values.update(overrides)
    return ComputeParams(**values)


def _timestamps(n: int = N) -> list[tuple[int, float]]:
    return [(i, i * 1000.0 / FPS) for i in range(n)]


def _fov(params: ComputeParams, n: int = N) -> float:
    values, _ = calculate_fovs(params, _timestamps(n))
    assert len(set(values)) == 1, "static zoom should give one value"
    return values[0]


def _stab_data(ibis_x: float, ibis_y: float):
    return {
        "crop_area": (0.0, 0.0, 1920.0, 1080.0),
        "pixel_pitch": (1.0, 1.0),
        "offset": 0.0,
        "ibis_spline": {
            "points": [[0.0, [ibis_x, ibis_y, 0.0]], [1080.0, [ibis_x, ibis_y, 0.0]]]
        },
        "ois_spline": {"points": [[0.0, [0.0, 0.0, 0.0]], [1080.0, [0.0, 0.0, 0.0]]]},
    }


def _focal_plane_mesh(slope: float) -> list:
    """A mesh buffer whose focal-plane table integrates to a y-dependent shift."""
    mesh = [0.0] * 120
    mesh[0] = 9.0  # offset to the focal-plane table
    mesh[1], mesh[2] = 3.0, 3.0
    mesh[3], mesh[4] = 1920.0, 1080.0
    mesh[5], mesh[6] = 0.0, 0.0
    mesh[7], mesh[8] = 1920.0, 1080.0
    mesh[9] = 1.0  # the table is present
    for index in range(8):
        mesh[9 + 4 + index * 2 + 0] = slope
        mesh[9 + 4 + index * 2 + 1] = slope / 2.0
    return mesh


class TestThePolygonUsesTheUpstreamPointsFamily:
    def test_the_local_copy_is_gone(self):
        """The reimplementation is what this gap was; leaving it would be a
        second implementation to keep in sync."""
        assert not hasattr(fov_mod, "_undistort_points_simple")

    def test_the_polygon_is_undistorted_by_the_shared_function(self, monkeypatch):
        calls = []
        real = fov_mod.undistort_points_with_rolling_shutter

        def spy(points, ts, frame, params, lens_correction_amount, use_fovs):
            calls.append(
                {
                    "count": len(points),
                    "frame": frame,
                    "lens_correction_amount": lens_correction_amount,
                    "use_fovs": use_fovs,
                }
            )
            return real(points, ts, frame, params, lens_correction_amount, use_fovs)

        monkeypatch.setattr(fov_mod, "undistort_points_with_rolling_shutter", spy)
        _fov(_make())

        assert calls, "the polygon was not undistorted through the shared function"
        # Upstream calls it with use_fovs=false: the FOVs are the thing being
        # computed, so they cannot also be an input.
        assert all(call["use_fovs"] is False for call in calls)
        # The initial polygon is 31x31 border points -> 2*30 per axis in the
        # layout _points_around_rect produces.
        assert calls[0]["count"] == 120
        # The refinement pass interpolates 30 steps between 3 points:
        # 31 * 3 - 30 = 63.
        assert any(call["count"] == 63 for call in calls)

    def test_the_contraction_loop_is_the_only_thing_after_it(self, monkeypatch):
        """With the undistortion stubbed, the result is a closed-form number.

        Pins the loop itself: four passes at most, and the FOV is twice the
        half-width of the inscribed rectangle over the output width. A
        polygon that is a fixed square of half-size 100 around the centre must
        therefore give 2 * (100 / inv_aspect) / 1920, with
        inv_aspect = 1080 / 1920 when the zoom is estimated at input size.
        """
        half = 100.0
        inv_aspect = 1080.0 / 1920.0

        def stub(points, ts, frame, params, lens_correction_amount, use_fovs):
            del ts, frame, params, lens_correction_amount, use_fovs
            return [(960.0 + half, 540.0 + half) for _ in points]

        monkeypatch.setattr(fov_mod, "undistort_points_with_rolling_shutter", stub)
        expected = 2.0 * (half / inv_aspect) / 1920.0
        assert _fov(_make()) == pytest.approx(expected, rel=1e-12)


class TestTheDataThatNowReachesTheCrop:
    """Each of these was silently dropped by the old local implementation."""

    def test_ibis_displacement_moves_the_crop(self):
        without = _fov(_make())
        with_ibis = _fov(_make(camera_stab_data=[_stab_data(60.0, 40.0)] * N))
        assert with_ibis != pytest.approx(without, rel=1e-6)
        # A displacement that pushes the polygon inward needs a *smaller*
        # scale factor to keep the frame covered: fov < 1 means cropping in.
        assert with_ibis < without

    def test_a_larger_ibis_displacement_moves_it_further(self):
        small = _fov(_make(camera_stab_data=[_stab_data(20.0, 10.0)] * N))
        large = _fov(_make(camera_stab_data=[_stab_data(80.0, 50.0)] * N))
        assert large < small

    def test_mesh_correction_moves_the_crop(self):
        without = _fov(_make())
        mesh = _focal_plane_mesh(5.0)
        with_mesh = _fov(_make(mesh_correction=[[mesh, mesh]] * N))
        assert with_mesh != pytest.approx(without, rel=1e-6)

    def test_the_digital_lens_moves_the_crop(self):
        from pygyroflow.stabilization.distortion_models import from_name

        without = _fov(_make())
        with_lens = _fov(_make(digital_lens=from_name("gopro_superview")))
        assert with_lens != pytest.approx(without, rel=1e-6)

    def test_the_optical_model_is_honoured(self):
        """Not new data, but the check that the model is not bypassed."""
        plain = _fov(_make())
        fisheye = _fov(
            _make(distortion_coeffs=[-0.2, 0.05, 0.0, 0.0] + [0.0] * 8)
        )
        assert fisheye != pytest.approx(plain, rel=1e-6)

    def test_a_non_zero_zoom_centre_moves_the_crop(self):
        centred = _fov(_make())
        offset = _fov(_make(adaptive_zoom_center_offset=(0.15, 0.05)))
        assert offset != pytest.approx(centred, rel=1e-6)


class TestTheWorkingCopyCarriesEverything:
    """The copy handed to ``FovIterative`` is a full clone plus overrides.

    It used to be built field by field, and anything nobody remembered to list
    silently fell back to its default. The lens maps are the newest family to
    have been dropped that way.
    """

    @staticmethod
    def _forwarded(compute_params, timestamps):
        captured = {}
        real = fov_mod.FovIterative

        class _Spy(real):  # type: ignore[misc, valid-type]
            def __init__(self, params, org_output_size):
                captured["params"] = params
                super().__init__(params, org_output_size)

        import pygyroflow.zooming as zooming_mod

        zooming_mod.FovIterative = _Spy
        try:
            calculate_fovs(compute_params, timestamps)
        finally:
            zooming_mod.FovIterative = real
        return captured["params"]

    def test_the_per_frame_lens_data_survives(self):
        from pygyroflow.stabilization.distortion_models import from_name
        from pygyroflow.util import ClosestMap

        class _Lens:
            """A stand-in for a lens profile with no interpolation table.

            `_get_lens_data_at_timestamp` asks the profile for an interpolated
            one when `lens_positions` is populated; returning None is the
            honest answer here, since this test is about whether the object is
            *carried*, not about what it contains. `distortion_coeffs` is read
            on the fallback path to decide whether the profile's own
            coefficients win over the per-frame ones.
            """

            input_horizontal_stretch = 1.0
            input_vertical_stretch = 1.0
            distortion_coeffs: list = []

            def get_interpolated_profile_at(self, position):
                del position
                return None

        mesh = _focal_plane_mesh(1.0)
        params = _make(
            lens_positions=ClosestMap({0: 35.0, 500_000: 70.0}),
            lens_params=ClosestMap(),
            mesh_correction=[[mesh, mesh]] * 2,
            camera_stab_data=[_stab_data(1.0, 1.0)] * 2,
            digital_lens=from_name("gopro_hyperview"),
            digital_lens_params=[1.0, 1.0, 0.0, 0.0],
            lens=_Lens(),
        )
        forwarded = self._forwarded(params, _timestamps())
        for name in (
            "lens",
            "lens_positions",
            "lens_params",
            "mesh_correction",
            "camera_stab_data",
            "digital_lens",
            "digital_lens_params",
        ):
            assert getattr(forwarded, name) is getattr(params, name), name

    def test_the_estimate_is_made_at_input_size_with_no_prior_zoom(self):
        params = _make(fovs=[1.3] * N, minimal_fovs=[1.3] * N, fov_scale=1.3)
        forwarded = self._forwarded(params, _timestamps())
        assert forwarded.output_width == params.width
        assert forwarded.output_height == params.height
        assert forwarded.fovs == []
        assert forwarded.minimal_fovs == []
        assert forwarded.fov_scale == 1.0


class TestTheKeyframedInputs:
    """Upstream looks three parameters up per frame, and so does this now.

    The values used to be read once from the static params, so keyframing any
    of them changed nothing.
    """

    @staticmethod
    def _captured_keyframe_values(params: ComputeParams):
        seen = []
        real = fov_mod.FovIterative._find_fov

        def spy(self, rect, ts, frame, center, keyframe_values):
            seen.append(keyframe_values)
            return real(self, rect, ts, frame, center, keyframe_values)

        original = fov_mod.FovIterative._find_fov
        fov_mod.FovIterative._find_fov = spy
        try:
            calculate_fovs(params, _timestamps(4), None) if False else None
            from pygyroflow.zooming import ZoomMethod

            calculate_fovs(params, _timestamps(4), ZoomMethod.EnvelopeFollower)
        finally:
            fov_mod.FovIterative._find_fov = original
        return seen

    def _keyframed(self, key, values):
        from pygyroflow.keyframes import KeyframeManager

        params = _make()
        params.keyframes = KeyframeManager()
        for timestamp_us, value in values:
            params.keyframes.set_keyframe(key, timestamp_us, value)
        # Static zoom keeps the result to one number; the spy sees the inputs.
        params.adaptive_zoom_window = -1.0
        return params

    def test_zoom_center_x_is_read_per_frame(self):
        from pygyroflow.keyframes import KeyframeType

        params = self._keyframed(
            KeyframeType.ZoomingCenterX, [(0, 0.0), (60_000, 0.3)]
        )
        seen = self._captured_keyframe_values(params)
        assert seen, "no frames were evaluated"
        assert seen[0][0] == pytest.approx(0.0)
        assert seen[-1][0] == pytest.approx(0.3, abs=0.02)

    def test_zoom_center_y_is_read_per_frame(self):
        from pygyroflow.keyframes import KeyframeType

        params = self._keyframed(
            KeyframeType.ZoomingCenterY, [(0, 0.0), (60_000, -0.2)]
        )
        seen = self._captured_keyframe_values(params)
        assert seen[0][1] == pytest.approx(0.0)
        assert seen[-1][1] == pytest.approx(-0.2, abs=0.02)

    def test_lens_correction_strength_is_read_per_frame(self):
        from pygyroflow.keyframes import KeyframeType

        params = self._keyframed(
            KeyframeType.LensCorrectionStrength, [(0, 1.0), (60_000, 0.2)]
        )
        seen = self._captured_keyframe_values(params)
        assert seen[0][2] == pytest.approx(1.0)
        assert seen[-1][2] == pytest.approx(0.2, abs=0.02)

    def test_without_keyframes_every_frame_gets_the_static_triple(self):
        params = _make(adaptive_zoom_center_offset=(0.1, -0.05))
        seen = self._captured_keyframe_values(params)
        assert seen
        assert all(
            value
            == pytest.approx(
                (0.1, -0.05, params.lens_correction_amount)
            )
            for value in seen
        )

    def test_a_keyframed_parameter_actually_changes_the_fov(self):
        from pygyroflow.keyframes import KeyframeType
        from pygyroflow.zooming import ZoomMethod

        plain = _make()
        # Keyframing the centre at the same value it already has must not
        # change anything — that isolates "the lookup happened" from "the
        # value moved".
        same = self._keyframed(KeyframeType.ZoomingCenterX, [(0, 0.0)])
        moved = self._keyframed(KeyframeType.ZoomingCenterX, [(0, 0.25)])

        a, _ = calculate_fovs(plain, _timestamps(), ZoomMethod.EnvelopeFollower)
        b, _ = calculate_fovs(same, _timestamps(), ZoomMethod.EnvelopeFollower)
        c, _ = calculate_fovs(moved, _timestamps(), ZoomMethod.EnvelopeFollower)
        assert b == pytest.approx(a)
        assert c != pytest.approx(a)


class TestNearestEdge:
    def test_the_aspect_comes_from_the_output_not_from_the_running_rect(self):
        """Upstream reads ``self.output_inv_aspect``; deriving it from the
        current best rectangle is only *approximately* the same, because the
        rectangle's ratio drifts by rounding each time it is recomputed."""
        params = _make()
        estimator = fov_mod.FovIterative(params, (params.output_width, params.output_height))
        aspect = estimator.output_inv_aspect

        center = (960.0, 540.0)
        polygon = [(1060.0, 620.0)]
        # A rectangle whose ratio is deliberately not the output aspect.
        wrong = (1000.0, 400.0)
        index, rect = estimator._nearest_edge(polygon, center, wrong)

        # ap = (100, 80). `80 > 100 * aspect` holds for the real aspect
        # (0.5625), so the height governs and the width follows from it.
        assert index == 0
        assert rect == pytest.approx((80.0 / aspect, 80.0))
        # Had the ratio been taken from `wrong` (0.4) the width would have come
        # out as 200 instead of 142.2 — a 40% error in the crop, twice over.
        assert rect[0] != pytest.approx(80.0 / (wrong[1] / wrong[0]))

    def test_points_exactly_on_the_rectangle_do_not_count(self):
        """Upstream's comparisons are strict (``<``, ``>``), so a point level
        with the current edge is not nearer than it."""
        params = _make()
        estimator = fov_mod.FovIterative(params, (params.output_width, params.output_height))
        initial = (100.0, 100.0 * estimator.output_inv_aspect)
        on_edge = [(1060.0, 540.0 + initial[1])]
        index, rect = estimator._nearest_edge(on_edge, (960.0, 540.0), initial)
        assert index is None
        assert rect == pytest.approx(initial)
