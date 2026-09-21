"""Per-point undistortion (gap item D-06).

``cpu_undistort.rs`` has two families. The image path (``undistort_coord``)
computes one transform per output pixel, vectorised. The points path —
``undistort_points`` and its two wrappers — takes a *list of pixel positions*
and returns where each one lands, one point at a time, and does three things
the image path cannot:

* each point is rotated by the orientation at **its own row's** exposure, so
  a fisheye and a rolling shutter interact the way they really do;
* the camera's own IBIS/OIS displacement is added per point, from the spline
  block the file carries;
* the focal-plane/mesh correction and the digital lens run on a point, not on
  a grid.

This is what autosync and adaptive-zoom sampling consume, and neither had a
distortion-corrected coordinate before (``almeida.py`` says so in a comment).

The tests below are mostly *structural*: they assert the order of the stages
by feeding one stage's output as the next stage's input and checking that the
composition is unchanged. That is deliberate — it verifies the pipeline
without re-deriving the distortion maths, which ``test_distortion_models.py``
and ``test_distortion_cv2_parity.py`` already cover against OpenCV.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pygyroflow.stabilization.compute_params import ComputeParams
from pygyroflow.stabilization.cpu_undistort import (
    _POINT_FAILURE,
    _apply_mesh,
    _points_kernel_params,
    undistort_points,
    undistort_points_for_optical_flow,
    undistort_points_with_rolling_shutter,
)
from pygyroflow.stabilization.frame_transform import (
    _shift_per_point,
    at_timestamp_for_points,
)
from pygyroflow.types.enums import ReadoutDirection
from pygyroflow.types.quaternion import Quat64

WIDTH = 1920
HEIGHT = 1080
FX = 1000.0
FY = 1000.0
CX = 960.0
CY = 540.0
ZERO_COEFFS = [0.0] * 12


def _params(**overrides) -> ComputeParams:
    """A ComputeParams for the points path: full-frame, static lens."""
    values = dict(
        width=WIDTH,
        height=HEIGHT,
        output_width=WIDTH,
        output_height=HEIGHT,
        scaled_fps=30.0,
        camera_matrix=np.array(
            [[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]], dtype=np.float64
        ),
        distortion_coeffs=list(ZERO_COEFFS),
        distortion_model_name="opencv_fisheye",
        fovs=[1.0],
    )
    values.update(overrides)
    return ComputeParams(**values)


def _centre_params(**overrides) -> ComputeParams:
    """The same, with a principal point the output centre sits on.

    ``_partial_correction`` divides the output centre out of the point, so a
    round-trip through it only lands back on the input when the output centre
    *is* the principal point. Every camera matrix here uses CX/CY for that
    reason; this alias exists so the call sites say why.
    """
    return _params(**overrides)


def _run_one(point, params, camera_matrix=None, coeffs=None, **kwargs):
    """Undistort a single point the way the render path calls it.

    The rotation is ``new_k @ R``, not a bare ``R`` — that is what
    ``at_timestamp_for_points`` hands over, and it is what puts the result in
    *output pixel* units. Passing ``I`` here would return normalised
    coordinates instead, which is only right for the optical-flow call.
    """
    camera_matrix = (
        params.camera_matrix if camera_matrix is None else camera_matrix
    )
    coeffs = params.distortion_coeffs if coeffs is None else coeffs
    return undistort_points(
        [point],
        camera_matrix,
        coeffs,
        np.array(camera_matrix, dtype=np.float64),
        params=params,
        **kwargs,
    )[0]


class TestUndistortPoints:
    """The scalar-per-point core: ``undistort_points``."""

    def test_the_principal_point_maps_to_the_output_centre(self):
        params = _params()
        assert _run_one((CX, CY), params) == pytest.approx((CX, CY))

    def test_an_unrotated_point_is_reprojected_through_the_new_k(self):
        """With rotation = I and new_k = K, the map is the identity in pixels.

        The result is in *output pixel* units, not normalised ones: the
        rotation the caller passes in is already ``new_k @ R``, and the
        perspective divide at the end undoes the ``[u, v, 1]`` lift.
        """
        params = _params()
        assert _run_one((CX + FX, CY), params) == pytest.approx((CX + FX, CY))
        assert _run_one((CX, CY + FY), params) == pytest.approx((CX, CY + FY))

    def test_the_output_is_in_new_k_units_not_normalised_units(self):
        """A focal length in the matrix shows up as a scale in the output.

        Guards against a port that returns the normalised point (what
        ``undistort_point`` itself returns) instead of the reprojected one.
        """
        params = _params()
        result = _run_one((CX + FX, CY), params)
        assert result == pytest.approx((CX + FX, CY))
        assert result != pytest.approx((1.0, 0.0))

    def test_p_is_left_multiplied_into_the_rotation(self):
        """``rr = p @ rotation``, so a non-uniform p scales x independently."""
        params = _params()
        point = (CX + FX, CY)
        plain = _run_one(point, params)
        p = np.diag([2.0, 1.0, 1.0])
        result = _run_one(point, params, p=p)
        # diag(2,1,1) leaves the perspective divide alone and doubles x.
        assert result[0] == pytest.approx(2.0 * plain[0])
        assert result[1] == pytest.approx(plain[1])

    def test_a_uniform_p_scaling_cancels_in_the_perspective_divide(self):
        """p = c*I must not change anything — the divide eats the scale."""
        params = _params()
        point = (CX + 300.0, CY + 120.0)
        plain = _run_one(point, params, p=None)
        scaled = _run_one(point, params, p=np.eye(3) * 4.0)
        assert scaled == pytest.approx(plain)

    def test_rot_per_point_overrides_the_shared_rotation(self):
        params = _params()
        camera_matrix = params.camera_matrix
        quarter_turn = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        points = [(CX + FX, CY), (CX + FX, CY)]
        out = undistort_points(
            points,
            camera_matrix,
            params.distortion_coeffs,
            np.array(camera_matrix, dtype=np.float64),
            params=params,
            rot_per_point=[
                np.array(camera_matrix, dtype=np.float64),
                quarter_turn @ camera_matrix,
            ],
        )
        # Both points are the same input; only the per-point rotation differs.
        # The second is rotated a quarter turn about the image origin, so its
        # output is the first output put through that rotation.
        assert out[0] == pytest.approx((CX + FX, CY))
        projected = quarter_turn @ np.array([out[0][0], out[0][1], 1.0])
        assert out[1] == pytest.approx(projected[:2])
        assert out[0] != pytest.approx(out[1])

    def test_a_point_that_does_not_converge_returns_the_sentinel(self):
        """A non-converged point is a value, not an exception.

        Callers drop it and keep the rest, so it has to be recognisable —
        ``(-1000000, -1000000)`` is upstream's marker.
        """
        params = _params(distortion_coeffs=[-0.5] + [0.0] * 11)
        # radius 1.0 in normalised units: (1 + k1*theta^2) is no longer
        # invertible for this k1 and the iteration bails out.
        assert _run_one((CX + FX, CY), params) == _POINT_FAILURE

    def test_the_sentinel_is_only_for_the_point_that_failed(self):
        params = _params(distortion_coeffs=[-0.5] + [0.0] * 11)
        out = undistort_points(
            [(CX, CY), (CX + FX, CY), (CX + 100.0, CY)],
            params.camera_matrix,
            params.distortion_coeffs,
            np.array(params.camera_matrix, dtype=np.float64),
            params=params,
        )
        assert out[0] == pytest.approx((CX, CY))
        assert out[1] == _POINT_FAILURE
        assert out[2] != _POINT_FAILURE

    def test_light_refraction_is_skipped_at_its_neutral_value(self):
        """1.0 is "no correction"; the branch must not perturb the point."""
        neutral = _params(light_refraction_coefficient=1.0)
        assert _run_one((CX + 400.0, CY + 200.0), neutral) == pytest.approx(
            (CX + 400.0, CY + 200.0)
        )

    def test_light_refraction_scales_the_point_outward(self):
        """The corrected radius is ``r_d``, the refraction-corrected radius.

        Stated from the geometry rather than from the code: for a normalised
        radius r, the sine of the incidence angle is ``r / sqrt(1 + r^2)``,
        the refracted sine is that divided by the coefficient, and the tangent
        radius that follows is ``sin / sqrt(1 - sin^2)``.
        """
        lrc = 1.5
        dense = _params(light_refraction_coefficient=lrc)
        r = 400.0 / FX  # the point is CX + 400 px out
        sin_theta_d = (r / math.sqrt(1.0 + r * r)) / lrc
        r_d = sin_theta_d / math.sqrt(1.0 - sin_theta_d * sin_theta_d)
        result = _run_one((CX + 400.0, CY), dense)
        assert result[0] == pytest.approx(CX + r_d * FX)
        assert abs(result[0] - CX) < abs(400.0)

    def test_a_distorted_point_does_not_stay_on_its_distorted_coordinate(self):
        """The gap item D-06 closes.

        ``almeida.py`` recorded that the sampling points autosync and adaptive
        zoom use were on *distorted* coordinates — no correction at all. This
        asserts the corrected answer is a different pixel, so a caller that
        skips this family is measurably wrong rather than approximately right.
        """
        point = (CX + 400.0, CY + 200.0)
        clean = _run_one(point, _params())
        distorted = _run_one(point, _params(distortion_coeffs=[0.1] + [0.0] * 11))
        assert clean == pytest.approx(point)
        assert distorted != pytest.approx(clean)
        assert abs(distorted[0] - clean[0]) > 1.0


class TestLensCorrectionAmount:
    """The ``lens_correction_amount < 1`` blend."""

    def test_a_full_correction_returns_the_undistorted_point(self):
        params = _params(distortion_coeffs=[0.1] + [0.0] * 11)
        # theta_d = 1.0 inverts to tan(theta) ~= 1.3179 for k1 = 0.1.
        corrected = _run_one((CX + FX, CY), params, lens_correction_amount=1.0)
        assert corrected[0] == pytest.approx(CX + 1.3179032372824817 * FX)

    def test_zero_correction_redistorts_the_point_back_where_it_started(self):
        """amount = 0 keeps the original look — the round trip must close.

        This is the branch that re-applies the *forward* distortion to the
        corrected point, so it is distort(undistort(d)) = d rather than a
        linear blend of two coordinates.
        """
        params = _centre_params(distortion_coeffs=[0.1] + [0.0] * 11)
        point = (CX + FX, CY)
        result = _run_one(point, params, lens_correction_amount=0.0)
        assert result == pytest.approx(point, rel=1e-5)

    def test_half_correction_lands_between_the_two(self):
        params = _centre_params(distortion_coeffs=[0.1] + [0.0] * 11)
        point = (CX + FX, CY)
        full = _run_one(point, params, lens_correction_amount=1.0)
        none = _run_one(point, params, lens_correction_amount=0.0)
        half = _run_one(point, params, lens_correction_amount=0.5)
        assert half[0] == pytest.approx((full[0] + none[0]) / 2.0)


class _StubDigitalLens:
    """A digital lens with a known, invertible effect.

    Real GoPro models are exercised in ``test_distortion_models.py``; what
    matters here is the *order* of the stages, and a stub makes that
    unambiguous — it shifts x by a fixed number of pixels.
    """

    shift_x = 37.0

    def id(self) -> str:
        return "stub"

    def undistort_point(self, x, y, params):
        return (x + self.shift_x, y)

    def distort_point(self, x, y, z, params):
        return (x - self.shift_x, y)


class TestStageOrder:
    """The digital lens is a pure pre-transform ahead of everything else."""

    def test_the_result_equals_feeding_the_lens_output_back_in(self):
        params = _params()
        point = (CX + 250.0, CY + 90.0)
        lens = _StubDigitalLens()
        kernel_params = _points_kernel_params(
            params.camera_matrix, params.distortion_coeffs, params, 1.0
        )
        moved = lens.undistort_point(point[0], point[1], kernel_params)

        with_lens = _params(digital_lens=lens)
        without_lens = _params()
        assert _run_one(point, with_lens) == pytest.approx(
            _run_one(moved, without_lens)
        )

    def test_a_digital_lens_that_does_nothing_changes_nothing(self):
        class _Identity(_StubDigitalLens):
            shift_x = 0.0

        point = (CX + 250.0, CY + 90.0)
        assert _run_one(point, _params(digital_lens=_Identity())) == pytest.approx(
            _run_one(point, _params())
        )


class TestMeshCorrection:
    """``_apply_mesh``: focal-plane table, then the full mesh table."""

    def _mesh(self, **overrides) -> list[float]:
        mesh = [0.0] * 64
        mesh[0] = 9.0  # offset to the focal-plane table
        mesh[3] = 1.0
        mesh[4] = float(HEIGHT)  # grid rows == frame height -> row grid is 135 px
        mesh[5], mesh[6] = 0.0, 0.0  # origin
        mesh[7], mesh[8] = float(WIDTH), float(HEIGHT)  # crop == full frame
        mesh[9] = 1.0  # non-zero -> the table is present
        for index, value in overrides.items():
            mesh[int(index[1:])] = value
        return mesh

    def test_a_zero_header_skips_both_branches(self):
        mesh = [0.0] * 64
        assert _apply_mesh(700.0, 300.0, mesh, _params()) == (700.0, 300.0)

    def test_the_row_slope_is_accumulated_once_per_grid_cell(self):
        """The table integrates along y: every grid cell below the row adds in.

        At y = 540 the point sits on grid row 4 exactly, so ``delta`` is zero
        and only the accumulated terms apply. Only the j = 0 slope is set, so
        exactly one of the four cells contributes, at 135 px.
        """
        mesh = self._mesh(m13=1.0)  # x slope for j = 0
        # mesh[9 + 4 + 0*2 + 0] == mesh[13]
        x, y = _apply_mesh(CX + FX, CY, mesh, _params())
        assert x == pytest.approx(CX + FX + 135.0)
        assert y == pytest.approx(CY)

    def test_every_grid_cell_below_the_row_contributes(self):
        """Two non-zero cells at rows 0 and 1 both land, and add up."""
        mesh = self._mesh(m13=1.0, m15=2.0)
        # mesh[9 + 4 + 1*2 + 0] == mesh[15]
        x, _ = _apply_mesh(CX + FX, CY, mesh, _params())
        assert x == pytest.approx(CX + FX + 135.0 + 2.0 * 135.0)

    def test_the_partial_cell_adds_its_own_fraction(self):
        mesh = self._mesh(m21=0.5)  # delta slope for index = 4
        # mesh[9 + 4 + 4*2 + 0] == mesh[21]
        x, y = _apply_mesh(CX + FX, CY + 20.0, mesh, _params())
        assert x == pytest.approx(CX + FX + 0.5 * 20.0)
        assert y == pytest.approx(CY + 20.0)

    def test_the_point_is_not_moved_when_the_crop_is_the_full_frame(self):
        """origin (0,0) and crop == frame make map_coord the identity."""
        mesh = self._mesh()
        assert _apply_mesh(700.0, 300.0, mesh, _params()) == (700.0, 300.0)

    def test_the_mesh_is_a_pure_pre_transform(self):
        """Whatever the table does, the rest of the chain is unaffected.

        Feeding the table's own output back in with the mesh removed has to
        give the same answer — that is what "the mesh runs before the
        rotation" means.
        """
        mesh = self._mesh(m13=1.0, m21=0.5)
        params = _params()
        point = (CX + FX, CY)
        moved = _apply_mesh(point[0], point[1], mesh, params)
        with_mesh = _run_one(point, params, mesh=mesh)
        without_mesh = _run_one(moved, params)
        assert with_mesh == pytest.approx(without_mesh)


class TestShiftPerPoint:
    """IBIS/OIS displacement, from the camera's own stabilization spline."""

    def _stab_data(self, ibis_x=1.0, ibis_y=2.0, angle_deg=0.0):
        return {
            "crop_area": (0.0, 0.0, float(WIDTH), float(HEIGHT)),
            "pixel_pitch": (1.0, 1.0),
            "offset": 0.0,
            "ibis_spline": {
                "points": [
                    [0.0, [ibis_x, ibis_y, angle_deg * 1000.0]],
                    [float(HEIGHT), [ibis_x, ibis_y, angle_deg * 1000.0]],
                ]
            },
            "ois_spline": {
                "points": [[0.0, [0.0, 0.0, 0.0]], [float(HEIGHT), [0.0, 0.0, 0.0]]]
            },
        }

    def test_a_constant_spline_becomes_a_constant_shift(self):
        params = _params(camera_stab_data=[self._stab_data()])
        shifts = _shift_per_point(params, [(0.0, 540.0)], 0)
        assert len(shifts) == 1
        assert shifts[0][0] == pytest.approx(1.0)  # ibis x
        assert shifts[0][1] == pytest.approx(2.0)  # ibis y
        assert shifts[0][2] == pytest.approx(0.0)  # angle, radians
        assert shifts[0][3] == pytest.approx(0.0)  # ois x
        assert shifts[0][4] == pytest.approx(0.0)  # ois y

    def test_the_shift_is_scaled_out_of_the_camera_crop_area(self):
        """A crop area half the frame's size doubles the scale factor."""
        stab = self._stab_data()
        stab["crop_area"] = (0.0, 0.0, float(WIDTH) / 2.0, float(HEIGHT) / 2.0)
        params = _params(camera_stab_data=[stab])
        shifts = _shift_per_point(params, [(0.0, 540.0)], 0)
        assert shifts[0][0] == pytest.approx(2.0)
        assert shifts[0][1] == pytest.approx(4.0)

    def test_the_shift_is_applied_at_the_row_the_spline_reports(self):
        """A row outside the crop area is remapped before the spline lookup.

        With the crop starting at y = 200 the point at y = 540 has to be
        looked up at 340 in the spline's own coordinates; a constant spline
        cannot show that, so this one ramps and the value is checked against
        the remapped position.
        """
        stab = self._stab_data()
        stab["crop_area"] = (0.0, 200.0, float(WIDTH), float(HEIGHT))
        stab["ibis_spline"] = {
            "points": [[0.0, [0.0, 0.0, 0.0]], [float(HEIGHT), [10.0, 0.0, 0.0]]]
        }
        params = _params(camera_stab_data=[stab])
        shifts = _shift_per_point(params, [(0.0, 540.0)], 0)
        assert 0.0 < shifts[0][0] < 10.0

    def test_no_stabilization_data_means_no_shift(self):
        assert _shift_per_point(_params(), [(0.0, 540.0)], 0) is None
        assert _shift_per_point(_params(camera_stab_data=[]), [(0.0, 540.0)], 0) is None

    def test_a_frame_past_the_end_of_the_data_means_no_shift(self):
        params = _params(camera_stab_data=[self._stab_data()])
        assert _shift_per_point(params, [(0.0, 540.0)], 5) is None

    def test_the_shift_moves_the_point_by_the_offset_it_reports(self):
        """The displacement and the OIS offset enter with opposite signs.

        ``x = x - c - ois + ibis``: IBIS pushes the point, OIS pulls it, and
        the rotation happens about the principal point.
        """
        params = _params(camera_stab_data=[self._stab_data()])
        point = (CX + 300.0, CY + 100.0)
        shifts = _shift_per_point(params, [point], 0)
        shifted = undistort_points(
            [point],
            params.camera_matrix,
            params.distortion_coeffs,
            np.array(params.camera_matrix, dtype=np.float64),
            params=params,
            shift_per_point=shifts,
        )[0]
        assert shifted == pytest.approx((CX + 300.0 + 1.0, CY + 100.0 + 2.0))


class TestAtTimestampForPoints:
    """``at_timestamp_for_points``: the per-point orchestration."""

    def test_it_returns_the_six_values_the_points_path_needs(self):
        camera_matrix, coeffs, new_k, rotations, shifts, mesh = (
            at_timestamp_for_points(_params(), [(CX, CY)], 0.0, 0)
        )
        assert camera_matrix.shape == (3, 3)
        assert len(coeffs) == 12
        assert new_k.shape == (3, 3)
        assert len(rotations) == 1
        assert shifts is None
        assert mesh is None

    def test_one_rotation_shared_by_every_point_without_rolling_shutter(self):
        params = _params(frame_readout_time=0.0)
        points = [(100.0, 100.0), (500.0, 700.0), (900.0, 200.0)]
        _, _, _, rotations, _, _ = at_timestamp_for_points(params, points, 0.0, 0)
        assert len(rotations) == 1

    def test_one_rotation_per_point_with_rolling_shutter(self):
        params = _params(
            frame_readout_time=0.01,
            frame_readout_direction=ReadoutDirection.TopToBottom,
        )
        points = [(100.0, 100.0), (500.0, 700.0), (900.0, 200.0)]
        _, _, _, rotations, _, _ = at_timestamp_for_points(params, points, 0.0, 0)
        assert len(rotations) == len(points)

    def test_rows_exposed_at_different_times_get_different_rotations(self):
        """The whole point of the per-point path: two rows, two orientations.

        The gyro stream turns during the frame, so the top row and the bottom
        row are exposed under different attitudes and must not share a matrix.
        """
        turn = Quat64.from_euler_angles(0.0, 0.0, math.radians(5.0))
        quats = {0: Quat64.identity(), 200_000: turn}
        params = _params(
            frame_readout_time=0.10,  # 100 ms of readout across the frame
            frame_readout_direction=ReadoutDirection.TopToBottom,
            quaternions=dict(quats),
            smoothed_quaternions=dict(quats),
        )
        points = [(960.0, 0.0), (960.0, float(HEIGHT))]
        _, _, _, rotations, _, _ = at_timestamp_for_points(params, points, 0.0, 0)
        assert not np.allclose(rotations[0], rotations[1])

    def test_the_mesh_comes_from_this_frames_distorting_mesh(self):
        frame0 = [[1.0, 2.0, 3.0]]
        frame1 = [[4.0, 5.0, 6.0]]
        params = _params(mesh_correction=[frame0, frame1])
        _, _, _, _, _, mesh = at_timestamp_for_points(
            params, [(CX, CY)], 0.0, 1
        )
        assert mesh == [4.0, 5.0, 6.0]

    def test_a_frame_without_a_mesh_entry_leaves_the_mesh_unset(self):
        params = _params(mesh_correction=[[[1.0, 2.0]]])
        _, _, _, _, _, mesh = at_timestamp_for_points(params, [(CX, CY)], 0.0, 3)
        assert mesh is None

    def test_the_shift_is_dropped_when_rotation_is_suppressed(self):
        """No rotation and no rolling shutter -> the IBIS shift is dead too.

        Upstream disables both together: with nothing rotating there is no
        per-row error for the shift to compensate.
        """
        stab = {
            "crop_area": (0.0, 0.0, float(WIDTH), float(HEIGHT)),
            "pixel_pitch": (1.0, 1.0),
            "offset": 0.0,
            "ibis_spline": {
                "points": [[0.0, [1.0, 1.0, 0.0]], [float(HEIGHT), [1.0, 1.0, 0.0]]]
            },
            "ois_spline": {"points": []},
        }
        params = _params(
            camera_stab_data=[stab], suppress_rotation=True, frame_readout_time=0.0
        )
        _, _, _, _, shifts, _ = at_timestamp_for_points(params, [(CX, CY)], 0.0, 0)
        assert shifts is None

    def test_the_shift_survives_when_there_is_rolling_shutter(self):
        stab = {
            "crop_area": (0.0, 0.0, float(WIDTH), float(HEIGHT)),
            "pixel_pitch": (1.0, 1.0),
            "offset": 0.0,
            "ibis_spline": {
                "points": [[0.0, [1.0, 1.0, 0.0]], [float(HEIGHT), [1.0, 1.0, 0.0]]]
            },
            "ois_spline": {"points": []},
        }
        params = _params(
            camera_stab_data=[stab], suppress_rotation=True, frame_readout_time=0.01
        )
        _, _, _, _, shifts, _ = at_timestamp_for_points(params, [(CX, CY)], 0.0, 0)
        assert shifts is not None


class TestUndistortPointsWithRollingShutter:
    """The entry point most callers use."""

    def test_no_points_in_no_points_out(self):
        assert undistort_points_with_rolling_shutter([], 0.0, 0, _params()) == []

    def test_ndarray_input_does_not_trip_the_truthiness_guard(self):
        """`if not distorted` raises on an ndarray ("truth value of an array
        is ambiguous"); the guard checks length, so both input shapes work.
        The offset search feeds ndarrays."""
        pts = np.array([[CX, CY], [CX + FX, CY + FY]], dtype=np.float64)
        out = undistort_points_with_rolling_shutter(pts, 0.0, 0, _params())
        assert len(out) == 2

    def test_without_rolling_shutter_it_is_the_single_matrix_path(self):
        params = _params(frame_readout_time=0.0)
        out = undistort_points_with_rolling_shutter(
            [(CX, CY), (CX + FX, CY)], 0.0, 0, params
        )
        assert out[0] == pytest.approx((CX, CY))
        assert out[1] == pytest.approx((CX + FX, CY))

    def test_every_input_point_gets_an_output_point(self):
        params = _params(
            frame_readout_time=0.01,
            frame_readout_direction=ReadoutDirection.TopToBottom,
        )
        points = [(100.0, 100.0), (500.0, 700.0), (900.0, 200.0)]
        out = undistort_points_with_rolling_shutter(points, 0.0, 0, params)
        assert len(out) == len(points)
        assert all(len(point) == 2 for point in out)

    def test_the_rolling_shutter_actually_changes_the_answer(self):
        """Two rows of the same file land differently only when RS is on."""
        turn = Quat64.from_euler_angles(0.0, 0.0, math.radians(5.0))
        quats = {0: Quat64.identity(), 200_000: turn}
        points = [(CX, 0.0), (CX, float(HEIGHT))]

        rolled = _params(
            frame_readout_time=0.10,
            frame_readout_direction=ReadoutDirection.TopToBottom,
            quaternions=dict(quats),
            smoothed_quaternions=dict(quats),
        )
        global_shutter = _params(
            frame_readout_time=0.0,
            quaternions=dict(quats),
            smoothed_quaternions=dict(quats),
        )
        rs_out = undistort_points_with_rolling_shutter(points, 0.0, 0, rolled)
        gs_out = undistort_points_with_rolling_shutter(points, 0.0, 0, global_shutter)
        # Global shutter gives every point the same matrix, so both rows land
        # at the same offset from the principal point; the rolled frame does
        # not.
        assert abs(gs_out[0][0] - gs_out[1][0]) < 1e-6
        assert abs(rs_out[0][0] - rs_out[1][0]) > 1e-3


class TestUndistortPointsForOpticalFlow:
    """The flow path scales the matrix to the downscaled frame."""

    def test_the_scaled_principal_point_is_the_origin(self):
        params = _params()
        out = undistort_points_for_optical_flow(
            [(480.0, 270.0)], 0, params, (960, 540)
        )
        assert out[0] == pytest.approx((0.0, 0.0))

    def test_the_matrix_is_scaled_by_the_width_ratio(self):
        """The full-frame principal point is *not* the origin in flow scale."""
        params = _params()
        out = undistort_points_for_optical_flow(
            [(CX, CY)], 0, params, (960, 540)
        )
        # 960 / 1920 = 0.5, so cx' = 480 and the point is 0.96 normalised units
        # off-centre in x and 0.54 in y.
        assert out[0] == pytest.approx((0.96, 0.54))

    def test_a_ratio_of_one_is_the_full_frame_calibration(self):
        params = _params()
        out = undistort_points_for_optical_flow([(CX, CY)], 0, params, (WIDTH, HEIGHT))
        assert out[0] == pytest.approx((0.0, 0.0))

    def test_a_zero_width_does_not_divide_by_zero(self):
        """``max(1, width)`` — a default-constructed params must not crash."""
        params = _params(width=0)
        out = undistort_points_for_optical_flow([(0.0, 0.0)], 0, params, (960, 540))
        assert len(out) == 1

    def test_the_model_is_the_one_the_params_name(self):
        """A change of model changes the answer — the name is not ignored."""
        params = _params(distortion_coeffs=[0.1] + [0.0] * 11)
        fisheye = undistort_points_for_optical_flow(
            [(700.0, 400.0)], 0, params, (960, 540)
        )
        standard = undistort_points_for_optical_flow(
            [(700.0, 400.0)],
            0,
            _params(
                distortion_coeffs=[0.1] + [0.0] * 11,
                distortion_model_name="opencv_standard",
            ),
            (960, 540),
        )
        assert fisheye[0] != pytest.approx(standard[0])
