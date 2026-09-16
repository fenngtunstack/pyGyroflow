# -*- coding: utf-8 -*-
"""Tests for the lens calibrator.

The strong test here is synthetic but complete: a chessboard is *rendered*
through a known fisheye model (corners projected with ``cv2.fisheye.
projectPoints``, squares filled between them), the renderer's own detector
finds the corners, and the calibrator has to recover the intrinsics from
images alone.  Success is measured by reprojection agreement on poses that
were never used for calibration — not by comparing coefficient vectors, which
the fisheye polynomial does not determine uniquely.
"""

from __future__ import annotations

import json
import logging
import sys
from fractions import Fraction

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="OpenCV required for calibration")
av = pytest.importorskip("av", reason="PyAV required for the video-source tests")

from pygyroflow.calibration import LensCalibrator  # noqa: E402
from pygyroflow.calibration.calibrator import SUPPORTED_MODELS, iter_gray_frames  # noqa: E402

# Ground truth of the synthetic camera.  Kept at 640x480: the scene is
# rendered once per pose and the pixel tolerances below scale with it.
WIDTH, HEIGHT = 640, 480
FOCAL = 300.0
K_TRUE = np.array([[FOCAL, 0.0, WIDTH / 2], [0.0, FOCAL, HEIGHT / 2], [0.0, 0.0, 1.0]])
D_TRUE = np.array([[0.05], [-0.01], [0.003], [-0.0008]], dtype=np.float64)

COLS, ROWS, SQUARE = 14, 8, 60  # inner corners per axis, and square size in world units

# Board poses.  The wide set walks the board into the frame corners, which is
# what makes the higher-order fisheye coefficients identifiable.
WIDE_TILTS = (
    (0.0, 0.0), (0.0, 0.35), (0.35, 0.0), (-0.3, 0.3),
    (0.2, 0.2), (-0.35, 0.0), (0.0, -0.35), (0.28, -0.28),
)
NARROW_TILTS = ((0.0, 0.0), (0.0, 0.1), (0.1, 0.0), (-0.08, 0.08))
WIDE_DEPTHS = (330.0, 400.0, 500.0)
NARROW_DEPTHS = (360.0, 430.0, 500.0)
# Detection is ~0.8 s per frame; tests that only need "a calibrator with
# detections" use these instead of the full grid.
SMALL_TILTS = WIDE_TILTS[:3]
SMALL_DEPTHS = (400.0,)
SMALL_FIT_TILTS = WIDE_TILTS[:6]


def render_board(rvec, tvec, width=WIDTH, height=HEIGHT):
    """Draw the chessboard as the fisheye camera would see it.

    *width*/*height* rescale the intrinsics, so a smaller render still frames
    the whole board.
    """
    scale = width / WIDTH
    K = K_TRUE * scale
    K[2, 2] = 1.0
    img = np.full((height, width), 255, np.uint8)
    n_sq_y, n_sq_x = ROWS + 1, COLS + 1
    xs = (np.arange(n_sq_x + 1) - n_sq_x / 2.0) * SQUARE
    ys = (np.arange(n_sq_y + 1) - n_sq_y / 2.0) * SQUARE
    grid = np.array([[(x, y, 0.0) for x in xs] for y in ys], dtype=np.float64)

    projected, _ = cv2.fisheye.projectPoints(grid.reshape(-1, 1, 3), rvec, tvec, K, D_TRUE)
    projected = projected.reshape(n_sq_y + 1, n_sq_x + 1, 2)

    for r in range(n_sq_y):
        for c in range(n_sq_x):
            if (r + c) % 2 == 0:
                quad = np.array([
                    projected[r, c], projected[r, c + 1],
                    projected[r + 1, c + 1], projected[r + 1, c],
                ])
                if np.all(np.isfinite(quad)) and np.all(np.abs(quad) < 1e5):
                    cv2.fillConvexPoly(img, np.round(quad).astype(np.int32), 0)
    return img


def write_board_video(path, frames=6, width=WIDTH, height=HEIGHT, fps=30):
    """Encode a video of the board moving between distinct poses.

    The poses have to differ: identical views make the extrinsics degenerate
    and ``cv2.fisheye.calibrate`` refuses them.
    """
    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=Fraction(fps, 1))
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    for i in range(frames):
        rvec = np.array([((i % 3) - 1) * 0.2, (((i + 1) % 3) - 1) * 0.2, 0.0])
        tvec = np.array([(i % 4 - 1.5) * 30.0, (((i + 2) % 3) - 1) * 25.0, 420.0])
        img = render_board(rvec, tvec, width, height)
        frame = av.VideoFrame.from_ndarray(np.repeat(img[:, :, None], 3, axis=2), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def feed_poses(calibrator, tilts, depths):
    for z in depths:
        for rx, ry in tilts:
            image = render_board(np.array([rx, ry, 0.0]), np.array([0.0, 0.0, z]))
            calibrator.feed_frame(image, frame_number=len(calibrator._all_matches))
    return calibrator


def make_calibrator(**kwargs):
    params = dict(columns=COLS, rows=ROWS, square_size=SQUARE, max_images=0, iterations=1)
    params.update(kwargs)
    return LensCalibrator(**params)


def max_reprojection_error(result, sample_seed=7):
    """Largest pixel disagreement between truth and the calibration on poses
    the calibration never saw."""
    Kc = result.camera_matrix
    Dc = np.asarray(result.dist_coeffs, dtype=np.float64).reshape(4, 1)
    rng = np.random.default_rng(sample_seed)
    worst = 0.0
    for z in (350.0, 460.0, 540.0):
        for rx, ry in ((0.15, 0.2), (0.3, 0.3), (-0.3, 0.25), (0.35, -0.3), (0.0, 0.0)):
            points = rng.uniform(-220, 220, (300, 1, 3))
            points[:, :, 2] = 0.0
            rvec, tvec = np.array([rx, ry, 0.0]), np.array([0.0, 0.0, z])
            truth, _ = cv2.fisheye.projectPoints(points, rvec, tvec, K_TRUE, D_TRUE)
            fitted, _ = cv2.fisheye.projectPoints(points, rvec, tvec, Kc, Dc)
            delta = np.linalg.norm(truth.reshape(-1, 2) - fitted.reshape(-1, 2), axis=-1)
            worst = max(worst, float(delta.max()))
    return worst


@pytest.fixture(scope="module")
def wide_calibration():
    """A calibration run over a board that visits the frame corners."""
    cal = feed_poses(make_calibrator(), WIDE_TILTS, WIDE_DEPTHS)
    return cal, cal.calibrate()


# --------------------------------------------------------------------------- #
#  Detection and acceptance                                                    #
# --------------------------------------------------------------------------- #


class TestDetection:
    def test_finds_and_accepts_the_synthetic_board(self):
        cal = feed_poses(make_calibrator(), SMALL_TILTS, SMALL_DEPTHS)

        assert cal.num_detected == len(SMALL_TILTS) > 0
        assert cal.num_candidates == cal.num_detected

    def test_sharpness_matches_the_opencv_metric(self):
        """The gate must use cv::estimateChessboardSharpness units, where the
        value is a transition width in pixels and *lower* is sharper."""
        sharp = make_calibrator()
        sharp.feed_frame(render_board(np.zeros(3), np.array([0.0, 0.0, 400.0])), frame_number=0)
        sharpness = sharp._all_matches[0].sharpness

        assert sharpness is not None
        assert 0.05 < sharpness < 5.0  # a rendered board is crisp

        blurred = make_calibrator()
        blurred.feed_frame(cv2.GaussianBlur(
            render_board(np.zeros(3), np.array([0.0, 0.0, 400.0])), (21, 21), 7.0
        ), frame_number=0)
        assert blurred._all_matches[0].sharpness > sharpness

    def test_blurred_frames_are_rejected_from_candidates(self):
        """A frame softer than the threshold is still detected, but must not
        feed the calibration."""
        image = cv2.GaussianBlur(render_board(np.zeros(3), np.array([0.0, 0.0, 400.0])), (21, 21), 7.0)

        cal = make_calibrator(max_sharpness=0.5)  # anything but a perfect board fails
        detected = cal.feed_frame(image, frame_number=0)

        assert detected is not None           # detected ...
        assert cal.num_detected == 1          # ... recorded ...
        assert cal.num_candidates == 0        # ... but not trusted

    def test_blank_frame_yields_nothing(self):
        cal = make_calibrator()
        assert cal.feed_frame(np.full((480, 640), 128, np.uint8), frame_number=0) is None
        assert cal.num_detected == 0

    def test_empty_image_is_ignored(self):
        cal = make_calibrator()
        assert cal.feed_frame(None, frame_number=0) is None
        assert cal.num_detected == 0

    def test_clear_resets_state(self):
        cal = feed_poses(make_calibrator(), SMALL_TILTS, SMALL_DEPTHS)
        assert cal.num_detected > 0

        cal.clear()

        assert cal.num_detected == 0
        assert cal.num_candidates == 0
        assert cal.get_result() is None


# --------------------------------------------------------------------------- #
#  Calibration accuracy                                                        #
# --------------------------------------------------------------------------- #


class TestFisheyeCalibration:
    def test_recovers_the_camera_matrix(self, wide_calibration):
        _cal, result = wide_calibration

        assert result.camera_matrix[0, 0] == pytest.approx(FOCAL, rel=0.01)
        assert result.camera_matrix[1, 1] == pytest.approx(FOCAL, rel=0.01)
        assert result.camera_matrix[0, 2] == pytest.approx(WIDTH / 2, abs=1.0)
        assert result.camera_matrix[1, 2] == pytest.approx(HEIGHT / 2, abs=1.0)

    def test_residual_is_sub_pixel(self, wide_calibration):
        _cal, result = wide_calibration
        assert result.rms < 1.0

    def test_reprojects_truth_on_unseen_poses(self, wide_calibration):
        """The coefficient vector is not unique, so agreement is measured on
        the thing that matters: where the model puts points it never saw."""
        _cal, result = wide_calibration
        # Sub-pixel: measured 0.60 px on this scene, against a 0.5 px corner
        # detection floor at 640x480.
        assert max_reprojection_error(result) < 1.0

    def test_fisheye_needs_no_model_conversion(self, wide_calibration):
        _cal, result = wide_calibration
        assert result.model_fit_error is None

    def test_used_frames_are_reported(self, wide_calibration):
        cal, result = wide_calibration
        # max_images=0 means "use everything that was accepted"
        assert len(result.used_frames) == cal.num_candidates

    def test_narrow_field_of_view_leaves_the_curve_wrong(self, wide_calibration):
        """Pins the sampling advice: with the board kept near the centre the
        RMS looks just as good but the model misplaces off-centre points.

        Measured on this scene: narrow RMS 0.467 px vs wide 0.453 px (no
        separation at all), while max reprojection is 4.37 px vs 0.60 px —
        a 7.3x gap that RMS does not see.
        """
        narrow = feed_poses(make_calibrator(), NARROW_TILTS, NARROW_DEPTHS).calibrate()
        _c, wide = wide_calibration

        assert narrow.rms < 1.0                       # looks fine ...
        assert max_reprojection_error(narrow) > 1.0   # ... but it is not
        assert max_reprojection_error(narrow) > 5 * max_reprojection_error(wide)


# --------------------------------------------------------------------------- #
#  Model conversion                                                            #
# --------------------------------------------------------------------------- #


class TestModelConversion:
    FISHEYE_D = np.array([[0.05], [-0.01], [0.003], [-0.0008]])

    def _convert(self, model):
        cal = LensCalibrator(distortion_model=model)
        return cal._convert_fisheye_coeffs(self.FISHEYE_D, K_TRUE, (WIDTH, HEIGHT))

    def test_poly3_is_a_fit_not_a_relabel(self):
        """Guards the previous behaviour, which copied k1 straight across."""
        coeffs, error = self._convert("poly3")

        assert coeffs.shape == (1, 1)
        assert float(coeffs[0, 0]) != pytest.approx(float(self.FISHEYE_D[0, 0]))
        assert error is not None and error > 0.0

    def test_more_coefficients_fit_better(self):
        """poly5 (2 params) and ptlens (3) must beat poly3 (1)."""
        _, poly3 = self._convert("poly3")
        poly5_coeffs, poly5 = self._convert("poly5")
        ptlens_coeffs, ptlens = self._convert("ptlens")

        assert poly5_coeffs.shape == (2, 1)
        assert ptlens_coeffs.shape == (3, 1)
        assert poly5 < poly3
        assert ptlens < poly5

    def test_fit_residual_is_finite_and_relative(self):
        for model in ("poly3", "poly5", "ptlens"):
            _coeffs, error = self._convert(model)
            assert np.isfinite(error)
            assert 0.0 < error < 10.0  # a relative error, not a pixel or an inf

    def test_conversion_lands_in_the_renderer_layout(self):
        """poly3 reads k1[0]; poly5 reads k1[0..1]; ptlens reads k1[0..2] as
        (a, b, c) for r_d = r_u*(a*r_u^3 + b*r_u^2 + c*r_u + 1)."""
        from pygyroflow.stabilization.distortion_models import from_name
        from pygyroflow.types.kernel_params import KernelParams

        for model, expected in (("poly3", 1), ("poly5", 2), ("ptlens", 3)):
            coeffs, _err = self._convert(model)
            params = KernelParams()
            for i, value in enumerate(coeffs.ravel()):
                params.k1[i] = float(value)

            # feeding the coefficients back through the render model must not
            # raise and must stay finite on a point inside the frame
            fx = float(K_TRUE[0, 0])
            x = (WIDTH * 0.75 - K_TRUE[0, 2]) / fx
            y = (HEIGHT * 0.75 - K_TRUE[1, 2]) / fx
            out = from_name(model).undistort_point(x, y, params)
            assert out is not None
            assert np.all(np.isfinite(out))
            assert len(coeffs.ravel()) == expected

    def test_bad_fit_is_warned_about(self, caplog):
        cal = LensCalibrator(distortion_model="poly3")
        with caplog.at_level(logging.WARNING):
            _coeffs, error = cal._convert_fisheye_coeffs(self.FISHEYE_D, K_TRUE, (WIDTH, HEIGHT))

        assert error > 0.05  # one coefficient cannot follow this fisheye curve
        assert "degrees of freedom" in caplog.text


# --------------------------------------------------------------------------- #
#  Refusals and error paths                                                    #
# --------------------------------------------------------------------------- #


class TestRefusals:
    def test_opencv_standard_is_refused_with_the_reason(self):
        with pytest.raises(ValueError) as exc:
            LensCalibrator(distortion_model="opencv_standard")

        assert "opencv_standard" in str(exc.value)
        assert "k6" in str(exc.value)  # names the layout clash

    def test_unknown_model_is_refused(self):
        with pytest.raises(ValueError):
            LensCalibrator(distortion_model="not_a_model")

    def test_supported_models_construct(self):
        for model in SUPPORTED_MODELS:
            assert LensCalibrator(distortion_model=model) is not None

    def test_digital_lens_calibration_is_refused(self):
        cal = make_calibrator(digital_lens="gopro_superview")
        feed_poses(cal, SMALL_TILTS, SMALL_DEPTHS)

        with pytest.raises(ValueError) as exc:
            cal.calibrate()
        assert "digital_lens" in str(exc.value)

    def test_too_few_corners_is_refused(self):
        with pytest.raises(ValueError):
            LensCalibrator(columns=1, rows=8)

    def test_calibrate_requires_two_frames(self):
        cal = make_calibrator()
        feed_poses(cal, WIDE_TILTS[:1], WIDE_DEPTHS[:1])

        with pytest.raises(RuntimeError):
            cal.calibrate()

    def test_only_used_reruns_on_the_same_frames(self, wide_calibration):
        cal, first = wide_calibration
        second = cal.calibrate(only_used=True)

        assert second.used_frames == first.used_frames
        assert second.camera_matrix == pytest.approx(first.camera_matrix)


# --------------------------------------------------------------------------- #
#  Profile export                                                              #
# --------------------------------------------------------------------------- #


class TestProfileExport:
    def test_profile_matches_the_lens_profile_schema(self, wide_calibration):
        cal, result = wide_calibration
        profile = cal.to_lens_profile_dict()

        assert profile["calib_dimension"] == {"w": WIDTH, "h": HEIGHT}
        assert profile["distortion_model"] == "opencv_fisheye"
        assert profile["num_images"] == len(result.used_frames)
        assert len(profile["fisheye_params"]["camera_matrix"]) == 3
        assert len(profile["fisheye_params"]["camera_matrix"][0]) == 3
        assert len(profile["fisheye_params"]["distortion_coeffs"]) == 4
        assert profile["fisheye_params"]["RMS_error"] == pytest.approx(result.rms)

    def test_profile_round_trips_through_lens_profile(self, wide_calibration):
        from pygyroflow.lens import LensProfile

        cal, result = wide_calibration
        restored = LensProfile.from_json(cal.to_lens_profile_dict())

        assert restored.calib_dimension == {"w": WIDTH, "h": HEIGHT}
        assert np.allclose(np.array(restored.camera_matrix), cal.camera_matrix)
        assert np.allclose(restored.distortion_coeffs, np.asarray(result.dist_coeffs).ravel())

    def test_converted_model_reports_its_residual(self):
        cal = feed_poses(make_calibrator(distortion_model="ptlens"), SMALL_FIT_TILTS, SMALL_DEPTHS)
        result = cal.calibrate()
        profile = cal.to_lens_profile_dict()

        assert "distortion_model_fit_error" in profile
        assert profile["distortion_model_fit_error"] == pytest.approx(result.model_fit_error)

    def test_empty_calibrator_exports_nothing(self):
        assert make_calibrator().to_lens_profile_dict() == {}


# --------------------------------------------------------------------------- #
#  Frame source                                                                #
# --------------------------------------------------------------------------- #


class TestIterGrayFrames:
    def _write_video(self, path, frames=6, width=640, height=480, fps=30):
        write_board_video(path, frames, width, height, fps)

    def test_reads_gray_frames_from_a_video(self, tmp_path):
        src = tmp_path / "clip.mp4"
        self._write_video(src)

        frames = list(iter_gray_frames(str(src)))

        assert len(frames) == 6
        assert frames[0].ndim == 2  # grayscale, not BGR
        assert frames[0].shape == (480, 640)

    def test_every_n_subsamples(self, tmp_path):
        src = tmp_path / "clip.mp4"
        self._write_video(src, frames=10)

        assert len(list(iter_gray_frames(str(src), every_n=5))) == 2

    def test_reads_an_image_sequence(self, tmp_path):
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for i in range(1, 5):
            board = render_board(np.zeros(3), np.array([0.0, 0.0, 400.0]), 640, 480)
            cv2.imwrite(str(frames_dir / f"f_{i:04d}.png"), board)

        frames = list(iter_gray_frames(str(frames_dir), fps=30.0))

        assert len(frames) == 4
        assert frames[0].shape == (480, 640)

    def test_reads_a_single_still(self, tmp_path):
        still = tmp_path / "board.png"
        cv2.imwrite(str(still), render_board(np.zeros(3), np.array([0.0, 0.0, 400.0]), 640, 480))

        assert len(list(iter_gray_frames(str(still)))) == 1

    def test_calibrates_from_a_video(self, tmp_path):
        """The whole chain: video file -> frames -> detection -> calibration."""
        src = tmp_path / "clip.mp4"
        self._write_video(src, frames=8)

        cal = make_calibrator(columns=COLS, rows=ROWS)
        found = cal.feed_frames(list(iter_gray_frames(str(src))), every_n=1)

        assert found == 8
        result = cal.calibrate()
        assert result.camera_matrix[0, 0] == pytest.approx(FOCAL, rel=0.15)
        assert np.isfinite(result.rms) and result.rms < 5.0

    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            list(iter_gray_frames(str(tmp_path / "nope.mp4")))


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #


class TestCliCalibration:
    def _run(self, monkeypatch, *argv):
        from pygyroflow.cli.main import main

        monkeypatch.setattr(sys, "argv", ["pygyroflow", *argv])
        with pytest.raises(SystemExit) as exc:
            main()
        return exc.value.code

    def test_writes_a_profile_next_to_the_input_by_default(self, tmp_path, monkeypatch):
        src = tmp_path / "board.mp4"
        write_board_video(src, frames=10)

        code = self._run(monkeypatch, str(src), "--calibrate", "--calib-every", "2")

        assert code == 0
        profile_path = tmp_path / "board_lens_profile.json"
        assert profile_path.exists()

        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        assert profile["distortion_model"] == "opencv_fisheye"
        assert profile["calib_dimension"] == {"w": WIDTH, "h": HEIGHT}
        assert profile["fisheye_params"]["RMS_error"] > 0
        assert profile["name"] and profile["date"]

        # the profile has to be loadable by --lens on the next run
        from pygyroflow.lens import LensProfile

        restored = LensProfile.from_json(profile)
        assert restored.camera_matrix[0][0] == pytest.approx(FOCAL, rel=0.05)

    def test_honours_an_explicit_output_path(self, tmp_path, monkeypatch):
        src = tmp_path / "board.mp4"
        write_board_video(src, frames=10)
        out = tmp_path / "custom.json"

        code = self._run(monkeypatch, str(src), "--calibrate", "--calib-every", "2", "-o", str(out))

        assert code == 0
        assert out.exists()

    def test_no_board_exits_nonzero(self, tmp_path, monkeypatch):
        src = tmp_path / "not_a_board.mp4"
        container = av.open(str(src), "w")
        stream = container.add_stream("libx264", rate=Fraction(30, 1))
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        rng = np.random.default_rng(5)
        for _ in range(6):
            noise = rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(noise, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()

        code = self._run(monkeypatch, str(src), "--calibrate", "--calib-every", "1")

        assert code == 1
        assert not (tmp_path / "not_a_board_lens_profile.json").exists()

    def test_two_inputs_is_refused(self, tmp_path, monkeypatch):
        assert self._run(monkeypatch, "a.mp4", "b.mp4", "--calibrate") == 2

    def test_unsupported_model_is_refused(self, tmp_path, monkeypatch):
        # opencv_standard is deliberately absent (see SUPPORTED_MODELS)
        assert self._run(monkeypatch, "a.mp4", "--calibrate", "--calib-model", "opencv_standard") == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
