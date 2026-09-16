"""Lens calibrator -- auto chessboard calibration.

Uses OpenCV chessboard detection to compute camera intrinsics and distortion
coefficients from a set of calibration frames.  Ported from Gyroflow's
core/calibration/mod.rs.

Workflow:
  1. feed_frame() -- detect chessboard corners in each video frame
  2. calibrate()  -- random-sampling calibration, keep lowest RMS result
  3. get_result()  -- retrieve camera matrix, distortion coefficients, RMS

Upstream's calibrator emits **fisheye** coefficients only (`cv::fisheye::
calibrate` -- the four k's of ``theta_d = theta*(1 + k0*t^2 + k1*t^4 + ...)``,
the same ones Gyroflow's ``opencv_fisheye`` render model reads back).  The
other models are offered here as conversions from that fisheye fit, and the
conversion residual is reported so the substitution can be judged.

Calibration quality is dominated by how much of the field of view the board
visits: with the board kept near the image centre the higher-order fisheye
coefficients are effectively unidentifiable (many k-vectors describe the same
central curve), and only the curve inside the covered radius is meaningful.
Move the board into the corners of the frame.
"""

from __future__ import annotations

import logging
import os
import random
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Supported distortion model identifiers.  ``opencv_standard`` is deliberately
# absent: the renderer's 12-parameter layout is Gyroflow's own forward/inverse
# split (k1,k2,p1,p2,k3,k6,k7,k8,s1..s4), and OpenCV's rational model produces
# (k1,k2,p1,p2,k3,k4,k5,k6) — positions 4..7 mean different things, so handing
# one to the other yields a profile that renders wrong.  Refusing is the safe
# behaviour; converting between the two parametrisations is a separate job.
SUPPORTED_MODELS = (
    "opencv_fisheye",
    "poly3",
    "poly5",
    "ptlens",
)

_REFUSED_MODELS = {
    "opencv_standard": (
        "the renderer's opencv_standard layout (k1,k2,p1,p2,k3,k6,k7,k8,s1..s4) is "
        "Gyroflow's forward/inverse split, while OpenCV's rational model returns "
        "(k1,k2,p1,p2,k3,k4,k5,k6) — the two disagree from the fifth coefficient on, "
        "so the coefficients cannot be passed through unchanged"
    ),
}

# Upstream's chessboard sharpness acceptance threshold, in the units of
# cv::estimateChessboardSharpness (transition width in pixels — smaller is
# sharper).  A well-focused board measures well under 3.0.
DEFAULT_MAX_SHARPNESS = 5.0

# Forward distortion models we can fit to the fisheye curve, keyed by the
# number of coefficients the resampling model expects.
_FIT_MODELS = {"poly3": 1, "poly5": 2, "ptlens": 3}


def iter_gray_frames(path: str, fps: float | None = None, every_n: int = 1) -> Iterator[np.ndarray]:
    """Yield grayscale frames from a video or an image sequence.

    Calibration only needs the frames, so this skips telemetry and takes the
    sequence-aware path (see :mod:`pygyroflow.rendering.image_sequence`) when
    the input is a directory, a printf pattern, or a single still.

    Parameters
    ----------
    path:
        Video file, or an image sequence.
    fps:
        Frame rate for image sequence input; ignored for video.
    every_n:
        Yield only every N-th frame.  Detection is the expensive part and
        neighbouring frames are near-duplicates, so upstream looks at every
        10th frame.

    Raises
    ------
    FileNotFoundError: *path* is neither an existing file nor a sequence.
    """
    import av

    from pygyroflow.rendering.image_sequence import (
        format_options,
        looks_like_image_sequence,
        resolve_image_sequence,
    )

    if not os.path.isfile(path) and not looks_like_image_sequence(path):
        raise FileNotFoundError(path)

    sequence = resolve_image_sequence(path) if looks_like_image_sequence(path) else None
    if sequence is not None:
        container = av.open(
            sequence.pattern,
            format="image2" if sequence.is_sequence else None,
            options=format_options(sequence, fps),
        )
    else:
        container = av.open(path)

    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        index = 0
        for packet in container.demux(stream):
            if packet.dts is None:
                continue
            for frame in packet.decode():
                if index % every_n == 0:
                    yield frame.to_ndarray(format="gray")
                index += 1
        # Flush the threaded decoder (trailing frames).
        for frame in stream.decode():
            if index % every_n == 0:
                yield frame.to_ndarray(format="gray")
            index += 1
    finally:
        container.close()


@dataclass
class DetectedCorners:
    """Chessboard corners detected in a single frame."""

    points: list[tuple[float, float]]
    frame_number: int
    timestamp_us: int
    sharpness: float | None
    is_forced: bool = False


@dataclass
class CalibrationResult:
    """Output of a successful calibration run."""

    camera_matrix: np.ndarray   # 3x3
    dist_coeffs: np.ndarray     # shape depends on model
    rms: float
    used_frames: list[int]
    image_size: tuple[int, int]  # (width, height)
    # Max relative error of the fitted distortion model against the fisheye
    # curve it was converted from.  None when no conversion happened (the
    # fisheye fit *is* the result).
    model_fit_error: float | None = None


class LensCalibrator:
    """Auto chessboard calibration engine.

    Parameters
    ----------
    columns, rows:
        Number of *inner* chessboard corners (not squares).
    square_size:
        Physical size of a chessboard square in arbitrary units (default 1.0).
    max_images:
        Maximum number of images per random-sampling iteration.
    iterations:
        Number of RANSAC iterations (more = better chance of good result).
    max_sharpness:
        Sharpness threshold for frame quality filtering, in transition-width
        pixels (``cv::estimateChessboardSharpness`` semantics — *lower* is
        sharper, so a frame is accepted when its value is **below** this).
        Upstream's default is 5.0; a focused board measures well under 3.0.
    distortion_model:
        One of ``SUPPORTED_MODELS``.  Determines the OpenCV calibration
        function and the number of output coefficients.
    digital_lens:
        Optional digital-lens name (e.g. ``"gopro_superview"``).
    digital_lens_params:
        Optional extra parameters for the digital lens model.
    asymmetrical:
        Whether the digital lens is asymmetric.
    """

    def __init__(
        self,
        columns: int = 14,
        rows: int = 8,
        square_size: float = 1.0,
        max_images: int = 10,
        iterations: int = 1000,
        max_sharpness: float = DEFAULT_MAX_SHARPNESS,
        distortion_model: str = "opencv_fisheye",
        digital_lens: str | None = None,
        digital_lens_params: list[float] | None = None,
        asymmetrical: bool = False,
    ) -> None:
        if columns < 2 or rows < 2:
            raise ValueError("Need at least 2x2 inner corners")
        if distortion_model in _REFUSED_MODELS:
            raise ValueError(
                f"Unsupported model '{distortion_model}': {_REFUSED_MODELS[distortion_model]}"
            )
        if distortion_model not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model '{distortion_model}', "
                f"choose from {SUPPORTED_MODELS}"
            )

        self.columns = columns
        self.rows = rows
        self.square_size = square_size
        self.max_images = max_images
        self.iterations = iterations
        self.max_sharpness = max_sharpness
        self.distortion_model = distortion_model
        self.digital_lens = digital_lens
        self.digital_lens_params = digital_lens_params
        self.asymmetrical = asymmetrical

        # Object points: (col, row, 0) on z=0 plane.
        self._obj_points = self._make_object_points()

        # Image dimensions (set on first frame).
        self._width = 0
        self._height = 0

        # Detection storage.
        self._all_matches: dict[int, DetectedCorners] = {}
        self._image_points: dict[int, DetectedCorners] = {}

        # Last calibration result.
        self._result: CalibrationResult | None = None
        self._warned_no_sharpness = False

    # ------------------------------------------------------------------
    # Object-point grid
    # ------------------------------------------------------------------

    def _make_object_points(self) -> np.ndarray:
        """Generate the (rows*columns, 3) world-coordinate grid."""
        obj = np.zeros((self.rows * self.columns, 3), dtype=np.float64)
        for r in range(self.rows):
            for c in range(self.columns):
                idx = r * self.columns + c
                obj[idx, 0] = c * self.square_size
                obj[idx, 1] = r * self.square_size
        return obj

    # ------------------------------------------------------------------
    # Frame feeding
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Reset all stored detections and calibration result."""
        self._all_matches.clear()
        self._image_points.clear()
        self._result = None

    def feed_frame(
        self,
        image: np.ndarray,
        frame_number: int = 0,
        timestamp_us: int = 0,
        enhance: bool = True,
    ) -> DetectedCorners | None:
        """Detect chessboard corners in a single frame.

        Parameters
        ----------
        image:
            BGR or grayscale image (numpy array).
        frame_number:
            Sequential frame index for bookkeeping.
        timestamp_us:
            Frame timestamp in microseconds.
        enhance:
            Apply contrast boost + histogram equalisation before detection,
            matching Gyroflow's preprocessing pipeline.

        Returns
        -------
        DetectedCorners or None if the chessboard was not found.
        """
        if image is None or image.size == 0:
            return None

        self._width = image.shape[1]
        self._height = image.shape[0]

        # Convert to grayscale if needed.
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if enhance:
            gray = self._enhance_image(gray)

        grid = (self.columns, self.rows)
        # flags=0 accepts boards without the white orientation marker.
        # Upstream defaults to CALIB_CB_MARKER (marker required) with a
        # "no marker" toggle; being permissive by default suits arbitrary
        # printed boards, and a marker-bearing board still detects.
        found, corners = cv2.findChessboardCornersSB(gray, grid, flags=0)

        if not found or corners is None or len(corners) == 0:
            return None

        corners = corners.reshape(-1, 2).astype(np.float64)

        # Sub-pixel refinement (findChessboardCornersSB already returns
        # sub-pixel accuracy, but a light refinement pass helps).
        corners_f32 = corners.astype(np.float32).reshape(-1, 1, 2)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        cv2.cornerSubPix(
            gray, corners_f32, (5, 5), (-1, -1), criteria
        )
        corners = corners_f32.reshape(-1, 2).astype(np.float64)

        # Sharpness: OpenCV's own chessboard sharpness, the same function
        # upstream calls.  The value is a transition width in pixels, so it is
        # not comparable with any home-grown focus metric.
        sharpness = self._estimate_sharpness(gray, corners)

        detected = DetectedCorners(
            points=[(float(p[0]), float(p[1])) for p in corners],
            frame_number=frame_number,
            timestamp_us=timestamp_us,
            sharpness=sharpness,
            is_forced=False,
        )

        # Cache.
        self._all_matches[frame_number] = detected

        # Accept if sharpness is below the threshold (lower = sharper).
        # When the metric is unavailable, accept rather than apply a
        # meaningless comparison.
        if sharpness is None or sharpness < self.max_sharpness:
            self._image_points[frame_number] = detected

        return detected

    def feed_frames(
        self,
        images: Sequence[np.ndarray],
        start_frame: int = 0,
        enhance: bool = True,
        every_n: int = 10,
        cancel_check=None,
    ) -> int:
        """Feed a sequence of frames, sampling every *every_n* frames.

        This matches Gyroflow's "detect every 10th frame" approach.

        Parameters
        ----------
        images:
            Iterable of BGR/grayscale frames.
        start_frame:
            Starting frame index.
        enhance:
            Apply contrast enhancement before detection.
        every_n:
            Only try detection on every N-th frame.
        cancel_check:
            Optional callable returning True to abort early.

        Returns
        -------
        Number of frames where chessboard was found.
        """
        found_count = 0
        for idx, img in enumerate(images):
            if cancel_check and cancel_check():
                break
            if idx % every_n != 0:
                continue
            result = self.feed_frame(
                img,
                frame_number=start_frame + idx,
                enhance=enhance,
            )
            if result is not None:
                found_count += 1
        return found_count

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        only_used: bool = False,
    ) -> CalibrationResult:
        """Run the RANSAC-style calibration.

        Mirrors Gyroflow's strategy: repeatedly pick a random subset of
        detected frames, run OpenCV calibration, keep the lowest-RMS result.

        Parameters
        ----------
        only_used:
            If True, reuse the frames from the last successful calibration
            (re-calibrate with the same subset).

        Returns
        -------
        CalibrationResult with camera_matrix, dist_coeffs, rms, etc.

        Raises
        ------
        RuntimeError
            If not enough frames have been detected (< 2).
        ValueError
            If ``digital_lens`` is set — upstream undistorts the detected
            corners through the digital lens before calibrating, and that
            step is not implemented here, so the calibration would silently
            describe the wrong geometry.
        """
        if self.digital_lens:
            raise ValueError(
                "digital_lens calibration is not supported: upstream undistorts the "
                "detected corners through the digital lens model before solving, and "
                "that step is missing here. Calibrate without digital_lens, or use "
                "upstream Gyroflow."
            )

        if only_used and self._result is not None:
            # Restrict to previously used frames.
            candidate_frames = set(self._result.used_frames)
        else:
            candidate_frames = set(self._image_points.keys())

        if len(candidate_frames) < 2:
            raise RuntimeError(
                f"Need at least 2 frames with detected chessboard corners, "
                f"got {len(candidate_frames)}"
            )

        # Decide iteration count.
        n_iter = self.iterations
        if len(candidate_frames) <= self.max_images or self.max_images == 0 or only_used:
            n_iter = 1

        image_size = (self._width, self._height)
        calib_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

        # Draw every subset up front: the sampling is the only random part, so
        # doing it sequentially keeps the run reproducible while the solves go
        # wide.  OpenCV releases the GIL inside fisheye::calibrate, so threads
        # really do overlap here — upstream gets the same effect from rayon.
        subsets = [self._pick_frames(candidate_frames) for _ in range(n_iter)]

        def solve(chosen: list[int]) -> tuple[float, np.ndarray, np.ndarray, float | None, list[int]] | None:
            obj_points_list: list[np.ndarray] = []
            img_points_list: list[np.ndarray] = []
            objp_prepped = self._obj_points.reshape(-1, 1, 3).astype(np.float32)
            for f in chosen:
                det = self._image_points.get(f)
                if det is None:
                    continue
                obj_points_list.append(objp_prepped)
                img_points_list.append(np.array(det.points, dtype=np.float32).reshape(-1, 1, 2))

            if len(obj_points_list) < 2:
                return None
            try:
                rms, K, D, fit_error = self._run_cv_calibration(
                    obj_points_list, img_points_list, image_size, calib_criteria
                )
            except cv2.error as exc:
                logger.warning("Calibration iteration failed: %s", exc)
                return None
            return rms, K, np.asarray(D, dtype=np.float64), fit_error, list(chosen)

        if n_iter > 1 and len(subsets) > 4:
            workers = min(os.cpu_count() or 1, len(subsets))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                solved = list(pool.map(solve, subsets))
        else:
            solved = [solve(chosen) for chosen in subsets]

        best = min((s for s in solved if s is not None), key=lambda s: s[0], default=None)
        if best is None:
            raise RuntimeError("All calibration iterations failed")

        best_rms, best_K, best_D, best_fit_error, best_frames = best

        self._result = CalibrationResult(
            camera_matrix=best_K,
            dist_coeffs=best_D,
            rms=best_rms,
            used_frames=best_frames,
            image_size=image_size,
            model_fit_error=best_fit_error,
        )
        return self._result

    # ------------------------------------------------------------------
    # Result access
    # ------------------------------------------------------------------

    def get_result(self) -> CalibrationResult | None:
        """Return the last CalibrationResult, or None if not yet calibrated."""
        return self._result

    @property
    def rms(self) -> float:
        """RMS reprojection error of the last calibration."""
        return self._result.rms if self._result else float("inf")

    @property
    def camera_matrix(self) -> np.ndarray | None:
        return self._result.camera_matrix if self._result else None

    @property
    def dist_coeffs(self) -> np.ndarray | None:
        return self._result.dist_coeffs if self._result else None

    @property
    def focal_length_px(self) -> tuple[float, float] | None:
        """(fx, fy) in pixels, or None."""
        if self._result is None:
            return None
        K = self._result.camera_matrix
        return (float(K[0, 0]), float(K[1, 1]))

    @property
    def principal_point(self) -> tuple[float, float] | None:
        """(cx, cy) in pixels, or None."""
        if self._result is None:
            return None
        K = self._result.camera_matrix
        return (float(K[0, 2]), float(K[1, 2]))

    @property
    def num_candidates(self) -> int:
        """Number of frames currently accepted for calibration."""
        return len(self._image_points)

    @property
    def num_detected(self) -> int:
        """Total frames where a chessboard was found (including low quality)."""
        return len(self._all_matches)

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def to_lens_profile_dict(self) -> dict:
        """Convert calibration result to a Gyroflow-compatible JSON dict.

        The dict matches the LensProfile JSON schema used by Gyroflow:
        ``fisheye_params.camera_matrix`` (3x3 list),
        ``fisheye_params.distortion_coeffs`` (list),
        ``distortion_model``, etc.

        ``distortion_model_fit_error`` is added when the coefficients came
        from a conversion (poly3/poly5/ptlens): it is the largest relative
        difference between the substituted model's undistortion scale and the
        calibrated fisheye curve over the covered image radius.
        """
        if self._result is None:
            return {}

        K = self._result.camera_matrix
        D = self._result.dist_coeffs

        profile = {
            "calib_dimension": {
                "w": self._width,
                "h": self._height,
            },
            "distortion_model": self.distortion_model,
            "fisheye_params": {
                "camera_matrix": K.tolist(),
                "distortion_coeffs": np.asarray(D).flatten().tolist(),
                "RMS_error": self._result.rms,
            },
            "digital_lens": self.digital_lens,
            "digital_lens_params": self.digital_lens_params,
            "asymmetrical": self.asymmetrical,
            "num_images": len(self._result.used_frames),
        }
        if self._result.model_fit_error is not None:
            profile["distortion_model_fit_error"] = self._result.model_fit_error
        return profile

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _enhance_image(gray: np.ndarray) -> np.ndarray:
        """Apply contrast boost + histogram equalisation (matches Gyroflow)."""
        contrast = 2.0
        brightness = -50.0
        enhanced = np.clip(gray.astype(np.float64) * contrast + brightness, 0, 255)
        enhanced = enhanced.astype(np.uint8)
        enhanced = cv2.equalizeHist(enhanced)
        return enhanced

    def _estimate_sharpness(self, gray: np.ndarray, corners: np.ndarray) -> float | None:
        """Chessboard sharpness via ``cv::estimateChessboardSharpness``.

        Returns the average transition width in pixels (upstream's element 0):
        *lower is sharper*.  A well-focused board lands well under 3.0 and a
        visibly blurred one climbs past 6.

        Returns ``None`` when the OpenCV build lacks the function (it appeared
        in OpenCV 4.7) — the caller then skips the sharpness gate instead of
        comparing against a number that has no meaning.
        """
        if not hasattr(cv2, "estimateChessboardSharpness"):
            if not self._warned_no_sharpness:
                self._warned_no_sharpness = True
                logger.warning(
                    "cv2.estimateChessboardSharpness unavailable (needs OpenCV >= 4.7) — "
                    "accepting every detected frame"
                )
            return None
        try:
            # rise_distance 0.8 (10%..90% of the edge step) is upstream's value.
            scalar, _per_view = cv2.estimateChessboardSharpness(
                gray, (self.columns, self.rows), corners, 0.8
            )
        except cv2.error as exc:
            logger.debug("Sharpness estimation failed: %s", exc)
            return None
        return float(np.asarray(scalar).ravel()[0])

    def _pick_frames(self, candidates: set[int]) -> list[int]:
        """Select a random subset of candidate frames for one calibration iteration.

        Strategy (matches Gyroflow):
          - If few frames, use all of them.
          - Otherwise, divide the frame range into max_images equal bins
            and randomly pick one frame from each bin for temporal spread.
        """
        sorted_frames = sorted(candidates)

        if len(sorted_frames) <= self.max_images or self.max_images == 0:
            return sorted_frames

        # Uniformly distributed sampling across frame range.
        min_f = sorted_frames[0]
        max_f = sorted_frames[-1]
        step = (max_f - min_f) / self.max_images
        chosen: list[int] = []

        for i in range(self.max_images):
            lo = min_f + int(i * step)
            hi = min_f + int((i + 1) * step)
            bin_frames = [f for f in sorted_frames if lo <= f < hi]
            if bin_frames:
                chosen.append(random.choice(bin_frames))

        # Deduplicate while preserving order.
        seen: set[int] = set()
        unique: list[int] = []
        for f in chosen:
            if f not in seen:
                seen.add(f)
                unique.append(f)
        return unique

    def _run_cv_calibration(
        self,
        obj_points: list[np.ndarray],
        img_points: list[np.ndarray],
        image_size: tuple[int, int],
        criteria: tuple,
    ) -> tuple[float, np.ndarray, np.ndarray, float | None]:
        """Dispatch to the correct OpenCV calibration function based on model.

        Returns (rms, K, D, model_fit_error).
        """

        if self.distortion_model == "opencv_fisheye":
            # Upstream's path: cv::fisheye::calibrate with FIX_SKEW |
            # RECOMPUTE_EXTRINSIC, four coefficients (k0-k3 of the
            # theta-polynomial that Gyroflow's opencv_fisheye model reads).
            K = np.zeros((3, 3), dtype=np.float64)
            D = np.zeros((4, 1), dtype=np.float64)
            flags = (
                cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                | cv2.fisheye.CALIB_FIX_SKEW
            )
            rms, _, _, _, _ = cv2.fisheye.calibrate(
                obj_points, img_points, image_size, K, D,
                flags=flags, criteria=criteria,
            )
            return float(rms), K, D, None

        # poly3 / poly5 / ptlens: fit the fisheye curve, then convert it into
        # the target model by least squares (see _convert_fisheye_coeffs).
        K = np.zeros((3, 3), dtype=np.float64)
        D = np.zeros((4, 1), dtype=np.float64)
        flags = (
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
            | cv2.fisheye.CALIB_FIX_SKEW
        )
        rms, _, _, _, _ = cv2.fisheye.calibrate(
            obj_points, img_points, image_size, K, D,
            flags=flags, criteria=criteria,
        )

        coeffs, fit_error = self._convert_fisheye_coeffs(D, K, image_size)
        return float(rms), K, coeffs, fit_error

    def _convert_fisheye_coeffs(
        self,
        fisheye_D: np.ndarray,
        K: np.ndarray,
        image_size: tuple[int, int],
    ) -> tuple[np.ndarray, float]:
        """Fit the target model to the calibrated fisheye curve.

        The fisheye model is a theta-polynomial — ``r_d = t*(1 + k0*t^2 + ...)``
        with ``r_u = tan(t)`` — while poly3/poly5/ptlens are radial polynomials
        in the undistorted radius.  There is no algebraic reparametrisation
        between them, so the target coefficients come from a least-squares fit
        of the fisheye curve over the radius range the calibration images
        actually cover.

        Only that covered range is meaningful: outside it the fisheye fit is an
        extrapolation and so is any conversion.  Returns ``(coeffs,
        max_relative_error)`` where the error is measured on the undistortion
        scale ``r_u/r_d`` — the quantity the resampler applies — so it says how
        much the substituted model bends the image differently.
        """
        k = np.asarray(fisheye_D, dtype=np.float64).ravel()[:4]
        n_coeffs = _FIT_MODELS[self.distortion_model]

        r_max = self._max_normalized_radius(K, image_size)
        theta = self._theta_for_radius(r_max, k)
        thetas = np.linspace(1e-6, theta, 512)

        t2 = thetas * thetas
        r_d = thetas * (1.0 + k[0] * t2 + k[1] * t2**2 + k[2] * t2**3 + k[3] * t2**4)
        r_u = np.tan(thetas)
        if not np.all(np.isfinite(r_d)) or not np.all(np.isfinite(r_u)):
            raise RuntimeError("Fisheye curve is not finite over the image radius")

        # All three target models are linear in their coefficients once the
        # ratio r_d/r_u - 1 is taken apart.  Columns are ordered so lstsq
        # returns the coefficients in the order the renderer stores them.
        if self.distortion_model == "poly3":
            # r_d = r_u * (1 + k1*r_u^2)
            design = np.stack([r_u**2], axis=-1)
        elif self.distortion_model == "poly5":
            # r_d = r_u * (1 + k1*r_u^2 + k2*r_u^4)
            design = np.stack([r_u**2, r_u**4], axis=-1)
        elif self.distortion_model == "ptlens":
            # r_d = r_u * (1 + c*r_u + b*r_u^2 + a*r_u^3), stored as (a, b, c)
            design = np.stack([r_u**3, r_u**2, r_u], axis=-1)
        else:  # pragma: no cover - guarded by SUPPORTED_MODELS
            raise ValueError(f"No fit for model '{self.distortion_model}'")

        target = r_d / r_u - 1.0
        coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
        coeffs = coeffs[:n_coeffs]

        fitted_d = r_u * (1.0 + design @ coeffs)
        truth_scale = np.where(r_d > 1e-12, r_u / r_d, 1.0)
        fit_scale = np.where(fitted_d > 1e-12, r_u / fitted_d, 1.0)
        scale = np.maximum(np.abs(truth_scale), 1e-9)
        fit_error = float(np.max(np.abs(fit_scale - truth_scale) / scale))

        if fit_error > 0.05:
            logger.warning(
                "Model '%s' reproduces the calibrated fisheye curve to only %.1f%% "
                "over the image radius — it has too few degrees of freedom for this "
                "lens. Prefer opencv_fisheye, or expect distortion at the frame edges.",
                self.distortion_model, fit_error * 100.0,
            )

        return coeffs.reshape(-1, 1), fit_error

    @staticmethod
    def _max_normalized_radius(K: np.ndarray, image_size: tuple[int, int]) -> float:
        """Largest distorted radius in the image, in normalised units."""
        w, h = image_size
        cx, cy = K[0, 2], K[1, 2]
        f = 0.5 * (K[0, 0] + K[1, 1])
        if f <= 0:
            return 0.0
        corners = [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h)]
        return max(float(np.hypot(x - cx, y - cy)) for x, y in corners) / f

    @staticmethod
    def _theta_for_radius(r_d: float, k: np.ndarray) -> float:
        """Invert the fisheye theta-polynomial by bisection."""
        if r_d <= 0:
            return 0.0
        # tan() blows up at pi/2; stay inside the model's valid range.
        hi = min(1.55, np.pi / 2 - 1e-3)
        lo = 0.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            m2 = mid * mid
            value = mid * (1.0 + k[0] * m2 + k[1] * m2**2 + k[2] * m2**3 + k[3] * m2**4)
            if value < r_d:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
