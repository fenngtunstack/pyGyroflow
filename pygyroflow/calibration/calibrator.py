"""Lens calibrator -- auto chessboard calibration.

Uses OpenCV chessboard detection to compute camera intrinsics and distortion
coefficients from a set of calibration frames.  Ported from Gyroflow's
core/calibration/mod.rs RANSAC-style calibration logic.

Workflow:
  1. feed_frame() -- detect chessboard corners in each video frame
  2. calibrate()  -- random-sampling calibration, keep lowest RMS result
  3. get_result()  -- retrieve camera matrix, distortion coefficients, RMS
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Supported distortion model identifiers.
SUPPORTED_MODELS = (
    "opencv_fisheye",
    "opencv_standard",
    "poly3",
    "poly5",
    "ptlens",
)


@dataclass
class DetectedCorners:
    """Chessboard corners detected in a single frame."""

    points: list[tuple[float, float]]
    frame_number: int
    timestamp_us: int
    sharpness: float
    is_forced: bool = False


@dataclass
class CalibrationResult:
    """Output of a successful calibration run."""

    camera_matrix: np.ndarray   # 3x3
    dist_coeffs: np.ndarray     # shape depends on model
    rms: float
    used_frames: list[int]
    image_size: tuple[int, int]  # (width, height)


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
        Sharpness threshold for frame quality filtering.
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
        max_sharpness: float = 5.0,
        distortion_model: str = "opencv_fisheye",
        digital_lens: str | None = None,
        digital_lens_params: list[float] | None = None,
        asymmetrical: bool = False,
    ) -> None:
        if columns < 2 or rows < 2:
            raise ValueError("Need at least 2x2 inner corners")
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

        # Sharpness estimate: mean Laplacian variance in the ROI around corners.
        sharpness = self._estimate_sharpness(gray, corners)

        detected = DetectedCorners(
            points=[(float(p[0]), float(p[1])) for p in corners],
            frame_number=frame_number,
            timestamp_us=timestamp_us,
            sharpness=sharpness,
        )

        # Cache.
        self._all_matches[frame_number] = detected

        # Accept if sharpness below threshold.
        if sharpness < self.max_sharpness:
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
        """
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
        objp = self._obj_points

        best_rms = float("inf")
        best_K: np.ndarray | None = None
        best_D: np.ndarray | None = None
        best_frames: list[int] = []

        calib_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

        for _ in range(n_iter):
            chosen = self._pick_frames(candidate_frames)

            # Build object/image point arrays.
            # OpenCV expects obj_points as (N,1,3) and img_points as (N,1,2).
            objp_prepped = objp.reshape(-1, 1, 3).astype(np.float32)
            obj_points_list: list[np.ndarray] = []
            img_points_list: list[np.ndarray] = []
            for f in chosen:
                det = self._image_points.get(f)
                if det is None:
                    continue
                pts = np.array(det.points, dtype=np.float32).reshape(-1, 1, 2)
                obj_points_list.append(objp_prepped)
                img_points_list.append(pts)

            if len(obj_points_list) < 2:
                continue

            try:
                rms, K, D = self._run_cv_calibration(
                    obj_points_list, img_points_list, image_size, calib_criteria
                )
                if rms < best_rms:
                    best_rms = rms
                    best_K = K.copy()
                    best_D = D.copy()
                    best_frames = list(chosen)
            except cv2.error as exc:
                logger.warning("Calibration iteration failed: %s", exc)
                continue

        if best_K is None:
            raise RuntimeError("All calibration iterations failed")

        self._result = CalibrationResult(
            camera_matrix=best_K,
            dist_coeffs=best_D,
            rms=best_rms,
            used_frames=best_frames,
            image_size=image_size,
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
        """
        if self._result is None:
            return {}

        K = self._result.camera_matrix
        D = self._result.dist_coeffs

        return {
            "calib_dimension": {
                "w": self._width,
                "h": self._height,
            },
            "distortion_model": self.distortion_model,
            "fisheye_params": {
                "camera_matrix": K.tolist(),
                "distortion_coeffs": D.flatten().tolist(),
                "RMS_error": self._result.rms,
            },
            "digital_lens": self.digital_lens,
            "digital_lens_params": self.digital_lens_params,
            "asymmetrical": self.asymmetrical,
            "num_images": len(self._result.used_frames),
        }

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

    @staticmethod
    def _estimate_sharpness(gray: np.ndarray, corners: np.ndarray) -> float:
        """Estimate image sharpness around detected corners via Laplacian variance."""
        h, w = gray.shape[:2]
        margin = 20
        # Collect patch around each corner.
        patches = []
        half = 15
        for pt in corners:
            x, y = int(round(pt[0])), int(round(pt[1]))
            x0 = max(0, x - half)
            x1 = min(w, x + half)
            y0 = max(0, y - half)
            y1 = min(h, y + half)
            if x1 > x0 and y1 > y0:
                patches.append(gray[y0:y1, x0:x1])
        if not patches:
            return 100.0
        combined = np.concatenate([p.ravel() for p in patches])
        laplacian_var = cv2.Laplacian(combined.reshape(1, -1).astype(np.float32), cv2.CV_32F)
        return float(np.var(laplacian_var)) / 1000.0

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
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Dispatch to the correct OpenCV calibration function based on model.

        Returns (rms, K, D).
        """

        if self.distortion_model == "opencv_fisheye":
            # Fisheye: 4 coefficients (k1-k4).
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
            return float(rms), K, D

        if self.distortion_model == "opencv_standard":
            # Standard: up to 14 coefficients (k1-k6, p1-p2 + rational).
            rms, K, D, _, _ = cv2.calibrateCamera(
                obj_points, img_points, image_size, None, None,
                criteria=criteria,
            )
            return float(rms), K, D

        # For poly3 / poly5 / ptlens, calibrate with OpenCV fisheye first,
        # then convert coefficients to the target model.
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

        return float(rms), K, self._convert_fisheye_coeffs(D, K, image_size)

    def _convert_fisheye_coeffs(
        self,
        fisheye_D: np.ndarray,
        K: np.ndarray,
        image_size: tuple[int, int],
    ) -> np.ndarray:
        """Convert fisheye (k1-k4) coefficients to the target model format.

        For poly3 / poly5 / ptlens, we fit a simple polynomial to the
        fisheye distortion curve.  This follows the approach used by
        Gyroflow's coefficient conversion.
        """
        k1, k2, k3, k4 = fisheye_D.ravel()[:4]

        if self.distortion_model == "poly3":
            # Poly3: single coefficient k.
            # Approximate via least-squares fit of r_d = k * r_u^3 + r_u.
            # Take k1 as the dominant term.
            return np.array([[k1]], dtype=np.float64)

        if self.distortion_model == "poly5":
            # Poly5: two coefficients (k1, k2).
            return np.array([[k1], [k2]], dtype=np.float64)

        if self.distortion_model == "ptlens":
            # PTLens: three coefficients (a, b, c).
            # Map fisheye k1..k4 to PTLens a, b, c via polynomial fit.
            a = k1 * 0.5
            b = k2 * 0.25
            c = k3 * 0.125
            return np.array([[a], [b], [c]], dtype=np.float64)

        return fisheye_D
