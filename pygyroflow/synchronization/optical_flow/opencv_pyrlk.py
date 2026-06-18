"""OpenCV Pyramidal Lucas-Kanade (PyrLK) sparse optical flow detector.

Port of Gyroflow's ``OFOpenCVPyrLK``.  The pipeline:

1.  Use Shi-Tomasi corner detection (``goodFeaturesToTrack``) on the
    *previous* frame to find up to 200 high-quality trackable points.
2.  Track those points into the *current* frame using
    ``calcOpticalFlowPyrLK`` with a 21x21 search window and 3 pyramid
    levels.
3.  Filter out points where tracking failed or the result fell outside
    the image bounds.

This is the fastest of the three detectors and works well when
inter-frame motion is moderate.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import numpy.typing as npt

from pygyroflow.synchronization.optical_flow.base import OpticalFlowDetector

logger = logging.getLogger(__name__)

_MIN_POINTS = 10

# goodFeaturesToTrack params (matching Gyroflow defaults)
_MAX_CORNERS = 200
_QUALITY_LEVEL = 0.01
_MIN_DISTANCE = 10.0
_BLOCK_SIZE = 3

# calcOpticalFlowPyrLK params
_WIN_SIZE = (21, 21)
_MAX_LEVEL = 3
_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01)
_MIN_EIG_THRESHOLD = 1e-4


class PyrLKDetector(OpticalFlowDetector):
    """Pyramidal Lucas-Kanade sparse optical flow using OpenCV."""

    def detect_and_track(
        self,
        prev_frame: npt.NDArray[np.uint8],
        curr_frame: npt.NDArray[np.uint8],
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        empty = np.empty((0, 2), dtype=np.float32)
        if prev_frame.size == 0 or curr_frame.size == 0:
            return empty, empty

        prev_gray = self._ensure_gray(prev_frame)
        curr_gray = self._ensure_gray(curr_frame)

        h, w = prev_gray.shape[:2]
        if w == 0 or h == 0:
            return empty, empty

        # Detect features in previous frame
        try:
            prev_pts = cv2.goodFeaturesToTrack(
                prev_gray,
                maxCorners=_MAX_CORNERS,
                qualityLevel=_QUALITY_LEVEL,
                minDistance=_MIN_DISTANCE,
                blockSize=_BLOCK_SIZE,
                useHarrisDetector=False,
                k=0.04,
            )
        except cv2.error as exc:
            logger.warning("goodFeaturesToTrack failed: %s", exc)
            return empty, empty

        if prev_pts is None or len(prev_pts) < _MIN_POINTS:
            return empty, empty

        # Track features into current frame
        try:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                curr_gray,
                prev_pts,
                None,
                winSize=_WIN_SIZE,
                maxLevel=_MAX_LEVEL,
                criteria=_CRITERIA,
                flags=0,
                minEigThreshold=_MIN_EIG_THRESHOLD,
            )
        except cv2.error as exc:
            logger.warning("PyrLK tracking failed: %s", exc)
            return empty, empty

        if curr_pts is None or status is None:
            return empty, empty

        # Filter: keep only successfully tracked points within bounds
        # prev_pts / curr_pts from calcOpticalFlowPyrLK may be (N, 1, 2) -- squeeze
        status_flat = status.ravel().astype(bool)
        prev_filtered = prev_pts[status_flat].reshape(-1, 2)
        curr_filtered = curr_pts[status_flat].reshape(-1, 2)

        # Boundary check
        in_bounds = (
            (curr_filtered[:, 0] >= 0)
            & (curr_filtered[:, 0] < w)
            & (curr_filtered[:, 1] >= 0)
            & (curr_filtered[:, 1] < h)
        )
        prev_filtered = prev_filtered[in_bounds]
        curr_filtered = curr_filtered[in_bounds]

        if len(prev_filtered) < _MIN_POINTS:
            return empty, empty

        return prev_filtered, curr_filtered

    def get_name(self) -> str:
        return "PyrLK"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_gray(
        frame: npt.NDArray[np.uint8],
    ) -> npt.NDArray[np.uint8]:
        if frame.ndim == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame
