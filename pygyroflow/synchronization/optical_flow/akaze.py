"""AKAZE feature detector + brute-force Hamming matching.

Port of Gyroflow's ``OFAkaze`` which uses a pure-Rust AKAZE implementation.
Here we use OpenCV's AKAZE with the same overall pipeline:

1.  Detect keypoints and compute binary descriptors on each frame.
2.  KNN-match descriptors (k=2).
3.  Apply Lowe's ratio test (threshold 0.7) to reject ambiguous matches.
4.  Return surviving point pairs.

The descriptor threshold is tunable via the constructor; lower values yield
more keypoints at higher computational cost.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import numpy.typing as npt

from pygyroflow.synchronization.optical_flow.base import OpticalFlowDetector

logger = logging.getLogger(__name__)

# Minimum number of matched pairs required to produce a valid result.
_MIN_MATCHES = 10


class AKazeDetector(OpticalFlowDetector):
    """AKAZE-based optical flow detector using OpenCV."""

    def __init__(self, threshold: float = 0.001) -> None:
        self._detector = cv2.AKAZE_create(threshold=threshold)
        self._matcher = cv2.DescriptorMatcher_create(
            cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING,
        )
        self._threshold = threshold

    # ------------------------------------------------------------------
    # OpticalFlowDetector interface
    # ------------------------------------------------------------------

    def detect_and_track(
        self,
        prev_frame: npt.NDArray[np.uint8],
        curr_frame: npt.NDArray[np.uint8],
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        empty = np.empty((0, 2), dtype=np.float32)
        if prev_frame.size == 0 or curr_frame.size == 0:
            return empty, empty

        # Ensure single-channel uint8 input
        prev_gray = self._ensure_gray(prev_frame)
        curr_gray = self._ensure_gray(curr_frame)

        # Detect & describe
        kp1, des1 = self._detector.detectAndCompute(prev_gray, None)
        kp2, des2 = self._detector.detectAndCompute(curr_gray, None)

        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
            return empty, empty

        # Match descriptors (knn, k=2) for Lowe's ratio test
        try:
            matches = self._matcher.knnMatch(des1, des2, k=2)
        except cv2.error as exc:
            logger.warning("AKAZE matching failed: %s", exc)
            return empty, empty

        # Lowe's ratio test
        good_indices: list[tuple[int, int]] = []
        for m_pair in matches:
            if len(m_pair) < 2:
                continue
            m, n = m_pair
            if m.distance < 0.7 * n.distance:
                good_indices.append((m.queryIdx, m.trainIdx))

        if len(good_indices) < _MIN_MATCHES:
            return empty, empty

        # Extract point coordinates
        prev_pts = np.array(
            [kp1[qi].pt for qi, _ in good_indices],
            dtype=np.float32,
        )
        curr_pts = np.array(
            [kp2[ti].pt for _, ti in good_indices],
            dtype=np.float32,
        )
        return prev_pts, curr_pts

    def get_name(self) -> str:
        return f"AKaze(threshold={self._threshold})"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_gray(
        frame: npt.NDArray[np.uint8],
    ) -> npt.NDArray[np.uint8]:
        """Convert to grayscale if the frame has 3 channels."""
        if frame.ndim == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame
