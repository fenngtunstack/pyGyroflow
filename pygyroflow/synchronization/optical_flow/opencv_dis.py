"""OpenCV Dense Inverse Search (DIS) optical flow detector.

Port of Gyroflow's ``OFOpenCVDis``.  DIS computes a dense per-pixel flow
field, then we sample it on a grid, filtering out low-texture regions
where flow is unreliable.

Algorithm
---------
1.  Compute dense optical flow between two grayscale frames using DIS
    (medium preset for speed/quality trade-off).
2.  Sample points on a uniform grid (~15 points per dimension).
3.  Discard samples in low-texture regions (local gray-level variance
    below a threshold).
4.  For each surviving sample, look up its flow vector (dx, dy) and
    produce a (source, target) point pair.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import numpy.typing as npt

from pygyroflow.synchronization.optical_flow.base import OpticalFlowDetector

logger = logging.getLogger(__name__)

_MIN_POINTS = 10
_GRID_POINTS_PER_AXIS = 15
_TEXTURE_WINDOW_FRACTION = 0.02   # 2 % of image width
_TEXTURE_WINDOW_MIN = 10          # minimum window size in pixels
_TEXTURE_VARIANCE_THRESHOLD = 3.0 # skip flat regions


class DISDetector(OpticalFlowDetector):
    """Dense Inverse Search optical flow using OpenCV."""

    def __init__(self) -> None:
        self._flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

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

        prev_gray = self._ensure_gray(prev_frame)
        curr_gray = self._ensure_gray(curr_frame)

        h, w = prev_gray.shape[:2]
        if w == 0 or h == 0:
            return empty, empty

        # Compute dense flow
        try:
            flow = self._flow.calc(prev_gray, curr_gray, None)
        except cv2.error as exc:
            logger.warning("DIS flow computation failed: %s", exc)
            return empty, empty

        # Grid sampling parameters
        step = max(w // _GRID_POINTS_PER_AXIS, 1)
        win_size = max(int(w * _TEXTURE_WINDOW_FRACTION), _TEXTURE_WINDOW_MIN)
        half_win = win_size // 2

        prev_pts_list: list[tuple[float, float]] = []
        curr_pts_list: list[tuple[float, float]] = []

        for x in range(0, w, step):
            for y in range(0, h, step):
                # Texture check: variance in local window
                if not self._has_enough_texture(prev_gray, x, y, half_win):
                    continue

                dx = flow[y, x, 0]
                dy = flow[y, x, 1]

                target_x = x + dx
                target_y = y + dy

                # Boundary check
                if 0 <= target_x < w and 0 <= target_y < h:
                    prev_pts_list.append((float(x), float(y)))
                    curr_pts_list.append((float(target_x), float(target_y)))

        if len(prev_pts_list) < _MIN_POINTS:
            return empty, empty

        prev_pts = np.array(prev_pts_list, dtype=np.float32)
        curr_pts = np.array(curr_pts_list, dtype=np.float32)
        return prev_pts, curr_pts

    def get_name(self) -> str:
        return "DIS"

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

    @staticmethod
    def _has_enough_texture(
        gray: npt.NDArray[np.uint8],
        cx: int,
        cy: int,
        half_win: int,
    ) -> bool:
        """Return True if local gray-level variance exceeds threshold."""
        h, w = gray.shape[:2]
        y0 = max(cy - half_win, 0)
        y1 = min(cy + half_win + 1, h)
        x0 = max(cx - half_win, 0)
        x1 = min(cx + half_win + 1, w)
        patch = gray[y0:y1, x0:x1].astype(np.float32)
        if patch.size == 0:
            return False
        variance = float(patch.var())
        return variance > _TEXTURE_VARIANCE_THRESHOLD
