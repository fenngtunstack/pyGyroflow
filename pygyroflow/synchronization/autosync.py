"""Automatic synchronization orchestrator.

``AutosyncProcess`` drives the full sync pipeline:

1.  Sample keyframes from the video at evenly spaced positions.
2.  For each consecutive frame pair, detect features and estimate camera
    rotation via optical flow.
3.  Convert visual rotations to angular velocity.
4.  Cross-correlate with real gyroscope data to find the optimal time
    offset.

This is the Python counterpart of Gyroflow's ``AutosyncProcess``.
"""

from __future__ import annotations

import logging
from typing import Callable, Sequence

import numpy as np
import numpy.typing as npt

from pygyroflow.synchronization.pose_estimator import PoseEstimator

logger = logging.getLogger(__name__)


class AutosyncProcess:
    """High-level auto-sync driver.

    Parameters
    ----------
    camera_matrix:
        3x3 camera intrinsic matrix.
    fps:
        Video frame rate.
    scaled_fps:
        Scaled frame rate (for slow-motion footage).  Defaults to ``fps``.
    of_method:
        Optical flow method index (0=AKaze, 1=PyrLK, 2=DIS).
    pose_method:
        Pose estimation method index.
    offset_method:
        Offset search method index.
    every_nth_frame:
        Process every N-th frame to speed things up.
    """

    def __init__(
        self,
        camera_matrix: npt.NDArray[np.float64] | None = None,
        fps: float = 30.0,
        scaled_fps: float | None = None,
        of_method: int = 2,
        pose_method: int = 0,
        offset_method: int = 1,
        every_nth_frame: int = 1,
    ) -> None:
        self._pose_estimator = PoseEstimator()
        self._pose_estimator.set_fps(fps, scaled_fps)
        self._pose_estimator.set_optical_flow_method(of_method)
        self._pose_estimator.set_pose_method(pose_method)
        self._pose_estimator.set_every_nth_frame(every_nth_frame)

        if camera_matrix is not None:
            self._pose_estimator.set_camera_matrix(camera_matrix)

        self._offset_method = offset_method
        self._fps = fps
        self._scaled_fps = scaled_fps or fps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def pose_estimator(self) -> PoseEstimator:
        return self._pose_estimator

    def run(
        self,
        frames: Sequence[tuple[int, npt.NDArray[np.uint8]]],
        gyro_data: list[tuple[int, np.ndarray]],
        search_range_ms: float = 500.0,
        sample_count: int | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> float | None:
        """Run the full automatic synchronization pipeline.

        Parameters
        ----------
        frames:
            Sequence of ``(timestamp_us, gray_frame)`` pairs.  The frames
            must be ordered by timestamp.  ``gray_frame`` is (H, W) uint8.
        gyro_data:
            ``[(timestamp_us, angular_velocity_3), ...]`` from the IMU.
            Angular velocity should be in degrees/second.
        search_range_ms:
            Total search window width (milliseconds).
        sample_count:
            If set, subsample *frames* to at most this many evenly spaced
            pairs for faster processing.  ``None`` = use all frames.
        progress_callback:
            ``Callable[[progress: float], None]`` with progress in [0, 1].

        Returns
        -------
        Time offset in milliseconds (``visual = gyro + offset``),
        or ``None`` if sync failed.
        """
        if len(frames) < 2:
            logger.warning("Need at least 2 frames for autosync")
            return None

        self._pose_estimator.clear()

        # Optionally subsample frames
        work_frames = self._subsample_frames(frames, sample_count)
        total = len(work_frames)

        # --- Phase 1: feed frames ---
        for i, (ts, gray) in enumerate(work_frames):
            self._pose_estimator.feed_frame(i, ts, gray)
            if progress_callback and i % 10 == 0:
                progress_callback(0.3 * i / max(total, 1))

        if progress_callback:
            progress_callback(0.3)

        # --- Phase 2: optical flow + pose estimation ---
        self._pose_estimator.process_all()

        if progress_callback:
            progress_callback(0.6)

        # --- Phase 3: offset search ---
        visual_rots = self._pose_estimator.get_visual_rotations()
        if len(visual_rots) < 5:
            logger.warning(
                "Only %d visual rotation estimates (need >= 5)",
                len(visual_rots),
            )
            return None

        from pygyroflow.synchronization.find_offset import find_time_offset

        offset = find_time_offset(
            visual_rots,
            gyro_data,
            method=self._offset_method,
            search_range_ms=search_range_ms,
            progress_callback=lambda p: (
                progress_callback(0.6 + 0.4 * p)
                if progress_callback
                else None
            ),
        )

        if progress_callback:
            progress_callback(1.0)

        return offset

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _subsample_frames(
        frames: Sequence[tuple[int, npt.NDArray[np.uint8]]],
        max_count: int | None,
    ) -> list[tuple[int, npt.NDArray[np.uint8]]]:
        """Return evenly spaced subset of *frames* with at most *max_count*."""
        if max_count is None or len(frames) <= max_count:
            return list(frames)
        indices = np.linspace(0, len(frames) - 1, max_count, dtype=int)
        return [frames[i] for i in indices]
