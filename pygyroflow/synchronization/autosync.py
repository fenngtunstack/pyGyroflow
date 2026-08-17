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
        quaternions: dict | None = None,
        frame_readout_time_ms: float = 0.0,
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
        quaternions:
            Gyro quaternion stream (timestamp_us -> Quat64).  When provided
            together with ``offset_method=2``, the rolling-shutter-aware
            per-point quaternion search runs instead of the 1-D magnitude
            cross-correlation.
        frame_readout_time_ms:
            Sensor readout time in ms for the RS-aware search.

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
        self._feed_and_estimate(work_frames, progress_callback)

        # --- Phase 3: offset search ---
        visual_rots = self._pose_estimator.get_visual_rotations()
        if len(visual_rots) < 5:
            logger.warning(
                "Only %d visual rotation estimates (need >= 5)",
                len(visual_rots),
            )
            return None

        # Rolling-shutter-aware per-point search: uses the matched point
        # pairs retained by the pose estimator plus the gyro quaternion
        # stream. Falls back to the cross-correlation when unavailable.
        if self._offset_method == 2 and quaternions:
            rs_offset = self._rs_sync_offset(
                frames, quaternions, frame_readout_time_ms, search_range_ms,
            )
            if rs_offset is not None:
                if progress_callback:
                    progress_callback(1.0)
                return rs_offset
            logger.warning(
                "RS-aware sync produced no result; "
                "falling back to cross-correlation"
            )

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

    def _rs_sync_offset(
        self,
        frames: Sequence[tuple[int, npt.NDArray[np.uint8]]],
        quaternions: dict,
        frame_readout_time_ms: float,
        search_range_ms: float,
    ) -> float | None:
        """Run the rolling-shutter-aware offset search.

        Builds ``RollingShutterSync`` tracks from the pose estimator's
        retained point pairs and runs the coarse-to-fine per-point
        quaternion error minimization. Returns the offset in the
        ``visual = gyro + offset`` convention (sign-flipped from the
        internal delay), or None when there is not enough data.
        """
        from pygyroflow.synchronization.find_offset.rs_sync import RollingShutterSync

        ordered = sorted(
            self._pose_estimator.get_frame_results().values(),
            key=lambda f: f.frame_no,
        )
        height = float(frames[0][1].shape[0])

        rs = RollingShutterSync(
            quaternions,
            frame_readout_time_ms=frame_readout_time_ms,
            fps=self._fps,
        )

        added = 0
        for a, b in zip(ordered, ordered[1:]):
            if a.prev_points is None or a.curr_points is None:
                continue
            if len(a.prev_points) < 2:
                continue
            rs.add_track_from_frames(
                a.timestamp_us,
                b.timestamp_us,
                a.prev_points,
                a.curr_points,
                height,
                camera_matrix=self._pose_estimator_camera_matrix(),
            )
            added += 1

        if added < 3:
            logger.warning("RS sync: only %d usable tracks (need >= 3)", added)
            return None

        ts_all = [f.timestamp_us for f in ordered if f.timestamp_us > 0]
        if len(ts_all) < 2:
            return None

        result = rs.full_sync(
            initial_delay_ms=0.0,
            from_ts_us=min(ts_all),
            to_ts_us=max(ts_all),
            coarse_step_ms=3.0,
            search_radius_ms=search_range_ms / 2.0,
        )
        if result is None:
            return None

        cost, delay_ms = result

        # Contrast guard: a flat cost landscape means the tracks carry no
        # usable sync signal (e.g. fast-motion optical-flow breakage); the
        # "minimum" is then noise. Compare the winner against the cost at
        # +/-100 ms and refuse to return an offset that is not
        # distinguishable from its neighbourhood.
        from_ts_s = min(ts_all) / 1e6
        to_ts_s = max(ts_all) / 1e6
        neighbours = [
            rs._compute_cost(delay_ms / 1000.0 + d, from_ts_s, to_ts_s)
            for d in (-0.1, -0.05, 0.05, 0.1)
        ]
        neighbour_med = sorted(neighbours)[len(neighbours) // 2]
        if cost > 0.97 * neighbour_med:
            logger.warning(
                "RS sync: cost landscape is flat (best %.4f vs neighbour "
                "median %.4f); offset not significant, rejecting",
                cost, neighbour_med,
            )
            return None

        # full_sync delay: gyro_ts = visual_ts + delay
        # convention:       visual_ts = gyro_ts + offset  =>  offset = -delay
        return -delay_ms

    def _pose_estimator_camera_matrix(self) -> np.ndarray | None:
        """Camera matrix set on the pose estimator (None if identity)."""
        K = getattr(self._pose_estimator, "_camera_matrix", None)
        if K is None:
            return None
        K = np.asarray(K, dtype=np.float64)
        if np.allclose(K, np.eye(3)):
            return None
        return K

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _feed_and_estimate(
        self,
        work_frames,
        progress_callback: Callable[[float], None] | None = None,
    ) -> None:
        """Feed frames into the pose estimator and run optical flow + pose.

        Phases 1-2 of the pipeline, factored out so both the offset search
        and IMU-orientation guessing can share one estimation pass.
        """
        total = len(work_frames)
        for i, (ts, gray) in enumerate(work_frames):
            self._pose_estimator.feed_frame(i, ts, gray)
            if progress_callback and i % 10 == 0:
                progress_callback(0.3 * i / max(total, 1))

        if progress_callback:
            progress_callback(0.3)

        self._pose_estimator.process_all()

        if progress_callback:
            progress_callback(0.6)

    def run_pose_only(
        self,
        frames,
        progress_callback: Callable[[float], None] | None = None,
    ):
        """Run only the optical-flow + pose estimation phase.

        Returns the pose estimator (with retained point tracks) for
        downstream consumers like IMU-orientation guessing; no offset
        search is performed.
        """
        if len(frames) < 2:
            logger.warning("Need at least 2 frames for pose estimation")
            return None
        self._pose_estimator.clear()
        work_frames = self._subsample_frames(frames, None)
        self._feed_and_estimate(work_frames, progress_callback)
        visual_rots = self._pose_estimator.get_visual_rotations()
        if len(visual_rots) < 5:
            logger.warning("Only %d visual rotation estimates", len(visual_rots))
            return None
        return self._pose_estimator

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
