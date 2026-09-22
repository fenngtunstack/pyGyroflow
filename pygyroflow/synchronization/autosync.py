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
    compute_params:
        ``ComputeParams`` for the pose estimators' lens. Upstream builds a
        dedicated one per sync run; ``None`` leaves them on a pinhole camera.
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
        compute_params=None,
        calc_initial_fast: bool = False,
        initial_offset_inv: bool = False,
    ) -> None:
        self._pose_estimator = PoseEstimator()
        self._pose_estimator.set_fps(fps, scaled_fps)
        self._pose_estimator.set_optical_flow_method(of_method)
        self._pose_estimator.set_pose_method(pose_method)
        self._pose_estimator.set_every_nth_frame(every_nth_frame)

        if camera_matrix is not None:
            self._pose_estimator.set_camera_matrix(camera_matrix)
        # The pose estimators need the lens, not just K. Upstream builds a
        # dedicated ComputeParams for the sync run (autosync.rs:86-89) with the
        # keyframes cleared and `lens_correction_amount = 1.0`, and hands it to
        # every estimator — and the RS-aware offset search undistorts its
        # tracks with it too.
        self._compute_params = compute_params
        self._pose_estimator.set_compute_params(compute_params)

        self._offset_method = offset_method
        # SyncParams.calc_initial_fast (rs_sync.rs:15-44): seed the RS search
        # with an essential-matrix estimate and narrow the window to it.
        # SyncParams.initial_offset_inv (autosync.rs:223-247): also try the
        # negated initial offset and keep whichever search does better.
        self._calc_initial_fast = calc_initial_fast
        self._initial_offset_inv = initial_offset_inv
        self._fps = fps
        self._scaled_fps = scaled_fps or fps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def pose_estimator(self) -> PoseEstimator:
        return self._pose_estimator

    def set_lpf(self, freq: float) -> None:
        """Low-pass cutoff for the estimated gyro signal, in Hz (0 = off).

        Upstream exposes this on ``PoseEstimator`` too and, like this one,
        nothing in the CLI drives it — it is a GUI-side knob. It is ported
        and reachable rather than dropped, because a caller that needs it
        (noisy optical flow on a low-contrast clip) has no other way in.
        """
        self._pose_estimator.lowpass_filter(freq, self._fps)

    def run(
        self,
        frames: Sequence[tuple[int, npt.NDArray[np.uint8]]],
        gyro_data: list[tuple[int, np.ndarray]],
        search_range_ms: float = 500.0,
        sample_count: int | None = None,
        progress_callback: Callable[[float], None] | None = None,
        quaternions: dict | None = None,
        frame_readout_time_ms: float = 0.0,
        initial_offset_ms: float = 0.0,
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

        # SyncParams.initial_offset_inv (autosync.rs:223-247): when set and
        # the initial offset is meaningful, run the search for BOTH signs of
        # the initial offset and keep whichever does better — more sync
        # points upstream (the port has one range, so: one that exists beats
        # one that doesn't, then the lower cost).
        check_negative = self._initial_offset_inv and abs(initial_offset_ms) > 1.0

        offset, cost = self._offset_dispatch(
            visual_rots, gyro_data, frames, quaternions, frame_readout_time_ms,
            search_range_ms, initial_offset_ms, progress_callback,
        )
        if check_negative:
            neg_offset, neg_cost = self._offset_dispatch(
                visual_rots, gyro_data, frames, quaternions,
                frame_readout_time_ms, search_range_ms, -initial_offset_ms,
                progress_callback,
            )
            if neg_offset is not None:
                if offset is None:
                    offset, cost = neg_offset, neg_cost
                elif neg_cost is not None and cost is not None and neg_cost < cost:
                    offset, cost = neg_offset, neg_cost

        if offset is not None:
            if progress_callback:
                progress_callback(1.0)
            return offset
        if check_negative:
            # The dispatch already logged per-method fallbacks; a failed
            # ±-trial pair is worth one line of its own.
            logger.warning(
                "Both ±initial-offset trials produced no result "
                "(initial %.2f ms)", initial_offset_ms,
            )

        from pygyroflow.synchronization.find_offset import find_time_offset

        offset = find_time_offset(
            visual_rots,
            gyro_data,
            method=self._offset_method,
            search_range_ms=search_range_ms,
            initial_offset_ms=initial_offset_ms,
            progress_callback=lambda p: (
                progress_callback(0.6 + 0.4 * p)
                if progress_callback
                else None
            ),
        )

        if progress_callback:
            progress_callback(1.0)

        return offset

    def _offset_dispatch(
        self,
        visual_rots: list[tuple[int, np.ndarray]],
        gyro_data: list[tuple[int, np.ndarray]],
        frames,
        quaternions: dict | None,
        frame_readout_time_ms: float,
        search_range_ms: float,
        initial_offset_ms: float,
        progress_callback: Callable[[float], None] | None,
    ) -> tuple[float | None, float | None]:
        """Run the configured offset search once (``estimator.find_offsets``,
        synchronization/mod.rs:384-386). Returns ``(offset, cost)`` — cost
        may be None for searches that don't expose one — or ``(None, None)``
        when the method produced nothing; the correlation fallback stays in
        ``run`` so the ±-trial wrapper can wrap exactly one dispatch."""
        # Rolling-shutter-aware per-point search: uses the matched point
        # pairs retained by the pose estimator plus the gyro quaternion
        # stream. Falls back to the cross-correlation when unavailable.
        if self._offset_method == 2 and quaternions:
            rs_offset = self._rs_sync_offset(
                frames, quaternions, frame_readout_time_ms, search_range_ms,
                initial_offset_ms=initial_offset_ms,
                visual_rots=visual_rots, gyro_data=gyro_data,
            )
            if rs_offset is not None:
                return rs_offset
            logger.warning(
                "RS-aware sync produced no result; "
                "falling back to cross-correlation"
            )

        if self._offset_method == 0:
            result = self._essential_matrix_offset(
                visual_rots,
                gyro_data,
                search_range_ms=search_range_ms,
                initial_offset_ms=initial_offset_ms,
                progress_callback=progress_callback,
            )
            return result if result is not None else (None, None)

        if self._offset_method == 1:
            offset = self._visual_features_offset(
                search_range_ms=search_range_ms,
                initial_offset_ms=initial_offset_ms,
                progress_callback=progress_callback,
            )
            return offset, None
        return None, None

    def _essential_matrix_offset(
        self,
        visual_rots: list[tuple[int, np.ndarray]],
        gyro_data: list[tuple[int, np.ndarray]],
        search_range_ms: float,
        initial_offset_ms: float,
        progress_callback: Callable[[float], None] | None = None,
    ) -> float | None:
        """Upstream's offset method 0 (``essential_matrix::find_offsets``).

        Matches the pose estimator's angular velocity against the raw IMU
        samples with weighted squared error. Returns ``None`` (so the
        caller falls back) when a range carries no movement or the minimum
        lands on the window edge.
        """
        from pygyroflow.types.time_types import TimeIMU

        of_samples = [
            TimeIMU(timestamp_ms=ts / 1000.0, gyro=np.asarray(v, dtype=np.float64))
            for ts, v in visual_rots
        ]
        gyro_samples = [
            TimeIMU(timestamp_ms=ts / 1000.0, gyro=np.asarray(v, dtype=np.float64))
            for ts, v in gyro_data
        ]

        from pygyroflow.synchronization.find_offset.essential_matrix import (
            find_offset_essential_matrix,
        )

        result = find_offset_essential_matrix(
            of_samples,
            gyro_samples,
            search_size_ms=search_range_ms / 2.0,
            initial_offset_ms=initial_offset_ms,
            scaled_fps=self._scaled_fps,
            progress_callback=progress_callback,
        )
        if result is None:
            return None
        return result

    def _visual_features_offset(
        self,
        search_range_ms: float,
        initial_offset_ms: float,
        progress_callback: Callable[[float], None] | None = None,
    ) -> float | None:
        """Upstream's offset method 1: point-pair distance minimization.

        Needs both the matched point pairs retained by the pose estimator
        and a ``ComputeParams`` with the lens and the integrated gyro
        streams. Returns ``None`` (so the caller falls back) when either is
        missing — callers without a lens profile kept working that way.
        """
        if self._compute_params is None:
            return None

        ordered = sorted(
            self._pose_estimator.get_frame_results().values(),
            key=lambda f: f.frame_no,
        )
        pairs = []
        for a, b in zip(ordered, ordered[1:]):
            if a.prev_points is None or a.curr_points is None:
                continue
            if len(a.prev_points) < 2 or len(a.prev_points) != len(a.curr_points):
                continue
            pairs.append((
                (a.timestamp_us, a.prev_points),
                (b.timestamp_us, a.curr_points),
            ))
        if len(pairs) < 3:
            logger.warning(
                "Visual-feature offset search: only %d usable pairs", len(pairs)
            )
            return None

        from pygyroflow.synchronization.find_offset.visual_features import (
            find_offset_visual_features,
        )

        result = find_offset_visual_features(
            pairs,
            self._compute_params,
            search_size_ms=search_range_ms,
            initial_offset_ms=initial_offset_ms,
            progress_callback=progress_callback,
        )
        if result is None:
            return None
        offset_ms, _cost = result
        return offset_ms

    def _rs_sync_offset(
        self,
        frames: Sequence[tuple[int, npt.NDArray[np.uint8]]],
        quaternions: dict,
        frame_readout_time_ms: float,
        search_range_ms: float,
        initial_offset_ms: float = 0.0,
        visual_rots: list[tuple[int, np.ndarray]] | None = None,
        gyro_data: list[tuple[int, np.ndarray]] | None = None,
    ) -> tuple[float, float] | None:
        """Run the rolling-shutter-aware offset search.

        Builds ``RollingShutterSync`` tracks from the pose estimator's
        retained point pairs and runs the coarse-to-fine per-point
        quaternion error minimization. Returns ``(offset, cost)`` in the
        ``visual = gyro + offset`` convention (sign-flipped from the
        internal delay), or None when there is not enough data.

        With ``calc_initial_fast`` set (``rs_sync.rs:15-44``) the search is
        seeded first with the essential-matrix estimate: the median of its
        offsets becomes the initial offset and the window widens to
        ±3000 ms around it — upstream's fast start, which trades a cheap
        pre-pass for a much narrower RS search.
        """
        from pygyroflow.synchronization.find_offset.rs_sync import RollingShutterSync

        if self._calc_initial_fast and visual_rots and gyro_data:
            seeded = self._essential_matrix_offset(
                visual_rots, gyro_data,
                search_range_ms=search_range_ms,
                initial_offset_ms=initial_offset_ms,
            )
            if seeded is not None:
                # Upstream sets search_size = 3000.0 unconditionally — even
                # if that *widens* a smaller user window.
                initial_offset_ms, _seed_cost = seeded
                search_range_ms = 6000.0
                logger.info(
                    "Fast start: essential-matrix offset %.2f ms, "
                    "RS window ±3000 ms", initial_offset_ms,
                )

        ordered = sorted(
            self._pose_estimator.get_frame_results().values(),
            key=lambda f: f.frame_no,
        )
        frame_height, frame_width = frames[0][1].shape[:2]
        height = float(frame_height)

        rs = RollingShutterSync(
            quaternions,
            frame_readout_time_ms=frame_readout_time_ms,
            fps=self._fps,
            scaled_fps=self._scaled_fps,
            compute_params=self._compute_params,
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
                compute_params=self._compute_params,
                points_dims=(frame_width, frame_height),
            )
            added += 1

        if added < 3:
            logger.warning("RS sync: only %d usable tracks (need >= 3)", added)
            return None

        ts_all = [f.timestamp_us for f in ordered if f.timestamp_us > 0]
        if len(ts_all) < 2:
            return None

        result = rs.full_sync(
            initial_delay_ms=-initial_offset_ms,  # delay = -offset convention
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

        # full_sync returns the delay in the same convention as upstream's
        # `offset` (gyro_ts = visual_ts + delay), so a perfect match lands on
        # the initial guess `-initial_offset_ms`.
        #
        # Upstream rejects results outside 90% of the search radius — at the
        # edge of the window the cost minimum is not a real match, it is the
        # boundary. Checked on the raw delay, before the readout term.
        radius_ms = search_range_ms / 2.0
        initial_delay_ms = -initial_offset_ms
        if abs(delay_ms - initial_delay_ms) >= radius_ms * 0.9:
            logger.warning(
                "RS sync: delay %.2f ms is outside 90%% of the search radius "
                "(%.2f ms from the initial guess, limit %.2f); rejecting",
                delay_ms, abs(delay_ms - initial_delay_ms), radius_ms * 0.9,
            )
            return None

        # Then subtract half the sensor readout time: the optical-flow pair
        # sits at the frame's *readout* midpoint, and a rolling shutter takes
        # frame_readout_time to sweep the frame, so the gyro↔frame
        # correspondence is late by half of it. Dropping this term (as this
        # did) biases every synced offset by a constant readout/2 — tens of
        # ms on a slow sensor.
        return -delay_ms - frame_readout_time_ms / 2.0, cost

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
