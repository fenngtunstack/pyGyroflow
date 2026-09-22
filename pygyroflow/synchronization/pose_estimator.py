"""Pose estimator -- manage per-frame rotation estimation from optical flow.

This is the Python counterpart of Gyroflow's ``PoseEstimator``.  It
stores detected optical flow results for each frame, estimates inter-frame
rotation using the chosen pose method, and converts rotation matrices to
angular velocity (euler) signals that can be cross-correlated with real
gyroscope data for time-offset search.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import numpy.typing as npt

from pygyroflow.synchronization.optical_flow import (
    OpticalFlowDetector,
    create_detector,
)
from pygyroflow.synchronization.estimate_pose import estimate_rotation

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """Per-frame analysis result.

    Attributes
    ----------
    timestamp_us:
        Frame timestamp in microseconds.
    frame_no:
        Sequential frame index.
    rotation:
        Estimated 3x3 rotation matrix (relative to next frame), or None.
    euler_angles:
        Angular velocity as (wx, wy, wz) in radians/second, or None.
        Already scaled by frame rate.
    prev_points:
        Matched feature points in this frame, shape (N, 2).
    curr_points:
        Corresponding points in the *next* frame, shape (N, 2).
    """

    timestamp_us: int
    frame_no: int = 0
    rotation: np.ndarray | None = None
    euler_angles: tuple[float, float, float] | None = None
    prev_points: np.ndarray | None = None
    curr_points: np.ndarray | None = None
    # OF pairs by frame distance (upstream FrameResult.optical_flow):
    # {d: ((ts_us, pts_prev), (ts_us, pts_curr))} for d = 1..N.
    optical_flow: dict[int, tuple] | None = None


class PoseEstimator:
    """Orchestrates optical flow detection and pose estimation across frames.

    Typical usage::

        est = PoseEstimator()
        est.set_camera_matrix(K)

        for frame_no, (ts, gray_frame) in enumerate(frames):
            est.feed_frame(frame_no, ts, gray_frame)

        est.process_all()
        rotations = est.get_visual_rotations()
    """

    def __init__(self) -> None:
        self._frames: dict[int, FrameResult] = {}  # timestamp_us -> result
        self._camera_matrix: np.ndarray = np.eye(3, dtype=np.float64)
        self._fps: float = 30.0
        self._scaled_fps: float = 30.0
        self._every_nth_frame: int = 1
        self._of_method: int = 2  # DIS
        self._pose_method: int = 0  # essential matrix
        self._lpf: float = 0.0  # 0 = no filtering (upstream's default)
        self._compute_params = None  # set_compute_params; None = pinhole
        self._detector: OpticalFlowDetector | None = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_camera_matrix(self, K: npt.NDArray[np.float64]) -> None:
        """Set the 3x3 camera intrinsic matrix."""
        self._camera_matrix = np.asarray(K, dtype=np.float64).reshape(3, 3)

    def set_fps(self, fps: float, scaled_fps: float | None = None) -> None:
        """Set video frame rate (and optionally scaled FPS for slow-mo)."""
        self._fps = float(fps)
        self._scaled_fps = float(scaled_fps) if scaled_fps is not None else float(fps)

    def set_every_nth_frame(self, n: int) -> None:
        """Process every N-th frame (1 = all frames)."""
        self._every_nth_frame = max(1, int(n))

    def set_optical_flow_method(self, method: int) -> None:
        """Set optical flow method index (0=AKaze, 1=PyrLK, 2=DIS)."""
        self._of_method = method
        self._detector = None  # force recreation

    def set_pose_method(self, method: int) -> None:
        """Set pose estimation method index."""
        self._pose_method = method

    def set_compute_params(self, params) -> None:
        """Give the estimators the ``ComputeParams`` upstream hands them.

        Upstream's ``EstimatePoseTrait::init`` receives a full ``ComputeParams``
        and the Almeida estimator reads the lens out of it on every call. The
        port used to pass only ``camera_matrix``, which silently reduced that
        estimator to a pinhole camera. ``None`` (the default) keeps that
        behaviour, so a caller with no lens information is unaffected.
        """
        self._compute_params = params

    def clear(self) -> None:
        """Drop all stored frame results."""
        self._frames.clear()

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def _get_detector(self) -> OpticalFlowDetector:
        if self._detector is None:
            self._detector = create_detector(self._of_method)
        return self._detector

    def feed_frame(
        self,
        frame_no: int,
        timestamp_us: int,
        gray_frame: npt.NDArray[np.uint8],
    ) -> None:
        """Run feature detection on a single frame and store the result.

        Parameters
        ----------
        frame_no:
            Sequential frame index.
        timestamp_us:
            Frame timestamp in microseconds.
        gray_frame:
            Grayscale image (H, W), uint8.
        """
        if timestamp_us in self._frames:
            return  # already processed

        detector = self._get_detector()

        result = FrameResult(
            timestamp_us=timestamp_us,
            frame_no=frame_no,
        )
        self._frames[timestamp_us] = result

        # Store the gray frame temporarily for optical flow computation
        # We attach it as a private attribute; it will be cleaned up after
        # pose estimation.
        result._gray_frame = gray_frame  # type: ignore[attr-defined]

    def process_frame_pair(
        self,
        prev_frame: npt.NDArray[np.uint8],
        curr_frame: npt.NDArray[np.uint8],
        timestamp_us: int,
        camera_matrix: npt.NDArray[np.float64] | None = None,
        optical_flow_method: int | None = None,
        next_timestamp_us: int | None = None,
    ) -> FrameResult | None:
        """Process a single frame pair and estimate rotation.

        This is a convenience method for one-shot use.  For batch
        processing, use ``feed_frame`` + ``process_all`` instead.

        Parameters
        ----------
        prev_frame, curr_frame:
            Consecutive grayscale frames.
        timestamp_us:
            Timestamp of *prev_frame* in microseconds.
        camera_matrix:
            Override camera matrix (uses default if None).
        optical_flow_method:
            Override OF method index.
        next_timestamp_us:
            Timestamp of *curr_frame* in microseconds. The pose estimators
            undistort the two point sets at their own frames' timestamps, so
            a zoom lens wants the real value here; ``None`` reuses
            *timestamp_us*.

        Returns
        -------
        ``FrameResult`` with rotation filled in, or None on failure.
        """
        K = np.asarray(
            camera_matrix if camera_matrix is not None else self._camera_matrix,
            dtype=np.float64,
        ).reshape(3, 3)

        method_idx = (
            optical_flow_method if optical_flow_method is not None else self._of_method
        )
        det = create_detector(method_idx)

        prev_pts, curr_pts = det.detect_and_track(prev_frame, curr_frame)
        if len(prev_pts) < 10:
            return None

        R = estimate_rotation(
            prev_pts, curr_pts, K, method=self._pose_method,
            size_wh=(prev_frame.shape[1], prev_frame.shape[0]),
            params=self._compute_params, timestamp_ms=timestamp_us / 1000.0,
            next_timestamp_ms=(
                next_timestamp_us / 1000.0 if next_timestamp_us is not None else None
            ),
        )
        if R is None:
            return None

        result = FrameResult(
            timestamp_us=timestamp_us,
            rotation=R,
            prev_points=prev_pts,
            curr_points=curr_pts,
        )
        return result

    def process_all(self) -> None:
        """Run optical flow + pose estimation on all fed frames.

        Iterates over frame pairs in ``frame_no`` order, computes optical
        flow between consecutive frames, estimates rotation, and converts
        to angular velocity.
        """
        # Sort frames by frame_no
        sorted_frames = sorted(self._frames.values(), key=lambda f: f.frame_no)
        if len(sorted_frames) < 2:
            return

        detector = self._get_detector()

        for i in range(len(sorted_frames) - 1):
            curr = sorted_frames[i]
            nxt = sorted_frames[i + 1]

            if curr.rotation is not None:
                continue  # already estimated

            # Retrieve cached gray frames
            prev_gray = getattr(curr, "_gray_frame", None)
            next_gray = getattr(nxt, "_gray_frame", None)
            if prev_gray is None or next_gray is None:
                continue

            # Optical flow
            prev_pts, curr_pts = detector.detect_and_track(prev_gray, next_gray)
            if len(prev_pts) < 10:
                continue

            curr.prev_points = prev_pts
            curr.curr_points = curr_pts

            # Pose estimation. The frame size is the optical-flow working
            # size (the frames fed to `feed_frame`, possibly smaller than the
            # video) — the undistortion inside `estimate_rotation` scales the
            # calibration to it — and each point set is undistorted at its
            # own frame's timestamp.
            R = estimate_rotation(
                prev_pts, curr_pts, self._camera_matrix, method=self._pose_method,
                size_wh=(prev_gray.shape[1], prev_gray.shape[0]),
                params=self._compute_params, timestamp_ms=curr.timestamp_us / 1000.0,
                next_timestamp_ms=nxt.timestamp_us / 1000.0,
            )
            if R is not None:
                curr.rotation = R
                # Convert rotation to angular velocity
                euler = self._rotation_to_angular_velocity(R)
                curr.euler_angles = euler

        # Clean up cached gray frames
        for f in sorted_frames:
            if hasattr(f, "_gray_frame"):
                del f._gray_frame

    # ------------------------------------------------------------------
    # Result access
    # ------------------------------------------------------------------

    def get_visual_rotations(
        self, final_pass: bool = True
    ) -> list[tuple[int, np.ndarray]]:
        """Return ``[(timestamp_us, angular_velocity_3), ...]`` for all
        frames with a usable rotation estimate, ordered by time.

        Angular velocity is in degrees/second, matching IMU output format.

        Port of upstream's ``recalculate_gyro_data`` (core/synchronization/
        mod.rs). Three details there are load-bearing and were all missing:

        * **The sample is stamped between this frame and the next**, not on
          the frame. The motion measured between two frames happened during
          the transition from one to the other, so attributing it to either
          endpoint biases the whole timeline by half a frame — which is
          exactly the quantity an offset search is trying to measure.
        * **On the final pass, a frame whose pose estimate failed borrows a
          linearly interpolated euler angle from its nearest successful
          neighbours.** Skipping it instead leaves a hole in the signal the
          correlation has to step over. ``final_pass=False`` is upstream's
          cheap intermediate pass, used only to update the UI while frames
          are still arriving.
        * **An optional low-pass filter** runs over the assembled samples
          (``lowpass_filter(freq, fps)``), zero-phase via forward-backward
          filtering.
        """
        entries = sorted(self._frames.items())  # by timestamp
        eulers: dict[int, tuple[float, float, float] | None] = {
            ts: fr.euler_angles for ts, fr in entries
        }
        if final_pass:
            eulers = self._interpolate_missing(eulers)

        result: list[tuple[int, np.ndarray]] = []
        for index, (ts, _frame) in enumerate(entries):
            eul = eulers.get(ts)
            if eul is None:
                continue
            wx, wy, wz = eul
            # Swap X/Y for the IMU coordinate convention, rad -> deg.
            av = np.array([wy, wx, wz], dtype=np.float64) * (180.0 / np.pi)
            result.append((self._midpoint_us(entries, index), av))

        if self._lpf > 0.0 and self._fps > 0.0:
            result = self._filtered(result)
        return result

    def lowpass_filter(self, freq: float, fps: float | None = None) -> None:
        """Set the low-pass cutoff applied to the estimated gyro signal.

        Mirrors upstream's ``PoseEstimator::lowpass_filter``, including its
        storage: the frequency is kept as hundredths of a Hz in an integer,
        so it is quantised to 0.01 Hz. Filtering only happens when the
        pipeline next rebuilds the signal.
        """
        self._lpf = float(int(freq * 100.0)) / 100.0
        if fps is not None:
            self._fps = float(fps)

    @staticmethod
    def _midpoint_us(
        entries: list[tuple[int, FrameResult]], index: int
    ) -> int:
        """This frame's timestamp moved half way towards the next one.

        The last frame has no successor, so it keeps its own timestamp —
        upstream's ``iter.peek()`` returning None does the same.
        """
        ts = entries[index][0]
        if index + 1 < len(entries):
            next_ts = entries[index + 1][0]
            return int(round(ts + (next_ts - ts) / 2.0))
        return int(ts)

    @staticmethod
    def _interpolate_missing(
        eulers: dict[int, tuple[float, float, float] | None],
    ) -> dict[int, tuple[float, float, float] | None]:
        """Fill gaps by linear interpolation between the nearest known values.

        A gap at either end of the clip is left as it is: there is nothing to
        interpolate from, and extrapolating a motion signal past its data
        would invent a direction rather than a magnitude.
        """
        keys = sorted(eulers)
        known = [k for k in keys if eulers[k] is not None]
        if not known:
            return dict(eulers)

        out = dict(eulers)
        for key in keys:
            if out[key] is not None:
                continue
            previous = next((k for k in reversed(known) if k < key), None)
            following = next((k for k in known if k > key), None)
            if previous is None or following is None:
                continue
            span = following - previous
            if span == 0:
                continue
            ratio = (key - previous) / span
            before, after = eulers[previous], eulers[following]
            out[key] = tuple(
                before[i] + (after[i] - before[i]) * ratio for i in range(3)
            )
        return out

    def _filtered(
        self, result: list[tuple[int, np.ndarray]]
    ) -> list[tuple[int, np.ndarray]]:
        """Zero-phase low-pass over the three angular-velocity channels."""
        from pygyroflow.filtering.lowpass import lowpass_filter_channels

        values = np.stack([av for _, av in result], axis=1)  # (3, N)
        filtered = lowpass_filter_channels(
            values, self._lpf, self._fps, forward_backward=True
        )
        return [
            (ts, np.asarray(filtered[:, i], dtype=np.float64))
            for i, (ts, _) in enumerate(result)
        ]

    def get_frame_results(self) -> dict[int, FrameResult]:
        """Return the full ``{timestamp_us: FrameResult}`` map."""
        return dict(self._frames)

    @staticmethod
    def filter_of_lines(
        lines: tuple[tuple[int, list], tuple[int, list]] | None,
        scale: float,
    ):
        """Drop flow lines whose direction deviates >30° from the mean
        (``synchronization/mod.rs:169-193``) and scale the survivors.

        A "line" is one matched pair: its direction is
        ``atan2(p2.y − p1.y, p2.x − p1.x)``. Averaging angles this way is
        circular-naive — upstream does it too; with the 30° gate it only
        bites when the motion is near the ±π wrap, where the port matches
        upstream's behaviour.
        """
        if lines is None:
            return None
        (ts1, pts1), (ts2, pts2) = lines
        if not len(pts1) or len(pts1) != len(pts2):
            return None
        angles = [
            math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            for p1, p2 in zip(pts1, pts2)
        ]
        avg_angle = sum(angles) / len(angles)
        limit = 30.0 * (math.pi / 180.0)
        out1, out2 = [], []
        for p1, p2, angle in zip(pts1, pts2, angles):
            if abs(angle - avg_angle) < limit:
                out1.append((p1[0] * scale, p1[1] * scale))
                out2.append((p2[0] * scale, p2[1] * scale))
        return ((ts1, out1), (ts2, out2))

    def get_of_lines_for_timestamp(
        self,
        timestamp_us: int,
        next_no: int = 0,
        scale: float = 1.0,
        num_frames: int = 1,
        apply_filter: bool = False,
    ):
        """The cached optical-flow lines at *timestamp_us*
        (``synchronization/mod.rs:228-247``): the frame closest within
        2 ms, then *next_no* frames later, returning that frame's stored
        point pair and its frame size.

        Uncached distances ``d > 1`` return ``(None, None)`` — the caller
        primes the multi-baseline cache with :meth:`cache_optical_flow`
        (B-14) rather than this silently returning the wrong baseline.
        """
        entries = sorted(self._frames)
        if not entries:
            return None, None
        closest = min(entries, key=lambda ts: abs(ts - timestamp_us))
        if abs(closest - timestamp_us) > 2000:
            return None, None
        idx = entries.index(closest) + next_no
        if idx >= len(entries):
            return None, None
        curr = self._frames[entries[idx]]

        # The multi-distance cache first (mod.rs:234-236): the caller
        # primes it with cache_optical_flow.
        cached = (curr.optical_flow or {}).get(num_frames)
        if cached is not None:
            (ts1, p1), (ts2, p2) = cached
            pts = (
                (ts1, [tuple(map(float, p)) for p in p1]),
                (ts2, [tuple(map(float, p)) for p in p2]),
            )
            if apply_filter:
                pts = self.filter_of_lines(pts, scale)
            frame_size = getattr(curr, "frame_size", None)
            return pts, tuple(frame_size) if frame_size else None

        # No cache entry: only the estimator's own d=1 pair is available
        # without a detector — return None rather than the wrong baseline.
        if num_frames != 1:
            return None, None
        if curr.prev_points is None or curr.curr_points is None:
            return None, None
        pts = (
            (curr.timestamp_us,
             [tuple(map(float, p)) for p in curr.prev_points]),
            (curr.timestamp_us,
             [tuple(map(float, p)) for p in curr.curr_points]),
        )
        if apply_filter:
            pts = self.filter_of_lines(pts, scale)
        frame_size = getattr(curr, "frame_size", None)
        return pts, tuple(frame_size) if frame_size else None

    def cache_optical_flow(self, num_frames: int = 1,
                           detector=None) -> None:
        """Fill each frame's OF cache for distances 1..num_frames
        (``synchronization/mod.rs:195-220``).

        Distance d pairs a frame with the frame d indices later
        (``frame_no + d`` — index arithmetic, not timestamps: dropped
        frames break the chain there). The d=1 entry is the pair the
        estimator already stored; larger distances re-run detection on
        the retained gray frames when *detector* is given (upstream
        reuses the per-frame OF method's detector; the port's estimator
        discards it after processing, so callers pass one back).
        """
        keys = sorted(self._frames)
        for i, ts in enumerate(keys):
            frame = self._frames[ts]
            if frame.optical_flow:
                continue  # already cached
            frame.optical_flow = {}
            for d in range(1, num_frames + 1):
                to_ts = keys[i + d] if i + d < len(keys) else None
                if to_ts is None:
                    continue
                to_frame = self._frames[to_ts]
                if frame.frame_no + d != to_frame.frame_no:
                    continue
                if d == 1 and frame.prev_points is not None \
                        and frame.curr_points is not None:
                    frame.optical_flow[d] = (
                        (frame.timestamp_us, frame.prev_points),
                        (to_frame.timestamp_us, frame.curr_points),
                    )
                    continue
                prev_gray = getattr(frame, "_gray_frame", None)
                next_gray = getattr(to_frame, "_gray_frame", None)
                if prev_gray is None or next_gray is None:
                    continue
                det = detector() if detector is not None else None
                if det is None:
                    continue
                pts1, pts2 = det.detect_and_track(prev_gray, next_gray)
                if len(pts1) < 2:
                    continue
                frame.optical_flow[d] = (
                    (frame.timestamp_us, pts1),
                    (to_frame.timestamp_us, pts2),
                )

    def cleanup(self) -> None:
        """Drop the cached gray frames (upstream ``cleanup``,
        ``mod.rs:221-226`` — release the image memory the OF caches
        held; the point pairs stay)."""
        for frame in self._frames.values():
            if hasattr(frame, "_gray_frame"):
                del frame._gray_frame

    def get_ranges(self) -> list[tuple[int, int]]:
        """Contiguous frame ranges, split at >100 ms gaps
        (``synchronization/mod.rs:363-378``).

        A dropped-frames hole in the middle of a clip is not one long
        track: motion across the hole is unobserved, and the per-range
        offset searches upstream runs must not bridge it.
        """
        ranges: list[tuple[int, int]] = []
        prev_ts = 0
        curr_range_start = 0
        for f in sorted(self._frames):
            if f - prev_ts > 100_000:  # 100 ms
                if curr_range_start != prev_ts:
                    ranges.append((curr_range_start, prev_ts))
                curr_range_start = f
            prev_ts = f
        if curr_range_start != prev_ts:
            ranges.append((curr_range_start, prev_ts))
        return ranges

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rotation_to_angular_velocity(
        self,
        R: np.ndarray,
    ) -> tuple[float, float, float]:
        """Convert a 3x3 rotation matrix to angular velocity (rad/s).

        Uses the rotation-vector (axis * angle) representation, scaled by
        the effective frame rate.  This matches Gyroflow's approach:
        ``rot.scaled_axis() * (scaled_fps / every_nth_frame)``.
        """
        # Compute rotation vector via cv2.Rodrigues
        import cv2

        rot_vec, _ = cv2.Rodrigues(R.astype(np.float64))
        scale = self._scaled_fps / self._every_nth_frame
        wx = float(rot_vec[0, 0]) * scale
        wy = float(rot_vec[1, 0]) * scale
        wz = float(rot_vec[2, 0]) * scale
        return (wx, wy, wz)
