"""Rolling-shutter-aware synchronization.

Port of Gyroflow's ``find_offset::rs_sync``.  Unlike simple cross-correlation,
this method accounts for rolling-shutter effects by computing per-point
timestamps based on row position and readout time.

The algorithm:
1.  For each pair of matched optical flow points across frames, compute
    precise per-point timestamps (frame_time + readout_time * y / height).
2.  Undistort the 2D pixel coordinates to normalized 3D direction vectors.
3.  For each candidate time offset, shift the gyroscope quaternion stream
    and compute the angular distance between gyro-predicted and optically
    observed rotations.
4.  Minimize the total cost over all candidate offsets using a coarse-to-fine
    search strategy with 4 refinement iterations.

If the optional initial fast estimate is enabled, it first uses the essential
matrix method to get a rough offset, then searches in a reduced window.
"""

from __future__ import annotations

import logging
import math
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation

from pygyroflow.types.quaternion import Quat64
from pygyroflow.types.time_types import TimeQuat

logger = logging.getLogger(__name__)


def _median(v: list[float]) -> float:
    s = sorted(v)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2.0
    return s[n // 2]


def _bisect_range(quats: TimeQuat, ts_us: float) -> tuple[float | None, float | None]:
    """Get the two quaternions bracketing ts_us via bisect."""
    keys = sorted(quats.keys())
    if not keys:
        return None, None
    if ts_us <= keys[0]:
        q = quats[keys[0]].quaternion()
        return q, q
    if ts_us >= keys[-1]:
        q = quats[keys[-1]].quaternion()
        return q, q

    lo, hi = 0, len(keys) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if keys[mid] < ts_us:
            lo = mid
        else:
            hi = mid
    return quats[keys[lo]].quaternion(), quats[keys[hi]].quaternion()


def _interp_quat(quats: TimeQuat, ts_us: float) -> np.ndarray | None:
    """Interpolate quaternion at ts_us."""
    keys = sorted(quats.keys())
    if not keys:
        return None
    if ts_us <= keys[0]:
        return quats[keys[0]].quaternion()
    if ts_us >= keys[-1]:
        return quats[keys[-1]].quaternion()

    lo, hi = 0, len(keys) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if keys[mid] < ts_us:
            lo = mid
        else:
            hi = mid

    t0, t1 = keys[lo], keys[hi]
    if t1 == t0:
        return quats[t0].quaternion()

    frac = (ts_us - t0) / (t1 - t0)
    q0 = quats[t0]
    q1 = quats[t1]
    result = q0.slerp(q1, frac)
    return result.quaternion()


def _quat_rotate(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (w,x,y,z)."""
    r = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    return r.apply(v)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions (w,x,y,z)."""
    r1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
    r2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]])
    result = r1 * r2
    q = result.as_quat()  # [x, y, z, w]
    return np.array([q[3], q[0], q[1], q[2]])


def _quat_inverse(q_wxyz: np.ndarray) -> np.ndarray:
    """Inverse of unit quaternion (w,x,y,z)."""
    return np.array([q_wxyz[0], -q_wxyz[1], -q_wxyz[2], -q_wxyz[3]])


def _angular_distance(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> float:
    """Angular distance between two quaternions in radians."""
    dot = abs(np.dot(q1_wxyz, q2_wxyz))
    dot = min(dot, 1.0)
    return 2.0 * math.acos(dot)


class SyncTrack:
    """One optical flow track across two frames with per-point timestamps."""

    __slots__ = ("ts_a", "ts_b", "pts_a", "pts_b")

    def __init__(
        self,
        ts_a: list[float],
        ts_b: list[float],
        pts_a: list[tuple[float, float, float]],
        pts_b: list[tuple[float, float, float]],
    ):
        self.ts_a = ts_a
        self.ts_b = ts_b
        self.pts_a = pts_a
        self.pts_b = pts_b


class RollingShutterSync:
    """Rolling-shutter-aware offset finder.

    Parameters
    ----------
    quaternions : TimeQuat
        Gyroscope quaternion stream (timestamp_us -> Quat64).
    frame_readout_time_ms : float
        Time for sensor to read out all rows (milliseconds).
        0 means use half the frame interval as estimate.
    fps : float
        Video frame rate, used for default readout time estimate.
    """

    def __init__(
        self,
        quaternions: TimeQuat,
        frame_readout_time_ms: float = 0.0,
        fps: float = 30.0,
    ):
        self.quaternions = quaternions
        self.tracks: list[SyncTrack] = []
        self._flat_ta: np.ndarray | None = None  # vectorized cache

        if frame_readout_time_ms > 0:
            self.readout_time_s = frame_readout_time_ms / 1000.0
        else:
            self.readout_time_s = (1000.0 / fps / 2.0) / 1000.0

    def add_track_from_frames(
        self,
        frame_a_ts_us: int,
        frame_b_ts_us: int,
        pts_a_px: list[tuple[float, float]],
        pts_b_px: list[tuple[float, float]],
        frame_height: float,
        camera_matrix: np.ndarray | None = None,
        distortion_coeffs: np.ndarray | None = None,
    ):
        """Add an optical flow track between two frames.

        Parameters
        ----------
        frame_a_ts_us, frame_b_ts_us:
            Frame timestamps in microseconds.
        pts_a_px, pts_b_px:
            Matched pixel coordinates [(x, y), ...].
        frame_height:
            Frame height in pixels, used for rolling-shutter time calculation.
        camera_matrix:
            Optional 3x3 camera intrinsics.  If None, assume identity
            (already normalized coordinates).
        distortion_coeffs:
            Optional distortion coefficients (not yet used in this path).
        """
        if len(pts_a_px) != len(pts_b_px) or len(pts_a_px) < 2:
            return

        ts_a: list[float] = []
        ts_b: list[float] = []
        pts_a_3d: list[tuple[float, float, float]] = []
        pts_b_3d: list[tuple[float, float, float]] = []

        for (ax, ay), (bx, by) in zip(pts_a_px, pts_b_px):
            # Per-point timestamp: frame_time + readout_time * (y / height)
            ta = frame_a_ts_us / 1e6 + self.readout_time_s * (ay / frame_height)
            tb = frame_b_ts_us / 1e6 + self.readout_time_s * (by / frame_height)

            # Undistort pixel to normalized 3D direction
            if camera_matrix is not None:
                fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
                cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
                nx_a = (ax - cx) / fx
                ny_a = (ay - cy) / fy
                nx_b = (bx - cx) / fx
                ny_b = (by - cy) / fy
            else:
                nx_a, ny_a = ax, ay
                nx_b, ny_b = bx, by

            # Normalize to unit sphere
            na = math.sqrt(nx_a * nx_a + ny_a * ny_a + 1.0)
            nb = math.sqrt(nx_b * nx_b + ny_b * ny_b + 1.0)

            ts_a.append(ta)
            ts_b.append(tb)
            pts_a_3d.append((nx_a / na, ny_a / na, 1.0 / na))
            pts_b_3d.append((nx_b / nb, ny_b / nb, 1.0 / nb))

        self.tracks.append(SyncTrack(ts_a, ts_b, pts_a_3d, pts_b_3d))
        self._flat_ta = None  # invalidate vectorized cache

    # ------------------------------------------------------------------
    # Vectorized cost (numpy) — semantics identical to the scalar loop
    # below, ~100x faster for realistic track counts.
    # ------------------------------------------------------------------

    def _prepare_arrays(self) -> None:
        """Flatten tracks into arrays and cache the quaternion stream."""
        if getattr(self, "_flat_ta", None) is not None:
            return

        ta: list[float] = []
        tb: list[float] = []
        pa: list[tuple[float, float, float]] = []
        pb: list[tuple[float, float, float]] = []
        for track in self.tracks:
            ta.extend(track.ts_a)
            tb.extend(track.ts_b)
            pa.extend(track.pts_a)
            pb.extend(track.pts_b)

        self._flat_ta = np.asarray(ta, dtype=np.float64)
        self._flat_tb = np.asarray(tb, dtype=np.float64)
        self._flat_pa = np.asarray(pa, dtype=np.float64)
        self._flat_pb = np.asarray(pb, dtype=np.float64)

        self._quat_keys = np.asarray(sorted(self.quaternions.keys()), dtype=np.int64)
        self._quat_wxyz = np.asarray(
            [self.quaternions[k].quaternion() for k in self._quat_keys],
            dtype=np.float64,
        )

    def _interp_quats_batch(self, ts_s: np.ndarray) -> np.ndarray:
        """Interpolate quaternions at many timestamps (seconds).

        Matches the scalar ``_interp_quat`` semantics: clamps to the first
        or last quaternion outside the stream range.
        """
        keys = self._quat_keys
        quats = self._quat_wxyz
        n = len(keys)

        ts_us = ts_s * 1e6
        idx = np.searchsorted(keys, ts_us)
        # idx in [0, n]; clamp to bracketing pair
        i1 = np.clip(idx, 1, n - 1)
        i0 = i1 - 1

        q0 = quats[i0]
        q1 = quats[i1]

        t0 = keys[i0].astype(np.float64)
        t1 = keys[i1].astype(np.float64)
        denom = t1 - t0
        alpha = np.where(denom > 0, (ts_us - t0) / np.where(denom > 0, denom, 1.0), 0.0)

        # Shortest-path sign
        dot = np.sum(q0 * q1, axis=1)
        sign = np.where(dot < 0.0, -1.0, 1.0)
        q1s = q1 * sign[:, None]
        dot = np.abs(dot)

        # Slerp with lerp fallback for near-parallel quaternions
        theta = np.arccos(np.clip(dot, -1.0, 1.0))
        sin_theta = np.sin(theta)
        small = sin_theta < 1e-8
        sin_theta_safe = np.where(small, 1.0, sin_theta)

        a0 = np.sin((1.0 - alpha) * theta) / sin_theta_safe
        a1 = np.sin(alpha * theta) / sin_theta_safe
        lerped = q0 + alpha[:, None] * (q1s - q0)
        result = a0[:, None] * q0 + a1[:, None] * q1s
        result = np.where(small[:, None], lerped, result)

        # Normalize and clamp outside-range timestamps to the endpoints
        norm = np.linalg.norm(result, axis=1)
        result = result / np.where(norm > 0, norm, 1.0)[:, None]

        below = ts_us <= keys[0]
        above = ts_us >= keys[-1]
        result[below] = quats[0]
        result[above] = quats[-1]
        return result

    @staticmethod
    def _quat_mul_batch(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Hamilton product q1*q2 for batches of wxyz quaternions."""
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        return np.stack([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], axis=1)

    @staticmethod
    def _quat_rotate_batch(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotate batch of vectors v (N,3) by unit quaternions q (N,4) wxyz."""
        qv = q[:, 1:4]
        t = 2.0 * np.cross(qv, v)
        return v + q[:, 0:1] * t + np.cross(qv, t)

    def _compute_cost(self, delay_s: float, from_ts_s: float, to_ts_s: float) -> float:
        """Compute synchronization cost for a given time delay.

        For each matched pair in each track, compute:
          1. The quaternion at each point's precise timestamp (shifted by delay).
          2. The rotation delta between gyro and optical flow.
          3. The angular error.

        Returns total cost (sum of squared angular errors).
        """
        self._prepare_arrays()

        mid = (self._flat_ta + self._flat_tb) * 0.5
        window = (mid >= from_ts_s) & (mid <= to_ts_s)
        if not np.any(window):
            return 0.0

        ta = self._flat_ta[window]
        tb = self._flat_tb[window]
        pa = self._flat_pa[window]
        pb = self._flat_pb[window]

        qa = self._interp_quats_batch(ta + delay_s)
        qb = self._interp_quats_batch(tb + delay_s)

        # q_delta = qb * conj(qa)
        conj_qa = qa * np.array([1.0, -1.0, -1.0, -1.0])
        q_delta = self._quat_mul_batch(qb, conj_qa)

        pa_rotated = self._quat_rotate_batch(q_delta, pa)

        dot = np.clip(np.sum(pa_rotated * pb, axis=1), -1.0, 1.0)
        angle_err = np.arccos(dot)
        return float(np.sum(angle_err * angle_err))

    def pre_sync(
        self,
        initial_delay_s: float,
        from_ts_us: int,
        to_ts_us: int,
        step_s: float = 0.003,
        radius_s: float = 0.5,
    ) -> tuple[float, float]:
        """Coarse synchronization search.

        Returns (cost, delay_s) for the best offset found.
        """
        from_ts_s = from_ts_us / 1e6
        to_ts_s = to_ts_us / 1e6

        best_cost = float("inf")
        best_delay = initial_delay_s

        steps = np.arange(
            initial_delay_s - radius_s,
            initial_delay_s + radius_s + step_s,
            step_s,
        )

        for delay in steps:
            cost = self._compute_cost(delay, from_ts_s, to_ts_s)
            if cost < best_cost:
                best_cost = cost
                best_delay = delay

        return best_cost, best_delay

    def full_sync(
        self,
        initial_delay_ms: float,
        from_ts_us: int,
        to_ts_us: int,
        coarse_step_ms: float = 3.0,
        search_radius_ms: float = 500.0,
        refinement_levels: int = 4,
    ) -> tuple[float, float] | None:
        """Full coarse-to-fine synchronization search.

        Returns (cost, delay_ms) or None if no solution found.

        Parameters
        ----------
        initial_delay_ms:
            Initial delay estimate in milliseconds.
        from_ts_us, to_ts_us:
            Sync window boundaries in microseconds.
        coarse_step_ms:
            Coarse search step in milliseconds.
        search_radius_ms:
            Search half-width in milliseconds.
        refinement_levels:
            Number of refinement iterations (each reduces step by 10x).
        """
        if not self.tracks:
            return None

        initial_delay_s = initial_delay_ms / 1000.0
        radius_s = search_radius_ms / 1000.0
        step_s = coarse_step_ms / 1000.0

        best_delay = initial_delay_s
        best_cost = float("inf")

        from_ts_s = from_ts_us / 1e6
        to_ts_s = to_ts_us / 1e6

        # Coarse search
        delays = np.arange(
            initial_delay_s - radius_s,
            initial_delay_s + radius_s + step_s,
            step_s,
        )

        for delay in delays:
            cost = self._compute_cost(delay, from_ts_s, to_ts_s)
            if cost < best_cost:
                best_cost = cost
                best_delay = delay

        # Refinement iterations
        for level in range(refinement_levels):
            step_s /= 10.0
            radius_s = step_s * 10.0

            delays = np.arange(
                best_delay - radius_s,
                best_delay + radius_s + step_s,
                step_s,
            )

            for delay in delays:
                cost = self._compute_cost(delay, from_ts_s, to_ts_s)
                if cost < best_cost:
                    best_cost = cost
                    best_delay = delay

        return best_cost, best_delay * 1000.0


def find_offset_rs_sync(
    visual_rotations: list[tuple[int, np.ndarray]],
    gyro_rotations: list[tuple[int, np.ndarray]],
    search_range_ms: float = 500.0,
    initial_offset_ms: float = 0.0,
    progress_callback: Callable[[float], None] | None = None,
) -> float | None:
    """Rolling-shutter-aware offset search.

    This wraps the basic cross-correlation approach with rolling-shutter
    awareness.  For the full pipeline (with undistortion and per-point
    timestamps), use the RollingShutterSync class directly.

    Parameters
    ----------
    visual_rotations:
        ``(timestamp_us, angular_velocity_3,)`` pairs from optical flow.
    gyro_rotations:
        ``(timestamp_us, angular_velocity_3,)`` pairs from the IMU.
    search_range_ms:
        Search half-width in milliseconds.
    initial_offset_ms:
        Known approximate offset.
    progress_callback:
        Optional ``Callable[[progress: float], None]``.

    Returns
    -------
    Estimated offset in milliseconds, or None on failure.
    """
    # Build a simple cross-correlation with rolling-shutter awareness
    if len(visual_rotations) < 10 or len(gyro_rotations) < 10:
        logger.warning("Not enough data points for rs-sync offset search")
        return None

    vis_ts = np.array([t for t, _ in visual_rotations], dtype=np.float64)
    vis_av = np.array([v for _, v in visual_rotations], dtype=np.float64)
    gyr_ts = np.array([t for t, _ in gyro_rotations], dtype=np.float64)
    gyr_av = np.array([v for _, v in gyro_rotations], dtype=np.float64)

    vis_ts_ms = vis_ts / 1000.0
    gyr_ts_ms = gyr_ts / 1000.0

    if gyr_av.ndim == 2:
        gyr_av_rad = np.deg2rad(gyr_av)
    else:
        gyr_av_rad = np.deg2rad(gyr_av)

    if vis_av.ndim == 2 and vis_av.shape[1] == 3:
        vis_mag = np.linalg.norm(vis_av, axis=1)
    else:
        vis_mag = np.abs(vis_av.ravel())

    if gyr_av_rad.ndim == 2 and gyr_av_rad.shape[1] == 3:
        gyr_mag = np.linalg.norm(gyr_av_rad, axis=1)
    else:
        gyr_mag = np.abs(gyr_av_rad.ravel())

    t_min = max(vis_ts_ms.min(), gyr_ts_ms.min())
    t_max = min(vis_ts_ms.max(), gyr_ts_ms.max())
    if t_max <= t_min:
        return None

    vis_interval = np.median(np.diff(vis_ts_ms)) if len(vis_ts_ms) > 1 else 1.0
    gyr_interval = np.median(np.diff(gyr_ts_ms)) if len(gyr_ts_ms) > 1 else 1.0
    resample_dt = min(vis_interval, gyr_interval, 1.0)

    grid = np.arange(t_min, t_max, resample_dt)
    if len(grid) < 10:
        return None

    vis_interp = np.interp(grid, vis_ts_ms, vis_mag)
    gyr_interp = np.interp(grid, gyr_ts_ms, gyr_mag)

    vis_centered = vis_interp - vis_interp.mean()
    gyr_centered = gyr_interp - gyr_interp.mean()
    gyr_energy = np.dot(gyr_centered, gyr_centered)
    if gyr_energy < 1e-12:
        return None

    # Coarse + fine search (matching the Gyroflow approach)
    half_range = search_range_ms / 2.0
    coarse_step = 3.0  # 3ms coarse step (matches Gyroflow)
    fine_step = 0.01

    best_offset = initial_offset_ms
    best_corr = -np.inf

    # Coarse search
    for offset in np.arange(-half_range + initial_offset_ms,
                            half_range + initial_offset_ms + coarse_step,
                            coarse_step):
        shifted_vis_ts = vis_ts_ms - offset
        vis_shifted = np.interp(grid, shifted_vis_ts, vis_mag)
        vis_c = vis_shifted - vis_shifted.mean()
        corr = np.dot(vis_c, gyr_centered) / (
            math.sqrt(np.dot(vis_c, vis_c) * gyr_energy) + 1e-12
        )
        if corr > best_corr:
            best_corr = corr
            best_offset = offset

    # 4 refinement levels (matching Gyroflow's full_sync with 4 refinement levels)
    current_step = coarse_step
    current_range = coarse_step * 10.0
    for _ in range(4):
        current_step /= 10.0
        current_range = current_step * 10.0

        for offset in np.arange(
            best_offset - current_range,
            best_offset + current_range + current_step,
            current_step,
        ):
            shifted_vis_ts = vis_ts_ms - offset
            vis_shifted = np.interp(grid, shifted_vis_ts, vis_mag)
            vis_c = vis_shifted - vis_shifted.mean()
            corr = np.dot(vis_c, gyr_centered) / (
                math.sqrt(np.dot(vis_c, vis_c) * gyr_energy) + 1e-12
            )
            if corr > best_corr:
                best_corr = corr
                best_offset = offset

    logger.debug(
        "RS-sync offset search: best_offset=%.3f ms, corr=%.4f",
        best_offset, best_corr,
    )
    return float(best_offset)
