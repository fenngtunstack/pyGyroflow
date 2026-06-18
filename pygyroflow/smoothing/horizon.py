"""Horizon lock post-processor -- corrects roll/pitch using gravity vectors.

Port of Gyroflow's core/smoothing/horizon.rs.

Works as a post-processing step after any smoothing algorithm. Corrects the
camera orientation to keep the horizon level. Two modes:
- Gravity vector mode: uses IMU accelerometer data for accurate horizon reference
- Pure gyroscope mode: extracts yaw/pitch from quaternion and applies fixed roll

Automatic lock mode dynamically adjusts the roll angle based on roll rate,
allowing natural banking during turns (like coordinated aircraft turns).
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from pygyroflow.keyframes import KeyframeManager, KeyframeType
from pygyroflow.types.quaternion import Quat64
from pygyroflow.types.time_types import TimeQuat, TimeVec


def _lock_horizon_angle(
    q: Quat64,
    roll_correction: float,
    lock_pitch: bool,
    pitch_correction: float,
) -> Quat64:
    """Compute horizon-locked quaternion from a given quaternion.

    Matches the Rust lock_horizon_angle function.

    Algorithm:
    1. Apply quaternion to Z axis to get a "forward" vector
    2. Extract pitch and yaw from the forward vector
    3. Replace roll with the correction angle
    4. Rebuild quaternion from corrected euler angles

    Args:
        q: Input quaternion (smoothed).
        roll_correction: Forced roll angle in radians.
        lock_pitch: Whether to also lock the pitch angle.
        pitch_correction: Forced pitch angle in radians (used if lock_pitch).

    Returns:
        Corrected quaternion.
    """
    # Apply quaternion to Z axis to get "forward" vector
    mat = q.to_rotation_matrix()
    test_vec = mat @ np.array([0.0, 0.0, 1.0])

    # Extract pitch: asin(-test_vec.z) unless locked
    if lock_pitch:
        pitch = pitch_correction
    else:
        pitch = math.asin(-test_vec[2])

    # Extract yaw: atan2(y, x)
    yaw = math.atan2(test_vec[1], test_vec[0])

    # Build rotations for yaw (Y axis), pitch (X axis), roll (Z axis)
    def _rot_x(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)

    def _rot_y(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)

    def _rot_z(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

    rot_yaw = _rot_y(yaw)
    rot_pitch = _rot_x(pitch)
    rot_roll = _rot_z(roll_correction)

    # Coordinate system correction: Y90 * Z90
    initial_mat = _rot_y(math.pi / 2.0) @ _rot_z(math.pi / 2.0)

    # Combine: correction * yaw * pitch * roll
    combined = initial_mat @ rot_yaw @ rot_pitch @ rot_roll
    return Quat64.from_rotation_matrix(combined)


def _interpolate_gravity_vector(
    gravs: TimeVec, timestamp_us: int
) -> Optional[np.ndarray]:
    """Linearly interpolate gravity vector at a given timestamp.

    Args:
        gravs: Gravity vector time series (keys = microseconds).
        timestamp_us: Query timestamp in microseconds.

    Returns:
        Interpolated 3D gravity vector, or None if not possible.
    """
    if not gravs:
        return None

    ts_list = sorted(gravs.keys())
    n = len(ts_list)

    if n == 1:
        return gravs[ts_list[0]].copy()

    # Clamp to data range
    lookup_ts = max(ts_list[0], min(ts_list[-1], timestamp_us))

    # Find the data point at or before lookup_ts
    # Binary search
    lo, hi = 0, n - 1
    idx1 = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if ts_list[mid] <= lookup_ts:
            idx1 = mid
            lo = mid + 1
        else:
            hi = mid - 1

    if ts_list[idx1] == lookup_ts:
        return gravs[ts_list[idx1]].copy()

    # Find the next data point after lookup_ts
    if idx1 + 1 < n:
        idx2 = idx1 + 1
    else:
        return gravs[ts_list[idx1]].copy()

    ts1 = ts_list[idx1]
    ts2 = ts_list[idx2]
    time_delta = ts2 - ts1
    if time_delta == 0:
        return gravs[ts1].copy()

    fract = (timestamp_us - ts1) / time_delta
    v1 = gravs[ts1]
    v2 = gravs[ts2]
    return v1 + (v2 - v1) * fract


class HorizonLock:
    """Horizon lock post-processor configuration and execution.

    Parameters:
        lock_enabled: Whether horizon lock is active.
        horizonlockpercent: Lock strength 0-100 (0=off, 100=full lock).
        horizonroll: Manual roll correction in degrees.
        lock_pitch: Whether to also lock pitch.
        horizonpitch: Manual pitch correction in degrees.
        automatic_lock: Enable automatic roll rate detection for natural banking.
        turn_threshold: Roll rate threshold (deg/s) to trigger dynamic tilt.
        turn_smoothing_ms: Smoothing time for dynamic tilt changes (ms).
        turn_multiplier: Scaling factor for dynamic tilt angle.
        tilt_accel_limit: Max tilt change rate (deg/s^2), inf = unlimited.
    """

    def __init__(self) -> None:
        self.lock_enabled: bool = False
        self.horizonlockpercent: float = 100.0
        self.horizonroll: float = 0.0
        self.lock_pitch: bool = False
        self.horizonpitch: float = 0.0
        self.automatic_lock: bool = False
        self.turn_threshold: float = 5.0
        self.turn_smoothing_ms: float = 500.0
        self.turn_multiplier: float = 1.0
        self.tilt_accel_limit: float = math.inf

    def set_horizon(
        self,
        lock_percent: float,
        roll: float,
        lock_pitch: bool,
        pitch: float,
        automatic_lock: bool = False,
        turn_threshold: float = 5.0,
        turn_smoothing_ms: float = 500.0,
        turn_multiplier: float = 1.0,
        tilt_accel_limit: float = math.inf,
    ) -> None:
        """Set all horizon lock parameters at once."""
        self.horizonroll = roll
        self.horizonlockpercent = lock_percent
        self.lock_enabled = self.horizonlockpercent > 1e-6
        self.horizonpitch = pitch
        self.lock_pitch = lock_pitch
        self.automatic_lock = automatic_lock
        self.turn_threshold = turn_threshold
        self.turn_smoothing_ms = turn_smoothing_ms
        self.turn_multiplier = turn_multiplier
        self.tilt_accel_limit = tilt_accel_limit

    def get_checksum(self) -> int:
        return hash((
            self.horizonlockpercent,
            self.horizonroll,
            self.lock_pitch,
            self.horizonpitch,
            self.turn_threshold,
            self.turn_smoothing_ms,
            self.turn_multiplier,
            self.tilt_accel_limit,
        ))

    def lock(
        self,
        quats: TimeQuat,
        org_quats: TimeQuat,
        grav: Optional[TimeVec],
        use_grav: bool,
        compute_params: Any,
    ) -> None:
        """Apply horizon lock correction in-place to quats.

        Args:
            quats: Smoothed quaternions (modified in place).
            org_quats: Original (unsmoothed) quaternions for roll rate computation.
            grav: Optional gravity vector time series.
            use_grav: Whether to use gravity vector mode.
            compute_params: Computation parameters (keyframes, etc.).
        """
        keyframes: KeyframeManager = compute_params.keyframes

        if not self.lock_enabled and not keyframes.is_keyframed(
            KeyframeType.LockHorizonAmount
        ):
            return

        tau_s: float = self.turn_smoothing_ms / 1000.0

        # ========== Automatic lock: compute roll rates ==========
        roll_rates: dict[int, float] = {}
        if self.automatic_lock:
            prev_roll: Optional[float] = None
            prev_ts: Optional[int] = None
            prev_smoothed: Optional[float] = None

            org_ts_sorted = sorted(org_quats.keys())
            for ts in org_ts_sorted:
                org_quat = org_quats[ts]
                current_euler = org_quat.euler_angles()
                current_roll: float = current_euler[2]  # Roll

                if prev_roll is not None and prev_ts is not None:
                    dt = (ts - prev_ts) / 1_000_000.0
                    if dt > 0.0 and dt < 1.0:
                        # Handle angle wrapping
                        diff_deg = math.degrees(current_roll - prev_roll)
                        while diff_deg > 180.0:
                            diff_deg -= 360.0
                        while diff_deg < -180.0:
                            diff_deg += 360.0
                        rate = diff_deg / dt

                        alpha = dt / (tau_s + dt) if tau_s > 0.0 else 1.0
                        if prev_smoothed is not None:
                            smoothed = prev_smoothed * (1.0 - alpha) + rate * alpha
                        else:
                            smoothed = rate
                        prev_smoothed = smoothed
                        roll_rates[ts] = smoothed

                prev_roll = current_roll
                prev_ts = ts

        ts_sorted = sorted(quats.keys())

        # ========== Gravity vector mode ==========
        if grav is not None and grav and use_grav:
            prev_tilt_smoothed: Optional[float] = None
            prev_tilt_ts: Optional[int] = None
            y_axis = np.array([0.0, 1.0, 0.0])

            for ts in ts_sorted:
                smoothed_ori = quats[ts]

                gv = _interpolate_gravity_vector(grav, ts)
                if gv is None:
                    gv = y_axis.copy()

                ori = org_quats.get(ts, smoothed_ori)
                ori_mat = ori.to_rotation_matrix()
                smoothed_mat = smoothed_ori.to_rotation_matrix()

                # Correction from original to smoothed
                correction = np.linalg.inv(ori_mat) @ smoothed_mat
                angle_corr = math.atan2(-correction[0, 1], correction[0, 0])

                # Get keyframed parameters
                timestamp_ms = ts / 1000.0
                video_rotation = keyframes.value_at_gyro_timestamp(
                    KeyframeType.VideoRotation, timestamp_ms
                )
                if video_rotation is None:
                    video_rotation = getattr(compute_params, "video_rotation", 0.0)

                horizonroll = keyframes.value_at_gyro_timestamp(
                    KeyframeType.LockHorizonRoll, timestamp_ms
                )
                if horizonroll is None:
                    horizonroll = self.horizonroll
                horizonroll += video_rotation

                horizonlockpercent = keyframes.value_at_gyro_timestamp(
                    KeyframeType.LockHorizonAmount, timestamp_ms
                )
                if horizonlockpercent is None:
                    horizonlockpercent = self.horizonlockpercent

                # Automatic lock: dynamic tilt
                dynamic_tilt_deg = 0.0
                if self.automatic_lock:
                    target = 0.0
                    if ts in roll_rates:
                        rr = roll_rates[ts]
                        if abs(rr) > self.turn_threshold:
                            target = rr * self.turn_multiplier

                    if prev_tilt_ts is not None:
                        dt_tilt = (ts - prev_tilt_ts) / 1_000_000.0
                        alpha_tilt = (
                            max(0.0, min(1.0, dt_tilt / (tau_s + dt_tilt)))
                            if tau_s > 0.0
                            else 1.0
                        )
                    else:
                        alpha_tilt = 1.0

                    if prev_tilt_smoothed is not None:
                        smoothed_tilt = prev_tilt_smoothed * (1.0 - alpha_tilt) + target * alpha_tilt
                    else:
                        smoothed_tilt = target

                    # Accel limit
                    accel_limited = smoothed_tilt
                    if math.isfinite(self.tilt_accel_limit) and prev_tilt_smoothed is not None and prev_tilt_ts is not None:
                        dt_acc = (ts - prev_tilt_ts) / 1_000_000.0
                        if dt_acc > 0.0:
                            max_change = self.tilt_accel_limit * dt_acc
                            change = smoothed_tilt - prev_tilt_smoothed
                            if abs(change) > max_change:
                                accel_limited = prev_tilt_smoothed + math.copysign(max_change, change)

                    prev_tilt_smoothed = accel_limited
                    prev_tilt_ts = ts
                    dynamic_tilt_deg = accel_limited

                total_horizonroll_deg = horizonroll + dynamic_tilt_deg

                # Gravity-based horizon correction
                locked_mat = (
                    smoothed_mat
                    @ _rot_z_matrix(
                        -angle_corr
                        + math.atan2(gv[0], gv[1])
                        + total_horizonroll_deg * math.pi / 180.0
                    )
                )
                locked_q = Quat64.from_rotation_matrix(locked_mat)
                quats[ts] = locked_q.slerp(smoothed_ori, 1.0 - horizonlockpercent / 100.0)

            return

        # ========== Pure gyroscope mode (no gravity vectors) ==========
        prev_tilt_smoothed: Optional[float] = None
        prev_tilt_ts: Optional[int] = None

        for ts in ts_sorted:
            smoothed_ori = quats[ts]

            # Get keyframed parameters
            timestamp_ms = ts / 1000.0
            video_rotation = keyframes.value_at_gyro_timestamp(
                KeyframeType.VideoRotation, timestamp_ms
            )
            if video_rotation is None:
                video_rotation = getattr(compute_params, "video_rotation", 0.0)

            horizonroll = keyframes.value_at_gyro_timestamp(
                KeyframeType.LockHorizonRoll, timestamp_ms
            )
            if horizonroll is None:
                horizonroll = self.horizonroll
            horizonroll += video_rotation

            horizonpitch = keyframes.value_at_gyro_timestamp(
                KeyframeType.LockHorizonPitch, timestamp_ms
            )
            if horizonpitch is None:
                horizonpitch = self.horizonpitch

            lock_pitch_val = keyframes.value_at_gyro_timestamp(
                KeyframeType.LockHorizonPitchEnabled, timestamp_ms
            )
            if lock_pitch_val is None:
                lock_pitch_val = 1.0 if self.lock_pitch else 0.0
            lock_pitch_flag = lock_pitch_val != 0.0

            horizonlockpercent = keyframes.value_at_gyro_timestamp(
                KeyframeType.LockHorizonAmount, timestamp_ms
            )
            if horizonlockpercent is None:
                horizonlockpercent = self.horizonlockpercent

            # Automatic lock: dynamic tilt
            dynamic_tilt_deg = 0.0
            if self.automatic_lock:
                target = 0.0
                if ts in roll_rates:
                    rr = roll_rates[ts]
                    if abs(rr) > self.turn_threshold:
                        target = rr * self.turn_multiplier

                if prev_tilt_ts is not None:
                    dt_tilt = (ts - prev_tilt_ts) / 1_000_000.0
                    alpha_tilt = (
                        max(0.0, min(1.0, dt_tilt / (tau_s + dt_tilt)))
                        if tau_s > 0.0
                        else 1.0
                    )
                else:
                    alpha_tilt = 1.0

                if prev_tilt_smoothed is not None:
                    smoothed_tilt = prev_tilt_smoothed * (1.0 - alpha_tilt) + target * alpha_tilt
                else:
                    smoothed_tilt = target

                # Accel limit
                accel_limited = smoothed_tilt
                if (
                    math.isfinite(self.tilt_accel_limit)
                    and prev_tilt_smoothed is not None
                    and prev_tilt_ts is not None
                ):
                    dt_acc = (ts - prev_tilt_ts) / 1_000_000.0
                    if dt_acc > 0.0:
                        max_change = self.tilt_accel_limit * dt_acc
                        change = smoothed_tilt - prev_tilt_smoothed
                        if abs(change) > max_change:
                            accel_limited = prev_tilt_smoothed + math.copysign(
                                max_change, change
                            )

                prev_tilt_smoothed = accel_limited
                prev_tilt_ts = ts
                dynamic_tilt_deg = accel_limited

            total_horizonroll_deg = horizonroll + dynamic_tilt_deg

            # Apply horizon lock
            locked_q = _lock_horizon_angle(
                smoothed_ori,
                total_horizonroll_deg * math.pi / 180.0,
                lock_pitch_flag,
                horizonpitch * math.pi / 180.0,
            )
            quats[ts] = locked_q.slerp(
                smoothed_ori, 1.0 - horizonlockpercent / 100.0
            )


def _rot_z_matrix(angle: float) -> np.ndarray:
    """Build a Z-axis rotation matrix."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
