"""Trim utilities for quaternion sequences.

Port of Gyroflow's get_trimmed_quats and get_max_angles from
core/smoothing/mod.rs.

get_trimmed_quats: Replaces quaternions outside trim ranges with SLERP
interpolants so smoothing only operates within the ranges.

get_max_angles: Computes max rotation angles between original and smoothed
quaternions, decomposed by axis (pitch, yaw, roll).
"""

from __future__ import annotations

import math
from typing import Any

from pygyroflow.types.quaternion import Quat64
from pygyroflow.types.time_types import TimeQuat


def get_trimmed_quats(
    quats: TimeQuat,
    duration_ms: float,
    trim_range_only: bool,
    trim_ranges: list[tuple[float, float]],
) -> TimeQuat:
    """Get quaternions with SLERP at boundaries outside trim ranges.

    When trim_range_only is True and trim_ranges is non-empty, quaternions
    outside the trim ranges are replaced with SLERP interpolants between
    the nearest valid quaternions at the range boundaries.

    Args:
        quats: Input quaternion sequence (keys = microseconds).
        duration_ms: Total data duration in milliseconds.
        trim_range_only: Whether to apply trimming.
        trim_ranges: List of (start_frac, end_frac) pairs in [0.0, 1.0].

    Returns:
        Trimmed quaternion sequence (may be the original if no trimming needed).
    """
    if not trim_range_only or not trim_ranges:
        return dict(quats)

    quats_copy = dict(quats)

    # Convert trim ranges from fractions to microsecond timestamps
    ranges_us = [
        (round(start * duration_ms * 1000.0), round(end * duration_ms * 1000.0))
        for start, end in trim_ranges
    ]

    ts_sorted = sorted(quats_copy.keys())

    if not ts_sorted or not ranges_us:
        return quats_copy

    # Initialize prev_q and next_q at the first range boundary
    first_range_start = ranges_us[0][0]

    # prev_q: nearest quat at or after the first range start
    prev_q_ts: int | None = None
    prev_q_val: Quat64 | None = None
    for ts in ts_sorted:
        if ts >= first_range_start:
            prev_q_ts = ts
            prev_q_val = quats_copy[ts]
            break

    if prev_q_ts is None:
        # All quats are before the first range; nothing to do
        return quats_copy

    next_q_ts = prev_q_ts
    next_q_val = prev_q_val

    current_range_idx = 0
    current_range = ranges_us[0]

    for ts in ts_sorted:
        quat = quats_copy[ts]

        # Advance to next range if past current range end
        while ts > current_range[1]:
            if current_range_idx + 1 < len(ranges_us):
                current_range_idx += 1
                current_range = ranges_us[current_range_idx]
                prev_q_ts = ts
                prev_q_val = quat
                # Find next_q at or after the new range start
                next_q_ts = None
                next_q_val = None
                for ts2 in ts_sorted:
                    if ts2 >= current_range[0]:
                        next_q_ts = ts2
                        next_q_val = quats_copy[ts2]
                        break
                if next_q_ts is None:
                    next_q_ts = prev_q_ts
                    next_q_val = prev_q_val
            else:
                # Past all ranges
                last_range_end = ranges_us[-1][1]
                for ts2 in reversed(ts_sorted):
                    if ts2 <= last_range_end:
                        prev_q_ts = ts2
                        prev_q_val = quats_copy[ts2]
                        break
                next_q_ts = prev_q_ts
                next_q_val = prev_q_val
                current_range = (ts + 1, ts + 2)  # Impossible range
                break

        # If outside current range, SLERP interpolate
        if not (ts >= current_range[0] and ts <= current_range[1]):
            if prev_q_ts is not None and next_q_ts is not None:
                if next_q_ts == prev_q_ts:
                    quats_copy[ts] = prev_q_val
                else:
                    dist_to_next = (ts - prev_q_ts) / (next_q_ts - prev_q_ts)
                    if abs(dist_to_next) < 1e-12:
                        quats_copy[ts] = prev_q_val
                    else:
                        quats_copy[ts] = prev_q_val.slerp(next_q_val, dist_to_next)

    return quats_copy


def get_max_angles(
    quats: TimeQuat,
    smoothed_quats: TimeQuat,
    compute_params: Any,
) -> tuple[float, float, float]:
    """Compute maximum rotation angles between original and smoothed quaternions.

    For each timestamp in the trim ranges, computes the incremental rotation
    from the smoothed quaternion to the original, decomposes it into euler
    angles, and tracks the maximum absolute value per axis.

    Args:
        quats: Original (unsmoothed) quaternions.
        smoothed_quats: Smoothed quaternions.
        compute_params: Contains trim_ranges and scaled_duration_ms.

    Returns:
        (max_pitch_deg, max_yaw_deg, max_roll_deg)
    """
    duration_ms = getattr(compute_params, "scaled_duration_ms", 0.0)
    trim_ranges = getattr(compute_params, "trim_ranges", [])

    # Convert trim ranges to microsecond timestamps
    ranges_us = [
        (round(start * duration_ms * 1000.0), round(end * duration_ms * 1000.0))
        for start, end in trim_ranges
    ]

    identity_quat = Quat64.identity()
    max_pitch = 0.0
    max_yaw = 0.0
    max_roll = 0.0

    for ts, quat in smoothed_quats.items():
        within_range = not ranges_us or any(
            ts >= r[0] and ts <= r[1] for r in ranges_us
        )
        if within_range:
            orig = quats.get(ts, identity_quat)
            dist = quat.inverse() * orig
            euler = dist.euler_angles()  # (roll, pitch, yaw)
            if abs(euler[2]) > max_roll:
                max_roll = abs(euler[2])
            if abs(euler[0]) > max_pitch:
                max_pitch = abs(euler[0])
            if abs(euler[1]) > max_yaw:
                max_yaw = abs(euler[1])

    RAD2DEG = 180.0 / math.pi
    return (max_pitch * RAD2DEG, max_yaw * RAD2DEG, max_roll * RAD2DEG)
