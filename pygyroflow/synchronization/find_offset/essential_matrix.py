"""Offset search via angular-velocity matching (upstream offset method 0).

Port of Gyroflow's ``find_offset::essential_matrix::find_offsets``. Despite
the file name, the shipped algorithm has no essential matrix in it: it
compares the optical-flow-derived angular velocity (the pose estimator's
output) against the *raw* IMU samples directly —

* both signals go through a 20 Hz forward-backward lowpass;
* for each candidate offset ``offs`` the real gyro sample at
  ``(sample_time − offs)`` is looked up and the cost
  ``Σ 70·(gx−ox)² + 70·(gy−oy)² + 100·(gz−oz)²`` accumulated — the yaw
  (z) axis is weighted 100 vs 70 for pitch/roll;
* the lookup takes the first sample at or after the query (a *ceil*, not an
  interpolation — ``essential_matrix.rs:138-140``);
* a range whose visual signal never exceeds 3 °/s carries no sync signal
  and is skipped; a candidate must match more than half of the samples to
  have a cost at all;
* the search sweeps ±``search_size`` in 1 ms steps, refines ±2 ms at
  0.01 ms, and rejects results within the outer 10 % of the window — here
  the guard is *live* (the coarse grid reaches the full ±search_size,
  unlike the method-1 sweep which only reaches ±0.5·search_size).
"""

from __future__ import annotations

import bisect
import logging
from collections.abc import Callable

import numpy as np

from pygyroflow.types.time_types import TimeIMU

logger = logging.getLogger(__name__)

# Cost weights (essential_matrix.rs:158-160): pitch/roll 70, yaw 100.
_COST_WEIGHTS = (70.0, 70.0, 100.0)
_MIN_MOVEMENT = 3.0  # deg/s — below this a range has no sync signal
_FINE_SEARCH_MS = 2.0
_FINE_STEPS = 200


def _gyro_at_timestamp(
    keys: list[int], gyro_map: dict[int, TimeIMU], timestamp_ms: float
) -> TimeIMU | None:
    """First sample at or after *timestamp_ms* (a ceil, not an interp).

    Upstream's ``gyro_at_timestamp`` is ``map.range(..=t).next()`` — no
    interpolation between IMU samples.
    """
    key = int(round(timestamp_ms * 1000.0))
    idx = bisect.bisect_left(keys, key)
    if idx >= len(keys):
        return None
    return gyro_map[keys[idx]]


def _calculate_cost(
    offs: float,
    of: list[TimeIMU],
    keys: list[int],
    gyro_map: dict[int, TimeIMU],
) -> float:
    """Mean weighted squared difference at *offs* (``essential_matrix.rs:144-166``)."""
    total = 0.0
    matches = 0
    w0, w1, w2 = _COST_WEIGHTS
    for o in of:
        g = _gyro_at_timestamp(keys, gyro_map, o.timestamp_ms - offs)
        if g is None or g.gyro is None or o.gyro is None:
            continue
        matches += 1
        total += (float(g.gyro[0]) - float(o.gyro[0])) ** 2 * w0
        total += (float(g.gyro[1]) - float(o.gyro[1])) ** 2 * w1
        total += (float(g.gyro[2]) - float(o.gyro[2])) ** 2 * w2
    # A candidate that matches less than half the samples is not a match.
    if of and matches > len(of) / 2:
        return total / matches
    return float("inf")


def _lowpass(samples: list[TimeIMU], sample_rate: float) -> list[TimeIMU]:
    """20 Hz forward-backward lowpass over the gyro channels in place."""
    from pygyroflow.filtering.lowpass import lowpass_filter_channels

    values = np.array([s.gyro for s in samples], dtype=np.float64).T  # (3, N)
    filtered = lowpass_filter_channels(
        values, 20.0, sample_rate, forward_backward=True
    )
    return [
        TimeIMU(timestamp_ms=s.timestamp_ms, gyro=filtered[:, i].copy())
        for i, s in enumerate(samples)
    ]


def find_offset_essential_matrix(
    of_samples: list[TimeIMU],
    gyro_samples: list[TimeIMU],
    search_size_ms: float,
    initial_offset_ms: float = 0.0,
    scaled_fps: float = 30.0,
    progress_callback: Callable[[float], None] | None = None,
) -> tuple[float, float] | None:
    """Find the offset aligning the visual angular velocity with the IMU.

    Port of ``essential_matrix.rs:13-114`` for a single sync range (the
    port searches one offset per clip, not a per-range track).

    Parameters
    ----------
    of_samples:
        The pose estimator's angular velocity as IMU-shaped samples
        (deg/s), i.e. ``PoseEstimator.get_visual_rotations()`` recast.
    gyro_samples:
        The raw IMU samples (deg/s).
    search_size_ms:
        The search *radius* around ``initial_offset_ms`` (upstream's
        ``SyncParams.search_size``). The port's ``search_range_ms`` is the
        full window, so callers pass half of it.
    initial_offset_ms:
        Centre of the window; ``visual = gyro + offset`` convention.

    Returns
    -------
    ``(offset_ms, cost)`` or ``None`` when there is no usable signal (no
    samples, under 3 °/s of movement, or the minimum sits within the outer
    10 % of the window).
    """
    if not of_samples or not gyro_samples:
        logger.warning("Essential-matrix offset search: no samples")
        return None

    ss = search_size_ms
    first_of_ts = of_samples[0].timestamp_ms
    last_of_ts = of_samples[-1].timestamp_ms

    gyro_item = [
        g for g in gyro_samples
        if first_of_ts - ss <= g.timestamp_ms + initial_offset_ms <= last_of_ts + ss
    ]
    if not gyro_item:
        logger.warning("Essential-matrix offset search: no IMU samples in window")
        return None

    max_angle = max(
        max(abs(float(c)) for c in o.gyro) if o.gyro is not None else 0.0
        for o in of_samples
    )
    if max_angle < _MIN_MOVEMENT:
        logger.info(
            "No movement detected, max gyro angle: %s. Skipping sync point.",
            max_angle,
        )
        return None

    span_ms = gyro_item[-1].timestamp_ms - gyro_item[0].timestamp_ms
    sample_rate = len(gyro_item) / (span_ms / 1000.0) if span_ms > 0 else scaled_fps
    of_item = _lowpass(of_samples, scaled_fps)
    gyro_item = _lowpass(gyro_item, sample_rate)

    gyro_map = {int(round(g.timestamp_ms * 1000.0)): g for g in gyro_item}
    keys = sorted(gyro_map)

    def cost_at(offs: float) -> float:
        return _calculate_cost(offs, of_item, keys, gyro_map)

    # Coarse: 1 ms steps over ±search_size (upstream's 0..search_size*2).
    steps = int(ss * 2)
    best_offs = initial_offset_ms - ss
    best_cost = float("inf")
    for i in range(steps):
        offs = initial_offset_ms - ss + float(i)
        cost = cost_at(offs)
        if cost < best_cost:
            best_cost, best_offs = cost, offs
        if progress_callback is not None:
            progress_callback(0.5 * (i + 1) / max(steps, 1))

    # Fine: ±2 ms around the coarse winner at 0.01 ms.
    for i in range(_FINE_STEPS):
        offs = best_offs + (-_FINE_SEARCH_MS + i * (_FINE_SEARCH_MS / _FINE_STEPS))
        cost = cost_at(offs)
        if cost < best_cost:
            best_cost, best_offs = cost, offs
        if progress_callback is not None:
            progress_callback(0.5 + 0.5 * (i + 1) / _FINE_STEPS)

    # Only accept offsets within 90 % of the radius — and unlike method 1,
    # the coarse grid reaches the full radius, so this guard is live.
    if abs(best_offs - initial_offset_ms) >= ss * 0.9:
        logger.warning(
            "Sync point out of acceptable range %s < %s",
            abs(best_offs - initial_offset_ms), ss * 0.9,
        )
        return None

    return best_offs, best_cost
