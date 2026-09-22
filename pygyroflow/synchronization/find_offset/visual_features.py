"""Offset search over matched feature points (upstream offset method 1).

Port of Gyroflow's ``find_offset::visual_features::find_offsets``. The idea:

for each candidate time offset ``offs``, map both point sets of every frame
pair through the lens **and the gyro orientation at (frame_time − offs)**
(``undistort_points_with_rolling_shutter``) and sum the squared distances of
the mapped pairs. When ``offs`` puts the gyro in sync with the footage, the
stabilizing rotation absorbs the observed motion and the mapped pairs land on
top of each other; at a wrong offset they keep a residual of the optical flow.
So the distance landscape has its minimum at the sync point. The search is a
1 ms coarse sweep over the window, then a 0.01 ms refinement over ±1 ms
around the coarse winner, with the result rejected if it lands within the
outer 10 % of the window (an edge minimum is the boundary, not a match).

This replaces the 1-D angular-velocity cross-correlation that used to live
here. Upstream has no signal-level offset search — method 1 *is* this
point-pair search — but the old function is kept (renamed
``..._correlation_fallback``) for callers without a ``ComputeParams``, who
cannot run the lens/gyro machinery.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _total_distance(matched_pairs, params, offs: float, w: float, h: float) -> float:
    """Sum of squared distances of the gyro-rotated point pairs at ``offs``.

    The ``calculate_distance`` closure of ``visual_features.rs:49-83``. One
    quirk is deliberate and pinned by tests: the squared distance is cast to
    ``u64`` (truncation), and the longest 10 % of each pair's distances are
    dropped ("often wrongly computed point matches") with
    ``(len * 0.9) as usize`` truncation — a single surviving point
    contributes nothing.
    """
    from pygyroflow.stabilization.cpu_undistort import (
        undistort_points_with_rolling_shutter,
    )

    total_dist = 0.0
    for (ts_us, pts1), (next_ts_us, pts2) in matched_pairs:
        ts_ms = ts_us / 1000.0
        ts2_ms = next_ts_us / 1000.0
        undistorted1 = undistort_points_with_rolling_shutter(
            pts1, ts_ms - offs, None, params, 1.0, False
        )
        undistorted2 = undistort_points_with_rolling_shutter(
            pts2, ts2_ms - offs, None, params, 1.0, False
        )
        distances: list[int] = []
        for (x1, y1), (x2, y2) in zip(undistorted1, undistorted2):
            # Strictly inside the frame; the points are in the flow's working
            # size while w/h are the calibration size — upstream has the same
            # mismatch and it only trims edge candidates.
            if 0.0 < x1 < w and 0.0 < y1 < h and 0.0 < x2 < w and 0.0 < y2 < h:
                # `dist as u64` truncates; so does int().
                distances.append(int((x2 - x1) ** 2 + (y2 - y1) ** 2))
        distances.sort()
        keep = int(len(distances) * 0.9)
        total_dist += float(sum(distances[:keep]))
    return total_dist


def find_offset_visual_features(
    matched_pairs,
    params,
    search_size_ms: float = 500.0,
    initial_offset_ms: float = 0.0,
    progress_callback: Callable[[float], None] | None = None,
) -> tuple[float, float] | None:
    """Find the time offset that best explains the optical flow with the gyro.

    Port of ``visual_features.rs::find_offsets`` (the ``for_rs = false``
    branch; the ``for_rs`` rolling-shutter estimator lives in
    :func:`estimate_rolling_shutter`).

    Parameters
    ----------
    matched_pairs:
        ``[((ts_us, pts1), (next_ts_us, pts2)), ...]`` — one entry per frame
        pair, points in the optical-flow working size. Upstream collects
        these per *sync range* from the estimator's stored feature lines;
        the port feeds one range per clip (its ``AutosyncProcess`` searches
        a single offset, not a per-range offset track).
    params:
        ``ComputeParams`` with the lens **and the integrated quaternion
        streams** (``quaternions`` / ``smoothed_quaternions``) — the mapped
        points are rotated by the gyro at ``(frame_time − offs)``. Upstream
        clears the gyro offsets on its private clone; this function does the
        same to its own (``sync_offsets_adjusted = {}``), so the sweep is
        not biased by an offset it is itself trying to find.
    search_size_ms:
        Total width of the coarse sweep, in ms, centred on
        ``initial_offset_ms``.
    initial_offset_ms:
        Centre of the search window (ms), ``visual = gyro + offset``
        convention.

    Returns
    -------
    ``(offset_ms, cost)`` of the refined minimum, or ``None`` when there is
    no data or the minimum sits within the outer 10 % of the window (the
    boundary of the search space is not a match — upstream rejects those
    with ``Sync point out of acceptable range``).
    """
    if not len(matched_pairs):
        logger.warning("Visual-feature offset search: no point pairs")
        return None

    import dataclasses

    # Upstream clones params and clears the gyro offsets on the clone
    # (`params.gyro.write().clear_offsets()`), so a previously synced offset
    # does not bias the sweep. The offsets reach the points path through
    # `sync_offsets_adjusted`; blanking it on a shallow copy is the same.
    params = dataclasses.replace(params, sync_offsets_adjusted={})

    w, h = float(params.width), float(params.height)

    # Coarse: 1 ms steps over the window.
    best_offs = 0.0
    best_cost = float("inf")
    steps = int(search_size_ms)
    for i in range(steps):
        offs = initial_offset_ms + (-search_size_ms / 2.0 + float(i))
        cost = _total_distance(matched_pairs, params, offs, w, h)
        if cost < best_cost:
            best_cost, best_offs = cost, offs
        if progress_callback is not None:
            progress_callback(0.5 * (i + 1) / max(steps, 1))

    # Fine: 0.01 ms steps over ±1 ms around the coarse winner.
    for i in range(200):
        offs = best_offs - 1.0 + float(i) * 0.01
        cost = _total_distance(matched_pairs, params, offs, w, h)
        if cost < best_cost:
            best_cost, best_offs = cost, offs
        if progress_callback is not None:
            progress_callback(0.5 + 0.5 * (i + 1) / 200.0)

    # Only accept offsets inside 90 % of the window: at the edge the minimum
    # is the boundary, not a match.
    if abs(best_offs - initial_offset_ms) >= search_size_ms * 0.9:
        logger.warning(
            "Visual-feature offset search: best offset %.2f ms is within the "
            "outer 10 %% of the window; rejecting", best_offs,
        )
        return None

    logger.debug(
        "Visual-feature offset search: best_offset=%.3f ms, cost=%.1f",
        best_offs, best_cost,
    )
    return best_offs, best_cost


def find_offset_visual_features_correlation_fallback(
    visual_rotations: Sequence[tuple[int, np.ndarray]],
    gyro_rotations: Sequence[tuple[int, np.ndarray]],
    search_range_ms: float = 500.0,
    coarse_step_ms: float = 1.0,
    fine_step_ms: float = 0.01,
    fine_range_ms: float = 2.0,
) -> float | None:
    """1-D cross-correlation of the angular-velocity magnitudes.

    **Not an upstream algorithm** — Gyroflow's offset method 1 is the
    point-pair search above. This is the port's own fallback for callers
    that have no ``ComputeParams`` (no lens profile), who cannot run the
    lens/gyro machinery. Its 0.5 peak-correlation guard is a deliberate
    port deviation: a weak peak means the visual estimates carry no usable
    sync signal and the "best" offset is noise.
    """
    if len(visual_rotations) < 10 or len(gyro_rotations) < 10:
        logger.warning("Not enough data points for offset search")
        return None

    vis_ts = np.array([t for t, _ in visual_rotations], dtype=np.float64)
    vis_av = np.array([v for _, v in visual_rotations], dtype=np.float64)

    gyr_ts = np.array([t for t, _ in gyro_rotations], dtype=np.float64)
    gyr_av = np.array([v for _, v in gyro_rotations], dtype=np.float64)

    vis_ts_ms = vis_ts / 1000.0
    gyr_ts_ms = gyr_ts / 1000.0

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

    gyr_interp = np.interp(grid, gyr_ts_ms, gyr_mag)
    gyr_centered = gyr_interp - gyr_interp.mean()
    gyr_energy = np.dot(gyr_centered, gyr_centered)
    if gyr_energy < 1e-12:
        return None

    half_range = search_range_ms / 2.0
    offsets_coarse = np.arange(-half_range, half_range + coarse_step_ms, coarse_step_ms)

    def correlation_at(offset: float) -> tuple[float, float]:
        shifted_vis_ts = vis_ts_ms - offset
        vis_shifted = np.interp(grid, shifted_vis_ts, vis_mag)
        vis_c = vis_shifted - vis_shifted.mean()
        corr = np.dot(vis_c, gyr_centered) / (
            np.sqrt(np.dot(vis_c, vis_c) * gyr_energy) + 1e-12
        )
        return corr, offset

    best_offset = 0.0
    best_corr = -np.inf

    for offset in offsets_coarse:
        corr, off = correlation_at(float(offset))
        if corr > best_corr:
            best_corr, best_offset = corr, off

    fine_offsets = np.arange(
        best_offset - fine_range_ms,
        best_offset + fine_range_ms + fine_step_ms,
        fine_step_ms,
    )
    for offset in fine_offsets:
        corr, off = correlation_at(float(offset))
        if corr > best_corr:
            best_corr, best_offset = corr, off

    if best_corr < 0.5:
        logger.warning(
            "Visual-feature offset search: peak correlation %.4f is too "
            "weak; returning no offset", best_corr,
        )
        return None

    logger.debug(
        "Visual-feature offset search: best_offset=%.3f ms, corr=%.4f",
        best_offset, best_corr,
    )
    return float(best_offset)


def estimate_rolling_shutter(
    matched_pairs,
    params,
    fps: float,
) -> tuple[float, float] | None:
    """Estimate the sensor's rolling-shutter readout time (``for_rs``).

    Port of ``visual_features.rs:87-110`` (the branch upstream's
    ``estimate_rolling_shutter`` autosync mode drives): instead of shifting
    the gyro timeline, every candidate is a *readout time* -- the per-point
    rolling-shutter model inside ``undistort_points_with_rolling_shutter``
    reads ``params.frame_readout_time``, so each candidate swaps that field
    on a private copy and measures the same point-pair distance at zero
    offset. The sweep runs +/-(1000/fps) in 1 ms steps, then +/-1 ms around
    the winner at 0.01 ms.

    Returns ``(readout_ms, cost)`` or ``None`` with no data.
    """
    if not len(matched_pairs):
        logger.warning("Rolling-shutter estimate: no point pairs")
        return None

    import dataclasses

    w, h = float(params.width), float(params.height)

    def distance_at(rs_ms: float) -> float:
        candidate = dataclasses.replace(params, frame_readout_time=rs_ms)
        return _total_distance(matched_pairs, candidate, 0.0, w, h)

    max_rs = 1000.0 / fps if fps > 0 else 33.0
    steps = int(max_rs)
    best_rs = 0.0
    best_cost = float("inf")
    for i in range(-steps, steps):
        cost = distance_at(float(i))
        if cost < best_cost:
            best_cost, best_rs = cost, float(i)

    for i in range(200):
        rs = best_rs - 1.0 + i * 0.01
        cost = distance_at(rs)
        if cost < best_cost:
            best_cost, best_rs = cost, rs

    logger.debug("rolling shutter estimate: %.2f ms (cost %.1f)", best_rs, best_cost)
    return best_rs, best_cost
