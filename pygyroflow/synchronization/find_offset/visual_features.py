"""Visual-features offset finder -- cross-correlate visual and gyro rotations.

Port of Gyroflow's ``find_offset::visual_features``.  The core idea:

1.  Angular velocities are estimated from optical flow (visual) and
    recorded from the gyroscope (gyro).
2.  Both signals are resampled onto a common time grid.
3.  Cross-correlation over a search window finds the time offset that
    aligns them best.

This is the simplest and fastest offset search method.  It works well
when the visual rotation estimates are reasonably accurate and the
motion is not purely rotational around one axis.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def find_offset_visual_features(
    visual_rotations: list[tuple[int, np.ndarray]],
    gyro_rotations: list[tuple[int, np.ndarray]],
    search_range_ms: float = 500.0,
    coarse_step_ms: float = 1.0,
    fine_step_ms: float = 0.01,
    fine_range_ms: float = 2.0,
) -> float | None:
    """Find the time offset that best aligns visual and gyro angular velocities.

    Parameters
    ----------
    visual_rotations:
        List of ``(timestamp_us, angular_velocity)`` pairs estimated from
        optical flow.  ``angular_velocity`` is shape (3,) in rad/s.
    gyro_rotations:
        List of ``(timestamp_us, angular_velocity)`` pairs from the IMU.
        ``angular_velocity`` is shape (3,) in deg/s (will be converted to
        rad/s internally).
    search_range_ms:
        Total search window in milliseconds (symmetric around 0).
    coarse_step_ms:
        Step size for the coarse search pass (milliseconds).
    fine_step_ms:
        Step size for the fine refinement pass.
    fine_range_ms:
        Half-width of the refinement window around the coarse optimum.

    Returns
    -------
    Offset in milliseconds such that ``visual_ts = gyro_ts + offset``,
    or ``None`` if there is insufficient data.
    """
    if len(visual_rotations) < 10 or len(gyro_rotations) < 10:
        logger.warning("Not enough data points for offset search")
        return None

    # Unpack into arrays
    vis_ts = np.array([t for t, _ in visual_rotations], dtype=np.float64)
    vis_av = np.array([v for _, v in visual_rotations], dtype=np.float64)

    gyr_ts = np.array([t for t, _ in gyro_rotations], dtype=np.float64)
    gyr_av = np.array([v for _, v in gyro_rotations], dtype=np.float64)

    # Convert timestamps from microseconds to milliseconds
    vis_ts_ms = vis_ts / 1000.0
    gyr_ts_ms = gyr_ts / 1000.0

    # Convert gyro from deg/s to rad/s for comparison
    if gyr_av.ndim == 2:
        gyr_av_rad = np.deg2rad(gyr_av)
    else:
        gyr_av_rad = np.deg2rad(gyr_av)

    # Compute per-sample rotation magnitude for 1-D cross-correlation
    if vis_av.ndim == 2 and vis_av.shape[1] == 3:
        vis_mag = np.linalg.norm(vis_av, axis=1)
    else:
        vis_mag = np.abs(vis_av.ravel())

    if gyr_av_rad.ndim == 2 and gyr_av_rad.shape[1] == 3:
        gyr_mag = np.linalg.norm(gyr_av_rad, axis=1)
    else:
        gyr_mag = np.abs(gyr_av_rad.ravel())

    # Build a common time grid covering both signals
    t_min = max(vis_ts_ms.min(), gyr_ts_ms.min())
    t_max = min(vis_ts_ms.max(), gyr_ts_ms.max())
    if t_max <= t_min:
        return None

    # Resample rate: use the higher of the two median sample intervals
    vis_interval = np.median(np.diff(vis_ts_ms)) if len(vis_ts_ms) > 1 else 1.0
    gyr_interval = np.median(np.diff(gyr_ts_ms)) if len(gyr_ts_ms) > 1 else 1.0
    resample_dt = min(vis_interval, gyr_interval, 1.0)  # at most 1 ms

    grid = np.arange(t_min, t_max, resample_dt)
    if len(grid) < 10:
        return None

    vis_interp = np.interp(grid, vis_ts_ms, vis_mag)
    gyr_interp = np.interp(grid, gyr_ts_ms, gyr_mag)

    # Normalize for correlation
    vis_centered = vis_interp - vis_interp.mean()
    gyr_centered = gyr_interp - gyr_interp.mean()
    gyr_energy = np.dot(gyr_centered, gyr_centered)
    if gyr_energy < 1e-12:
        return None

    # Coarse search: try offsets from -search_range/2 to +search_range/2
    half_range = search_range_ms / 2.0
    offsets_coarse = np.arange(-half_range, half_range + coarse_step_ms, coarse_step_ms)

    best_offset = 0.0
    best_corr = -np.inf

    for offset in offsets_coarse:
        # Shift visual signal by offset (positive offset = visual is ahead)
        shifted_vis_ts = vis_ts_ms - offset
        vis_shifted = np.interp(grid, shifted_vis_ts, vis_mag)
        vis_c = vis_shifted - vis_shifted.mean()

        corr = np.dot(vis_c, gyr_centered) / (
            np.sqrt(np.dot(vis_c, vis_c) * gyr_energy) + 1e-12
        )
        if corr > best_corr:
            best_corr = corr
            best_offset = offset

    # Fine search around the coarse optimum
    fine_offsets = np.arange(
        best_offset - fine_range_ms,
        best_offset + fine_range_ms + fine_step_ms,
        fine_step_ms,
    )

    for offset in fine_offsets:
        shifted_vis_ts = vis_ts_ms - offset
        vis_shifted = np.interp(grid, shifted_vis_ts, vis_mag)
        vis_c = vis_shifted - vis_shifted.mean()

        corr = np.dot(vis_c, gyr_centered) / (
            np.sqrt(np.dot(vis_c, vis_c) * gyr_energy) + 1e-12
        )
        if corr > best_corr:
            best_corr = corr
            best_offset = offset

    # Confidence guard: a weak peak means the visual rotation estimates
    # carry little sync signal and the "best" offset is noise (observed on
    # the GoPro pan clip: corr ~0.2 -> -242 ms, which destabilizes the
    # render). 0.5 keeps only unambiguous peaks; upstream has no such
    # guard, so this is a deliberate port deviation for safety.
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
