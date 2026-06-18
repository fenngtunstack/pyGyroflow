"""Offset-finding sub-package -- align visual and gyro timelines.

Three search strategies:

* ``visual_features`` -- Cross-correlate rotation magnitudes (fast, simple)
* ``rs_sync``         -- Rolling-shutter-aware search (stub, falls back)

The public entry point is ``find_time_offset()``, which dispatches by
method index.
"""

from pygyroflow.synchronization.find_offset.visual_features import (
    find_offset_visual_features,
)
from pygyroflow.synchronization.find_offset.rs_sync import (
    find_offset_rs_sync,
)

import logging
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

# Method index -- mirrors Gyroflow's ``offset_method``
# 0 = essential_matrix (same engine as visual_features here)
# 1 = visual_features
# 2 = rs_sync


def find_time_offset(
    visual_rotations: list[tuple[int, np.ndarray]],
    gyro_rotations: list[tuple[int, np.ndarray]],
    method: int = 1,
    search_range_ms: float = 500.0,
    initial_offset_ms: float = 0.0,
    progress_callback: Callable[[float], None] | None = None,
) -> float | None:
    """Find the time offset between visual and gyro signals.

    Parameters
    ----------
    visual_rotations:
        ``[(timestamp_us, angular_velocity_3), ...]`` from optical flow.
    gyro_rotations:
        ``[(timestamp_us, angular_velocity_3), ...]`` from the IMU.
    method:
        0 or 1 = visual_features cross-correlation; 2 = rs_sync stub.
    search_range_ms:
        Total search window width (milliseconds).
    initial_offset_ms:
        Approximate offset hint (used by rs_sync stub).
    progress_callback:
        Optional progress reporter ``Callable[[0..1], None]``.

    Returns
    -------
    Offset in milliseconds or None.
    """
    if method == 2:
        return find_offset_rs_sync(
            visual_rotations,
            gyro_rotations,
            search_range_ms=search_range_ms,
            initial_offset_ms=initial_offset_ms,
            progress_callback=progress_callback,
        )

    # Methods 0 and 1 both use cross-correlation
    return find_offset_visual_features(
        visual_rotations,
        gyro_rotations,
        search_range_ms=search_range_ms,
    )


__all__ = [
    "find_time_offset",
    "find_offset_visual_features",
    "find_offset_rs_sync",
]
