"""Offset-finding sub-package -- align visual and gyro timelines.

Upstream's offset methods (``synchronization/mod.rs:384-386``):

* ``0`` ``essential_matrix`` -- point pairs + essential matrices per
  offset candidate. **Not ported**; the signal-level shim below runs the
  correlation fallback for this index.
* ``1`` ``visual_features``  -- the faithful port
  (:func:`find_offset_visual_features`) minimizes the gyro-rotated point
  distances over candidate offsets. It needs matched point pairs and a
  ``ComputeParams`` with the integrated quaternion streams, so it is
  driven from ``AutosyncProcess.run``; callers holding only reduced
  rotation signals go through the correlation fallback.
* ``2`` ``rs_sync``          -- rolling-shutter-aware search. The
  per-point quaternion-error minimization lives in ``RollingShutterSync``
  and is driven by ``AutosyncProcess.run(quaternions=...)`` /
  ``StabilizationManager.synchronize()``; the function exported here
  works on pre-reduced rotation lists and falls back to magnitude
  correlation.

The public entry point here, ``find_time_offset()``, is a *signal-level*
shim (rotation lists in, offset out); the point-level searches live on
``AutosyncProcess``.
"""

from pygyroflow.synchronization.find_offset.visual_features import (
    find_offset_visual_features,
    find_offset_visual_features_correlation_fallback,
)
from pygyroflow.synchronization.find_offset.rs_sync import (
    find_offset_rs_sync,
)

import logging
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

# Method index -- mirrors Gyroflow's ``offset_method``
# 0 = essential_matrix (not ported; see module docstring)
# 1 = visual_features (point-pair search; signal callers get the fallback)
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
        0 or 1 = correlation fallback (upstream's method 1 is the point-pair
        search on ``AutosyncProcess`` — this signal-level shim cannot run
        it); 2 = rs_sync stub.
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
    return find_offset_visual_features_correlation_fallback(
        visual_rotations,
        gyro_rotations,
        search_range_ms=search_range_ms,
    )


__all__ = [
    "find_time_offset",
    "find_offset_visual_features",
    "find_offset_visual_features_correlation_fallback",
    "find_offset_rs_sync",
]
