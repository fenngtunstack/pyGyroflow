"""Smoothing algorithm registry and manager.

Port of Gyroflow's core/smoothing/mod.rs Smoothing struct.

Manages the list of available smoothing algorithms and the currently selected
one. Also holds the HorizonLock post-processor.
"""

from __future__ import annotations

from typing import Any

from pygyroflow.smoothing.base import SmoothingAlgorithm
from pygyroflow.smoothing.default_algo import DefaultAlgo
from pygyroflow.smoothing.fixed import FixedSmoothing
from pygyroflow.smoothing.horizon import HorizonLock
from pygyroflow.smoothing.none import NoSmoothing
from pygyroflow.smoothing.plain import PlainSmoothing
from pygyroflow.types.time_types import TimeQuat


class Smoothing:
    """Top-level smoothing manager.

    Holds all algorithm instances, the current selection index,
    and the horizon lock post-processor.

    Algorithm index mapping:
        0: NoSmoothing
        1: DefaultAlgo (selected by default)
        2: PlainSmoothing
        3: FixedSmoothing
    """

    def __init__(self) -> None:
        self.algorithms: list[SmoothingAlgorithm] = [
            NoSmoothing(),
            DefaultAlgo(),
            PlainSmoothing(),
            FixedSmoothing(),
        ]
        self.current_index: int = 1  # DefaultAlgo by default
        self.horizon_lock: HorizonLock = HorizonLock()

    def set_current(self, idx: int) -> None:
        """Set the current algorithm index (clamped to valid range)."""
        self.current_index = min(idx, len(self.algorithms) - 1)

    def current(self) -> SmoothingAlgorithm:
        """Get the currently selected algorithm."""
        return self.algorithms[self.current_index]

    def get_names(self) -> list[str]:
        """Get display names of all algorithms."""
        return [alg.get_name() for alg in self.algorithms]

    def get_state_checksum(self, gyro_checksum: int) -> int:
        """Compute a combined checksum for cache invalidation.

        Includes gyro data checksum, current algorithm index,
        algorithm parameters, and horizon lock config.
        """
        return hash((
            gyro_checksum,
            self.current_index,
            self.current().get_checksum(),
            self.horizon_lock.get_checksum(),
        ))

    def smooth(
        self,
        quats: TimeQuat,
        duration_ms: float,
        compute_params: Any,
    ) -> TimeQuat:
        """Run the currently selected smoothing algorithm.

        The horizon lock is *not* applied here. Upstream composes the two in
        ``GyroSource::recompute_smoothness``, and the order matters: it locks
        the horizon on the original orientations and then smooths. Applying
        it here as well (as this did) locked a second time after smoothing —
        ``lock()`` slerps toward the locked orientation by a percentage, so
        two passes are not the same as one. See
        ``StabilizationManager.recompute_smoothing``.
        """
        return self.current().smooth(quats, duration_ms, compute_params)
