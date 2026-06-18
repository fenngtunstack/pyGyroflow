"""Abstract base class for all smoothing algorithms.

Port of Gyroflow's core/smoothing/mod.rs SmoothingAlgorithm trait.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pygyroflow.types.time_types import TimeQuat

if TYPE_CHECKING:
    pass


class SmoothingAlgorithm(ABC):
    """Abstract base for all smoothing algorithms.

    Each algorithm takes a TimeQuat (timestamp_us -> Quat64) and returns
    a smoothed TimeQuat. The compute_params argument carries keyframes,
    FOV data, trim ranges, etc.
    """

    @abstractmethod
    def get_name(self) -> str:
        """Return the display name of this algorithm."""

    @abstractmethod
    def get_parameters_json(self) -> list[dict]:
        """Return parameter descriptions for UI rendering.

        Each dict contains name, description, type, from, to, value, default, etc.
        """

    @abstractmethod
    def set_parameter(self, name: str, val: float) -> None:
        """Set a parameter value by name."""

    @abstractmethod
    def get_parameter(self, name: str) -> float:
        """Get a parameter value by name."""

    @abstractmethod
    def get_checksum(self) -> int:
        """Return a checksum of the current parameter configuration.

        Used to detect parameter changes and invalidate cached results.
        """

    @abstractmethod
    def smooth(
        self,
        quats: TimeQuat,
        duration_ms: float,
        compute_params: Any,
    ) -> TimeQuat:
        """Execute the smoothing algorithm.

        Args:
            quats: Input quaternion sequence (keys = microseconds).
            duration_ms: Total data duration in milliseconds.
            compute_params: Computation parameters (keyframes, FOV, etc.).

        Returns:
            Smoothed quaternion sequence with the same timestamps.
        """
