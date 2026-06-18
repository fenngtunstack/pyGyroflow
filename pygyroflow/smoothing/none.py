"""No-op smoothing algorithm -- returns input unchanged.

Port of Gyroflow's core/smoothing/none.rs.
"""

from __future__ import annotations

from typing import Any

from pygyroflow.smoothing.base import SmoothingAlgorithm
from pygyroflow.types.time_types import TimeQuat


class NoSmoothing(SmoothingAlgorithm):
    """Passthrough smoothing: returns the input quaternions unchanged.

    Useful as a diagnostic baseline to see raw gyro stabilization.
    """

    def get_name(self) -> str:
        return "No smoothing"

    def get_parameters_json(self) -> list[dict]:
        return []

    def set_parameter(self, name: str, val: float) -> None:
        pass  # No parameters

    def get_parameter(self, name: str) -> float:
        return 0.0

    def get_checksum(self) -> int:
        return 0

    def smooth(
        self,
        quats: TimeQuat,
        duration_ms: float,
        compute_params: Any,
    ) -> TimeQuat:
        return dict(quats)
