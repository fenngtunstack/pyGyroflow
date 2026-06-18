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
        org_quats: TimeQuat | None = None,
        grav: Any | None = None,
        use_grav: bool = False,
    ) -> TimeQuat:
        """Execute smoothing with the current algorithm and optional horizon lock.

        Args:
            quats: Input quaternion sequence.
            duration_ms: Total duration in milliseconds.
            compute_params: Computation parameters.
            org_quats: Original quaternions for horizon lock roll rate.
            grav: Gravity vectors for horizon lock.
            use_grav: Whether to use gravity vectors in horizon lock.

        Returns:
            Smoothed (and optionally horizon-locked) quaternion sequence.
        """
        result = self.current().smooth(quats, duration_ms, compute_params)

        # Apply horizon lock if enabled
        if self.horizon_lock.lock_enabled or (
            hasattr(compute_params, "keyframes")
            and compute_params.keyframes.is_keyframed(
                __import__(
                    "pygyroflow.keyframes", fromlist=["KeyframeType"]
                ).KeyframeType.LockHorizonAmount
            )
        ):
            if org_quats is None:
                org_quats = quats
            self.horizon_lock.lock(result, org_quats, grav, use_grav, compute_params)

        return result

    def clone(self) -> Smoothing:
        """Create a copy of this Smoothing manager.

        Copies parameters from each algorithm to a new instance.
        """
        ret = Smoothing()
        ret.current_index = self.current_index
        ret.horizon_lock = HorizonLock()
        # Copy horizon lock params
        ret.horizon_lock.lock_enabled = self.horizon_lock.lock_enabled
        ret.horizon_lock.horizonlockpercent = self.horizon_lock.horizonlockpercent
        ret.horizon_lock.horizonroll = self.horizon_lock.horizonroll
        ret.horizon_lock.lock_pitch = self.horizon_lock.lock_pitch
        ret.horizon_lock.horizonpitch = self.horizon_lock.horizonpitch
        ret.horizon_lock.automatic_lock = self.horizon_lock.automatic_lock
        ret.horizon_lock.turn_threshold = self.horizon_lock.turn_threshold
        ret.horizon_lock.turn_smoothing_ms = self.horizon_lock.turn_smoothing_ms
        ret.horizon_lock.turn_multiplier = self.horizon_lock.turn_multiplier
        ret.horizon_lock.tilt_accel_limit = self.horizon_lock.tilt_accel_limit

        # Copy current algorithm parameters
        params = self.current().get_parameters_json()
        for p in params:
            name = p.get("name")
            value = p.get("value", 0.0)
            if name is not None:
                ret.current().set_parameter(name, value)

        return ret
