"""Abstract base class for all IMU integration algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pygyroflow.types.time_types import TimeIMU, TimeQuat


class GyroIntegrator(ABC):
    """Base class for IMU integration algorithms.

    All integrators convert raw IMU samples (gyro, accel, mag) into
    a time-indexed sequence of unit quaternion orientations.
    """

    @abstractmethod
    def integrate(self, imu_data: list[TimeIMU], duration_ms: float) -> TimeQuat:
        """Integrate IMU data into quaternion orientations.

        Args:
            imu_data: List of IMU samples with timestamp_ms, gyro, accl, magn.
            duration_ms: Total duration in milliseconds.

        Returns:
            TimeQuat mapping timestamp_us -> Quat64.
        """
        ...
