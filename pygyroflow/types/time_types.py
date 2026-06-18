"""Time-indexed data types matching Gyroflow's Rust time types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from pygyroflow.types.quaternion import Quat64

# Type aliases matching Gyroflow's Rust types.
# Keys are timestamps in microseconds (i64), values are quaternions or 3D vectors.
TimeQuat = dict[int, Quat64]
TimeVec = dict[int, NDArray[np.float64]]


@dataclass
class TimeIMU:
    """Single IMU sample. Matches Gyroflow's TimeIMU struct.

    Attributes:
        timestamp_ms: Timestamp in milliseconds.
        gyro: Gyroscope data in degrees/second, shape (3,). None if not available.
        accl: Accelerometer data in m/s^2, shape (3,). None if not available.
        magn: Magnetometer data in microtesla, shape (3,). None if not available.
    """

    timestamp_ms: float
    gyro: NDArray[np.float64] | None = None
    accl: NDArray[np.float64] | None = None
    magn: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        if self.gyro is not None:
            self.gyro = np.asarray(self.gyro, dtype=np.float64)
            if self.gyro.shape != (3,):
                raise ValueError(f"gyro must have shape (3,), got {self.gyro.shape}")
        if self.accl is not None:
            self.accl = np.asarray(self.accl, dtype=np.float64)
            if self.accl.shape != (3,):
                raise ValueError(f"accl must have shape (3,), got {self.accl.shape}")
        if self.magn is not None:
            self.magn = np.asarray(self.magn, dtype=np.float64)
            if self.magn.shape != (3,):
                raise ValueError(f"magn must have shape (3,), got {self.magn.shape}")
