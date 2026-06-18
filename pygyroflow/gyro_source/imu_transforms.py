"""IMU transform pipeline — orientation remapping, rotation, bias, filtering.

Port of Gyroflow's src/core/gyro_source/imu_transforms.rs.
Applies user-configurable transforms to raw IMU samples before integration:
  1. Gyro bias subtraction
  2. Orientation string remapping (e.g. "XYZ" -> "xYz")
  3. Additional rotation matrix (separate for gyro and accel)
  4. Low-pass filter (Butterworth, forward-backward)
  5. Median filter
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

_DEG2RAD = math.pi / 180.0


@dataclass
class IMUTransforms:
    """Configuration for IMU data transforms.

    Mirrors Gyroflow's IMUTransforms struct. Rotation matrices are derived
    from the angle fields on demand rather than stored separately (Python
    doesn't need the pre-computed Rotation3 optimization).
    """

    imu_orientation: str | None = "XYZ"
    imu_rotation_angles: tuple[float, float, float] | None = None  # (pitch, roll, yaw) degrees
    acc_rotation_angles: tuple[float, float, float] | None = None
    imu_lpf: float = 0.0   # Low-pass filter cutoff Hz, 0 = disabled
    imu_mf: int = 0         # Median filter window size, 0 = disabled
    gyro_bias: list[float] | None = None  # [bx, by, bz]

    # Derived rotation matrices (populated by set_imu_rotation / set_acc_rotation)
    _imu_rotation_matrix: np.ndarray | None = field(default=None, repr=False, compare=False)
    _acc_rotation_matrix: np.ndarray | None = field(default=None, repr=False, compare=False)

    def transform(self, v: np.ndarray, is_acc: bool) -> np.ndarray:
        """Apply all transforms to a single 3-element sample.

        Args:
            v: Shape (3,) array — gyro or accel sample, modified in-place.
            is_acc: True for accelerometer data (uses acc_rotation), False for gyro.
        """
        if self.gyro_bias is not None:
            v += np.array(self.gyro_bias)

        if self.imu_orientation is not None and self.imu_orientation != "XYZ":
            v[:] = _orient(v, self.imu_orientation)

        if is_acc and self._acc_rotation_matrix is not None:
            v[:] = self._acc_rotation_matrix @ v
        elif self._imu_rotation_matrix is not None:
            v[:] = self._imu_rotation_matrix @ v

        return v

    def has_any(self) -> bool:
        """Check if any transform is active."""
        if self.imu_orientation is not None and self.imu_orientation != "XYZ":
            return True
        if self._imu_rotation_matrix is not None:
            return True
        if self._acc_rotation_matrix is not None:
            return True
        if self.gyro_bias is not None and any(abs(b) > 0.0 for b in self.gyro_bias):
            return True
        if self.imu_lpf > 0.0:
            return True
        if self.imu_mf > 0:
            return True
        return False

    def set_imu_rotation(self, pitch_deg: float, roll_deg: float, yaw_deg: float) -> None:
        """Set additional IMU rotation from Euler angles in degrees."""
        if abs(pitch_deg) > 0.0 or abs(roll_deg) > 0.0 or abs(yaw_deg) > 0.0:
            self.imu_rotation_angles = (pitch_deg, roll_deg, yaw_deg)
            self._imu_rotation_matrix = _euler_to_rotation_matrix(yaw_deg, pitch_deg, roll_deg)
        else:
            self.imu_rotation_angles = None
            self._imu_rotation_matrix = None

    def set_acc_rotation(self, pitch_deg: float, roll_deg: float, yaw_deg: float) -> None:
        """Set additional accelerometer rotation from Euler angles in degrees."""
        if abs(pitch_deg) > 0.0 or abs(roll_deg) > 0.0 or abs(yaw_deg) > 0.0:
            self.acc_rotation_angles = (pitch_deg, roll_deg, yaw_deg)
            self._acc_rotation_matrix = _euler_to_rotation_matrix(yaw_deg, pitch_deg, roll_deg)
        else:
            self.acc_rotation_angles = None
            self._acc_rotation_matrix = None


def _orient(inp: np.ndarray, orientation: str) -> np.ndarray:
    """Remap axes based on orientation string (e.g. "XYZ", "xYz").

    Uppercase = positive axis, lowercase = negated axis.
    """
    def map_axis(o: str) -> float:
        if o == "X":
            return inp[0]
        if o == "x":
            return -inp[0]
        if o == "Y":
            return inp[1]
        if o == "y":
            return -inp[1]
        if o == "Z":
            return inp[2]
        if o == "z":
            return -inp[2]
        raise ValueError(f"Invalid orientation character: {o!r}")

    return np.array([map_axis(orientation[0]), map_axis(orientation[1]), map_axis(orientation[2])])


def _euler_to_rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Build a 3x3 rotation matrix from Euler angles in degrees (ZYX intrinsic).

    Matches nalgebra's Rotation3::from_euler_angles(yaw, pitch, roll).
    """
    from scipy.spatial.transform import Rotation

    return Rotation.from_euler("ZYX", [yaw_deg * _DEG2RAD, pitch_deg * _DEG2RAD, roll_deg * _DEG2RAD]).as_matrix()
