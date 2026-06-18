"""Unit quaternion wrapper matching nalgebra::UnitQuaternion<f64> behavior.

Internal storage uses scipy.spatial.transform.Rotation, which stores quaternions
in [x, y, z, w] format. All public APIs use nalgebra's [w, x, y, z] convention.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


class Quat64:
    """Unit quaternion with float64 precision.

    Wraps scipy.spatial.transform.Rotation to provide an interface compatible
    with nalgebra::UnitQuaternion<f64> used in Gyroflow's Rust codebase.
    """

    __slots__ = ("_rot",)

    def __init__(self, rotation: Rotation) -> None:
        self._rot = rotation

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def identity(cls) -> Quat64:
        """Return the identity quaternion (no rotation)."""
        return cls(Rotation.identity())

    @classmethod
    def from_euler_angles(cls, roll: float, pitch: float, yaw: float) -> Quat64:
        """Create from Euler angles in radians (ZYX intrinsic convention).

        This matches nalgebra's `UnitQuaternion::from_euler_angles(roll, pitch, yaw)`.
        scipy's 'ZYX' uppercase means intrinsic rotations (same as 'zyx' extrinsic).
        """
        rot = Rotation.from_euler("ZYX", [yaw, pitch, roll])
        return cls(rot)

    @classmethod
    def from_quaternion(cls, wxyz: NDArray[np.floating]) -> Quat64:
        """Create from a [w, x, y, z] quaternion array (nalgebra convention).

        scipy expects [x, y, z, w], so we reorder.
        """
        wxyz = np.asarray(wxyz, dtype=np.float64)
        if wxyz.shape != (4,):
            raise ValueError(f"Expected shape (4,), got {wxyz.shape}")
        # Normalize to unit quaternion
        norm = np.linalg.norm(wxyz)
        if norm < 1e-15:
            raise ValueError("Quaternion norm is zero")
        wxyz = wxyz / norm
        # [w, x, y, z] -> [x, y, z, w]
        xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
        return cls(Rotation.from_quat(xyzw))

    @classmethod
    def from_scaled_axis(cls, axis_angle: NDArray[np.floating]) -> Quat64:
        """Create from a rotation vector (scaled axis / axis-angle).

        axis_angle is a 3-element vector whose direction is the rotation axis
        and whose magnitude is the rotation angle in radians.
        """
        axis_angle = np.asarray(axis_angle, dtype=np.float64)
        if axis_angle.shape != (3,):
            raise ValueError(f"Expected shape (3,), got {axis_angle.shape}")
        return cls(Rotation.from_rotvec(axis_angle))

    @classmethod
    def from_rotation_matrix(cls, mat: NDArray[np.floating]) -> Quat64:
        """Create from a 3x3 rotation matrix."""
        mat = np.asarray(mat, dtype=np.float64)
        if mat.shape != (3, 3):
            raise ValueError(f"Expected shape (3, 3), got {mat.shape}")
        return cls(Rotation.from_matrix(mat))

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def inverse(self) -> Quat64:
        """Return the inverse (conjugate) quaternion."""
        return Quat64(self._rot.inv())

    def slerp(self, other: Quat64, t: float) -> Quat64:
        """Spherical linear interpolation to *other* at parameter *t* in [0, 1]."""
        # Use scipy's Slerp for correctness
        from scipy.spatial.transform import Slerp

        key_rots = Rotation.concatenate([self._rot, other._rot])
        key_times = [0, 1]
        slerp = Slerp(key_times, key_rots)
        return Quat64(slerp(t))

    def angle(self) -> float:
        """Return the rotation angle in radians.

        Matches nalgebra's UnitQuaternion::angle().
        For a unit quaternion q = cos(theta/2) + sin(theta/2)*v,
        the angle is 2 * acos(|w|).
        """
        wxyz = self.quaternion()
        w = float(wxyz[0])
        # Clamp to [-1, 1] to avoid nan from acos
        w = max(-1.0, min(1.0, abs(w)))
        return 2.0 * math.acos(w)

    def to_rotation_matrix(self) -> NDArray[np.float64]:
        """Return the 3x3 rotation matrix."""
        return self._rot.as_matrix().astype(np.float64)

    def quaternion(self) -> NDArray[np.float64]:
        """Return the quaternion as [w, x, y, z] (nalgebra convention)."""
        xyzw = self._rot.as_quat().astype(np.float64)
        # [x, y, z, w] -> [w, x, y, z]
        return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

    def euler_angles(self) -> tuple[float, float, float]:
        """Return (roll, pitch, yaw) in radians (ZYX intrinsic convention).

        Matches nalgebra's euler_angles() output order.
        """
        # scipy 'ZYX' intrinsic: returns [yaw, pitch, roll]
        ypr = self._rot.as_euler("ZYX")
        return (float(ypr[2]), float(ypr[1]), float(ypr[0]))

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __mul__(self, other: Quat64) -> Quat64:
        """Quaternion composition: self * other.

        Result represents applying *other* first, then *self* (same as nalgebra).
        """
        if not isinstance(other, Quat64):
            return NotImplemented
        # scipy: r1 * r2 means apply r1 then r2.
        # nalgebra: q1 * q2 means apply q2 then q1 (Hamilton product convention).
        # To match nalgebra: self * other = apply other first, then self.
        # In scipy terms: self._rot * other._rot already means "apply self then other",
        # but nalgebra q1*q2 means "apply q2 then q1", so we need other._rot * self._rot.
        # Wait, let's think again:
        #   - nalgebra: q1 * q2 rotates a vector v as q1 * (q2 * v * q2^-1) * q1^-1
        #     i.e., q2 applied first, then q1.
        #   - scipy: r1 * r2 means r2 applied first, then r1 (matrix composition r1 @ r2).
        # So nalgebra q1*q2 == scipy r1*r2. Direct multiplication is correct.
        return Quat64(self._rot * other._rot)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quat64):
            return NotImplemented
        # Compare quaternions, accounting for double cover (q and -q represent same rotation)
        q1 = self.quaternion()
        q2 = other.quaternion()
        return np.allclose(q1, q2, atol=1e-12) or np.allclose(q1, -q2, atol=1e-12)

    def __repr__(self) -> str:
        w, x, y, z = self.quaternion()
        return f"Quat64(w={w:.6f}, x={x:.6f}, y={y:.6f}, z={z:.6f})"

    def __hash__(self) -> int:
        # Use the rotation matrix for a canonical hash (immune to q/-q)
        mat = self.to_rotation_matrix()
        return hash(mat.tobytes())
