"""Tests for Quat64 quaternion class."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pygyroflow.types.quaternion import Quat64


class TestQuat64Identity:
    def test_identity_is_unit_quaternion(self):
        q = Quat64.identity()
        wxyz = q.quaternion()
        assert_allclose(np.linalg.norm(wxyz), 1.0, atol=1e-12)

    def test_identity_components(self):
        q = Quat64.identity()
        wxyz = q.quaternion()
        assert_allclose(wxyz, [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_identity_rotation_matrix(self):
        q = Quat64.identity()
        mat = q.to_rotation_matrix()
        assert_allclose(mat, np.eye(3), atol=1e-12)

    def test_identity_angle_is_zero(self):
        q = Quat64.identity()
        assert_allclose(q.angle(), 0.0, atol=1e-12)


class TestQuat64FromEulerAngles:
    def test_zero_angles_produce_identity(self):
        q = Quat64.from_euler_angles(0.0, 0.0, 0.0)
        expected = Quat64.identity()
        assert q == expected

    def test_90_degree_roll(self):
        q = Quat64.from_euler_angles(math.pi / 2, 0.0, 0.0)
        mat = q.to_rotation_matrix()
        # 90 degree rotation around X: Y->Z, Z->-Y
        assert_allclose(mat @ np.array([0, 1, 0]), np.array([0, 0, 1]), atol=1e-10)
        assert_allclose(mat @ np.array([0, 0, 1]), np.array([0, -1, 0]), atol=1e-10)

    def test_90_degree_pitch(self):
        q = Quat64.from_euler_angles(0.0, math.pi / 2, 0.0)
        mat = q.to_rotation_matrix()
        # 90 degree rotation around Y: X->-Z, Z->X
        assert_allclose(mat @ np.array([1, 0, 0]), np.array([0, 0, -1]), atol=1e-10)
        assert_allclose(mat @ np.array([0, 0, 1]), np.array([1, 0, 0]), atol=1e-10)

    def test_90_degree_yaw(self):
        q = Quat64.from_euler_angles(0.0, 0.0, math.pi / 2)
        mat = q.to_rotation_matrix()
        # 90 degree rotation around Z: X->Y, Y->-X
        assert_allclose(mat @ np.array([1, 0, 0]), np.array([0, 1, 0]), atol=1e-10)
        assert_allclose(mat @ np.array([0, 1, 0]), np.array([-1, 0, 0]), atol=1e-10)

    def test_euler_roundtrip(self):
        """euler_angles -> from_euler_angles should recover the same rotation."""
        roll, pitch, yaw = 0.3, -0.5, 1.2
        q = Quat64.from_euler_angles(roll, pitch, yaw)
        r, p, y = q.euler_angles()
        assert_allclose([r, p, y], [roll, pitch, yaw], atol=1e-10)


class TestQuat64FromQuaternion:
    def test_roundtrip(self):
        wxyz = np.array([0.5, 0.5, 0.5, 0.5])
        q = Quat64.from_quaternion(wxyz)
        result = q.quaternion()
        # May differ by sign (double cover)
        if np.dot(result, wxyz) < 0:
            result = -result
        assert_allclose(result, wxyz / np.linalg.norm(wxyz), atol=1e-12)

    def test_normalization(self):
        wxyz = np.array([2.0, 0.0, 0.0, 0.0])
        q = Quat64.from_quaternion(wxyz)
        result = q.quaternion()
        assert_allclose(np.linalg.norm(result), 1.0, atol=1e-12)

    def test_rejects_zero_norm(self):
        with pytest.raises(ValueError, match="norm is zero"):
            Quat64.from_quaternion(np.array([0.0, 0.0, 0.0, 0.0]))

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="shape"):
            Quat64.from_quaternion(np.array([1.0, 0.0, 0.0]))


class TestQuat64Inverse:
    def test_self_times_inverse_is_identity(self, identity_quat):
        q = Quat64.from_euler_angles(0.3, -0.5, 1.2)
        product = q * q.inverse()
        assert product == identity_quat

    def test_identity_inverse_is_identity(self):
        q = Quat64.identity()
        assert q.inverse() == q


class TestQuat64Slerp:
    def test_slerp_at_zero_returns_start(self):
        q1 = Quat64.from_euler_angles(0.0, 0.0, 0.0)
        q2 = Quat64.from_euler_angles(0.0, 0.0, 1.0)
        result = q1.slerp(q2, 0.0)
        assert result == q1

    def test_slerp_at_one_returns_end(self):
        q1 = Quat64.from_euler_angles(0.0, 0.0, 0.0)
        q2 = Quat64.from_euler_angles(0.0, 0.0, 1.0)
        result = q1.slerp(q2, 1.0)
        assert result == q2

    def test_slerp_same_quaternion(self):
        q = Quat64.from_euler_angles(0.3, -0.5, 1.2)
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = q.slerp(q, t)
            assert result == q

    def test_slerp_midpoint_is_half_angle(self):
        q1 = Quat64.identity()
        angle = math.pi / 4
        q2 = Quat64.from_euler_angles(0.0, 0.0, angle)
        mid = q1.slerp(q2, 0.5)
        # Midpoint should have half the angle
        expected = Quat64.from_euler_angles(0.0, 0.0, angle / 2)
        assert mid == expected


class TestQuat64Composition:
    def test_composition_produces_combined_rotation(self):
        q1 = Quat64.from_euler_angles(0.0, 0.0, math.pi / 2)  # 90 yaw
        q2 = Quat64.from_euler_angles(0.0, math.pi / 2, 0.0)  # 90 pitch
        combined = q1 * q2
        mat = combined.to_rotation_matrix()
        # Both rotations applied should transform X axis
        x_axis = np.array([1.0, 0.0, 0.0])
        result = mat @ x_axis
        # Not identity — some vector has been rotated
        assert not np.allclose(result, x_axis, atol=1e-6)

    def test_composition_with_identity(self):
        q = Quat64.from_euler_angles(0.3, -0.5, 1.2)
        identity = Quat64.identity()
        assert q * identity == q
        assert identity * q == q


class TestQuat64FromScaledAxis:
    def test_zero_axis_is_identity(self):
        q = Quat64.from_scaled_axis(np.array([0.0, 0.0, 0.0]))
        assert q == Quat64.identity()

    def test_small_angle_approx(self):
        """Small rotation: q ~ identity + perturbation."""
        angle = 1e-6
        axis = np.array([0.0, 0.0, angle])
        q = Quat64.from_scaled_axis(axis)
        wxyz = q.quaternion()
        # w should be ~1, z should be ~angle/2
        assert_allclose(wxyz[0], 1.0, atol=1e-5)
        assert_allclose(abs(wxyz[3]), angle / 2, atol=1e-5)

    def test_90_deg_around_z(self):
        q = Quat64.from_scaled_axis(np.array([0.0, 0.0, math.pi / 2]))
        mat = q.to_rotation_matrix()
        x_axis = np.array([1.0, 0.0, 0.0])
        rotated = mat @ x_axis
        assert_allclose(rotated, np.array([0.0, 1.0, 0.0]), atol=1e-10)


class TestQuat64FromRotationMatrix:
    def test_identity_matrix(self):
        q = Quat64.from_rotation_matrix(np.eye(3))
        assert q == Quat64.identity()

    def test_roundtrip(self):
        q_orig = Quat64.from_euler_angles(0.3, -0.5, 1.2)
        mat = q_orig.to_rotation_matrix()
        q_recovered = Quat64.from_rotation_matrix(mat)
        assert q_recovered == q_orig


class TestQuat64Equality:
    def test_double_cover(self):
        """q and -q represent the same rotation."""
        wxyz = np.array([0.5, 0.5, 0.5, 0.5])
        q1 = Quat64.from_quaternion(wxyz)
        q2 = Quat64.from_quaternion(-wxyz)
        assert q1 == q2

    def test_different_rotations_not_equal(self):
        q1 = Quat64.from_euler_angles(0.0, 0.0, 0.0)
        q2 = Quat64.from_euler_angles(0.0, 0.0, 1.0)
        assert q1 != q2
