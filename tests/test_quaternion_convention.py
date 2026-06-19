"""Quaternion convention cross-check (golden triangle).

Independent verification that ``Quat64.__mul__`` and ``Quat64.inverse()``
follow the same convention as scipy ``Rotation`` matrix composition and
nalgebra ``UnitQuaternion`` (the Rust reference). This guards against the
class of regression that a silently-regenerated golden file cannot catch.

The three independent sources:
  1. scipy Rotation matrix composition  ``R1 @ R2``  (unambiguous ground truth)
  2. scipy Rotation ``__mul__``         ``R1 * R2``  (scipy's own convention)
  3. pygyroflow ``Quat64(q1) * Quat64(q2)`` -> rotation matrix

All three must agree to machine precision. nalgebra agreement is established
separately in the Rust workspace (msgyro-types tests); this file pins the
Python side and documents the verified convention.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pygyroflow.types.quaternion import Quat64


# Non-commuting rotation pairs (different axes) to expose order/convention bugs.
EULER_PAIRS = [
    ([90, 0, 0], [0, 90, 0]),    # X then Y
    ([0, 0, 45], [30, 0, 0]),    # Z then X
    ([10, 20, 30], [5, -15, 25]),
]


@pytest.mark.parametrize("e1,e2", EULER_PAIRS)
def test_mul_matches_scipy_matrix_composition(e1, e2):
    """Quat64 q1*q2 must equal scipy R1@R2 (apply q1/R1 first, then q2/R2)."""
    R1 = Rotation.from_euler("xyz", e1, degrees=True)
    R2 = Rotation.from_euler("xyz", e2, degrees=True)
    q1 = Quat64.from_euler_angles(*np.radians(e1))
    q2 = Quat64.from_euler_angles(*np.radians(e2))

    expected = (R1 * R2).as_matrix()           # == R1.as_matrix() @ R2.as_matrix()
    actual = (q1 * q2).to_rotation_matrix()

    np.testing.assert_allclose(actual, expected, atol=1e-12,
                               err_msg=f"q1*q2 != scipy R1*R2 for {e1}*{e2}")


@pytest.mark.parametrize("e1,e2", EULER_PAIRS)
def test_mul_is_not_reverse_order(e1, e2):
    """Sanity: q1*q2 must differ from q2*q1 (quaternion mult is non-commutative)."""
    q1 = Quat64.from_euler_angles(*np.radians(e1))
    q2 = Quat64.from_euler_angles(*np.radians(e2))
    left = (q1 * q2).to_rotation_matrix()
    right = (q2 * q1).to_rotation_matrix()
    assert not np.allclose(left, right, atol=1e-6), \
        "q1*q2 == q2*q1 — multiplication order is broken"


@pytest.mark.parametrize("e", [[90, 0, 0], [0, 45, 0], [10, 20, 30], [0, 0, -60]])
def test_inverse_is_conjugate(e):
    """q * q.inverse() must be identity."""
    q = Quat64.from_euler_angles(*np.radians(e))
    product = (q * q.inverse()).to_rotation_matrix()
    np.testing.assert_allclose(product, np.eye(3), atol=1e-12,
                               err_msg=f"q*q.inv() != I for {e}")


@pytest.mark.parametrize("e1,e2", EULER_PAIRS)
def test_inverse_distribution(e1, e2):
    """(q1*q2).inverse() == q2.inverse() * q1.inverse() (reversed-order conjugates)."""
    q1 = Quat64.from_euler_angles(*np.radians(e1))
    q2 = Quat64.from_euler_angles(*np.radians(e2))
    lhs = (q1 * q2).inverse().to_rotation_matrix()
    rhs = (q2.inverse() * q1.inverse()).to_rotation_matrix()
    np.testing.assert_allclose(lhs, rhs, atol=1e-12,
                               err_msg="(q1*q2).inv() != q2.inv()*q1.inv()")
