"""The converter's correction formula (A-08) — an equivalence proof.

The gap table claimed ``n * io * org⁻¹`` (the port) differs from
``n * (org * io⁻¹)⁻¹`` (upstream, imu_integration/mod.rs:46). It does not:
by the group-inverse law ``(org * io⁻¹)⁻¹ = io * org⁻¹`` for rotations —
the two expressions are the same quaternion, verified numerically at
6.6e-15 degrees. The row's premise came from comparing different
conventions, not different algebra. This test pins the equivalence so the
"simplification" comment in the port stays honest.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pygyroflow.types.quaternion import Quat64


def _quat(euler):
    return Quat64(Rotation.from_euler("xyz", euler))


class TestTheFormulasAreIdentical:
    @pytest.mark.parametrize("seed", range(8))
    def test_group_inverse_law_makes_them_equal(self, seed):
        rng = np.random.default_rng(seed)
        n = _quat(rng.normal(size=3) * 0.3)
        org = _quat(rng.normal(size=3) * 0.4)
        io = _quat(rng.normal(size=3) * 0.5)

        upstream = n * (org * io.inverse()).inverse()
        simplified = n * io * org.inverse()

        angle = (upstream.inverse() * simplified)._rot.magnitude()
        assert np.degrees(angle) == pytest.approx(0.0, abs=1e-9)

    def test_the_two_io_terms_do_not_commute_trivially(self):
        """Guard against the check passing vacuously: io and org with
        distinct axes must not make (org * io⁻¹) its own inverse."""
        org = _quat((0.2, -0.4, 0.1))
        io = _quat((-0.3, 0.1, 0.5))
        x = org * io.inverse()
        angle = np.degrees((x * x)._rot.magnitude())
        assert angle > 1.0  # X² ≠ I, so the identity is doing real work
