"""Tests for smoothing algorithms."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pygyroflow.types.quaternion import Quat64
from pygyroflow.types.time_types import TimeQuat
from pygyroflow.keyframes import KeyframeManager


def _make_noisy_quats(n=200, dt_us=10000):
    """Generate quaternions with high-frequency noise on top of a slow rotation."""
    quats = {}
    rng = np.random.RandomState(42)
    for i in range(n):
        slow = i * 0.005  # Slow drift
        noise = 0.1 * rng.randn()  # High-freq noise
        angle = slow + noise
        quats[i * dt_us] = Quat64.from_euler_angles(0.0, 0.0, angle)
    return quats


def _make_compute_params(quats, duration_ms):
    """Create a minimal compute_params-like object for smoothing tests."""
    from pygyroflow.stabilization import ComputeParams
    params = ComputeParams(
        scaled_duration_ms=duration_ms,
        scaled_fps=len(quats) / (duration_ms / 1000.0),
    )
    return params


def _angular_velocity(quats: TimeQuat) -> list[float]:
    """Compute per-frame angular velocity in radians."""
    ts_sorted = sorted(quats.keys())
    velocities = [0.0]
    for i in range(1, len(ts_sorted)):
        q_prev = quats[ts_sorted[i - 1]]
        q_curr = quats[ts_sorted[i]]
        delta = q_prev.inverse() * q_curr
        dt = (ts_sorted[i] - ts_sorted[i - 1]) / 1e6  # us -> s
        if dt > 0:
            velocities.append(delta.angle() / dt)
        else:
            velocities.append(0.0)
    return velocities


class TestNoSmoothing:
    def test_passthrough(self):
        from pygyroflow.smoothing import NoSmoothing
        quats = _make_noisy_quats()
        duration_ms = 2000.0
        params = _make_compute_params(quats, duration_ms)

        algo = NoSmoothing()
        result = algo.smooth(quats, duration_ms, params)

        assert set(result.keys()) == set(quats.keys())
        for ts in quats:
            assert result[ts] == quats[ts]

    def test_name(self):
        from pygyroflow.smoothing import NoSmoothing
        assert NoSmoothing().get_name() == "No smoothing"


class TestPlainSmoothing:
    def test_preserves_timestamps(self):
        from pygyroflow.smoothing import PlainSmoothing
        quats = _make_noisy_quats()
        duration_ms = 2000.0
        params = _make_compute_params(quats, duration_ms)

        algo = PlainSmoothing()
        result = algo.smooth(quats, duration_ms, params)
        assert set(result.keys()) == set(quats.keys())

    def test_output_is_smoother(self):
        """Smoothed output has lower angular velocity variance than input."""
        from pygyroflow.smoothing import PlainSmoothing
        quats = _make_noisy_quats()
        duration_ms = 2000.0
        params = _make_compute_params(quats, duration_ms)

        algo = PlainSmoothing()
        algo.set_parameter("time_constant", 0.5)
        result = algo.smooth(quats, duration_ms, params)

        vel_input = _angular_velocity(quats)
        vel_output = _angular_velocity(result)

        # Variance of output velocity should be less than input
        assert np.var(vel_output) < np.var(vel_input)


class TestDefaultAlgo:
    def test_preserves_timestamps(self):
        from pygyroflow.smoothing import DefaultAlgo
        quats = _make_noisy_quats()
        duration_ms = 2000.0
        params = _make_compute_params(quats, duration_ms)

        algo = DefaultAlgo()
        result = algo.smooth(quats, duration_ms, params)
        assert set(result.keys()) == set(quats.keys())

    def test_output_is_smoother(self):
        from pygyroflow.smoothing import DefaultAlgo
        quats = _make_noisy_quats()
        duration_ms = 2000.0
        params = _make_compute_params(quats, duration_ms)

        algo = DefaultAlgo()
        algo.set_parameter("smoothness", 0.8)
        result = algo.smooth(quats, duration_ms, params)

        vel_input = _angular_velocity(quats)
        vel_output = _angular_velocity(result)

        assert np.var(vel_output) < np.var(vel_input)

    def test_name(self):
        from pygyroflow.smoothing import DefaultAlgo
        assert DefaultAlgo().get_name() == "Default"


class TestFixedSmoothing:
    def test_produces_constant_orientation(self):
        from pygyroflow.smoothing import FixedSmoothing
        quats = _make_noisy_quats()
        duration_ms = 2000.0
        params = _make_compute_params(quats, duration_ms)

        algo = FixedSmoothing()
        result = algo.smooth(quats, duration_ms, params)

        # All timestamps present, all quaternions identical
        assert set(result.keys()) == set(quats.keys())
        ts_sorted = sorted(result.keys())
        first = result[ts_sorted[0]]
        for ts in ts_sorted[1:]:
            assert result[ts] == first
