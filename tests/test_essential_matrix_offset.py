"""Offset method 0: angular-velocity matching (``essential_matrix.rs``).

The visual signal is the pose estimator's angular velocity; the search
sweeps candidate offsets against the raw IMU with weighted squared error
(pitch/roll 70, yaw 100), a ceil sample lookup, and a live 90 % window
guard. Conventions are pinned against an independently constructed scene:
the visual signal is the true gyro delayed by ``delta``, which must yield
``offset = -delta`` (``visual = gyro + offset``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pygyroflow.synchronization.autosync import AutosyncProcess
from pygyroflow.synchronization.find_offset import essential_matrix as em
from pygyroflow.types.time_types import TimeIMU

FPS = 250.0  # dense IMU so the ceil lookup is fine-grained


def _omega(t: float) -> float:
    return 25.0 * math.sin(2.0 * math.pi * t / 1.5)


def _gyro_samples(n=750, fps=FPS, omega=_omega) -> list[TimeIMU]:
    return [
        TimeIMU(timestamp_ms=k / fps * 1000.0,
                gyro=np.array([0.0, omega(k / fps), 0.0]))
        for k in range(n)
    ]


def _visual_samples(delta_s: float, n=300, fps=30.0, omega=_omega,
                    noise: float = 0.0, seed=3) -> list[TimeIMU]:
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        t = k / fps + delta_s
        v = omega(t) + (rng.uniform(-noise, noise) if noise else 0.0)
        out.append(TimeIMU(timestamp_ms=k / fps * 1000.0,
                           gyro=np.array([0.0, v, 0.0])))
    return out


class TestTheSearch:
    def test_footage_late_by_40ms_yields_minus_40(self):
        # Both signals span ~3 s: lookups past the IMU's end would leave
        # fewer than half the samples matched and cost inf everywhere.
        of = _visual_samples(delta_s=0.040, noise=0.5, n=90)
        gyro = _gyro_samples()
        result = em.find_offset_essential_matrix(
            of, gyro, search_size_ms=150.0, scaled_fps=30.0,
        )
        assert result is not None
        assert result[0] == pytest.approx(-40.0, abs=2.0)

    def test_quiet_signal_is_skipped(self):
        """Under 3 deg/s of movement there is no sync signal (upstream
        skips the range with ``No movement detected``)."""
        of = [
            TimeIMU(timestamp_ms=k / 30.0 * 1000.0,
                    gyro=np.array([0.0, 1.0, 0.0]))
            for k in range(100)
        ]
        assert em.find_offset_essential_matrix(
            of, _gyro_samples(), search_size_ms=100.0
        ) is None

    def test_window_edge_minimum_is_rejected(self):
        """Method 0's guard is live: the coarse grid reaches the full
        ±search_size, so a minimum at the edge gets rejected by the 90 %
        rule — unlike method 1, whose grid stops at half the window."""
        of = _visual_samples(delta_s=-0.140)  # truth (+140) beyond ±100
        gyro = _gyro_samples()
        assert em.find_offset_essential_matrix(
            of, gyro, search_size_ms=100.0
        ) is None

    def test_no_imu_in_window_is_no_result(self):
        of = _visual_samples(delta_s=0.0)
        assert em.find_offset_essential_matrix(
            of, [], search_size_ms=100.0
        ) is None

    def test_progress_is_reported(self):
        of = _visual_samples(delta_s=0.0, n=60)
        seen = []
        em.find_offset_essential_matrix(
            of, _gyro_samples(), search_size_ms=40.0,
            progress_callback=seen.append,
        )
        assert seen == sorted(seen)
        assert seen[-1] == pytest.approx(1.0)


class TestTheCostFunction:
    def test_yaw_is_weighted_heavier_than_pitch(self):
        """The 70/70/100 weights: identical error on z costs more than on
        x, so a scene differing only in z can't tie with one differing in
        x."""
        of = [TimeIMU(timestamp_ms=0.0, gyro=np.array([1.0, 0.0, 0.0]))]
        keys = [0]
        gyro_map = {0: TimeIMU(timestamp_ms=0.0, gyro=np.array([0.0, 0.0, 0.0]))}
        # of's x = 1 vs gyro x = 0 -> error on the 70-weighted axis.
        cost_x = em._calculate_cost(0.0, of, keys, gyro_map)
        of_z = [TimeIMU(timestamp_ms=0.0, gyro=np.array([0.0, 0.0, 1.0]))]
        cost_z = em._calculate_cost(0.0, of_z, keys, gyro_map)
        assert cost_z == pytest.approx(cost_x * 100.0 / 70.0)

    def test_under_half_matches_is_infinite(self):
        """A candidate that reaches fewer than half the samples has no
        cost (upstream returns f64::MAX so the candidate can't win)."""
        of = [TimeIMU(timestamp_ms=t, gyro=np.array([1.0, 0.0, 0.0]))
              for t in (0.0, 10.0, 20.0, 30.0)]
        # IMU only covers the first sample -> 1 match of 4, not > half.
        keys = [0]
        gyro_map = {0: TimeIMU(timestamp_ms=0.0, gyro=np.array([1.0, 0.0, 0.0]))}
        assert em._calculate_cost(0.0, of, keys, gyro_map) == float("inf")

    def test_ceil_lookup_not_nearest(self):
        """The lookup takes the first sample *at or after* the query: at
        t=9 ms with samples at 0 and 10, the value at 10 is used — the
        0-sample one is never considered, even if it were closer."""
        of = [TimeIMU(timestamp_ms=9.0, gyro=np.array([5.0, 0.0, 0.0]))]
        keys = [0, 10_000]
        gyro_map = {
            0: TimeIMU(timestamp_ms=0.0, gyro=np.array([0.0, 0.0, 0.0])),
            10_000: TimeIMU(timestamp_ms=10.0, gyro=np.array([5.0, 0.0, 0.0])),
        }
        # ceil(9 ms) = the 10 ms sample -> error 0 -> cost 0. Nearest would
        # also pick 10 here; the discriminator is that the 0 sample's value
        # is never blended in.
        assert em._calculate_cost(0.0, of, keys, gyro_map) == pytest.approx(0.0)


class TestTheAutosyncIntegration:
    def test_method_zero_branch_reaches_the_search(self, monkeypatch):
        proc = AutosyncProcess(camera_matrix=np.eye(3), fps=30.0)
        seen = {}

        def fake(of_samples, gyro_samples, **kwargs):
            seen["n_of"] = len(of_samples)
            seen["search_size_ms"] = kwargs["search_size_ms"]
            return (-37.0, 1.0)

        monkeypatch.setattr(
            em, "find_offset_essential_matrix", fake
        )
        gyro = [(k * 33_333, np.array([0.0, 10.0, 0.0])) for k in range(40)]
        visual = [(k * 33_333, np.array([0.0, 10.0, 0.0])) for k in range(30)]
        offset = proc._essential_matrix_offset(
            visual, gyro, search_range_ms=200.0, initial_offset_ms=0.0
        )
        assert offset == pytest.approx(-37.0)
        assert seen["n_of"] == 30
        assert seen["search_size_ms"] == pytest.approx(100.0)  # half window

    def test_run_dispatches_method_zero(self, monkeypatch):
        proc = AutosyncProcess(camera_matrix=np.eye(3), fps=30.0, offset_method=0)
        monkeypatch.setattr(
            proc, "_essential_matrix_offset",
            lambda *a, **k: 12.0,
        )
        gyro = [(k * 33_333, np.array([0.0, 10.0, 0.0])) for k in range(40)]
        frames = [(k * 33_333, np.zeros((16, 16), dtype=np.uint8))
                  for k in range(40)]
        # run() needs the pose stage; stub it out — the offset path under
        # test is what comes after.
        monkeypatch.setattr(
            proc._pose_estimator, "get_visual_rotations",
            lambda **k: [(k2 * 33_333, np.array([0.0, 10.0, 0.0]))
                         for k2 in range(30)],
        )
        offset = proc.run(frames, gyro, search_range_ms=200.0)
        assert offset == pytest.approx(12.0)
