"""The rs-sync optimizer port (B-04): end-to-end against ground truth.

The scene is built entirely from first principles: a sinusoidal pan from
scipy-free math, world rays fixed in space, camera rays as the gyro
quaternion at ``(track_time − D)`` applied to those rays. When the
optimizer is faithful, ``full_sync`` recovers the delay ``−D`` with cost
≈ 0 (the residual is the cross product of gyro-rotated ray pairs, which
vanishes at the true delay).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pygyroflow.synchronization.rs_sync_problem import (
    Backtrack,
    FrameState,
    NdSpline,
    Spline,
    SyncProblem,
    clamp_k,
    opt_compute_problem,
    opt_guess_translational_motion,
    quat_conj,
    quat_prod,
    quat_rotate_point,
    quat_slerp,
    safe_normalize,
)


def _q_pan(angle: float) -> np.ndarray:
    return np.array([
        math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0,
    ])


def _scene(delay_s: float = 0.05, n_pairs: int = 5, n_points: int = 25,
           seed: int = 1) -> SyncProblem:
    """Camera rays = pan(t − delay) applied to fixed world rays: the
    footage lags the gyro by *delay_s*, so the true sync delay is −delay."""
    fps = 200.0
    n = 800
    ts_us = [int(k / fps * 1e6) for k in range(n)]

    def theta(t: float) -> float:
        return 0.5 * math.sin(2.0 * math.pi * t / 1.5)

    def q(t: float) -> np.ndarray:
        return _q_pan(theta(t))

    sp = SyncProblem()
    sp.set_gyro_quaternions(ts_us, [tuple(q(k / fps)) for k in range(n)])

    rng = np.random.default_rng(seed)
    w = np.column_stack([
        rng.uniform(-0.8, 0.8, n_points),
        rng.uniform(-0.8, 0.8, n_points),
        np.ones(n_points),
    ])
    w /= np.linalg.norm(w, axis=1, keepdims=True)
    for fp in range(n_pairs):
        base_t = 1.0 + fp * 0.033
        ts_a = [base_t + rng.uniform(0, 0.004) for _ in range(n_points)]
        ts_b = [t + 1.0 / 30 for t in ts_a]
        ra = np.array([
            quat_rotate_point(q(t - delay_s), w[i]) for i, t in enumerate(ts_a)
        ])
        rb = np.array([
            quat_rotate_point(q(t - delay_s), w[i]) for i, t in enumerate(ts_b)
        ])
        sp.set_track_result(
            int(base_t * 1e6), ts_a, ts_b,
            [tuple(r) for r in ra], [tuple(r) for r in rb],
        )
    return sp


class TestQuatHelpers:
    def test_prod_is_associative_and_rotates(self):
        q = _q_pan(0.3)
        p = np.array([0.0, 0.2, 0.5, 1.0])
        double = quat_prod(quat_prod(q, p), quat_conj(q))
        assert double[0] == pytest.approx(0.0, abs=1e-12)

        v = np.array([0.1, -0.4, 0.9])
        r = quat_rotate_point(q, v)
        assert np.linalg.norm(r) == pytest.approx(np.linalg.norm(v))
        # Rotation about Y: the Y component is invariant and the x-z
        # radius is preserved (a full dot-product angle check only holds
        # for vectors perpendicular to the axis).
        assert r[1] == pytest.approx(v[1])
        assert math.hypot(r[0], r[2]) == pytest.approx(math.hypot(v[0], v[2]))
        back = quat_rotate_point(quat_conj(q), r)
        assert np.allclose(back, v, atol=1e-12)

    def test_slerp_endpoints_and_midpoint(self):
        a = _q_pan(0.0)
        b = _q_pan(1.0)
        assert np.allclose(quat_slerp(a, b, 0.0), a)
        assert np.allclose(quat_slerp(a, b, 1.0), b)
        mid = quat_slerp(a, b, 0.5)
        # (w, x, y, z): the pan lives in w and y. Halfway between angle 0
        # and 1 is angle 0.5 -> half-angle 0.25.
        assert mid[0] == pytest.approx(math.cos(0.25), abs=1e-12)
        assert mid[2] == pytest.approx(math.sin(0.25), abs=1e-12)

    def test_slerp_handles_opposite_hemisphere(self):
        a = _q_pan(0.0)
        b = -_q_pan(1.0)  # same rotation, negated quaternion
        mid = quat_slerp(a, b, 0.5)
        assert np.isfinite(mid).all()

    def test_safe_normalize_leaves_tiny_vectors(self):
        v = np.array([1e-13, 0.0, 0.0])
        assert safe_normalize(v) is v  # untouched below the threshold

    def test_clamp_k(self):
        assert clamp_k(1.0) == 1e1
        assert clamp_k(5e2) == 5e2
        assert clamp_k(1e6) == 1e3


class TestSpline:
    def test_reproduces_a_linear_function(self):
        xs = np.arange(20, dtype=np.float64)
        sp = Spline(3.0 * xs + 1.0)
        for x in (0.5, 5.25, 17.9):
            assert sp.eval(x) == pytest.approx(3.0 * x + 1.0, abs=1e-10)
            assert sp.deriv(x) == pytest.approx(3.0, abs=1e-8)

    def test_interpolates_at_the_knots(self):
        # Any interpolating spline reproduces the data at the knots.
        xs = np.arange(30, dtype=np.float64)
        data = 2.0 * xs ** 3 - xs + 4.0
        sp = Spline(data)
        for i in (0, 7, 29):
            assert sp.eval(float(i)) == pytest.approx(data[i], abs=1e-9)

    def test_natural_boundaries_flatten_the_ends(self):
        # The natural-ish boundary (c[0] = c[n-1] = 0) means x^2 is NOT
        # reproduced away from the knots: the end segments sag below the
        # true parabola (measured: eval(9.5) < 90.25, eval(11) < 121).
        # Pinned so a "fix" to the spline changes this test visibly.
        xs = np.arange(10, dtype=np.float64)
        sp = Spline(xs ** 2)
        assert sp.eval(9.5) < 9.5 ** 2
        assert sp.eval(11.0) < 11.0 ** 2

    def test_extrapolates_with_the_last_segment(self):
        xs = np.arange(10, dtype=np.float64)
        sp = Spline(xs ** 2)
        # beyond the last knot the last segment's polynomial continues:
        # above the last knot's value (81), if below the true 121.
        assert sp.eval(11.0) > 81.0

    def test_ndspline_evals_each_component(self):
        quats = np.array([
            [1.0, 0.9, 0.8, 0.7],
            [0.0, 0.1, 0.2, 0.3],
            [0.0, 0.2, 0.4, 0.6],
            [0.0, 0.3, 0.6, 0.9],
        ])
        nd = NdSpline.make(quats)
        out = nd.eval(1.0)
        assert out[0] == pytest.approx(0.9)
        assert out[2] == pytest.approx(0.2)
        assert not nd.is_empty()

    def test_set_gyro_quaternions_resamples_to_50hz(self):
        # 190 Hz input over 2 s -> 1.0 s span/380 samples = 190 Hz -> rounds
        # to 200 Hz; the resampled grid is uniform at 200 Hz.
        ts = [int(k / 190.0 * 1e6) for k in range(380)]
        quats = [tuple(_q_pan(0.01 * k)) for k in range(380)]
        sp = SyncProblem()
        sp.set_gyro_quaternions(ts, quats)
        assert sp.problem.sample_rate == pytest.approx(200.0)
        assert sp.problem.quats_start == pytest.approx(ts[0] / 1e6)


class TestTheGradientChain:
    def test_motion_gradient_matches_finite_differences(self):
        sp = _scene()
        fs = FrameState(1_000_000, sp.problem)
        fs.motion_vec = np.array([0.01, 0.9, -0.03])
        fs.var_k = fs.guess_k(0.0)

        _, _, jac = fs.loss(0.0, fs.motion_vec)
        eps = 1e-7
        for ax in range(3):
            m1 = fs.motion_vec.copy()
            m1[ax] += eps
            m2 = fs.motion_vec.copy()
            m2[ax] -= eps
            fd = (fs.loss_single(0.0, m1) - fs.loss_single(0.0, m2)) / (2 * eps)
            assert jac[ax] == pytest.approx(fd, abs=1e-5), ax

    def test_delay_gradient_matches_finite_differences(self):
        sp = _scene()
        fs = FrameState(1_000_000, sp.problem)
        fs.motion_vec = np.array([0.2, 0.7, 0.1])
        fs.var_k = 1e2
        _, jac_delay, _ = fs.loss(0.0, fs.motion_vec)
        eps = 1e-7
        fd = (
            fs.loss_single(eps, fs.motion_vec)
            - fs.loss_single(-eps, fs.motion_vec)
        ) / (2 * eps)
        assert jac_delay == pytest.approx(fd, abs=1e-4)

    def test_loss_single_is_zero_when_rotation_explains_everything(self):
        sp = _scene(delay_s=0.0)
        fs = FrameState(1_000_000, sp.problem)
        # With the true delay 0, the problem rows are ~0: any motion gives 0.
        fs.motion_vec = np.array([1.0, 0.0, 0.0])
        fs.var_k = 1e3
        # ~1e-7 floor: the spline's interpolation error in the quats.
        assert fs.loss_single(0.0, fs.motion_vec) == pytest.approx(0.0, abs=1e-5)


class TestBacktrack:
    def test_descends_a_quadratic(self):
        bt = Backtrack()
        bt.set_hyper(2e-4, 0.1, 1e-3, 10)
        # f(x) = (x-3)^2: grad 2(x-3). At x=0 the step moves toward 3.
        bt.set_objective(lambda x: ((x - 3.0) ** 2, 2.0 * (x - 3.0)))
        step = bt.step(0.0)
        assert step > 0.0
        # upstream's step returns -t*p: negative gradient direction
        assert bt._f_only(0.0 + step) < 4.0 * 9.0  # improved from f(0)=9

    def test_set_objective_installs_f_only(self):
        bt = Backtrack()
        bt.set_objective(lambda x: (x * x, 2.0 * x))
        assert bt._f_only is not None
        assert bt._f_only(2.0) == pytest.approx(4.0)


class TestTheResidual:
    def test_residual_vanishes_at_the_true_delay(self):
        sp = _scene(delay_s=0.05)
        total = sum(
            float(np.linalg.norm(opt_compute_problem(ts, -0.05, sp.problem)))
            for ts in sp.problem.frame_data
        )
        assert total == pytest.approx(0.0, abs=1e-3)

    def test_residual_is_nonzero_away_from_it(self):
        sp = _scene(delay_s=0.05)
        total = sum(
            float(np.linalg.norm(opt_compute_problem(ts, 0.0, sp.problem)))
            for ts in sp.problem.frame_data
        )
        assert total > 0.1

    def test_lmeds_guess_finds_the_degenerate_direction(self):
        # rows all orthogonal to (0,1,0): the motion direction that
        # minimizes |n·v| squared for the quartile is ±(0,1,0).
        rows = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.6, 0.0, 0.8],
            [-0.3, 0.0, 0.95],
        ])
        v = opt_guess_translational_motion(rows, 50)
        assert abs(v[1]) == pytest.approx(1.0, abs=1e-9)

    def test_empty_problem_returns_zero_vector(self):
        out = opt_guess_translational_motion(np.zeros((0, 3)), 10)
        assert out.shape == (0,)


class TestFullSync:
    def test_recovers_the_true_delay(self):
        sp = _scene(delay_s=0.05)
        result = sp.full_sync(0.0, 0, 2_000_000, 0.003, 0.15, 4)
        assert result is not None
        cost, delay = result
        assert delay == pytest.approx(-0.05, abs=0.003)
        assert cost == pytest.approx(0.0, abs=1e-3)

    def test_recovers_a_delay_near_zero(self):
        sp = _scene(delay_s=0.008, n_pairs=4)
        result = sp.full_sync(0.0, 0, 2_000_000, 0.003, 0.05, 4)
        assert result is not None
        assert result[1] == pytest.approx(-0.008, abs=0.003)

    def test_progress_callback_runs(self):
        sp = _scene(delay_s=0.02, n_pairs=3)
        seen = []
        sp.on_progress(lambda p: (seen.append(p), True)[1])
        assert sp.full_sync(0.0, 0, 2_000_000, 0.003, 0.05, 2) is not None
        assert seen  # fired at least once
        assert seen[-1] == pytest.approx(1.0)

    def test_cancel_stops(self):
        sp = _scene(delay_s=0.02, n_pairs=3)
        sp.on_progress(lambda p: False)  # cancel immediately
        assert sp.full_sync(0.0, 0, 2_000_000, 0.003, 0.05, 2) is None

    def test_empty_quats_is_none(self):
        sp = SyncProblem()
        assert sp.full_sync(0.0, 0, 1000, 0.003, 0.05, 2) is None
