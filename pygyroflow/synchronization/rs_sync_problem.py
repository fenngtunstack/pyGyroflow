"""Port of the ``rs-sync`` crate (github.com/gyroflow/rs-sync @ fc5cf51).

This is the optimization engine behind Gyroflow's rolling-shutter-aware
sync: for every tracked frame pair it builds a small least-squares problem
whose unknowns are a 3-D **translational motion** estimate, solved
repeatedly while a scalar **gyro delay** is optimized around it — with a
robust ``log1p`` loss so outlier tracks don't drag the solution.

Structure mirrors the crate file by file:

* :class:`Spline` / :class:`NdSpline` — ``minispline.rs`` / ``ndspline.rs``:
  cubic spline over the gyro quaternion components.
* :mod:`quat helpers` — ``quat.rs``: (w, x, y, z) convention.
* :class:`Backtrack` — ``backtrack.rs``: 1-D backtracking line search on
  the delay (sufficient-decrease criterion).
* :class:`SyncProblem` — ``lib.rs``: the problem state, the per-frame
  :class:`FrameState` (motion + scale ``var_k``), the robust loss with its
  analytic gradient chain, ``pre_sync`` (coarse grid with LMedS motion
  guesses), ``sync`` (outer loop: per-frame L-BFGS on motion, backtracking
  step on delay), ``full_sync`` (pre_sync + ``iterations`` × sync).

One deliberate deviation: upstream solves the per-frame motion with
argmin's LBFGS (Armijo backtracking, memory 10, grad-tol 1e-4); the port
uses ``scipy.optimize.minimize(method="L-BFGS-B")`` with the *same*
analytic gradient — the same quasi-Newton family, verified by convergence
to ground truth rather than bit-exactness (the loss surface is smooth and
convex near the solution, so both land on it).
"""

from __future__ import annotations

import bisect
import logging
import math
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)

_NUMERIC_DIFF_STEP = 1e-6


# ---------------------------------------------------------------------------
# quat.rs
# ---------------------------------------------------------------------------


def quat_prod(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton product, (w, x, y, z) convention (``quat.rs:37-41``)."""
    w1, x1, y1, z1 = p
    w2, x2, y2, z2 = q
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate_point(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector by a quaternion (``quat.rs:53-55``)."""
    return quat_prod(quat_prod(q, np.array([0.0, *p])), quat_conj(q))[1:]


def quat_slerp(p: np.ndarray, q: np.ndarray, t: float) -> np.ndarray:
    """Slerp with hemisphere alignment (``quat.rs:62-78``)."""
    q = q.copy()
    if float(np.dot(p, q)) < 0.0:
        q = -q
    theta = math.acos(float(np.dot(p, q)))
    if theta > 1e-9:
        sin_theta = math.sin(theta)
        mult1 = math.sin((1.0 - t) * theta) / sin_theta
        mult2 = math.sin(t * theta) / sin_theta
    else:
        mult1, mult2 = 1.0 - t, t
    return mult1 * p + mult2 * q


def safe_normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        return v
    return v / norm


def clamp_k(k: float) -> float:
    return min(max(k, 1e1), 1e3)


# ---------------------------------------------------------------------------
# minispline.rs — cubic spline over one component, unit-spacing knots
# ---------------------------------------------------------------------------


class Spline:
    """Natural-ish cubic spline (``minispline.rs``): tridiagonal second
    derivatives, unit knot spacing, evaluated past the ends with the first
    resp. last segment's polynomial."""

    def __init__(self, y: list[float] | np.ndarray) -> None:
        y = np.asarray(y, dtype=np.float64)
        n = len(y)
        self.m_y = y.copy()
        self.m_b = np.zeros(n)
        self.m_c = np.zeros(n)
        self.m_d = np.zeros(n)
        if n == 0:
            return
        if n == 1:
            # Upstream underflows here (n-2 on usize); a single sample is a
            # constant spline.
            return

        # Tridiagonal system for the second derivatives (m_c).
        a = np.zeros((n, 3))  # sub, diag, super
        if n > 2:
            a[1:n - 1, 0] = 1.0 / 3.0
            a[1:n - 1, 1] = 2.0 / 3.0 * 2.0
            a[1:n - 1, 2] = 1.0 / 3.0
            self.m_c[1:n - 1] = y[2:n] - 2.0 * y[1:n - 1] + y[0:n - 2]
        if n >= 2:
            a[0, 1] = 2.0
            self.m_c[0] = 0.0
            a[n - 1, 1] = 2.0
            a[n - 1, 0] = 0.0
            self.m_c[n - 1] = 0.0

        # Forward + backward elimination, as upstream writes it.
        for i in range(n - 2):
            k = 1.0 / a[i, 1] * a[i + 1, 0]
            a[i + 1, 0] -= a[i, 1] * k
            self.m_c[i + 1] -= self.m_c[i] * k
        for i in range(n - 2, 0, -1):
            k = 1.0 / a[i, 1] * a[i - 1, 2]
            a[i - 1, 1] -= a[i, 0] * k
            self.m_c[i - 1] -= self.m_c[i] * k
        self.m_c /= a[:, 1]

        self.m_d[:n - 1] = 1.0 / 3.0 * (self.m_c[1:n] - self.m_c[:n - 1])
        self.m_b[:n - 1] = (
            (y[1:n] - y[:n - 1])
            - 1.0 / 3.0 * (2.0 * self.m_c[:n - 1] + self.m_c[1:n])
        )
        if n >= 2:
            self.m_d[n - 1] = 0.0
            self.m_b[n - 1] = (
                3.0 * self.m_d[n - 2] + 2.0 * self.m_c[n - 2] + self.m_b[n - 2]
            )

    def eval(self, x: float) -> float | None:
        n = len(self.m_b)
        if n == 0:
            return None
        idx = max(0, min(int(math.floor(x)), n))
        h = x - idx
        if x < idx:  # x negative, idx clamped to 0
            return (self.m_c[0] * h + self.m_b[0]) * h + self.m_y[0]
        if x > n - 1.0:
            return (self.m_c[n - 1] * h + self.m_b[n - 1]) * h + self.m_y[n - 1]
        return ((self.m_d[idx] * h + self.m_c[idx]) * h + self.m_b[idx]) * h \
            + self.m_y[idx]

    def deriv(self, x: float) -> float | None:
        n = len(self.m_b)
        if n == 0:
            return None
        idx = max(0, min(int(math.floor(x)), n))
        h = x - idx
        if x < 0.0:
            return 2.0 * self.m_c[0] * h + self.m_b[0]
        if x > n - 1.0:
            return 2.0 * self.m_c[n - 1] * h + self.m_b[n - 1]
        return (3.0 * self.m_d[idx] * h + 2.0 * self.m_c[idx]) * h + self.m_b[idx]


class NdSpline:
    """Four splines, one per quaternion component (``ndspline.rs``)."""

    def __init__(self) -> None:
        self.splines: list[Spline] = []

    @staticmethod
    def make(quats: np.ndarray) -> "NdSpline":
        """*quats*: shape (4, N) — one row per component."""
        ret = NdSpline()
        for row in np.asarray(quats, dtype=np.float64):
            ret.splines.append(Spline(row))
        return ret

    def eval(self, t: float) -> np.ndarray:
        ret = np.zeros(4)
        for i, sp in enumerate(self.splines):
            val = sp.eval(t)
            if val is not None:
                ret[i] = val
        return ret

    def deriv(self, t: float) -> np.ndarray:
        ret = np.zeros(4)
        for i, sp in enumerate(self.splines):
            val = sp.deriv(t)
            if val is not None:
                ret[i] = val
        return ret

    def is_empty(self) -> bool:
        return not self.splines


# ---------------------------------------------------------------------------
# Backtrack (backtrack.rs)
# ---------------------------------------------------------------------------


class Backtrack:
    """1-D backtracking line search step on the delay (``backtrack.rs``)."""

    def __init__(self) -> None:
        self.sufficient_decrease = 0.7
        self.decay = 0.1
        self.initial_step = 1.0
        self.max_iterations = 20
        self._f_and_grad: Callable[[float], tuple[float, float]] | None = None
        self._f_only: Callable[[float], float] | None = None

    def set_hyper(self, sufficient_decrease: float, decay: float,
                  initial_step: float, max_iterations: int) -> None:
        self.sufficient_decrease = sufficient_decrease
        self.decay = decay
        self.initial_step = initial_step
        self.max_iterations = max_iterations

    def set_objective(self, f_and_grad: Callable[[float], tuple[float, float]]) -> None:
        self._f_and_grad = f_and_grad
        if self._f_only is None:
            self._f_only = lambda x: f_and_grad(x)[0]

    def set_objective_f_only(self, f_only: Callable[[float], float]) -> None:
        self._f_only = f_only

    def step(self, x0: float) -> float:
        assert self._f_and_grad is not None and self._f_only is not None
        v, p = self._f_and_grad(x0)
        m = p * p
        t = self.initial_step
        for _ in range(self.max_iterations):
            v1 = self._f_only(x0 - t * p)
            if v - v1 >= t * self.sufficient_decrease * m:
                break
            t *= self.decay
        return -t * p


# ---------------------------------------------------------------------------
# The problem (lib.rs)
# ---------------------------------------------------------------------------


class FrameData:
    """One tracked frame pair: per-point exposure times and unit rays."""

    __slots__ = ("ts_a", "ts_b", "rays_a", "rays_b")

    def __init__(self, ts_a: np.ndarray, ts_b: np.ndarray,
                 rays_a: np.ndarray, rays_b: np.ndarray) -> None:
        self.ts_a = ts_a
        self.ts_b = ts_b
        self.rays_a = rays_a
        self.rays_b = rays_b


class OptData:
    def __init__(self) -> None:
        self.quats_start = 0.0
        self.sample_rate = 0.0
        self.quats = NdSpline()
        self.frame_data: dict[int, FrameData] = {}


def opt_compute_problem(timestamp_us: int, gyro_delay: float,
                        data: OptData) -> np.ndarray:
    """(N, 3) cross products of the gyro-rotated ray pairs
    (``lib.rs:406-428``): the residual of a pure-rotation model is zero for
    a correct delay, so what remains is explained by camera translation —
    the quantity the motion estimate captures."""
    flow = data.frame_data.get(timestamp_us)
    if flow is None:
        return np.zeros((0, 3))
    at = (flow.ts_a - data.quats_start + gyro_delay) * data.sample_rate
    bt = (flow.ts_b - data.quats_start + gyro_delay) * data.sample_rate
    problem = np.zeros((len(at), 3))
    for i in range(len(at)):
        a = safe_normalize(data.quats.eval(at[i]))
        b = safe_normalize(data.quats.eval(bt[i]))
        ar = quat_rotate_point(quat_conj(a), flow.rays_a[i])
        br = quat_rotate_point(quat_conj(b), flow.rays_b[i])
        problem[i] = np.cross(ar, br)
    return problem


def opt_guess_translational_motion(problem: np.ndarray,
                                   max_iters: int) -> np.ndarray:
    """LMedS translational-motion guess (``lib.rs:430-464``): random ray
    pairs, the cross product normalised is a candidate motion direction,
    scored by the first-quartile squared residual."""
    if len(problem) == 0:
        return np.zeros(0)
    nproblem = np.array([
        safe_normalize(row) for row in problem
    ], dtype=np.float64)

    rng = np.random.default_rng()
    best_sol = np.zeros(3)
    least_med = float("inf")
    n = len(problem)
    for _ in range(max_iters):
        # mtrand(0, n-1) upstream is inclusive; numpy's integers() is
        # half-open, so [0, n) is the same range.
        vs = [int(rng.integers(0, n)), int(rng.integers(0, n))]
        while n > 1 and vs[0] == vs[1]:
            vs[1] = int(rng.integers(0, n))
        v = safe_normalize(np.cross(problem[vs[0]], problem[vs[1]]))
        residuals = nproblem @ v
        med = np.sort(residuals ** 2)[len(residuals) // 4]
        if med < least_med:
            least_med = med
            best_sol = v
    return best_sol


def _sqr_jac(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Upstream's ``sqr_jac`` — kept for reference/tests of the chain."""
    return x * x, np.diag(2.0 * x)


class FrameState:
    """Per-frame loss with its analytic gradient chain (``lib.rs:39-100``)."""

    def __init__(self, timestamp_us: int, problem: OptData) -> None:
        self.timestamp_us = timestamp_us
        self.problem = problem
        self.motion_vec = np.zeros(3)
        self.var_k = 1e3

    def loss_single(self, gyro_delay: float, motion: np.ndarray) -> float:
        p = opt_compute_problem(self.timestamp_us, gyro_delay, self.problem)
        r = (p @ motion) * (self.var_k / np.linalg.norm(motion))
        return float(np.log1p(r * r).sum())

    def loss(self, gyro_delay: float, motion: np.ndarray):
        """(cost, d/delay, d/dmotion) — upstream ``FrameState::loss``
        (``lib.rs:57-79``).

        The cost is ``Σ log1p((Pm)_i² · var_k² / ‖m‖²)``. Upstream builds
        it as a chain of jacobian-carrying operators (``sqr_jac`` /
        ``sum_jac`` / ``div_jac`` / ``log1p_jac``, with a numeric
        difference for the delay); written out algebraically the same chain
        is, with ``v1 = Pm``, ``v5 = ‖m‖²/var_k²``::

            dcost/d(Pm)_i = 1/(1+v6_i) · 2·v1_i/v5
            dcost/dv5     = Σ_i 1/(1+v6_i) · −v2_i/v5² ,  dv5/dm = 2m/var_k²

        which is what the two ``jac_motion`` terms below are, term by term.
        """
        p = opt_compute_problem(self.timestamp_us, gyro_delay, self.problem)

        loss_l = self.loss_single(gyro_delay - _NUMERIC_DIFF_STEP, motion)
        loss_r = self.loss_single(gyro_delay + _NUMERIC_DIFF_STEP, motion)

        v1 = p @ motion                # (N,)
        v2 = v1 * v1                   # (N,)
        v5 = float(motion @ motion) / (self.var_k * self.var_k)  # scalar
        v6 = v2 / v5                   # (N,)
        cost = float(np.log1p(v6).sum())

        g_v6 = 1.0 / (1.0 + v6)                       # dcost/dv6
        jac_motion = p.T @ (g_v6 * (2.0 * v1 / v5))   # j8·j7·j6a·j2·j1
        g_v5 = float(np.dot(g_v6, -(v2 / (v5 * v5))))  # j8·j7·j6b (scalar)
        jac_motion = jac_motion + g_v5 * (2.0 * motion / (self.var_k * self.var_k))

        jac_delay = (loss_r - loss_l) / 2.0 / _NUMERIC_DIFF_STEP

        return cost, jac_delay, jac_motion

    def guess_motion(self, gyro_delay: float) -> np.ndarray:
        p = opt_compute_problem(self.timestamp_us, gyro_delay, self.problem)
        return opt_guess_translational_motion(p, 200)

    def guess_k(self, gyro_delay: float) -> float:
        p = opt_compute_problem(self.timestamp_us, gyro_delay, self.problem)
        return clamp_k(1.0 / np.linalg.norm(p @ self.motion_vec) * 1e2)


class SyncProblem:
    """The full sync problem (``lib.rs:103-404``)."""

    def __init__(self) -> None:
        self.problem = OptData()
        self._progress_cb = None

    def on_progress(self, cb) -> None:
        self._progress_cb = cb

    # -- inputs ------------------------------------------------------------

    def set_gyro_quaternions_fixed(
        self, data: list[tuple[float, float, float, float]],
        sample_rate: float, first_timestamp: float,
    ) -> None:
        self.problem.sample_rate = sample_rate
        self.problem.quats_start = first_timestamp
        self.problem.quats = NdSpline.make(np.array(data, dtype=np.float64).T)

    def set_gyro_quaternions(
        self, timestamps_us: list[int],
        quats: list[tuple[float, float, float, float]],
    ) -> None:
        """Resample onto a uniform grid rounded to the nearest 50 Hz
        (``lib.rs:120-174``), slerp-interpolating between input samples."""
        if not timestamps_us:
            logger.error("Empty timestamps! quats.len: %s", len(quats))
            return
        us_in_sec = 1_000_000
        count = len(timestamps_us)

        actual_sr_uhz = 1_000_000 * us_in_sec * count // max(
            1, timestamps_us[count - 1] - timestamps_us[0]
        )
        rounded_sr_hz = int(round(actual_sr_uhz / 50.0 / 1_000_000.0)) * 50
        if rounded_sr_hz <= 0:
            logger.error(
                "Invalid sample rate, count: %s, ts diff: %s",
                count, timestamps_us[count - 1] - timestamps_us[0],
            )
            return

        new_ts: list[int] = []
        sample = timestamps_us[0] * rounded_sr_hz // us_in_sec
        while us_in_sec * sample // rounded_sr_hz < timestamps_us[count - 1]:
            new_ts.append(us_in_sec * sample // rounded_sr_hz)
            sample += 1

        for i in range(1, count):
            if timestamps_us[i - 1] > timestamps_us[i]:
                logger.error("timestamps out of order at pos %s", i)

        new_quats = np.zeros((4, len(new_ts)))
        for i, ts in enumerate(new_ts):
            idx = bisect.bisect_left(timestamps_us, ts)
            if idx > 0:
                denom = timestamps_us[idx] - timestamps_us[idx - 1]
                t = (ts - timestamps_us[idx - 1]) / denom if denom else 0.0
                a = np.array(quats[idx - 1], dtype=np.float64)
                b = np.array(quats[idx], dtype=np.float64)
                new_quats[:, i] = quat_slerp(a, b, t)
            else:
                new_quats[:, i] = quats[idx]

        if not new_ts:
            logger.error("Invalid new timestamps")
            return
        self.problem.sample_rate = float(rounded_sr_hz)
        self.problem.quats_start = new_ts[0] / us_in_sec
        self.problem.quats = NdSpline.make(new_quats)

    def set_track_result(
        self, timestamp_us: int, ts_a: list[float], ts_b: list[float],
        rays_a: list[tuple[float, float, float]],
        rays_b: list[tuple[float, float, float]],
    ) -> None:
        self.problem.frame_data[timestamp_us] = FrameData(
            np.asarray(ts_a, dtype=np.float64),
            np.asarray(ts_b, dtype=np.float64),
            np.asarray(rays_a, dtype=np.float64),
            np.asarray(rays_b, dtype=np.float64),
        )

    # -- search ------------------------------------------------------------

    def pre_sync(self, rough_delay: float, ts_from: int, ts_to: int,
                 search_step: float, search_radius: float) -> tuple[float, float] | None:
        """Coarse grid with per-frame LMedS motion guesses
        (``lib.rs:192-226``)."""
        if self.problem.quats.is_empty() or search_step <= 0.0 or search_radius <= 0.0:
            logger.error("Invalid params for pre_sync")
            return None
        timestamps = sorted(
            k for k in self.problem.frame_data if ts_from <= k < ts_to
        )
        results: list[tuple[float, float]] = []
        delay = rough_delay - search_radius
        while delay < rough_delay + search_radius:
            cost = 0.0
            for ts in timestamps:
                p = opt_compute_problem(ts, delay, self.problem)
                m = opt_guess_translational_motion(p, 20)
                k = clamp_k(1.0 / np.linalg.norm(p @ m) * 1e2)
                r = (p @ m) * (k / np.linalg.norm(m))
                cost += float(np.sqrt(np.sqrt(np.log1p(r * r)).sum()))
            results.append((cost, delay))
            delay += search_step
        results.sort(key=lambda x: x[0])
        return results[0]

    def sync(self, initial_delay: float, ts_from: int, ts_to: int,
             search_center: float, search_radius: float) -> tuple[float, float] | None:
        """Outer loop (``lib.rs:228-362``): per-frame L-BFGS on the motion,
        backtracking step on the delay, until the delay step collapses or
        the solution walks out of the radius."""
        if self.problem.quats.is_empty():
            logger.error("Empty quats!")
            return None
        gyro_delay = initial_delay

        costs = [
            FrameState(ts, self.problem)
            for ts in sorted(k for k in self.problem.frame_data
                             if ts_from <= k < ts_to)
        ]
        for cost in costs:
            cost.motion_vec = cost.guess_motion(gyro_delay)
            cost.var_k = cost.guess_k(gyro_delay)

        delay_optimizer = Backtrack()
        # Upstream hyper set: 2e-4 sufficient decrease, 0.1 decay,
        # 1e-3 initial step, 10 iterations.
        delay_optimizer.set_hyper(2e-4, 0.1, 1e-3, 10)

        def total_loss_and_grad(x: float) -> tuple[float, float]:
            total = 0.0
            grad = 0.0
            for fs in costs:
                c, g_delay, _ = fs.loss(x, fs.motion_vec)
                total += c
                grad += g_delay
            return total, grad

        delay_optimizer.set_objective(total_loss_and_grad)

        def simple_objective(x: float) -> float:
            return sum(fs.loss_single(x, fs.motion_vec) for fs in costs)

        delay_optimizer.set_objective_f_only(simple_objective)

        delay_b = 0.3
        delay_v = 0.0
        converge_counter = 0

        for _ in range(400):
            # Optimize each frame's motion with L-BFGS on the robust loss.
            from scipy.optimize import minimize

            for fs in costs:
                gyro_delay_f = gyro_delay

                def grad_fn(w, fs=fs, gyro_delay_f=gyro_delay_f):
                    _, _, gm = fs.loss(gyro_delay_f, w)
                    return gm

                def cost_fn(w, fs=fs, gyro_delay_f=gyro_delay_f):
                    return fs.loss_single(gyro_delay_f, w)

                res = minimize(
                    cost_fn, fs.motion_vec, jac=grad_fn, method="L-BFGS-B",
                    options={"maxiter": 200, "gtol": 1e-4},
                )
                if np.isfinite(res.x).all():
                    fs.motion_vec = res.x

            # Optimize the delay with the backtracking step.
            step = delay_optimizer.step(gyro_delay - delay_b * delay_v)
            delay_v = delay_b * delay_v + step
            gyro_delay += delay_v

            step_size = abs(step)
            if step_size < 1e-4:
                converge_counter += 1
            else:
                converge_counter = 0
            if converge_counter > 5:
                break
            if abs(gyro_delay - search_center) > search_radius:
                break

        return simple_objective(gyro_delay), gyro_delay

    def full_sync(self, initial_delay: float, ts_from: int, ts_to: int,
                  search_step: float, search_radius: float,
                  iterations: int) -> tuple[float, float] | None:
        """``lib.rs:364-383``: pre_sync, then ``iterations`` sync rounds."""
        delay = self.pre_sync(initial_delay, ts_from, ts_to,
                              search_step, search_radius)
        if delay is None:
            return None
        for i in range(iterations):
            if self._progress_cb is not None:
                if not self._progress_cb(0.5 + (i / iterations) * 0.5):
                    return None
            d = self.sync(delay[1], ts_from, ts_to, initial_delay, search_radius)
            if d is not None:
                delay = d
        if self._progress_cb is not None:
            self._progress_cb(1.0)
        return delay
