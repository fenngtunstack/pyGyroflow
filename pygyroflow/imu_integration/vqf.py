"""VQF (Versatile Quaternion-based Filter) integrator.

Complete Python port of Gyroflow's VQF implementation (vqf.rs, ~1485 lines).
Based on: D. Laidig, T. Seel. "VQF: Highly Accurate IMU Orientation Estimation
with Bias Estimation and Magnetic Disturbance Rejection."
https://arxiv.org/abs/2203.17024

The VQF algorithm separates 3D attitude estimation into:
1. 3D tilt (pitch+roll) via gyro integration + accelerometer correction
2. Heading (yaw) via magnetometer (optional)
3. Gyroscope bias estimation via Extended Kalman Filter (EKF)

Gyroflow uses the offline mode (forward-backward bidirectional filtering) for
maximum accuracy, with large tau values (tau_acc=40, tau_mag=40) because the
forward-backward processing compensates for individual direction latency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from pygyroflow.types.quaternion import Quat64
from pygyroflow.types.time_types import TimeIMU, TimeQuat
from pygyroflow.imu_integration.base import GyroIntegrator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EPS: float = float(np.finfo(np.float64).eps)
NAN: float = float("nan")
DEG2RAD: float = math.pi / 180.0
M_PI: float = math.pi
M_SQRT2: float = math.sqrt(2.0)


def _square(x: float) -> float:
    return x * x


# ---------------------------------------------------------------------------
# VQFParams
# ---------------------------------------------------------------------------
@dataclass
class VQFParams:
    """VQF algorithm parameters."""

    tau_acc: float = 3.0
    tau_mag: float = 9.0
    motion_bias_est_enabled: bool = True
    rest_bias_est_enabled: bool = True
    mag_dist_rejection_enabled: bool = True
    bias_sigma_init: float = 0.5
    bias_forgetting_time: float = 100.0
    bias_clip: float = 2.0
    bias_sigma_motion: float = 0.1
    bias_vertical_forgetting_factor: float = 0.0001
    bias_sigma_rest: float = 0.03
    rest_min_t: float = 1.5
    rest_filter_tau: float = 0.5
    rest_th_gyr: float = 2.0
    rest_th_acc: float = 0.5
    mag_current_tau: float = 0.05
    mag_ref_tau: float = 20.0
    mag_norm_th: float = 0.1
    mag_dip_th: float = 10.0
    mag_new_time: float = 20.0
    mag_new_first_time: float = 5.0
    mag_new_min_gyr: float = 20.0
    mag_min_undisturbed_time: float = 0.5
    mag_max_rejection_time: float = 60.0
    mag_rejection_factor: float = 2.0


# ---------------------------------------------------------------------------
# VQFCoefficients
# ---------------------------------------------------------------------------
@dataclass
class VQFCoefficients:
    """Pre-computed filter coefficients."""

    gyr_ts: float = 0.0
    acc_ts: float = 0.0
    mag_ts: float = 0.0
    acc_lp_b: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    acc_lp_a: list[float] = field(default_factory=lambda: [0.0, 0.0])
    k_mag: float = 0.0
    bias_p0: float = 0.0
    bias_v: float = 0.0
    bias_motion_w: float = 0.0
    bias_vertical_w: float = 0.0
    bias_rest_w: float = 0.0
    rest_gyr_lp_b: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rest_gyr_lp_a: list[float] = field(default_factory=lambda: [0.0, 0.0])
    rest_acc_lp_b: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rest_acc_lp_a: list[float] = field(default_factory=lambda: [0.0, 0.0])
    k_mag_ref: float = 0.0
    mag_norm_dip_lp_b: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    mag_norm_dip_lp_a: list[float] = field(default_factory=lambda: [0.0, 0.0])


# ---------------------------------------------------------------------------
# VQFState
# ---------------------------------------------------------------------------
@dataclass
class VQFState:
    """VQF algorithm internal state."""

    gyr_quat: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    acc_quat: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    delta: float = 0.0
    rest_detected: bool = False
    mag_dist_detected: bool = True
    last_acc_lp: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    acc_lp_state: list[float] = field(default_factory=lambda: [NAN] * 6)
    last_acc_corr_angular_rate: float = 0.0
    k_mag_init: float = 1.0
    last_mag_dis_angle: float = 0.0
    last_mag_corr_angular_rate: float = 0.0
    bias: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    bias_p: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    motion_bias_est_r_lp_state: list[float] = field(default_factory=lambda: [NAN] * 18)
    motion_bias_est_bias_lp_state: list[float] = field(default_factory=lambda: [NAN] * 4)
    rest_last_squared_deviations: list[float] = field(default_factory=lambda: [0.0, 0.0])
    rest_t: float = 0.0
    rest_last_gyr_lp: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rest_gyr_lp_state: list[float] = field(default_factory=lambda: [NAN] * 6)
    rest_last_acc_lp: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rest_acc_lp_state: list[float] = field(default_factory=lambda: [NAN] * 6)
    mag_ref_norm: float = 0.0
    mag_ref_dip: float = 0.0
    mag_undisturbed_t: float = 0.0
    mag_reject_t: float = 0.0
    mag_candidate_norm: float = -1.0
    mag_candidate_dip: float = 0.0
    mag_candidate_t: float = 0.0
    mag_norm_dip: list[float] = field(default_factory=lambda: [0.0, 0.0])
    mag_norm_dip_lp_state: list[float] = field(default_factory=lambda: [NAN] * 4)


# ---------------------------------------------------------------------------
# Standalone math helpers (module-level, matching Rust's associated functions)
# ---------------------------------------------------------------------------

def quat_multiply(q1: list[float], q2: list[float]) -> list[float]:
    w = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
    x = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
    y = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
    z = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
    return [w, x, y, z]


def quat_conj(q: list[float]) -> list[float]:
    return [q[0], -q[1], -q[2], -q[3]]


def quat_apply_delta(q: list[float], delta: float) -> list[float]:
    c = math.cos(delta / 2.0)
    s = math.sin(delta / 2.0)
    w = c*q[0] - s*q[3]
    x = c*q[1] - s*q[2]
    y = c*q[2] + s*q[1]
    z = c*q[3] + s*q[0]
    return [w, x, y, z]


def quat_rotate(q: list[float], v: list[float]) -> list[float]:
    x = (1.0 - 2.0*q[2]*q[2] - 2.0*q[3]*q[3])*v[0] + 2.0*v[1]*(q[2]*q[1] - q[0]*q[3]) + 2.0*v[2]*(q[0]*q[2] + q[3]*q[1])
    y = 2.0*v[0]*(q[0]*q[3] + q[2]*q[1]) + v[1]*(1.0 - 2.0*q[1]*q[1] - 2.0*q[3]*q[3]) + 2.0*v[2]*(q[2]*q[3] - q[1]*q[0])
    z = 2.0*v[0]*(q[3]*q[1] - q[0]*q[2]) + 2.0*v[1]*(q[0]*q[1] + q[3]*q[2]) + v[2]*(1.0 - 2.0*q[1]*q[1] - 2.0*q[2]*q[2])
    return [x, y, z]


def norm(vec: list[float], n: int) -> float:
    s = 0.0
    for i in range(n):
        s += vec[i] * vec[i]
    return math.sqrt(s)


def normalize(vec: list[float], n: int) -> None:
    l = norm(vec, n)
    if l < EPS:
        return
    for i in range(n):
        vec[i] /= l


def clip_vec(vec: list[float], n: int, lo: float, hi: float) -> None:
    for i in range(n):
        if vec[i] < lo:
            vec[i] = lo
        elif vec[i] > hi:
            vec[i] = hi


def gain_from_tau(tau: float, ts: float) -> float:
    assert ts > 0.0
    if tau < 0.0:
        return 0.0
    elif tau == 0.0:
        return 1.0
    else:
        return 1.0 - math.exp(-ts / tau)


def filter_coeffs(tau: float, ts: float) -> tuple[list[float], list[float]]:
    assert tau > 0.0
    assert ts > 0.0
    fc = (M_SQRT2 / (2.0 * M_PI)) / tau
    c = math.tan(M_PI * fc * ts)
    d = c*c + M_SQRT2*c + 1.0
    b0 = c*c / d
    b = [b0, 2.0*b0, b0]
    a = [2.0*(c*c - 1.0)/d, (1.0 - M_SQRT2*c + c*c)/d]
    return b, a


def filter_initial_state(x0: float, b: list[float], a: list[float]) -> list[float]:
    return [x0 * (1.0 - b[0]), x0 * (b[2] - a[1])]


def filter_step(x: float, b: list[float], a: list[float], state: list[float], off: int = 0) -> float:
    """One IIR step updating ``state`` IN PLACE at ``state[off:off+2]``.

    The offset variant matters: passing ``state[off:off+2]`` as a slice
    would hand filter_step a COPY and silently drop every state update
    (the bug that made the whole VQF filter stateless in Python).
    """
    y = b[0]*x + state[off]
    state[off] = b[1]*x - a[0]*y + state[off+1]
    state[off+1] = b[2]*x - a[1]*y
    return y


def filter_vec(
    x: list[float],
    n: int,
    tau: float,
    ts: float,
    b: list[float],
    a: list[float],
    state: list[float],
    out: list[float],
) -> None:
    assert n >= 2

    if math.isnan(state[0]):
        # initialization phase
        if math.isnan(state[1]):
            state[1] = 0.0  # sample count
            for i in range(n):
                state[2 + i] = 0.0  # sum accumulator
        state[1] += 1.0
        for i in range(n):
            state[2 + i] += x[i]
            out[i] = state[2 + i] / state[1]
        if state[1] * ts >= tau:
            for i in range(n):
                init = filter_initial_state(out[i], b, a)
                state[2*i] = init[0]
                state[2*i + 1] = init[1]
        return

    for i in range(n):
        out[i] = filter_step(x[i], b, a, state, 2*i)


def matrix3_set_to_scaled_identity(scale: float) -> list[float]:
    return [scale, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, scale]


def matrix3_multiply(in1: list[float], in2: list[float]) -> list[float]:
    return [
        in1[0]*in2[0] + in1[1]*in2[3] + in1[2]*in2[6],
        in1[0]*in2[1] + in1[1]*in2[4] + in1[2]*in2[7],
        in1[0]*in2[2] + in1[1]*in2[5] + in1[2]*in2[8],
        in1[3]*in2[0] + in1[4]*in2[3] + in1[5]*in2[6],
        in1[3]*in2[1] + in1[4]*in2[4] + in1[5]*in2[7],
        in1[3]*in2[2] + in1[4]*in2[5] + in1[5]*in2[8],
        in1[6]*in2[0] + in1[7]*in2[3] + in1[8]*in2[6],
        in1[6]*in2[1] + in1[7]*in2[4] + in1[8]*in2[7],
        in1[6]*in2[2] + in1[7]*in2[5] + in1[8]*in2[8],
    ]


def matrix3_multiply_tps_first(in1: list[float], in2: list[float]) -> list[float]:
    return [
        in1[0]*in2[0] + in1[3]*in2[3] + in1[6]*in2[6],
        in1[0]*in2[1] + in1[3]*in2[4] + in1[6]*in2[7],
        in1[0]*in2[2] + in1[3]*in2[5] + in1[6]*in2[8],
        in1[1]*in2[0] + in1[4]*in2[3] + in1[7]*in2[6],
        in1[1]*in2[1] + in1[4]*in2[4] + in1[7]*in2[7],
        in1[1]*in2[2] + in1[4]*in2[5] + in1[7]*in2[8],
        in1[2]*in2[0] + in1[5]*in2[3] + in1[8]*in2[6],
        in1[2]*in2[1] + in1[5]*in2[4] + in1[8]*in2[7],
        in1[2]*in2[2] + in1[5]*in2[5] + in1[8]*in2[8],
    ]


def matrix3_multiply_tps_second(in1: list[float], in2: list[float]) -> list[float]:
    return [
        in1[0]*in2[0] + in1[1]*in2[1] + in1[2]*in2[2],
        in1[0]*in2[3] + in1[1]*in2[4] + in1[2]*in2[5],
        in1[0]*in2[6] + in1[1]*in2[7] + in1[2]*in2[8],
        in1[3]*in2[0] + in1[4]*in2[1] + in1[5]*in2[2],
        in1[3]*in2[3] + in1[4]*in2[4] + in1[5]*in2[5],
        in1[3]*in2[6] + in1[4]*in2[7] + in1[5]*in2[8],
        in1[6]*in2[0] + in1[7]*in2[1] + in1[8]*in2[2],
        in1[6]*in2[3] + in1[7]*in2[4] + in1[8]*in2[5],
        in1[6]*in2[6] + in1[7]*in2[7] + in1[8]*in2[8],
    ]


def matrix3_inv(mat: list[float]) -> list[float]:
    a = mat[4]*mat[8] - mat[5]*mat[7]
    d = mat[2]*mat[7] - mat[1]*mat[8]
    g = mat[1]*mat[5] - mat[2]*mat[4]
    b = mat[5]*mat[6] - mat[3]*mat[8]
    e = mat[0]*mat[8] - mat[2]*mat[6]
    h = mat[2]*mat[3] - mat[0]*mat[5]
    c = mat[3]*mat[7] - mat[4]*mat[6]
    f = mat[1]*mat[6] - mat[0]*mat[7]
    i = mat[0]*mat[4] - mat[1]*mat[3]

    det = mat[0]*a + mat[1]*b + mat[2]*c

    if -EPS <= det <= EPS:
        return [0.0] * 9

    return [a/det, d/det, g/det, b/det, e/det, h/det, c/det, f/det, i/det]


def matrix3_multiply_vec(r: list[float], v: list[float]) -> list[float]:
    return [
        r[0]*v[0] + r[1]*v[1] + r[2]*v[2],
        r[3]*v[0] + r[4]*v[1] + r[5]*v[2],
        r[6]*v[0] + r[7]*v[1] + r[8]*v[2],
    ]


# ---------------------------------------------------------------------------
# VQF class — online filter
# ---------------------------------------------------------------------------

class VQF:
    """Online VQF filter.

    Combines params, state, and coefficients. Provides single-step and batch
    update interfaces. Gyroflow mainly uses offline_vqf (forward-backward).
    """

    def __init__(self, params: VQFParams | None, gyr_ts: float, acc_ts: float = 0.0, mag_ts: float = 0.0) -> None:
        self.params = params if params is not None else VQFParams()
        self.state = VQFState()
        self.coeffs = VQFCoefficients()
        self.coeffs.gyr_ts = gyr_ts
        self.coeffs.acc_ts = acc_ts if acc_ts > 0.0 else gyr_ts
        self.coeffs.mag_ts = mag_ts if mag_ts > 0.0 else gyr_ts
        self._setup()

    # ---- setup / reset ----

    def _setup(self) -> None:
        assert self.coeffs.gyr_ts > 0.0
        assert self.coeffs.acc_ts > 0.0
        assert self.coeffs.mag_ts > 0.0

        b, a = filter_coeffs(self.params.tau_acc, self.coeffs.acc_ts)
        self.coeffs.acc_lp_b = b
        self.coeffs.acc_lp_a = a

        self.coeffs.k_mag = gain_from_tau(self.params.tau_mag, self.coeffs.mag_ts)

        self.coeffs.bias_p0 = _square(self.params.bias_sigma_init * 100.0)
        self.coeffs.bias_v = _square(0.1 * 100.0) * self.coeffs.acc_ts / self.params.bias_forgetting_time

        p_motion = _square(self.params.bias_sigma_motion * 100.0)
        self.coeffs.bias_motion_w = _square(p_motion) / self.coeffs.bias_v + p_motion
        self.coeffs.bias_vertical_w = self.coeffs.bias_motion_w / max(self.params.bias_vertical_forgetting_factor, 1e-10)

        p_rest = _square(self.params.bias_sigma_rest * 100.0)
        self.coeffs.bias_rest_w = _square(p_rest) / self.coeffs.bias_v + p_rest

        b, a = filter_coeffs(self.params.rest_filter_tau, self.coeffs.gyr_ts)
        self.coeffs.rest_gyr_lp_b = b
        self.coeffs.rest_gyr_lp_a = a
        b, a = filter_coeffs(self.params.rest_filter_tau, self.coeffs.acc_ts)
        self.coeffs.rest_acc_lp_b = b
        self.coeffs.rest_acc_lp_a = a

        self.coeffs.k_mag_ref = gain_from_tau(self.params.mag_ref_tau, self.coeffs.mag_ts)
        if self.params.mag_current_tau > 0.0:
            b, a = filter_coeffs(self.params.mag_current_tau, self.coeffs.mag_ts)
            self.coeffs.mag_norm_dip_lp_b = b
            self.coeffs.mag_norm_dip_lp_a = a
        else:
            self.coeffs.mag_norm_dip_lp_b = [NAN, NAN, NAN]
            self.coeffs.mag_norm_dip_lp_a = [NAN, NAN]

        self.reset_state()

    def reset_state(self) -> None:
        s = self.state
        s.gyr_quat = [1.0, 0.0, 0.0, 0.0]
        s.acc_quat = [1.0, 0.0, 0.0, 0.0]
        s.delta = 0.0
        s.rest_detected = False
        s.mag_dist_detected = True
        s.last_acc_lp = [0.0, 0.0, 0.0]
        s.acc_lp_state = [NAN] * 6
        s.last_acc_corr_angular_rate = 0.0
        s.k_mag_init = 1.0
        s.last_mag_dis_angle = 0.0
        s.last_mag_corr_angular_rate = 0.0
        s.bias = [0.0, 0.0, 0.0]
        s.bias_p = matrix3_set_to_scaled_identity(self.coeffs.bias_p0)
        s.motion_bias_est_r_lp_state = [NAN] * 18
        s.motion_bias_est_bias_lp_state = [NAN] * 4
        s.rest_last_squared_deviations = [0.0, 0.0]
        s.rest_t = 0.0
        s.rest_last_gyr_lp = [NAN, NAN, NAN]
        s.rest_gyr_lp_state = [NAN] * 6
        s.rest_last_acc_lp = [0.0, 0.0, 0.0]
        s.rest_acc_lp_state = [NAN] * 6
        s.mag_ref_norm = 0.0
        s.mag_ref_dip = 0.0
        s.mag_undisturbed_t = 0.0
        s.mag_reject_t = self.params.mag_max_rejection_time
        s.mag_candidate_norm = -1.0
        s.mag_candidate_dip = 0.0
        s.mag_candidate_t = 0.0
        s.mag_norm_dip = [0.0, 0.0]
        s.mag_norm_dip_lp_state = [NAN] * 4

    # ---- main update steps ----

    def update_gyr(self, gyr: list[float]) -> None:
        s = self.state
        c = self.coeffs
        p = self.params

        # rest detection
        if p.rest_bias_est_enabled or p.mag_dist_rejection_enabled:
            gyr_copy = list(gyr)
            filter_vec(
                gyr_copy, 3, p.rest_filter_tau, c.gyr_ts,
                c.rest_gyr_lp_b, c.rest_gyr_lp_a,
                s.rest_gyr_lp_state, s.rest_last_gyr_lp,
            )

            s.rest_last_squared_deviations[0] = (
                _square(gyr[0] - s.rest_last_gyr_lp[0])
                + _square(gyr[1] - s.rest_last_gyr_lp[1])
                + _square(gyr[2] - s.rest_last_gyr_lp[2])
            )

            bias_clip = p.bias_clip * DEG2RAD
            if (
                s.rest_last_squared_deviations[0] >= _square(p.rest_th_gyr * DEG2RAD)
                or abs(s.rest_last_gyr_lp[0]) > bias_clip
                or abs(s.rest_last_gyr_lp[1]) > bias_clip
                or abs(s.rest_last_gyr_lp[2]) > bias_clip
            ):
                s.rest_t = 0.0
                s.rest_detected = False

        # remove estimated bias
        gyr_no_bias = [gyr[0] - s.bias[0], gyr[1] - s.bias[1], gyr[2] - s.bias[2]]

        # gyro prediction step
        gyr_norm = norm(gyr_no_bias, 3)
        angle = gyr_norm * c.gyr_ts
        if gyr_norm > EPS:
            cos_half = math.cos(angle / 2.0)
            sin_half_norm = math.sin(angle / 2.0) / gyr_norm
            gyr_step_quat = [cos_half, sin_half_norm*gyr_no_bias[0], sin_half_norm*gyr_no_bias[1], sin_half_norm*gyr_no_bias[2]]
            s.gyr_quat = quat_multiply(s.gyr_quat, gyr_step_quat)
            normalize(s.gyr_quat, 4)

    def update_acc(self, acc: list[float]) -> None:
        s = self.state
        c = self.coeffs
        p = self.params

        # ignore [0 0 0]
        if acc[0] == 0.0 and acc[1] == 0.0 and acc[2] == 0.0:
            return

        # rest detection (accel dimension)
        if p.rest_bias_est_enabled:
            acc_copy = list(acc)
            filter_vec(
                acc_copy, 3, p.rest_filter_tau, c.acc_ts,
                c.rest_acc_lp_b, c.rest_acc_lp_a,
                s.rest_acc_lp_state, s.rest_last_acc_lp,
            )
            s.rest_last_squared_deviations[1] = (
                _square(acc[0] - s.rest_last_acc_lp[0])
                + _square(acc[1] - s.rest_last_acc_lp[1])
                + _square(acc[2] - s.rest_last_acc_lp[2])
            )
            if s.rest_last_squared_deviations[1] >= _square(p.rest_th_acc):
                s.rest_t = 0.0
                s.rest_detected = False
            else:
                s.rest_t += c.acc_ts
                if s.rest_t >= p.rest_min_t:
                    s.rest_detected = True

        # rotate acc to inertial frame, low-pass filter
        acc_earth = quat_rotate(s.gyr_quat, acc)
        filter_vec(
            acc_earth, 3, p.tau_acc, c.acc_ts,
            c.acc_lp_b, c.acc_lp_a, s.acc_lp_state, s.last_acc_lp,
        )

        # transform to 6D earth frame and normalize
        acc_earth = quat_rotate(s.acc_quat, s.last_acc_lp)
        normalize(acc_earth, 3)

        # inclination correction
        q_w = math.sqrt((acc_earth[2] + 1.0) / 2.0)
        if q_w > 1e-6:
            acc_corr_quat = [q_w, 0.5*acc_earth[1]/q_w, -0.5*acc_earth[0]/q_w, 0.0]
        else:
            acc_corr_quat = [0.0, 1.0, 0.0, 0.0]
        s.acc_quat = quat_multiply(acc_corr_quat, s.acc_quat)
        normalize(s.acc_quat, 4)

        s.last_acc_corr_angular_rate = math.acos(acc_earth[2]) / c.acc_ts

        # bias estimation (EKF)
        if p.motion_bias_est_enabled or p.rest_bias_est_enabled:
            bias_clip_val = p.bias_clip * DEG2RAD

            r = [NAN] * 9
            bias_lp = [NAN, NAN]

            # construct rotation matrix from 6D quaternion
            acc_gyr_quat = self.get_quat6d()
            r[0] = 1.0 - 2.0*_square(acc_gyr_quat[2]) - 2.0*_square(acc_gyr_quat[3])
            r[1] = 2.0*(acc_gyr_quat[2]*acc_gyr_quat[1] - acc_gyr_quat[0]*acc_gyr_quat[3])
            r[2] = 2.0*(acc_gyr_quat[0]*acc_gyr_quat[2] + acc_gyr_quat[3]*acc_gyr_quat[1])
            r[3] = 2.0*(acc_gyr_quat[0]*acc_gyr_quat[3] + acc_gyr_quat[2]*acc_gyr_quat[1])
            r[4] = 1.0 - 2.0*_square(acc_gyr_quat[1]) - 2.0*_square(acc_gyr_quat[3])
            r[5] = 2.0*(acc_gyr_quat[2]*acc_gyr_quat[3] - acc_gyr_quat[1]*acc_gyr_quat[0])
            r[6] = 2.0*(acc_gyr_quat[3]*acc_gyr_quat[1] - acc_gyr_quat[0]*acc_gyr_quat[2])
            r[7] = 2.0*(acc_gyr_quat[0]*acc_gyr_quat[1] + acc_gyr_quat[3]*acc_gyr_quat[2])
            r[8] = 1.0 - 2.0*_square(acc_gyr_quat[1]) - 2.0*_square(acc_gyr_quat[2])

            bias_lp[0] = r[0]*s.bias[0] + r[1]*s.bias[1] + r[2]*s.bias[2]
            bias_lp[1] = r[3]*s.bias[0] + r[4]*s.bias[1] + r[5]*s.bias[2]

            # low-pass filter R and R*bias
            r_in = list(r)
            filter_vec(
                r_in, 9, p.tau_acc, c.acc_ts,
                c.acc_lp_b, c.acc_lp_a, s.motion_bias_est_r_lp_state, r,
            )
            bias_lp_in = list(bias_lp)
            filter_vec(
                bias_lp_in, 2, p.tau_acc, c.acc_ts,
                c.acc_lp_b, c.acc_lp_a, s.motion_bias_est_bias_lp_state, bias_lp,
            )

            w = [0.0, 0.0, 0.0]
            e = [0.0, 0.0, 0.0]
            if s.rest_detected and p.rest_bias_est_enabled:
                e[0] = s.rest_last_gyr_lp[0] - s.bias[0]
                e[1] = s.rest_last_gyr_lp[1] - s.bias[1]
                e[2] = s.rest_last_gyr_lp[2] - s.bias[2]
                r = matrix3_set_to_scaled_identity(1.0)
                w = [c.bias_rest_w, c.bias_rest_w, c.bias_rest_w]
            elif p.motion_bias_est_enabled:
                e[0] = -acc_earth[1]/c.acc_ts + bias_lp[0] - r[0]*s.bias[0] - r[1]*s.bias[1] - r[2]*s.bias[2]
                e[1] = acc_earth[0]/c.acc_ts + bias_lp[1] - r[3]*s.bias[0] - r[4]*s.bias[1] - r[5]*s.bias[2]
                e[2] = -r[6]*s.bias[0] - r[7]*s.bias[1] - r[8]*s.bias[2]
                w[0] = c.bias_motion_w
                w[1] = c.bias_motion_w
                w[2] = c.bias_vertical_w
            else:
                w = [-1.0, -1.0, -1.0]  # disable update

            # Kalman filter prediction
            if s.bias_p[0] < c.bias_p0:
                s.bias_p[0] += c.bias_v
            if s.bias_p[4] < c.bias_p0:
                s.bias_p[4] += c.bias_v
            if s.bias_p[8] < c.bias_p0:
                s.bias_p[8] += c.bias_v

            if w[0] >= 0.0:
                clip_vec(e, 3, -bias_clip_val, bias_clip_val)

                # K = P * R^T * inv(W + R*P*R^T)
                k = matrix3_multiply_tps_second(s.bias_p, r)  # k = P * R^T
                k = matrix3_multiply(r, k)                     # k = R * P * R^T
                k[0] += w[0]
                k[4] += w[1]
                k[8] += w[2]                                  # k = W + R*P*R^T
                k = matrix3_inv(k)                             # k = inv(...)
                k = matrix3_multiply_tps_first(r, k)           # k = R^T * inv(...)
                k = matrix3_multiply(s.bias_p, k)              # k = P * R^T * inv(...)

                # bias = bias + K*e
                s.bias[0] += k[0]*e[0] + k[1]*e[1] + k[2]*e[2]
                s.bias[1] += k[3]*e[0] + k[4]*e[1] + k[5]*e[2]
                s.bias[2] += k[6]*e[0] + k[7]*e[1] + k[8]*e[2]

                # P = P - K*R*P
                k = matrix3_multiply(k, r)
                k = matrix3_multiply(k, s.bias_p)
                for i in range(9):
                    s.bias_p[i] -= k[i]

                clip_vec(s.bias, 3, -bias_clip_val, bias_clip_val)

    def update_mag(self, mag: list[float]) -> None:
        s = self.state
        c = self.coeffs
        p = self.params

        if mag[0] == 0.0 and mag[1] == 0.0 and mag[2] == 0.0:
            return

        acc_gyr_quat = self.get_quat6d()
        mag_earth = quat_rotate(acc_gyr_quat, mag)

        if p.mag_dist_rejection_enabled:
            s.mag_norm_dip[0] = norm(mag_earth, 3)
            s.mag_norm_dip[1] = -math.asin(mag_earth[2] / s.mag_norm_dip[0])

            if p.mag_current_tau > 0.0:
                nd_in = list(s.mag_norm_dip)
                filter_vec(
                    nd_in, 2, p.mag_current_tau, c.mag_ts,
                    c.mag_norm_dip_lp_b, c.mag_norm_dip_lp_a,
                    s.mag_norm_dip_lp_state, s.mag_norm_dip,
                )

            # mag disturbance detection
            if (
                abs(s.mag_norm_dip[0] - s.mag_ref_norm) < p.mag_norm_th * s.mag_ref_norm
                and abs(s.mag_norm_dip[1] - s.mag_ref_dip) < p.mag_dip_th * DEG2RAD
            ):
                s.mag_undisturbed_t += c.mag_ts
                if s.mag_undisturbed_t >= p.mag_min_undisturbed_time:
                    s.mag_dist_detected = False
                    s.mag_ref_norm += c.k_mag_ref * (s.mag_norm_dip[0] - s.mag_ref_norm)
                    s.mag_ref_dip += c.k_mag_ref * (s.mag_norm_dip[1] - s.mag_ref_dip)
            else:
                s.mag_undisturbed_t = 0.0
                s.mag_dist_detected = True

            # new mag reference acceptance
            if (
                abs(s.mag_norm_dip[0] - s.mag_candidate_norm) < p.mag_norm_th * s.mag_candidate_norm
                and abs(s.mag_norm_dip[1] - s.mag_candidate_dip) < p.mag_dip_th * DEG2RAD
            ):
                if norm(s.rest_last_gyr_lp, 3) >= p.mag_new_min_gyr * DEG2RAD:
                    s.mag_candidate_t += c.mag_ts
                s.mag_candidate_norm += c.k_mag_ref * (s.mag_norm_dip[0] - s.mag_candidate_norm)
                s.mag_candidate_dip += c.k_mag_ref * (s.mag_norm_dip[1] - s.mag_candidate_dip)

                if s.mag_dist_detected and (
                    s.mag_candidate_t >= p.mag_new_time
                    or (s.mag_ref_norm == 0.0 and s.mag_candidate_t >= p.mag_new_first_time)
                ):
                    s.mag_ref_norm = s.mag_candidate_norm
                    s.mag_ref_dip = s.mag_candidate_dip
                    s.mag_dist_detected = False
                    s.mag_undisturbed_t = p.mag_min_undisturbed_time
            else:
                s.mag_candidate_t = 0.0
                s.mag_candidate_norm = s.mag_norm_dip[0]
                s.mag_candidate_dip = s.mag_norm_dip[1]

        # disagreement angle
        s.last_mag_dis_angle = math.atan2(mag_earth[0], mag_earth[1]) - s.delta

        if s.last_mag_dis_angle > M_PI:
            s.last_mag_dis_angle -= 2.0 * M_PI
        elif s.last_mag_dis_angle < -M_PI:
            s.last_mag_dis_angle += 2.0 * M_PI

        k = c.k_mag

        if p.mag_dist_rejection_enabled:
            if s.mag_dist_detected:
                if s.mag_reject_t <= p.mag_max_rejection_time:
                    s.mag_reject_t += c.mag_ts
                    k = 0.0
                else:
                    k /= p.mag_rejection_factor
            else:
                s.mag_reject_t = max(s.mag_reject_t - p.mag_rejection_factor * c.mag_ts, 0.0)

        # fast initial convergence
        if s.k_mag_init != 0.0:
            if k < s.k_mag_init:
                k = s.k_mag_init
            s.k_mag_init = s.k_mag_init / (s.k_mag_init + 1.0)
            if s.k_mag_init * p.tau_mag < c.mag_ts:
                s.k_mag_init = 0.0

        s.delta += k * s.last_mag_dis_angle
        s.last_mag_corr_angular_rate = k * s.last_mag_dis_angle / c.mag_ts

        if s.delta > M_PI:
            s.delta -= 2.0 * M_PI
        elif s.delta < -M_PI:
            s.delta += 2.0 * M_PI

    def update(self, gyr: list[float], acc: list[float], mag: list[float] | None = None) -> None:
        self.update_gyr(gyr)
        self.update_acc(acc)
        if mag is not None:
            self.update_mag(mag)

    # ---- getters ----

    def get_quat6d(self) -> list[float]:
        return quat_multiply(self.state.acc_quat, self.state.gyr_quat)

    def get_quat9d(self) -> list[float]:
        return quat_apply_delta(quat_multiply(self.state.acc_quat, self.state.gyr_quat), self.state.delta)

    def get_bias_estimate(self) -> tuple[list[float], float]:
        bp = self.state.bias_p
        sum1 = abs(bp[0]) + abs(bp[1]) + abs(bp[2])
        sum2 = abs(bp[3]) + abs(bp[4]) + abs(bp[5])
        sum3 = abs(bp[6]) + abs(bp[7]) + abs(bp[8])
        p = max(sum1, sum2, sum3)
        p = min(p, self.coeffs.bias_p0)
        return list(self.state.bias), math.sqrt(p) * M_PI / 100.0 / 180.0

    def get_rest_detected(self) -> bool:
        return self.state.rest_detected

    def get_mag_dist_detected(self) -> bool:
        return self.state.mag_dist_detected


# ---------------------------------------------------------------------------
# Offline VQF helper functions
# ---------------------------------------------------------------------------

def _integrate_gyr(gyr: list[float], bias: list[float], n: int, ts: float) -> list[float]:
    """Integrate gyroscope with bias correction."""
    out = [0.0] * (n * 4)
    q = [1.0, 0.0, 0.0, 0.0]
    for i in range(n):
        gyr_no_bias = [gyr[3*i] - bias[3*i], gyr[3*i+1] - bias[3*i+1], gyr[3*i+2] - bias[3*i+2]]
        gyrnorm = norm(gyr_no_bias, 3)
        angle = gyrnorm * ts
        if gyrnorm > EPS:
            cos_h = math.cos(angle / 2.0)
            sin_h_n = math.sin(angle / 2.0) / gyrnorm
            gyr_step_quat = [cos_h, sin_h_n*gyr_no_bias[0], sin_h_n*gyr_no_bias[1], sin_h_n*gyr_no_bias[2]]
            q = quat_multiply(q, gyr_step_quat)
            normalize(q, 4)
        out[4*i] = q[0]
        out[4*i+1] = q[1]
        out[4*i+2] = q[2]
        out[4*i+3] = q[3]
    return out


def _lowpass_butter_filtfilt(acc_i: list[float], n: int, ts: float, tau: float) -> None:
    """Forward-backward Butterworth low-pass filter (filtfilt)."""
    b, a = filter_coeffs(tau, ts)
    state = [NAN] * 6

    # forward pass
    for i in range(n):
        inp = acc_i[3*i:3*i+3]
        aout = list(inp)  # will be overwritten by filter_vec
        filter_vec(inp, 3, tau, ts, b, a, state, aout)
        acc_i[3*i] = aout[0]
        acc_i[3*i+1] = aout[1]
        acc_i[3*i+2] = aout[2]

    # backward pass
    for j in range(3):
        init = filter_initial_state(acc_i[3*n - 3 + j], b, a)
        state[2*j] = init[0]
        state[2*j+1] = init[1]

    for i in range(n - 1, -1, -1):
        inp = acc_i[3*i:3*i+3]
        aout = list(inp)
        filter_vec(inp, 3, tau, ts, b, a, state, aout)
        acc_i[3*i] = aout[0]
        acc_i[3*i+1] = aout[1]
        acc_i[3*i+2] = aout[2]


def _acc_correction(quat3d: list[float], acc_i: list[float], n: int) -> list[float]:
    """Accelerometer tilt correction."""
    quat6d = [0.0] * (n * 4)
    acc_quat = [1.0, 0.0, 0.0, 0.0]

    for i in range(n):
        # transform acc from inertial frame to 6D earth frame and normalize
        acc_earth = quat_rotate(acc_quat, acc_i[3*i:3*i+3])
        normalize(acc_earth, 3)

        # inclination correction
        q_w = math.sqrt((acc_earth[2] + 1.0) / 2.0)
        if q_w > 1e-6:
            acc_corr_quat = [q_w, 0.5*acc_earth[1]/q_w, -0.5*acc_earth[0]/q_w, 0.0]
        else:
            acc_corr_quat = [0.0, 1.0, 0.0, 0.0]
        acc_quat = quat_multiply(acc_corr_quat, acc_quat)
        normalize(acc_quat, 4)

        # output quaternion
        qtemp = quat_multiply(acc_quat, quat3d[4*i:4*i+4])
        quat6d[4*i] = qtemp[0]
        quat6d[4*i+1] = qtemp[1]
        quat6d[4*i+2] = qtemp[2]
        quat6d[4*i+3] = qtemp[3]

    return quat6d


def _calculate_delta(quat6d: list[float], mag: list[float], n: int) -> list[float]:
    """Calculate heading disagreement angle delta."""
    delta = [0.0] * n
    for i in range(n):
        mag_earth = quat_rotate(quat6d[4*i:4*i+4], mag[3*i:3*i+3])
        delta[i] = math.atan2(mag_earth[0], mag_earth[1])
    return delta


def _filter_delta(
    mag_dist: list[bool],
    n: int,
    ts: float,
    params: VQFParams,
    backward: bool,
    delta: list[float],
) -> None:
    """First-order low-pass filter on delta (supports forward or backward)."""
    d = delta[n - 1] if backward else delta[0]
    k_mag = gain_from_tau(params.tau_mag, ts)
    k_mag_init = 1.0
    mag_reject_t = 0.0

    for i in range(n):
        j = n - i - 1 if backward else i
        dis_angle = delta[j] - d

        if dis_angle > M_PI:
            dis_angle -= 2.0 * M_PI
        elif dis_angle < -M_PI:
            dis_angle += 2.0 * M_PI

        k = k_mag

        if params.mag_dist_rejection_enabled:
            if mag_dist[j]:
                if mag_reject_t <= params.mag_max_rejection_time:
                    mag_reject_t += ts
                    k = 0.0
                else:
                    k /= params.mag_rejection_factor
            else:
                mag_reject_t = max(mag_reject_t - params.mag_rejection_factor * ts, 0.0)

        # fast initial convergence
        if k_mag_init != 0.0:
            if k < k_mag_init:
                k = k_mag_init
            k_mag_init = k_mag_init / (k_mag_init + 1.0)
            if k_mag_init * params.tau_mag < ts:
                k_mag_init = 0.0

        d += k * dis_angle

        if d > M_PI:
            d -= 2.0 * M_PI
        elif d < -M_PI:
            d += 2.0 * M_PI

        delta[j] = d


# ---------------------------------------------------------------------------
# offline_vqf — the main entry point used by Gyroflow
# ---------------------------------------------------------------------------

def offline_vqf(
    gyr: list[float],
    acc: list[float],
    mag: list[float] | None,
    n: int,
    ts: float,
    params: VQFParams,
) -> list[float]:
    """Run offline VQF (forward-backward bidirectional filtering).

    This is the Gyroflow entry point. Steps:
    1. Forward pass → save bias and covariance
    2. Backward pass (negated gyro) → save bias and covariance
    3. Covariance-weighted bias merge
    4. Re-integrate gyro with merged bias
    5. Forward-backward Butterworth filter on accelerometer (in inertial frame)
    6. Tilt correction
    7. If magnetometer: calculate delta, forward-backward filter on delta

    Returns:
        Flat quaternion array [w0,x0,y0,z0, w1,x1,y1,z1, ...] (6D or 9D).
    """
    quat6d = [0.0] * (n * 4)
    bias = [0.0] * (n * 3)
    mag_dist: list[bool] | None = [False] * n if mag is not None else None
    delta: list[float] | None = [0.0] * n if mag is not None else None

    # Step 1: forward pass
    vqf = VQF(params, ts, 0.0, 0.0)
    bias_p_inv1: list[list[float]] = [[] for _ in range(n)]

    for i in range(n):
        g = gyr[3*i:3*i+3]
        a = acc[3*i:3*i+3]
        if mag is not None:
            m = mag[3*i:3*i+3]
            vqf.update(g, a, m)
        else:
            vqf.update(g, a, None)

        if mag is not None:
            mag_dist[i] = vqf.get_mag_dist_detected()  # type: ignore[index]

        bias_est, _ = vqf.get_bias_estimate()
        bias[3*i] = bias_est[0]
        bias[3*i+1] = bias_est[1]
        bias[3*i+2] = bias_est[2]
        bias_p_inv1[i] = matrix3_inv(vqf.state.bias_p)

    # Step 2: backward pass (negated gyro)
    vqf.reset_state()
    for i in range(n - 1, -1, -1):
        temp_gyr = [-gyr[3*i], -gyr[3*i+1], -gyr[3*i+2]]
        a = acc[3*i:3*i+3]
        if mag is not None:
            m = mag[3*i:3*i+3]
            vqf.update(temp_gyr, a, m)
        else:
            vqf.update(temp_gyr, a, None)

        if mag is not None:
            mag_dist[i] = mag_dist[i] and vqf.get_mag_dist_detected()  # type: ignore[index]

        bias2, _ = vqf.get_bias_estimate()
        bias_p_inv2 = matrix3_inv(vqf.state.bias_p)

        # Step 3: covariance-weighted bias merge
        # P1^-1 * b1
        b1 = matrix3_multiply_vec(bias_p_inv1[i], bias[3*i:3*i+3])
        # P2^-1 * b2
        b2 = matrix3_multiply_vec(bias_p_inv2, bias2)
        # P1^-1 * b1 - P2^-1 * b2
        merged = [b1[0] - b2[0], b1[1] - b2[1], b1[2] - b2[2]]
        # (P1^-1 + P2^-1)
        for j in range(9):
            bias_p_inv1[i][j] += bias_p_inv2[j]
        bias_p_inv1[i] = matrix3_inv(bias_p_inv1[i])
        # (P1^-1 + P2^-1)^-1 * (P1^-1 * b1 - P2^-1 * b2)
        result = matrix3_multiply_vec(bias_p_inv1[i], merged)
        bias[3*i] = result[0]
        bias[3*i+1] = result[1]
        bias[3*i+2] = result[2]

    # Step 4: re-integrate gyro with merged bias
    quat3d = _integrate_gyr(gyr, bias, n, ts)

    # Transform acceleration to inertial frame
    acc_i: list[float] = []
    for i in range(n):
        acc_i.extend(quat_rotate(quat3d[4*i:4*i+4], acc[3*i:3*i+3]))

    # Step 5: forward-backward Butterworth low-pass filter
    _lowpass_butter_filtfilt(acc_i, n, ts, params.tau_acc)

    # Step 6: tilt correction
    quat6d = _acc_correction(quat3d, acc_i, n)

    # Step 7: heading correction (if magnetometer available)
    if mag is not None and delta is not None and mag_dist is not None:
        delta = _calculate_delta(quat6d, mag, n)
        _filter_delta(mag_dist, n, ts, params, False, delta)   # forward
        _filter_delta(mag_dist, n, ts, params, True, delta)    # backward

        # apply delta to get 9D quaternions
        for i in range(n):
            q9d = quat_apply_delta(quat6d[4*i:4*i+4], delta[i])
            quat6d[4*i] = q9d[0]
            quat6d[4*i+1] = q9d[1]
            quat6d[4*i+2] = q9d[2]
            quat6d[4*i+3] = q9d[3]

    return quat6d


# ---------------------------------------------------------------------------
# VQFIntegrator — GyroIntegrator interface
# ---------------------------------------------------------------------------

class VQFIntegrator(GyroIntegrator):
    """VQF integrator for PyGyroFlow.

    Collects gyro, accel, mag into flat arrays, runs offline_vqf with
    tau_acc=40.0 and tau_mag=40.0, then converts output quaternions to
    the TimeQuat format used by the rest of the codebase.

    Coordinate transform is applied before calling VQF:
        gyro: (-g[1]*DEG2RAD, g[0]*DEG2RAD, g[2]*DEG2RAD)
        accel: (-a[1], a[0], a[2])
        mag: (-m[1], m[0], m[2])
    """

    def integrate(self, imu_data: list[TimeIMU], duration_ms: float) -> TimeQuat:
        if not imu_data:
            return {}

        num_samples = len(imu_data)
        sample_time = duration_ms / (num_samples * 1000.0)

        # Collect flattened arrays with coordinate transform
        gyr: list[float] = []
        acc: list[float] = []
        mag: list[float] = []

        for v in imu_data:
            g = v.gyro if v.gyro is not None else np.zeros(3)
            a = v.accl if v.accl is not None else np.zeros(3)
            m = v.magn if v.magn is not None else np.zeros(3)

            # Coordinate transform: (x,y,z) -> (-y, x, z)
            # Gyro: degrees/s -> rad/s
            gyr.extend([-g[1] * DEG2RAD, g[0] * DEG2RAD, g[2] * DEG2RAD])
            acc.extend([-a[1], a[0], a[2]])
            mag.extend([-m[1], m[0], m[2]])

        params = VQFParams(
            tau_acc=40.0,
            tau_mag=40.0,
        )

        # ALWAYS pass mag through (zeros when absent), exactly like the
        # Rust wrapper: offline_vqf runs the 9D path with Some(&zeros).
        # Downgrading all-zero mag to the 6D path diverges from the
        # reference - zero-mag updates still change the VQF class state
        # (mag_dist / rest detection interplay).
        quat_flat = offline_vqf(
            gyr=gyr,
            acc=acc,
            mag=mag,
            n=num_samples,
            ts=sample_time,
            params=params,
        )

        # Convert to TimeQuat
        quats: TimeQuat = {}
        for i, v in enumerate(imu_data):
            w = quat_flat[4*i]
            x = quat_flat[4*i+1]
            y = quat_flat[4*i+2]
            z = quat_flat[4*i+3]
            ts_us = int(v.timestamp_ms * 1000.0)
            quats[ts_us] = Quat64.from_quaternion(np.array([w, x, y, z]))

        return quats
