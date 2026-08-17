"""Complementary filter integrators — port of the full paper implementation.

Faithful port of ``msgyro-imu-integration/src/complementary.rs`` (itself the
"Keeping a Good Attitude: A Quaternion-Based Orientation Filter for IMUs and
MAVs" algorithm, Valenti et al.). The previous Python port mirrored an early
8-line simplified golden generator and diverged from the reference by ~0.71;
this port matches the Rust V2 filter.

Layout mirrors the Rust file: quaternion helpers, ComplementaryFilterV1
(original), ComplementaryFilterV2 (IIR-filtered accel, gravity autoscale,
adaptive gain), and the GyroIntegrator wrapper driving V2.
"""

from __future__ import annotations

import math

import numpy as np

from pygyroflow.types.quaternion import Quat64
from pygyroflow.types.time_types import TimeIMU, TimeQuat
from pygyroflow.imu_integration.base import GyroIntegrator

DEG2RAD: float = math.pi / 180.0

# ---------------------------------------------------------------------------
# Quaternion helpers (Rust: normalize_vector .. rotate_vector_by_quaternion)
# ---------------------------------------------------------------------------


def _normalize_vector(x: float, y: float, z: float) -> tuple[float, float, float]:
    norm = math.sqrt(x * x + y * y + z * z)
    if math.isfinite(norm) and norm != 0.0:
        return x / norm, y / norm, z / norm
    return x, y, z


def _normalize_quaternion(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if math.isfinite(norm) and norm != 0.0:
        return q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm
    return q


def _invert_quaternion(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (q[0], -q[1], -q[2], -q[3])


def _scale_quaternion(gain: float, dq: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Slerp between identity and dq by ``gain`` (Rust scale_quaternion)."""
    dq0, dq1, dq2, dq3 = dq
    if dq0 < 0.0:
        angle = math.acos(dq0)
        sin_angle = math.sin(angle)
        a = math.sin(angle * (1.0 - gain)) / sin_angle
        b = math.sin(angle * gain) / sin_angle
        dq0 = a + b * dq0
        dq1 *= b
        dq2 *= b
        dq3 *= b
    else:
        dq0 = (1.0 - gain) + gain * dq0
        dq1 *= gain
        dq2 *= gain
        dq3 *= gain
    return _normalize_quaternion((dq0, dq1, dq2, dq3))


def _quat_mul(
    p: tuple[float, float, float, float], q: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    p0, p1, p2, p3 = p
    q0, q1, q2, q3 = q
    return (
        p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3,
        p0 * q1 + p1 * q0 + p2 * q3 - p3 * q2,
        p0 * q2 - p1 * q3 + p2 * q0 + p3 * q1,
        p0 * q3 + p1 * q2 - p2 * q1 + p3 * q0,
    )


def _rotate_vector_by_quaternion(
    x: float, y: float, z: float, q: tuple[float, float, float, float]
) -> tuple[float, float, float]:
    q0, q1, q2, q3 = q
    return (
        (q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3) * x + 2.0 * (q1 * q2 - q0 * q3) * y + 2.0 * (q1 * q3 + q0 * q2) * z,
        2.0 * (q1 * q2 + q0 * q3) * x + (q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3) * y + 2.0 * (q2 * q3 - q0 * q1) * z,
        2.0 * (q1 * q3 - q0 * q2) * x + 2.0 * (q2 * q3 + q0 * q1) * y + (q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3) * z,
    )


# ---------------------------------------------------------------------------
# Complementary Filter V1 (original)
# ---------------------------------------------------------------------------

_V1_GRAVITY = 9.81
_V1_ANGULAR_VELOCITY_THRESHOLD = 0.2
_V1_ACCELERATION_THRESHOLD = 0.1
_V1_DELTA_ANGULAR_VELOCITY_THRESHOLD = 0.01


class ComplementaryFilterV1:
    """Original Valenti et al. filter (kept for parity with the Rust file)."""

    def __init__(self) -> None:
        self.gain_acc = 0.01
        self.gain_mag = 0.01
        self.bias_alpha = 0.01
        self.do_bias_estimation = True
        self.do_adaptive_gain = True
        self.initialized = False
        self.steady_state = False
        self.q: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        self.w_prev = (0.0, 0.0, 0.0)
        self.w_bias = (0.0, 0.0, 0.0)

    def get_orientation(self) -> tuple[float, float, float, float]:
        return _invert_quaternion(self.q)

    def set_orientation(self, q: tuple[float, float, float, float]) -> None:
        self.q = _invert_quaternion(q)

    def update(self, ax: float, ay: float, az: float, wx: float, wy: float, wz: float, dt: float) -> None:
        if not self.initialized:
            self.q = self._get_measurement(ax, ay, az)
            self.initialized = True
            return
        if self.do_bias_estimation:
            self._update_biases(ax, ay, az, wx, wy, wz)
        pred = self._get_prediction(wx, wy, wz, dt)
        dq_acc = self._get_acc_correction(ax, ay, az, pred)
        gain = self._get_adaptive_gain(self.gain_acc, ax, ay, az) if self.do_adaptive_gain else self.gain_acc
        dq_acc = _scale_quaternion(gain, dq_acc)
        self.q = _normalize_quaternion(_quat_mul(pred, dq_acc))

    def _update_biases(self, ax: float, ay: float, az: float, wx: float, wy: float, wz: float) -> None:
        self.steady_state = self._check_state(ax, ay, az, wx, wy, wz)
        if self.steady_state:
            self.w_bias = (
                self.w_bias[0] + self.bias_alpha * (wx - self.w_bias[0]),
                self.w_bias[1] + self.bias_alpha * (wy - self.w_bias[1]),
                self.w_bias[2] + self.bias_alpha * (wz - self.w_bias[2]),
            )
        self.w_prev = (wx, wy, wz)

    def _check_state(self, ax: float, ay: float, az: float, wx: float, wy: float, wz: float) -> bool:
        acc_magnitude = math.sqrt(ax * ax + ay * ay + az * az)
        if abs(acc_magnitude - _V1_GRAVITY) > _V1_ACCELERATION_THRESHOLD:
            return False
        if (
            abs(wx - self.w_prev[0]) > _V1_DELTA_ANGULAR_VELOCITY_THRESHOLD
            or abs(wy - self.w_prev[1]) > _V1_DELTA_ANGULAR_VELOCITY_THRESHOLD
            or abs(wz - self.w_prev[2]) > _V1_DELTA_ANGULAR_VELOCITY_THRESHOLD
        ):
            return False
        if (
            abs(wx - self.w_bias[0]) > _V1_ANGULAR_VELOCITY_THRESHOLD
            or abs(wy - self.w_bias[1]) > _V1_ANGULAR_VELOCITY_THRESHOLD
            or abs(wz - self.w_bias[2]) > _V1_ANGULAR_VELOCITY_THRESHOLD
        ):
            return False
        return True

    def _get_prediction(self, wx: float, wy: float, wz: float, dt: float) -> tuple[float, float, float, float]:
        wxu = wx - self.w_bias[0]
        wyu = wy - self.w_bias[1]
        wzu = wz - self.w_bias[2]
        q0, q1, q2, q3 = self.q
        return _normalize_quaternion((
            q0 + 0.5 * dt * (wxu * q1 + wyu * q2 + wzu * q3),
            q1 + 0.5 * dt * (-wxu * q0 - wyu * q3 + wzu * q2),
            q2 + 0.5 * dt * (wxu * q3 - wyu * q0 - wzu * q1),
            q3 + 0.5 * dt * (-wxu * q2 + wyu * q1 - wzu * q0),
        ))

    def _get_measurement(self, ax: float, ay: float, az: float) -> tuple[float, float, float, float]:
        ax, ay, az = _normalize_vector(ax, ay, az)
        if az >= 0.0:
            q0 = math.sqrt((az + 1.0) * 0.5)
            return (q0, -ay / (2.0 * q0), ax / (2.0 * q0), 0.0)
        x = math.sqrt((1.0 - az) * 0.5)
        return (-ay / (2.0 * x), x, 0.0, ax / (2.0 * x))

    def _get_acc_correction(self, ax: float, ay: float, az: float, p: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        ax, ay, az = _normalize_vector(ax, ay, az)
        g = _rotate_vector_by_quaternion(ax, ay, az, (p[0], -p[1], -p[2], -p[3]))
        dq0 = math.sqrt((g[2] + 1.0) * 0.5)
        return (dq0, -g[1] / (2.0 * dq0), g[0] / (2.0 * dq0), 0.0)

    def _get_adaptive_gain(self, alpha: float, ax: float, ay: float, az: float) -> float:
        a_mag = math.sqrt(ax * ax + ay * ay + az * az)
        error = abs(a_mag - _V1_GRAVITY) / _V1_GRAVITY
        error1, error2 = 0.1, 0.2
        m = 1.0 / (error1 - error2)
        b = 1.0 - m * error1
        factor = 1.0 if error < error1 else (m * error + b if error < error2 else 0.0)
        return factor * alpha


# ---------------------------------------------------------------------------
# Complementary Filter V2 (IIR-filtered accel, gravity autoscale, adaptive gain)
# ---------------------------------------------------------------------------

_V2_GRAVITY = 9.81
_V2_ANGULAR_VELOCITY_THRESHOLD = 0.01
_V2_ACCELERATION_THRESHOLD = 0.1
_V2_DELTA_ANGULAR_VELOCITY_THRESHOLD = 0.01
_V2_DELTA_ACCELERATION_THRESHOLD = 0.05
_V2_GRAV_AUTOSCALE_THRESHOLD = 1.0
_V2_ACC_FILT_TIMECONSTANT = 0.1
_V2_GRAV_AUTOSCALE_ALPHA = 0.005
_V2_STEADY_WAIT_THRESHOLD = 0.2


class ComplementaryFilterV2:
    """Improved filter — the one driven by the integrator wrapper."""

    def __init__(self) -> None:
        self.gain_acc = 0.0004
        self.prev_gain_acc = 0.0
        self.gain_mag = 0.0004
        self.bias_alpha = 0.001
        self.do_bias_estimation = True
        self.do_adaptive_gain = True
        self.do_gravity_autoscale = True
        self.gravity = 9.81
        self.initialized = False
        self.steady_state = False
        self.partial_steady_state = False
        self.q: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        self.a_filt = (0.0, 0.0, 0.0)
        self.a_prev = (0.0, 0.0, 0.0)
        self.w_prev = (0.0, 0.0, 0.0)
        self.w_bias = (0.0, 0.0, 0.0)
        self.time = 0.0
        self.time_steady = 0.0
        self.initial_settle_time = 2.0

    def set_initial_settle_time(self, settle_time: float) -> None:
        self.initial_settle_time = settle_time

    def get_orientation(self) -> tuple[float, float, float, float]:
        return _invert_quaternion(self.q)

    def set_orientation(self, q: tuple[float, float, float, float]) -> None:
        self.q = _invert_quaternion(q)

    def update(self, ax: float, ay: float, az: float, wx: float, wy: float, wz: float, dt: float) -> None:
        if not self.initialized:
            self.q = self._get_measurement(ax, ay, az)
            self.a_filt = (ax, ay, az)
            self.a_prev = (ax, ay, az)
            self.initialized = True
            return

        axf, ayf, azf = self._filter_acc(ax, ay, az, dt)
        self.steady_state = self._check_state(ax, ay, az, wx, wy, wz)
        self.time_steady = self.time_steady + dt if self.steady_state else 0.0

        if self.do_bias_estimation:
            self._update_biases(wx, wy, wz)
        if self.do_gravity_autoscale:
            self._autoscale_gravity()

        pred = self._get_prediction(wx, wy, wz, dt)
        dq_acc = self._get_acc_correction(axf, ayf, azf, pred)
        gain = self._get_adaptive_gain(self.gain_acc, axf, ayf, azf, dt)
        dq_acc = _scale_quaternion(gain, dq_acc)
        self.q = _normalize_quaternion(_quat_mul(pred, dq_acc))
        self.time += dt

    def _filter_acc(self, ax: float, ay: float, az: float, dt: float) -> tuple[float, float, float]:
        iir_alpha = 1.0 - math.exp(-dt / _V2_ACC_FILT_TIMECONSTANT)
        self.a_filt = (
            iir_alpha * ax + (1.0 - iir_alpha) * self.a_filt[0],
            iir_alpha * ay + (1.0 - iir_alpha) * self.a_filt[1],
            iir_alpha * az + (1.0 - iir_alpha) * self.a_filt[2],
        )
        return self.a_filt

    def _update_biases(self, wx: float, wy: float, wz: float) -> None:
        if self.time_steady > _V2_STEADY_WAIT_THRESHOLD:
            self.w_bias = (
                self.w_bias[0] + self.bias_alpha * (wx - self.w_bias[0]),
                self.w_bias[1] + self.bias_alpha * (wy - self.w_bias[1]),
                self.w_bias[2] + self.bias_alpha * (wz - self.w_bias[2]),
            )

    def _autoscale_gravity(self) -> None:
        if self.partial_steady_state:
            acc_magnitude = math.sqrt(
                self.a_filt[0] ** 2 + self.a_filt[1] ** 2 + self.a_filt[2] ** 2
            )
            if abs(acc_magnitude - _V2_GRAVITY) < _V2_GRAV_AUTOSCALE_THRESHOLD:
                self.gravity = self.gravity * (1.0 - _V2_GRAV_AUTOSCALE_ALPHA) + _V2_GRAV_AUTOSCALE_ALPHA * acc_magnitude

    def _check_state(self, ax: float, ay: float, az: float, wx: float, wy: float, wz: float) -> bool:
        acc_magnitude = math.sqrt(ax * ax + ay * ay + az * az)

        acc_th = abs(acc_magnitude - self.gravity) < _V2_ACCELERATION_THRESHOLD
        acc_component_steady = (
            abs(ax - self.a_filt[0]) < _V2_DELTA_ACCELERATION_THRESHOLD
            or abs(ay - self.a_filt[1]) < _V2_DELTA_ACCELERATION_THRESHOLD
            or abs(az - self.a_filt[2]) < _V2_DELTA_ACCELERATION_THRESHOLD
        )
        acc_delta_th = (
            abs(ax - self.a_prev[0]) < _V2_DELTA_ACCELERATION_THRESHOLD
            or abs(ay - self.a_prev[1]) < _V2_DELTA_ACCELERATION_THRESHOLD
            or abs(az - self.a_prev[2]) < _V2_DELTA_ACCELERATION_THRESHOLD
        )
        gyro_delta_th = (
            abs(wx - self.w_prev[0]) < _V2_DELTA_ANGULAR_VELOCITY_THRESHOLD
            or abs(wy - self.w_prev[1]) < _V2_DELTA_ANGULAR_VELOCITY_THRESHOLD
            or abs(wz - self.w_prev[2]) < _V2_DELTA_ANGULAR_VELOCITY_THRESHOLD
        )
        gyro_th = (
            abs(wx - self.w_bias[0]) < _V2_ANGULAR_VELOCITY_THRESHOLD
            or abs(wy - self.w_bias[1]) < _V2_ANGULAR_VELOCITY_THRESHOLD
            or abs(wz - self.w_bias[2]) < _V2_ANGULAR_VELOCITY_THRESHOLD
        )

        self.w_prev = (wx, wy, wz)
        self.a_prev = (ax, ay, az)

        self.partial_steady_state = acc_component_steady and acc_delta_th and gyro_delta_th and gyro_th
        return acc_th and self.partial_steady_state

    def _get_prediction(self, wx: float, wy: float, wz: float, dt: float) -> tuple[float, float, float, float]:
        wxu = wx - self.w_bias[0]
        wyu = wy - self.w_bias[1]
        wzu = wz - self.w_bias[2]
        q0, q1, q2, q3 = self.q
        return _normalize_quaternion((
            q0 + 0.5 * dt * (wxu * q1 + wyu * q2 + wzu * q3),
            q1 + 0.5 * dt * (-wxu * q0 - wyu * q3 + wzu * q2),
            q2 + 0.5 * dt * (wxu * q3 - wyu * q0 - wzu * q1),
            q3 + 0.5 * dt * (-wxu * q2 + wyu * q1 - wzu * q0),
        ))

    def _get_measurement(self, ax: float, ay: float, az: float) -> tuple[float, float, float, float]:
        ax, ay, az = _normalize_vector(ax, ay, az)
        if az >= 0.0:
            q0 = math.sqrt((az + 1.0) * 0.5)
            return (q0, -ay / (2.0 * q0), ax / (2.0 * q0), 0.0)
        x = math.sqrt((1.0 - az) * 0.5)
        return (-ay / (2.0 * x), x, 0.0, ax / (2.0 * x))

    def _get_acc_correction(self, ax: float, ay: float, az: float, p: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        ax, ay, az = _normalize_vector(ax, ay, az)
        g = _rotate_vector_by_quaternion(ax, ay, az, (p[0], -p[1], -p[2], -p[3]))
        dq0 = math.sqrt((g[2] + 1.0) * 0.5)
        return (dq0, -g[1] / (2.0 * dq0), g[0] / (2.0 * dq0), 0.0)

    def _get_adaptive_gain(self, alpha: float, ax: float, ay: float, az: float, dt: float) -> float:
        if not self.do_adaptive_gain:
            return alpha
        a_mag = math.sqrt(ax * ax + ay * ay + az * az)
        w_mag = math.sqrt(self.w_prev[0] ** 2 + self.w_prev[1] ** 2 + self.w_prev[2] ** 2)
        error = abs(a_mag - self.gravity) / self.gravity

        gain_iir_alpha = 1.0 - math.exp(-dt / 0.15)

        if self.time_steady > _V2_STEADY_WAIT_THRESHOLD:
            new_gain = 8.0 * alpha
        else:
            settle = max(15.0 - self.time / self.initial_settle_time * 14.0, 8.0) if self.time < self.initial_settle_time else 1.0
            new_gain = math.exp(-40.0 * error - 1.0 * w_mag) * alpha * settle
        if new_gain < self.prev_gain_acc:
            gain = new_gain
        else:
            gain = gain_iir_alpha * new_gain + (1.0 - gain_iir_alpha) * self.prev_gain_acc
        self.prev_gain_acc = gain
        return gain


# ---------------------------------------------------------------------------
# GyroIntegrator wrapper (drives V2, same input transforms as the Rust wrapper)
# ---------------------------------------------------------------------------


class ComplementaryIntegrator(GyroIntegrator):
    """Complementary filter (paper V2) — full port of the Rust integrator."""

    def integrate(self, imu_data: list[TimeIMU], duration_ms: float) -> TimeQuat:
        if not imu_data:
            return {}

        quats: TimeQuat = {}
        sample_time_ms = duration_ms / len(imu_data)

        f = ComplementaryFilterV2()
        f.set_initial_settle_time(min(duration_ms / 1000.0 * 0.05, 2.0))

        prev_time = imu_data[0].timestamp_ms - sample_time_ms

        for v in imu_data:
            if v.gyro is None:
                continue
            g = v.gyro
            a = list(v.accl if v.accl is not None else (0.0, 0.0, 0.0))
            # Rust nudges exactly-zero accel to avoid a degenerate measurement
            if abs(a[0]) == 0.0 and abs(a[1]) == 0.0 and abs(a[2]) == 0.0:
                a[0] += 0.0000001

            f.update(
                -a[1], a[0], a[2],
                -g[1] * DEG2RAD, g[0] * DEG2RAD, g[2] * DEG2RAD,
                (v.timestamp_ms - prev_time) / 1000.0,
            )

            w, x, y, z = f.get_orientation()
            quats[int(v.timestamp_ms * 1000.0)] = Quat64.from_quaternion(
                np.array([w, x, y, z])
            )
            prev_time = v.timestamp_ms

        return quats
