"""GyroSource — raw IMU data management, integration, and quaternion lookup.

Port of Gyroflow's src/core/gyro_source/mod.rs GyroSource struct.
Central data holder for gyroscope data throughout the stabilization pipeline:
  - Stores raw IMU samples and integrated quaternions
  - Applies configurable transforms (orientation, rotation, filtering, bias)
  - Integrates raw IMU into quaternion orientations
  - Provides interpolated quaternion lookup at arbitrary timestamps
  - Manages gyro-to-video sync offsets
"""

from __future__ import annotations

import bisect
import logging
import math
from dataclasses import dataclass, field

import numpy as np

from pygyroflow.types.quaternion import Quat64
from pygyroflow.types.time_types import TimeIMU, TimeQuat, TimeVec
from pygyroflow.gyro_source.file_metadata import FileMetadata
from pygyroflow.gyro_source.imu_transforms import IMUTransforms
from pygyroflow.filtering import lowpass_filter_imu, median_filter_imu

log = logging.getLogger(__name__)


@dataclass
class FileLoadOptions:
    """Options for loading a telemetry file."""

    sample_index: int | None = None
    project_version: int = 0


class GyroSource:
    """Central gyro data manager.

    Holds raw IMU data, integrated quaternions, smoothed quaternions,
    and all metadata needed for the stabilization pipeline.

    Integration method index mapping (matches Rust):
        0: Use pre-integrated quaternions from telemetry
        1: Complementary filter
        2: VQF (default in Rust, falls back to Complementary if unavailable)
        3: Simple gyro integration
        4: Simple gyro + accel
        5: Mahony
        6: Madgwick
    """

    def __init__(self) -> None:
        self.file_load_options: FileLoadOptions = FileLoadOptions()

        self.duration_ms: float = 0.0

        self.raw_imu: list[TimeIMU] = []

        self.imu_transforms: IMUTransforms = IMUTransforms()

        self.integration_method: int = 2  # VQF (same default as Rust)

        self.quaternions: TimeQuat = {}          # Original (unsmoothed)
        self.smoothed_quaternions: TimeQuat = {}  # Smoothed

        self.use_gravity_vectors: bool = False
        self.horizon_lock_integration_method: int = 1
        # Upstream GyroSource::use_gravity_vectors (default off): feed the
        # accelerometer-derived gravity series into HorizonLock's gravity mode
        self.use_gravity_vectors: bool = False

        self.max_angles: tuple[float, float, float] = (0.0, 0.0, 0.0)  # (pitch, yaw, roll) deg

        self.smoothing_status: dict = {}

        self.prevent_recompute: bool = False

        self.file_metadata: FileMetadata = FileMetadata()

        self.offsets: dict[int, float] = {}         # timestamp_us -> offset_ms
        self.offsets_linear: dict[int, float] = {}   # linear fit offsets
        self.offsets_adjusted: dict[int, float] = {}  # (timestamp + offset) -> offset

        self.file_url: str = ""

    def has_motion(self) -> bool:
        """Check if any motion data is available."""
        return self.file_metadata.has_motion()

    def init_from_params(self, duration_ms: float) -> None:
        """Set duration from stabilization params."""
        self.duration_ms = duration_ms

    def clear(self) -> None:
        """Reset all data to initial state.

        ``imu_orientation`` and ``gyro_bias`` deliberately survive, matching
        upstream's ``GyroSource::clear`` (gyro_source/mod.rs): they are user
        settings, not per-load state. Replacing the whole ``IMUTransforms``
        object here (as this used to) silently dropped a gyro bias the caller
        had set before loading — and because ``has_any()`` then went False,
        ``apply_transforms`` took the "no transforms" branch and cleared
        ``raw_imu`` entirely, leaving ``_get_imu_data()`` to fall back to the
        untransformed samples. The bias was ignored with no error anywhere.
        """
        self.quaternions.clear()
        self.smoothed_quaternions.clear()
        self.raw_imu.clear()
        self.imu_transforms.imu_rotation_angles = None
        self.imu_transforms._imu_rotation_matrix = None
        self.imu_transforms.acc_rotation_angles = None
        self.imu_transforms._acc_rotation_matrix = None
        self.imu_transforms.imu_lpf = 0.0
        self.imu_transforms.imu_mf = 0
        self.file_metadata = FileMetadata()
        self.clear_offsets()

    def load_from_telemetry(self, telemetry: FileMetadata) -> None:
        """Load gyro data from parsed telemetry.

        Mirrors Gyroflow's GyroSource::load_from_telemetry:
        1. Clear existing data
        2. Copy IMU orientation from telemetry
        3. If quaternions present, use them directly (integration_method = 0)
        4. If raw IMU present, apply transforms then integrate
        5. Fall back to integration if neither has data
        """
        if self.duration_ms <= 0.0:
            log.error("Invalid duration_ms %s", self.duration_ms)
            return

        self.clear()

        self.imu_transforms.imu_orientation = telemetry.imu_orientation

        has_quats = bool(telemetry.quaternions)
        has_raw_imu = bool(telemetry.raw_imu)

        self.file_metadata = telemetry

        if has_quats:
            self.quaternions = dict(telemetry.quaternions)
            self.integration_method = 0

            # Recalculate duration from quaternion timestamps
            ts_keys = sorted(self.quaternions.keys())
            if ts_keys:
                n = len(ts_keys)
                first_ts = ts_keys[0] / 1000.0  # us -> ms
                last_ts = ts_keys[-1] / 1000.0
                imu_duration = (last_ts - first_ts) * ((n + 1) / n)
                if abs(imu_duration - self.duration_ms) > 0.01:
                    log.warning(
                        "IMU duration %.1f is different than video duration (%.1f)",
                        imu_duration, self.duration_ms,
                    )
                    if imu_duration > 0.0:
                        self.duration_ms = imu_duration

        if has_raw_imu:
            # Recalculate duration from raw IMU timestamps
            if self.file_metadata.raw_imu:
                n = len(self.file_metadata.raw_imu)
                first_ts = self.file_metadata.raw_imu[0].timestamp_ms
                last_ts = self.file_metadata.raw_imu[-1].timestamp_ms
                imu_duration = (last_ts - first_ts) * ((n + 1) / n)
                if abs(imu_duration - self.duration_ms) > 0.01:
                    log.warning(
                        "IMU duration %.1f is different than video duration (%.1f)",
                        imu_duration, self.duration_ms,
                    )
                    if imu_duration > 0.0:
                        self.duration_ms = imu_duration
            self.apply_transforms()
        elif not self.quaternions:
            self.integrate()

    def apply_transforms(self) -> None:
        """Apply IMU transforms to raw data and integrate.

        Mirrors Gyroflow's GyroSource::apply_transforms:
        1. If transforms are active, copy raw IMU and apply each transform
        2. Apply low-pass filter if enabled
        3. Apply median filter if enabled
        4. Integrate
        """
        if self.imu_transforms.has_any():
            self.raw_imu = []
            for sample in self.file_metadata.raw_imu:
                # Deep copy the sample
                gyro = sample.gyro.copy() if sample.gyro is not None else None
                accl = sample.accl.copy() if sample.accl is not None else None
                magn = sample.magn.copy() if sample.magn is not None else None

                if gyro is not None:
                    self.imu_transforms.transform(gyro, is_acc=False)
                if accl is not None:
                    self.imu_transforms.transform(accl, is_acc=True)
                if magn is not None:
                    self.imu_transforms.transform(magn, is_acc=False)

                self.raw_imu.append(TimeIMU(
                    timestamp_ms=sample.timestamp_ms,
                    gyro=gyro,
                    accl=accl,
                    magn=magn,
                ))

            # Low-pass filter
            if self.imu_transforms.imu_lpf > 0.0 and self.raw_imu and self.duration_ms > 0.0:
                sample_rate = len(self.raw_imu) / (self.duration_ms / 1000.0)
                self.raw_imu = lowpass_filter_imu(
                    self.raw_imu, self.imu_transforms.imu_lpf, sample_rate, forward_backward=True
                )

            # Median filter
            if self.imu_transforms.imu_mf > 0 and self.raw_imu and self.duration_ms > 0.0:
                sample_rate = len(self.raw_imu) / (self.duration_ms / 1000.0)
                self.raw_imu = median_filter_imu(
                    self.raw_imu, self.imu_transforms.imu_mf, forward_backward=True
                )
        else:
            self.raw_imu.clear()

        self.integrate()

    def integrate(self) -> None:
        """Integrate raw IMU data into quaternions using selected method.

        Mirrors Gyroflow's GyroSource::integrate dispatch.
        """
        from pygyroflow.imu_integration import (
            GyroIntegrator,
            ComplementaryIntegrator,
            VQFIntegrator,
            SimpleGyroIntegrator,
            SimpleGyroAccelIntegrator,
            MahonyIntegrator,
            MadgwickIntegrator,
        )

        # Determine which IMU data to use
        imu_data = self._get_imu_data()

        integrators: dict[int, type[GyroIntegrator]] = {
            1: ComplementaryIntegrator,
            2: VQFIntegrator,
            3: SimpleGyroIntegrator,
            4: SimpleGyroAccelIntegrator,
            5: MahonyIntegrator,
            6: MadgwickIntegrator,
        }

        if self.integration_method == 0:
            # Use pre-integrated quaternions from file metadata
            self.quaternions = dict(self.file_metadata.quaternions)

            # Apply low-pass filter to quaternions if enabled
            if self.imu_transforms.imu_lpf > 0.0 and self.quaternions and self.duration_ms > 0.0:
                sample_rate = len(self.quaternions) / (self.duration_ms / 1000.0)
                self.quaternions = self._lowpass_filter_quats(
                    self.quaternions, self.imu_transforms.imu_lpf, sample_rate
                )

            # Apply additional rotation to quaternions
            if self.imu_transforms._imu_rotation_matrix is not None:
                from scipy.spatial.transform import Rotation
                rot = Rotation.from_matrix(self.imu_transforms._imu_rotation_matrix)
                rot_quat = Quat64(rot)
                for ts in self.quaternions:
                    self.quaternions[ts] = rot_quat * self.quaternions[ts]

            return

        integrator_cls = integrators.get(self.integration_method)
        if integrator_cls is None:
            log.error("Unknown integration method: %s", self.integration_method)
            return

        integrator = integrator_cls()
        self.quaternions = integrator.integrate(imu_data, self.duration_ms)

    def _get_imu_data(self) -> list[TimeIMU]:
        """Get the best available IMU data (raw_imu overrides file_metadata)."""
        if self.raw_imu:
            return self.raw_imu
        return self.file_metadata.raw_imu

    def get_quat_at_timestamp(self, timestamp_ms: float, quats: TimeQuat | None = None) -> Quat64:
        """Get interpolated quaternion at given timestamp.

        Uses SLERP between bracketing quaternions with optional offset correction.

        Args:
            timestamp_ms: Timestamp in milliseconds.
            quats: Quaternion dict to search (defaults to self.quaternions).

        Returns:
            Interpolated quaternion, or identity if no data.
        """
        if quats is None:
            quats = self.quaternions

        if len(quats) < 2 or self.duration_ms <= 0.0:
            return Quat64.identity()

        # Apply offset correction
        timestamp_ms -= self.offset_at_video_timestamp(timestamp_ms)

        keys = sorted(quats.keys())
        first_ts = keys[0]
        last_ts = keys[-1]

        lookup_us = round(timestamp_ms * 1000.0)
        lookup_us = max(first_ts, min(last_ts, lookup_us))

        # Binary search for the right position
        idx = bisect.bisect_right(keys, lookup_us) - 1
        if idx < 0:
            idx = 0

        if keys[idx] == lookup_us:
            return quats[keys[idx]]

        if idx + 1 < len(keys):
            t0, t1 = keys[idx], keys[idx + 1]
            time_delta = t1 - t0
            if time_delta == 0:
                return quats[t0]
            fract = (lookup_us - t0) / time_delta
            return quats[t0].slerp(quats[t1], fract)

        return quats[keys[idx]]

    def org_quat_at_timestamp(self, timestamp_ms: float) -> Quat64:
        """Get original (unsmoothed) quaternion at timestamp."""
        return self.get_quat_at_timestamp(timestamp_ms, self.quaternions)

    def smoothed_quat_at_timestamp(self, timestamp_ms: float) -> Quat64:
        """Get smoothed quaternion at timestamp."""
        return self.get_quat_at_timestamp(timestamp_ms, self.smoothed_quaternions)

    # ------------------------------------------------------------------
    # Offset management
    # ------------------------------------------------------------------

    def set_use_gravity_vectors(self, v: bool) -> None:
        """Enable gravity-vector horizon mode (needs accelerometer data)."""
        self.use_gravity_vectors = bool(v)

    def set_offset(self, timestamp_us: int, offset_ms: float) -> None:
        """Set sync offset at a given timestamp."""
        if math.isfinite(offset_ms) and not math.isnan(offset_ms):
            self.offsets[timestamp_us] = offset_ms
            self._adjust_offsets()

    def remove_offset(self, timestamp_us: int) -> None:
        """Remove sync offset at a given timestamp."""
        self.offsets.pop(timestamp_us, None)
        self._adjust_offsets()

    def clear_offsets(self) -> None:
        """Clear all sync offsets."""
        self.offsets.clear()
        self.offsets_adjusted.clear()
        self.offsets_linear.clear()

    def get_offsets(self) -> dict[int, float]:
        """Get all sync offsets."""
        return dict(self.offsets)

    def set_offsets(self, offsets: dict[int, float]) -> None:
        """Replace all sync offsets."""
        self.offsets = dict(offsets)
        self._adjust_offsets()

    def offset_at_video_timestamp(self, timestamp_ms: float) -> float:
        """Get interpolated offset at a video timestamp."""
        return self.offset_at_timestamp(self.offsets_adjusted, timestamp_ms)

    def offset_at_gyro_timestamp(self, timestamp_ms: float) -> float:
        """Get interpolated offset at a gyro timestamp."""
        return self.offset_at_timestamp(self.offsets, timestamp_ms)

    @staticmethod
    def offset_at_timestamp(offsets: dict[int, float], timestamp_ms: float) -> float:
        """Interpolate offset at given timestamp.

        Shared by GyroSource lookups and the stabilization layer
        (frame_transform applies the same correction to quaternion
        lookups, mirroring Gyroflow's ``quat_at_timestamp``). The
        implementation lives in :mod:`pygyroflow.util` so the keyframe
        manager can use it too without a Layer-1 cross-dependency.
        """
        from pygyroflow.util import offset_at_timestamp as _offset_at_timestamp

        return _offset_at_timestamp(offsets, timestamp_ms)

    @staticmethod
    def _line_fit(offsets: dict[int, float]) -> tuple[float, float, float] | None:
        """Least-squares line over the offset map, as residual sum too
        (``mod.rs:683-698``): ``[slope, intercept, residual_ss]``."""
        keys = sorted(offsets)
        if len(keys) < 2:
            return None
        a = np.column_stack([np.asarray(keys, dtype=np.float64),
                             np.ones(len(keys))])
        b = np.asarray([offsets[k] for k in keys], dtype=np.float64)
        try:
            solution, *_ = np.linalg.lstsq(a, b, rcond=None)
        except np.linalg.LinAlgError:
            return None
        residuals = float(np.sum((a @ solution - b) ** 2))
        return float(solution[0]), float(solution[1]), residuals

    def _adjust_offsets(self) -> None:
        """Recalculate linear fit and adjusted offsets.

        Port of ``adjust_offsets`` (``mod.rs:700-776``): pairwise-slope
        consensus with a 5 ms inlier band, not a plain least squares over
        everything — one bad sync point must not drag the drift curve.
        The best consensus wins on inlier count first (among >2-point
        near-constant solutions, lowest residual), and the fitted line
        then spans **all** keys, extrapolating over the rejected outliers.
        """
        if len(self.offsets) <= 1:
            self.offsets_linear = dict(self.offsets)
        else:
            keys = sorted(self.offsets)
            best_offsets: dict[int, float] = {}
            best_rsquared = 1000.0
            best_coeffs = (0.0, 0.0)
            max_fitting_error = 5.0  # ms

            for i in keys:
                for j in keys:
                    if i == j:
                        continue
                    slope = (self.offsets[j] - self.offsets[i]) / (j - i)
                    intersect = self.offsets[i] - i * slope
                    within = {
                        k: v for k, v in self.offsets.items()
                        if abs((k * slope + intersect) - v) < max_fitting_error
                    }
                    if len(within) >= len(best_offsets) and within != best_offsets:
                        solution = self._line_fit(within)
                        if solution is None:
                            continue
                        close_constant = abs(solution[0]) < 0.1
                        if len(within) > 2 and close_constant:
                            if solution[2] < best_rsquared:
                                best_rsquared = solution[2]
                                best_offsets = dict(within)
                                best_coeffs = (solution[0], solution[1])
                        elif close_constant:
                            best_offsets = dict(within)
                            best_coeffs = (solution[0], solution[1])

            if best_offsets:
                self.offsets_linear = {
                    k: k * best_coeffs[0] + best_coeffs[1]
                    for k in self.offsets
                }
            else:
                solution = self._line_fit(self.offsets)
                if solution is not None:
                    self.offsets_linear = {
                        k: k * solution[0] + solution[1]
                        for k in self.offsets
                    }
                else:
                    self.offsets_linear = dict(self.offsets)

        # Build adjusted offsets: key = timestamp + offset in us
        self.offsets_adjusted = {
            k + round(v * 1000.0): v for k, v in self.offsets.items()
        }

    # ------------------------------------------------------------------
    # Checksum and sample rate
    # ------------------------------------------------------------------

    def get_checksum(self) -> int:
        """Compute a hash of all gyro data for cache invalidation.

        Covers upstream's whole field set (gyro_source/mod.rs::get_checksum),
        not just the source and the two lengths: the transform angles, the
        gyro bias, the integration method, the sync offsets and the endpoints
        of the quaternion stream all change the pipeline's output, so leaving
        them out let a stale cache survive a parameter change.
        """
        import hashlib
        import struct as _struct

        hasher = hashlib.sha256()

        def _text(value: str | None) -> None:
            if value:
                hasher.update(value.encode())

        def _f64(value: float) -> None:
            hasher.update(_struct.pack('d', float(value)))

        t = self.imu_transforms
        _text(self.file_metadata.detected_source)
        _text(t.imu_orientation)
        for angles in (t.imu_rotation_angles, t.acc_rotation_angles, t.gyro_bias):
            if angles is not None:
                for component in angles:
                    _f64(component)
        _text(self.file_url)
        _f64(self.duration_ms)
        _f64(t.imu_lpf)
        hasher.update(_struct.pack('i', int(t.imu_mf)))

        io_stream = self.file_metadata.image_orientations
        for count in (
            len(self.raw_imu),
            len(self.file_metadata.raw_imu),
            len(self.quaternions),
            len(self.file_metadata.quaternions),
            len(io_stream) if io_stream else 0,
            len(self.file_metadata.lens_positions),
            len(self.file_metadata.lens_params),
        ):
            hasher.update(_struct.pack('Q', count))
        hasher.update(_struct.pack('I', 1 if self.use_gravity_vectors else 0))
        hasher.update(_struct.pack('Q', int(self.integration_method)))

        for ts in sorted(self.offsets):
            hasher.update(_struct.pack('q', int(ts)))
            _f64(self.offsets[ts])

        for ts in (min(self.quaternions), max(self.quaternions)) if self.quaternions else ():
            q = self.quaternions[ts].quaternion()
            hasher.update(_struct.pack('q', int(ts)))
            for component in q:
                _f64(component)

        return int(hasher.hexdigest()[:16], 16)

    @staticmethod
    def get_sample_rate(file_metadata: FileMetadata) -> float:
        """Estimate IMU sample rate from metadata."""
        if len(file_metadata.raw_imu) > 2:
            n = len(file_metadata.raw_imu)
            duration_ms = file_metadata.raw_imu[-1].timestamp_ms - file_metadata.raw_imu[0].timestamp_ms
            duration_ms *= (n + 1) / n
            return n / (duration_ms / 1000.0) if duration_ms > 0 else 0.0

        if len(file_metadata.quaternions) > 2:
            keys = sorted(file_metadata.quaternions.keys())
            n = len(keys)
            first_ms = keys[0] / 1000.0
            last_ms = keys[-1] / 1000.0
            duration_ms = (last_ms - first_ms) * (n + 1) / n
            return n / (duration_ms / 1000.0) if duration_ms > 0 else 0.0

        return 0.0

    def find_bias(self, timestamp_start: float, timestamp_stop: float) -> tuple[float, float, float]:
        """Find average gyro bias in a time window.

        The window arrives on the *video* timeline; the raw IMU lives on
        the gyro one, so both ends shift by the sync offset at that
        instant (``mod.rs:933-935``) — without the shift a synced clip
        averages the wrong stretch of samples.
        """
        ts_start = timestamp_start - self.offset_at_video_timestamp(
            timestamp_start
        )
        ts_stop = timestamp_stop - self.offset_at_video_timestamp(
            timestamp_stop
        )
        bias_vals = [0.0, 0.0, 0.0]
        n = 0

        for sample in self.file_metadata.raw_imu:
            if sample.gyro is not None and ts_start < sample.timestamp_ms < ts_stop:
                bias_vals[0] -= sample.gyro[0]
                bias_vals[1] -= sample.gyro[1]
                bias_vals[2] -= sample.gyro[2]
                n += 1

        if n > 0:
            bias_vals = [b / n for b in bias_vals]

        return (bias_vals[0], bias_vals[1], bias_vals[2])

    @staticmethod
    def _lowpass_filter_quats(quats: TimeQuat, cutoff_freq: float, sample_rate: float) -> TimeQuat:
        """Apply lowpass filter to quaternion components."""
        from pygyroflow.filtering import lowpass_filter

        keys = sorted(quats.keys())
        if len(keys) < 4:
            return quats

        # Extract quaternion components
        wxyz = np.array([quats[k].quaternion() for k in keys])  # shape (N, 4)

        # Filter each component independently
        for ch in range(4):
            wxyz[:, ch] = lowpass_filter(wxyz[:, ch], cutoff_freq, sample_rate, forward_backward=True)

        # Re-normalize and rebuild dict
        result: TimeQuat = {}
        for i, k in enumerate(keys):
            result[k] = Quat64.from_quaternion(wxyz[i])

        return result
