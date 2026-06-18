"""OptimSync -- frequency-domain analysis for optimal sync-point selection.

Port of Gyroflow's ``OptimSync`` module.  The goal is to choose video
segments that contain the richest rotational motion for synchronization.

Algorithm
---------
1.  Resample gyroscope data to a uniform rate.
2.  Compute short-time FFT (STFT) with a Blackman window.
3.  Band-split into low (0-2 Hz), mid (2-30 Hz), and high (30+ Hz) energy.
4.  Score each time bin: high mid-frequency energy is rewarded (hand-shake
    is the best sync signal), while high low/high energy is penalised.
5.  Apply non-maximum suppression (NMS) so sync points are at least
    ``nms_gap_s`` seconds apart.
6.  Divide the timeline into equal segments and pick the highest-scoring
    bin in each segment.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Frequency band boundaries (Hz)
_LOW_BAND = (0.0, 2.0)
_MID_BAND = (2.0, 30.0)
_HIGH_BAND = (30.0, 2000.0)

# Scoring penalty thresholds
_HIGH_PENALTY_TRIP = 450.0
_LOW_PENALTY_TRIP = 650.0
_MIN_RANK = 50.0

# Minimum sync-point score to be considered valid
_MIN_SCORE = 0.1


def _blackman(width: int) -> np.ndarray:
    """Generate a Blackman window of given length."""
    a0 = 7938.0 / 18608.0
    a1 = 9240.0 / 18608.0
    a2 = 1430.0 / 18608.0
    n = np.arange(width, dtype=np.float32)
    N = width - 1
    return a0 - a1 * np.cos(2.0 * np.pi * n / N) + a2 * np.cos(4.0 * np.pi * n / N)


def _nlfunc(arg: float, trip_point: float) -> float:
    """Dead-zone penalty: returns 0 below *trip_point*, else arg - trip_point."""
    return max(0.0, arg - trip_point)


class OptimSync:
    """Frequency-domain optimal sync-point selector.

    Parameters
    ----------
    gyro_timestamps_ms:
        1-D array of gyroscope timestamps in milliseconds.
    gyro_data:
        (N, 3) array of angular velocities in degrees/second.
    """

    def __init__(
        self,
        gyro_timestamps_ms: np.ndarray,
        gyro_data: np.ndarray,
    ) -> None:
        ts = np.asarray(gyro_timestamps_ms, dtype=np.float64)
        data = np.asarray(gyro_data, dtype=np.float64)
        if data.ndim == 1:
            data = data[:, np.newaxis]

        self._duration_ms = ts[-1] - ts[0]
        self._sample_rate = len(ts) / (self._duration_ms / 1000.0)

        # Resample to uniform rate via linear interpolation
        n_samples = int(self._duration_ms * self._sample_rate / 1000.0)
        uniform_ts = np.linspace(ts[0], ts[-1], n_samples)
        self._gyro = np.column_stack(
            [np.interp(uniform_ts, ts, data[:, ax]) for ax in range(data.shape[1])]
        )

    def run(
        self,
        target_sync_points: int = 3,
        trim_ranges_s: list[tuple[float, float]] | None = None,
        nms_gap_s: float = 8.0,
    ) -> tuple[list[float], np.ndarray, float]:
        """Select optimal sync points.

        Parameters
        ----------
        target_sync_points:
            Desired number of sync points.
        trim_ranges_s:
            Valid time ranges in seconds.  Only points inside these ranges
            are candidates.  ``None`` means the entire duration.
        nms_gap_s:
            Minimum gap between sync points in seconds.

        Returns
        -------
        (sync_points_ms, rank_curve, time_step_s)
        * sync_points_ms: selected point timestamps in milliseconds.
        * rank_curve: per-bin scoring curve (for visualisation).
        * time_step_s: time resolution of the rank curve.
        """
        if trim_ranges_s is None:
            trim_ranges_s = [(0.0, self._duration_ms / 1000.0)]

        sr = self._sample_rate
        step_size = 16
        fft_size = max(int(round(sr)), 64)
        scale = np.sqrt(1.0 / fft_size) / fft_size * 256.0
        win = _blackman(fft_size)

        # STFT per axis
        stft_axes: list[list[np.ndarray]] = []
        for ax in range(self._gyro.shape[1]):
            signal = self._gyro[:, ax].astype(np.float32)
            axis_ffts: list[np.ndarray] = []
            for start in range(0, len(signal) - fft_size + 1, step_size):
                chunk = signal[start : start + fft_size] * win
                spectrum = np.fft.rfft(chunk)
                magnitude = np.abs(spectrum) * scale
                axis_ffts.append(magnitude)
            stft_axes.append(axis_ffts)

        if not stft_axes or not stft_axes[0]:
            return [], np.array([]), 0.0

        n_bins = len(stft_axes[0][0])
        n_windows = len(stft_axes[0])

        # Merge axes by summation
        merged = np.zeros((n_windows, n_bins), dtype=np.float32)
        for ax_ffts in stft_axes:
            for i, mag in enumerate(ax_ffts):
                merged[i, : len(mag)] += mag

        # Map frequency to bin index
        def freq_to_bin(freq: float) -> int:
            return min(
                max(int(round(fft_size / sr * freq)), 0), n_bins - 1
            )

        # Band energy
        def band_energy(lo: float, hi: float) -> np.ndarray:
            b0 = freq_to_bin(lo)
            b1 = freq_to_bin(hi)
            if b1 <= b0:
                return np.zeros(n_windows, dtype=np.float32)
            return merged[:, b0:b1].sum(axis=1)

        lf = band_energy(*_LOW_BAND)
        mf = band_energy(*_MID_BAND)
        hf = band_energy(*_HIGH_BAND)

        # Score
        rank = np.array(
            [
                m / (1.0 + _nlfunc(h, _HIGH_PENALTY_TRIP) * 0.003)
                    / (1.0 + _nlfunc(l, _LOW_PENALTY_TRIP) * 0.003)
                for l, m, h in zip(lf, mf, hf)
            ],
            dtype=np.float32,
        )

        ratio = step_size / sr  # seconds per rank bin
        total_duration = n_windows * ratio

        # Zero out low-rank bins and bins outside trim ranges
        for i in range(n_windows):
            t = i * ratio
            if rank[i] < _MIN_RANK:
                rank[i] = 0.0
            if not any(lo <= t <= hi for lo, hi in trim_ranges_s):
                rank[i] = 0.0

        # Exclude first/last 2 seconds if long enough
        if total_duration > 12.0:
            for i in range(n_windows):
                t = i * ratio
                if t < 2.0 or t >= total_duration - 2.0:
                    rank[i] = 0.0

        # NMS
        nms_radius = int(sr / step_size / 2.0 * nms_gap_s)
        rank_nms = rank.copy()
        for i in range(n_windows):
            for j in range(max(0, i - nms_radius), min(n_windows, i + nms_radius + 1)):
                if rank[j] < rank[i]:
                    rank_nms[j] = 0.0

        # Select top bins per segment
        segment_size = (n_windows + target_sync_points - 1) // target_sync_points
        selected: list[float] = []
        for seg in range(target_sync_points):
            start = seg * segment_size
            end = min(start + segment_size, n_windows)
            if start >= n_windows:
                break
            segment = rank_nms[start:end]
            if len(segment) == 0:
                continue
            best_idx = int(np.argmax(segment))
            if segment[best_idx] < _MIN_SCORE:
                continue
            absolute_idx = start + best_idx
            time_ms = (
                (absolute_idx * step_size + fft_size / 2.0) / sr * 1000.0
            )
            selected.append(time_ms)

        return selected, rank, ratio
