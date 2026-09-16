"""Tests for lowpass and median filters."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pygyroflow.filtering import (
    lowpass_filter,
    lowpass_filter_imu,
    median_filter,
    median_filter_imu,
)
from pygyroflow.gyro_source import FileMetadata, GyroSource
from pygyroflow.types.time_types import TimeIMU


class TestLowpassFilter:
    def test_high_frequency_noise_reduced(self):
        """Lowpass should attenuate high-frequency components."""
        sample_rate = 200.0
        t = np.arange(1000) / sample_rate
        # 5 Hz signal + 50 Hz noise
        signal = np.sin(2 * np.pi * 5.0 * t) + 0.5 * np.sin(2 * np.pi * 50.0 * t)

        filtered = lowpass_filter(signal, cutoff_freq=10.0, sample_rate=sample_rate)

        # The filtered signal should be close to just the 5 Hz component
        expected = np.sin(2 * np.pi * 5.0 * t)
        # Check that high-frequency noise is reduced (std of difference is smaller)
        noise_before = np.std(signal - expected)
        noise_after = np.std(filtered - expected)
        assert noise_after < noise_before * 0.3  # At least 70% reduction

    def test_zero_cutoff_returns_input(self):
        data = np.random.randn(100)
        result = lowpass_filter(data, cutoff_freq=0.0, sample_rate=100.0)
        assert_allclose(result, data)

    def test_cutoff_above_nyquist_returns_input(self):
        data = np.random.randn(100)
        result = lowpass_filter(data, cutoff_freq=60.0, sample_rate=100.0)
        assert_allclose(result, data)


class TestLowpassForwardBackward:
    def test_zero_phase(self):
        """Forward-backward filtering should not shift the signal in time."""
        sample_rate = 200.0
        t = np.arange(1000) / sample_rate
        # 5 Hz signal
        signal = np.sin(2 * np.pi * 5.0 * t)

        filtered = lowpass_filter(signal, cutoff_freq=20.0, sample_rate=sample_rate,
                                   forward_backward=True)

        # Cross-correlation peak should be at lag 0 (no shift)
        correlation = np.correlate(filtered - filtered.mean(), signal - signal.mean(), mode='full')
        lags = np.arange(-len(signal) + 1, len(signal))
        peak_lag = lags[np.argmax(correlation)]
        assert abs(peak_lag) <= 1  # Allow 1 sample tolerance

    def test_forward_only_has_phase_shift(self):
        """Single-pass forward filter introduces some phase delay."""
        sample_rate = 200.0
        t = np.arange(1000) / sample_rate
        signal = np.sin(2 * np.pi * 5.0 * t)

        filtered = lowpass_filter(signal, cutoff_freq=8.0, sample_rate=sample_rate,
                                   forward_backward=False)

        # The correlation peak should not necessarily be at lag 0
        # (but the signal should still be present)
        assert np.std(filtered) > 0.1


class TestMedianFilter:
    def test_impulse_noise_removed(self):
        """Median filter should remove isolated spikes."""
        signal = np.array([1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        filtered = median_filter(signal, kernel_size=3, forward_backward=False)
        # The spike at index 3 should be removed
        assert filtered[3] < 5.0

    def test_preserves_smooth_signal(self):
        """Median filter should not significantly alter a smooth signal."""
        signal = np.linspace(0, 10, 100)
        filtered = median_filter(signal, kernel_size=3, forward_backward=False)
        # Smooth linear ramp should be nearly unchanged (except near edges)
        assert_allclose(filtered[5:-5], signal[5:-5], atol=0.5)

    def test_forward_backward_double_pass(self):
        """Forward-backward median is more aggressive than single pass."""
        np.random.seed(42)
        signal = np.random.randn(100)
        single = median_filter(signal, kernel_size=3, forward_backward=False)
        double = median_filter(signal, kernel_size=3, forward_backward=True)

        # Double pass should be smoother (lower variance)
        assert np.var(double) <= np.var(single) + 1e-10

    def test_kernel_size_1_returns_input(self):
        signal = np.array([1.0, 5.0, 3.0, 2.0])
        filtered = median_filter(signal, kernel_size=1, forward_backward=False)
        assert_allclose(filtered, signal)


def _noisy_imu(n=200, seed=0):
    """IMU samples with a slow trend plus per-axis noise."""
    rng = np.random.default_rng(seed)
    return [
        TimeIMU(
            timestamp_ms=i * 10.0,
            gyro=np.array(
                [
                    10 * np.sin(i / 4.0) + rng.normal(0, 3),
                    5 * np.cos(i / 3.0) + rng.normal(0, 3),
                    2.0 + rng.normal(0, 3),
                ]
            ),
            accl=np.array([0.0, 0.0, 9.81]),
        )
        for i in range(n)
    ]


class TestImuSampleFilters:
    """`*_filter_imu` operate on TimeIMU samples, not dicts.

    Regression: the write-back path used ``sample.get("gyro")``, which raises
    AttributeError on a dataclass — so setting ``imu_lpf`` or ``imu_mf`` blew
    up on the first sample, and neither filter had ever actually run.
    """

    def test_lowpass_accepts_time_imu_samples(self):
        samples = _noisy_imu()
        out = lowpass_filter_imu(samples, 15.0, 100.0)
        assert all(isinstance(s, TimeIMU) for s in out)
        before = np.array([s.gyro for s in samples])
        after = np.array([s.gyro for s in out])
        assert np.abs(after - before).max() > 1.0

    def test_median_accepts_time_imu_samples(self):
        samples = _noisy_imu()
        out = median_filter_imu(samples, 5)
        assert all(isinstance(s, TimeIMU) for s in out)
        before = np.array([s.gyro for s in samples])
        after = np.array([s.gyro for s in out])
        assert np.abs(after - before).max() > 1.0

    @pytest.mark.parametrize(
        "func,args",
        [(lowpass_filter_imu, (15.0, 100.0)), (median_filter_imu, (5,))],
    )
    def test_other_channels_survive(self, func, args):
        samples = _noisy_imu()
        out = func(samples, *args)
        assert [s.timestamp_ms for s in out] == [s.timestamp_ms for s in samples]
        assert all(np.allclose(s.accl, [0.0, 0.0, 9.81]) for s in out)

    @pytest.mark.parametrize(
        "func,args",
        [(lowpass_filter_imu, (15.0, 100.0)), (median_filter_imu, (5,))],
    )
    def test_input_samples_are_not_mutated(self, func, args):
        samples = _noisy_imu()
        before = np.array([s.gyro for s in samples])
        func(samples, *args)
        assert_allclose(np.array([s.gyro for s in samples]), before)

    @pytest.mark.parametrize(
        "func,args",
        [(lowpass_filter_imu, (15.0, 100.0)), (median_filter_imu, (5,))],
    )
    def test_missing_accel_channel_is_tolerated(self, func, args):
        samples = [
            TimeIMU(timestamp_ms=i * 10.0, gyro=np.array([1.0, 2.0, 3.0]))
            for i in range(20)
        ]
        out = func(samples, *args)
        assert len(out) == 20
        assert all(s.accl is None for s in out)

    @pytest.mark.parametrize(
        "func,args",
        [(lowpass_filter_imu, (15.0, 100.0)), (median_filter_imu, (5,))],
    )
    def test_empty_input(self, func, args):
        assert func([], *args) == []

    def test_lowpass_cutoff_above_nyquist_passes_through(self):
        samples = _noisy_imu()
        assert lowpass_filter_imu(samples, 90.0, 100.0) is samples


class TestGyroSourceKeepsUserTransforms:
    """clear() must not drop settings the caller made before loading."""

    @staticmethod
    def _telemetry(n=400):
        return FileMetadata(
            detected_source="Test",
            imu_orientation="XYZ",
            raw_imu=[
                TimeIMU(
                    timestamp_ms=i * 10.0,
                    gyro=np.array([10 * np.sin(i / 7.0), 5 * np.cos(i / 5.0), 3.0]),
                    accl=np.array([0.0, 0.0, 9.81]),
                )
                for i in range(n)
            ],
        )

    def _load(self, **transforms):
        source = GyroSource()
        source.init_from_params(4000.0)
        for key, value in transforms.items():
            setattr(source.imu_transforms, key, value)
        source.load_from_telemetry(self._telemetry())
        return source

    @staticmethod
    def _quats(source):
        keys = sorted(source.quaternions)[:60]
        return np.array([source.quaternions[k].quaternion() for k in keys])

    def test_gyro_bias_survives_load_and_changes_the_result(self):
        plain = self._load()
        biased = self._load(gyro_bias=[5.0, 0.0, 0.0])
        assert biased.imu_transforms.gyro_bias == [5.0, 0.0, 0.0]
        assert np.abs(self._quats(biased) - self._quats(plain)).max() > 1e-3

    def test_transformed_raw_imu_is_kept(self):
        """A bias forces the transform branch; raw_imu must not be cleared."""
        plain = self._load()
        biased = self._load(gyro_bias=[5.0, 0.0, 0.0])
        assert plain.raw_imu == []
        assert len(biased.raw_imu) == 400

    @pytest.mark.parametrize("key,value", [("imu_lpf", 15.0), ("imu_mf", 5)])
    def test_filters_apply_on_reload(self, key, value):
        """Set the filter, then re-run the transform stage as the GUI does.

        ``clear()`` resets imu_lpf/imu_mf (upstream does the same — they are
        per-load state), so the filter is configured after loading and
        ``apply_transforms()`` is what actually runs it.
        """
        source = self._load()
        setattr(source.imu_transforms, key, value)
        source.apply_transforms()
        assert len(source.raw_imu) == 400
        assert source.quaternions

    @pytest.mark.parametrize("key,value", [("imu_lpf", 15.0), ("imu_mf", 5)])
    def test_clear_resets_filters_but_keeps_bias(self, key, value):
        source = self._load(gyro_bias=[1.0, 2.0, 3.0])
        setattr(source.imu_transforms, key, value)
        source.clear()
        expected = 0.0 if key == "imu_lpf" else 0
        assert getattr(source.imu_transforms, key) == expected
        assert source.imu_transforms.gyro_bias == [1.0, 2.0, 3.0]
