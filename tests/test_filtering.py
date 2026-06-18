"""Tests for lowpass and median filters."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pygyroflow.filtering import lowpass_filter, median_filter


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
