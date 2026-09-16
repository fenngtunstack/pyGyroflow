"""Butterworth lowpass filter for gyroscope and IMU data.

Port of Gyroflow's src/core/filtering.rs Lowpass struct.
Uses scipy.signal for Butterworth filter design and SOS (second-order sections)
application, which matches the biquad DirectForm2Transposed approach in Rust.
"""

import numpy as np
from scipy import signal

from pygyroflow.filtering.imu_channels import filter_imu_channels
from pygyroflow.types.time_types import TimeIMU


def lowpass_filter(
    data: np.ndarray,
    cutoff_freq: float,
    sample_rate: float,
    forward_backward: bool = True,
) -> np.ndarray:
    """Apply 2nd-order Butterworth lowpass filter.

    Mirrors the Rust Lowpass which uses biquad Coefficients with Q_BUTTERWORTH
    and DirectForm2Transposed state. The scipy SOS path is numerically equivalent.

    Args:
        data: Input signal (1D array).
        cutoff_freq: Cutoff frequency in Hz.
        sample_rate: Sample rate in Hz.
        forward_backward: If True, apply zero-phase forward-backward filtering
            (equivalent to filter_gyro_forward_backward in Rust). If False,
            single-pass forward-only (equivalent to filter_gyro).

    Returns:
        Filtered signal, same shape as input.
    """
    if cutoff_freq <= 0 or cutoff_freq >= sample_rate / 2:
        return data.copy() if isinstance(data, np.ndarray) else np.array(data)

    # 2nd-order Butterworth, SOS output for numerical stability
    sos = signal.butter(2, cutoff_freq, btype="low", fs=sample_rate, output="sos")

    if forward_backward:
        # Zero-phase: forward then backward, same as Rust's filter_gyro_forward_backward
        return signal.sosfiltfilt(sos, data)

    return signal.sosfilt(sos, data)


def lowpass_filter_channels(
    data: np.ndarray,
    cutoff_freq: float,
    sample_rate: float,
    forward_backward: bool = True,
) -> np.ndarray:
    """Apply lowpass filter independently to each channel (row) of a 2D array.

    Used for multi-channel IMU data where each row is one axis.

    Args:
        data: Input signal (2D array, shape [n_channels, n_samples]).
        cutoff_freq: Cutoff frequency in Hz.
        sample_rate: Sample rate in Hz.
        forward_backward: If True, zero-phase forward-backward filtering.

    Returns:
        Filtered signal, same shape as input.
    """
    if cutoff_freq <= 0 or cutoff_freq >= sample_rate / 2:
        return data.copy()

    out = np.empty_like(data)
    for ch in range(data.shape[0]):
        out[ch] = lowpass_filter(data[ch], cutoff_freq, sample_rate, forward_backward)
    return out


def lowpass_filter_imu(
    imu_data: list[TimeIMU],
    cutoff_freq: float,
    sample_rate: float,
    forward_backward: bool = True,
) -> list[TimeIMU]:
    """Apply a Butterworth lowpass to the gyro and accel axes of IMU samples.

    Mirrors upstream's ``IMUTransforms`` lowpass step: the Rust implementation
    runs 6 independent biquads (3 gyro + 3 accel). Each axis is lifted out as
    an array, filtered, and written back.

    Args:
        imu_data: IMU samples (``TimeIMU`` objects).
        cutoff_freq: Cutoff frequency in Hz.
        sample_rate: Sample rate in Hz.
        forward_backward: If True, zero-phase forward-backward filtering.

    Returns:
        New list of ``TimeIMU`` samples with filtered gyro/accl.
    """
    if cutoff_freq <= 0 or cutoff_freq >= sample_rate / 2:
        return imu_data

    return filter_imu_channels(
        imu_data,
        lambda arr: lowpass_filter_channels(arr, cutoff_freq, sample_rate, forward_backward),
    )
