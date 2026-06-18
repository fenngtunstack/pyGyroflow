"""Butterworth lowpass filter for gyroscope and IMU data.

Port of Gyroflow's src/core/filtering.rs Lowpass struct.
Uses scipy.signal for Butterworth filter design and SOS (second-order sections)
application, which matches the biquad DirectForm2Transposed approach in Rust.
"""

import numpy as np
from scipy import signal


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
    imu_data: list,
    cutoff_freq: float,
    sample_rate: float,
    forward_backward: bool = True,
) -> list:
    """Apply lowpass filter to IMU data channels.

    Each element of imu_data is expected to be a dict-like with optional
    'gyro' ([x, y, z]) and 'accl' ([x, y, z]) fields, mirroring the Rust
    TimeIMU struct.

    The Rust implementation runs 6 independent biquad filters (3 gyro + 3 accl).
    We achieve the same by extracting each axis as an array, filtering, and
    writing back.

    Args:
        imu_data: List of IMU samples. Each sample is a dict with optional
            'gyro' (list/array of 3) and 'accl' (list/array of 3).
        cutoff_freq: Cutoff frequency in Hz.
        sample_rate: Sample rate in Hz.
        forward_backward: If True, zero-phase forward-backward filtering.

    Returns:
        New list with filtered IMU data (same structure, new objects).
    """
    if cutoff_freq <= 0 or cutoff_freq >= sample_rate / 2:
        return imu_data

    n = len(imu_data)

    # Collect gyro and accl arrays
    has_gyro = any(sample.get("gyro") is not None for sample in imu_data)
    has_accl = any(sample.get("accl") is not None for sample in imu_data)

    if has_gyro:
        gyro_arr = np.array(
            [sample.get("gyro", [0.0, 0.0, 0.0]) for sample in imu_data]
        ).T  # shape (3, n)
        gyro_filtered = lowpass_filter_channels(
            gyro_arr, cutoff_freq, sample_rate, forward_backward
        )
    else:
        gyro_filtered = None

    if has_accl:
        accl_arr = np.array(
            [sample.get("accl", [0.0, 0.0, 0.0]) for sample in imu_data]
        ).T  # shape (3, n)
        accl_filtered = lowpass_filter_channels(
            accl_arr, cutoff_freq, sample_rate, forward_backward
        )
    else:
        accl_filtered = None

    # Write back
    result = []
    for i, sample in enumerate(imu_data):
        new_sample = dict(sample)
        if gyro_filtered is not None and sample.get("gyro") is not None:
            new_sample["gyro"] = [float(gyro_filtered[axis, i]) for axis in range(3)]
        if accl_filtered is not None and sample.get("accl") is not None:
            new_sample["accl"] = [float(accl_filtered[axis, i]) for axis in range(3)]
        result.append(new_sample)

    return result
