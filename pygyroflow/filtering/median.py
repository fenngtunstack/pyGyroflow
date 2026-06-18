"""Median filter for gyroscope and IMU data.

Port of Gyroflow's src/core/filtering.rs Median struct.
Uses scipy.signal.medfilt which applies a sliding-window median.
"""

import numpy as np
from scipy.signal import medfilt


def median_filter(
    data: np.ndarray,
    kernel_size: int = 3,
    forward_backward: bool = True,
) -> np.ndarray:
    """Apply median filter.

    The Rust implementation uses a streaming median filter (median crate) applied
    forward then backward when forward_backward is True. We replicate this by
    running medfilt twice (forward, then backward on the forward result).

    Args:
        data: Input signal (1D array).
        kernel_size: Size of the median window. Must be odd and >= 1.
        forward_backward: If True, apply forward-backward (double pass) to
            reduce phase distortion.

    Returns:
        Filtered signal, same shape as input.
    """
    if kernel_size < 1:
        return data.copy() if isinstance(data, np.ndarray) else np.array(data)

    # Ensure odd kernel size (scipy requirement)
    ks = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    if forward_backward:
        # Double pass: forward then backward, matching Rust's filter_gyro_forward_backward
        return medfilt(medfilt(data, ks), ks)

    return medfilt(data, ks)


def median_filter_channels(
    data: np.ndarray,
    kernel_size: int = 3,
    forward_backward: bool = True,
) -> np.ndarray:
    """Apply median filter independently to each channel of a 2D array.

    Args:
        data: Input signal (2D array, shape [n_channels, n_samples]).
        kernel_size: Size of the median window.
        forward_backward: If True, double-pass median filter.

    Returns:
        Filtered signal, same shape as input.
    """
    out = np.empty_like(data)
    for ch in range(data.shape[0]):
        out[ch] = median_filter(data[ch], kernel_size, forward_backward)
    return out


def median_filter_imu(
    imu_data: list,
    kernel_size: int = 3,
    forward_backward: bool = True,
) -> list:
    """Apply median filter to IMU data channels.

    Each element of imu_data is a dict-like with optional 'gyro' and 'accl'
    fields, matching the TimeIMU structure.

    Args:
        imu_data: List of IMU samples.
        kernel_size: Size of the median window.
        forward_backward: If True, double-pass median filter.

    Returns:
        New list with filtered IMU data.
    """
    n = len(imu_data)

    has_gyro = any(sample.get("gyro") is not None for sample in imu_data)
    has_accl = any(sample.get("accl") is not None for sample in imu_data)

    if has_gyro:
        gyro_arr = np.array(
            [sample.get("gyro", [0.0, 0.0, 0.0]) for sample in imu_data]
        ).T
        gyro_filtered = median_filter_channels(gyro_arr, kernel_size, forward_backward)
    else:
        gyro_filtered = None

    if has_accl:
        accl_arr = np.array(
            [sample.get("accl", [0.0, 0.0, 0.0]) for sample in imu_data]
        ).T
        accl_filtered = median_filter_channels(accl_arr, kernel_size, forward_backward)
    else:
        accl_filtered = None

    result = []
    for i, sample in enumerate(imu_data):
        new_sample = dict(sample)
        if gyro_filtered is not None and sample.get("gyro") is not None:
            new_sample["gyro"] = [float(gyro_filtered[axis, i]) for axis in range(3)]
        if accl_filtered is not None and sample.get("accl") is not None:
            new_sample["accl"] = [float(accl_filtered[axis, i]) for axis in range(3)]
        result.append(new_sample)

    return result
