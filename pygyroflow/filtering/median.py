"""Median filter for gyroscope and IMU data.

Port of Gyroflow's src/core/filtering.rs Median struct.
Uses scipy.signal.medfilt which applies a sliding-window median.
"""

import numpy as np
from scipy.signal import medfilt

from pygyroflow.filtering.imu_channels import filter_imu_channels
from pygyroflow.types.time_types import TimeIMU


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
    imu_data: list[TimeIMU],
    kernel_size: int = 3,
    forward_backward: bool = True,
) -> list[TimeIMU]:
    """Apply a median filter to the gyro and accel axes of IMU samples.

    Args:
        imu_data: IMU samples (``TimeIMU`` objects).
        kernel_size: Size of the median window.
        forward_backward: If True, double-pass median filter.

    Returns:
        New list of ``TimeIMU`` samples with filtered gyro/accl.
    """
    return filter_imu_channels(
        imu_data,
        lambda arr: median_filter_channels(arr, kernel_size, forward_backward),
    )
