"""Channel-wise plumbing shared by the IMU-level filters.

``lowpass_filter_imu`` and ``median_filter_imu`` both work the same way:
lift the gyro and accel axes into a (3, n) matrix, run the per-channel
filter, then write the result back into copies of the samples. Only this
wrapper is shared — the filters themselves stay in their own modules.

The samples are ``TimeIMU`` dataclasses, not dicts: reading them with
``.get()`` (as this plumbing once did) raises ``AttributeError`` on the
first sample, which meant ``imu_lpf``/``imu_mf`` never filtered anything.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np

from pygyroflow.types.time_types import TimeIMU

_ZERO = (0.0, 0.0, 0.0)


def _channel_matrix(samples: list[TimeIMU], attr: str) -> np.ndarray:
    """(3, n) matrix of one attribute, zero-filled where a sample is empty."""
    return np.array(
        [getattr(s, attr) if getattr(s, attr) is not None else _ZERO for s in samples],
        dtype=np.float64,
    ).T


def filter_imu_channels(
    imu_data: list[TimeIMU],
    transform: Callable[[np.ndarray], np.ndarray],
) -> list[TimeIMU]:
    """Run *transform* over the gyro and the accel axes of every sample.

    *transform* maps a (3, n) matrix to a filtered one. A channel is filtered
    only when at least one sample carries it; samples that are empty for a
    channel are returned untouched.
    """
    if not imu_data:
        return imu_data

    has_gyro = any(s.gyro is not None for s in imu_data)
    has_accl = any(s.accl is not None for s in imu_data)

    gyro_filtered = transform(_channel_matrix(imu_data, "gyro")) if has_gyro else None
    accl_filtered = transform(_channel_matrix(imu_data, "accl")) if has_accl else None

    result: list[TimeIMU] = []
    for i, sample in enumerate(imu_data):
        gyro = sample.gyro
        accl = sample.accl
        if gyro_filtered is not None and gyro is not None:
            gyro = np.asarray(gyro_filtered[:, i], dtype=np.float64)
        if accl_filtered is not None and accl is not None:
            accl = np.asarray(accl_filtered[:, i], dtype=np.float64)
        result.append(replace(sample, gyro=gyro, accl=accl))
    return result
