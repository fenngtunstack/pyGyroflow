"""PyGyroFlow — Python port of Gyroflow video stabilization.

Production-grade video stabilization library with bit-exact algorithms
matching the original Gyroflow (Rust) implementation.
"""

from pygyroflow._version import __version__

from pygyroflow.filtering import (
    lowpass_filter,
    lowpass_filter_channels,
    lowpass_filter_imu,
    median_filter,
    median_filter_channels,
    median_filter_imu,
)
from pygyroflow.keyframes import (
    Easing,
    Keyframe,
    KeyframeManager,
    KeyframeType,
)
from pygyroflow.gyro_source import GyroSource, FileMetadata, IMUTransforms
from pygyroflow.manager import StabilizationManager
from pygyroflow.stabilization_params import StabilizationParams
from pygyroflow.util import timestamp_at_frame, frame_at_timestamp

__all__ = [
    "__version__",
    # filtering
    "lowpass_filter",
    "lowpass_filter_channels",
    "lowpass_filter_imu",
    "median_filter",
    "median_filter_channels",
    "median_filter_imu",
    # keyframes
    "Easing",
    "Keyframe",
    "KeyframeManager",
    "KeyframeType",
    # gyro source
    "GyroSource",
    "FileMetadata",
    "IMUTransforms",
    # manager
    "StabilizationManager",
    "StabilizationParams",
    # utilities
    "timestamp_at_frame",
    "frame_at_timestamp",
]
