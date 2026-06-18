"""PyGyroFlow core types — foundation layer for all other modules."""

from pygyroflow.types.quaternion import Quat64
from pygyroflow.types.time_types import TimeIMU, TimeQuat, TimeVec
from pygyroflow.types.kernel_params import KernelParams, kernel_params_from_dict
from pygyroflow.types.enums import (
    BackgroundMode,
    DistortionModelType,
    Interpolation,
    ReadoutDirection,
)
from pygyroflow.types.errors import (
    GPUError,
    GyroflowError,
    LensProfileError,
    StabilizationError,
    TelemetryParseError,
    VideoIOError,
)

__all__ = [
    # Quaternion
    "Quat64",
    # Time-indexed types
    "TimeIMU",
    "TimeQuat",
    "TimeVec",
    # GPU kernel params
    "KernelParams",
    "kernel_params_from_dict",
    # Enums
    "BackgroundMode",
    "DistortionModelType",
    "Interpolation",
    "ReadoutDirection",
    # Errors
    "GyroflowError",
    "TelemetryParseError",
    "LensProfileError",
    "StabilizationError",
    "GPUError",
    "VideoIOError",
]
