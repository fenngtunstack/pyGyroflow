"""Pixel format types for CPU undistortion pipeline.

Simplified from Gyroflow's pixel_formats.rs which defines GPU-compatible
pixel types with bytemuck traits. Here we focus on the CPU path with numpy,
providing pixel type identification and byte-size calculations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from enum import IntEnum


class PixelType(IntEnum):
    """Scalar pixel component type, matching GPU shader conventions."""

    U8 = 0
    U16 = 1
    F32 = 2
    F16 = 3


def bytes_per_pixel(pixel_type: PixelType) -> int:
    """Return bytes per scalar component for a pixel type."""
    return {
        PixelType.U8:  1,
        PixelType.U16: 2,
        PixelType.F32: 4,
        PixelType.F16: 2,
    }[pixel_type]


def get_pixel_type(frame: NDArray) -> PixelType:
    """Infer PixelType from a numpy array's dtype.

    Args:
        frame: Input image array.

    Returns:
        The matching PixelType enum value.

    Raises:
        ValueError: If the dtype is not recognized.
    """
    dt = frame.dtype
    if dt == np.uint8:
        return PixelType.U8
    if dt == np.uint16:
        return PixelType.U16
    if dt == np.float32:
        return PixelType.F32
    if dt == np.float16:
        return PixelType.F16
    raise ValueError(f"Unsupported pixel dtype: {dt}")
