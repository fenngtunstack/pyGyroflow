"""Stabilization pipeline — frame transforms and CPU undistortion.

This package provides the core stabilization algorithm:
  - ComputeParams: parameter snapshot for the transform
  - FrameTransform: per-frame (or per-row) transformation matrices
  - cpu_undistort: CPU-based image warping with bilinear interpolation

Usage:
    from pygyroflow.stabilization import ComputeParams, FrameTransform, cpu_undistort

    params = ComputeParams(width=1920, height=1080, ...)
    transform = FrameTransform.at_timestamp(params, timestamp_ms=1000.0, frame=30)
    output = cpu_undistort(input_frame, transform)
"""

from pygyroflow.stabilization.compute_params import ComputeParams
from pygyroflow.stabilization.frame_transform import FrameTransform
from pygyroflow.stabilization.cpu_undistort import cpu_undistort
from pygyroflow.stabilization.pixel_formats import PixelType, get_pixel_type, bytes_per_pixel

__all__ = [
    "ComputeParams",
    "FrameTransform",
    "cpu_undistort",
    "PixelType",
    "get_pixel_type",
    "bytes_per_pixel",
]
