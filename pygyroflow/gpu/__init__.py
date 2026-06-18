"""GPU backend module for PyGyroFlow.

Provides the wgpu-based GPU undistortion pipeline and shader builder.
"""

from pygyroflow.gpu.buffers import BufferDescription, BufferSource
from pygyroflow.gpu.shader_builder import build_undistort_shader, SHADER_DIR
from pygyroflow.gpu.backend import WgpuBackend

__all__ = [
    "BufferDescription",
    "BufferSource",
    "WgpuBackend",
    "build_undistort_shader",
    "SHADER_DIR",
]
