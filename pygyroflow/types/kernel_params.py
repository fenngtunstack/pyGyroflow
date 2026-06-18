"""KernelParams ctypes structure matching the WGSL KernelParams struct byte-for-byte.

Layout derived from wgpu_undistort.wgsl lines 9-52. Total size: 320 bytes.

WGSL alignment rules applied:
  - vec2<T> aligns to 8 bytes
  - vec4<T> aligns to 16 bytes
  - f32/i32 are 4 bytes each

Since all fields are 4-byte scalars or arrays thereof, the ctypes natural layout
produces the exact same byte sequence as the WGSL struct (no padding needed).
"""

from __future__ import annotations

import ctypes
from typing import Any


class KernelParams(ctypes.Structure):
    """GPU undistortion kernel parameters.

    Must match the WGSL struct layout exactly. Total: 320 bytes.
    """

    _fields_ = [
        # -- offset 0-15: image dimensions --
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("stride", ctypes.c_int32),
        ("output_width", ctypes.c_int32),
        # -- offset 16-31: output dimensions --
        ("output_height", ctypes.c_int32),
        ("output_stride", ctypes.c_int32),
        ("matrix_count", ctypes.c_int32),
        ("interpolation", ctypes.c_int32),
        # -- offset 32-47: mode flags --
        ("background_mode", ctypes.c_int32),
        ("flags", ctypes.c_int32),
        ("bytes_per_pixel", ctypes.c_int32),
        ("pix_element_count", ctypes.c_int32),
        # -- offset 48-63: background color --
        ("background", ctypes.c_float * 4),
        # -- offset 64-71: focal length in pixels --
        ("f", ctypes.c_float * 2),
        # -- offset 72-79: lens center --
        ("c", ctypes.c_float * 2),
        # -- offset 80-127: distortion coefficients --
        ("k1", ctypes.c_float * 4),
        ("k2", ctypes.c_float * 4),
        ("k3", ctypes.c_float * 4),
        # -- offset 128-143: lens correction --
        ("fov", ctypes.c_float),
        ("r_limit", ctypes.c_float),
        ("lens_correction_amount", ctypes.c_float),
        ("input_vertical_stretch", ctypes.c_float),
        # -- offset 144-159: margins --
        ("input_horizontal_stretch", ctypes.c_float),
        ("background_margin", ctypes.c_float),
        ("background_margin_feather", ctypes.c_float),
        ("canvas_scale", ctypes.c_float),
        # -- offset 160-175: rotation and 2D translation --
        ("input_rotation", ctypes.c_float),
        ("output_rotation", ctypes.c_float),
        ("translation2d", ctypes.c_float * 2),
        # -- offset 176-191: 3D translation --
        ("translation3d", ctypes.c_float * 4),
        # -- offset 192-207: source rectangle (x, y, w, h) --
        ("source_rect", ctypes.c_int32 * 4),
        # -- offset 208-223: output rectangle (x, y, w, h) --
        ("output_rect", ctypes.c_int32 * 4),
        # -- offset 224-239: digital lens parameters --
        ("digital_lens_params", ctypes.c_float * 4),
        # -- offset 240-255: safe area rectangle --
        ("safe_area_rect", ctypes.c_float * 4),
        # -- offset 256-271: pixel limits and model --
        ("max_pixel_value", ctypes.c_float),
        ("distortion_model", ctypes.c_int32),
        ("digital_lens", ctypes.c_int32),
        ("pixel_value_limit", ctypes.c_float),
        # -- offset 272-287: optics --
        ("light_refraction_coefficient", ctypes.c_float),
        ("plane_index", ctypes.c_int32),
        ("reserved1", ctypes.c_float),
        ("reserved2", ctypes.c_float),
        # -- offset 288-303: EWA coefficients --
        ("ewa_coeffs_p", ctypes.c_float * 4),
        # -- offset 304-319: EWA coefficients --
        ("ewa_coeffs_q", ctypes.c_float * 4),
    ]

    def to_bytes(self) -> bytes:
        """Serialize to raw bytes suitable for GPU uniform buffer upload."""
        return ctypes.string_at(ctypes.byref(self), ctypes.sizeof(self))

    @classmethod
    def from_bytes(cls, data: bytes) -> KernelParams:
        """Deserialize from raw bytes."""
        if len(data) < ctypes.sizeof(cls):
            raise ValueError(
                f"Need at least {ctypes.sizeof(cls)} bytes, got {len(data)}"
            )
        params = cls()
        ctypes.memmove(ctypes.byref(params), data, ctypes.sizeof(cls))
        return params


def kernel_params_from_dict(d: dict[str, Any]) -> KernelParams:
    """Create KernelParams from a dict with human-readable keys.

    Accepts both scalar values and tuples/lists for vector fields.
    Missing fields default to zero.
    """
    p = KernelParams()

    # Scalar i32 fields
    _set_int_fields(p, d, [
        "width", "height", "stride", "output_width",
        "output_height", "output_stride", "matrix_count", "interpolation",
        "background_mode", "flags", "bytes_per_pixel", "pix_element_count",
    ])

    # Scalar f32 fields
    _set_float_fields(p, d, [
        "fov", "r_limit", "lens_correction_amount", "input_vertical_stretch",
        "input_horizontal_stretch", "background_margin", "background_margin_feather",
        "canvas_scale", "input_rotation", "output_rotation",
        "max_pixel_value", "pixel_value_limit",
        "light_refraction_coefficient", "reserved1", "reserved2",
    ])

    # i32 fields (after distortion_model group)
    _set_int_fields(p, d, ["distortion_model", "digital_lens", "plane_index"])

    # vec4<f32> fields
    _set_vec4_float(p, d, "background")
    _set_vec4_float(p, d, "k1")
    _set_vec4_float(p, d, "k2")
    _set_vec4_float(p, d, "k3")
    _set_vec4_float(p, d, "translation3d")
    _set_vec4_float(p, d, "digital_lens_params")
    _set_vec4_float(p, d, "safe_area_rect")
    _set_vec4_float(p, d, "ewa_coeffs_p")
    _set_vec4_float(p, d, "ewa_coeffs_q")

    # vec2<f32> fields
    _set_vec2_float(p, d, "f")
    _set_vec2_float(p, d, "c")
    _set_vec2_float(p, d, "translation2d")

    # vec4<i32> fields
    _set_vec4_int(p, d, "source_rect")
    _set_vec4_int(p, d, "output_rect")

    return p


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _set_int_fields(p: KernelParams, d: dict[str, Any], names: list[str]) -> None:
    for name in names:
        if name in d:
            setattr(p, name, int(d[name]))


def _set_float_fields(p: KernelParams, d: dict[str, Any], names: list[str]) -> None:
    for name in names:
        if name in d:
            setattr(p, name, float(d[name]))


def _set_vec4_float(p: KernelParams, d: dict[str, Any], name: str) -> None:
    if name not in d:
        return
    arr = d[name]
    arr = [float(x) for x in arr]
    if len(arr) != 4:
        raise ValueError(f"{name} must have 4 elements, got {len(arr)}")
    setattr(p, name, (ctypes.c_float * 4)(*arr))


def _set_vec2_float(p: KernelParams, d: dict[str, Any], name: str) -> None:
    if name not in d:
        return
    arr = d[name]
    arr = [float(x) for x in arr]
    if len(arr) != 2:
        raise ValueError(f"{name} must have 2 elements, got {len(arr)}")
    setattr(p, name, (ctypes.c_float * 2)(*arr))


def _set_vec4_int(p: KernelParams, d: dict[str, Any], name: str) -> None:
    if name not in d:
        return
    arr = d[name]
    arr = [int(x) for x in arr]
    if len(arr) != 4:
        raise ValueError(f"{name} must have 4 elements, got {len(arr)}")
    setattr(p, name, (ctypes.c_int32 * 4)(*arr))
