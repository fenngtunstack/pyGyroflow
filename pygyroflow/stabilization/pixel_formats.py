"""Pixel format types for the stabilization pipeline.

Port of Gyroflow's ``pixel_formats.rs`` (the pixel types) plus the
format-to-planes table from ``rendering/mod.rs:563-651`` — together they are
what lets the render path work on the decoder's *native* pixel format
instead of converting everything through 8-bit RGB first.

Two layers:

* :class:`PixelFormatType` — one per upstream pixel struct (``Luma8``,
  ``UV16``, ``AYUV16``, ...): component count, scalar dtype, max value, and
  :meth:`~PixelFormatType.from_rgb_color` — how an RGBA background lands in
  this type's channels. YUV types go through :func:`rgb_to_yuv` (Rec709,
  with the full→limited range remap for MPEG-range output).
* :func:`planes_for_format` — a decoder pixel format name → the list of
  plane specs to process (plane index, channel mapping, value limit),
  including upstream's quirks: NV21's swapped UV, the GBR plane order, the
  P010-is-actually-16-bit max value, and the unknown-format fallback that
  converts to YUV444P16LE first.

The scalar-dtype layer that the GPU path needs (bytemuck/OCL/wgpu format
tables) has no CPU counterpart and is not ported.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray


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


# ---------------------------------------------------------------------------
# RGB ↔ YUV (pixel_formats.rs:24-48)
# ---------------------------------------------------------------------------

# Rec709 constants, exactly as upstream names them.
_KR = 0.2126
_KB = 0.0722
_KG = 1.0 - _KR - _KB
_US = 1.0 / (2.0 - 2.0 * _KB)
_VS = 1.0 / (2.0 - 2.0 * _KR)


def _remap_to_limited(v: NDArray[np.float64], is_y: bool) -> NDArray[np.float64]:
    """0-255 (JPEG/full) → 16-235 (MPEG/limited), in 0..1 units."""
    if is_y:
        return (16.0 / 255.0) + v * ((235.0 - 16.0) / 255.0)
    return (16.0 / 255.0) + v * ((240.0 - 16.0) / 255.0)


def rgb_to_yuv(
    v: NDArray[np.float64], is_limited: bool
) -> NDArray[np.float64]:
    """Rec709 RGB(A) → YUV(A), last component passed through.

    *v* is a float array whose last axis holds (R, G, B, A) in 0..1; the
    result has the same shape with (Y, U, V, A) in 0..1. With *is_limited*
    the YUV channels are remapped into MPEG range (16-235 / 16-240).
    """
    v = np.asarray(v, dtype=np.float64)
    r, g, b, a = v[..., 0], v[..., 1], v[..., 2], v[..., 3]
    y = np.clip(_KR * r + _KG * g + _KB * b, 0.0, 1.0)
    u = np.clip((-_KR * _US) * r + (-_KG * _US) * g + ((1.0 - _KB) * _US) * b + 0.5,
                0.0, 1.0)
    vv = np.clip(((1.0 - _KR) * _VS) * r + (-_KG * _VS) * g + (-_KB * _VS) * b + 0.5,
                 0.0, 1.0)
    out = np.stack([y, u, vv, np.clip(a, 0.0, 1.0)], axis=-1)
    if is_limited:
        out[..., 0] = _remap_to_limited(out[..., 0], True)
        out[..., 1] = _remap_to_limited(out[..., 1], False)
        out[..., 2] = _remap_to_limited(out[..., 2], False)
    return out


# ---------------------------------------------------------------------------
# Pixel types (pixel_formats.rs:65-302)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PixelFormatType:
    """One upstream pixel struct: the CPU-visible parts.

    ``from_rgb_color`` maps an (..., 4) RGBA background into this type's
    channel layout — ``channels`` selects which converted components land in
    which slots, per the table at the top of each upstream impl.
    """

    name: str
    dtype: np.dtype
    count: int
    max_value: float | None
    # (v_rgba, yuv, channels, is_limited) -> (..., count) float array
    from_rgb: Callable[
        [NDArray, NDArray, Sequence[int], bool], NDArray
    ]
    scalar_pixel_type: PixelType = field(default=PixelType.U8)

    def from_rgb_color(
        self,
        v: NDArray[np.float64],
        channels: Sequence[int],
        is_limited: bool,
    ) -> NDArray[np.float64]:
        """RGBA background → this pixel type's channels (float, unclamped
        to the type's range — the write-back casts)."""
        v = np.asarray(v, dtype=np.float64)
        yuv = rgb_to_yuv(v, is_limited)
        return self.from_rgb(v, yuv, channels, is_limited)


def _make_passthrough(count: int):
    def from_rgb(v, yuv, channels, is_limited):
        return v[..., :count]
    return from_rgb


def _make_luma():
    def from_rgb(v, yuv, channels, is_limited):
        out = np.zeros(v.shape[:-1] + (1,), dtype=np.float64)
        out[..., 0] = yuv[..., channels[0]]
        return out
    return from_rgb


def _make_bgra(count: int):
    def from_rgb(v, yuv, channels, is_limited):
        out = v[..., :4].copy()
        return out[..., [2, 1, 0, 3]]
    return from_rgb


def _make_yuv_indexed(count: int):
    """AYUV16 / UV8 / UV16: components pulled from the converted YUV by
    the format's channel map."""
    def from_rgb(v, yuv, channels, is_limited):
        picks = [yuv[..., channels[k]] for k in range(count)]
        return np.stack(picks, axis=-1)
    return from_rgb


def _make_r32f():
    def from_rgb(v, yuv, channels, is_limited):
        out = np.zeros(v.shape[:-1] + (1,), dtype=np.float64)
        out[..., 0] = v[..., channels[0]]
        return out
    return from_rgb


LUMA8 = PixelFormatType("Luma8", np.dtype(np.uint8), 1, 255.0,
                        _make_luma(), PixelType.U8)
LUMA16 = PixelFormatType("Luma16", np.dtype(np.uint16), 1, 65535.0,
                         _make_luma(), PixelType.U16)
RGB8 = PixelFormatType("RGB8", np.dtype(np.uint8), 3, 255.0,
                       _make_passthrough(3), PixelType.U8)
RGBA8 = PixelFormatType("RGBA8", np.dtype(np.uint8), 4, 255.0,
                        _make_passthrough(4), PixelType.U8)
BGRA8 = PixelFormatType("BGRA8", np.dtype(np.uint8), 4, 255.0,
                        _make_bgra(4), PixelType.U8)
RGB16 = PixelFormatType("RGB16", np.dtype(np.uint16), 3, 65535.0,
                        _make_passthrough(3), PixelType.U16)
RGBA16 = PixelFormatType("RGBA16", np.dtype(np.uint16), 4, 65535.0,
                         _make_passthrough(4), PixelType.U16)
AYUV16 = PixelFormatType("AYUV16", np.dtype(np.uint16), 4, 65535.0,
                         _make_yuv_indexed(4), PixelType.U16)
RGBAF = PixelFormatType("RGBAf", np.dtype(np.float32), 4, None,
                        _make_passthrough(4), PixelType.F32)
RGBAF16 = PixelFormatType("RGBAf16", np.dtype(np.float16), 4, None,
                          _make_passthrough(4), PixelType.F16)
R32F = PixelFormatType("R32f", np.dtype(np.float32), 1, None,
                       _make_r32f(), PixelType.F32)
UV8 = PixelFormatType("UV8", np.dtype(np.uint8), 2, 255.0,
                      _make_yuv_indexed(2), PixelType.U8)
UV16 = PixelFormatType("UV16", np.dtype(np.uint16), 2, 65535.0,
                       _make_yuv_indexed(2), PixelType.U16)


# ---------------------------------------------------------------------------
# Decoder format → planes (rendering/mod.rs:563-651)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlaneSpec:
    """One plane to process: pixel type, plane index in the frame, the
    channel map into the converted YUV/RGB components, and the value limit
    that lands in ``kernel_params.pixel_value_limit`` / ``max_pixel_value``.
    """

    pixel_type: PixelFormatType
    plane_index: int
    channels: tuple[int, ...]
    max_val: float


_YCBCR_16BIT_FORMATS = (
    "yuv420p10le", "yuv422p10le", "yuv444p10le",
    "yuv420p12le", "yuv422p12le", "yuv444p12le",
    "yuv420p14le", "yuv422p14le", "yuv444p14le",
    "yuv420p16le", "yuv422p16le", "yuv444p16le",
)
_YCBCR_MAX_BY_DEPTH = {
    "10": 1023.0,
    "12": 4095.0,
    "14": 16383.0,
    "16": 65535.0,
}

# The unknown-format fallback: convert to YUV444P16LE first, then process
# three Luma16 planes (rendering/mod.rs:645-651).
FALLBACK_CONVERT_TO = "yuv444p16le"


def planes_for_format(pix_fmt: str) -> tuple[list[PlaneSpec], str | None]:
    """Plane specs for a decoder pixel format name.

    Returns ``(planes, convert_to)``: *convert_to* is ``None`` when the
    format is handled natively, otherwise the pixel format the renderer must
    convert to first (upstream converts unknowns to YUV444P16LE — 4:4:4 so
    the chroma planes keep even dimensions).

    PyAV's format names are the ffmpeg ones (``"nv12"``, ``"yuv420p10le"``,
    ``"gbrpf32le"``, ...), lower case — matching the table in
    ``rendering/mod.rs:563-643``.
    """
    fmt = (pix_fmt or "").lower()

    if fmt == "nv12":
        return [PlaneSpec(LUMA8, 0, (0,), 255.0),
                PlaneSpec(UV8, 1, (1, 2), 255.0)], None
    if fmt == "nv21":
        return [PlaneSpec(LUMA8, 0, (0,), 255.0),
                PlaneSpec(UV8, 1, (2, 1), 255.0)], None
    if fmt in ("p010le", "p016le", "p210le", "p216le", "p410le", "p416le"):
        # Upstream note: P010LE appears to use full 16-bit values despite
        # being 10-bit, so every P-plane gets 65535.0.
        return [PlaneSpec(LUMA16, 0, (0,), 65535.0),
                PlaneSpec(UV16, 1, (1, 2), 65535.0)], None
    if fmt in ("yuv420p", "yuvj420p"):
        return [PlaneSpec(LUMA8, 0, (0,), 255.0),
                PlaneSpec(LUMA8, 1, (1,), 255.0),
                PlaneSpec(LUMA8, 2, (2,), 255.0)], None
    if fmt in _YCBCR_16BIT_FORMATS:
        depth = fmt[-4:-2]  # "10" .. "16" from e.g. yuv420p10le
        max_val = _YCBCR_MAX_BY_DEPTH[depth]
        return [PlaneSpec(LUMA16, 0, (0,), max_val),
                PlaneSpec(LUMA16, 1, (1,), max_val),
                PlaneSpec(LUMA16, 2, (2,), max_val)], None
    if fmt in ("yuva444p10le", "yuva444p12le", "yuva444p16le"):
        max_val = {"10": 1023.0, "12": 4095.0}.get(fmt[-4:-2], 65535.0)
        return [PlaneSpec(LUMA16, 0, (0,), max_val),
                PlaneSpec(LUMA16, 1, (1,), max_val),
                PlaneSpec(LUMA16, 2, (2,), max_val),
                PlaneSpec(LUMA16, 3, (3,), max_val)], None
    if fmt == "gbrapf32le":
        # GBR(A) plane order — the channel indices are upstream's verbatim.
        return [PlaneSpec(R32F, 0, (2,), 255.0),
                PlaneSpec(R32F, 0, (0,), 255.0),
                PlaneSpec(R32F, 0, (1,), 255.0),
                PlaneSpec(R32F, 0, (3,), 255.0)], None
    if fmt == "gbrpf32le":
        return [PlaneSpec(R32F, 0, (2,), 255.0),
                PlaneSpec(R32F, 0, (0,), 255.0),
                PlaneSpec(R32F, 0, (1,), 255.0)], None
    if fmt == "ayuv64le":
        return [PlaneSpec(AYUV16, 0, (3, 0, 1, 2), 65535.0)], None
    if fmt == "rgb24":
        return [PlaneSpec(RGB8, 0, (), 255.0)], None
    if fmt == "rgba":
        return [PlaneSpec(RGBA8, 0, (), 255.0)], None
    if fmt == "rgb48be":
        return [PlaneSpec(RGB16, 0, (), 65535.0)], None
    if fmt == "rgba64be":
        return [PlaneSpec(RGBA16, 0, (), 65535.0)], None

    return ([PlaneSpec(LUMA16, 0, (0,), 65535.0),
             PlaneSpec(LUMA16, 1, (1,), 65535.0),
             PlaneSpec(LUMA16, 2, (2,), 65535.0)],
            FALLBACK_CONVERT_TO)
