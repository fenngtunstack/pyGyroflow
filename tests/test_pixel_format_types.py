"""The pixel-format type system (C-03, pixel_formats.rs + the format table).

These pin the formulas and channel maps the per-plane render pipeline
relies on. The RGB→YUV constants are Rec709; the channel permutations are
upstream's verbatim — NV21's swapped UV, GBR's plane order, AYUV's packed
(A, Y, U, V) — and the value limits carry each format's bit depth.
"""

from __future__ import annotations

import numpy as np
import pytest

from pygyroflow.stabilization.pixel_formats import (
    AYUV16,
    BGRA8,
    FALLBACK_CONVERT_TO,
    LUMA16,
    LUMA8,
    RGBAF,
    RGB8,
    R32F,
    UV8,
    UV16,
    PixelType,
    planes_for_format,
    rgb_to_yuv,
)  # noqa: I001 - grouped by theme; ruff's sorted order splits the pairs

_WHITE = np.array([[[1.0, 1.0, 1.0, 1.0]]])
_RED = np.array([[[1.0, 0.0, 0.0, 1.0]]])


class TestRgbToYuv:
    def test_rec709_grey_is_y_only(self):
        out = rgb_to_yuv(_WHITE, is_limited=False)
        # Y = 0.2126 + 0.7678 + 0.0722 = 1.0; U/V sit at 0.5 (chroma null).
        assert out[..., 0] == pytest.approx(1.0)
        assert out[..., 1] == pytest.approx(0.5)
        assert out[..., 2] == pytest.approx(0.5)
        assert out[..., 3] == pytest.approx(1.0)  # alpha passes through

    def test_full_range_red(self):
        out = rgb_to_yuv(_RED, is_limited=False)
        # Y = KR = 0.2126; U = -KR*US*1 + 0.5; V = (1-KR)*VS*1 + 0.5.
        us = 1.0 / (2.0 - 2.0 * 0.0722)
        vs = 1.0 / (2.0 - 2.0 * 0.2126)
        assert out[..., 0] == pytest.approx(0.2126)
        assert out[..., 1] == pytest.approx(-0.2126 * us + 0.5)
        assert out[..., 2] == pytest.approx(0.7874 * vs + 0.5)

    def test_limited_range_remaps_y_and_uv_differently(self):
        y_limited = rgb_to_yuv(_WHITE, is_limited=True)[..., 0]
        # Y 1.0 -> 235/255 (span 235-16); U/V 0.5 -> 16/255 + 0.5*(240-16)/255
        # = 128/255 (span 240-16, neutral chroma keeps its half offset).
        assert y_limited[0] == pytest.approx(235.0 / 255.0)
        assert rgb_to_yuv(_WHITE, True)[..., 1][0] == pytest.approx(128.0 / 255.0)
        # The Y span differs from the UV span: Y-only remap to the UV formula
        # would give 0.5 + 0.5*((235-16)/255 - (240-16)/255) off.
        y_uv_span = (16.0 / 255.0) + 1.0 * ((240.0 - 16.0) / 255.0)
        assert y_limited[0] != pytest.approx(y_uv_span)

    def test_black_in_limited_is_not_pedestal_free(self):
        out = rgb_to_yuv(np.zeros((1, 1, 4)), is_limited=True)
        # Y drops to the 16 pedestal; neutral chroma (the +0.5 offset) maps
        # to 128, not the pedestal — black is not green.
        assert out[..., 0][0] == pytest.approx(16.0 / 255.0)
        assert out[..., 1][0] == pytest.approx(128.0 / 255.0)
        assert out[..., 2][0] == pytest.approx(128.0 / 255.0)


class TestFromRgbColor:
    def test_luma_takes_the_requested_yuv_component(self):
        grey = np.full((1, 1, 4), 0.5)
        out = LUMA8.from_rgb_color(grey, (0,), is_limited=False)
        want = float(rgb_to_yuv(grey, False)[0, 0, 0])
        assert float(out[0, 0, 0]) == pytest.approx(want)
        assert out.shape[-1] == 1

    def test_rgb_passthrough_ignores_channels_and_range(self):
        rgb = np.array([[[0.1, 0.5, 0.9, 1.0]]])
        out = RGB8.from_rgb_color(rgb, (), is_limited=True)
        assert out[0, 0, :3] == pytest.approx([0.1, 0.5, 0.9])

    def test_bgra_swaps_r_and_b(self):
        rgb = np.array([[[0.1, 0.5, 0.9, 0.25]]])
        out = BGRA8.from_rgb_color(rgb, (), is_limited=False)
        assert out[0, 0] == pytest.approx([0.9, 0.5, 0.1, 0.25])

    def test_uv_picks_components_by_index(self):
        grey = np.full((1, 1, 4), 0.5)
        yuv = rgb_to_yuv(grey, False)
        out = UV8.from_rgb_color(grey, (1, 2), is_limited=False)
        assert out[0, 0, 0] == pytest.approx(yuv[0, 0, 1])
        assert out[0, 0, 1] == pytest.approx(yuv[0, 0, 2])

    def test_ayuv_packs_alpha_first(self):
        rgb = np.array([[[0.3, 0.6, 0.9, 0.75]]])
        yuv = rgb_to_yuv(rgb, False)
        out = AYUV16.from_rgb_color(rgb, (3, 0, 1, 2), is_limited=False)
        assert out[0, 0, 0] == pytest.approx(0.75)          # A from v[3]
        assert out[0, 0, 1] == pytest.approx(yuv[0, 0, 0])  # Y
        assert out[0, 0, 2] == pytest.approx(yuv[0, 0, 1])  # U
        assert out[0, 0, 3] == pytest.approx(yuv[0, 0, 2])  # V

    def test_r32f_takes_raw_rgb_not_yuv(self):
        rgb = np.array([[[0.3, 0.6, 0.9, 1.0]]])
        out = R32F.from_rgb_color(rgb, (2,), is_limited=False)
        assert out[0, 0, 0] == pytest.approx(0.9)  # v[2], not yuv[2]

    def test_limited_range_reaches_the_yuv_types(self):
        grey = np.full((1, 1, 4), 1.0)
        out = LUMA16.from_rgb_color(grey, (0,), is_limited=True)
        assert out[..., 0][0] == pytest.approx(235.0 / 255.0)


class TestTypeDescriptors:
    def test_max_values_match_upstream(self):
        assert LUMA8.max_value == 255.0
        assert LUMA16.max_value == 65535.0
        assert UV8.max_value == 255.0
        assert UV16.max_value == 65535.0
        assert AYUV16.max_value == 65535.0
        assert RGBAF.max_value is None  # float types have no limit
        assert R32F.max_value is None

    def test_scalar_types(self):
        assert LUMA8.scalar_pixel_type == PixelType.U8
        assert LUMA16.scalar_pixel_type == PixelType.U16
        assert RGBAF.scalar_pixel_type == PixelType.F32
        assert RGB8.scalar_pixel_type == PixelType.U8


class TestTheFormatTable:
    def test_nv12_is_luma_plus_interleaved_uv(self):
        planes, conv = planes_for_format("nv12")
        assert conv is None
        assert [(p.pixel_type.name, p.plane_index, p.channels, p.max_val)
                for p in planes] == [
            ("Luma8", 0, (0,), 255.0),
            ("UV8", 1, (1, 2), 255.0),
        ]

    def test_nv21_swaps_uv(self):
        planes, _ = planes_for_format("nv21")
        assert planes[1].channels == (2, 1)

    def test_p010_quirk_is_sixteen_bit(self):
        """Upstream's own comment: P010LE appears to carry full 16-bit
        values, so every P-plane gets 65535 — not 1023."""
        planes, conv = planes_for_format("p010le")
        assert conv is None
        assert all(p.max_val == 65535.0 for p in planes)

    @pytest.mark.parametrize("fmt,max_val", [
        ("yuv420p10le", 1023.0), ("yuv422p12le", 4095.0),
        ("yuv444p14le", 16383.0), ("yuv420p16le", 65535.0),
    ])
    def test_yuv_bit_depths_carry_their_limits(self, fmt, max_val):
        planes, conv = planes_for_format(fmt)
        assert conv is None
        assert len(planes) == 3
        assert all(p.pixel_type is LUMA16 for p in planes)
        assert all(p.max_val == max_val for p in planes)

    def test_yuv420p8_is_three_luma8_planes(self):
        planes, conv = planes_for_format("yuv420p")
        assert conv is None
        assert all(p.pixel_type is LUMA8 for p in planes)
        assert [p.channels for p in planes] == [(0,), (1,), (2,)]

    def test_gbrp_plane_order_is_upstreams_verbatim(self):
        planes, conv = planes_for_format("gbrpf32le")
        assert conv is None
        assert [p.channels for p in planes] == [(2,), (0,), (1,)]
        assert all(p.pixel_type is R32F for p in planes)

    def test_gbrapf32_has_the_alpha_plane(self):
        planes, _ = planes_for_format("gbrapf32le")
        assert [p.channels for p in planes] == [(2,), (0,), (1,), (3,)]

    def test_ayuv64_channel_map(self):
        planes, _ = planes_for_format("ayuv64le")
        assert planes[0].channels == (3, 0, 1, 2)
        assert planes[0].pixel_type is AYUV16

    def test_rgb_packed_formats(self):
        for fmt, ptype in (("rgb24", "RGB8"), ("rgba", "RGBA8"),
                           ("rgb48be", "RGB16"), ("rgba64be", "RGBA16")):
            planes, conv = planes_for_format(fmt)
            assert conv is None and len(planes) == 1
            assert planes[0].pixel_type.name == ptype

    def test_unknown_format_falls_back_to_yuv444p16(self):
        planes, conv = planes_for_format("mjpeg-something-weird")
        assert conv == FALLBACK_CONVERT_TO
        assert len(planes) == 3
        assert all(p.pixel_type is LUMA16 and p.max_val == 65535.0
                   for p in planes)
