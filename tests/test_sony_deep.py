"""The gyro_source/sony.py port (A-06): time offsets, IBIS/OIS splines,
LensDistortion profile, mesh correction — unit-level with synthetic tags."""

from __future__ import annotations

import struct

import numpy as np  # noqa: F401  (kept for parity with sibling test files)
import pytest

from pygyroflow.gyro_source import sony as S
from pygyroflow.gyro_source.file_metadata import FileMetadata
from pygyroflow.gyro_source.splines import interpolate_mesh

# ---------------------------------------------------------------------------
# Tag payload builders
# ---------------------------------------------------------------------------


def _ibis1(entries):
    return struct.pack(">ii", len(entries), 16) + b"".join(
        struct.pack(">4i", *e) for e in entries)


def _ois(entries):
    return struct.pack(">ii", len(entries), 16) + b"".join(
        struct.pack(">4i", *e) for e in entries)


def _lens_distortion(focal_nm, height_nm, coeff_scale, coeffs):
    out = struct.pack(">IIBfII", focal_nm, height_nm, 0, coeff_scale,
                      len(coeffs), 2)
    out += b"".join(struct.pack(">H", c) for c in coeffs)
    return out


def _focal_plane(pairs):
    # unk1 i32, unk2 i16, cc i16 (scale denominator), count i32, size i32
    out = struct.pack(">ihhii", 0, 0, 8, len(pairs), 4)
    out += b"".join(struct.pack(">hh", *p) for p in pairs)
    return out


def _mesh(div_x=9, div_y=9, size=(5028, 2828), value_scale=1.0):
    xs = [int(i * value_scale) for i in range(81)]
    ys = [int(-i * value_scale) for i in range(81)]
    out = struct.pack(">hii HH", 0, 0, 0, size[0], size[1])
    out += struct.pack(">81h", *xs) + struct.pack(">81h", *ys)
    out += struct.pack(">4B", 0, 0, div_x, div_y)  # 2^0 subdivision
    return out


def _imager_tags(**over):
    tags = {
        S.T_SENSOR_SIZE_PX: struct.pack(">HH", 5028, 2828),
        S.T_PIXEL_PITCH: struct.pack(">HH", 2400, 2400),
        S.T_CAPTURE_ORIGIN: struct.pack(">II", 2, 1),
        S.T_CAPTURE_SIZE: struct.pack(">II", 5024, 2826),
        S.T_FIRST_FRAME_TS: struct.pack(">i", 34659),
        S.T_EXPOSURE_TIME: struct.pack(">i", 3598),
        S.T_GYRO_FREQUENCY: struct.pack(">i", 2000),
        S.T_GYRO_SCALER: struct.pack(">i", 1000000),
        S.T_GYRO_TIME_OFFSET: struct.pack(">i", -8909),
    }
    tags.update(over)
    return tags


# ---------------------------------------------------------------------------
# get_time_offset (sony.rs:212-230, verbatim arithmetic)
# ---------------------------------------------------------------------------


class TestGetTimeOffset:
    def test_exact_synthetic_values(self):
        md = FileMetadata(detected_source="Sony")
        md.frame_readout_time = 14.297
        orig, offset = S.get_time_offset(
            md, _imager_tags(), 2005.0, "DSC-RX0M2")
        # first=34.659ms, exp/2=1.799, readout/2=7.1485, model=1.5 (RX0M2),
        # offset=-8.909ms: rounded=-8909, period=500,
        # offset_diff=round(-8909 - 500*floor(-17.818))/1000 = 91/1000
        expected_ms = 34.659 - 1.799 + 7.1485 + 1.5 + 0.091 - (-8.909)
        assert orig == 2000.0
        assert offset == pytest.approx(expected_ms / 2000.0 * 2005.0, rel=1e-9)

    def test_readout_term_scales_as_ms(self):
        """frame_offset comes out in ms; with sample_rate == gyro rate the
        scaling is 1:1, so the readout term is exactly frt/2."""
        md = FileMetadata(detected_source="Sony")
        md.frame_readout_time = 14.297
        a = S.get_time_offset(md, _imager_tags(), 2000.0, None)
        md.frame_readout_time = 0.0
        b = S.get_time_offset(md, _imager_tags(), 2000.0, None)
        assert a[1] - b[1] == pytest.approx(14.297 / 2, rel=1e-9)

    def test_missing_tag_returns_none(self):
        tags = _imager_tags()
        del tags[S.T_GYRO_TIME_OFFSET]
        assert S.get_time_offset(FileMetadata(detected_source="Sony"),
                                 tags, 2000.0) is None


# ---------------------------------------------------------------------------
# stab_collect / stab_calc_splines
# ---------------------------------------------------------------------------


class TestStabSplines:
    def test_ois_only_file(self):
        """OIS without IBIS: `t` comes from the OIS table itself
        (sony.rs:349-351) and the splines carry OIS points only."""
        md = FileMetadata(detected_source="Sony")
        md.frame_readout_time = 14.297
        is_temp = S.ISTemp()
        entries = [(i * 500, 100 * i, -50 * i, 0) for i in range(16)]
        for frame in range(3):
            tags = _imager_tags()
            tags[S.T_OIS_DATA] = _ois(entries)
            tags[S.T_FIRST_FRAME_TS] = struct.pack(
                ">i", 34659 + frame * 33367)
            assert S.stab_collect(is_temp, tags, 29.97)
        assert is_temp.original_sample_rate == 2000.0
        assert len(is_temp.per_frame_start_idx) == 3
        assert is_temp.t  # OIS-only pushed the time column

        stab = S.stab_calc_splines(md, is_temp)
        assert stab is not None and len(stab) == 3
        for cs in stab:
            assert len(cs.ois_spline) > 0
            assert len(cs.ibis_spline) == 0
        v = stab[0].ois_spline.interpolate(0.5)
        assert v is not None and len(v) == 3

    def test_missing_imager_tags_abort_frame(self):
        is_temp = S.ISTemp()
        tags = _imager_tags()
        del tags[S.T_PIXEL_PITCH]
        assert not S.stab_collect(is_temp, tags, 29.97)
        assert S.stab_calc_splines(
            FileMetadata(detected_source="Sony"), is_temp) is None


# ---------------------------------------------------------------------------
# init_lens_profile
# ---------------------------------------------------------------------------


class TestInitLensProfile:
    # Real RX100M7 LensDistortion table (coeff_scale 200): the angles are
    # c/scale/180*pi radians. Used so the SVD path runs on plausible data.
    REAL_COEFFS = [738, 1477, 2218, 2962, 3708, 4456, 5206, 5955, 6701, 7437]

    def test_profile_from_lens_distortion(self):
        md = FileMetadata(detected_source="Sony")
        md.frame_readout_time = 14.297
        tags = _imager_tags()
        tags[S.T_LENS_DIST_DATA] = _lens_distortion(
            9_332_000, 6_012_884, 200.0, self.REAL_COEFFS)

        S.init_lens_profile(md, tags, (3840, 2160), 0.0, "DSC-RX100M7")

        assert md.lens_profile is not None
        lp = md.lens_profile
        assert lp["camera_brand"] == "Sony"
        assert lp["camera_model"] == "DSC-RX100M7"
        assert lp["distortion_model"] == "sony"
        assert lp["official"] is True
        assert lp["sync_settings"]["search_size"] == 0.3
        fx = lp["fisheye_params"]["camera_matrix"][0][0]
        # matches the real-footage parse (fx ≈ 2823 px on the actual clip)
        assert 1000.0 < fx < 6000.0

    def test_fallback_when_focal_ratio_is_bad(self):
        """Claimed 20 mm (0x8005) vs true 9 mm (focal_length_nm):
        ratio 2.2 → the "Not calibrated" fallback profile."""
        md = FileMetadata(detected_source="Sony")
        tags = _imager_tags()
        tags[S.T_LENS_DIST_DATA] = _lens_distortion(
            9_000_000, 6_012_884, 200.0, self.REAL_COEFFS)
        # f16 of 0.02: exponent nibble 12 encodes 10^-4, mantissa 200
        tags[S.T_LENS_FOCAL] = struct.pack(">H", (12 << 12) | 200)
        tags[S.T_SENSOR_WIDTH_MM] = struct.pack(">H", 13200)
        tags[S.T_SENSOR_HEIGHT_MM] = struct.pack(">H", 8800)

        S.init_lens_profile(md, tags, (3840, 2160), 0.0, None)

        assert md.lens_profile is not None
        lp = md.lens_profile
        assert lp["calibrated_by"] == "Not calibrated"
        assert lp["official"] is False
        assert "distortion_model" not in lp  # fallback has no distortion
        fx = lp["fisheye_params"]["camera_matrix"][0][0]
        fy = lp["fisheye_params"]["camera_matrix"][1][1]
        # fx = focal/sw*w = 20/13.2*3840 = 5818; fy = 20/8.8*2160 = 4909
        assert fx == pytest.approx(20.0 / 13.2 * 3840, rel=1e-3)
        assert fy == pytest.approx(20.0 / 8.8 * 2160, rel=1e-3)

    def test_no_sensor_dimensions_and_no_focal_is_none(self):
        md = FileMetadata(detected_source="Sony")
        tags = _imager_tags()
        tags[S.T_LENS_DIST_DATA] = _lens_distortion(
            9_000_000, 6_012_884, 200.0, [])
        # empty coeffs enter the fallback; without 0x8104/0x8105 (sw/sh = 0)
        # and without a 0x8005 focal tag nothing can be built
        S.init_lens_profile(md, tags, (3840, 2160), 0.0, None)
        assert md.lens_profile is None


# ---------------------------------------------------------------------------
# get_mesh_correction
# ---------------------------------------------------------------------------


class TestMeshCorrection:
    def test_forward_mesh_hits_grid_values_and_inverse_roundtrips(self):
        tags = _imager_tags()
        tags[S.T_MESH_DATA] = _mesh()
        tags[S.T_FPD_DATA] = _focal_plane([(0, 0)] * 8)

        out = S.get_mesh_correction(tags, {})
        assert out is not None
        mesh, inv = out
        assert mesh[0] == float(len(mesh) - 20)  # focal section offset
        assert (mesh[1], mesh[2]) == (9.0, 9.0)

        size = (5028.0, 2828.0)
        # (0,0) is a grid node; node values carry the raw-mesh correction:
        # xs[80] = 80, ys[80] = -80
        x0, y0 = interpolate_mesh(0.0, 0.0, size, mesh)
        assert x0 == pytest.approx(80.0, abs=2.0)
        assert y0 == pytest.approx(-80.0, abs=2.0)

        # definitional inverse check: forward(inverse(node)) == node
        p = interpolate_mesh(0.0, 0.0, size, inv)
        f = interpolate_mesh(p[0], p[1], size, mesh)
        assert f[0] == pytest.approx(0.0, abs=2.0)
        assert f[1] == pytest.approx(0.0, abs=2.0)

    def test_zero_mesh_with_focal_plane_emits_header_plus_focal(self):
        tags = _imager_tags()
        tags[S.T_MESH_DATA] = _mesh(value_scale=0)  # all-zero corrections
        tags[S.T_FPD_DATA] = _focal_plane([(100, -100)] * 8)
        out = S.get_mesh_correction(tags, {})
        assert out is not None
        mesh, _ = out
        # has_any_mesh_value is False: no mesh body, no coefficient blocks
        assert len(mesh) == 9 + 20
        # the FPD section decodes: [count=8, unk1, unk2, scale, pairs/32768]
        assert mesh[9] == 8.0
        assert mesh[12] == pytest.approx(32768.0 / 8.0)  # scale = 32768/cc

    def test_nothing_usable_is_none(self):
        tags = _imager_tags()
        tags[S.T_MESH_DATA] = _mesh(value_scale=0)
        assert S.get_mesh_correction(tags, {}) is None

    def test_cache_hit(self):
        tags = _imager_tags()
        tags[S.T_MESH_DATA] = _mesh()
        tags[S.T_FPD_DATA] = _focal_plane([(0, 0)] * 8)
        cache = {}
        first = S.get_mesh_correction(tags, cache)
        second = S.get_mesh_correction(tags, cache)
        assert first is not None and second is first


# ---------------------------------------------------------------------------
# End-to-end wiring through _parse_sony (synthetic rtmd mp4)
# ---------------------------------------------------------------------------


class TestParseSonyWiring:
    def test_deep_fields_populated_from_synthetic_file(self, tmp_path):
        """One rtmd sample carrying the imager/timing/OIS tags: the parse
        must produce the time offset, the OIS spline and the rescaled
        readout time — without any real footage."""
        from pygyroflow.telemetry.parser import _parse_sony

        gyro = struct.pack(">ii", 200, 6) + struct.pack(
            f">{200 * 3}h", *([100, 0, 0] * 200))
        tlvs = b"".join(
            struct.pack(">HH", tag, len(p)) + p for tag, p in [
                (0xE43B, gyro),                     # gyro rows
                (0xE435, struct.pack(">i", 2000)),  # gyro frequency
                (0xE436, struct.pack(">i", 1000000)),
                (0xE437, struct.pack(">i", -8909)),
                (0xE40C, struct.pack(">i", 34659)),
                (0xE40D, struct.pack(">i", 3598)),
                (0xE40E, struct.pack(">i", 14297)),
                (0xE405, struct.pack(">HH", 5028, 2828)),
                (0xE407, struct.pack(">HH", 2400, 2400)),
                (0xE409, struct.pack(">II", 2, 1)),
                (0xE40A, struct.pack(">II", 5024, 2826)),
                (0xE416, _ois([(i * 500, 100 * i, -50 * i, 0)
                               for i in range(16)])),
            ])
        chunk = struct.pack(">HH", 0x001C, 0) + b"\x00" * 24 + tlvs
        data = _build_rtmd_mp4([chunk], stts_delta_ms=100.0)

        md = _parse_sony(data, 29.97, (3840, 2160))
        assert md.detected_source == "Sony"
        assert len(md.raw_imu) == 200
        # deep pass
        assert len(md.per_frame_time_offsets) == 1
        assert md.per_frame_time_offsets[0] != 0.0
        assert len(md.camera_stab_data) == 1
        cs = md.camera_stab_data[0]
        assert len(cs.ois_spline) > 0
        assert cs.sensor_size == (5028, 2828)
        # readout 14.297 ms rescaled by gyro rate vs the IMU-grid rate:
        # the synthetic grid lands at ≈2000.05 Hz, so the ratio ≈ 1.000025
        assert md.frame_readout_time == pytest.approx(14.297, rel=1e-4)


def _mp4_box(fourcc: str, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + fourcc.encode() + payload


def _build_rtmd_mp4(chunks: list[bytes], stts_delta_ms: float | None = None) -> bytes:
    """Minimal mp4 with an rtmd track (same structure as the Sony fixture
    builder in test_metadata_fields.py)."""

    def hdlr_plain():
        return _mp4_box("hdlr", b"\x00" * 4 + b"mhlr" + b"meta" + b"\x00" * 12)

    def make_moov(sample_offset: int, sample_size: int) -> bytes:
        stsd = _mp4_box("stsd", struct.pack(">II", 0, 1)
                        + struct.pack(">I", 16) + b"rtmd" + b"\x00" * 8)
        stco = _mp4_box("stco", struct.pack(">II", 0, 1)
                        + struct.pack(">I", sample_offset))
        stsz = _mp4_box("stsz", struct.pack(">III", 0, 0, 1)
                        + struct.pack(">I", sample_size))
        stsc = _mp4_box("stsc", struct.pack(">II", 0, 1)
                        + struct.pack(">III", 1, 1, 1))
        boxes = stsd + stco + stsz + stsc
        if stts_delta_ms is not None:
            boxes += _mp4_box("stts", struct.pack(">II", 0, 1)
                              + struct.pack(">II", 1, int(stts_delta_ms)))
        mdhd = _mp4_box("mdhd", struct.pack(">IIIII", 0, 0, 0, 1000, 30))
        minf = _mp4_box("minf", _mp4_box("stbl", boxes))
        trak = _mp4_box("trak", _mp4_box("tkhd", b"\x00" * 84)
                        + _mp4_box("mdia", mdhd + hdlr_plain() + minf))
        return _mp4_box("moov", _mp4_box("mvhd", b"\x00" * 96) + trak)

    moov = make_moov(0, len(chunks[0]))
    moov = make_moov(len(moov), len(chunks[0]))
    return moov + b"".join(chunks)
