"""A-05: wiring the FileMetadata fields parsers never wrote.

Upstream fills these in gyro_source/mod.rs:137-501 from parser tag maps:
GoPro IORI -> image_orientations, GoPro GRAV -> gravity_vectors (rotated
by IORI), GoPro DZST/DZMX -> digital_zoom, frame_readout_direction from
the signed readout time, Insta360 exposure/TimeMap records ->
per_frame_time_offsets. Sony lens_positions was already written.
"""

from __future__ import annotations

import struct

import pytest

from pygyroflow.telemetry.parser import (
    _parse_gopro,
    _parse_insta360,
    _parse_sony,
    _readout_direction_from,
)
from pygyroflow.types.enums import ReadoutDirection

_MAGIC = b"8db42d694ccc418790edff439fe026bf"


# ---------------------------------------------------------------------------
# Synthetic GoPro gpmd mp4 (structure mirrors tests/test_e2e.py's builder,
# extended with GRAV / CORI / IORI / DZST streams)
# ---------------------------------------------------------------------------

def _gpmf_klv(fourcc: str, type_byte: int, payload: bytes, struct_size: int = 0,
              repeat: int = 1) -> bytes:
    payload = payload + b"\x00" * (-len(payload) % 4)
    if struct_size == 0:
        if type_byte == 0:
            struct_size, repeat = 1, len(payload)
        else:
            struct_size, repeat = max(len(payload), 1), 1
    return (fourcc.encode() + bytes([type_byte, struct_size])
            + struct.pack(">H", repeat) + payload)


def _mp4_box(fourcc: str, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + fourcc.encode() + payload


def _quat_stream(fourcc: str, name: str,
                 raw: list[tuple[int, int, int, int]]) -> bytes:
    """An orientation STRM: STNM + SCAL(1000) + the i16 quaternion array."""
    payload = struct.pack(f">{len(raw) * 4}h", *[v for q in raw for v in q])
    strm = b"".join([
        _gpmf_klv("STNM", ord("c"), name.encode() + b"\x00" * 4),
        _gpmf_klv("SCAL", ord("s"), struct.pack(">h", 1000)),
        _gpmf_klv(fourcc, ord("s"), payload, struct_size=8, repeat=len(raw)),
    ])
    return _gpmf_klv("STRM", 0, strm)


def _build_gopro_mp4(dzst: int | None = None, dzmx: float | None = None,
                     with_grav: bool = False, grav_raw=None,
                     with_quats: bool = False) -> bytes:
    gyro = _gpmf_klv("GYRO", ord("s"),
                     struct.pack(">3h", 100, 0, 0), struct_size=6, repeat=1)
    accl = _gpmf_klv("ACCL", ord("s"),
                     struct.pack(">3h", 0, 0, 1000), struct_size=6, repeat=1)
    strm_parts = [
        _gpmf_klv("STNM", ord("c"), b"Angular velocity\x00\x00"),
        _gpmf_klv("SCAL", ord("s"), struct.pack(">h", 100)),
        gyro, accl,
    ]
    if with_grav:
        raw = grav_raw if grav_raw is not None else (32767, 0, 0)
        # GRAV is its own STRM with no SCAL — upstream defaults the gravity
        # group's scale to 32767 (mod.rs:270).
        grav_strm = _gpmf_klv("STRM", 0, b"".join([
            _gpmf_klv("STNM", ord("c"), b"Gravity vector\x00\x00\x00"),
            _gpmf_klv("GRAV", ord("s"), struct.pack(">3h", *raw),
                      struct_size=6, repeat=1),
        ]))
    else:
        grav_strm = b""
    strm = _gpmf_klv("STRM", 0, b"".join(strm_parts))

    devc_children = [strm, grav_strm, _gpmf_klv("DVNM", ord("c"), b"HERO12\x00\x00")]
    if with_quats:
        # CORI = identity; IORI raw (0,0,0,1000): the mod-variant (b,c,d,a)
        # / 1000 = (0,0,1,0) — a 180 deg rotation about Y.
        devc_children.append(_quat_stream("CORI", "CameraOrientation",
                                          [(1000, 0, 0, 0)]))
        devc_children.append(_quat_stream("IORI", "ImageOrientation",
                                          [(0, 0, 0, 1000)]))
    if dzst is not None:
        devc_children.append(_gpmf_klv("DZST", ord("L"), struct.pack(">I", dzst)))
    if dzmx is not None:
        devc_children.append(_gpmf_klv("DZMX", ord("F"), struct.pack(">f", dzmx)))
    devc = _gpmf_klv("DEVC", 0, b"".join(devc_children) + b"DVC1\x00\x00\x00")

    def make_moov(sample_offset: int, sample_size: int) -> bytes:
        stsd = _mp4_box("stsd", struct.pack(">II", 0, 1)
                        + struct.pack(">I", 16) + b"gpmd" + b"\x00" * 8)
        stco = _mp4_box("stco", struct.pack(">II", 0, 1)
                        + struct.pack(">I", sample_offset))
        stsz = _mp4_box("stsz", struct.pack(">III", 0, 0, 1)
                        + struct.pack(">I", sample_size))
        stsc = _mp4_box("stsc", struct.pack(">II", 0, 1) + struct.pack(">III", 1, 1, 1))
        hdlr = _mp4_box("hdlr", b"\x00" * 4 + b"mhlr" + b"meta" + b"\x00" * 12
                        + bytes([11]) + b"GoPro MET  ")
        mdhd = _mp4_box("mdhd", struct.pack(">IIIII", 0, 0, 0, 1000, 30))
        minf = _mp4_box("minf", _mp4_box("stbl", stsd + stco + stsz + stsc))
        trak = _mp4_box("trak", _mp4_box("tkhd", b"\x00" * 84)
                        + _mp4_box("mdia", mdhd + hdlr + minf))
        return _mp4_box("moov", _mp4_box("mvhd", b"\x00" * 96) + trak)

    moov = make_moov(0, len(devc))
    moov = make_moov(len(moov), len(devc))
    return moov + devc


# ---------------------------------------------------------------------------
# frame_readout_direction (mod.rs:398-405)
# ---------------------------------------------------------------------------

class TestReadoutDirection:
    @pytest.mark.parametrize("fr,expected", [
        (15.0, ReadoutDirection.TopToBottom),
        (None, ReadoutDirection.TopToBottom),
        (-15.0, ReadoutDirection.BottomToTop),
        (10015.0, ReadoutDirection.LeftToRight),
        (-10015.0, ReadoutDirection.RightToLeft),
    ])
    def test_the_sign_and_sentinel_table(self, fr, expected):
        assert _readout_direction_from(fr) is expected

    def test_gopro_srot_is_top_to_bottom(self):
        data = _build_gopro_mp4() + _gpmf_klv("SROT", ord("f"),
                                              struct.pack(">f", 12.5))
        md = _parse_gopro(data, 30.0)
        assert md.frame_readout_time == pytest.approx(12.5)
        assert md.frame_readout_direction is ReadoutDirection.TopToBottom

    def test_sony_signed_readout_flips_direction(self):
        """0xe40e is i32 µs (rtmd_tags.rs:414): a negative value must give
        BottomToTop, not just a negative frame_readout_time."""
        tlv = struct.pack(">HHi", 0xE40E, 4, -8000)  # -8 ms
        chunk = struct.pack(">HH", 0x001C, 0) + b"\x00" * 24 + tlv
        data = _build_rtmd_mp4([chunk])
        md = _parse_sony(data, 30.0)
        assert md.frame_readout_time == pytest.approx(-8.0)
        assert md.frame_readout_direction is ReadoutDirection.BottomToTop


def _build_rtmd_mp4(chunks: list[bytes]) -> bytes:
    """Minimal mp4 with an rtmd track: one chunk = one sample, no stts."""

    def hdlr_plain():
        return _mp4_box("hdlr", b"\x00" * 4 + b"mhlr" + b"meta" + b"\x00" * 12)

    def make_moov(sample_offset: int, sample_size: int) -> bytes:
        stsd = _mp4_box("stsd", struct.pack(">II", 0, 1)
                        + struct.pack(">I", 16) + b"rtmd" + b"\x00" * 8)
        stco = _mp4_box("stco", struct.pack(">II", 0, 1)
                        + struct.pack(">I", sample_offset))
        stsz = _mp4_box("stsz", struct.pack(">III", 0, 0, 1)
                        + struct.pack(">I", sample_size))
        stsc = _mp4_box("stsc", struct.pack(">II", 0, 1) + struct.pack(">III", 1, 1, 1))
        mdhd = _mp4_box("mdhd", struct.pack(">IIIII", 0, 0, 0, 1000, 30))
        minf = _mp4_box("minf", _mp4_box("stbl", stsd + stco + stsz + stsc))
        trak = _mp4_box("trak", _mp4_box("tkhd", b"\x00" * 84)
                        + _mp4_box("mdia", mdhd + hdlr_plain() + minf))
        return _mp4_box("moov", _mp4_box("mvhd", b"\x00" * 96) + trak)

    moov = make_moov(0, len(chunks[0]))
    moov = make_moov(len(moov), len(chunks[0]))
    return moov + b"".join(chunks)


# ---------------------------------------------------------------------------
# GoPro: image_orientations, gravity_vectors, digital_zoom
# ---------------------------------------------------------------------------

class TestGoProImageOrientations:
    def test_iori_lands_rearranged_on_quat_timestamps(self):
        """mod.rs:314-343 builds the IORI quat via
        Quaternion::from_vector(Vector4::new(v.x, v.y, v.z, v.w)) — i.e.
        payload (a,b,c,d) becomes (w,x,y,z) = (b,c,d,a) — and zips it onto
        the combined-quaternion timestamps. The -x flip of the CORI*IORI
        path does NOT apply here."""
        data = _build_gopro_mp4(with_quats=True)
        md = _parse_gopro(data, 30.0)
        assert md.quaternions and md.image_orientations
        assert set(md.image_orientations) == set(md.quaternions)
        (q,) = md.image_orientations.values()
        assert q.quaternion() == pytest.approx([0.0, 0.0, 1.0, 0.0])

    def test_identity_iori_gives_identity_image_orientation(self):
        """Raw (1000,0,0,0): rearranged (0,0,0,1) is NOT identity — the
        quirk makes the identity IORI encode as (0,0,0,1), not (1,0,0,0).
        Pinned to document the upstream behavior verbatim."""
        from pygyroflow.telemetry.parser import _parse_gpmf_orientation_chunk

        chunk = _gpmf_klv("DEVC", 0, _quat_stream(
            "IORI", "ImageOrientation", [(1000, 0, 0, 0)]))
        _, _, iori_mod = _parse_gpmf_orientation_chunk(chunk)
        assert iori_mod == [[0.0, 0.0, 0.0, 1.0]]


class TestGoProGravity:
    def test_grav_scaled_by_32767_default_and_rotated_by_iori(self):
        """GRAV with no SCAL divides by 32767 (mod.rs:270), pairs 1:1 with
        the quats and is rotated by the (rearranged) IORI: 180 deg about Y
        maps (1,0,0) to (-1,0,0)."""
        data = _build_gopro_mp4(with_grav=True, with_quats=True)
        md = _parse_gopro(data, 30.0)
        assert md.gravity_vectors
        (g,) = md.gravity_vectors.values()
        assert g == pytest.approx([-1.0, 0.0, 0.0])

    def test_all_zero_grav_is_discarded(self):
        data = _build_gopro_mp4(with_grav=True, grav_raw=(0, 0, 0),
                                with_quats=True)
        md = _parse_gopro(data, 30.0)
        assert md.gravity_vectors is None

    def test_no_grav_stream_stays_none(self):
        md = _parse_gopro(_build_gopro_mp4(), 30.0)
        assert md.gravity_vectors is None


class TestGoProDigitalZoom:
    def test_dzst_over_dzmx(self):
        # mod.rs:262-266: 1 + (DZST/100) * (DZMX - 1), DZMX defaults 1.4
        data = _build_gopro_mp4(dzst=40, dzmx=2.0)
        md = _parse_gopro(data, 30.0)
        assert md.digital_zoom == pytest.approx(1.0 + 0.4 * 1.0)

    def test_dzmx_defaults_to_1_4(self):
        data = _build_gopro_mp4(dzst=50)
        md = _parse_gopro(data, 30.0)
        assert md.digital_zoom == pytest.approx(1.0 + 0.5 * 0.4)

    def test_zero_dzst_means_no_zoom(self):
        data = _build_gopro_mp4(dzst=0, dzmx=2.0)
        md = _parse_gopro(data, 30.0)
        assert md.digital_zoom is None


# ---------------------------------------------------------------------------
# Insta360: per_frame_time_offsets from exposure + TimeMap records
# ---------------------------------------------------------------------------

def _insta_record(rid: int, payload: bytes) -> bytes:
    return payload + bytes([0, rid]) + struct.pack("<I", len(payload))


def _build_insta360(extra_records: list[bytes]) -> bytes:
    records = b"".join(extra_records)
    # Header (72 bytes): 32 unknown + extra_size + version + 32-byte magic.
    # extra_size spans the whole file — upstream compares it against an
    # end-relative offset that includes the header itself.
    return records + (b"\x00" * 32 + struct.pack("<II", len(records) + 72, 1)
                      + _MAGIC)


class TestInsta360PerFrameOffsets:
    @staticmethod
    def _exposure_payload(ts_ms, shutter):
        return b"".join(struct.pack("<Qd", t, shutter) for t in ts_ms)

    def test_offsets_follow_the_upstream_walk(self):
        records = [
            _insta_record(1, b""),  # empty metadata: defaults everywhere
            _insta_record(4, self._exposure_payload([0, 1000, 2000], 0.01)),
        ]
        md = _parse_insta360(_build_insta360(records), 30.0)
        # mod.rs:470-492, fft=0, raw gyro off, no TimeMap:
        #   offset = -(v*500) - 0.9 - (video_ts - t)*1000 - 0
        v = 0.01
        assert md.per_frame_time_offsets[0] == pytest.approx(-v * 500 - 0.9)
        video_ts = 1.0 / 30.0
        assert md.per_frame_time_offsets[1] == pytest.approx(
            -v * 500 - 0.9 - (video_ts - 1.0) * 1000.0)
        assert len(md.per_frame_time_offsets) == 3

    def test_timemap_entry_joins_the_offset(self):
        # TimeMap: 4×u32 header + f64 + (t, v) pairs
        tm_payload = (struct.pack("<IIII", 0, 0, 0, 0)
                      + struct.pack("<d", 0.0) + struct.pack("<dd", 5.0, 1.0))
        records = [
            _insta_record(1, b""),
            _insta_record(4, self._exposure_payload([0], 0.02)),
            _insta_record(128, tm_payload),
        ]
        md = _parse_insta360(_build_insta360(records), 30.0)
        assert md.per_frame_time_offsets[0] == pytest.approx(
            -(0.02 * 500) - 0.9 - 0.0 - (5.0 - 1.0) - 0.0)

    def test_zero_fps_skips_the_walk(self):
        records = [
            _insta_record(1, b""),
            _insta_record(4, self._exposure_payload([0, 1000], 0.01)),
        ]
        md = _parse_insta360(_build_insta360(records), 0.0)
        assert md.per_frame_time_offsets == []
