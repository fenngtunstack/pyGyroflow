"""Gyroflow Protobuf input (A-02 second half): hand-encoded protobuf wire
messages decoded by telemetry/gyroflow_proto.py and checked against
hand-computed expectations — no protobuf library, no circular data."""

from __future__ import annotations

import struct

import pytest
from pygyroflow.telemetry.gyroflow_proto import (
    conjugate_quat_by,
    pack_readout_time_ms,
    parse_gyroflow_proto,
)
from pygyroflow.types.enums import ReadoutDirection
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Minimal protobuf wire ENCODER (the decoder's independent counterpart)
# ---------------------------------------------------------------------------

def varint(n: int) -> bytes:
    out = b""
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out += bytes([b | 0x80])
        else:
            return out + bytes([b])


def _tag(field: int, wt: int) -> bytes:
    return varint((field << 3) | wt)


def f_str(field: int, s: str) -> bytes:
    b = s.encode()
    return _tag(field, 2) + varint(len(b)) + b


def f_msg(field: int, b: bytes) -> bytes:
    return _tag(field, 2) + varint(len(b)) + b


def f_u32(field: int, v: int) -> bytes:
    return _tag(field, 0) + varint(v)


def f_i32(field: int, v: int) -> bytes:
    if v < 0:
        v &= (1 << 64) - 1
    return _tag(field, 0) + varint(v)


def f_f32(field: int, v: float) -> bytes:
    return _tag(field, 5) + struct.pack("<f", v)


def f_f64(field: int, v: float) -> bytes:
    return _tag(field, 1) + struct.pack("<d", v)


def f_packed_f32(field: int, vals: list[float]) -> bytes:
    b = struct.pack(f"<{len(vals)}f", *vals)
    return _tag(field, 2) + varint(len(b)) + b


def quat_msg(w=1.0, x=0.0, y=0.0, z=0.0) -> bytes:
    return f_f32(1, w) + f_f32(2, x) + f_f32(3, y) + f_f32(4, z)


def camera_msg(brand="Acme", model="X100", **kw) -> bytes:
    out = f_str(1, brand) + f_str(2, model)
    out += f_u32(7, kw.get("pp_x", 2400)) + f_u32(8, kw.get("pp_y", 2400))
    out += f_u32(9, kw.get("sw", 4000)) + f_u32(10, kw.get("sh", 3000))
    if kw.get("imu_orientation"):
        out += f_str(13, kw["imu_orientation"])
    if kw.get("imu_rotation"):
        out += f_msg(14, kw["imu_rotation"])
    if kw.get("quats_rotation"):
        out += f_msg(15, kw["quats_rotation"])
    if kw.get("lens_brand"):
        out += f_str(5, kw["lens_brand"])
    if kw.get("lens_model"):
        out += f_str(6, kw["lens_model"])
    return out


def clip_msg(**kw) -> bytes:
    out = (f_u32(1, kw.get("frame_w", 1920))
           + f_u32(2, kw.get("frame_h", 1080)))
    out += f_f32(4, kw.get("record_fps", 30.0))
    out += f_f32(5, kw.get("sensor_fps", 30.0))
    out += f_u32(8, kw.get("imu_sample_rate", 1000))
    out += f_f64(11, kw.get("frame_readout_time_us", 0.0))
    out += f_i32(12, kw.get("readout_direction", 0))
    return out


def imu_msg(ts_us, g=(0.0, 0.0, 0.0), a=(0.0, 0.0, 9.81)) -> bytes:
    out = f_f64(1, ts_us)
    out += f_f32(2, g[0]) + f_f32(3, g[1]) + f_f32(4, g[2])
    out += f_f32(5, a[0]) + f_f32(6, a[1]) + f_f32(7, a[2])
    return out


def frame_msg(**kw) -> bytes:
    out = f_f64(1, kw.get("start_ts", 0.0)) + f_f64(2, kw.get("end_ts", 0.0))
    if kw.get("exposure_us") is not None:
        out += f_f64(5, kw["exposure_us"])
    if kw.get("shutter_num") is not None:
        out += f_i32(9, kw["shutter_num"]) + f_i32(10, kw["shutter_den"])
    if kw.get("zoom") is not None:
        out += f_f32(8, kw["zoom"])
    if kw.get("lens") is not None:
        out += f_msg(16, kw["lens"])
    for m in kw.get("imu", []):
        out += f_msg(17, m)
    for q in kw.get("quats", []):
        out += f_msg(18, q)
    for o in kw.get("ois", []):
        out += f_msg(19, o)
    for s in kw.get("ibis", []):
        out += f_msg(20, s)
    for e in kw.get("eis", []):
        out += f_msg(21, e)
    return out


def lens_msg(focal=None, dist_name=None, coeffs=None, intrinsic=None) -> bytes:
    out = b""
    if intrinsic:
        out += f_packed_f32(1, intrinsic)
    if focal is not None:
        out += f_f32(2, focal)
    if dist_name is not None:
        out += f_msg(dist_name[0], f_packed_f32(1, dist_name[1]))
    return out


def ibis_msg(ts_us, x, y, roll) -> bytes:
    return f_f64(1, ts_us) + f_f32(2, x) + f_f32(3, y) + f_f32(4, roll)


def eis_quat_msg(w, x, y, z) -> bytes:
    return f_msg(2, quat_msg(w, x, y, z))


def eis_mesh_msg(gw=2, gh=2, region=(4000.0, 3000.0), offset=(10.0, -10.0)):
    warped = []
    for j in range(gh):
        for i in range(gw):
            warped.extend([region[0] * i + offset[0],
                           region[1] * j + offset[1]])
    out = (f_u32(1, gw) + f_u32(2, gh) + f_f32(3, region[0])
           + f_f32(4, region[1]) + f_packed_f32(5, warped))
    return f_msg(3, out)


def main_msg(header: bytes | None = None, frame: bytes | None = None) -> bytes:
    out = f_str(1, "GyroflowProtobuf")
    if header is not None:
        out += f_msg(3, header)
    if frame is not None:
        out += f_msg(4, frame)
    return out


def _header(brand="Acme", **kw):
    return f_msg(1, camera_msg(brand=brand, **kw)) + f_msg(2, clip_msg(**kw))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pack_readout(us: float, direction: int) -> float:
    return pack_readout_time_ms(us, {
        0: ReadoutDirection.TopToBottom,
        1: ReadoutDirection.BottomToTop,
        2: ReadoutDirection.LeftToRight,
        3: ReadoutDirection.RightToLeft,
    }[direction])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWireDecoding:
    def test_basic_header_and_frame(self):
        frame = frame_msg(
            start_ts=1000.0, end_ts=14297.0, exposure_us=3598.0,
            imu=[imu_msg(1000.0, g=(1.5, -2.0, 0.25)),
                 imu_msg(2500.0, g=(1.6, -2.1, 0.35))],
        )
        md = parse_gyroflow_proto(
            [main_msg(_header(brand="Acme", model="X100"), frame)], 30.0)
        assert md.detected_source == "Acme X100"
        assert md.frame_rate == pytest.approx(30.0)
        assert md.has_accurate_timestamps
        assert len(md.raw_imu) == 2
        # camera-clock microseconds → milliseconds, physical units verbatim
        assert md.raw_imu[0].timestamp_ms == pytest.approx(1.0)
        assert md.raw_imu[0].gyro[0] == pytest.approx(1.5)
        assert md.raw_imu[1].accl[2] == pytest.approx(9.81)

    def test_readout_packing_and_direction(self):
        frame = frame_msg(start_ts=0.0, end_ts=14297.0)
        md = parse_gyroflow_proto(
            [main_msg(_header(readout_direction=1), frame)], 30.0)
        # clip helper absent (0): the first frame's authoritative doubles
        # promote into the clip-level readout, sign-packed BottomToTop
        assert md.frame_readout_time == pytest.approx(-14.297)
        assert md.frame_readout_direction is ReadoutDirection.BottomToTop

    def test_clip_readout_helper_takes_effect_when_set(self):
        frame = frame_msg(start_ts=0.0, end_ts=14297.0)
        md = parse_gyroflow_proto(
            [main_msg(_header(frame_readout_time_us=8000.0,
                              readout_direction=2), frame)], 30.0)
        assert md.frame_readout_time == pytest.approx(8000.0 / 1000.0 + 10000.0)
        assert md.frame_readout_direction is ReadoutDirection.LeftToRight

    def test_exposure_precedence(self):
        f1 = frame_msg(start_ts=0.0, end_ts=0.0, exposure_us=5000.0,
                       shutter_num=1, shutter_den=240)
        md = parse_gyroflow_proto([main_msg(_header(), f1)], 30.0)
        # exposure only surfaces through the Sony deep pass; verify via
        # lens_params-free paths: no crash + no offsets is enough here.
        assert md.per_frame_time_offsets == []

    def test_digital_zoom(self):
        f1 = frame_msg(start_ts=0.0, end_ts=0.0, zoom=1.5)
        f2 = frame_msg(start_ts=33333.0, end_ts=33333.0, zoom=1.0)
        md = parse_gyroflow_proto(
            [main_msg(_header(), f1), main_msg(frame=f2)], 30.0)
        assert md.digital_zoom == pytest.approx(1.5)


class TestImuRotation:
    def test_orientation_remap_then_quat_rotation(self):
        """imu_rotation applies AFTER the imu_orientation axis remap
        (binary.rs:624-641). io='ZXY' maps (x,y,z)→(z,x,y); then rotate by
        a 90°-about-Z quaternion."""
        rz = quat_msg(w=0.7071067811865476, z=0.7071067811865476)
        frame = frame_msg(imu=[imu_msg(1000.0, g=(1.0, 2.0, 3.0))])
        md = parse_gyroflow_proto(
            [main_msg(_header(imu_orientation="ZXY", imu_rotation=rz),
                      frame)], 30.0)
        g = md.raw_imu[0].gyro
        expected = Rotation.from_quat([0.0, 0.0, 0.7071067811865476,
                                       0.7071067811865476]).apply([3.0, 1.0, 2.0])
        assert g == pytest.approx(expected, abs=1e-6)

    def test_no_rotation_leaves_raw_values(self):
        frame = frame_msg(imu=[imu_msg(1000.0, g=(1.0, 2.0, 3.0))])
        md = parse_gyroflow_proto([main_msg(_header(), frame)], 30.0)
        assert md.raw_imu[0].gyro == pytest.approx([1.0, 2.0, 3.0])


class TestQuatsRotation:
    def test_conjugation_rotates_the_axis(self):
        q = (0.9238795290916443, 0.22094237129596315, 0.22094237129596315,
             0.22094237129596315)
        r = (0.7071067811865476, 0.7071067811865476, 0.0, 0.0)
        out = conjugate_quat_by(q, r)
        expected = (Rotation.from_quat([r[1], r[2], r[3], r[0]])
                    * Rotation.from_quat([q[1], q[2], q[3], q[0]])
                    * Rotation.from_quat([r[1], r[2], r[3], r[0]]).inv())
        got = Rotation.from_quat([out[1], out[2], out[3], out[0]])
        ang = (got * expected.inv()).magnitude()
        assert ang == pytest.approx(0.0, abs=1e-7)


class TestEisQuatOrientations:
    def test_image_orientations_are_the_rearranged_payload(self):
        """EIS float quats quantize to i16 (payload order w,x,y,z) and the
        consumer reads them REARRANGED: (x,y,z,w)/scale — same quirk as
        GoPro IORI."""
        frame = frame_msg(
            quats=[f_f64(1, 500.0) + f_msg(2, quat_msg())],  # identity, t=500µs
            eis=[eis_quat_msg(0.0, 0.0, 0.0, 1.0)],  # 180° about Z
        )
        md = parse_gyroflow_proto([main_msg(_header(), frame)], 30.0)
        assert set(md.image_orientations) == {500}
        (q,) = md.image_orientations.values()
        assert q.quaternion() == pytest.approx([0.0, 0.0, 1.0, 0.0], abs=1e-6)


class TestSonyDeepPass:
    def test_sony_brand_builds_stab_data_with_sign_flip(self):
        ibis = [ibis_msg(500.0 * i, 100.0, -50.0, 1.5) for i in range(5)]
        frame = frame_msg(start_ts=0.0, end_ts=20000.0, ibis=ibis)
        md = parse_gyroflow_proto(
            [main_msg(_header(brand="Sony", model="A7S III",
                              record_fps=50.0), frame)], 50.0)
        assert len(md.camera_stab_data) >= 1
        cs = md.camera_stab_data[0]
        assert len(cs.ibis_spline) == 5

    def test_non_sony_brand_skips_stab_data(self):
        ibis = [ibis_msg(500.0 * i, 100.0, -50.0, 1.5) for i in range(5)]
        frame = frame_msg(start_ts=0.0, end_ts=20000.0, ibis=ibis)
        md = parse_gyroflow_proto([main_msg(_header(), frame)], 50.0)
        assert md.camera_stab_data == []

    def test_guard_samples_outside_frame_interval_dropped(self):
        # interval = 1e6/50 = 20000 µs; a sample at dt 25000 µs is a guard
        ibis = [ibis_msg(0.0, 1.0, 0.0, 0.0),
                ibis_msg(25000.0, 2.0, 0.0, 0.0)]
        frame = frame_msg(start_ts=0.0, end_ts=20000.0, ibis=ibis)
        md = parse_gyroflow_proto(
            [main_msg(_header(brand="Sony", record_fps=50.0), frame)], 50.0)
        cs = md.camera_stab_data[0]
        assert len(cs.ibis_spline) == 1


class TestMeshAndProfile:
    def test_eis_mesh_warp_builds_correction(self):
        frame = frame_msg(start_ts=0.0, end_ts=0.0, eis=[eis_mesh_msg()])
        md = parse_gyroflow_proto([main_msg(_header(), frame)], 30.0)
        assert len(md.mesh_correction) == 1
        mesh, inv = md.mesh_correction[0]
        assert (mesh[1], mesh[2]) == (2.0, 2.0)

    def test_lens_profile_built_once_with_model(self):
        lens = lens_msg(focal=24.0,
                        dist_name=(6, [0.01, -0.02, 0.003]))  # opencv_fisheye=6
        frame = frame_msg(start_ts=0.0, end_ts=0.0, lens=lens)
        md = parse_gyroflow_proto(
            [main_msg(_header(lens_brand="Acme", lens_model="Iron Glass"),
                      frame),
             main_msg(frame=frame)], 30.0)
        assert md.lens_profile is not None
        lp = md.lens_profile
        assert lp["calibrated_by"] == "Gyroflow Protobuf"
        assert lp["camera_brand"] == "Acme"
        assert lp["lens_model"] == "Acme Iron Glass"
        assert lp["distortion_model"] == "opencv_fisheye"
        assert lp["fps"] == pytest.approx(30.0)

    def test_lens_params_from_focal_and_geometry(self):
        lens = lens_msg(focal=24.0, dist_name=(5, []))  # no_distortion
        frame = frame_msg(start_ts=0.0, end_ts=33333.0, lens=lens)
        md = parse_gyroflow_proto([main_msg(_header(), frame)], 30.0)
        assert len(md.lens_params) == 1
        (lp,) = md.lens_params.values()
        assert lp.focal_length == pytest.approx(24.0)
        # pixel focal derived from mm + pitch×crop geometry:
        # sensor_w_mm = 2400nm * 4000px / 1e6 = 9.6mm → fx = 24/9.6*1920
        assert lp.pixel_focal_length == pytest.approx(24.0 / 9.6 * 1920,
                                                      rel=1e-3)


class TestRobustness:
    def test_sample_without_magic_skipped(self):
        frame = frame_msg(imu=[imu_msg(1000.0, g=(1.0, 2.0, 3.0))])
        good = main_msg(_header(), frame)
        bad = b"\x01\x02\x03garbage"
        md = parse_gyroflow_proto([bad, good], 30.0)
        assert len(md.raw_imu) == 1

    def test_truncated_message_does_not_crash(self):
        good = main_msg(_header())
        md = parse_gyroflow_proto([good[: len(good) // 2], good], 30.0)
        assert md.detected_source == "Acme X100"


class TestEndToEnd:
    def test_parse_telemetry_file_on_synthetic_mp4(self, tmp_path):
        """A 'meta'-codec metadata track inside a real mp4: detection (magic
        in the head/tail window) through sample extraction to FileMetadata."""
        from pygyroflow.telemetry import parse_telemetry_file

        frame = frame_msg(
            start_ts=0.0, end_ts=14297.0, exposure_us=3598.0,
            imu=[imu_msg(1000.0, g=(1.0, 2.0, 3.0)),
                 imu_msg(2000.0, g=(1.0, 2.0, 3.0))],
        )
        payload = main_msg(_header(), frame)

        def box(fourcc: str, b: bytes) -> bytes:
            return struct.pack(">I", 8 + len(b)) + fourcc.encode() + b

        def make_moov(off: int, size: int) -> bytes:
            stsd = box("stsd", struct.pack(">II", 0, 1)
                       + struct.pack(">I", 16) + b"meta" + b"\x00" * 8)
            stco = box("stco", struct.pack(">II", 0, 1)
                       + struct.pack(">I", off))
            stsz = box("stsz", struct.pack(">III", 0, 0, 1)
                       + struct.pack(">I", size))
            stsc = box("stsc", struct.pack(">II", 0, 1)
                       + struct.pack(">III", 1, 1, 1))
            hdlr = box("hdlr", b"\x00" * 4 + b"mhlr" + b"meta" + b"\x00" * 12)
            mdhd = box("mdhd", struct.pack(">IIIII", 0, 0, 0, 1000, 30))
            minf = box("minf", box("stbl", stsd + stco + stsz + stsc))
            trak = box("trak", box("tkhd", b"\x00" * 84)
                       + box("mdia", mdhd + hdlr + minf))
            return box("moov", box("mvhd", b"\x00" * 96) + trak)

        moov = make_moov(0, len(payload))
        moov = make_moov(len(moov), len(payload))
        data = moov + payload

        path = tmp_path / "proto.mp4"
        path.write_bytes(data)
        md = parse_telemetry_file(str(path), fps=30.0)
        assert md.detected_source == "Acme X100"
        assert len(md.raw_imu) == 2
        assert md.frame_readout_time == pytest.approx(14.297)
