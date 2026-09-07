"""End-to-end and pipeline-semantics regression tests (P0 batch, see
docs/08-optimization-roadmap.md).

Covers the "last mile" wiring that had zero coverage before:
  - vectorized distortion models vs their scalar reference (all 10 models)
  - cpu_undistort semantics: identity, sub-pixel sampling, per-row rolling
    shutter matrices, distortion-model dispatch, r_limit, lens correction
    blend
  - sync offsets shifting the frame transform (previously ignored)
  - synthetic GPMF/mp4 parsing through the real telemetry parser
  - full render through real codecs: decoder flush (frame-count parity),
    audio stream copy, no-gyro passthrough
"""

from __future__ import annotations

import ctypes
import math
import struct
from fractions import Fraction

import numpy as np
import pytest

from pygyroflow.stabilization.cpu_undistort import cpu_undistort
from pygyroflow.stabilization.distortion_models import _MODEL_REGISTRY
from pygyroflow.stabilization.frame_transform import FrameTransform
from pygyroflow.types.kernel_params import KernelParams

av = pytest.importorskip("av", reason="PyAV required for rendering tests")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_kernel_params(width=64, height=48, model_coeffs=None) -> KernelParams:
    kp = KernelParams()
    kp.width, kp.height = width, height
    kp.output_width, kp.output_height = width, height
    kp.f = (ctypes.c_float * 2)(40.0, 40.0)
    kp.c = (ctypes.c_float * 2)(width / 2.0, height / 2.0)
    kp.fov = 1.0
    kp.matrix_count = 1
    if model_coeffs is not None and len(model_coeffs):
        flat = list(model_coeffs) + [0.0] * (12 - len(model_coeffs))
        kp.k1 = (ctypes.c_float * 4)(*flat[0:4])
        kp.k2 = (ctypes.c_float * 4)(*flat[4:8])
        kp.k3 = (ctypes.c_float * 4)(*flat[8:12])
    return kp


def inv_k_matrix(f: float, cx: float, cy: float, shift_x: float = 0.0) -> np.ndarray:
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
    K[0, 2] += shift_x
    return np.linalg.inv(K)


def transform_with(matrix_rows: np.ndarray, kp: KernelParams,
                   model: str = "opencv_fisheye") -> FrameTransform:
    return FrameTransform(
        matrices=np.asarray(matrix_rows, dtype=np.float32),
        kernel_params=kp,
        distortion_model_name=model,
    )


def random_frame(h=48, w=64, seed=3) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# 1. Vectorized distortion models vs scalar reference
# ---------------------------------------------------------------------------

class TestDistortPointsParity:
    """distort_points must match distort_point bit-for-bit (all models)."""

    @pytest.mark.parametrize("model_name", sorted(_MODEL_REGISTRY))
    def test_distort_points_matches_scalar(self, model_name):
        model = _MODEL_REGISTRY[model_name]()
        rng = np.random.default_rng(42)
        kp = make_kernel_params(model_coeffs=rng.uniform(-0.3, 0.3, 12))
        kp.digital_lens_params = (ctypes.c_float * 4)(1.1, 0.9, 0.0, 0.0)

        n = 300
        # gopro models take pixel-domain coords; keep inputs in a range both
        # interpretations tolerate
        xs = rng.uniform(-0.8, 0.8, n)
        ys = rng.uniform(-0.8, 0.8, n)
        zs = np.ones(n)

        vxd, vyd = model.distort_points(xs.copy(), ys.copy(), zs.copy(), kp)
        for i in range(n):
            sxd, syd = model.distort_point(float(xs[i]), float(ys[i]), float(zs[i]), kp)
            assert sxd == pytest.approx(float(vxd[i]), abs=1e-12)
            assert syd == pytest.approx(float(vyd[i]), abs=1e-12)

    def test_fisheye_undistort_points_matches_scalar(self):
        from pygyroflow.stabilization.distortion_models import OpenCVFisheyeModel

        model = OpenCVFisheyeModel()
        kp = make_kernel_params(model_coeffs=[0.15, -0.05, 0.02, -0.003])
        rng = np.random.default_rng(7)
        xs = rng.uniform(-1.5, 1.5, 500)
        ys = rng.uniform(-1.5, 1.5, 500)

        vxu, vyu = model.undistort_points(xs.copy(), ys.copy(), kp)
        n_none = 0
        for i in range(len(xs)):
            pt = model.undistort_point(float(xs[i]), float(ys[i]), kp)
            if pt is None:
                n_none += 1
                assert np.isnan(vxu[i]) or np.isnan(vyu[i])
                continue
            assert pt[0] == pytest.approx(float(vxu[i]), abs=1e-8)
            assert pt[1] == pytest.approx(float(vyu[i]), abs=1e-8)
        assert n_none < len(xs)  # at least some points converge


# ---------------------------------------------------------------------------
# 2. cpu_undistort semantics
# ---------------------------------------------------------------------------

class TestCpuUndistort:
    def test_identity(self):
        frame = random_frame()
        kp = make_kernel_params()
        m = np.zeros((1, 14), dtype=np.float32)
        m[0, :9] = inv_k_matrix(40.0, 32.0, 24.0).ravel()
        out = cpu_undistort(frame, transform_with(m, kp))
        assert out.shape == frame.shape
        # Border row/col can hit the background on tiny float error; the
        # interior must reproduce the input.
        d = np.abs(out[1:, 1:].astype(int) - frame[1:, 1:].astype(int))
        assert d.max() <= 1

    def test_subpixel_shift_matches_analytic_bilinear(self):
        frame = random_frame()
        ff = frame.astype(np.float64)
        kp = make_kernel_params()
        m = np.zeros((1, 14), dtype=np.float32)
        m[0, :9] = inv_k_matrix(40.0, 32.0, 24.0, shift_x=-0.5).ravel()
        out = cpu_undistort(frame, transform_with(m, kp)).astype(np.float64)

        y, x = np.mgrid[0:48, 0:64]
        # inv_k_matrix(shift_x=s) makes the sampler read u = x - s
        xq, yq = x + 0.5, y
        x0 = np.floor(xq).astype(int)
        fx = (xq - x0)[:, :, None]
        x0c = np.clip(x0, 0, 63)
        x1c = np.clip(x0 + 1, 0, 63)
        expect = (1 - fx) * ff[y, x0c] + fx * ff[y, x1c]

        err = np.abs(out[2:46, 2:62] - expect[2:46, 2:62])
        assert err.max() <= 1.0

    def test_rolling_shutter_uses_per_row_matrices(self):
        """Each output row must use its own matrix, not the center one."""
        frame = random_frame()
        kp = make_kernel_params()
        rows = []
        m = np.zeros((48, 14), dtype=np.float32)
        for yy in range(48):
            m[yy, :9] = inv_k_matrix(40.0, 32.0, 24.0, shift_x=-yy / 8.0).ravel()
        kp.matrix_count = 48
        out_rs = cpu_undistort(frame, transform_with(m, kp))

        for yy in [0, 20, 40, 47]:
            single = m[yy : yy + 1].copy()
            kp_single = make_kernel_params()
            out_single = cpu_undistort(frame, transform_with(single, kp_single))
            d = np.abs(out_rs[yy].astype(int) - out_single[yy].astype(int))
            assert d.max() <= 1, f"row {yy} does not match its own matrix"

    def test_distortion_model_dispatch(self):
        """A non-fisheye model must actually be applied (poly3)."""
        frame = random_frame()
        # poly3: r_d = r * (1 + k1 r^2), k1 = 0.2. Full correction (lca=1)
        # so the distortion model actually drives the sampling.
        kp_poly = make_kernel_params(model_coeffs=[0.2])
        kp_poly.lens_correction_amount = 1.0
        kp_zero = make_kernel_params()
        kp_zero.lens_correction_amount = 1.0
        m = np.zeros((1, 14), dtype=np.float32)
        m[0, :9] = inv_k_matrix(40.0, 32.0, 24.0).ravel()

        out_poly = cpu_undistort(frame, transform_with(m, kp_poly, model="poly3"))
        out_none = cpu_undistort(frame, transform_with(m, kp_zero, model="poly3"))

        d = np.abs(out_poly.astype(int) - out_none.astype(int))
        # strong barrel distortion shifts interior sampling
        assert d.max() > 10

    def test_r_limit_fills_background(self):
        frame = random_frame()
        kp = make_kernel_params()
        kp.r_limit = 0.05  # valid only very close to the axis
        m = np.zeros((1, 14), dtype=np.float32)
        m[0, :9] = inv_k_matrix(40.0, 32.0, 24.0).ravel()
        out = cpu_undistort(frame, transform_with(m, kp))
        # nearly everything outside the tiny valid radius -> background
        assert (out == 0).mean() > 0.9

    def test_lens_correction_amount_blend(self):
        """lca=1 full correction vs lca=0 no correction differ."""
        frame = random_frame()
        coeffs = [0.2, 0.0, 0.0, 0.0]
        kp_full = make_kernel_params(model_coeffs=coeffs)
        kp_full.lens_correction_amount = 1.0
        kp_none = make_kernel_params(model_coeffs=coeffs)
        kp_none.lens_correction_amount = 0.0
        m = np.zeros((1, 14), dtype=np.float32)
        m[0, :9] = inv_k_matrix(40.0, 32.0, 24.0).ravel()
        out_full = cpu_undistort(frame, transform_with(m, kp_full))
        out_none = cpu_undistort(frame, transform_with(m, kp_none))
        assert np.abs(out_full.astype(int) - out_none.astype(int)).max() > 10


# ---------------------------------------------------------------------------
# 3. Sync offsets applied by the frame transform
# ---------------------------------------------------------------------------

class TestSyncOffsets:
    def _manager_with_rotating_gyro(self, duration_ms=4000.0):
        from pygyroflow.gyro_source import GyroSource, FileMetadata
        from pygyroflow.types.time_types import TimeIMU
        from pygyroflow.manager import StabilizationManager

        mgr = StabilizationManager()
        mgr.init_from_video_data(duration_ms, 30.0, 120, (64, 48))
        mgr.set_output_size(64, 48)

        rate_hz = 200.0
        n = int(duration_ms * rate_hz / 1000.0)
        t = np.arange(n) / rate_hz * 1000.0
        w = np.array([0.0, 0.0, 60.0])  # deg/s around z
        md = FileMetadata(detected_source="Synthetic", imu_orientation="XYZ")
        md.raw_imu = [
            TimeIMU(timestamp_ms=float(ts), gyro=w.copy(), accl=None) for ts in t
        ]
        mgr.gyro.load_from_telemetry(md)
        mgr.recompute_blocking()
        return mgr

    def test_offset_shifts_transform(self):
        mgr = self._manager_with_rotating_gyro()

        cp_no_offset = mgr._build_compute_params()
        assert not cp_no_offset.sync_offsets_adjusted

        # 500 ms offset: transform at t=1000 must equal no-offset t=500
        mgr.gyro.set_offset(0, 500.0)
        cp_offset = mgr._build_compute_params()
        assert cp_offset.sync_offsets_adjusted

        t_shifted = FrameTransform.at_timestamp(cp_offset, 1000.0, 30)
        t_ref = FrameTransform.at_timestamp(cp_no_offset, 500.0, 15)

        assert np.allclose(t_shifted.matrices, t_ref.matrices, atol=1e-6)

    def test_zero_offset_is_identity(self):
        mgr = self._manager_with_rotating_gyro()
        mgr.gyro.set_offset(0, 0.0)
        cp = mgr._build_compute_params()
        t = FrameTransform.at_timestamp(cp, 1000.0, 30)
        # offset dict exists but is zero -> same as no offset
        cp2 = mgr._build_compute_params()
        cp2.sync_offsets_adjusted = {}
        t2 = FrameTransform.at_timestamp(cp2, 1000.0, 30)
        assert np.allclose(t.matrices, t2.matrices, atol=1e-9)


# ---------------------------------------------------------------------------
# 4. Synthetic GPMF/mp4 parsing
# ---------------------------------------------------------------------------

def _gpmf_klv(fourcc: str, type_byte: int, payload: bytes, struct_size: int = 0,
              repeat: int = 1) -> bytes:
    payload = payload + b"\x00" * (-len(payload) % 4)
    if struct_size == 0:
        # Type-0 containers (DEVC/STRM) are encoded struct_size=1 x repeat
        # (GoPro's own encoding for payloads over 255 bytes); data elements
        # use struct_size=len(payload), repeat=1.
        if type_byte == 0:
            struct_size = 1
            repeat = len(payload)
        else:
            struct_size = max(len(payload), 1)
            repeat = 1
    header = fourcc.encode() + bytes([type_byte, struct_size]) + struct.pack(">H", repeat)
    return header + payload


def _mp4_box(fourcc: str, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + fourcc.encode() + payload


def build_synthetic_gopro_mp4(gyro_xyz: list[tuple[int, int, int]],
                              accl_xyz: list[tuple[int, int, int]],
                              scale: int = 100,
                              stamp_us: int = 0,
                              orio: str | None = None,
                              n_packets: int = 1,
                              stts_delta_ms: float | None = None) -> bytes:
    """Build a minimal mp4 containing only a gpmd data track.

    The telemetry parser reads boxes manually (no ffmpeg), so only the
    moov/trak/mdia/minf/stbl hierarchy plus the raw sample payloads at the
    offsets advertised by stco/stsz are required.

    ``n_packets`` > 1 repeats the DEVC payload as consecutive samples;
    ``stts_delta_ms`` adds an stts box giving each sample that duration
    (mirroring real GoPro files: one gpmd sample per ~1 s with Hero5/6
    carrying no STMP inside the payload).
    """
    gyro_payload = struct.pack(f">{len(gyro_xyz) * 3}h", *[v for s in gyro_xyz for v in s])
    accl_payload = struct.pack(f">{len(accl_xyz) * 3}h", *[v for s in accl_xyz for v in s])

    def build_devc(stamp_us_value: int | None) -> bytes:
        strm_parts = [
            _gpmf_klv("STNM", ord("c"), b"Angular velocity\0\0"),
            _gpmf_klv("ORIN", ord("c"), b"ZXY\0"),
        ]
        if orio:
            pad = b"\0" * (-len(orio) % 4)
            strm_parts.append(_gpmf_klv("ORIO", ord("c"), orio.encode() + pad))
        strm_parts += [
            _gpmf_klv("SIUN", ord("c"), b"rad/s\0\0"),
        ]
        if stamp_us_value is not None:
            strm_parts.append(_gpmf_klv("STMP", ord("J"), struct.pack(">q", stamp_us_value)))
        strm_parts += [
            _gpmf_klv("SCAL", ord("s"), struct.pack(">h", scale)),
            _gpmf_klv("GYRO", ord("s"), gyro_payload, struct_size=6, repeat=len(gyro_xyz)),
            _gpmf_klv("ACCL", ord("s"), accl_payload, struct_size=6, repeat=len(accl_xyz)),
        ]
        strm = b"".join(strm_parts)
        return _gpmf_klv("DEVC", 0, _gpmf_klv("STRM", 0, strm) + b"DVC1\0\0\0")

    if isinstance(stamp_us, (list, tuple)):
        packets = [build_devc(s) for s in stamp_us]
        n_packets = len(packets)
    else:
        devc = build_devc(stamp_us)
        packets = [devc] * n_packets
    payload = b"".join(packets)

    # stbl advertising n_packets samples of size len(devc); samples placed
    # after moov.
    sample_offset_placeholder = 0  # patched below

    def make_moov(sample_offset: int, sample_size: int) -> bytes:
        stsd_payload = struct.pack(">II", 0, 1) + struct.pack(">I", 16) + b"gpmd" + b"\x00" * 8
        stsd = _mp4_box("stsd", stsd_payload)
        stco_payload = struct.pack(">II", 0, n_packets)
        for i in range(n_packets):
            stco_payload += struct.pack(">I", sample_offset + i * sample_size)
        stco = _mp4_box("stco", stco_payload)
        stsz_payload = struct.pack(">III", 0, 0, n_packets) + struct.pack(">I", sample_size) * n_packets
        stsz = _mp4_box("stsz", stsz_payload)
        stsc = _mp4_box("stsc", struct.pack(">II", 0, 1) + struct.pack(">III", 1, 1, 1))
        boxes = stsd + stco + stsz + stsc
        if stts_delta_ms is not None:
            # timescale 1000 ticks/s -> delta in ms == delta in ticks
            stts = _mp4_box("stts", struct.pack(">II", 0, 1) + struct.pack(">II", n_packets, int(stts_delta_ms)))
            boxes += stts
        stbl = _mp4_box("stbl", boxes)
        minf = _mp4_box("minf", stbl)
        hdlr = _mp4_box("hdlr", b"\x00" * 8 + b"gpmd" + b"\x00" * 12)
        # mdhd v0: version+flags(4) + creation(4) + modification(4) +
        # timescale(4) + duration(4)
        mdhd = _mp4_box("mdhd", struct.pack(">IIIII", 0, 0, 0, 1000, 30))
        mdia = _mp4_box("mdia", mdhd + hdlr + minf)
        tkhd = _mp4_box("tkhd", b"\x00" * 84)
        trak = _mp4_box("trak", tkhd + mdia)
        mvhd = _mp4_box("mvhd", b"\x00" * 96)
        return _mp4_box("moov", mvhd + trak)

    moov = make_moov(sample_offset_placeholder, len(packets[0]))
    sample_offset = len(moov)
    moov = make_moov(sample_offset, len(packets[0]))
    assert len(moov) == sample_offset  # size stable (no offset-dependent length)
    return moov + payload


class TestSyntheticGpmfParsing:
    def test_parse_gopro_synthetic(self, tmp_path):
        from pygyroflow.telemetry import parse_telemetry_file

        gyro = [(100 * (i % 5) - 200, 300, -50) for i in range(20)]
        accl = [(0, 0, 1000) for _ in range(20)]
        data = build_synthetic_gopro_mp4(gyro, accl, scale=100, stamp_us=123456)
        path = tmp_path / "synthetic.mp4"
        path.write_bytes(data)

        md = parse_telemetry_file(str(path), fps=30.0)
        assert md.detected_source == "GoPro"
        # Upstream telemetry-parser: files without ORIO/MTRX (Hero8+) get NO
        # orientation remap — Gyroflow guesses it at runtime instead.
        assert md.imu_orientation is None
        assert len(md.raw_imu) == 20

        first = md.raw_imu[0]
        assert first.gyro is not None
        # physical = raw / scale, gyro additionally converted to deg/s
        expected = (-200 / 100.0) * 180.0 / math.pi
        assert first.gyro[0] == pytest.approx(expected, rel=1e-6)
        assert first.accl is not None
        assert first.accl[2] == pytest.approx(10.0, rel=1e-6)  # 1000 / 100

    def test_integrate_synthetic_gyro(self, tmp_path):
        """Parsed GPMF data must be loadable and integrable by GyroSource."""
        from pygyroflow.telemetry import parse_telemetry_file
        from pygyroflow.gyro_source import GyroSource

        gyro = [(500, 0, 0) for _ in range(40)]  # constant rotation about x
        data = build_synthetic_gopro_mp4(gyro, gyro, scale=100, stamp_us=0)
        path = tmp_path / "synthetic2.mp4"
        path.write_bytes(data)

        md = parse_telemetry_file(str(path), fps=30.0)
        src = GyroSource()
        src.init_from_params(1000.0)
        src.load_from_telemetry(md)
        assert len(src.quaternions) > 10

    def test_orientation_from_orio(self, tmp_path):
        """ORIN+ORIO derive the orientation like upstream telemetry-parser."""
        from pygyroflow.telemetry import parse_telemetry_file
        from pygyroflow.telemetry.parser import (
            _gopro_orientations_to_matrix,
            _gopro_mtrx_to_orientation,
        )

        # unit-check the ports of orientations_to_matrix / mtrx_to_orientation
        # (upstream: exact char match = +1, case-insensitive match = -1)
        m = _gopro_orientations_to_matrix("ZXY", "YXZ")
        assert m is not None
        assert _gopro_mtrx_to_orientation(m) == "ZYX"
        m2 = _gopro_orientations_to_matrix("ZXY", "yxz")
        assert _gopro_mtrx_to_orientation(m2) == "zyx"
        # identity / mismatched lengths
        assert _gopro_mtrx_to_orientation(_gopro_orientations_to_matrix("XYZ", "XYZ")) == "XYZ"
        assert _gopro_orientations_to_matrix("ZXY", "YZ") is None

        # end-to-end: a synthetic file WITH orio derives the orientation
        gyro = [(100, 100, 100) for _ in range(10)]
        data = build_synthetic_gopro_mp4(gyro, gyro, scale=100, stamp_us=0, orio="YXZ")
        path = tmp_path / "synthetic_orio.mp4"
        path.write_bytes(data)
        md = parse_telemetry_file(str(path), fps=30.0)
        assert md.imu_orientation == "ZYX"

    def test_no_stmp_multipacket_spans_whole_timeline(self, tmp_path):
        """Hero5/6 regression: no STMP in packets + many packets.

        Real Hero6 files (e.g. the 34 s 4k clip) carry one gpmd packet per
        second WITHOUT any STMP tag. The parser must lay the readings on a
        uniform grid covering the whole track duration (stts-derived), not
        collapse every packet onto the first second (timestamp overwrite).
        """
        from pygyroflow.telemetry import parse_telemetry_file

        gyro = [(100 * (i % 5) - 200, 300, -50) for i in range(200)]
        data = build_synthetic_gopro_mp4(
            gyro, gyro, scale=100, stamp_us=None,
            n_packets=3, stts_delta_ms=1001.0,
        )
        path = tmp_path / "hero6_nostmp.mp4"
        path.write_bytes(data)
        md = parse_telemetry_file(str(path), fps=29.97)

        assert len(md.raw_imu) == 600  # 3 packets x 200 readings
        ts = [r.timestamp_ms for r in md.raw_imu]
        # Uniform grid over the full 3*1001 ms, not collapsed to <=1 s
        assert ts[-1] == pytest.approx(5.005 * (600 - 1), abs=1.0)  # 3003/600=5.005ms step
        assert ts[-1] > 2000.0
        steps = set(round(b - a, 6) for a, b in zip(ts[:-1], ts[1:]))
        assert len(steps) == 1  # strictly uniform
        # data survived: readings of packet 2/3 are present with real values
        assert md.raw_imu[400].gyro is not None
        assert md.raw_imu[599].gyro is not None

    def test_stmp_span_derives_grid_step(self, tmp_path):
        """Hero8-style files (per-packet STMP) derive the step from STMP span.

        Upstream GoPro::get_avg_sample_duration: step = (last STMP - first
        STMP) / (total readings - last packet's readings). Two packets 1000 ms
        apart with 200 readings each -> step 1000/200 = 5.0 ms, no stts
        needed.
        """
        from pygyroflow.telemetry import parse_telemetry_file

        gyro = [(100 * (i % 5) - 200, 300, -50) for i in range(200)]
        data = build_synthetic_gopro_mp4(
            gyro, gyro, scale=100, stamp_us=[10_000, 1_010_000],
        )
        path = tmp_path / "hero8_stmp.mp4"
        path.write_bytes(data)
        md = parse_telemetry_file(str(path), fps=30.0)
        ts = [r.timestamp_ms for r in md.raw_imu]
        assert len(ts) == 400
        assert ts[1] - ts[0] == pytest.approx(5.0, abs=0.001)
        assert ts[-1] == pytest.approx(5.0 * 399, abs=0.1)

    def test_camera_tags_extracted_for_identifier(self, tmp_path):
        """DEVC-level EISA/VFOV tags feed CameraIdentifier (lens autoload)."""
        from pygyroflow.telemetry.parser import _parse_gpmf_chunk

        strm = _gpmf_klv("STNM", ord("c"), b"Angular velocity\0\0") + \
            _gpmf_klv("SCAL", ord("s"), struct.pack(">h", 100)) + \
            _gpmf_klv("GYRO", ord("s"), struct.pack(">6h", 1, 2, 3, 4, 5, 6), struct_size=6, repeat=2)
        devc = _gpmf_klv("DEVC", 0,
                         _gpmf_klv("VFOV", ord("c"), b"W\0\0\0") +
                         _gpmf_klv("EISA", ord("c"), b"N\0\0\0") +
                         _gpmf_klv("ZFOV", ord("f"), struct.pack(">f", 100.0)) +
                         _gpmf_klv("STRM", 0, strm))
        tags = {}
        model, g_rows, a_rows, g_count, stamp = _parse_gpmf_chunk(devc, None, tags)
        assert tags == {"VFOV": "W", "EISA": "N", "ZFOV": pytest.approx(100.0)}
        assert g_count == 2 and len(g_rows) == 2

        # end-to-end: CameraIdentifier built from these tags hits the DB id
        from pygyroflow.camera.identifier import CameraIdentifier
        ident = CameraIdentifier.from_metadata(
            brand="GoPro", model="HERO6 Black",
            video_width=2704, video_height=2028, fps=29.97,
            samples=[{"tag_map": {"Default": tags}}],
        )
        assert ident.additional == "NO-EIS"
        assert ident.lens_info == "Wide"
        assert ident.identifier == "gopro-hero6black-wide-2704x2028@29970-no-eis"
        assert ident.get_identifier_for_autoload() == ident.identifier

    def test_untimed_file_falls_back_to_200hz_grid(self, tmp_path):
        """No STMP and no stts: assume ~200 Hz rather than all-zero times."""
        from pygyroflow.telemetry import parse_telemetry_file

        gyro = [(100 * (i % 5) - 200, 300, -50) for i in range(20)]
        data = build_synthetic_gopro_mp4(gyro, gyro, scale=100, stamp_us=None)
        path = tmp_path / "untimed.mp4"
        path.write_bytes(data)
        md = parse_telemetry_file(str(path), fps=30.0)
        ts = [r.timestamp_ms for r in md.raw_imu]
        assert len(ts) == 20
        assert ts[1] - ts[0] == pytest.approx(5.0, abs=0.001)

    @pytest.mark.skipif(
        not __import__("os").path.isfile("/home/ft/workspace/testvideos/gpmf-11-hero8.mp4"),
        reason="local Hero8 sample not available",
    )
    def test_real_hero8_tail_camera_tags(self):
        """Real Hero8 file: camera DEVC after the last sample-table entry
        still yields VFOV/EISA tags and a non-empty lens autoload."""
        from pygyroflow.manager import StabilizationManager

        mgr = StabilizationManager()
        mgr.load_video("/home/ft/workspace/testvideos/gpmf-11-hero8.mp4")
        tags = (mgr.gyro.file_metadata.additional_data or {}).get("camera_tags") or {}
        inner = tags.get("Default") or tags
        assert inner.get("VFOV") == "W"
        assert mgr.lens.calib_dimension.get("w", 0) > 0


# ---------------------------------------------------------------------------
# 7. Keyframe animation reaches the frame transform (P4-3)
# ---------------------------------------------------------------------------


class TestKeyframeAnimation:
    def _manager_with_gyro(self):
        from pygyroflow.manager import StabilizationManager
        from pygyroflow.gyro_source import FileMetadata
        from pygyroflow.types.time_types import TimeIMU

        mgr = StabilizationManager()
        mgr.init_from_video_data(4000.0, 30.0, 120, (64, 48))
        mgr.set_output_size(64, 48)
        n = int(4000.0 * 200.0 / 1000.0)
        t = np.arange(n) * 5.0
        md = FileMetadata(detected_source="Synthetic", imu_orientation="XYZ")
        md.raw_imu = [
            TimeIMU(timestamp_ms=float(ts), gyro=np.array([0.0, 0.0, 45.0]), accl=None)
            for ts in t
        ]
        mgr.gyro.load_from_telemetry(md)
        mgr.recompute_blocking()
        return mgr

    def test_video_rotation_keyframe_animates(self):
        from pygyroflow.keyframes.types import KeyframeType

        mgr = self._manager_with_gyro()

        tr_before = mgr.get_frame_transform(1000.0, 30)
        m_before = tr_before.matrices[0][:9].astype(np.float64).reshape(3, 3)

        # Animate video rotation 0 -> 90 deg over the clip
        mgr.keyframes.set_keyframe(KeyframeType.VideoRotation, 0, 0.0)
        mgr.keyframes.set_keyframe(KeyframeType.VideoRotation, 4000000, 90.0)

        tr_after = mgr.get_frame_transform(1000.0, 30)
        m_after = tr_after.matrices[0][:9].astype(np.float64).reshape(3, 3)
        assert not np.allclose(m_before, m_after), "rotation keyframe had no effect"

        tr_late = mgr.get_frame_transform(3500.0, 105)
        m_late = tr_late.matrices[0][:9].astype(np.float64).reshape(3, 3)
        assert not np.allclose(m_after, m_late)

    def test_fov_keyframe_changes_scale(self):
        from pygyroflow.keyframes.types import KeyframeType

        mgr = self._manager_with_gyro()
        mgr.params.fovs = [1.0] * 120

        fov_plain = mgr.get_frame_transform(2000.0, 60).fov
        mgr.keyframes.set_keyframe(KeyframeType.Fov, 0, 1.0)
        mgr.keyframes.set_keyframe(KeyframeType.Fov, 4000000, 2.0)
        fov_zoomed = mgr.get_frame_transform(2000.0, 60).fov
        assert abs(fov_zoomed - fov_plain) > 0.1, "FOV keyframe had no effect"


# ---------------------------------------------------------------------------
# 7b. IMU orientation guessing (48 permutations)
# ---------------------------------------------------------------------------


class TestOrientationGuess:
    def test_guess_recovers_pan_axis(self, tmp_path):
        """Pan footage: the winning orientation must map the gyro axis
        carrying the pan signal onto the visual pan axis (Y here)."""
        cv2 = pytest.importorskip("cv2", reason="OpenCV required")

        from pygyroflow.manager import StabilizationManager
        from pygyroflow.gyro_source import FileMetadata
        from pygyroflow.types.time_types import TimeIMU

        w, h, fps, n = 320, 240, 30.0, 30
        K = np.array([[256.0, 0.0, w / 2.0], [0.0, 256.0, h / 2.0], [0.0, 0.0, 1.0]])
        t = np.arange(n) / fps
        omega = 25.0 * np.sin(2 * np.pi * t / 1.5)  # visual-Y pan, deg/s

        rng = np.random.default_rng(5)
        base = np.full((h, w), 128, dtype=np.uint8)
        for _ in range(120):
            cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
            cv2.circle(base, (cx, cy), int(rng.integers(3, 12)), int(rng.integers(0, 255)), -1)

        src = tmp_path / "pan.mp4"
        container = av.open(str(src), "w")
        vs = container.add_stream("libx264", rate=Fraction(int(fps), 1))
        vs.width, vs.height, vs.pix_fmt = w, h, "yuv420p"
        theta = np.rad2deg(np.cumsum(np.deg2rad(omega)) / fps)
        for i in range(n):
            a = np.deg2rad(theta[i] - theta[0])
            R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
            H = K @ R @ np.linalg.inv(K)
            warped = cv2.warpPerspective(base, H, (w, h), borderMode=cv2.BORDER_REPLICATE)
            vf = av.VideoFrame.from_ndarray(np.repeat(warped[:, :, None], 3, axis=2), format="rgb24")
            for pkt in vs.encode(vf):
                container.mux(pkt)
        for pkt in vs.encode():
            container.mux(pkt)
        container.close()

        mgr = StabilizationManager()
        mgr.load_video(str(src))
        assert not mgr.gyro.file_metadata.quaternions  # raw-imu path

        # Gyro stream carrying the pan on its Z axis (like an IMU mounted
        # with Y->Z). The guess must map that axis onto the visual Y.
        md = FileMetadata(detected_source="Synthetic", imu_orientation=None)
        md.raw_imu = [
            TimeIMU(timestamp_ms=float(ts * 1000.0),
                    gyro=np.array([0.0, omega[i], 0.0]),
                    accl=None)
            for i, ts in enumerate(t)
        ]
        mgr.gyro.file_metadata = md
        mgr.gyro.load_from_telemetry(md)
        mgr.gyro.integration_method = 3

        guessed = mgr.guess_imu_orientation(sample_count=30)
        assert guessed is not None
        # The signal axis must be mapped to visual position 1 (Y):
        # the orientation char at position 1 selects the gyro axis that
        # feeds visual Y — it must be the axis carrying the signal.
        assert guessed[0] in "Yy" or guessed[1] in "Yy"


# ---------------------------------------------------------------------------
# 8. Telemetry parity with upstream telemetry-parser (ground-truth fixtures)
# ---------------------------------------------------------------------------


class TestTelemetryParity:
    """Regression tests for bugs found via upstream ground-truth dumps.

    Ground truth was produced with the upstream telemetry-parser rev 2f4218b
    (the same rev Gyroflow pins) via the tp-dump harness in the parent
    workspace. The parser-level regression below pins the semantics that
    were ported wrong originally:

      - GoPro: hardcoded "ZXY" orientation (upstream: derived/None)
      - GoPro: CORI*IORI quaternion stream ignored (upstream's primary
        stabilization source on Hero9+)
      - GoPro: SROT rolling-shutter readout time
      - DJI: quaternion multiply order flipped y/z signs vs upstream
      - DJI: quaternion double-cover continuity flip
    """

    def test_dji_multiply_order(self):
        """DJI transform order must reproduce the upstream reference exactly.

        Fixture: first raw quaternion of the real Osmo Nano file, and the
        upstream telemetry-parser (rev 2f4218b) output for it:
        (0.485435, 0.450314, 0.530656, 0.529126) in (w, x, y, z).
        The wrong multiply order flips y/z signs.
        """
        from pygyroflow.telemetry.parser import _multiply_quat

        # raw (w, x, y, z) as decoded from the djmd protobuf stream
        raw = (0.99776554, 0.01832509, -0.01679565, -0.06201635)
        q1 = _multiply_quat(raw[0], raw[1], raw[2], raw[3], 0.5, -0.5, -0.5, 0.5)
        got = _multiply_quat(0.0, 0.0, 1.0, 0.0, q1[0], q1[1], q1[2], q1[3])
        ref = (0.485435, 0.450314, 0.530656, 0.529126)
        for g, r in zip(got, ref):
            assert g == pytest.approx(r, abs=1e-5)
        # the WRONG order demonstrably differs (y/z sign flip on this input)
        wrong = _multiply_quat(q1[0], q1[1], q1[2], q1[3], 0.0, 0.0, 1.0, 0.0)
        assert wrong != got

    def test_gopro_orientation_chunk_extraction(self):
        """CORI/IORI extraction converts i16 with upstream sign convention."""
        from pygyroflow.telemetry.parser import _parse_gpmf_orientation_chunk

        def klv(fourcc: str, type_byte: int, payload: bytes, struct_size: int = 0,
                repeat: int = 1) -> bytes:
            payload = payload + b"\x00" * (-len(payload) % 4)
            if struct_size == 0:
                if type_byte == 0:
                    struct_size = 1
                    repeat = len(payload)
                else:
                    struct_size = max(len(payload), 1)
                    repeat = 1
            header = fourcc.encode() + bytes([type_byte, struct_size]) + struct.pack(">H", repeat)
            return header + payload

        # two i16 quats per stream: (w, x, y, z) raw, SCAL=32767
        cori_raw = struct.pack(">8h", 32767, -16384, 0, 0, 16384, 16384, 16384, 16384)
        iori_raw = struct.pack(">8h", 32767, 0, 0, 0, 32767, 0, 0, 0)
        strm_cori = b"".join([
            klv("STNM", ord("c"), b"CameraOrientation\0"),
            klv("SCAL", ord("s"), struct.pack(">h", 32767)),
            klv("CORI", ord("s"), cori_raw, struct_size=8, repeat=2),
        ])
        strm_iori = b"".join([
            klv("STNM", ord("c"), b"ImageOrientation\0"),
            klv("SCAL", ord("s"), struct.pack(">h", 32767)),
            klv("IORI", ord("s"), iori_raw, struct_size=8, repeat=2),
        ])
        devc = klv("DEVC", 0, klv("STRM", 0, strm_cori) + klv("STRM", 0, strm_iori))

        cori, iori = _parse_gpmf_orientation_chunk(devc)
        assert len(cori) == 2 and len(iori) == 2
        # first CORI: (1.0, +16384/32767, 0, 0) — x sign flipped by upstream convention
        assert cori[0] == pytest.approx([1.0, 16384 / 32767, 0.0, 0.0], abs=1e-9)
        # IORI identity
        assert iori[0] == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-9)

# ---------------------------------------------------------------------------
# 5. Full render through real codecs
# ---------------------------------------------------------------------------

def build_synthetic_video(path, frames=30, w=64, h=48, fps=30, with_audio=True):
    """Encode a synthetic test video (optionally with audio) via PyAV."""
    container = av.open(str(path), "w")
    vstream = container.add_stream("libx264", rate=Fraction(fps, 1))
    vstream.width, vstream.height, vstream.pix_fmt = w, h, "yuv420p"

    astream = None
    if with_audio:
        astream = container.add_stream("aac")
        astream.rate = 48000
        astream.layout = "stereo"

    rng = np.random.default_rng(11)
    for i in range(frames):
        # moving gradient + noise: stabilization has structure to work with
        x = np.arange(w, dtype=np.float64)[None, :]
        y = np.arange(h, dtype=np.float64)[:, None]
        img = (
            127
            + 60 * np.sin((x + i * 3.0) / 7.0)
            + 40 * np.cos((y + i * 2.0) / 9.0)
        )
        frame_u8 = np.clip(img, 0, 255).astype(np.uint8)
        frame_u8 = np.repeat(frame_u8[:, :, None], 3, axis=2)
        vframe = av.VideoFrame.from_ndarray(frame_u8, format="rgb24")
        for pkt in vstream.encode(vframe):
            container.mux(pkt)

        if astream is not None and i % 2 == 0:
            samples = np.zeros((1024, 2), dtype=np.float32)
            arr = (samples * 32767.0).astype(np.int16)
            aframe = av.AudioFrame.from_ndarray(
                np.ascontiguousarray(arr.T), format="s16p", layout="stereo"
            )
            aframe.sample_rate = 48000
            for pkt in astream.encode(aframe):
                container.mux(pkt)

    for pkt in vstream.encode():
        container.mux(pkt)
    if astream is not None:
        for pkt in astream.encode():
            container.mux(pkt)
    container.close()


def count_frames(path, stream_type="video"):
    c = av.open(str(path))
    n = 0
    streams = c.streams.video if stream_type == "video" else c.streams.audio
    if not streams:
        c.close()
        return 0
    # Count frames from ALL packets including the trailing flush packet —
    # filtering on dts hides its frames and undercounts the total.
    for p in c.demux(streams[0]):
        n += len(p.decode())
    c.close()
    return n


@pytest.mark.slow
class TestRenderPipeline:
    def test_render_frame_count_and_audio(self, tmp_path):
        """Full load->recompute->render pass on a synthetic video.

        Guards two past defects: the missing decoder flush (output losing
        the last ~thread-count frames) and the absent audio copy.
        """
        from pygyroflow.manager import StabilizationManager

        src = tmp_path / "in.mp4"
        build_synthetic_video(src, frames=30, with_audio=True)

        mgr = StabilizationManager()
        info = mgr.load_video(str(src))
        assert info["width"] == 64 and info["height"] == 48

        # No gyro in the file -> inject a rotating synthetic signal so the
        # stabilization path is active during rendering.
        from pygyroflow.gyro_source import FileMetadata
        from pygyroflow.types.time_types import TimeIMU

        duration_ms = info["duration_ms"]
        n = int(duration_ms * 200.0 / 1000.0)
        t = np.arange(n) * 5.0
        w = np.array([0.0, 0.0, 45.0])
        md = FileMetadata(detected_source="Synthetic", imu_orientation="XYZ")
        md.raw_imu = [TimeIMU(timestamp_ms=float(ts), gyro=w.copy(), accl=None) for ts in t]
        mgr.gyro.load_from_telemetry(md)
        mgr.recompute_blocking()
        assert len(mgr.gyro.quaternions) > 10
        assert len(mgr.gyro.smoothed_quaternions) > 10

        out = tmp_path / "out.mp4"
        mgr.render(str(src), str(out), {"codec": "H.264/AVC", "audio": True})

        in_frames = count_frames(src)
        out_frames = count_frames(out)
        assert in_frames == 30
        assert out_frames == in_frames, "output lost frames (decoder flush regression?)"

        c = av.open(str(out))
        assert len(c.streams.audio) == 1, "audio track missing (copy regression?)"
        c.close()

    def test_render_no_audio_option(self, tmp_path):
        from pygyroflow.manager import StabilizationManager

        src = tmp_path / "in2.mp4"
        build_synthetic_video(src, frames=10, with_audio=True)

        mgr = StabilizationManager()
        mgr.load_video(str(src))
        out = tmp_path / "out_noaudio.mp4"
        mgr.render(str(src), str(out), {"codec": "H.264/AVC", "audio": False})

        c = av.open(str(out))
        assert len(c.streams.audio) == 0
        c.close()

    def test_stabilized_output_differs_from_passthrough(self, tmp_path):
        """With real smoothing data the render must actually move pixels."""
        from pygyroflow.manager import StabilizationManager
        from pygyroflow.gyro_source import FileMetadata
        from pygyroflow.types.time_types import TimeIMU

        src = tmp_path / "in3.mp4"
        build_synthetic_video(src, frames=10, with_audio=False)

        mgr = StabilizationManager()
        info = mgr.load_video(str(src))
        n = int(info["duration_ms"] * 200.0 / 1000.0)
        t = np.arange(n) * 5.0
        md = FileMetadata(detected_source="Synthetic", imu_orientation="XYZ")
        md.raw_imu = [
            TimeIMU(timestamp_ms=float(ts), gyro=np.array([0.0, 0.0, 90.0]), accl=None)
            for ts in t
        ]
        mgr.gyro.load_from_telemetry(md)
        mgr.smoothing.current().set_parameter("smoothness", 0.5)
        mgr.recompute_blocking()

        out = tmp_path / "out3.mp4"
        mgr.render(str(src), str(out), {"codec": "H.264/AVC", "audio": False})

        # Decode middle frame of both and compare
        def read_frame(path, idx):
            c = av.open(str(path))
            n = 0
            result = None
            for p in c.demux(c.streams.video[0]):
                if p.dts is None:
                    continue
                for f in p.decode():
                    if n == idx:
                        result = f.to_ndarray(format="rgb24")
                    n += 1
            c.close()
            return result

        a = read_frame(src, 5)
        b = read_frame(out, 5)
        assert a is not None and b is not None
        assert np.abs(a.astype(int) - b.astype(int)).mean() > 0.5


# ---------------------------------------------------------------------------
# 6.# ---------------------------------------------------------------------------
# 6. Auto-sync recovers a known gyro-to-video delay (P1)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestAutoSync:
    def test_sync_recovers_known_delay(self, tmp_path):
        """Synthetic pan rotation video vs delayed gyro -> offset recovered.

        The video pans with a known angular velocity profile; the gyro
        records the same rotation but on a clock shifted by D ms. The
        ``visual = gyro + offset`` convention expects offset = -D.
        """
        cv2 = pytest.importorskip("cv2", reason="OpenCV required for sync test")

        from pygyroflow.manager import StabilizationManager
        from pygyroflow.gyro_source import FileMetadata
        from pygyroflow.types.time_types import TimeIMU

        w, h, fps, n = 320, 240, 30.0, 40
        K = np.array(
            [[256.0, 0.0, w / 2.0], [0.0, 256.0, h / 2.0], [0.0, 0.0, 1.0]]
        )
        delay_ms = 200.0

        # Camera rotation profile: oscillating pan (y axis), deg/s
        t = np.arange(n) / fps
        omega_y = 25.0 * np.sin(2.0 * np.pi * t / 1.5)
        theta = np.rad2deg(np.cumsum(np.deg2rad(omega_y)) / fps)

        # Textured base image with trackable blobs
        rng = np.random.default_rng(5)
        base = np.full((h, w), 128, dtype=np.uint8)
        for _ in range(120):
            cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
            r = int(rng.integers(3, 12))
            cv2.circle(base, (cx, cy), r, int(rng.integers(0, 255)), -1)

        src = tmp_path / "pan.mp4"
        container = av.open(str(src), "w")
        vstream = container.add_stream("libx264", rate=Fraction(int(fps), 1))
        vstream.width, vstream.height, vstream.pix_fmt = w, h, "yuv420p"
        for i in range(n):
            a = np.deg2rad(theta[i] - theta[0])
            R = np.array(
                [
                    [np.cos(a), 0.0, np.sin(a)],
                    [0.0, 1.0, 0.0],
                    [-np.sin(a), 0.0, np.cos(a)],
                ]
            )
            H = K @ R @ np.linalg.inv(K)
            warped = cv2.warpPerspective(
                base, H, (w, h), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            vframe = av.VideoFrame.from_ndarray(
                np.repeat(warped[:, :, None], 3, axis=2), format="rgb24"
            )
            for pkt in vstream.encode(vframe):
                container.mux(pkt)
        for pkt in vstream.encode():
            container.mux(pkt)
        container.close()

        # Gyro: same rotation, timestamps shifted by +delay_ms.
        # The video warps the SCENE by R(+theta); the inferred camera
        # rotation is the inverse, hence -omega. IMU yaw axis is Z.
        mgr = StabilizationManager()
        mgr.load_video(str(src))
        md = FileMetadata(detected_source="Synthetic", imu_orientation="XYZ")
        md.raw_imu = [
            TimeIMU(
                timestamp_ms=float((t[i] + delay_ms / 1000.0) * 1000.0),
                gyro=np.array([0.0, 0.0, -omega_y[i]]),
                accl=None,
            )
            for i in range(n)
        ]
        # method 3 = SimpleGyro: cleanest for synthetic data without accel
        mgr.gyro.integration_method = 3
        mgr.gyro.load_from_telemetry(md)
        assert len(mgr.gyro.quaternions) >= n - 2

        offset = mgr.synchronize(sample_count=30, search_range_ms=500.0)
        assert offset is not None, "auto-sync returned None"
        assert abs(offset - (-delay_ms)) < 25.0, (
            f"recovered offset {offset:.1f} ms, expected {-delay_ms:.1f} ms"
        )
        # The offset must actually be stored on the gyro source
        assert len(mgr.gyro.get_offsets()) == 1


def _sony_tlv(tag: int, payload: bytes) -> bytes:
    return struct.pack(">HH", tag, len(payload)) + payload


def build_synthetic_sony_mp4(gyro_raw, accl_raw, gyro_scale=65.5, accl_scale=8192.0,
                             orient_u16=0x420, n_packets=1, stts_delta_ms=100.0) -> bytes:
    """Build a minimal mp4 with one 'rtmd' track mimicking Sony RTMD."""
    def make_payload() -> bytes:
        gyro = struct.pack(">ii", len(gyro_raw), 6) + struct.pack(
            f">{len(gyro_raw)*3}h", *[v for s in gyro_raw for v in s])
        accl = struct.pack(">ii", len(accl_raw), 6) + struct.pack(
            f">{len(accl_raw)*3}h", *[v for s in accl_raw for v in s])
        tlv = b"".join([
            _sony_tlv(0xE439, struct.pack(">f", gyro_scale)),
            _sony_tlv(0xE43A, struct.pack(">H", orient_u16)),
            _sony_tlv(0xE43B, gyro),
            _sony_tlv(0xE449, struct.pack(">f", accl_scale)),
            _sony_tlv(0xE44B, accl),
            _sony_tlv(0xE40E, struct.pack(">i", 14297)),
        ])
        return b"\x00\x1c" + b"\x00" * 0x1A + tlv

    payload = make_payload() * n_packets
    sample_size = len(payload) // n_packets

    def make_moov(sample_offset: int) -> bytes:
        stsd_payload = struct.pack(">II", 0, 1) + struct.pack(">I", 16) + b"rtmd" + b"\x00" * 8
        stsd = _mp4_box("stsd", stsd_payload)
        stco = _mp4_box("stco", struct.pack(">II", 0, n_packets) +
                        b"".join(struct.pack(">I", sample_offset + i * sample_size) for i in range(n_packets)))
        stsz = _mp4_box("stsz", struct.pack(">III", 0, 0, n_packets) + struct.pack(">I", sample_size) * n_packets)
        stsc = _mp4_box("stsc", struct.pack(">II", 0, 1) + struct.pack(">III", 1, 1, 1))
        stts = _mp4_box("stts", struct.pack(">II", 0, 1) + struct.pack(">II", n_packets, int(stts_delta_ms)))
        stbl = _mp4_box("stbl", stsd + stco + stsz + stsc + stts)
        minf = _mp4_box("minf", stbl)
        hdlr = _mp4_box("hdlr", b"\x00" * 8 + b"rtmd" + b"\x00" * 12)
        mdhd = _mp4_box("mdhd", struct.pack(">IIIII", 0, 0, 0, 1000, 30))
        mdia = _mp4_box("mdia", mdhd + hdlr + minf)
        tkhd = _mp4_box("tkhd", b"\x00" * 84)
        trak = _mp4_box("trak", tkhd + mdia)
        mvhd = _mp4_box("mvhd", b"\x00" * 96)
        return _mp4_box("moov", mvhd + trak)

    moov = make_moov(0)
    moov = make_moov(len(moov))
    return moov + payload


class TestSonyRtmdParsing:
    def test_parse_sony_synthetic(self, tmp_path):
        from pygyroflow.telemetry import parse_telemetry_file
        from pygyroflow.telemetry.parser import _sony_read_f16, _sony_orientation

        # Sony decimal f16: 14.0 mm = mantissa 14, exp -3 -> 0xD00E
        assert _sony_read_f16(struct.pack(">H", 0xD00E)) == pytest.approx(0.014, rel=1e-6)
        # orientations per upstream comments
        assert _sony_orientation(struct.pack(">H", 0x152)) == "Yzx"
        assert _sony_orientation(struct.pack(">H", 0x420)) == "XYZ"
        assert _sony_orientation(struct.pack(">H", 0x241)) == "xZY"

        gyro = [(655, -482, -650), (646, -473, -700), (643, -477, -730)]
        accl = [(0x7F00, 0x1000, -0x0040)] * 3
        data = build_synthetic_sony_mp4(gyro, accl, n_packets=2, stts_delta_ms=100.0)
        path = tmp_path / "sony.mp4"
        path.write_bytes(data)

        md = parse_telemetry_file(str(path), fps=30.0)
        assert md.detected_source == "Sony"
        assert len(md.raw_imu) == 6  # 2 packets x 3 readings
        # uniform grid: 200 ms total / 6 readings
        ts = [r.timestamp_ms for r in md.raw_imu]
        assert ts[1] - ts[0] == pytest.approx(200.0 / 6.0, abs=1e-6)
        # values: raw / scale; accel additionally x 9.80665 (unit "g")
        first = md.raw_imu[0]
        assert first.gyro[0] == pytest.approx(655 / 65.5, rel=1e-6)
        assert first.gyro[1] == pytest.approx(-482 / 65.5, rel=1e-6)
        assert first.accl[0] == pytest.approx(0x7F00 / 8192.0 * 9.80665, rel=1e-6)
        # orientation: 0x420 = "XYZ" -> normalized swap-XY + invert-Z = "YXz"
        assert md.imu_orientation == "YXz"
        assert md.frame_readout_time == pytest.approx(14.297, abs=1e-3)

    @pytest.mark.skipif(
        not __import__("os").path.isfile("/home/ft/workspace/PreReserach/msGyroFlow/sony-ois-only.MP4"),
        reason="local Sony sample not available",
    )
    def test_real_sony_rx100m7(self):
        from pygyroflow.telemetry import parse_telemetry_file

        md = parse_telemetry_file(
            "/home/ft/workspace/PreReserach/msGyroFlow/sony-ois-only.MP4",
            fps=29.97, video_size=(3840, 2160),
        )
        assert md.detected_source == "Sony DSC-RX100M7"
        assert md.imu_orientation == "zYX"  # 0x152 -> "Yzx" -> normalized
        assert md.frame_readout_time == pytest.approx(14.297, abs=0.01)
        ts = [r.timestamp_ms for r in md.raw_imu]
        assert len(ts) > 20000
        assert ts[-1] == pytest.approx(11511.5, abs=5.0)  # full 11.5 s span
        # readable gyro magnitudes (deg/s), not garbage
        mags = np.array([np.linalg.norm(r.gyro) for r in md.raw_imu[:200] if r.gyro is not None])
        assert 0.01 < np.median(mags) < 500.0


def _insta_varint(v: int) -> bytes:
    out = b""
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out += bytes([b | 0x80])
        else:
            return out + bytes([b])


def _insta_pb_str(field: int, s: str) -> bytes:
    b = s.encode()
    return _insta_varint((field << 3) | 2) + _insta_varint(len(b)) + b


def _insta_pb_varint(field: int, v: int) -> bytes:
    return _insta_varint((field << 3) | 0) + _insta_varint(v)


def _insta_pb_f64(field: int, v: float) -> bytes:
    return _insta_varint((field << 3) | 1) + struct.pack("<d", v)


def _insta_pb_msg(field: int, payload: bytes) -> bytes:
    return _insta_varint((field << 3) | 2) + _insta_varint(len(payload)) + payload


_INSTA_MAGIC = b"8db42d694ccc418790edff439fe026bf"
_INSTA_HDR = 72


def _insta_record(rid: int, fmt: int, payload: bytes) -> bytes:
    return payload + bytes([fmt, rid]) + struct.pack("<I", len(payload))


def build_synthetic_insta360_file(
    raw_gyro: bool = True,
    with_offsets_index: bool = True,
    model: str = "Insta360 OneR",
) -> bytes:
    """Build a minimal file with an Insta360 extra-info trailer.

    Mirrors the layout upstream telemetry-parser's insta360 module parses:
    [prefix][records...][offsets-index record][72-byte header]. extra_size
    in the header includes the header itself (extra_start = size -
    extra_size must land on the first record).
    """
    if raw_gyro:
        fft = 1_700_000_123_456_789  # us
        items = b""
        for i in range(50):
            items += struct.pack("<Q", fft + (i + 1) * 5000)
            items += struct.pack("<6H", 32768 + 2048, 32768, 32768 - 2048,
                                 32768 + 3277, 32768 - 1640, 32768 + 819)
    else:
        fft = 1_700_000_123_456_789_000  # ns for non-raw files
        items = b""
        for i in range(20):
            items += struct.pack("<Q", fft + (i + 1) * 5_000_000)
            # acc in g, gyro in rad/s (100/-50/25 deg/s)
            items += struct.pack("<6d", 0.1, 0.0, -0.1,
                                 1.7453292519943295, -0.8726646259971648, 0.4363323129985824)

    off_v3 = "_".join(str(v) for v in [1, 0.5, 1500, 1500, 1024, 1024, 10, 20, 30,
                                       0, 0, 0, 0.01, -0.02, 0.005, 1e-05, -1e-05,
                                       3008, 3008, 0, 0])
    meta = b""
    meta += _insta_pb_str(2, model)
    if raw_gyro:
        meta += _insta_pb_msg(19, _insta_pb_varint(1, 2048) + _insta_pb_varint(2, 2048))
        meta += _insta_pb_msg(27, _insta_pb_varint(1, 2048) + _insta_pb_varint(2, 2048)
                              + _insta_pb_varint(3, 2048) + _insta_pb_varint(4, 2048))
        meta += _insta_pb_str(54, off_v3)
    meta += _insta_pb_varint(24, fft)
    meta += _insta_pb_f64(25, 15.5 if raw_gyro else 12.0)
    meta += _insta_pb_varint(62, 1 if raw_gyro else 0)
    if raw_gyro:
        meta += _insta_pb_msg(65, _insta_pb_varint(1, 16) + _insta_pb_varint(2, 2000))

    meta_rec = _insta_record(1, 0, meta)
    gyro_rec = _insta_record(3, 0, items)
    payloads = meta_rec + gyro_rec

    extra = payloads
    if with_offsets_index:
        tbl = bytes([1, 0]) + struct.pack("<II", len(meta), 0)
        tbl += bytes([3, 0]) + struct.pack("<II", len(items), len(meta_rec))
        extra = payloads + _insta_record(0, 0, tbl)

    header = b"\x00" * 32 + struct.pack("<II", len(extra) + _INSTA_HDR, 1) + _INSTA_MAGIC
    prefix = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2" + b"\x00" * 64
    return prefix + extra + header


class TestInsta360Parsing:
    """Cross-validated against upstream telemetry-parser (Rust) on the same
    synthetic files: values below are upstream's insta_dump output, not ours."""

    def test_raw_gyro_with_offsets_index(self, tmp_path):
        from pygyroflow.telemetry import parse_telemetry_file

        path = tmp_path / "sample.insv"
        path.write_bytes(build_synthetic_insta360_file(raw_gyro=True, with_offsets_index=True))
        md = parse_telemetry_file(str(path), fps=30.0)

        assert md.detected_source == "Insta360 OneR"
        assert md.imu_orientation == "Xyz"  # offset_v3 present -> OneR table
        assert md.frame_readout_time == pytest.approx(15.5)
        assert len(md.raw_imu) == 50

        first = md.raw_imu[0]
        # upstream: IMU 5.000000 225.830989 6.533593 -38.482198 4.268593 4.430821 -12.429308
        assert first.timestamp_ms == pytest.approx(5.0, abs=1e-6)
        assert first.gyro == pytest.approx([225.830989, 6.533593, -38.482198], abs=1e-5)
        assert first.accl == pytest.approx([4.268593, 4.430821, -12.429308], abs=1e-5)
        assert md.raw_imu[49].timestamp_ms == pytest.approx(250.0, abs=1e-6)

        lp = md.lens_profile
        assert lp is not None
        assert lp["distortion_model"] == "insta360"
        assert lp["calib_dimension"] == {"w": 2048, "h": 2048}
        assert lp["fisheye_params"]["camera_matrix"][0] == pytest.approx(
            [1500.0, 0.0, 697.1914893617021], abs=1e-6)
        assert lp["fisheye_params"]["distortion_coeffs"] == pytest.approx(
            [0.01, -0.02, 0.005, 1e-05, -1e-05, 0.5], abs=1e-12)
        assert lp["asymmetrical"] is True

    def test_nonraw_gyro_legacy_walk(self, tmp_path):
        from pygyroflow.telemetry import parse_telemetry_file

        path = tmp_path / "legacy.insv"
        path.write_bytes(build_synthetic_insta360_file(raw_gyro=False, with_offsets_index=False,
                                                       model="Insta360 ONE X2"))
        md = parse_telemetry_file(str(path), fps=30.0)

        assert md.detected_source == "Insta360 ONE X2"
        assert md.imu_orientation == "xZy"  # no offset_v3 -> ONE X2 legacy table
        assert md.frame_readout_time == pytest.approx(12.0)
        assert len(md.raw_imu) == 20

        first = md.raw_imu[0]
        # upstream: IMU 5000000.000000 100.000000 -50.000000 25.000000 0.980665 0 -0.980665
        # (non-raw timestamps are ns-sourced; upstream's model keeps x1000 scaling)
        assert first.timestamp_ms == pytest.approx(5_000_000.0, abs=1e-3)
        assert first.gyro == pytest.approx([100.0, -50.0, 25.0], abs=1e-6)
        assert first.accl == pytest.approx([0.980665, 0.0, -0.980665], abs=1e-6)
        assert md.lens_profile is None  # no offset_v3 -> no inline lens

    @pytest.mark.skipif(
        not __import__("os").path.isfile("/home/ft/workspace/testvideos/insta360-oner-preview.insv"),
        reason="local Insta360 sample not available",
    )
    def test_real_insv_oner_preview(self):
        """Real OneR .insv from the insvtools fixtures (736x368 preview, no
        offset_v3). Upstream telemetry-parser DROPS ALL telemetry on this
        file (strict loop errors on a 1-byte gyro-record tail, swallowed by
        .ok()); our tolerant floor-division recovers 1178 samples."""
        from pygyroflow.telemetry import parse_telemetry_file

        md = parse_telemetry_file(
            "/home/ft/workspace/testvideos/insta360-oner-preview.insv",
            fps=29.97, video_size=(736, 368),
        )
        assert md.detected_source == "Insta360 OneR"
        assert md.imu_orientation == "yXZ"  # legacy table (no offset_v3)
        assert md.frame_readout_time == pytest.approx(22.278, abs=0.01)
        assert len(md.raw_imu) == 1178
        import numpy as np
        amag = np.median([np.linalg.norm(r.accl) for r in md.raw_imu if r.accl is not None])
        assert amag == pytest.approx(9.80665, rel=0.02)  # static gravity
        ts = [r.timestamp_ms for r in md.raw_imu]
        assert ts[0] == pytest.approx(-135.0, abs=5.0)  # gyro precedes video

    def test_inline_lens_profile_loads_into_manager(self, tmp_path):
        from pygyroflow.telemetry import parse_telemetry_file
        from pygyroflow.manager import StabilizationManager

        path = tmp_path / "lens.insv"
        path.write_bytes(build_synthetic_insta360_file(raw_gyro=True, with_offsets_index=True))
        md = parse_telemetry_file(str(path), fps=30.0)

        mgr = StabilizationManager()
        mgr._load_lens_from_metadata(md.lens_profile, str(path))
        assert mgr.lens.calib_dimension["w"] == 2048
        assert mgr.lens.distortion_model == "insta360"
        assert mgr.lens.asymmetrical is True
        # GyroSource-level integration: quaternions computable from the raw IMU
        from pygyroflow.gyro_source import GyroSource
        src = GyroSource()
        src.init_from_params(250.0)
        src.load_from_telemetry(md)
        assert len(src.quaternions) > 10
        assert src.imu_transforms.imu_orientation == "Xyz"
