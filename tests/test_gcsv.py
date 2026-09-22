"""GCSV parsing and the sidecar fallback (A-02 text half, A-03)."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from pygyroflow.telemetry.gcsv import detect_gcsv, parse_gcsv
from pygyroflow.telemetry.parser import parse_telemetry_file
from pygyroflow.types.enums import ReadoutDirection

HEADER = """GYROFLOW IMU LOG
id,Test_Logger
vendor,pytest
tscale,0.001
gscale,1.0
ascale,1.0
orientation,xzY
t,gx,gy,gz,ax,ay,az
"""


def _write_gcsv(path, rows, header=HEADER):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header)
        for row in rows:
            fh.write(",".join(str(v) for v in row) + "\n")
    return str(path)


class TestDetect:
    def test_both_magic_lines(self, tmp_path):
        assert detect_gcsv(b"GYROFLOW IMU LOG\nrest")
        assert detect_gcsv(b"CAMERA IMU LOG\nrest")
        assert not detect_gcsv(b"something else")


class TestParseGcsv:
    def test_scales_and_units(self, tmp_path):
        """gscale is a divisor and adds π/180 (deg/s → rad/s consumers get
        deg/s back through the tag system); tscale converts the time
        column; timestamps land in milliseconds."""
        path = _write_gcsv(tmp_path / "log.gcsv", [
            (10, 100.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (20, 0.0, 100.0, 0.0, 0.0, 1.0, 0.0),
        ])
        meta = parse_gcsv(open(path, "rb").read())
        assert meta.detected_source == "GCSV"
        assert meta.imu_orientation == "xzY"
        assert len(meta.raw_imu) == 2
        first = meta.raw_imu[0]
        assert first.timestamp_ms == pytest.approx(10.0)
        # gscale 1.0 → the π/180 factor is the only change
        assert first.gyro[0] == pytest.approx(100.0 * math.pi / 180.0)
        assert first.accl is not None
        assert first.accl[2] == pytest.approx(1.0)

    def test_gscale_divides(self, tmp_path):
        header = HEADER.replace("gscale,1.0", "gscale,2.0")
        path = _write_gcsv(tmp_path / "log.gcsv",
                           [(0, 100.0, 0, 0, 0, 0, 0)], header)
        meta = parse_gcsv(open(path, "rb").read())
        assert meta.raw_imu[0].gyro[0] == pytest.approx(
            100.0 / 2.0 * math.pi / 180.0)

    def test_magnetometer_by_row_width(self, tmp_path):
        header = HEADER.replace(
            "t,gx,gy,gz,ax,ay,az", "t,gx,gy,gz,ax,ay,az,mx,my,mz")
        path = _write_gcsv(tmp_path / "log.gcsv",
                           [(0, 1, 2, 3, 0, 0, 0, 10.0, 0.0, 0.0)], header)
        meta = parse_gcsv(open(path, "rb").read())
        magn = meta.additional_data["magnetometer"]
        assert magn[0][1] == pytest.approx(1000.0)  # Gauss → μT (×100)

    def test_readout_direction_codes(self, tmp_path):
        cases = [
            ("0", 15.0, 15.0, ReadoutDirection.TopToBottom),
            ("TopToBottom", 15.0, 15.0, ReadoutDirection.TopToBottom),
            ("1", 15.0, -15.0, ReadoutDirection.BottomToTop),
            ("180", 15.0, -15.0, ReadoutDirection.BottomToTop),
            ("2", 15.0, 10015.0, ReadoutDirection.LeftToRight),
            ("3", 15.0, -10015.0, ReadoutDirection.RightToLeft),
        ]
        for code, declared, expected, direction in cases:
            # The extra keys belong BEFORE the t-row — after it they are
            # data rows and get dropped.
            header = HEADER.replace(
                "t,gx",
                f"frame_readout_direction,{code}\n"
                f"frame_readout_time,{declared}\nt,gx")
            path = _write_gcsv(tmp_path / "log.gcsv", [(0, 1, 0, 0)], header)
            meta = parse_gcsv(open(path, "rb").read())
            assert meta.frame_readout_time == pytest.approx(expected), code
            assert meta.frame_readout_direction == direction, code

    def test_no_readout_header_leaves_none(self, tmp_path):
        path = _write_gcsv(tmp_path / "log.gcsv", [(0, 1, 0, 0)])
        meta = parse_gcsv(open(path, "rb").read())
        assert meta.frame_readout_time is None

    def test_lensprofile_header_lands_in_metadata(self, tmp_path):
        header = HEADER.replace(
            "t,gx", "lensprofile,my-lens.json\nt,gx")
        path = _write_gcsv(tmp_path / "log.gcsv", [(0, 1, 0, 0)], header)
        meta = parse_gcsv(open(path, "rb").read())
        assert meta.lens_profile == "my-lens.json"

    def test_late_data_header_uses_tscale(self, tmp_path):
        """tscale is read when the t-row appears, not before."""
        path = _write_gcsv(tmp_path / "log.gcsv", [(5, 1, 0, 0)])
        meta = parse_gcsv(open(path, "rb").read())
        assert meta.raw_imu[0].timestamp_ms == pytest.approx(5.0)


class TestSidecarFallback:
    def test_empty_mp4_falls_back_to_gcsv(self, tmp_path):
        import av

        clip = tmp_path / "clip.mp4"
        container = av.open(str(clip), mode="w")
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
        img = np.zeros((24, 32, 3), np.uint8)
        img[:, :, 0] = 128
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        frame.pts = 0
        for pkt in stream.encode(frame):
            container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
        container.close()

        _write_gcsv(tmp_path / "clip.gcsv", [(0, 1, 0, 0), (10, 1, 0, 0)])
        meta = parse_telemetry_file(str(clip))
        assert meta.detected_source == "GCSV"
        assert len(meta.raw_imu) == 2

    def test_sidecar_extension_priority(self, tmp_path):
        (tmp_path / "clip.csv").write_text("not really")
        _write_gcsv(tmp_path / "clip.gcsv", [(0, 1, 0, 0)])
        empty = tmp_path / "clip.mp4"
        empty.write_bytes(b"\x00" * 64)  # unparseable
        # _find_sidecar picks .gcsv before .csv
        from pygyroflow.telemetry.parser import _find_sidecar

        assert _find_sidecar(str(empty)).endswith("clip.gcsv")

    def test_direct_gcsv_file_parses(self, tmp_path):
        path = _write_gcsv(tmp_path / "log.gcsv", [(0, 1, 0, 0)])
        meta = parse_telemetry_file(path)
        assert meta.detected_source == "GCSV"
        assert len(meta.raw_imu) == 1

    def test_non_gcsv_text_file_still_refused(self, tmp_path):
        path = tmp_path / "log.csv"
        path.write_text("a,b\n1,2\n")
        from pygyroflow.types.errors import TelemetryParseError

        with pytest.raises(TelemetryParseError):
            parse_telemetry_file(str(path))
