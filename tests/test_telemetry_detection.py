"""Telemetry format detection.

Regression coverage for the full-file substring search that used to decide
the format: every marker is a fourcc, so searching the whole file finds them
inside compressed video data. On a 470 MB Sony clip `dvtm` occurs at 201 MB
and `djmd` at 236 MB, both inside the H.264 payload — the parser called the
file DJI, never reached the Sony branch, and returned zero IMU samples, so
the render "succeeded" with no stabilization at all.

Upstream only ever looks at the head and the tail (telemetry-parser
``util::read_beginning_and_end``, 5 MB each side below 5 GB).
"""

import os

import pytest

from pygyroflow.telemetry.parser import (
    _DETECT_WINDOW,
    _detect_buffer,
    _detect_dji,
    _detect_gopro,
    _detect_insta360,
    _detect_sony,
    detect_telemetry_format,
)

W = _DETECT_WINDOW


def _write(path, head: bytes, middle: bytes = b"", tail: bytes = b"", size: int = 0):
    """Write a file with *head* at offset 0, *tail* at EOF, *middle* centred.

    *size* == 0 concatenates the three parts; otherwise the file is zero
    padded to *size* bytes and the middle sits in the dead centre — well
    outside the head and tail detection windows.
    """
    if not size:
        path.write_bytes(head + middle + tail)
        return path

    assert size >= len(head) + len(middle) + len(tail)
    body = bytearray(size)
    body[: len(head)] = head
    if tail:
        body[-len(tail):] = tail
    if middle:
        at = (size - len(middle)) // 2
        body[at:at + len(middle)] = middle
    path.write_bytes(bytes(body))
    return path


# ---------------------------------------------------------------------------
# Marker semantics
# ---------------------------------------------------------------------------

class TestMarkerSemantics:
    def test_gopro_raw_stream_declared_by_devc_at_offset_zero(self):
        assert _detect_gopro(b"DEVC" + b"\x00" * 64)

    def test_gopro_detected_by_gpmf_box_marker(self):
        """Hero6 and later: the GPMF box abuts the stream start."""
        assert _detect_gopro(b"\x00" * 16 + b"GPMFDEVC" + b"\x00" * 16)

    def test_gopro_detected_by_metadata_track_handler(self):
        """Hero5/Session/Karma carry no GPMFDEVC — only the handler name."""
        assert _detect_gopro(b"\x00" * 16 + b"GoPro MET" + b"\x00" * 16)
        assert not _detect_gopro(b"\x00" * 64)

    def test_gopro_not_matched_by_bare_gpmd_fourcc(self):
        """`gpmd` alone is a track codec tag — it also occurs in H.264 data."""
        assert not _detect_gopro(b"\x00" * 16 + b"gpmd" + b"\x00" * 16)

    def test_dji_requires_the_handler_name(self):
        assert not _detect_dji(b"\x00" * 16 + b"djmd" + b"\x00" * 16)
        assert _detect_dji(b"\x00" * 8 + b"djmd" + b"\x00" * 8 + b"CAM meta")
        assert _detect_dji(b"\x00" * 8 + b"djmd" + b"\x00" * 8 + b"DJI meta")

    def test_dji_flight_log_csv(self):
        assert _detect_dji(b"Clock:Tick" + b"\x00" * 32 + b"IMU_ATTI(0):gyroX")
        assert not _detect_dji(b"Clock:Tick")

    def test_dji_not_matched_by_dvtm_alone(self):
        """The old rule was `dvtm` AND `DJI` — both occur in H.264 payloads."""
        assert not _detect_dji(b"\x00" * 16 + b"dvtm" + b"\x00" * 16 + b"DJI")

    def test_sony_manufacturer_xml(self):
        assert _detect_sony(b'<Device manufacturer="Sony" modelName="ZV-E1"/>')
        assert not _detect_sony(b'manufacturer="Canon"')

    def test_insta360_trailer_magic(self):
        magic = b"8db42d694ccc418790edff439fe026bf"
        assert _detect_insta360(b"\x00" * 32 + magic)
        assert not _detect_insta360(b"\x00" * 32 + magic + b"\x00")


# ---------------------------------------------------------------------------
# Bounded window
# ---------------------------------------------------------------------------

class TestDetectionWindow:
    def test_window_is_head_plus_tail(self, tmp_path):
        path = tmp_path / "big.mp4"
        _write(path, b"HEADMARK", tail=b"TAILMARK", size=4 * W)
        buf = _detect_buffer(str(path))
        assert len(buf) == 2 * W
        assert buf.startswith(b"HEADMARK")
        assert buf.endswith(b"TAILMARK")

    def test_small_file_is_read_whole(self, tmp_path):
        path = tmp_path / "small.mp4"
        _write(path, b"DEVC" + b"\x00" * 100)
        assert _detect_buffer(str(path)) == path.read_bytes()

    def test_marker_buried_in_the_middle_is_invisible(self, tmp_path):
        """The exact shape of the bug: a fourcc deep inside the payload."""
        path = tmp_path / "buried.mp4"
        _write(path, b"\x00" * 64, middle=b"djmd\x00CAM meta", size=4 * W)
        assert path.stat().st_size > 2 * W
        assert b"djmd" in path.read_bytes()  # present in the file ...
        assert not _detect_dji(_detect_buffer(str(path)))  # ... but not seen


class TestFalsePositiveRegression:
    """A Sony clip with video-data fourccs must still be a Sony clip."""

    @pytest.mark.parametrize(
        "buried",
        [
            b"\x00djmd\x00",          # issue-44-12: djmd at 236 MB
            b"\x00gpmd\x00",          # issue-44-23: gpmd at 33 MB
            b"\x00dvtm\x00DJI\x00",   # issue-44-30: dvtm at 201 MB, DJI at 8.9 MB
        ],
    )
    def test_sony_survives_buried_markers(self, tmp_path, buried):
        path = tmp_path / "sony.mp4"
        _write(
            path,
            head=b'<Device manufacturer="Sony" modelName="ZV-E1"/>',
            middle=b"\x00" * 1024 + buried,
            size=4 * W,
        )
        assert detect_telemetry_format(str(path)) == "Sony"

    def test_gopro_survives_buried_djmd(self, tmp_path):
        path = tmp_path / "gopro.mp4"
        _write(path, head=b"GPMFDEVC", middle=b"djmd C CAM meta", size=4 * W)
        assert detect_telemetry_format(str(path)) == "GoPro"

    def test_format_order_prefers_gopro_then_dji(self, tmp_path):
        path = tmp_path / "both.mp4"
        _write(path, head=b"GoPro MET" + b"CAM meta" + b"djmd")
        assert detect_telemetry_format(str(path)) == "GoPro"


class TestDetectTelemetryFormatDispatch:
    def test_unknown_for_garbage(self, tmp_path):
        path = tmp_path / "junk.mp4"
        _write(path, b"\x00" * 4096)
        assert detect_telemetry_format(str(path)) == "Unknown"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(OSError):
            detect_telemetry_format(str(tmp_path / "nope.mp4"))


# ---------------------------------------------------------------------------
# Real footage (skipped when the reference corpus is not present)
# ---------------------------------------------------------------------------

TESTVIDEOS = "/home/ft/workspace/testvideos"


@pytest.mark.skipif(
    not os.path.isdir(TESTVIDEOS), reason="reference footage not available"
)
class TestRealFootage:
    @pytest.mark.parametrize(
        "name,expected",
        [
            # All three were misdetected before the fix (zero IMU samples).
            ("issue-44-12-C1781---14-24mm-zooming.MP4", "Sony"),
            ("issue-44-23-C1792.MP4", "Sony"),
            ("issue-44-30-C1800---28-200-zooming-dist-compensation-on.MP4", "Sony"),
            # Older GoPro bodies: handler name only, no GPMFDEVC.
            ("extra-10-GoPro-Hero5-Session.MP4", "GoPro"),
            ("gpmf-03-hero5.mp4", "GoPro"),
            ("gpmf-12-karma.mp4", "GoPro"),
            ("extra-04-GoPro-Hero-6.MP4", "GoPro"),
            ("extra-01-DJI-Avata-4k60.MP4", "DJI"),
        ],
    )
    def test_detected_format(self, name, expected):
        path = os.path.join(TESTVIDEOS, name)
        if not os.path.isfile(path):
            pytest.skip(f"{name} not present")
        assert detect_telemetry_format(path) == expected
