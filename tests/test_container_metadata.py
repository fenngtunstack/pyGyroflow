"""Container display rotation (tkhd matrix).

A phone or action camera records one way and stores a rotation in the track
header; players apply it at playback. PyAV does not surface it, so it is read
from the box structure directly. Upstream uses ``av_display_rotation_get``
and folds the result in as ``(360 - rotation) % 360`` (render_queue.rs).

The fixtures are hand-built MP4 headers, so the parser is tested without
depending on what an encoder happens to write.
"""

import math
import shutil
import struct
import subprocess

import numpy as np
import pytest

from pygyroflow.rendering.container_metadata import read_display_rotation

_IDENTITY = [65536, 0, 0, 0, 65536, 0, 0, 0, 1073741824]


def _box(fourcc: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + fourcc + payload


def _tkhd(matrix, version=0, handler=b"vide"):
    """A tkhd box laid out exactly as ISO/IEC 14496-12 specifies.

    Version 0 puts the matrix at body offset 40, version 1 at 52.
    """
    if version == 0:
        # version/flags, creation, modification, track_ID, reserved, duration
        head = struct.pack(">IIIIII", 0, 0, 0, 1, 0, 1000)
    else:
        head = struct.pack(">I", 0x01000000)          # version + flags
        head += struct.pack(">QQ", 0, 0)              # creation, modification
        head += struct.pack(">II", 1, 0)              # track_ID, reserved
        head += struct.pack(">Q", 1000)               # duration
    head += struct.pack(">II", 0, 0)  # reserved[2]
    head += struct.pack(">HH", 0, 0)  # layer, alternate_group
    head += struct.pack(">HH", 0, 0)  # volume, reserved
    body = head + struct.pack(">9i", *matrix)
    body += struct.pack(">II", 320 << 16, 240 << 16)
    hdlr = _box(b"hdlr", b"\x00" * 8 + handler + b"\x00" * 12)
    mdia = _box(b"mdia", hdlr)
    return _box(b"tkhd", body), mdia


def _mp4(tracks):
    """*tracks* is a list of (tkhd_matrix, version, handler)."""
    traks = b""
    for matrix, version, handler in tracks:
        tkhd, mdia = _tkhd(matrix, version=version, handler=handler)
        traks += _box(b"trak", tkhd + mdia)
    return _box(b"moov", _box(b"mvhd", b"\x00" * 96) + traks)


def _rotation_matrix(degrees):
    """The matrix FFmpeg writes for a display rotation of *degrees*.

    ``av_display_rotation_get`` returns ``-atan2(m[1], m[0])``, so a rotation
    of theta needs ``m[0] = cos(theta)`` and ``m[1] = -sin(theta)``.
    Confirmed against a real file: ``-display_rotation 90`` yields
    ``[0, -65536, 0, 65536, ...]``.
    """
    rad = math.radians(degrees)
    cos = round(math.cos(rad) * 65536)
    sin = round(math.sin(rad) * 65536)
    return [cos, -sin, 0, sin, cos, 0, 0, 0, 1073741824]


def _normalise(degrees):
    return degrees % 360.0


class TestDisplayRotation:
    @pytest.mark.parametrize("degrees", [0, 90, 180, 270])
    def test_reads_the_matrix(self, tmp_path, degrees):
        """Compared modulo 360: +180 and -180 are the same rotation."""
        path = tmp_path / "r.mp4"
        path.write_bytes(_mp4([(_rotation_matrix(degrees), 0, b"vide")]))
        assert _normalise(read_display_rotation(str(path))) == pytest.approx(
            _normalise(degrees), abs=0.01
        )

    def test_identity_is_zero(self, tmp_path):
        path = tmp_path / "r.mp4"
        path.write_bytes(_mp4([(_IDENTITY, 0, b"vide")]))
        assert read_display_rotation(str(path)) == 0.0

    def test_version_1_header(self, tmp_path):
        """v1 widens the timestamps, moving the matrix 12 bytes later."""
        path = tmp_path / "r.mp4"
        path.write_bytes(_mp4([(_rotation_matrix(90), 1, b"vide")]))
        assert _normalise(read_display_rotation(str(path))) == pytest.approx(
            90.0, abs=0.01
        )

    def test_video_track_wins_over_audio(self, tmp_path):
        """An audio track carries an identity matrix and must not answer."""
        path = tmp_path / "r.mp4"
        path.write_bytes(
            _mp4(
                [
                    (_IDENTITY, 0, b"soun"),
                    (_rotation_matrix(270), 0, b"vide"),
                ]
            )
        )
        assert _normalise(read_display_rotation(str(path))) == pytest.approx(
            270.0, abs=0.01
        )

    def test_missing_moov_is_zero(self, tmp_path):
        path = tmp_path / "r.mp4"
        path.write_bytes(b"\x00" * 64)
        assert read_display_rotation(str(path)) == 0.0

    def test_missing_file_is_zero(self, tmp_path):
        assert read_display_rotation(str(tmp_path / "nope.mp4")) == 0.0

    def test_truncated_matrix_is_zero(self, tmp_path):
        """A tkhd that stops before the matrix must not read past the end."""
        path = tmp_path / "r.mp4"
        tkhd = _box(b"tkhd", b"\x00" * 20)
        path.write_bytes(_box(b"moov", _box(b"trak", tkhd)))
        assert read_display_rotation(str(path)) == 0.0


_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")


@pytest.mark.skipif(not (_FFMPEG and _FFPROBE), reason="ffmpeg/ffprobe missing")
class TestAgainstFfmpeg:
    """Cross-check against the tool that writes the matrix we parse."""

    @staticmethod
    def _make(tmp_path, degrees):
        import av

        src = tmp_path / "src.mp4"
        container = av.open(str(src), mode="w")
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(4):
            frame = av.VideoFrame.from_ndarray(
                np.zeros((48, 64, 3), np.uint8), format="rgb24"
            )
            frame.pts = i
            for pkt in stream.encode(frame):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
        container.close()

        out = tmp_path / f"rot{degrees}.mp4"
        subprocess.run(
            [_FFMPEG, "-y", "-loglevel", "error", "-display_rotation", str(degrees),
             "-i", str(src), "-c", "copy", str(out)],
            check=True,
        )
        return out

    @staticmethod
    def _ffprobe(path):
        result = subprocess.run(
            [_FFPROBE, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "side_data=rotation", "-of", "default=nw=1:nk=1",
             str(path)],
            capture_output=True, text=True, check=True,
        )
        first = result.stdout.strip().split("\n")[0]
        return float(first) if first else 0.0

    @pytest.mark.parametrize("degrees", [90, 180, 270])
    def test_matches_ffprobe(self, tmp_path, degrees):
        path = self._make(tmp_path, degrees)
        assert read_display_rotation(str(path)) == pytest.approx(
            self._ffprobe(path), abs=0.01
        )

    def test_unrotated_is_zero(self, tmp_path):
        path = self._make(tmp_path, 90)
        assert read_display_rotation(str(tmp_path / "src.mp4")) == 0.0
        assert self._ffprobe(path) != 0.0

@pytest.mark.skipif(not (_FFMPEG and _FFPROBE), reason="ffmpeg/ffprobe missing")
class TestAgainstFfmpeg:
    """Cross-check against the tool that writes the matrix we parse."""

    @staticmethod
    def _make(tmp_path, degrees):
        import av

        src = tmp_path / "src.mp4"
        container = av.open(str(src), mode="w")
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(6):
            frame = av.VideoFrame.from_ndarray(
                np.zeros((48, 64, 3), np.uint8), format="rgb24"
            )
            frame.pts = i
            for pkt in stream.encode(frame):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
        container.close()

        out = tmp_path / f"rot{degrees}.mp4"
        subprocess.run(
            [_FFMPEG, "-y", "-loglevel", "error", "-display_rotation", str(degrees),
             "-i", str(src), "-c", "copy", str(out)],
            check=True,
        )
        return out

    @staticmethod
    def _ffprobe(path):
        result = subprocess.run(
            [_FFPROBE, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "side_data=rotation", "-of", "default=nw=1:nk=1",
             str(path)],
            capture_output=True, text=True, check=True,
        )
        first = result.stdout.strip().split("\n")[0]
        return float(first) if first else 0.0

    @pytest.mark.parametrize("degrees", [90, 180, 270])
    def test_matches_ffprobe(self, tmp_path, degrees):
        path = self._make(tmp_path, degrees)
        assert _normalise(read_display_rotation(str(path))) == pytest.approx(
            _normalise(self._ffprobe(path)), abs=0.01
        )

    def test_unrotated_source_is_zero(self, tmp_path):
        path = self._make(tmp_path, 90)
        assert read_display_rotation(str(tmp_path / "src.mp4")) == 0.0
        assert self._ffprobe(path) != 0.0

    @pytest.mark.parametrize(
        "degrees,expected",
        [(90, 270.0), (180, 180.0), (270, 90.0)],
    )
    def test_load_video_folds_it_into_video_rotation(
        self, tmp_path, degrees, expected
    ):
        """load_video applies (360 - rotation) % 360, as upstream does."""
        from pygyroflow.manager import StabilizationManager

        path = self._make(tmp_path, degrees)
        mgr = StabilizationManager()
        mgr.load_video(str(path))
        assert mgr.params.video_rotation == pytest.approx(expected, abs=0.01)

    def test_load_video_leaves_unrotated_alone(self, tmp_path):
        from pygyroflow.manager import StabilizationManager

        self._make(tmp_path, 90)  # writes src.mp4 too
        mgr = StabilizationManager()
        mgr.load_video(str(tmp_path / "src.mp4"))
        assert mgr.params.video_rotation == 0.0
