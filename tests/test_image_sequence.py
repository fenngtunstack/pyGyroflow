# -*- coding: utf-8 -*-
"""Tests for image sequence (EXR/PNG/...) input.

Covers the three ways a sequence can be addressed (directory, printf pattern,
single frame), the resolution of padding/start-number against the files on
disk, the avformat options handed to FFmpeg's image2 demuxer, and a full
load -> stabilize -> render pass on a synthetic PNG sequence.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from fractions import Fraction

import numpy as np
import pytest

from pygyroflow.rendering.image_sequence import (
    format_options,
    is_image_file,
    looks_like_image_sequence,
    resolve_image_sequence,
    sequence_output_stem,
)
from pygyroflow.types.errors import VideoIOError

av = pytest.importorskip("av", reason="PyAV required for image sequence tests")


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #


def write_png_frames(directory, indexes, name="frame_{:04d}.png", width=64, height=48):
    """Write a numbered PNG run; returns the list of paths."""
    import cv2

    os.makedirs(directory, exist_ok=True)
    paths = []
    for i in indexes:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :, 1] = 32
        cv2.circle(img, (8 + i * 4, height // 2), 6, (255, 255, 255), -1)
        path = os.path.join(directory, name.format(i))
        cv2.imwrite(path, img)
        paths.append(path)
    return paths


def open_sequence(path, fps=None):
    from pygyroflow.rendering import FfmpegProcessor

    proc = FfmpegProcessor()
    return proc, proc.open_input(str(path), fps=fps)


def decode_all(path, fps=None) -> list[np.ndarray]:
    """Decode a sequence the way FfmpegProcessor does: resolve -> av.open
    with the image2 options -> rgb24 arrays.  Used where the assertions are
    about pixels rather than about the processor's bookkeeping."""
    sequence = resolve_image_sequence(str(path))
    container = av.open(
        sequence.pattern,
        format="image2" if sequence.is_sequence else None,
        options=format_options(sequence, fps),
    )
    try:
        stream = container.streams.video[0]
        return [f.to_ndarray(format="rgb24") for pkt in container.demux(stream) for f in pkt.decode()]
    finally:
        container.close()


def count_frames(path):
    container = av.open(str(path))
    total = 0
    for packet in container.demux(container.streams.video[0]):
        total += len(packet.decode())
    container.close()
    return total


def _write_video(path, frames=10, fps=30, width=64, height=48):
    """Encode a tiny synthetic video, so the sequence path can be contrasted
    with the plain video path."""
    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=Fraction(fps, 1))
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    for i in range(frames):
        img = np.full((height, width, 3), i * 3 % 255, np.uint8)
        for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def first_frame(path) -> np.ndarray:
    """First decodable frame of a video, as an RGB array."""
    container = av.open(str(path))
    try:
        for packet in container.demux(container.streams.video[0]):
            for frame in packet.decode():
                return frame.to_ndarray(format="rgb24")
    finally:
        container.close()
    raise AssertionError(f"no frames decoded from {path}")


# --------------------------------------------------------------------------- #
#  Detection                                                                   #
# --------------------------------------------------------------------------- #


class TestDetection:
    def test_image_file(self, tmp_path):
        assert is_image_file("shot.EXR")
        assert is_image_file("/a/b/frame_0001.png")
        assert not is_image_file("clip.mp4")
        assert not is_image_file("notes.txt")
        assert not is_image_file("project.gyroflow")

    def test_printf_pattern(self):
        assert looks_like_image_sequence("/shots/frame_%04d.exr")
        assert looks_like_image_sequence("frame_%d.png")
        assert not looks_like_image_sequence("/shots/clip.mp4")

    def test_directory_of_images(self, tmp_path):
        write_png_frames(str(tmp_path), range(1, 4))
        assert looks_like_image_sequence(str(tmp_path))

    def test_directory_without_images(self, tmp_path):
        (tmp_path / "clip.mp4").write_bytes(b"not a video")
        (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
        assert not looks_like_image_sequence(str(tmp_path))

    def test_empty_directory(self, tmp_path):
        assert not looks_like_image_sequence(str(tmp_path))

    def test_missing_path(self, tmp_path):
        # the shape check is about the extension; existence is resolve's job
        assert looks_like_image_sequence(str(tmp_path / "nope.exr"))
        with pytest.raises(FileNotFoundError):
            resolve_image_sequence(str(tmp_path / "nope.exr"))


# --------------------------------------------------------------------------- #
#  Resolution                                                                  #
# --------------------------------------------------------------------------- #


class TestResolve:
    def test_directory_uses_first_frame_as_template(self, tmp_path):
        write_png_frames(str(tmp_path), range(1, 6))

        seq = resolve_image_sequence(str(tmp_path))

        assert seq.is_sequence
        assert seq.pattern == os.path.join(str(tmp_path), "frame_%04d.png")
        assert seq.start_number == 1
        assert seq.frame_count == 5
        assert seq.pad_width == 4
        assert seq.extension == ".png"

    def test_natural_order_beats_lexicographic(self, tmp_path):
        # "p10_..." sorts before "p2_..." as a string; numerically p2 is first.
        write_png_frames(str(tmp_path), [1], name="p2_{:04d}.png")
        write_png_frames(str(tmp_path), [1], name="p10_{:04d}.png")

        seq = resolve_image_sequence(str(tmp_path))

        assert os.path.basename(seq.pattern) == "p2_%04d.png"

    def test_padded_series_is_counted_whole(self, tmp_path):
        write_png_frames(str(tmp_path), range(1, 13), name="f{:02d}.png")

        seq = resolve_image_sequence(str(tmp_path))

        assert os.path.basename(seq.pattern) == "f%02d.png"
        assert seq.start_number == 1
        assert seq.frame_count == 12

    def test_single_frame_infers_padding_and_start(self, tmp_path):
        write_png_frames(str(tmp_path), range(1, 8))

        seq = resolve_image_sequence(os.path.join(str(tmp_path), "frame_0003.png"))

        assert seq.start_number == 1  # inferred from the run on disk
        assert seq.frame_count == 7
        assert seq.pad_width == 4

    def test_zero_based_sequence(self, tmp_path):
        write_png_frames(str(tmp_path), range(0, 6), name="z{:03d}.png")

        seq = resolve_image_sequence(str(tmp_path))

        assert seq.start_number == 0
        assert seq.frame_count == 6
        assert seq.pattern.endswith("z%03d.png")

    def test_printf_pattern_passthrough(self, tmp_path):
        write_png_frames(str(tmp_path), range(1, 4))

        seq = resolve_image_sequence(os.path.join(str(tmp_path), "frame_%04d.png"))

        assert seq.pattern.endswith("frame_%04d.png")
        assert seq.frame_count == 3

    def test_gap_is_rejected_with_the_missing_frame_named(self, tmp_path):
        write_png_frames(str(tmp_path), [1, 2, 3, 5, 6])

        with pytest.raises(VideoIOError) as exc:
            resolve_image_sequence(str(tmp_path))

        assert "frame_0004.png" in str(exc.value)

    def test_literal_still_without_numbers(self, tmp_path):
        import cv2

        path = os.path.join(str(tmp_path), "poster.png")
        cv2.imwrite(path, np.zeros((16, 16, 3), np.uint8))

        seq = resolve_image_sequence(path)

        assert not seq.is_sequence
        assert seq.frame_count == 1
        assert seq.pattern == path

    def test_other_files_do_not_confuse_the_template(self, tmp_path):
        import cv2

        write_png_frames(str(tmp_path), range(1, 4))
        (tmp_path / "readme.txt").write_text("x", encoding="utf-8")
        (tmp_path / "thumb.jpg").write_bytes(b"jpeg bytes")  # not decodable, but an image by extension
        cv2.imwrite(os.path.join(str(tmp_path), "aaa_first.png"), np.zeros((16, 16, 3), np.uint8))

        seq = resolve_image_sequence(str(tmp_path))

        # the template is chosen by natural order, and only its own series counted
        assert seq.frame_count == 1
        assert os.path.basename(seq.pattern) == "aaa_first.png"

    def test_mixed_padding_is_not_merged(self, tmp_path):
        write_png_frames(str(tmp_path), [1, 2, 3], name="q{}.png")
        write_png_frames(str(tmp_path), [10, 11, 12], name="q{:03d}.png")

        seq = resolve_image_sequence(str(tmp_path))

        # one series only — the other padding width lives in a different pattern
        assert seq.pattern.endswith("q%01d.png")
        assert seq.pad_width == 1
        assert seq.frame_count == 3

    def test_directory_with_a_single_image_is_a_still(self, tmp_path):
        import cv2

        cv2.imwrite(os.path.join(str(tmp_path), "a.png"), np.zeros((8, 8, 3), np.uint8))
        (tmp_path / "b.mp4").write_bytes(b"")

        seq = resolve_image_sequence(str(tmp_path))
        assert seq.frame_count == 1

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_image_sequence(str(tmp_path))


# --------------------------------------------------------------------------- #
#  avformat options                                                            #
# --------------------------------------------------------------------------- #


class TestFormatOptions:
    def _seq(self, tmp_path, **kw):
        write_png_frames(str(tmp_path), range(1, 4), **kw)
        return resolve_image_sequence(str(tmp_path))

    def test_start_number_and_rate(self, tmp_path):
        options = format_options(self._seq(tmp_path), 30.0)
        assert options == {"start_number": "1", "framerate": "30"}

    def test_zero_start_number_is_preserved(self, tmp_path):
        write_png_frames(str(tmp_path), range(0, 4), name="z{:03d}.png")
        options = format_options(resolve_image_sequence(str(tmp_path)), 30.0)
        assert options["start_number"] == "0"

    def test_missing_fps_falls_back_to_ffmpeg_default(self, tmp_path):
        for fps in (None, 0.0):
            assert format_options(self._seq(tmp_path), fps)["framerate"] == "25"

    def test_ntsc_rates_go_over_1001(self, tmp_path):
        # upstream rendering::fps_to_rational — the container's rate for NTSC
        # material is x*1000/1001, not the decimal the user typed
        assert format_options(self._seq(tmp_path), 29.97)["framerate"] == "30000/1001"
        assert format_options(self._seq(tmp_path), 59.94)["framerate"] == "60000/1001"
        assert format_options(self._seq(tmp_path), 23.976)["framerate"] == "24000/1001"

    def test_integral_rates_are_plain_integers(self, tmp_path):
        assert format_options(self._seq(tmp_path), 30.0)["framerate"] == "30"
        assert format_options(self._seq(tmp_path), 24)["framerate"] == "24"

    def test_single_still_gets_no_start_number(self, tmp_path):
        import cv2

        path = os.path.join(str(tmp_path), "poster.png")
        cv2.imwrite(path, np.zeros((16, 16, 3), np.uint8))

        options = format_options(resolve_image_sequence(path), 30.0)
        assert "start_number" not in options
        assert options["framerate"] == "30"


class TestOutputStem:
    def test_directory(self, tmp_path):
        target = tmp_path / "shots"
        target.mkdir()
        assert sequence_output_stem(str(target)) == str(target)

    def test_directory_with_trailing_separator(self, tmp_path):
        target = tmp_path / "shots"
        target.mkdir()
        assert sequence_output_stem(str(target) + os.sep) == str(target)

    def test_pattern(self):
        assert sequence_output_stem("/a/shots/frame_%04d.exr") == "/a/shots/frame"

    def test_pattern_without_padding(self):
        assert sequence_output_stem("frame_%d.png") == "frame"

    def test_single_file(self):
        assert sequence_output_stem("/a/poster.png") == "/a/poster"


# --------------------------------------------------------------------------- #
#  FFmpeg open                                                                 #
# --------------------------------------------------------------------------- #


class TestOpenInput:
    def test_directory_sequence(self, tmp_path):
        write_png_frames(str(tmp_path), range(1, 11))

        _proc, info = open_sequence(tmp_path, fps=30.0)

        assert info["width"] == 64 and info["height"] == 48
        assert info["fps"] == 30.0
        assert info["frames"] == 10
        assert info["duration"] == pytest.approx(1000.0 / 3.0, rel=1e-3)
        assert info["codec"] == "png"

    def test_rate_comes_from_the_caller(self, tmp_path):
        write_png_frames(str(tmp_path), range(1, 11))

        _proc, info = open_sequence(tmp_path, fps=60.0)

        assert info["fps"] == 60.0
        assert info["duration"] == pytest.approx(1000.0 / 6.0, rel=1e-3)

    def test_zero_based_run_decodes_every_frame(self, tmp_path):
        write_png_frames(str(tmp_path), range(0, 6), name="z{:03d}.png")

        _proc, info = open_sequence(tmp_path, fps=30.0)

        # An off-by-one start_number silently drops the leading frame
        assert info["frames"] == 6

    def test_single_still_opens(self, tmp_path):
        import cv2

        path = tmp_path / "poster.png"
        cv2.imwrite(str(path), np.full((24, 32, 3), 77, np.uint8))

        _proc, info = open_sequence(path, fps=30.0)

        assert info["frames"] == 1
        assert info["width"] == 32 and info["height"] == 24

    def test_pattern_decodes_every_frame(self, tmp_path):
        write_png_frames(str(tmp_path), range(1, 8))

        decoded = decode_all(tmp_path, fps=30.0)

        assert len(decoded) == 7
        assert decoded[0].shape == (48, 64, 3)

    def test_gap_raises_before_opening(self, tmp_path):
        write_png_frames(str(tmp_path), [1, 2, 4])

        with pytest.raises(VideoIOError):
            open_sequence(tmp_path, fps=30.0)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg binary required to author EXR")
class TestExrInput:
    @pytest.fixture
    def exr_dir(self, tmp_path):
        pngs = tmp_path / "png"
        exrs = tmp_path / "exr"
        exrs.mkdir()
        write_png_frames(str(pngs), range(1, 6))
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-framerate", "30", "-start_number", "1",
                "-i", str(pngs / "frame_%04d.png"),
                "-pix_fmt", "gbrpf32le",
                str(exrs / "frame_%04d.exr"),
            ],
            check=True, capture_output=True, timeout=120,
        )
        return exrs

    def test_exr_sequence_resolves(self, exr_dir):
        seq = resolve_image_sequence(str(exr_dir))

        assert seq.extension == ".exr"
        assert seq.frame_count == 5
        assert seq.start_number == 1

    def test_exr_sequence_decodes_to_uint8(self, exr_dir):
        proc, info = open_sequence(exr_dir, fps=30.0)

        assert info["codec"] == "exr"
        assert info["frames"] == 5
        assert info["fps"] == 30.0
        proc.close()

        frames = np.stack(decode_all(exr_dir, fps=30.0))
        assert frames.shape == (5, 48, 64, 3)
        assert frames.dtype == np.uint8
        # the PNGs were written with a green background at g=32 (RGB after
        # the bgr24 -> rgb24 conversion, so the corner pixel is roughly
        # (0, 32, 0))
        assert int(frames[0, 2, 2, 1]) == pytest.approx(32, abs=2)
        assert int(frames[0, 2, 2, 0]) < 8 and int(frames[0, 2, 2, 2]) < 8


# --------------------------------------------------------------------------- #
#  Manager integration                                                         #
# --------------------------------------------------------------------------- #


class TestManagerSequence:
    def test_load_video_records_sequence_metadata(self, tmp_path):
        from pygyroflow.manager import StabilizationManager

        write_png_frames(str(tmp_path), range(1, 13))
        mgr = StabilizationManager()

        info = mgr.load_video(str(tmp_path), fps=30.0)

        assert info["frame_count"] == 12
        assert info["fps"] == 30.0
        assert info["image_sequence"] is not None
        assert mgr.input_file.image_sequence_fps == 30.0
        assert mgr.input_file.image_sequence_start == 1
        assert mgr.params.fps == 30.0
        assert mgr.params.frame_count == 12
        assert mgr.params.size == (64, 48)
        assert mgr.params.duration_ms == pytest.approx(400.0, rel=1e-3)

    def test_load_video_does_not_parse_telemetry_from_frames(self, tmp_path):
        from pygyroflow.manager import StabilizationManager

        write_png_frames(str(tmp_path), range(1, 6))
        mgr = StabilizationManager()

        mgr.load_video(str(tmp_path), fps=25.0)

        assert mgr.gyro.raw_imu == []
        assert mgr.gyro.quaternions == {}

    def test_load_video_default_fps_is_ffmpeg_default(self, tmp_path):
        from pygyroflow.manager import StabilizationManager

        write_png_frames(str(tmp_path), range(1, 6))
        mgr = StabilizationManager()

        info = mgr.load_video(str(tmp_path))

        assert info["fps"] == 25.0

    def test_load_video_rejects_missing_path(self, tmp_path):
        from pygyroflow.manager import StabilizationManager

        mgr = StabilizationManager()
        with pytest.raises(VideoIOError):
            mgr.load_video(str(tmp_path / "nope.mp4"))

    def test_load_video_reports_gap(self, tmp_path):
        from pygyroflow.manager import StabilizationManager

        write_png_frames(str(tmp_path), [1, 2, 4])
        mgr = StabilizationManager()

        with pytest.raises(VideoIOError) as exc:
            mgr.load_video(str(tmp_path), fps=30.0)
        assert "gap" in str(exc.value)

    def test_empty_directory_is_a_missing_file(self, tmp_path):
        from pygyroflow.manager import StabilizationManager

        mgr = StabilizationManager()
        with pytest.raises(VideoIOError):
            mgr.load_video(str(tmp_path))

    def test_fps_argument_does_not_override_a_video_container(self, tmp_path):
        """--fps speaks for sequences only; a video knows its own rate."""
        from pygyroflow.manager import StabilizationManager

        src = tmp_path / "clip.mp4"
        _write_video(src, frames=10, fps=30)

        mgr = StabilizationManager()
        info = mgr.load_video(str(src), fps=60.0)

        assert info["fps"] == pytest.approx(30.0)
        assert info["image_sequence"] is None
        assert mgr.input_file.image_sequence_fps == 0.0


# --------------------------------------------------------------------------- #
#  Full render                                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.slow
class TestSequenceRender:
    def test_sequence_renders_with_injected_gyro(self, tmp_path):
        """The --gyro use case: frames from disk + an external gyro timeline."""
        from pygyroflow.gyro_source import FileMetadata
        from pygyroflow.manager import StabilizationManager
        from pygyroflow.types.time_types import TimeIMU

        frames_dir = tmp_path / "frames"
        write_png_frames(str(frames_dir), range(1, 31), width=64, height=48)

        mgr = StabilizationManager()
        info = mgr.load_video(str(frames_dir), fps=30.0)
        assert info["frame_count"] == 30

        # 30 frames at 30 fps = 1 s; feed a rotating synthetic IMU signal the
        # way --gyro would supply real telemetry.  load_gyro_data() does the
        # same init before parsing, and load_from_telemetry rejects an
        # uninitialised (zero) duration.
        n = 200
        md = FileMetadata(detected_source="Synthetic", imu_orientation="XYZ")
        md.raw_imu = [
            TimeIMU(timestamp_ms=i * 5.0, gyro=np.array([0.0, 0.0, 45.0]), accl=None)
            for i in range(n)
        ]
        mgr.gyro.init_from_params(mgr.params.get_scaled_duration_ms())
        mgr.gyro.load_from_telemetry(md)
        mgr.recompute_blocking()
        assert len(mgr.gyro.quaternions) > 10

        out = tmp_path / "out.mp4"
        mgr.render(str(frames_dir), str(out), {"codec": "H.264/AVC", "audio": False})

        assert out.exists()
        assert count_frames(out) == 30

        container = av.open(str(out))
        stream = container.streams.video[0]
        assert (stream.width, stream.height) == (64, 48)
        assert float(stream.average_rate) == pytest.approx(30.0)
        container.close()

    def test_render_output_is_stabilized_not_passthrough(self, tmp_path):
        """With a gyro signal present the frames must actually be warped."""
        from pygyroflow.gyro_source import FileMetadata
        from pygyroflow.manager import StabilizationManager
        from pygyroflow.types.time_types import TimeIMU

        frames_dir = tmp_path / "frames"
        write_png_frames(str(frames_dir), range(1, 21), width=64, height=48)

        def render(with_gyro: bool, name: str) -> np.ndarray:
            mgr = StabilizationManager()
            mgr.load_video(str(frames_dir), fps=30.0)
            if with_gyro:
                md = FileMetadata(detected_source="Synthetic", imu_orientation="XYZ")
                md.raw_imu = [
                    TimeIMU(timestamp_ms=i * 5.0, gyro=np.array([8.0, 0.0, 0.0]), accl=None)
                    for i in range(160)
                ]
                mgr.gyro.init_from_params(mgr.params.get_scaled_duration_ms())
                mgr.gyro.load_from_telemetry(md)
            mgr.recompute_blocking()
            out = tmp_path / name
            mgr.render(str(frames_dir), str(out), {"codec": "H.264/AVC", "audio": False})
            return first_frame(out)

        plain = render(False, "plain.mp4")
        stabilized = render(True, "stab.mp4")

        assert plain.shape == stabilized.shape
        assert not np.array_equal(plain, stabilized)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
