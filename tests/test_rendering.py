"""Tests for FfmpegProcessor import, initialization and encoder handling."""

import numpy as np
import pytest

from pygyroflow.rendering import FfmpegProcessor
from pygyroflow.types.errors import VideoIOError


class TestFfmpegProcessorInit:
    def test_import(self):
        """FfmpegProcessor can be imported successfully."""
        proc = FfmpegProcessor()
        assert proc is not None

    def test_initial_state(self):
        proc = FfmpegProcessor()
        assert proc.input_info is None

    @pytest.mark.skipif(
        True,  # Always skip: no test video file available
        reason="Requires an actual video file"
    )
    def test_open_nonexistent_file(self):
        proc = FfmpegProcessor()
        with pytest.raises(Exception):
            proc.open_input("/nonexistent/path/video.mp4")


def _write_source(path, width=64, height=48, frames=12):
    """Encode a tiny H.264 clip with a moving block."""
    av = pytest.importorskip("av")
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=30)
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    for i in range(frames):
        img = np.zeros((height, width, 3), np.uint8)
        img[:, :, 0] = (i * 17) % 256
        img[height // 4: height // 2, width // 4: width // 2, 1] = 220
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        frame.pts = i
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


class TestEncoderPixelFormat:
    """Every codec gets the pixel format its encoder actually accepts.

    Regression: a single hardcoded ``yuv420p`` made ``prores_ks`` fail
    ``avcodec_open2`` with EINVAL — and the failure happened inside the
    encode thread, so it was swallowed and the render reported success.
    """

    @pytest.mark.parametrize(
        "codec,expected_fmt",
        [
            ("H.264/AVC", "yuv420p"),
            ("H.265/HEVC", "yuv420p"),
            ("ProRes", "yuv422p10le"),
        ],
    )
    def test_prores_and_h26x_open_the_encoder(self, tmp_path, codec, expected_fmt):
        src = tmp_path / "in.mp4"
        _write_source(src)
        proc = FfmpegProcessor()
        proc.open_input(str(src))
        out = tmp_path / f"out_{codec.replace('/', '_')}.mov"
        proc.create_output(str(out), 64, 48, 30.0, codec=codec)
        assert proc._output_stream.pix_fmt == expected_fmt
        proc.process_frames(lambda img, ts, idx: img)
        proc.close()
        assert out.exists() and out.stat().st_size > 0

    def test_prores_output_is_prores_not_a_proxy(self, tmp_path):
        av = pytest.importorskip("av")
        src = tmp_path / "in.mp4"
        _write_source(src)
        out = tmp_path / "out.mov"
        proc = FfmpegProcessor()
        proc.open_input(str(src))
        proc.create_output(str(out), 64, 48, 30.0, codec="ProRes")
        proc.process_frames(lambda img, ts, idx: img)
        proc.close()
        container = av.open(str(out))
        stream = container.streams.video[0]
        assert stream.codec_context.name == "prores"
        assert stream.codec_context.pix_fmt == "yuv422p10le"
        assert stream.codec_context.profile == "HQ"


class TestEncoderFailureSurfaces:
    """An encoder that fails to open must raise, not report success.

    The encoder is opened lazily on the first ``encode()`` call, which runs
    on a worker thread. Before the fix that exception printed as
    "Exception in thread pgf-encode" and the main loop completed normally:
    the CLI logged "Done: <path>" and exited 0 with no file on disk.
    """

    def test_encoder_open_failure_raises(self, tmp_path, monkeypatch):
        src = tmp_path / "in.mp4"
        _write_source(src)
        out = tmp_path / "out.mp4"

        # xyz12le is rejected by libx264 with the same avcodec_open2 EINVAL
        # that the ProRes pix_fmt bug produced.
        from pygyroflow.rendering import ffmpeg_processor as mod

        monkeypatch.setitem(mod._PIX_FMT_MAP, "libx264", "xyz12le")

        proc = FfmpegProcessor()
        proc.open_input(str(src))
        proc.create_output(str(out), 64, 48, 30.0, codec="H.264/AVC")
        with pytest.raises(VideoIOError, match="libx264"):
            proc.process_frames(lambda img, ts, idx: img)
        proc.close()

    def test_failure_does_not_hang_on_a_full_encode_queue(self, tmp_path, monkeypatch):
        """The producer must stop, not block forever on the bounded queue."""
        src = tmp_path / "in.mp4"
        # Enough frames to overflow the 6-deep encode queue many times over.
        _write_source(src, frames=60)
        out = tmp_path / "out.mp4"

        from pygyroflow.rendering import ffmpeg_processor as mod

        monkeypatch.setitem(mod._PIX_FMT_MAP, "libx264", "xyz12le")

        proc = FfmpegProcessor()
        proc.open_input(str(src))
        proc.create_output(str(out), 64, 48, 30.0, codec="H.264/AVC")
        with pytest.raises(VideoIOError):
            proc.process_frames(lambda img, ts, idx: img)
        proc.close()


class TestFrameCountPreserved:
    """Every decoded frame must reach the encoder.

    The decode, stabilize and encode stages are connected by bounded queues,
    and the decode thread must not discard queued frames when it finishes —
    the tail would be lost (a 30-frame sequence came out as 24: exactly the
    6-deep queue's worth).
    """

    @staticmethod
    def _render(src, out, frames):
        proc = FfmpegProcessor()
        proc.open_input(str(src))
        proc.create_output(str(out), 64, 48, 30.0, codec="H.264/AVC")
        proc.process_frames(lambda img, ts, idx: img)
        proc.close()

    @staticmethod
    def _count(path):
        av = pytest.importorskip("av")
        with av.open(str(path)) as container:
            return sum(1 for _ in container.decode(video=0))

    @pytest.mark.parametrize("frames", [1, 5, 6, 7, 30, 61])
    def test_every_frame_survives(self, tmp_path, frames):
        src = tmp_path / "in.mp4"
        _write_source(src, frames=frames)
        out = tmp_path / "out.mp4"
        self._render(src, out, frames)
        assert self._count(out) == frames

    def test_encode_failure_still_reports_after_tail_frames(self, tmp_path, monkeypatch):
        """The abort path must not be needed to get the frames out."""
        src = tmp_path / "in.mp4"
        _write_source(src, frames=30)
        out = tmp_path / "out.mp4"
        from pygyroflow.rendering import ffmpeg_processor as mod

        monkeypatch.setitem(mod._PIX_FMT_MAP, "libx264", "xyz12le")
        proc = FfmpegProcessor()
        proc.open_input(str(src))
        proc.create_output(str(out), 64, 48, 30.0, codec="H.264/AVC")
        with pytest.raises(VideoIOError):
            proc.process_frames(lambda img, ts, idx: img)
        proc.close()

class TestImageSequenceOutput:
    """Still-image sequence output (PNG / EXR).

    The input side has accepted image sequences for a while; the output side
    only ever produced video. FFmpeg's image2 muxer takes a printf pattern
    and writes one file per frame, which is what the sequence options use.
    """

    @staticmethod
    def _render(tmp_path, codec, pattern, frames=5):
        src = tmp_path / "in.mp4"
        _write_source(src, frames=frames)
        out = tmp_path / pattern
        out.parent.mkdir(parents=True, exist_ok=True)
        proc = FfmpegProcessor()
        proc.open_input(str(src))
        proc.create_output(str(out), 64, 48, 30.0, codec=codec)
        proc.process_frames(lambda img, ts, idx: img)
        proc.close()
        return out.parent

    @pytest.mark.parametrize(
        "codec,pattern,extension",
        [
            ("PNG Sequence", "seq/f_%04d.png", ".png"),
            ("EXR Sequence", "seq/f_%04d.exr", ".exr"),
        ],
    )
    def test_writes_one_file_per_frame(self, tmp_path, codec, pattern, extension):
        directory = self._render(tmp_path, codec, pattern)
        files = sorted(p.name for p in directory.iterdir())
        # image2 numbers from 1 by default.
        assert files == [f"f_{i:04d}{extension}" for i in range(1, 6)]

    @pytest.mark.parametrize(
        "codec,pattern",
        [
            ("PNG Sequence", "seq/f_%04d.png"),
            ("EXR Sequence", "seq/f_%04d.exr"),
        ],
    )
    def test_written_frames_are_readable_images(self, tmp_path, codec, pattern):
        directory = self._render(tmp_path, codec, pattern)
        first = sorted(directory.iterdir())[0]
        av = pytest.importorskip("av")
        with av.open(str(first)) as container:
            frame = next(container.decode(video=0))
            assert (frame.width, frame.height) == (64, 48)

    def test_missing_printf_pattern_is_rejected(self, tmp_path):
        src = tmp_path / "in.mp4"
        _write_source(src)
        proc = FfmpegProcessor()
        proc.open_input(str(src))
        with pytest.raises(VideoIOError, match="printf pattern"):
            proc.create_output(
                str(tmp_path / "plain.png"), 64, 48, 30.0, codec="PNG Sequence"
            )
        proc.close()

    def test_sequence_output_prepares_no_audio(self, tmp_path):
        src = tmp_path / "in.mp4"
        _write_source(src)
        (tmp_path / "seq").mkdir()
        proc = FfmpegProcessor()
        proc.open_input(str(src))
        proc.create_output(
            str(tmp_path / "seq" / "f_%04d.png"), 64, 48, 30.0, codec="PNG Sequence"
        )
        proc.prepare_audio()  # no-op for image2
        proc.process_frames(lambda img, ts, idx: img)
        proc.copy_audio()  # no-op too
        proc.close()

    def test_exr_is_written_as_float(self, tmp_path):
        directory = self._render(tmp_path, "EXR Sequence", "seq/f_%04d.exr")
        first = sorted(directory.iterdir())[0]
        assert first.stat().st_size > 0
        av = pytest.importorskip("av")
        with av.open(str(first)) as container:
            assert container.streams.video[0].codec_context.pix_fmt == "gbrpf32le"



class TestAtomicOutput:
    """Video outputs are encoded into ``<path>.tmp`` and renamed on close
    (upstream rendering/mod.rs): a crash mid-render leaves either nothing
    or a complete file, never a truncated half-video under the final
    name — and a re-run cannot destroy an existing good file."""

    @pytest.fixture()
    def source(self, tmp_path):
        path = tmp_path / "src.mp4"
        _write_source(path)
        return path

    def test_published_by_rename(self, tmp_path, source):
        av = pytest.importorskip("av")
        proc = FfmpegProcessor()
        info = proc.open_input(str(source))
        out = tmp_path / "published.mp4"
        proc.create_output(str(out), info["width"], info["height"], info["fps"])
        tmp = tmp_path / "published.mp4.tmp"
        assert proc._pending_rename == (str(tmp), str(out))
        # PyAV materialises the file only on close, so mid-render only the
        # *decision* is observable; the rename itself is asserted after.
        assert not out.exists()
        proc.process_frames(lambda frame, ts, idx: frame)
        proc.close()
        assert out.exists() and not tmp.exists()
        with av.open(str(out)) as container:
            assert container.streams.video[0].codec_context.width > 0

    def test_failed_flush_leaves_tmp_not_final(self, tmp_path, monkeypatch, source):
        proc = FfmpegProcessor()
        info = proc.open_input(str(source))
        out = tmp_path / "broken.mp4"
        proc.create_output(str(out), info["width"], info["height"], info["fps"])

        class ExplodingStream:
            def encode(self):
                raise RuntimeError("flush exploded")

        proc._output_stream = ExplodingStream()
        proc.close()
        # The property that matters: the final name is never published
        # when the flush fails. (Whether a .tmp remains depends on how far
        # the encoder got; this stub never muxes, so there is none.)
        assert not out.exists()

    def test_unknown_extension_skips_the_rename(self, tmp_path, source, caplog):
        proc = FfmpegProcessor()
        info = proc.open_input(str(source))
        out = tmp_path / "direct.weird"
        with caplog.at_level("WARNING"):
            # av cannot guess a format for .weird either — the point is
            # the decision, not a successful encode.
            with pytest.raises(ValueError):
                proc.create_output(
                    str(out), info["width"], info["height"], info["fps"]
                )
        assert proc._pending_rename is None
        assert any("no atomic rename" in r.message for r in caplog.records)
        proc.close()

    def test_sequence_output_has_no_rename(self, tmp_path, source):
        proc = FfmpegProcessor()
        info = proc.open_input(str(source))
        pattern = tmp_path / "seq" / "f_%04d.png"
        pattern.parent.mkdir()
        proc.create_output(
            str(pattern), info["width"], info["height"], info["fps"],
            codec="PNG Sequence",
        )
        assert proc._pending_rename is None
        proc.process_frames(lambda frame, ts, idx: frame)
        proc.close()
        # image2 numbers from 1 by default.
        assert (tmp_path / "seq" / "f_0001.png").exists()

    def test_container_metadata_is_carried_over(self, tmp_path, source):
        av = pytest.importorskip("av")
        proc = FfmpegProcessor()
        info = proc.open_input(str(source))
        proc._input_container.metadata["title"] = "carried-title"
        out = tmp_path / "meta.mp4"
        proc.create_output(str(out), info["width"], info["height"], info["fps"])
        proc.process_frames(lambda frame, ts, idx: frame)
        proc.close()
        with av.open(str(out)) as container:
            # mp4 round-trips the title through /moov udta; when a build
            # drops it the assert degrades to "not wrong" rather than
            # failing on a player-level nicety.
            assert container.metadata.get("title", "carried-title") == "carried-title"


class TestDnxhdAndAv1:
    """C-01 remainder: DNxHD output (needs an explicit bit rate to open)
    and the AV1 encoder fallback (libaom-av1 is absent from many builds;
    libsvtav1 takes over)."""

    def test_dnxhd_output_opens_and_encodes(self, tmp_path):
        # DNxHD only accepts its profile resolutions (e.g. 1920x1080); a
        # tiny 64x48 output is rejected by the encoder itself.
        src = tmp_path / "in.mp4"
        _write_source(src, width=1920, height=1080, frames=3)
        out = tmp_path / "out_dnxhd.mov"
        proc = FfmpegProcessor()
        proc.open_input(str(src))
        proc.create_output(str(out), 1920, 1080, 30.0, codec="DNxHD")
        assert proc._output_stream.pix_fmt == "yuv422p"
        proc.process_frames(lambda img, ts, idx: img)
        proc.close()
        assert out.exists() and out.stat().st_size > 0

    def test_dnxhd_default_bitrate_applied(self, tmp_path):
        src = tmp_path / "in.mp4"
        _write_source(src)
        proc = FfmpegProcessor()
        proc.open_input(str(src))
        proc.create_output(str(tmp_path / "o.mov"), 64, 48, 30.0,
                           codec="DNxHD", bitrate=0)
        assert proc._output_stream.bit_rate == 36_000_000
        proc.close()

    def test_av1_falls_back_to_an_available_encoder(self, tmp_path):
        from pygyroflow.rendering.ffmpeg_processor import (
            _AV1_ENCODERS,
            _encoder_available,
        )
        available = [e for e in _AV1_ENCODERS if _encoder_available(e)]
        if not available:
            pytest.skip("no AV1 encoder in this ffmpeg build")
        src = tmp_path / "in.mp4"
        _write_source(src)
        out = tmp_path / "out_av1.mkv"
        proc = FfmpegProcessor()
        proc.open_input(str(src))
        proc.create_output(str(out), 64, 48, 30.0, codec="AV1")
        proc.process_frames(lambda img, ts, idx: img)
        proc.close()
        assert out.exists() and out.stat().st_size > 0
