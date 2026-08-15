"""PyAV-based video processor implementation.

Uses PyAV (Pythonic bindings for FFmpeg) for video decoding, filtering,
and encoding. Handles variable frame rate, missing audio streams, and
graceful resource cleanup.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from pygyroflow.types.errors import VideoIOError
from pygyroflow.rendering.video_processor import VideoProcessor, FrameCallback
from pygyroflow.rendering.audio_resampler import AudioResampler

log = logging.getLogger(__name__)

# Map from user-facing codec names to FFmpeg encoder names.
_CODEC_MAP: dict[str, str] = {
    "H.264/AVC": "libx264",
    "H.265/HEVC": "libx265",
    "ProRes": "prores_ks",
    "AV1": "libaom-av1",
    "VP9": "libvpx-vp9",
    "MPEG-4": "mpeg4",
}

# Default codec when an unknown name is given.
_DEFAULT_CODEC = "libx265"


class FfmpegProcessor(VideoProcessor):
    """Video processor backed by PyAV (FFmpeg).

    Typical usage::

        proc = FfmpegProcessor()
        info = proc.open_input("input.mp4")
        proc.create_output("output.mp4", info["width"], info["height"], info["fps"])
        proc.process_frames(my_stabilize_callback)
        proc.close()
    """

    def __init__(self) -> None:
        self._input_container: object | None = None
        self._output_container: object | None = None
        self._input_stream: object | None = None
        self._output_stream: object | None = None
        self._input_info: dict | None = None
        self._frame_index: int = 0
        self._audio_pairs: list = []

    # ------------------------------------------------------------------
    # VideoProcessor interface
    # ------------------------------------------------------------------

    @property
    def input_info(self) -> dict | None:
        return self._input_info

    def open_input(self, path: str) -> dict:
        """Open an input video and return its metadata."""
        try:
            import av  # type: ignore[import-untyped]
        except ImportError as exc:
            raise VideoIOError(
                "PyAV is required for video I/O. Install with: pip install av"
            ) from exc

        try:
            self._input_container = av.open(path)
        except av.error.InvalidDataError as exc:
            raise VideoIOError(f"Cannot open video file: {path}") from exc

        video_streams = self._input_container.streams.video
        if not video_streams:
            self._input_container.close()
            raise VideoIOError(f"No video stream found in: {path}")

        self._input_stream = video_streams[0]
        self._input_stream.thread_type = "AUTO"

        # Compute fps safely -- some containers report None.
        fps = 30.0
        if self._input_stream.average_rate is not None:
            fps = float(self._input_stream.average_rate)
        elif self._input_stream.framerate is not None:
            fps = float(self._input_stream.framerate)

        # Duration in milliseconds.
        duration_ms = 0.0
        if self._input_stream.duration is not None:
            tb = float(self._input_stream.time_base)
            duration_ms = float(self._input_stream.duration) * tb * 1000.0
        elif self._input_container.duration is not None:
            duration_ms = float(self._input_container.duration) / 1000.0

        info = {
            "width": self._input_stream.width,
            "height": self._input_stream.height,
            "fps": fps,
            "frames": (
                self._input_stream.frames
                if self._input_stream.frames
                else 0
            ),
            "duration": duration_ms,
            "codec": (
                self._input_stream.codec_context.name
                if self._input_stream.codec_context
                else "unknown"
            ),
        }
        self._input_info = info
        self._frame_index = 0
        return info

    def create_output(
        self,
        path: str,
        width: int,
        height: int,
        fps: float,
        codec: str = "H.265/HEVC",
        bitrate: float = 0.0,
    ) -> None:
        """Set up the output encoder."""
        import av  # type: ignore[import-untyped]

        if self._input_container is None:
            raise VideoIOError("No input is open; call open_input() first")

        codec_name = _CODEC_MAP.get(codec, _DEFAULT_CODEC)
        self._output_container = av.open(path, mode="w")
        from fractions import Fraction
        fps_frac = Fraction(fps).limit_denominator(100000)
        self._output_stream = self._output_container.add_stream(
            codec_name, rate=fps_frac
        )
        self._output_stream.width = width
        self._output_stream.height = height
        # pix_fmt must be yuv420p for most encoder compatibility.
        self._output_stream.pix_fmt = "yuv420p"

        if bitrate > 0:
            self._output_stream.bit_rate = int(bitrate * 1_000_000)

    def prepare_audio(self) -> None:
        """Add output audio streams mirroring the input's audio streams.

        Must run before any packet is muxed (i.e. before ``process_frames``):
        the container header is written on the first mux, and streams added
        afterwards get a zero time base and cannot be muxed
        (``ValueError: Cannot rebase to zero time``).
        The actual audio packets are moved later by :meth:`copy_audio`.
        """
        if self._input_container is None or self._output_container is None:
            raise VideoIOError("Both input and output must be opened first")

        from pygyroflow.rendering.audio_resampler import prepare_audio_streams

        self._audio_pairs = prepare_audio_streams(
            self._input_container, self._output_container
        )

    def process_frames(self, callback: FrameCallback) -> None:
        """Decode all frames, apply *callback*, encode to output.

        The input decoder is explicitly flushed after the demux loop: with
        multi-threaded decode (``thread_type="AUTO"``) the decoder buffers
        up to ~thread-count frames that only come out on flush. The demux
        loop's ``dts is None`` packets are container flush markers and are
        skipped, so without this the trailing frames would be lost
        (~11 frames on a 438-frame GoPro clip).
        """
        import av  # type: ignore[import-untyped]

        if self._input_container is None or self._output_container is None:
            raise VideoIOError("Both input and output must be opened first")

        self._frame_index = 0
        in_info = self._input_info or {}
        tb = float(self._input_stream.time_base)
        fallback_fps = in_info.get("fps", 30.0)

        def to_item(frame: av.VideoFrame, seq: int):
            img = frame.to_ndarray(format="rgb24")
            timestamp_ms = (
                float(frame.pts) * tb * 1000.0
                if frame.pts is not None
                else seq * (1000.0 / fallback_fps)
            )
            return img, timestamp_ms

        # Three-stage pipeline: decode thread -> stabilize (this thread) ->
        # encode thread. Both C stages release the GIL (PyAV / numpy /
        # x264), so they overlap with the Python-heavy stabilize stage.
        import queue
        import threading

        decode_q: "queue.Queue" = queue.Queue(maxsize=6)
        encode_q: "queue.Queue" = queue.Queue(maxsize=6)
        _DONE = object()

        def decode_loop():
            seq = 0
            try:
                for packet in self._input_container.demux(self._input_stream):
                    if packet.dts is None:
                        continue
                    for frame in packet.decode():
                        decode_q.put(to_item(frame, seq))
                        seq += 1
                # Flush the threaded decoder (trailing buffered frames).
                for frame in self._input_stream.decode():
                    decode_q.put(to_item(frame, seq))
                    seq += 1
            except Exception as exc:  # surfaced to the main loop
                decode_q.put(exc)
            finally:
                decode_q.put(_DONE)

        def encode_loop():
            while True:
                item = encode_q.get()
                if item is _DONE or isinstance(item, BaseException):
                    return
                out_frame = av.VideoFrame.from_ndarray(item, format="rgb24")
                for pkt in self._output_stream.encode(out_frame):
                    self._output_container.mux(pkt)

        decoder = threading.Thread(target=decode_loop, name="pgf-decode", daemon=True)
        encoder = threading.Thread(target=encode_loop, name="pgf-encode", daemon=True)
        decoder.start()
        encoder.start()

        pipeline_error: BaseException | None = None
        try:
            while True:
                item = decode_q.get()
                if item is _DONE:
                    break
                if isinstance(item, Exception):
                    pipeline_error = item
                    break
                img, timestamp_ms = item
                processed = callback(img, timestamp_ms, self._frame_index)
                # Ensure uint8 for encoding.
                if processed.dtype != np.uint8:
                    processed = np.clip(processed, 0, 255).astype(np.uint8)
                encode_q.put(processed)
                self._frame_index += 1
        except BaseException as exc:
            pipeline_error = exc
            raise
        finally:
            # Always release the encoder, even on stabilize errors.
            encode_q.put(pipeline_error if pipeline_error is not None else _DONE)
            encoder.join(timeout=120.0)
            decoder.join(timeout=120.0)

        if pipeline_error is not None:
            raise pipeline_error

        log.info("Processed %d frames", self._frame_index)

    def copy_audio(self) -> None:
        """Mux audio packets into the streams added by ``prepare_audio``.

        Call after ``process_frames`` and before ``close``: the video demux
        only consumes video-stream packets, so audio packets are still
        unread when this runs. Direct stream copy is used when the output
        container supports the input codec, with an AAC re-encode fallback.
        """
        if self._input_container is None or self._output_container is None:
            raise VideoIOError("Both input and output must be opened first")
        if not self._audio_pairs:
            return

        from pygyroflow.rendering.audio_resampler import mux_audio

        mux_audio(self._input_container, self._output_container, self._audio_pairs)
        self._audio_pairs = []

    def close(self) -> None:
        """Flush the encoder and close containers."""
        try:
            import av  # type: ignore[import-untyped]
        except ImportError:
            # Nothing to clean up if PyAV was never used.
            self._output_container = None
            self._input_container = None
            self._output_stream = None
            self._input_stream = None
            return

        if self._output_container is not None and self._output_stream is not None:
            try:
                for pkt in self._output_stream.encode():
                    self._output_container.mux(pkt)
            except Exception:
                log.warning("Error flushing encoder", exc_info=True)

            try:
                self._output_container.close()
            except Exception:
                log.warning("Error closing output container", exc_info=True)

        if self._input_container is not None:
            try:
                self._input_container.close()
            except Exception:
                log.warning("Error closing input container", exc_info=True)

        self._output_container = None
        self._input_container = None
        self._output_stream = None
        self._input_stream = None
