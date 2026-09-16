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

# Pixel format per encoder.  Encoders are not interchangeable here: prores_ks
# only accepts 10-bit 4:2:2 (yuv422p10le), and hardcoding yuv420p for every
# codec makes avcodec_open2 fail with EINVAL.  The 8-bit encoders are happy
# with yuv420p.
_PIX_FMT_MAP: dict[str, str] = {
    "libx264": "yuv420p",
    "libx265": "yuv420p",
    "prores_ks": "yuv422p10le",
    "libaom-av1": "yuv420p",
    "libvpx-vp9": "yuv420p",
    "mpeg4": "yuv420p",
}

# ProRes needs an explicit profile: prores_ks defaults to "proxy" (0), which
# is a quarter-resolution mezzanine — the wrong thing to hand someone who
# asked for ProRes output.  3 = HQ.
_ENCODER_PROFILE: dict[str, int] = {"prores_ks": 3}


def _put_unless_failed(q, item, failed: list, stop=None) -> bool:
    """Put *item* on *q*, giving up if the consumer has died or *stop* is set.

    The queues are bounded, so a plain ``put`` blocks forever once one fills
    — and it fills for good as soon as the consumer exits.  Poll instead: the
    moment the pipeline is aborting the producer has to stop, not wait.
    """
    import queue

    while not failed and not (stop is not None and stop.is_set()):
        try:
            q.put(item, timeout=0.1)
            return True
        except queue.Full:
            continue
    return False


def _drain(q) -> None:
    """Discard everything currently queued."""
    import queue

    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return


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
        self._image_sequence = None
        self._frame_index: int = 0
        self._audio_pairs: list = []
        self._output_codec_name: str | None = None

    # ------------------------------------------------------------------
    # VideoProcessor interface
    # ------------------------------------------------------------------

    @property
    def input_info(self) -> dict | None:
        return self._input_info

    def open_input(self, path: str, fps: float | None = None) -> dict:
        """Open an input video (or image sequence) and return its metadata.

        *path* may be an image sequence — a directory, a printf pattern like
        ``shots/frame_%04d.exr``, or a single frame.  FFmpeg's image2 demuxer
        takes those as a pattern plus ``start_number``/``framerate`` options,
        so the sequence is resolved to that form here and *fps* supplies the
        rate (sequences have none of their own).
        """
        try:
            import av  # type: ignore[import-untyped]
        except ImportError as exc:
            raise VideoIOError(
                "PyAV is required for video I/O. Install with: pip install av"
            ) from exc

        from pygyroflow.rendering.image_sequence import (
            format_options,
            looks_like_image_sequence,
            resolve_image_sequence,
        )

        sequence = resolve_image_sequence(path) if looks_like_image_sequence(path) else None
        self._image_sequence = sequence

        try:
            if sequence is not None:
                self._input_container = av.open(
                    sequence.pattern,
                    format="image2" if sequence.is_sequence else None,
                    options=format_options(sequence, fps),
                )
            else:
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

        frame_count = self._input_stream.frames or 0
        if sequence is not None:
            # image2 reports no frame count; the files on disk are the truth.
            frame_count = sequence.frame_count
            if duration_ms <= 0 and fps > 0:
                duration_ms = frame_count / fps * 1000.0

        info = {
            "width": self._input_stream.width,
            "height": self._input_stream.height,
            "fps": fps,
            "frames": frame_count,
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
        self._output_codec_name = codec_name
        self._output_container = av.open(path, mode="w")
        from fractions import Fraction
        fps_frac = Fraction(fps).limit_denominator(100000)
        self._output_stream = self._output_container.add_stream(
            codec_name, rate=fps_frac
        )
        self._output_stream.width = width
        self._output_stream.height = height
        self._output_stream.pix_fmt = _PIX_FMT_MAP.get(codec_name, "yuv420p")

        profile = _ENCODER_PROFILE.get(codec_name)
        if profile is not None:
            try:
                self._output_stream.codec_context.profile = profile
            except Exception:
                log.warning(
                    "Encoder %s accepted no profile %d; using its default",
                    codec_name, profile,
                )

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

        # Set when the main loop gives up (encode failure, callback error).
        # Without it the decode thread keeps filling a queue nobody drains,
        # blocks forever on its own put, and the join below burns its full
        # timeout — 120 s of hang for a failure that was already known.
        stop = threading.Event()

        # Failures from either worker thread, read by the main loop.
        fail: list[BaseException] = []

        def decode_loop():
            seq = 0
            aborted = False
            try:
                for packet in self._input_container.demux(self._input_stream):
                    if stop.is_set():
                        aborted = True
                        break
                    if packet.dts is None:
                        continue
                    for frame in packet.decode():
                        if not _put_unless_failed(
                            decode_q, to_item(frame, seq), fail, stop
                        ):
                            aborted = True
                            return
                        seq += 1
                if not aborted:
                    # Flush the threaded decoder (trailing buffered frames).
                    for frame in self._input_stream.decode():
                        if not _put_unless_failed(
                            decode_q, to_item(frame, seq), fail, stop
                        ):
                            aborted = True
                            return
                        seq += 1
            except Exception as exc:  # surfaced to the main loop
                fail.append(exc)
                aborted = True
            finally:
                if aborted:
                    # The main loop is gone or going: drop what is queued so
                    # this put cannot block. On the normal path the queue
                    # still holds frames the main loop has not read yet —
                    # draining there would silently lose the tail.
                    _drain(decode_q)
                decode_q.put(_DONE)

        # Encoder failures (a bad pix_fmt, an unavailable encoder) happen on
        # the FIRST encode call inside this thread, not at add_stream. An
        # exception here used to die silently as "Exception in thread
        # pgf-encode": the main loop never saw it, so render() returned
        # normally, the CLI printed "Done: <path>", and no file existed.
        def encode_loop():
            try:
                while True:
                    item = encode_q.get()
                    if item is _DONE or isinstance(item, BaseException):
                        return
                    out_frame = av.VideoFrame.from_ndarray(item, format="rgb24")
                    for pkt in self._output_stream.encode(out_frame):
                        self._output_container.mux(pkt)
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                fail.append(exc)

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
                if not _put_unless_failed(encode_q, processed, fail):
                    pipeline_error = fail[0]
                    break
                self._frame_index += 1
        except BaseException as exc:
            pipeline_error = exc
            raise
        finally:
            stop.set()
            _drain(decode_q)  # unblock a decode thread parked on a full queue
            # Always release the encoder, even on stabilize errors. When the
            # encoder already died its queue can be full and the final put
            # would block forever, so drain it first.
            if pipeline_error is not None or fail:
                _drain(encode_q)
            encode_q.put(pipeline_error if pipeline_error is not None else _DONE)
            encoder.join(timeout=30.0)
            decoder.join(timeout=30.0)

        if pipeline_error is not None:
            raise pipeline_error

        if fail and not isinstance(fail[0], Exception):
            raise fail[0]

        if fail:
            raise VideoIOError(
                f"Encoder '{self._output_codec_name}' failed after "
                f"{self._frame_index} frame(s): {fail[0]}"
            ) from fail[0]

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
