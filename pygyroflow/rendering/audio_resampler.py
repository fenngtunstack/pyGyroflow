"""Audio passthrough and resampling for video rendering.

Handles copying audio streams from input to output during stabilisation,
with optional resampling to match the output container format.

PyAV requires every output stream to exist before the first packet is
muxed (the container header is written on first mux), so the flow is
two-phase: ``prepare_audio_streams`` adds the streams right after the
video encoder is created, and ``mux_audio`` moves the packets later.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import av

log = logging.getLogger(__name__)


def _add_stream_from_template(
    output_container: "av.container.OutputContainer",
    audio_stream: "av.audio.stream.AudioStream",
) -> "av.stream.Stream":
    """Add an output stream mirroring the input audio stream's parameters.

    PyAV >= 14 moved the ``template=`` keyword to a dedicated
    ``add_stream_from_template`` method; older versions only support the
    keyword form.
    """
    add_from_template = getattr(output_container, "add_stream_from_template", None)
    if add_from_template is not None:
        return add_from_template(audio_stream)
    return output_container.add_stream(template=audio_stream)  # type: ignore[call-arg]


def prepare_audio_streams(
    input_container: "av.container.InputContainer",
    output_container: "av.container.OutputContainer",
) -> list[tuple["av.audio.stream.AudioStream", "av.stream.Stream"]]:
    """Add output audio streams mirroring the input's audio streams.

    Must be called before any packet is muxed into ``output_container``
    (i.e. before video processing starts), otherwise PyAV raises
    ``ValueError: Cannot rebase to zero time`` for the late-added stream.

    Returns pairs of (input_stream, output_stream) to feed to
    :func:`mux_audio`. If a template stream cannot be created for an
    input stream, an AAC encoder stream is prepared instead and the pair
    is muxed via re-encode.
    """
    pairs: list[tuple["av.audio.stream.AudioStream", "av.stream.Stream"]] = []
    for audio_stream in input_container.streams.audio:
        try:
            out_stream = _add_stream_from_template(output_container, audio_stream)
        except Exception:
            log.warning(
                "Cannot add template stream for audio %s, preparing AAC "
                "re-encode instead",
                audio_stream.name if hasattr(audio_stream, "name") else "?",
                exc_info=True,
            )
            out_stream = _prepare_aac_stream(output_container)
        pairs.append((audio_stream, out_stream))
    return pairs


def _prepare_aac_stream(
    output_container: "av.container.OutputContainer",
) -> "av.stream.Stream":
    """Add a widely-compatible AAC stream for re-encoded audio."""
    out_stream = output_container.add_stream("aac")
    out_stream.rate = 48000
    out_stream.layout = "stereo"
    return out_stream


def _make_rebaser(ranges_ms):
    """Build ``(keep, rebase)`` for a trimmed render, or ``(None, None)``.

    The video pass drops whole ranges and writes the survivors back-to-back;
    audio has to be cut the same way or it drifts further ahead of the video
    with every range removed. A packet in range *k* is kept if it starts
    inside it and is shifted back by however much was cut before it plus the
    range's own offset from the start of the clip — so range *k*'s audio
    lands exactly where range *k-1*'s ended.

    Packets, not samples: a packet straddling a boundary is kept or dropped
    whole. That is the resolution of the container's own framing (a few ms
    for AAC), well under one video frame.
    """
    if not ranges_ms:
        return None, None

    starts = [0.0 if r[0] is None else float(r[0]) for r in ranges_ms]
    ends = [None if r[1] is None else float(r[1]) for r in ranges_ms]
    offsets = []
    acc = 0.0
    for k, start in enumerate(starts):
        offsets.append(acc)
        end = ends[k]
        if end is not None:
            acc += end - start

    def locate(ts_ms):
        """Index of the range holding *ts_ms*, or None."""
        for k, start in enumerate(starts):
            end = ends[k]
            if ts_ms < start:
                return None
            if end is None or ts_ms <= end:
                return k
        return None

    def keep(ts_ms):
        return locate(ts_ms) is not None

    def rebase(ts_ms):
        k = locate(ts_ms)
        if k is None:
            return None
        return ts_ms - starts[k] + offsets[k]

    return keep, rebase


def mux_audio(
    input_container: "av.container.InputContainer",
    output_container: "av.container.OutputContainer",
    pairs: list[tuple["av.audio.stream.AudioStream", "av.stream.Stream"]],
    ranges_ms: list[tuple[float | None, float | None]] | None = None,
) -> None:
    """Mux audio packets through the prepared stream pairs.

    Direct stream copy is used when the output stream was created from the
    input template; pairs whose output stream is a bare AAC encoder are
    re-encoded. A stream that fails mid-copy is skipped with an error log
    (streams cannot be replaced after muxing has started).

    *ranges_ms* mirrors the video trim: only audio inside the ranges is
    written, and the kept spans are concatenated (see :func:`_make_rebaser`).
    """
    keep, rebase = _make_rebaser(ranges_ms)
    for i, (in_stream, out_stream) in enumerate(pairs):
        from_template = out_stream.codec_context.name == in_stream.codec_context.name
        try:
            if from_template:
                # The video pass already demuxed the input to EOF; seek back
                # to the start or this stream's packets are all gone and
                # the muxer would silently drop the (empty) audio track.
                input_container.seek(0)
                for packet in input_container.demux(in_stream):
                    if packet.dts is None:
                        continue
                    if keep is not None:
                        ts_ms = _packet_ms(packet, in_stream)
                        if ts_ms is None or not keep(ts_ms):
                            continue
                        _rebase_packet(packet, in_stream, rebase(ts_ms))
                    packet.stream = out_stream
                    output_container.mux(packet)
                log.info("Audio stream %d copied (stream copy)", i)
            else:
                reencode_audio(
                    input_container, output_container, in_stream, out_stream, i,
                    ranges_ms=ranges_ms,
                )
        except Exception:
            log.error(
                "Failed to mux audio stream %d, skipping (output will miss "
                "this audio track)",
                i,
                exc_info=True,
            )


def _packet_ms(packet, stream) -> float | None:
    """A packet's start time in milliseconds, or None if it has no pts."""
    if packet.pts is None:
        return None
    time_base = packet.time_base or stream.time_base
    if time_base is None:
        return None
    return float(packet.pts) * float(time_base) * 1000.0


def _rebase_packet(packet, stream, ts_ms: float) -> None:
    """Move a packet to *ts_ms*, in the stream's own time base."""
    time_base = packet.time_base or stream.time_base
    if time_base is None:
        return
    new_pts = int(round(ts_ms / (float(time_base) * 1000.0)))
    packet.pts = new_pts
    packet.dts = new_pts


def reencode_audio(
    input_container: "av.container.InputContainer",
    output_container: "av.container.OutputContainer",
    audio_stream: "av.audio.stream.AudioStream",
    out_stream: "av.stream.Stream",
    index: int = 0,
    ranges_ms: list[tuple[float | None, float | None]] | None = None,
) -> None:
    """Re-encode an audio stream through a prepared AAC output stream."""
    import av  # type: ignore[import-untyped]

    resampler = av.AudioResampler(
        format="fltp",
        layout="stereo",
        rate=48000,
    )

    keep, _ = _make_rebaser(ranges_ms)
    input_container.seek(0)
    for packet in input_container.demux(audio_stream):
        if packet.dts is None:
            continue
        for frame in packet.decode():
            # Trim before the resampler: the encoder assigns output
            # timestamps from arrival order, so dropping frames here is
            # already a gapless concatenation.
            if keep is not None:
                if frame.pts is None:
                    continue
                ts_ms = float(frame.pts) * float(frame.time_base) * 1000.0
                if not keep(ts_ms):
                    continue
            for r_frame in resampler.resample(frame):
                for out_pkt in out_stream.encode(r_frame):
                    output_container.mux(out_pkt)

    # Flush the resampler and the encoder.
    for r_frame in resampler.resample(None):
        for out_pkt in out_stream.encode(r_frame):
            output_container.mux(out_pkt)
    for out_pkt in out_stream.encode():
        output_container.mux(out_pkt)

    log.info("Audio stream %d re-encoded to AAC 48kHz stereo", index)


class AudioResampler:
    """Handle audio passthrough and resampling during rendering.

    Historical convenience wrapper combining prepare + mux in one call;
    only usable when no packets have been muxed yet. The rendering
    pipeline uses the module-level two-phase functions instead.
    """

    @staticmethod
    def copy_audio(
        input_path: str,
        output_container: "av.container.OutputContainer",
        input_container: "av.container.InputContainer",
    ) -> None:
        """Copy all audio streams from input to output container.

        Args:
            input_path: Path to the original input video (not used directly,
                        kept for API compatibility).
            output_container: Already-opened output container (no packets
                        muxed yet).
            input_container: Already-opened input container.
        """
        pairs = prepare_audio_streams(input_container, output_container)
        if not pairs:
            log.info("No audio streams found, skipping audio passthrough")
            return
        mux_audio(input_container, output_container, pairs)
