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


def mux_audio(
    input_container: "av.container.InputContainer",
    output_container: "av.container.OutputContainer",
    pairs: list[tuple["av.audio.stream.AudioStream", "av.stream.Stream"]],
) -> None:
    """Mux audio packets through the prepared stream pairs.

    Direct stream copy is used when the output stream was created from the
    input template; pairs whose output stream is a bare AAC encoder are
    re-encoded. A stream that fails mid-copy is skipped with an error log
    (streams cannot be replaced after muxing has started).
    """
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
                    packet.stream = out_stream
                    output_container.mux(packet)
                log.info("Audio stream %d copied (stream copy)", i)
            else:
                reencode_audio(input_container, output_container, in_stream, out_stream, i)
        except Exception:
            log.error(
                "Failed to mux audio stream %d, skipping (output will miss "
                "this audio track)",
                i,
                exc_info=True,
            )


def reencode_audio(
    input_container: "av.container.InputContainer",
    output_container: "av.container.OutputContainer",
    audio_stream: "av.audio.stream.AudioStream",
    out_stream: "av.stream.Stream",
    index: int = 0,
) -> None:
    """Re-encode an audio stream through a prepared AAC output stream."""
    import av  # type: ignore[import-untyped]

    resampler = av.AudioResampler(
        format="fltp",
        layout="stereo",
        rate=48000,
    )

    input_container.seek(0)
    for packet in input_container.demux(audio_stream):
        if packet.dts is None:
            continue
        for frame in packet.decode():
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
