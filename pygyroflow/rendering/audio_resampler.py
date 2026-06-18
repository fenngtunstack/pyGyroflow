"""Audio passthrough and resampling for video rendering.

Handles copying audio streams from input to output during stabilisation,
with optional resampling to match the output container format.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import av

log = logging.getLogger(__name__)


class AudioResampler:
    """Handle audio passthrough and resampling during rendering."""

    @staticmethod
    def copy_audio(
        input_path: str,
        output_container: av.container.OutputContainer,
        input_container: av.container.InputContainer,
    ) -> None:
        """Copy all audio streams from input to output container.

        This performs a direct stream copy (no re-encoding) when possible.
        Falls back to re-encoding if the output container format does not
        support the input audio codec.

        Args:
            input_path: Path to the original input video (not used directly,
                        kept for API compatibility).
            output_container: Already-opened output container.
            input_container: Already-opened input container.
        """
        audio_streams = input_container.streams.audio
        if not audio_streams:
            log.info("No audio streams found, skipping audio passthrough")
            return

        for i, audio_stream in enumerate(audio_streams):
            try:
                # Try direct stream copy first.
                out_stream = output_container.add_stream(template=audio_stream)
                for packet in input_container.demux(audio_stream):
                    if packet.dts is None:
                        continue
                    packet.stream = out_stream
                    output_container.mux(packet)
                log.info("Audio stream %d copied (stream copy)", i)
            except Exception:
                # Fall back to decode-encode if muxing fails.
                log.warning(
                    "Stream copy failed for audio stream %d, "
                    "falling back to re-encode",
                    i,
                    exc_info=True,
                )
                _reencode_audio(
                    input_container, output_container, audio_stream, i
                )


def _reencode_audio(
    input_container: av.container.InputContainer,
    output_container: av.container.OutputContainer,
    audio_stream: av.audio.stream.AudioStream,
    index: int,
) -> None:
    """Re-encode an audio stream into the output container.

    Used as a fallback when direct stream copy is not possible
    (e.g., container format mismatch).
    """
    import av  # type: ignore[import-untyped]

    try:
        # Add an AAC audio stream -- widely compatible.
        out_stream = output_container.add_stream("aac")
        out_stream.rate = 48000
        out_stream.layout = "stereo"

        resampler = av.AudioResampler(
            format="fltp",
            layout="stereo",
            rate=48000,
        )

        for packet in input_container.demux(audio_stream):
            for frame in packet.decode():
                resampled = resampler.resample(frame)
                for r_frame in resampled:
                    for out_pkt in out_stream.encode(r_frame):
                        output_container.mux(out_pkt)

        # Flush the resampler.
        for r_frame in resampler.resample(None):
            for out_pkt in out_stream.encode(r_frame):
                output_container.mux(out_pkt)

        # Flush encoder.
        for out_pkt in out_stream.encode():
            output_container.mux(out_pkt)

        log.info("Audio stream %d re-encoded to AAC 48kHz stereo", index)
    except Exception:
        log.error(
            "Failed to re-encode audio stream %d, skipping", index,
            exc_info=True,
        )
