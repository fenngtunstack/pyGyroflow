"""Abstract base class for video processing backends.

Defines the interface that all video I/O implementations must follow
(PyAV-based, OpenCV-based, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

# Type alias for the frame-processing callback.
# Receives (frame_rgb, timestamp_ms, frame_index) and returns the processed frame.
FrameCallback = Callable[[np.ndarray, float, int], np.ndarray]


class VideoProcessor(ABC):
    """Interface for video input/output backends.

    Subclasses handle opening video files, decoding frames, encoding
    output, and managing the full frame-processing loop.
    """

    @abstractmethod
    def open_input(self, path: str) -> dict:
        """Open an input video file.

        Args:
            path: Filesystem path to the video.

        Returns:
            Dict with keys: width, height, fps, frames, duration (ms), codec.

        Raises:
            VideoIOError: If the file cannot be opened.
        """

    @abstractmethod
    def create_output(
        self,
        path: str,
        width: int,
        height: int,
        fps: float,
        codec: str = "H.265/HEVC",
        bitrate: float = 0.0,
    ) -> None:
        """Configure the output video encoder.

        Args:
            path: Output file path.
            width: Frame width in pixels.
            height: Frame height in pixels.
            fps: Target frame rate.
            codec: Codec name (e.g. "H.264/AVC", "H.265/HEVC").
            bitrate: Target bitrate in Mbps. 0 = use codec default.

        Raises:
            VideoIOError: If the encoder cannot be created.
        """

    @abstractmethod
    def process_frames(self, callback: FrameCallback) -> None:
        """Iterate over all input frames, apply *callback*, write to output.

        Args:
            callback: Function receiving (frame, timestamp_ms, frame_index)
                      and returning the processed frame.
        """

    @abstractmethod
    def close(self) -> None:
        """Flush encoders and release all resources."""

    @property
    @abstractmethod
    def input_info(self) -> dict | None:
        """Return metadata of the currently opened input, or None."""

    # Convenience context manager support.

    def __enter__(self) -> VideoProcessor:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
