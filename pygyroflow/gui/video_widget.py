"""Video preview widget — displays the current frame in the central area.

Supports displaying video info text as a placeholder and rendering numpy
frames as QPixmap images.  GPU-accelerated rendering can be layered on
top of the set_frame() path later.
"""

from __future__ import annotations

import logging

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

log = logging.getLogger(__name__)


class VideoWidget(QWidget):
    """Central video preview area.

    Shows a placeholder when no video is loaded, video metadata when loaded,
    and full frames when a decoder feeds them via :meth:`set_frame`.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 270)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label = QLabel("No video loaded")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet(
            "background-color: #1a1a2e; color: #e0e0e0; font-size: 18px;"
        )
        layout.addWidget(self._label)

        self._info: dict = {}

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def set_video_info(self, info: dict) -> None:
        """Display video metadata as placeholder text."""
        self._info = info
        w = info.get("width", "?")
        h = info.get("height", "?")
        fps = info.get("fps", 0)
        frames = info.get("frame_count", "?")
        fps_str = f"{fps:.2f}" if isinstance(fps, (int, float)) and fps > 0 else "?"
        duration_ms = info.get("duration_ms", 0)
        if isinstance(duration_ms, (int, float)) and duration_ms > 0:
            dur_s = duration_ms / 1000.0
            dur_str = f"{int(dur_s // 60)}:{int(dur_s % 60):02d}"
        else:
            dur_str = "?"

        text = (
            f"{w} x {h}  |  {fps_str} fps  |  {frames} frames  |  {dur_str}"
        )
        self._label.setText(text)

    def set_frame(self, frame: np.ndarray) -> None:
        """Render a numpy frame (H, W, C) as the preview image.

        Supports RGB (3ch), RGBA (4ch), and grayscale (1ch / 2ch).
        """
        if frame is None or frame.size == 0:
            return

        h, w = frame.shape[:2]
        ch = frame.shape[2] if frame.ndim == 3 else 1

        # Ensure contiguous buffer for QImage
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)

        if ch == 3:
            fmt = QImage.Format.Format_RGB888
            bpl = 3 * w
        elif ch == 4:
            fmt = QImage.Format.Format_RGBA8888
            bpl = 4 * w
        else:
            fmt = QImage.Format.Format_Grayscale8
            bpl = w

        qimg = QImage(frame.data, w, h, bpl, fmt)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self._label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(pixmap)

    def clear(self) -> None:
        """Reset to the placeholder state."""
        self._info = {}
        self._label.clear()
        self._label.setText("No video loaded")
