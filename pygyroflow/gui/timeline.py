"""Timeline widget — playback scrubber and gyro data chart placeholder.

Provides a horizontal slider for frame-by-frame navigation, a frame counter
label, and a placeholder area where the gyro waveform chart will be rendered
in a future iteration.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class TimelineWidget(QWidget):
    """Bottom-docked timeline with playback controls.

    Signals
    -------
    position_changed : int
        Emitted when the user moves the scrubber (frame index).
    play_toggled : bool
        Emitted when the play/pause button is toggled.
    """

    position_changed = Signal(int)
    play_toggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        # -- Gyro chart placeholder --
        self._chart_label = QLabel("Gyro data chart")
        self._chart_label.setMinimumHeight(60)
        self._chart_label.setStyleSheet(
            "background-color: #16213e; color: #8899aa; "
            "padding: 4px; font-size: 12px; border-radius: 3px;"
        )
        layout.addWidget(self._chart_label)

        # -- Playback controls --
        controls = QHBoxLayout()

        self._play_btn = QPushButton("▶")
        self._play_btn.setFixedWidth(36)
        self._play_btn.setCheckable(True)
        self._play_btn.toggled.connect(self._on_play_toggle)
        controls.addWidget(self._play_btn)

        self._slider = QSlider()
        self._slider.setOrientation(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.valueChanged.connect(self._on_slider)
        controls.addWidget(self._slider, stretch=1)

        self._frame_label = QLabel("0 / 0")
        self._frame_label.setMinimumWidth(100)
        controls.addWidget(self._frame_label)

        layout.addLayout(controls)

        self._frame_count: int = 0

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def set_frame_count(self, count: int) -> None:
        """Set the total number of frames for the scrubber range."""
        self._frame_count = max(0, count)
        self._slider.setRange(0, max(0, self._frame_count - 1))
        self._update_label(0)

    def set_position(self, frame: int) -> None:
        """Move the scrubber to *frame* without emitting a signal."""
        self._slider.blockSignals(True)
        self._slider.setValue(frame)
        self._slider.blockSignals(False)
        self._update_label(frame)

    def set_chart_text(self, text: str) -> None:
        """Update the gyro chart placeholder text."""
        self._chart_label.setText(text)

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _on_slider(self, value: int) -> None:
        self._update_label(value)
        self.position_changed.emit(value)

    def _on_play_toggle(self, checked: bool) -> None:
        self._play_btn.setText("⏸" if checked else "▶")
        self.play_toggled.emit(checked)

    def _update_label(self, frame: int) -> None:
        self._frame_label.setText(f"{frame} / {self._frame_count}")
