"""Settings panel — right-docked stabilization parameter controls.

Groups smoothing, FOV/zoom, synchronization, horizon lock, and
adaptive zoom into collapsible sections inside a scrollable area.
Controls emit change signals that the main window wires to the
StabilizationManager.
"""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

log = logging.getLogger(__name__)


class SettingsPanel(QScrollArea):
    """Scrollable panel of stabilization settings.

    Signals
    -------
    smoothing_changed : str
        Algorithm name.
    smoothness_changed : float
        Smoothness value 0.0 - 1.0.
    fov_changed : float
        FOV scale factor (1.0 = native).
    zoom_window_changed : float
        Adaptive zoom window in seconds.
    horizon_amount_changed : float
        Horizon lock amount 0.0 - 1.0.
    sync_requested : ()
        User clicked Auto Sync.
    """

    smoothing_changed = Signal(str)
    smoothness_changed = Signal(float)
    fov_changed = Signal(float)
    zoom_window_changed = Signal(float)
    horizon_amount_changed = Signal(float)
    sync_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setMinimumWidth(260)
        self.setMaximumWidth(400)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)

        self._build_smoothing_group(layout)
        self._build_fov_group(layout)
        self._build_sync_group(layout)
        self._build_horizon_group(layout)
        self._build_trim_group(layout)

        layout.addStretch()
        self.setWidget(container)

    # ================================================================== #
    #  Smoothing                                                           #
    # ================================================================== #

    def _build_smoothing_group(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("Smoothing")
        form = QFormLayout()

        self._smooth_algo = QComboBox()
        self._smooth_algo.addItems(["None", "Default", "Plain", "Fixed"])
        self._smooth_algo.setCurrentIndex(1)
        self._smooth_algo.currentTextChanged.connect(self.smoothing_changed.emit)
        form.addRow("Algorithm:", self._smooth_algo)

        self._smoothness = QSlider()
        self._smoothness.setOrientation(Qt.Orientation.Horizontal)
        self._smoothness.setRange(0, 100)
        self._smoothness.setValue(50)
        self._smoothness_label = QLabel("0.50")
        self._smoothness.valueChanged.connect(self._on_smoothness)
        form.addRow("Smoothness:", self._smoothness)
        form.addRow("", self._smoothness_label)

        group.setLayout(form)
        parent_layout.addWidget(group)

    def _on_smoothness(self, value: int) -> None:
        f = value / 100.0
        self._smoothness_label.setText(f"{f:.2f}")
        self.smoothness_changed.emit(f)

    # ================================================================== #
    #  FOV / Zoom                                                          #
    # ================================================================== #

    def _build_fov_group(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("FOV & Zoom")
        form = QFormLayout()

        # FOV slider (0.50 - 2.00, default 1.00)
        self._fov = QSlider()
        self._fov.setOrientation(Qt.Orientation.Horizontal)
        self._fov.setRange(50, 200)
        self._fov.setValue(100)
        self._fov_label = QLabel("1.00")
        self._fov.valueChanged.connect(self._on_fov)
        form.addRow("FOV:", self._fov)
        form.addRow("", self._fov_label)

        # Adaptive zoom window
        self._zoom_window = QDoubleSpinBox()
        self._zoom_window.setRange(0.0, 30.0)
        self._zoom_window.setValue(4.0)
        self._zoom_window.setSingleStep(0.5)
        self._zoom_window.setSuffix(" s")
        self._zoom_window.valueChanged.connect(self.zoom_window_changed.emit)
        form.addRow("Zoom window:", self._zoom_window)

        # Max zoom
        self._max_zoom = QDoubleSpinBox()
        self._max_zoom.setRange(1.0, 20.0)
        self._max_zoom.setValue(5.0)
        self._max_zoom.setSingleStep(0.5)
        form.addRow("Max zoom:", self._max_zoom)

        group.setLayout(form)
        parent_layout.addWidget(group)

    def _on_fov(self, value: int) -> None:
        f = value / 100.0
        self._fov_label.setText(f"{f:.2f}")
        self.fov_changed.emit(f)

    # ================================================================== #
    #  Synchronization                                                     #
    # ================================================================== #

    def _build_sync_group(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("Synchronization")
        vbox = QVBoxLayout()

        self._auto_sync_btn = QPushButton("Auto Sync")
        self._auto_sync_btn.clicked.connect(self.sync_requested.emit)
        vbox.addWidget(self._auto_sync_btn)

        row = QHBoxLayout()
        self._offset_spin = QDoubleSpinBox()
        self._offset_spin.setRange(-10.0, 10.0)
        self._offset_spin.setDecimals(4)
        self._offset_spin.setSuffix(" s")
        self._offset_spin.setSingleStep(0.001)
        row.addWidget(QLabel("Offset:"))
        row.addWidget(self._offset_spin)
        vbox.addLayout(row)

        group.setLayout(vbox)
        parent_layout.addWidget(group)

    # ================================================================== #
    #  Horizon lock                                                        #
    # ================================================================== #

    def _build_horizon_group(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("Horizon Lock")
        form = QFormLayout()

        self._horizon_amount = QSlider()
        self._horizon_amount.setOrientation(Qt.Orientation.Horizontal)
        self._horizon_amount.setRange(0, 100)
        self._horizon_amount.setValue(0)
        self._horizon_amount_label = QLabel("0.00")
        self._horizon_amount.valueChanged.connect(self._on_horizon)
        form.addRow("Amount:", self._horizon_amount)
        form.addRow("", self._horizon_amount_label)

        self._horizon_roll = QDoubleSpinBox()
        self._horizon_roll.setRange(-180.0, 180.0)
        self._horizon_roll.setSingleStep(0.5)
        self._horizon_roll.setSuffix(" deg")
        form.addRow("Roll offset:", self._horizon_roll)

        group.setLayout(form)
        parent_layout.addWidget(group)

    def _on_horizon(self, value: int) -> None:
        f = value / 100.0
        self._horizon_amount_label.setText(f"{f:.2f}")
        self.horizon_amount_changed.emit(f)

    # ================================================================== #
    #  Trim / Speed                                                        #
    # ================================================================== #

    def _build_trim_group(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("Trim & Speed")
        form = QFormLayout()

        self._speed_spin = QDoubleSpinBox()
        self._speed_spin.setRange(0.1, 10.0)
        self._speed_spin.setValue(1.0)
        self._speed_spin.setSingleStep(0.1)
        self._speed_spin.setSuffix("x")
        form.addRow("Speed:", self._speed_spin)

        self._lens_correction = QSlider()
        self._lens_correction.setOrientation(Qt.Orientation.Horizontal)
        self._lens_correction.setRange(0, 100)
        self._lens_correction.setValue(100)
        self._lens_correction_label = QLabel("1.00")
        self._lens_correction.valueChanged.connect(
            lambda v: self._lens_correction_label.setText(f"{v / 100:.2f}")
        )
        form.addRow("Lens correction:", self._lens_correction)
        form.addRow("", self._lens_correction_label)

        group.setLayout(form)
        parent_layout.addWidget(group)

    # ================================================================== #
    #  Value accessors (for wiring to StabilizationManager)                #
    # ================================================================== #

    def get_smoothing_algorithm(self) -> str:
        return self._smooth_algo.currentText()

    def get_smoothness(self) -> float:
        return self._smoothness.value() / 100.0

    def get_fov(self) -> float:
        return self._fov.value() / 100.0

    def get_zoom_window(self) -> float:
        return self._zoom_window.value()

    def get_horizon_amount(self) -> float:
        return self._horizon_amount.value() / 100.0

    def get_sync_offset(self) -> float:
        return self._offset_spin.value()
