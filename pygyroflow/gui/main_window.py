"""Main application window — orchestrates all GUI panels and the StabilizationManager.

Provides the top-level window with:
- Central video preview
- Right-docked settings panel
- Left-docked lens profile browser
- Bottom-docked timeline
- Menu bar with File / Tools actions
- Status bar with current operation feedback
"""

from __future__ import annotations

import logging
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QWidget,
)

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Top-level PyGyroFlow window.

    Usage::

        app = MainWindow.create_app()
        window = MainWindow()
        window.show()
        app.exec()
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PyGyroFlow — Video Stabilization")
        self.setMinimumSize(1280, 720)
        self.resize(1440, 900)

        # -- Central widget: video preview --
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        from pygyroflow.gui.video_widget import VideoWidget
        self.video_widget = VideoWidget()
        layout.addWidget(self.video_widget, stretch=3)

        # -- Right panel: stabilization settings --
        from pygyroflow.gui.settings_panel import SettingsPanel
        self.settings_panel = SettingsPanel()
        right_dock = QDockWidget("Stabilization Settings", self)
        right_dock.setWidget(self.settings_panel)
        right_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, right_dock)

        # -- Left panel: lens profiles --
        from pygyroflow.gui.lens_dialog import LensProfileBrowser
        self.lens_browser = LensProfileBrowser()
        left_dock = QDockWidget("Lens Profiles", self)
        left_dock.setWidget(self.lens_browser)
        left_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, left_dock)

        # -- Bottom panel: timeline --
        from pygyroflow.gui.timeline import TimelineWidget
        self.timeline = TimelineWidget()
        bottom_dock = QDockWidget("Timeline", self)
        bottom_dock.setWidget(self.timeline)
        bottom_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, bottom_dock)

        # -- Menu bar --
        self._setup_menu()

        # -- Status bar --
        self.statusBar().showMessage("Ready")

        # -- Stabilization manager --
        from pygyroflow.manager import StabilizationManager
        self.manager = StabilizationManager()

        # -- Wire signals --
        self._connect_signals()

    # ================================================================== #
    #  Application factory                                                 #
    # ================================================================== #

    @staticmethod
    def create_app() -> QApplication:
        """Create (or return the existing) QApplication instance.

        Call this before constructing MainWindow if you don't already
        have a QApplication.
        """
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        return app

    # ================================================================== #
    #  Menu bar                                                            #
    # ================================================================== #

    def _setup_menu(self) -> None:
        menu = self.menuBar()

        # File menu
        file_menu = menu.addMenu("&File")
        file_menu.addAction("&Open Video...", self._open_video, "Ctrl+O")
        file_menu.addAction("&Export...", self._export_video, "Ctrl+E")
        file_menu.addSeparator()
        file_menu.addAction("&Quit", self.close, "Ctrl+Q")

        # Tools menu
        tools_menu = menu.addMenu("&Tools")
        tools_menu.addAction("&Auto Sync", self._auto_sync, "Ctrl+T")
        tools_menu.addAction("Load &Lens Profile...", self._load_lens)
        tools_menu.addSeparator()
        tools_menu.addAction("&Recompute", self._recompute, "Ctrl+R")

        # View menu
        view_menu = menu.addMenu("&View")
        view_menu.addAction("Toggle &Fullscreen", self._toggle_fullscreen, "F11")

    # ================================================================== #
    #  Signal wiring                                                       #
    # ================================================================== #

    def _connect_signals(self) -> None:
        # Settings panel -> manager
        self.settings_panel.fov_changed.connect(self._on_fov_changed)
        self.settings_panel.zoom_window_changed.connect(self._on_zoom_window_changed)
        self.settings_panel.smoothing_changed.connect(self._on_smoothing_changed)
        self.settings_panel.smoothness_changed.connect(self._on_smoothness_changed)
        self.settings_panel.horizon_amount_changed.connect(self._on_horizon_changed)
        self.settings_panel.sync_requested.connect(self._auto_sync)

        # Lens browser -> manager
        self.lens_browser.profile_selected.connect(self._on_lens_selected)

        # Timeline -> video position
        self.timeline.position_changed.connect(self._on_frame_seek)

    # ================================================================== #
    #  Menu actions                                                        #
    # ================================================================== #

    def _open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            "",
            "Video Files (*.mp4 *.mov *.mkv *.avi *.mxf *.webm);;All Files (*)",
        )
        if not path:
            return

        self.statusBar().showMessage(f"Loading: {path}")
        try:
            info = self.manager.load_video(path)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to load video:\n{exc}")
            self.statusBar().showMessage("Load failed")
            return

        w = info.get("width", "?")
        h = info.get("height", "?")
        fps = info.get("fps", 0)
        frames = info.get("frame_count", "?")
        fps_str = f"{fps:.2f}" if isinstance(fps, (int, float)) and fps > 0 else "?"

        self.statusBar().showMessage(
            f"Loaded: {w}x{h} @ {fps_str} fps, {frames} frames"
        )
        self.video_widget.set_video_info(info)
        self.timeline.set_frame_count(frames if isinstance(frames, int) else 0)

        # Trigger recomputation
        try:
            self.manager.recompute_blocking()
            self.statusBar().showMessage(
                f"Ready — {w}x{h} @ {fps_str} fps, {frames} frames"
            )
        except Exception as exc:
            log.warning("Recompute failed: %s", exc)
            self.statusBar().showMessage(
                f"Loaded (recompute pending) — {w}x{h} @ {fps_str} fps"
            )

    def _export_video(self) -> None:
        if not self.manager.input_file.url:
            QMessageBox.information(self, "Export", "No video loaded.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Video",
            "",
            "MP4 (*.mp4);;MOV (*.mov);;MKV (*.mkv);;All Files (*)",
        )
        if not path:
            return

        self.statusBar().showMessage(f"Exporting to: {path}")
        try:
            self.manager.render(self.manager.input_file.url, path)
            self.statusBar().showMessage(f"Export complete: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", f"Export failed:\n{exc}")
            self.statusBar().showMessage("Export failed")

    def _auto_sync(self) -> None:
        if not self.manager.input_file.url:
            QMessageBox.information(self, "Auto Sync", "No video loaded.")
            return

        # Auto Sync is not wired up: the previous code imported a non-existent
        # ``AutoSync`` class (the real one is ``AutosyncProcess``, with a
        # different API) and swallowed the resulting ImportError, so clicking
        # the menu silently did nothing. Surface this honestly instead.
        QMessageBox.information(
            self,
            "Auto Sync",
            "Auto Sync is not yet implemented in the GUI. Use the CLI or the "
            "synchronization module (find_offset_rs_sync / AutosyncProcess) directly.",
        )
        self.statusBar().showMessage("Auto Sync: not implemented")

    def _load_lens(self) -> None:
        self.lens_browser.browse()
        # Focus the lens browser dock
        for dock in self.findChildren(QDockWidget):
            if dock.widget() is self.lens_browser:
                dock.raise_()
                break

    def _recompute(self) -> None:
        if not self.manager.input_file.url:
            return
        self.statusBar().showMessage("Recomputing...")
        try:
            self.manager.recompute_blocking()
            self.statusBar().showMessage("Recomputation complete")
        except Exception as exc:
            log.warning("Recompute failed: %s", exc)
            self.statusBar().showMessage(f"Recompute failed: {exc}")

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    # ================================================================== #
    #  Slot handlers for settings changes                                  #
    # ================================================================== #

    def _on_fov_changed(self, value: float) -> None:
        self.manager.set_fov(value)

    def _on_zoom_window_changed(self, value: float) -> None:
        self.manager.set_adaptive_zoom(value)

    def _on_smoothing_changed(self, name: str) -> None:
        algo_map = {"None": 0, "Default": 1, "Plain": 2, "Fixed": 3}
        idx = algo_map.get(name, 1)
        self.manager.set_smoothing_method(idx)

    def _on_smoothness_changed(self, value: float) -> None:
        self.manager.set_smoothing_param("smoothness", value)

    def _on_horizon_changed(self, value: float) -> None:
        # Horizon lock maps to additional_rotation roll
        self.manager.params.additional_rotation[2] = value
        self.manager._invalidate_smoothing()

    def _on_lens_selected(self, profile_name: str) -> None:
        self.statusBar().showMessage(f"Loading lens profile: {profile_name}")
        try:
            self.manager.load_lens_profile(profile_name)
            self.statusBar().showMessage(f"Lens profile loaded: {profile_name}")
        except Exception as exc:
            log.warning("Failed to load lens profile %s: %s", profile_name, exc)
            self.statusBar().showMessage(f"Failed to load profile: {exc}")

    def _on_frame_seek(self, frame: int) -> None:
        """Handle timeline scrubber position change."""
        fps = self.manager.params.fps
        if fps > 0:
            timestamp_ms = frame * 1000.0 / fps
            self.statusBar().showMessage(
                f"Frame {frame} @ {timestamp_ms:.1f} ms"
            )
