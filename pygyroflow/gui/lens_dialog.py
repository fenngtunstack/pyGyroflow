"""Lens profile browser — left-docked search and selection widget.

Searches the LensProfileDatabase and displays matching profiles in a list.
Selecting a profile loads it into the StabilizationManager.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

log = logging.getLogger(__name__)


class LensProfileBrowser(QWidget):
    """Left-docked lens profile search and selection panel.

    Signals
    -------
    profile_selected : str
        Emitted with the profile name when the user selects a profile.
    """

    profile_selected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        # -- Search bar --
        search_row = QHBoxLayout()
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search lens profiles...")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._on_search)
        search_row.addWidget(self._search)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setFixedWidth(70)
        self._refresh_btn.clicked.connect(self.browse)
        search_row.addWidget(self._refresh_btn)
        layout.addLayout(search_row)

        # -- Profile list --
        self._list = QListWidget()
        self._list.currentItemChanged.connect(self._on_item_changed)
        layout.addWidget(self._list, stretch=1)

        # -- Profile info --
        self._info = QLabel("No profile selected")
        self._info.setWordWrap(True)
        self._info.setStyleSheet("color: #aaa; font-size: 11px; padding: 4px;")
        self._info.setMinimumHeight(48)
        layout.addWidget(self._info)

        # -- Database --
        from pygyroflow.lens import LensProfileDatabase
        self._db = LensProfileDatabase()
        self._loaded = False

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def browse(self) -> None:
        """Load the database (if needed) and refresh the profile list."""
        if not self._loaded:
            try:
                self._db.load_all()
                self._loaded = True
            except Exception as exc:
                log.warning("Failed to load lens database: %s", exc)
                self._info.setText(f"Error loading database: {exc}")
                return

        self._on_search(self._search.text())

    def get_selected_profile_name(self) -> str | None:
        """Return the name of the currently selected profile, or None."""
        item = self._list.currentItem()
        if item is not None:
            return item.data(Qt.ItemDataRole.UserRole)
        return None

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _on_search(self, text: str) -> None:
        self._list.clear()
        if not self._loaded:
            return

        try:
            results = self._db.search(text)
        except Exception as exc:
            log.warning("Search error: %s", exc)
            return

        for profile in results[:100]:
            display = profile.get_display_name()
            item = QListWidgetItem(display)
            item.setData(Qt.ItemDataRole.UserRole, profile.name or display)
            self._list.addItem(item)

        if not results and text:
            item = QListWidgetItem(f"No results for \"{text}\"")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self._list.addItem(item)

    def _on_item_changed(
        self, current: QListWidgetItem | None, _previous: QListWidgetItem | None
    ) -> None:
        if current is None:
            self._info.setText("No profile selected")
            return

        profile_name = current.data(Qt.ItemDataRole.UserRole)
        profile = self._db.get_by_name(profile_name) if self._loaded else None

        if profile is not None:
            w = profile.calib_dimension["w"]
            h = profile.calib_dimension["h"]
            ar = profile.get_aspect_ratio()
            size = profile.get_size_str()
            model = profile.distortion_model or "none"
            info_text = (
                f"{profile.camera_brand} {profile.camera_model}\n"
                f"{w}x{h} ({size}, {ar})\n"
                f"Distortion: {model}"
            )
            self._info.setText(info_text)
            self.profile_selected.emit(profile_name)
        else:
            self._info.setText(profile_name)
