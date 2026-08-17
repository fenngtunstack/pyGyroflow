"""Settings — persistent application settings.

Port of Gyroflow's settings.rs. Stores key-value settings in a JSON file
in the platform-specific app data directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pygyroflow.filesystem import get_app_data_dir

log = logging.getLogger(__name__)


class Settings:
    """Persistent key-value settings stored in JSON.

    Usage::

        settings = Settings()
        settings.load()
        value = settings.get("key", default=42)
        settings.set("key", 100)
        settings.save()
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._path: Path | None = None

    def load(self, path: str | None = None) -> None:
        """Load settings from a JSON file.

        Args:
            path: Optional path override. Defaults to <app_data>/settings.json.
        """
        if path is not None:
            self._path = Path(path)
        else:
            self._path = get_app_data_dir() / "settings.json"

        if self._path.is_file():
            try:
                text = self._path.read_text(encoding="utf-8")
                self._data = json.loads(text)
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("Failed to load settings from %s: %s", self._path, exc)
                self._data = {}
        else:
            self._data = {}

    def save(self) -> None:
        """Save settings to the JSON file."""
        if self._path is None:
            self._path = get_app_data_dir() / "settings.json"

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        except OSError as exc:
            log.error("Failed to write settings to %s: %s", self._path, exc)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value."""
        return self._data.get(key, default)

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get a boolean setting."""
        val = self._data.get(key)
        if isinstance(val, bool):
            return val
        return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get a float setting."""
        val = self._data.get(key)
        if isinstance(val, (int, float)):
            return float(val)
        return default

    def get_str(self, key: str, default: str = "") -> str:
        """Get a string setting."""
        val = self._data.get(key)
        if isinstance(val, str):
            return val
        return default

    def set(self, key: str, value: Any) -> None:
        """Set a setting value."""
        self._data[key] = value

    def contains(self, key: str) -> bool:
        """Check if a setting exists."""
        return key in self._data

    def clear(self) -> None:
        """Clear all settings."""
        self._data.clear()

    def get_all(self) -> dict[str, Any]:
        """Get a copy of all settings."""
        return dict(self._data)
