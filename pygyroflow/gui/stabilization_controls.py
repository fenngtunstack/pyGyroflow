"""Stabilization controls — thin re-export wrapper.

All stabilization controls live in :mod:`pygyroflow.gui.settings_panel`.
This module exists as a convenience import point and for backward
compatibility if any code imports from ``stabilization_controls`` directly.
"""

from pygyroflow.gui.settings_panel import SettingsPanel  # noqa: F401

__all__ = ["SettingsPanel"]
