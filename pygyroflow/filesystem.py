"""File system utilities — platform-specific paths and file operations.

Provides helpers for locating app data directories, lens profile directories,
and common file operations.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def get_app_data_dir() -> Path:
    """Get platform-specific application data directory.

    Returns:
        Path to the app data directory (created if it doesn't exist).
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        data_dir = base / "PyGyroFlow"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
        data_dir = base / "PyGyroFlow"
    else:
        # Linux / other Unix
        xdg = os.environ.get("XDG_DATA_HOME")
        base = Path(xdg) if xdg else Path.home() / ".local" / "share"
        data_dir = base / "pygyroflow"

    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_lens_profiles_dir() -> Path:
    """Get the lens profiles directory.

    Checks in order:
    1. <app_data>/lens_profiles
    2. <package_dir>/../../resources/camera_presets

    Returns:
        Path to the lens profiles directory (created if using app_data).
    """
    # User's local profiles
    user_dir = get_app_data_dir() / "lens_profiles"
    user_dir.mkdir(parents=True, exist_ok=True)

    # Bundled profiles
    package_dir = Path(__file__).parent
    bundled = package_dir.parent / "resources" / "camera_presets"

    if bundled.is_dir():
        return bundled

    return user_dir


def url_to_path(url: str) -> str:
    """Convert a file URL to a local path."""
    if url.startswith("file://"):
        return url[7:]
    return url


def path_to_url(path: str) -> str:
    """Convert a local path to a file URL."""
    return f"file://{os.path.abspath(path)}"
