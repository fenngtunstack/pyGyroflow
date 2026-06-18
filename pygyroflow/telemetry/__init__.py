"""Telemetry module — gyro data extraction from video files.

Provides parse_telemetry_file() which tries the native bridge first,
then falls back to a basic GPMF parser for GoPro files.
"""

from pygyroflow.telemetry.parser import parse_telemetry_file

__all__ = ["parse_telemetry_file"]
