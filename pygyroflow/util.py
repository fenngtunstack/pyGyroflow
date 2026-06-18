"""Shared utility functions.

Port of Gyroflow's timestamp_at_frame / frame_at_timestamp from lib.rs.
"""

from __future__ import annotations


def timestamp_at_frame(frame: int, fps: float) -> float:
    """Convert frame index to timestamp in milliseconds.

    Args:
        frame: Zero-based frame index.
        fps: Video frame rate.

    Returns:
        Timestamp in milliseconds.
    """
    return frame * 1000.0 / fps if fps > 0 else 0.0


def frame_at_timestamp(timestamp_ms: float, fps: float) -> int:
    """Convert timestamp to frame index.

    Args:
        timestamp_ms: Timestamp in milliseconds.
        fps: Video frame rate.

    Returns:
        Zero-based frame index (rounded).
    """
    return round(timestamp_ms * fps / 1000.0) if fps > 0 else 0
