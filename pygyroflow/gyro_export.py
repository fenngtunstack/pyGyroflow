"""Gyro data export — CSV and JSON output of quaternion streams.

Port of Gyroflow's gyro_export functionality. Exports original and stabilized
quaternion data in common interchange formats.
"""

from __future__ import annotations

import csv
import json
import math
from typing import Any

from pygyroflow.types.time_types import TimeQuat


def export_gyro_csv(
    quaternions: TimeQuat,
    path: str,
    include_header: bool = True,
) -> None:
    """Export quaternions to CSV file.

    Args:
        quaternions: Timestamp_us -> Quat64 mapping.
        path: Output file path.
        include_header: Whether to include column headers.
    """
    keys = sorted(quaternions.keys())

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        if include_header:
            writer.writerow(["timestamp_us", "quat_w", "quat_x", "quat_y", "quat_z", "timestamp_ms"])
        for ts in keys:
            q = quaternions[ts].quaternion()
            writer.writerow([ts, f"{q[0]:.10f}", f"{q[1]:.10f}", f"{q[2]:.10f}", f"{q[3]:.10f}", f"{ts / 1000.0:.6f}"])


def export_gyro_json(
    quaternions: TimeQuat,
    path: str,
    pretty: bool = True,
) -> None:
    """Export quaternions to JSON file.

    Args:
        quaternions: Timestamp_us -> Quat64 mapping.
        path: Output file path.
        pretty: Whether to pretty-print JSON.
    """
    keys = sorted(quaternions.keys())
    data = {
        str(ts): {
            "w": float(quaternions[ts].quaternion()[0]),
            "x": float(quaternions[ts].quaternion()[1]),
            "y": float(quaternions[ts].quaternion()[2]),
            "z": float(quaternions[ts].quaternion()[3]),
            "timestamp_ms": ts / 1000.0,
        }
        for ts in keys
    }

    with open(path, "w") as f:
        if pretty:
            json.dump(data, f, indent=2)
        else:
            json.dump(data, f)


def export_gyro_csv_full(
    original: TimeQuat,
    smoothed: TimeQuat,
    path: str,
    fovs: list[float] | None = None,
    fps: float = 0.0,
) -> None:
    """Export full gyro data (original + stabilized) to CSV.

    Args:
        original: Original quaternion stream.
        smoothed: Smoothed quaternion stream.
        path: Output file path.
        fovs: Per-frame FOV values.
        fps: Video FPS for frame calculation.
    """
    rad2deg = 180.0 / math.pi

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "timestamp_us", "timestamp_ms",
            "org_quat_w", "org_quat_x", "org_quat_y", "org_quat_z",
            "stab_quat_w", "stab_quat_x", "stab_quat_y", "stab_quat_z",
        ]
        if fovs is not None:
            header.append("fov_scale")
        writer.writerow(header)

        for ts in sorted(original.keys()):
            ts_ms = ts / 1000.0
            oq = original[ts].quaternion()
            sq = smoothed.get(ts, original[ts]).quaternion()

            row = [
                ts, f"{ts_ms:.6f}",
                f"{oq[0]:.10f}", f"{oq[1]:.10f}", f"{oq[2]:.10f}", f"{oq[3]:.10f}",
                f"{sq[0]:.10f}", f"{sq[1]:.10f}", f"{sq[2]:.10f}", f"{sq[3]:.10f}",
            ]

            if fovs is not None and fps > 0:
                frame = int(ts_ms * fps / 1000.0)
                fov_val = fovs[frame] if frame < len(fovs) else 1.0
                row.append(f"{fov_val:.6f}")

            writer.writerow(row)
