"""File metadata from telemetry parsing.

Port of Gyroflow's src/core/gyro_source/file_metadata.rs FileMetadata struct.
Stores parsed telemetry data: raw IMU, quaternions, gravity vectors,
lens parameters, camera identifier, and per-frame time offsets.

In the Rust codebase this is wrapped in ReadOnlyFileMetadata (Arc<RwLock>).
Python doesn't need the thread-safe wrapper since we don't have the same
concurrency model — the GyroSource holds the metadata directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pygyroflow.types.enums import ReadoutDirection
from pygyroflow.types.time_types import TimeIMU, TimeQuat, TimeVec


@dataclass
class LensParams:
    """Per-frame lens metadata from telemetry."""

    focal_length: float | None = None       # mm
    pixel_pitch: tuple[int, int] | None = None  # nm
    sensor_size_px: tuple[int, int] | None = None
    capture_area_origin: tuple[float, float] | None = None
    capture_area_size: tuple[float, float] | None = None
    pixel_focal_length: float | None = None  # pixels
    distortion_coefficients: list[float] = field(default_factory=list)
    focus_distance: float | None = None


@dataclass
class FileMetadata:
    """Parsed telemetry metadata from a video file.

    Once populated, this data is not modified — it serves as the immutable
    reference for raw IMU data and pre-integrated quaternions.
    """

    imu_orientation: str | None = None
    raw_imu: list[TimeIMU] = field(default_factory=list)
    quaternions: TimeQuat = field(default_factory=dict)
    gravity_vectors: TimeVec | None = None
    image_orientations: TimeQuat | None = None
    detected_source: str | None = None
    frame_readout_time: float | None = None
    frame_readout_direction: ReadoutDirection = ReadoutDirection.TopToBottom
    frame_rate: float | None = None
    camera_identifier: Any | None = None  # CameraIdentifier when available
    lens_profile: Any | None = None       # JSON value or string
    lens_positions: dict[int, float] = field(default_factory=dict)
    lens_params: dict[int, LensParams] = field(default_factory=dict)
    digital_zoom: float | None = None
    has_accurate_timestamps: bool = False
    additional_data: dict = field(default_factory=dict)
    per_frame_time_offsets: list[float] = field(default_factory=list)
    camera_stab_data: list = field(default_factory=list)
    mesh_correction: list = field(default_factory=list)

    def has_motion(self) -> bool:
        """Check if the file contains any motion data."""
        return len(self.raw_imu) > 0 or len(self.quaternions) > 0

    def thin(self) -> FileMetadata:
        """Return a lightweight copy without large data arrays.

        Used for project file export — keeps metadata but drops raw IMU
        and quaternions to save space.
        """
        return FileMetadata(
            imu_orientation=self.imu_orientation,
            raw_imu=[],
            quaternions={},
            gravity_vectors=None,
            image_orientations=None,
            detected_source=self.detected_source,
            frame_readout_time=self.frame_readout_time,
            frame_readout_direction=self.frame_readout_direction,
            frame_rate=self.frame_rate,
            camera_identifier=self.camera_identifier,
            lens_profile=self.lens_profile,
            lens_positions={},
            lens_params={},
            digital_zoom=self.digital_zoom,
            has_accurate_timestamps=self.has_accurate_timestamps,
            additional_data=dict(self.additional_data),
            per_frame_time_offsets=[],
            camera_stab_data=[],
            mesh_correction=[],
        )
