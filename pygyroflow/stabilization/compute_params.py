"""Compute parameters snapshot for stabilization pipeline.

Port of Gyroflow's ComputeParams struct. This is a frozen snapshot of all
parameters needed for frame transform computation, extracted from the
stabilization manager to avoid holding locks during rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from pygyroflow.types.enums import BackgroundMode, ReadoutDirection
from pygyroflow.types.time_types import TimeQuat, TimeVec


@dataclass
class ComputeParams:
    """Snapshot of all parameters needed for frame transform computation.

    Created from the stabilization manager state, this dataclass holds
    everything FrameTransform.at_timestamp needs without requiring
    access to the manager's locks.

    Field groups mirror the Rust struct layout:
      - Video: dimensions, frame count, rotation, fps
      - Gyro: raw and smoothed quaternion streams
      - FOV: per-frame FOV scale factors
      - Lens: camera intrinsics and distortion coefficients
      - Rolling shutter: readout time and direction
      - Background: fill color and mode
      - Adaptive zoom: window size, center offset, method
      - Transforms: additional rotation/translation, video speed
    """

    # Video dimensions
    width: int = 0
    height: int = 0
    output_width: int = 0
    output_height: int = 0
    frame_count: int = 0
    video_rotation: float = 0.0
    scaled_fps: float = 0.0
    scaled_duration_ms: float = 0.0

    # Gyro data — timestamp_us -> Quat64
    quaternions: TimeQuat = field(default_factory=dict)
    smoothed_quaternions: TimeQuat = field(default_factory=dict)

    # Gyro-to-video sync offsets (timestamp_us -> offset_ms), applied when
    # looking up quaternions by video timestamp (mirrors Gyroflow's
    # GyroSource::quat_at_timestamp offset correction).
    sync_offsets_adjusted: dict[int, float] = field(default_factory=dict)

    # Per-frame FOV scale factors (from adaptive zoom)
    fovs: list[float] = field(default_factory=list)
    minimal_fovs: list[float] = field(default_factory=list)

    # Camera diagonal FOV in degrees per frame (from the lens intrinsics).
    # Consumed by DefaultAlgo to scale max_velocity (fov_ratio = dfov / 120).
    camera_diagonal_fovs: list[float] = field(default_factory=lambda: [120.0])

    # Cached pre-sorted quaternion keys for the zooming hot path (not part
    # of the upstream struct; plain perf caches, never serialized).
    _fov_org_keys: list[int] | None = None
    _fov_smoothed_keys: list[int] | None = None

    # FOV control
    fov_scale: float = 1.0
    fov_overview: bool = False
    show_safe_area: bool = False
    max_zoom: Optional[float] = 130.0
    max_zoom_iterations: int = 5

    # Lens parameters
    # Camera intrinsics matrix K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    camera_matrix: NDArray[np.float64] = field(
        default_factory=lambda: np.eye(3, dtype=np.float64)
    )
    # Distortion coefficients (up to 12 for various models)
    distortion_coeffs: list[float] = field(default_factory=lambda: [0.0] * 12)
    distortion_model_name: str = "opencv_fisheye"
    lens_correction_amount: float = 1.0
    light_refraction_coefficient: float = 1.0

    # Rolling shutter
    frame_readout_time: float = 0.0
    frame_readout_direction: ReadoutDirection = ReadoutDirection.TopToBottom

    # Background fill
    background: NDArray[np.float32] = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    )
    background_mode: BackgroundMode = BackgroundMode.SolidColor
    background_margin: float = 0.0
    background_margin_feather: float = 0.0

    # Adaptive zoom
    adaptive_zoom_window: float = 4.0
    adaptive_zoom_center_offset: tuple[float, float] = (0.0, 0.0)
    adaptive_zoom_method: int = 1

    # Additional transforms
    additional_rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    additional_translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    video_speed: float = 1.0
    video_speed_affects_smoothing: bool = False
    video_speed_affects_zooming: bool = False

    # Misc
    framebuffer_inverted: bool = False
    suppress_rotation: bool = False
    trim_ranges: list[tuple[float, float]] = field(default_factory=list)
    fov_algorithm_margin: float = 2.0
    horizontal_stretch: float = 0.0

    # Lens calibration dimension (0 means use video dimensions)
    calib_width: int = 0
    calib_height: int = 0
    input_horizontal_stretch: float = 1.0
    input_vertical_stretch: float = 1.0

    # Focal length in mm (None if unknown)
    focal_length: Optional[float] = None

    # Keyframes
    keyframes: 'KeyframeManager' = field(default_factory=lambda: __import__('pygyroflow.keyframes', fromlist=['KeyframeManager']).KeyframeManager())

    # Radial distortion limit
    radial_distortion_limit: float = 0.0
