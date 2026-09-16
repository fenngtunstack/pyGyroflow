"""Compute parameters snapshot for stabilization pipeline.

Port of Gyroflow's ComputeParams struct. This is a frozen snapshot of all
parameters needed for frame transform computation, extracted from the
stabilization manager to avoid holding locks during rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from pygyroflow.types.enums import BackgroundMode, ReadoutDirection
from pygyroflow.types.time_types import TimeQuat, TimeVec
from pygyroflow.util import ClosestMap


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

    # --- Focal length smoothing (upstream compute_params.rs:65-68) ---
    # A zoom lens changes focal length mid-clip, and cameras report it in
    # coarse steps. `focal_lengths` holds the DEQUANTIZED curve (the short
    # Gaussian pass, smoothing/focal_length.py), not the raw metadata: it is
    # the denominator of the compensation ratio, so stairs in it would become
    # stairs in the sampling position. `smoothed_focal_lengths` is
    # dequantized/adaptive-filtered — the curve the output actually tracks.
    # Both are frame-indexed and only populated when smoothing is active; the
    # raw curve for the UI timeline lives on StabilizationParams.
    focal_lengths: list[float | None] = field(default_factory=list)
    smoothed_focal_lengths: list[float | None] = field(default_factory=list)
    focal_length_smoothing_enabled: bool = False
    focal_length_smoothing_strength: float = 0.0

    # --- Per-timestamp lens data (FileMetadata.lens_positions / lens_params) ---
    # A zoom lens changes focal length during the clip, so its calibration
    # changes with it. Both maps are empty for a fixed-focal-length clip,
    # which is the common case and leaves every consumer on the static path
    # above; when they are not, `FrameTransform.get_lens_data_at_timestamp`
    # prefers them (upstream frame_transform.rs).
    #
    # `lens_positions` is a scalar per time — a focal length in mm, or a Sony
    # crop score — used to interpolate *within* the profile's own
    # `interpolations` table. `lens_params` carries raw per-frame intrinsics
    # that override the profile outright. They are independent: one does not
    # feed the other.
    lens_positions: ClosestMap = field(default_factory=ClosestMap)
    lens_params: ClosestMap = field(default_factory=ClosestMap)
    # The LensProfile itself, so a position lookup can interpolate a profile
    # out of its `interpolations` table (LensProfile.get_interpolated_profile_at).
    lens: Any = None
    # FileMetadata.digital_zoom: a crop factor the camera decided on.
    digital_zoom: Optional[float] = None

    # --- Per-frame stabilization data, consumed only by the points path ---
    # FileMetadata.mesh_correction: one (distorting mesh, undistorting mesh)
    # pair per frame. The points path reads the first element. Upstream reaches
    # into the gyro source for these; copying them here keeps FrameTransform
    # free of a gyro reference, the same way `per_frame_time_offsets` works.
    mesh_correction: list = field(default_factory=list)
    # FileMetadata.camera_stab_data: one CameraStabData per frame — the IBIS
    # and OIS displacement splines plus the crop area they are expressed in.
    camera_stab_data: list = field(default_factory=list)
    # The lens's *digital* lens, a second distortion model applied on top of
    # the optical one (GoPro's SuperView/HyperView stretch). `digital_lens` is
    # the model, `digital_lens_params` its coefficients.
    digital_lens: Any = None
    digital_lens_params: list | None = None

    def calculate_camera_fovs(self) -> None:
        """Fill ``camera_diagonal_fovs``, one value per frame.

        Port of ``ComputeParams::calculate_camera_fovs`` (compute_params.rs).
        A fixed-focal-length clip gets a single value and is left at that:
        the FOV cannot change, so a per-frame pass would be ``frame_count``
        identical lookups. Only a clip whose calibration actually moves —
        a zoom lens, i.e. more than one ``lens_params`` entry — needs the
        per-frame array, which DefaultAlgo consumes to scale its velocity
        threshold by ``dfov / 120``.
        """
        import math

        from pygyroflow.stabilization.frame_transform import (
            _get_lens_data_at_timestamp,
        )
        from pygyroflow.util import timestamp_at_frame

        frame_count = self.frame_count if len(self.lens_params) > 1 else 1
        diagonal_px = math.hypot(self.width, self.height)
        fovs: list[float] = []
        for frame in range(frame_count):
            timestamp_ms = timestamp_at_frame(frame, self.scaled_fps)
            matrix, _, _, _, _, _ = _get_lens_data_at_timestamp(self, timestamp_ms)
            fy = matrix[1, 1] if matrix[1, 1] else 0.0
            if fy == 0.0:
                fovs.append(120.0)
                continue
            fovs.append(
                2.0 * math.degrees(math.atan(diagonal_px / (2.0 * fy)))
            )
        self.camera_diagonal_fovs = fovs

    # Keyframes
    keyframes: 'KeyframeManager' = field(default_factory=lambda: __import__('pygyroflow.keyframes', fromlist=['KeyframeManager']).KeyframeManager())

    # Radial distortion limit
    radial_distortion_limit: float = 0.0
    # LensProfile.optimal_fov: the FOV the lens is sharpest at, from the
    # profile JSON. Upstream divides the *UI* fov by it when per-frame fovs
    # exist, and multiplies the render fov by it when they don't
    # (frame_transform.rs). None means "no adjustment".
    optimal_fov: float | None = None
    # FileMetadata.per_frame_time_offsets: per-frame timestamp corrections
    # (Sony/DJI/RED). Upstream adds offsets[frame] to the video timestamp
    # before the rolling-shutter and quaternion lookups.
    per_frame_time_offsets: list[float] = field(default_factory=list)
    # Per-frame relaxation of the smoothing (upstream
    # ComputeParams.smoothing_fov_limit_per_frame). The max-zoom feedback
    # loop fills it: where the required crop would exceed the zoom limit,
    # smoothing is relaxed instead of exceeding it. Consumed by
    # DefaultAlgo/PlainSmoothing as a multiplier on max_velocity.
    smoothing_fov_limit_per_frame: list[float] = field(default_factory=list)
    video_speed_affects_zooming_limit: bool = True
