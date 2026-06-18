"""Stabilization parameters — user-adjustable settings for the pipeline.

Port of Gyroflow's src/core/stabilization_params.rs StabilizationParams struct.
This is the user-facing settings bag that controls all stabilization behavior.
The ComputeParams snapshot is derived from these settings + GyroSource + LensProfile.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pygyroflow.types.enums import BackgroundMode, ReadoutDirection


@dataclass
class StabilizationParams:
    """All user-adjustable stabilization parameters.

    Field groups mirror the Rust struct layout.
    """

    # Video dimensions
    size: tuple[int, int] = (0, 0)            # Input (width, height)
    output_size: tuple[int, int] = (0, 0)     # Output (width, height)

    # Background fill
    background: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    )

    # Rolling shutter
    frame_readout_time: float = 0.0
    frame_readout_direction: ReadoutDirection = ReadoutDirection.TopToBottom

    # Adaptive zoom
    adaptive_zoom_window: float = 4.0
    adaptive_zoom_center_offset: tuple[float, float] = (0.0, 0.0)
    adaptive_zoom_method: int = 1

    # Additional transforms
    additional_rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    additional_translation: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # FOV control
    fov: float = 1.0
    fov_overview: bool = False
    max_zoom: float | None = 130.0
    max_zoom_iterations: int = 5
    show_safe_area: bool = False

    # Per-frame FOV data (populated by adaptive zoom)
    fovs: list[float] = field(default_factory=list)
    minimal_fovs: list[float] = field(default_factory=list)
    min_fov: float = 1.0

    # Video timing
    fps: float = 0.0
    fps_scale: float | None = None
    video_speed: float = 1.0
    video_speed_affects_smoothing: bool = True
    video_speed_affects_zooming: bool = True
    video_speed_affects_zooming_limit: bool = True
    speed_ramped_timestamps: dict[int, int] | None = None
    frame_count: int = 0
    duration_ms: float = 0.0
    video_created_at: int | None = None

    # Trim ranges
    trim_ranges: list[tuple[float, float]] = field(default_factory=list)

    # Rotation
    video_rotation: float = 0.0

    # Lens
    lens_correction_amount: float = 1.0
    light_refraction_coefficient: float = 1.0

    # Background mode
    background_mode: BackgroundMode = BackgroundMode.SolidColor
    background_margin: float = 0.0
    background_margin_feather: float = 0.0

    # Misc flags
    framebuffer_inverted: bool = False
    is_calibrator: bool = False
    stab_enabled: bool = True
    show_detected_features: bool = True
    show_optical_flow: bool = True

    # Frame offset
    frame_offset: int = 0

    # Optical flow method
    of_method: int = 2
    current_device: int = 0

    # Zooming debug
    zooming_debug_points: dict[int, list[tuple[float, float]]] = field(default_factory=dict)

    # Focal length smoothing
    focal_lengths: list[float | None] = field(default_factory=list)
    smoothed_focal_lengths: list[float | None] = field(default_factory=list)
    focal_length_smoothing_enabled: bool = False
    focal_length_smoothing_strength: float = 0.5

    # ------------------------------------------------------------------
    # Computed helpers
    # ------------------------------------------------------------------

    def get_trim_ratio(self) -> float:
        """Total duration fraction covered by trim ranges."""
        if not self.trim_ranges:
            return 1.0
        return sum(end - start for start, end in self.trim_ranges)

    def get_scaled_duration_ms(self) -> float:
        """Duration adjusted for FPS scaling."""
        if self.fps_scale is not None:
            return self.duration_ms / self.fps_scale
        return self.duration_ms

    def get_scaled_fps(self) -> float:
        """FPS adjusted for scaling."""
        if self.fps_scale is not None:
            return self.fps * self.fps_scale
        return self.fps

    def set_fovs(self, fovs: list[float], lens_fov_adjustment: float = 1.0) -> None:
        """Set per-frame FOV values and compute min_fov."""
        if fovs:
            min_fov = min(fovs)
            min_fov *= self.size[0] / max(1, self.output_size[0])
            if lens_fov_adjustment <= 0.0001:
                lens_fov_adjustment = 1.0
            self.min_fov = min_fov / lens_fov_adjustment
        else:
            self.min_fov = 1.0
        self.fovs = fovs

    def clear(self) -> None:
        """Reset to defaults, preserving UI preferences."""
        preserved = StabilizationParams(
            stab_enabled=self.stab_enabled,
            show_detected_features=self.show_detected_features,
            show_optical_flow=self.show_optical_flow,
            background=self.background.copy(),
            adaptive_zoom_window=self.adaptive_zoom_window,
            framebuffer_inverted=self.framebuffer_inverted,
            lens_correction_amount=self.lens_correction_amount,
            video_speed=self.video_speed,
            video_speed_affects_smoothing=self.video_speed_affects_smoothing,
            video_speed_affects_zooming=self.video_speed_affects_zooming,
            video_speed_affects_zooming_limit=self.video_speed_affects_zooming_limit,
            light_refraction_coefficient=self.light_refraction_coefficient,
            background_mode=self.background_mode,
            background_margin=self.background_margin,
            background_margin_feather=self.background_margin_feather,
            of_method=self.of_method,
            current_device=self.current_device,
            adaptive_zoom_method=self.adaptive_zoom_method,
            fov_overview=self.fov_overview,
            show_safe_area=self.show_safe_area,
            max_zoom=self.max_zoom,
            max_zoom_iterations=self.max_zoom_iterations,
            focal_length_smoothing_enabled=self.focal_length_smoothing_enabled,
            focal_length_smoothing_strength=self.focal_length_smoothing_strength,
        )
        self.__dict__.update(preserved.__dict__)
