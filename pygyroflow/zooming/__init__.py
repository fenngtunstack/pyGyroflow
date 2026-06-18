"""Adaptive zoom — per-frame FOV computation and smoothing.

This package provides the adaptive zoom pipeline:
  1. FovIterative computes raw per-frame FOV (minimum zoom without black borders)
  2. ZoomDynamic smooths the FOV sequence over time
  3. calculate_fovs() orchestrates both and applies the zoom strategy

Usage:
    from pygyroflow.zooming import calculate_fovs, ZoomMethod

    fovs, minimal_fovs = calculate_fovs(params, timestamps, ZoomMethod.EnvelopeFollower)
"""

from __future__ import annotations

import hashlib
import struct
from typing import Optional

from pygyroflow.stabilization.compute_params import ComputeParams
from pygyroflow.zooming.fov_iterative import FovIterative
from pygyroflow.zooming.zoom_dynamic import ZoomMethod, compute as zoom_dynamic_compute


def calculate_fovs(
    compute_params: ComputeParams,
    timestamps: list[tuple[int, float]],
    method: ZoomMethod = ZoomMethod.EnvelopeFollower,
) -> tuple[list[float], list[float]]:
    """Calculate per-frame FOV scale factors.

    Entry point for the adaptive zoom system. Three strategies based on
    adaptive_zoom_window:
      - window < -0.9: static zoom (use global minimum FOV for all frames)
      - window > 0.0001: dynamic zoom (smooth with Gaussian or envelope)
      - otherwise: disabled (FOV = 1.0 for all frames)

    Args:
        compute_params: Stabilization parameters. Modified internally to
            use original dimensions for FOV estimation.
        timestamps: List of (frame_index, timestamp_ms).
        method: Dynamic smoothing algorithm.

    Returns:
        (final_fovs, minimal_fovs) where minimal_fovs are the raw per-frame
        minimum FOVs and final_fovs are the smoothed/static/disabled result.
    """
    if not timestamps:
        return ([], [])

    # Work on a copy with original dimensions for FOV estimation
    params = ComputeParams(
        width=compute_params.width,
        height=compute_params.height,
        output_width=compute_params.width,  # Use input dims for FOV estimation
        output_height=compute_params.height,
        frame_count=compute_params.frame_count,
        video_rotation=compute_params.video_rotation,
        scaled_fps=compute_params.scaled_fps,
        scaled_duration_ms=compute_params.scaled_duration_ms,
        quaternions=compute_params.quaternions,
        smoothed_quaternions=compute_params.smoothed_quaternions,
        fovs=[],  # Clear for fresh computation
        minimal_fovs=[],
        fov_scale=1.0,  # Reset to neutral
        fov_overview=compute_params.fov_overview,
        show_safe_area=compute_params.show_safe_area,
        max_zoom=compute_params.max_zoom,
        max_zoom_iterations=compute_params.max_zoom_iterations,
        camera_matrix=compute_params.camera_matrix.copy(),
        distortion_coeffs=list(compute_params.distortion_coeffs),
        distortion_model_name=compute_params.distortion_model_name,
        lens_correction_amount=compute_params.lens_correction_amount,
        light_refraction_coefficient=compute_params.light_refraction_coefficient,
        frame_readout_time=compute_params.frame_readout_time,
        frame_readout_direction=compute_params.frame_readout_direction,
        background=compute_params.background.copy(),
        background_mode=compute_params.background_mode,
        background_margin=compute_params.background_margin,
        background_margin_feather=compute_params.background_margin_feather,
        adaptive_zoom_window=compute_params.adaptive_zoom_window,
        adaptive_zoom_center_offset=compute_params.adaptive_zoom_center_offset,
        adaptive_zoom_method=compute_params.adaptive_zoom_method,
        additional_rotation=compute_params.additional_rotation,
        additional_translation=compute_params.additional_translation,
        video_speed=compute_params.video_speed,
        video_speed_affects_smoothing=compute_params.video_speed_affects_smoothing,
        video_speed_affects_zooming=compute_params.video_speed_affects_zooming,
        framebuffer_inverted=compute_params.framebuffer_inverted,
        suppress_rotation=compute_params.suppress_rotation,
        trim_ranges=list(compute_params.trim_ranges),
        fov_algorithm_margin=compute_params.fov_algorithm_margin,
        horizontal_stretch=compute_params.horizontal_stretch,
        calib_width=compute_params.calib_width,
        calib_height=compute_params.calib_height,
        input_horizontal_stretch=compute_params.input_horizontal_stretch,
        input_vertical_stretch=compute_params.input_vertical_stretch,
        focal_length=compute_params.focal_length,
        radial_distortion_limit=compute_params.radial_distortion_limit,
    )

    # The actual output dimensions (for aspect ratio calculation)
    org_output_size = (compute_params.output_width, compute_params.output_height)

    fov_estimator = FovIterative(params, org_output_size)
    fov_values = fov_estimator.compute(timestamps, params.trim_ranges)

    window = params.adaptive_zoom_window

    if window < -0.9:
        # Static zoom: all frames use the same (maximum needed) FOV
        fov_minimal = list(fov_values)
        if fov_values:
            max_f = min(fov_values)
            fov_values = [max_f] * len(fov_values)
        return fov_values, fov_minimal

    elif window > 0.0001:
        # Dynamic zoom: smooth with chosen method
        return zoom_dynamic_compute(params, fov_values, timestamps, method)

    else:
        # Disabled zoom: no cropping
        return ([1.0] * len(fov_values), fov_values)


def get_checksum(compute_params: ComputeParams) -> str:
    """Generate a hash of parameters affecting FOV computation.

    Used for caching FOV results to avoid recomputation when
    parameters haven't changed.

    Args:
        compute_params: Stabilization parameters.

    Returns:
        Hex digest string.
    """
    hasher = hashlib.sha256()

    for x in compute_params.distortion_coeffs:
        hasher.update(struct.pack('d', x))

    for val in [
        compute_params.width,
        compute_params.height,
        compute_params.output_width,
        compute_params.output_height,
        compute_params.scaled_fps,
        compute_params.max_zoom if compute_params.max_zoom is not None else 0.0,
        compute_params.max_zoom_iterations,
        compute_params.video_rotation,
        compute_params.adaptive_zoom_window,
    ]:
        hasher.update(struct.pack('d', float(val)))

    for start, end in compute_params.trim_ranges:
        hasher.update(struct.pack('d', start))
        hasher.update(struct.pack('d', end))

    return hasher.hexdigest()


__all__ = [
    "calculate_fovs",
    "get_checksum",
    "FovIterative",
    "ZoomMethod",
]
