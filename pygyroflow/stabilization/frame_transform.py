"""Frame transform computation — the core stabilization algorithm.

Port of Gyroflow's frame_transform.rs. Computes per-frame (or per-row for
rolling shutter) transformation matrices from gyro quaternion data, lens
parameters, and FOV scaling.

The key function is FrameTransform.at_timestamp(), which generates all data
needed for GPU/CPU undistortion:
  - matrices: list of 14-float entries (9 for 3x3 rotation + 5 IBIS params)
  - kernel_params: ctypes struct for the GPU compute shader
  - fov: the FOV scale factor used
"""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from pygyroflow.types.enums import BackgroundMode, ReadoutDirection
from pygyroflow.types.kernel_params import KernelParams
from pygyroflow.types.quaternion import Quat64
from pygyroflow.gyro_source.source import GyroSource
from pygyroflow.keyframes.types import KeyframeType as KT
from pygyroflow.stabilization.compute_params import ComputeParams


def _quat_at_timestamp(
    quats: dict[int, Quat64],
    timestamp_us: float,
    keys: list[int] | None = None,
) -> Quat64:
    """Interpolate a quaternion at the given timestamp (microseconds).

    Uses nearest-neighbor lookup within the sorted timestamp keys,
    falling back to slerp between bracketing keys.

    Args:
        quats: Map of timestamp_us -> Quat64.
        timestamp_us: Query timestamp in microseconds.
        keys: Optional pre-sorted list of ``quats`` keys. Pass this when
            calling in a tight loop (e.g. per rolling-shutter row) to avoid
            re-sorting on every call — sorting is O(N log N) and dominated
            per-frame cost when called once per output row.

    Returns:
        Interpolated quaternion, or identity if no data.
    """
    if not quats:
        return Quat64.identity()

    ts = round(timestamp_us)
    if keys is None:
        keys = sorted(quats.keys())

    if ts <= keys[0]:
        return quats[keys[0]]
    if ts >= keys[-1]:
        return quats[keys[-1]]

    # Binary search for bracketing keys
    lo, hi = 0, len(keys) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if keys[mid] <= ts:
            lo = mid
        else:
            hi = mid

    if keys[lo] == ts:
        return quats[keys[lo]]

    t0, t1 = keys[lo], keys[hi]
    alpha = (ts - t0) / (t1 - t0) if t1 != t0 else 0.0
    return quats[t0].slerp(quats[t1], alpha)


def _get_frame_readout_time(
    params: ComputeParams,
    timestamp_ms: float,
    can_invert: bool = False,
) -> float:
    """Compute effective frame readout time in seconds.

    Accounts for readout direction and framebuffer inversion.

    Args:
        params: Compute parameters.
        timestamp_ms: Frame timestamp in ms.
        can_invert: Whether to invert for framebuffer direction.

    Returns:
        Frame readout time (may be negative for inverted directions).
    """
    frt = abs(params.frame_readout_time)

    if can_invert and params.framebuffer_inverted and not params.frame_readout_direction.is_horizontal():
        frt *= -1.0
    if params.frame_readout_direction.is_inverted():
        frt *= -1.0

    return frt


def _get_new_k(
    params: ComputeParams,
    camera_matrix: NDArray[np.float64],
    fov: float,
) -> NDArray[np.float64]:
    """Build output camera intrinsics matrix (new K).

    Scales focal length by horizontal stretch and FOV, re-centers principal
    point to output image center.

    Args:
        params: Compute parameters with output dimensions.
        camera_matrix: Original 3x3 camera matrix.
        fov: FOV scale factor (>1 = zoom in).

    Returns:
        New 3x3 camera matrix for the output image.
    """
    horizontal_ratio = params.input_horizontal_stretch if params.input_horizontal_stretch > 0.01 else 1.0
    img_dim_ratio = 1.0 / horizontal_ratio

    out_w = float(params.output_width)
    out_h = float(params.output_height)

    new_k = camera_matrix.copy()
    new_k[0, 0] = new_k[0, 0] * img_dim_ratio / fov
    new_k[1, 1] = new_k[1, 1] * img_dim_ratio / fov
    new_k[0, 2] = out_w / 2.0
    new_k[1, 2] = out_h / 2.0

    return new_k


def _get_fov(
    params: ComputeParams,
    frame: int,
    use_fovs: bool,
    timestamp_ms: float,
    for_ui: bool = False,
) -> float:
    """Compute FOV scale factor for a given frame.

    Combines keyframe animation, adaptive zoom FOVs, and input/output
    resolution ratio.

    Args:
        params: Compute parameters.
        frame: Frame index for per-frame FOV lookup.
        use_fovs: Whether to use the per-frame FOV list.
        timestamp_ms: Frame timestamp in ms.
        for_ui: If True, skip the +1.0 overview offset.

    Returns:
        FOV scale factor (clamped to >= 0.001).
    """
    fov_scale = params.fov_scale
    if params.keyframes:
        kf_val = params.keyframes.value_at_video_timestamp(KT.Fov, timestamp_ms)
        if kf_val is not None:
            fov_scale = kf_val

    fov_overview_offset = 1.0 if params.fov_overview and use_fovs and not for_ui else 0.0
    fov_scale += fov_overview_offset

    if use_fovs and params.fovs:
        per_frame = params.fovs[frame] if frame < len(params.fovs) else (
            params.fovs[-1] if len(params.fovs) > 1 else 1.0
        )
        fov = per_frame * fov_scale
    else:
        fov = 1.0

    fov = max(fov, 0.001)

    # Scale by input/output resolution ratio
    fov *= params.width / max(1, params.output_width)

    return fov


def _get_lens_data_at_timestamp(
    params: ComputeParams,
    timestamp_ms: float,
) -> tuple[NDArray[np.float64], list[float], float]:
    """Get camera intrinsics and distortion coefficients at a timestamp.

    For zoom lenses, interpolates between calibration entries.

    Args:
        params: Compute parameters.
        timestamp_ms: Frame timestamp in ms.

    Returns:
        Tuple of (camera_matrix 3x3, distortion_coeffs list[12], radial_distortion_limit).
    """
    camera_matrix = params.camera_matrix.copy()
    distortion_coeffs = list(params.distortion_coeffs)

    # Scale camera matrix from calibration resolution to video resolution
    calib_w = params.calib_width if params.calib_width > 0 else params.width
    calib_h = params.calib_height if params.calib_height > 0 else params.height

    if calib_w > 0 and calib_h > 0:
        ratio_x = (params.width / calib_w) * params.input_horizontal_stretch
        ratio_y = (params.height / calib_h) * params.input_vertical_stretch
        camera_matrix[0, 0] *= ratio_x
        camera_matrix[1, 1] *= ratio_y
        camera_matrix[0, 2] *= ratio_x
        camera_matrix[1, 2] *= ratio_y

    return camera_matrix, distortion_coeffs, params.radial_distortion_limit


@dataclass
class FrameTransform:
    """Result of computing stabilization transform for a single frame.

    Attributes:
        matrices: Shape (N, 14) array. N=1 for global shutter, N=height/width
            for rolling shutter. Each row: 9 floats for 3x3 inverse transform
            matrix + 5 floats for IBIS/OIS compensation (sx, sy, angle, ox, oy).
        kernel_params: GPU compute shader parameters.
        fov: FOV scale factor used for rendering.
        minimal_fov: Minimum FOV from adaptive zoom for this frame.
        focal_length: Lens focal length in mm, if known.
    """

    matrices: NDArray[np.float32]
    kernel_params: KernelParams
    fov: float = 1.0
    minimal_fov: float = 1.0
    focal_length: Optional[float] = None
    distortion_model_name: str = "opencv_fisheye"

    @staticmethod
    def at_timestamp(
        params: ComputeParams,
        timestamp_ms: float,
        frame: int,
    ) -> FrameTransform:
        """Compute frame transform at a given timestamp.

        This is the core stabilization algorithm. Steps:
          1. Get lens parameters at this timestamp
          2. Get original and smoothed quaternions
          3. For each row (rolling shutter) compute:
             a. Quaternion at that row's exposure time
             b. Rotation difference: smoothed * org_inv * org_at_time
             c. Combine with camera intrinsics into inverse transform matrix
          4. Pack into 14-float matrix entries + KernelParams

        Args:
            params: Complete stabilization parameters.
            timestamp_ms: Frame center timestamp in milliseconds.
            frame: Frame index for FOV/IBIS lookup.

        Returns:
            FrameTransform with matrices and kernel params.
        """
        # --- 1. Keyframe parameter queries ---
        # Animated values are queried per timestamp like upstream
        # frame_transform.rs (queries return None when not keyframed —
        # fall back to the static param in that case).
        keyframes = params.keyframes
        video_rotation = params.video_rotation
        background_margin = params.background_margin
        background_feather = params.background_margin_feather
        lens_correction_amount = params.lens_correction_amount
        adaptive_zoom_center_x = params.adaptive_zoom_center_offset[0]
        adaptive_zoom_center_y = params.adaptive_zoom_center_offset[1]
        if keyframes:
            def _kf(kt: "KT", fallback: float) -> float:
                v = keyframes.value_at_video_timestamp(kt, timestamp_ms)
                return v if v is not None else fallback

            video_rotation = _kf(KT.VideoRotation, video_rotation)
            background_margin = _kf(KT.BackgroundMargin, background_margin)
            background_feather = _kf(KT.BackgroundFeather, background_feather)
            lens_correction_amount = _kf(KT.LensCorrectionStrength, lens_correction_amount)
            adaptive_zoom_center_x = _kf(KT.ZoomingCenterX, adaptive_zoom_center_x)
            adaptive_zoom_center_y = _kf(KT.ZoomingCenterY, adaptive_zoom_center_y)
        light_refraction_coefficient = params.light_refraction_coefficient

        # --- 2. Lens data ---
        camera_matrix, distortion_coeffs, radial_distortion_limit = \
            _get_lens_data_at_timestamp(params, timestamp_ms)

        # --- 3. FOV computation ---
        fov = _get_fov(params, frame, True, timestamp_ms, False)
        ui_fov = _get_fov(params, frame, True, timestamp_ms, True)

        # Apply optimal FOV adjustment if lens provides one
        # (simplified: not handling lens.optimal_fov here)

        scaled_k = camera_matrix.copy()
        new_k = _get_new_k(params, camera_matrix, fov)

        # --- 4. Rolling shutter parameters ---
        frame_readout_time = _get_frame_readout_time(
            params, timestamp_ms, can_invert=True
        )

        is_horizontal = params.frame_readout_direction.is_horizontal()
        rs_dim = params.width if is_horizontal else params.height
        row_readout_time = frame_readout_time / rs_dim if rs_dim > 0 else 0.0

        # Start of readout = center time - half readout time
        start_ts = timestamp_ms - (frame_readout_time / 2.0)

        # --- 5. Video rotation matrix ---
        rot_rad = video_rotation * (math.pi / 180.0)
        image_rotation = np.array(Rotation.from_euler("z", rot_rad).as_matrix(), dtype=np.float64)

        # --- 6. Quaternion lookups at frame center ---
        # Align video timestamp to quaternion timestamp space:
        # Quaternions may start at a non-zero offset (e.g. GoPro IMU absolute time).
        # The first quaternion key corresponds to the start of the video.
        quat_offset_us = 0.0
        if params.quaternions:
            quat_keys = sorted(params.quaternions.keys())
            if quat_keys:
                quat_offset_us = float(quat_keys[0])

        # Pre-sort keys ONCE per at_timestamp call — the rolling-shutter path
        # calls _quat_at_timestamp once per output row (up to 1080), and each
        # call previously re-sorted all keys (O(N log N) per row).
        org_keys = sorted(params.quaternions.keys()) if params.quaternions else None

        def lookup_quat(quats: dict[int, Quat64], ts_ms: float, keys: list[int] | None = None) -> Quat64:
            """Look up a quaternion at a video timestamp.

            Applies the sync-offset correction first, mirroring Gyroflow's
            ``GyroSource::quat_at_timestamp`` (``ts -= offset_at_video_timestamp(ts)``),
            so offsets set by synchronization actually shift the lookup.
            """
            corrected = ts_ms - GyroSource.offset_at_timestamp(params.sync_offsets_adjusted, ts_ms)
            return _quat_at_timestamp(quats, corrected * 1000.0 + quat_offset_us, keys)

        org_quat_center = lookup_quat(params.quaternions, timestamp_ms, org_keys).inverse()
        smoothed_quat_center = lookup_quat(params.smoothed_quaternions, timestamp_ms)

        # --- 7. Compute per-row matrices ---
        has_rolling_shutter = abs(frame_readout_time) > 0.0
        num_rows = rs_dim if has_rolling_shutter else 1

        matrices = np.zeros((num_rows, 14), dtype=np.float32)

        for y in range(num_rows):
            # Time for this row's exposure. Without rolling shutter this is
            # the frame center time, so the org lookups cancel and the
            # composite degenerates to the ABSOLUTE smoothed orientation —
            # matching upstream frame_transform.rs (quat = smoothed * org_c⁻¹
            # * org_row). The old two-factor form (smoothed * org_c⁻¹) was a
            # porting bug: it stabilized by the org-relative delta and the
            # output kept following the original shake.
            quat_time = timestamp_ms if not has_rolling_shutter else start_ts + row_readout_time * y
            org_at_time = lookup_quat(params.quaternions, quat_time, org_keys)
            quat = smoothed_quat_center * org_quat_center * org_at_time


            # Quaternion -> 3x3 rotation matrix
            r = image_rotation @ quat.to_rotation_matrix()

            # Coordinate system adaptation:
            # Flip Y-related elements to convert from camera coords (Y down)
            # to graphics coords (Y up)
            if params.framebuffer_inverted:
                r[0, 2] *= -1.0
                r[1, 2] *= -1.0
                r[2, 0] *= -1.0
                r[2, 1] *= -1.0
            else:
                r[0, 1] *= -1.0
                r[0, 2] *= -1.0
                r[1, 0] *= -1.0
                r[2, 0] *= -1.0

            # Suppress rotation mode: keep only translation
            if params.suppress_rotation:
                r = np.eye(3, dtype=np.float64)
                if not has_rolling_shutter:
                    # Disable IBIS too when no rolling shutter
                    matrices[y, :] = 0.0
                    matrices[y, 0] = 1.0
                    matrices[y, 4] = 1.0
                    matrices[y, 8] = 1.0
                    continue

            # Compute inverse transform: (new_k * R)^-1
            combined = new_k @ r
            try:
                i_r = np.linalg.inv(combined)
            except np.linalg.LinAlgError:
                i_r = np.zeros((3, 3), dtype=np.float64)

            # Pack: 9 matrix entries + 5 IBIS zeros
            matrices[y, 0:3] = i_r[0, :].astype(np.float32)
            matrices[y, 3:6] = i_r[1, :].astype(np.float32)
            matrices[y, 6:9] = i_r[2, :].astype(np.float32)
            # IBIS/OIS entries [9:14] remain zero (no sensor stabilization data)

        # --- 8. Build KernelParams ---
        kernel_params = KernelParams()

        kernel_params.width = params.width
        kernel_params.height = params.height
        kernel_params.stride = params.width * 3  # Assume 3 channels for now
        kernel_params.output_width = params.output_width
        kernel_params.output_height = params.output_height
        kernel_params.matrix_count = num_rows
        kernel_params.interpolation = 2  # Bilinear

        kernel_params.background_mode = int(params.background_mode)
        # HORIZONTAL_RS flag (bit 4, value 16): rolling-shutter direction is
        # horizontal (matrices indexed by source column instead of row).
        # Mirrors upstream kernel_flags.set(HORIZONTAL_RS, direction.is_horizontal()).
        kernel_params.flags = 16 if is_horizontal else 0
        kernel_params.bytes_per_pixel = 3  # Will be set properly by caller
        kernel_params.pix_element_count = 3

        # Background color
        bg = params.background
        kernel_params.background = (ctypes.c_float * 4)(
            float(bg[0]), float(bg[1]), float(bg[2]), float(bg[3])
        )

        # Focal length and center from scaled_k
        kernel_params.f = (ctypes.c_float * 2)(
            float(scaled_k[0, 0]), float(scaled_k[1, 1])
        )
        kernel_params.c = (ctypes.c_float * 2)(
            float(scaled_k[0, 2]), float(scaled_k[1, 2])
        )

        # Distortion coefficients (pack 12 into 3 vec4s)
        k_floats = [float(x) for x in distortion_coeffs]
        while len(k_floats) < 12:
            k_floats.append(0.0)
        kernel_params.k1 = (ctypes.c_float * 4)(k_floats[0], k_floats[1], k_floats[2], k_floats[3])
        kernel_params.k2 = (ctypes.c_float * 4)(k_floats[4], k_floats[5], k_floats[6], k_floats[7])
        kernel_params.k3 = (ctypes.c_float * 4)(k_floats[8], k_floats[9], k_floats[10], k_floats[11])

        kernel_params.fov = float(fov)
        kernel_params.r_limit = float(radial_distortion_limit)
        kernel_params.lens_correction_amount = float(lens_correction_amount)
        kernel_params.input_vertical_stretch = float(params.input_vertical_stretch)
        kernel_params.input_horizontal_stretch = float(params.input_horizontal_stretch)
        kernel_params.background_margin = float(background_margin)
        kernel_params.background_margin_feather = float(background_feather)

        # 2D translation from adaptive zoom center
        fov_f = float(fov)
        kernel_params.translation2d = (ctypes.c_float * 2)(
            float(adaptive_zoom_center_x * params.width / fov_f),
            float(adaptive_zoom_center_y * params.height / fov_f),
        )
        kernel_params.translation3d = (ctypes.c_float * 4)(0.0, 0.0, 0.0, 0.0)

        kernel_params.digital_lens_params = (ctypes.c_float * 4)(0.0, 0.0, 0.0, 0.0)
        kernel_params.light_refraction_coefficient = float(light_refraction_coefficient)

        # Source/output rectangles (full image by default)
        kernel_params.source_rect = (ctypes.c_int32 * 4)(0, 0, params.width, params.height)
        kernel_params.output_rect = (ctypes.c_int32 * 4)(0, 0, params.output_width, params.output_height)

        # Safe-area rect must cover the full frame: the shader's
        # draw_safe_area dims pixels OUTSIDE this rect by 0.5x (preview
        # overlay). The (0,0,0,0) default left only pixel (0,0) "safe",
        # halving every other pixel of the GPU output.
        kernel_params.safe_area_rect = (
            ctypes.c_float * 4
        )(0.0, 0.0, float(params.output_width), float(params.output_height))

        return FrameTransform(
            matrices=matrices,
            kernel_params=kernel_params,
            fov=ui_fov,
            minimal_fov=params.minimal_fovs[frame] if frame < len(params.minimal_fovs) else 1.0,
            focal_length=params.focal_length,
            distortion_model_name=params.distortion_model_name,
        )
