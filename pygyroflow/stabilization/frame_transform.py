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

from pygyroflow.types.enums import BackgroundMode, Interpolation, ReadoutDirection
from pygyroflow.types.kernel_params import KernelParams
from pygyroflow.types.quaternion import Quat64
from pygyroflow.gyro_source.source import GyroSource
from pygyroflow.gyro_source.splines import as_catmull_rom
from pygyroflow.util import frame_at_timestamp, map_coord
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

    # A sensor that reads out only part of its height finishes the frame
    # proportionally sooner (upstream frame_transform.rs).
    if params.lens_params:
        entry = params.lens_params.get_closest(
            round(timestamp_ms * 1000.0), LENS_LOOKUP_MAX_DIFF_US
        )
        if (
            entry is not None
            and entry.capture_area_size is not None
            and entry.sensor_size_px is not None
            and entry.sensor_size_px[1]
        ):
            frt *= entry.capture_area_size[1] / entry.sensor_size_px[1]

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


def _focal_length_fov_compensation(params: ComputeParams, frame: int) -> float:
    """Digital zoom-out factor that tracks the smoothed focal length.

    Port of upstream ``FrameTransform::focal_length_fov_compensation``.

    Returns ``dequantized / smoothed``. When the true optical focal length is
    longer than the smoothed target, the ratio is above 1 and the render zooms
    out to compensate, so the apparent zoom follows the smoothed curve instead
    of the raw metadata. The factor goes into ``fov`` only — ``scaled_k`` keeps
    the raw pixel focal length — so it lands on ``new_k`` but not on the
    forward projection, which is what makes it a visible digital zoom.

    Both curves are frame-indexed, so no timestamp lookup is involved and the
    two stay aligned by construction.
    """
    if not params.focal_length_smoothing_enabled:
        return 1.0
    if frame >= len(params.focal_lengths) or frame >= len(params.smoothed_focal_lengths):
        return 1.0
    dequantized = params.focal_lengths[frame]
    smoothed = params.smoothed_focal_lengths[frame]
    if dequantized is None or smoothed is None:
        return 1.0
    if dequantized > 0.0 and smoothed > 0.0:
        return dequantized / smoothed
    return 1.0


# The per-frame lens maps are consulted within this window; a frame further
# than that from any entry uses the profile as-is (upstream's 100000, µs).
LENS_LOOKUP_MAX_DIFF_US = 100_000


def _get_lens_data_at_timestamp(
    params: ComputeParams,
    timestamp_ms: float,
    invert_asym_lens: bool = False,
) -> tuple[NDArray[np.float64], list[float], float, float, float, float | None]:
    """Camera intrinsics and distortion coefficients at a timestamp.

    Port of ``FrameTransform::get_lens_data_at_timestamp`` (frame_transform.rs).
    Two independent per-frame channels can override the static profile:

    ``lens_positions``
        A scalar per time (a focal length in mm, or a Sony crop score) that
        selects an entry from the profile's own ``interpolations`` table,
        interpolating between two of them. This is the zoom-lens path.
    ``lens_params``
        Raw per-frame intrinsics that override the camera matrix and
        distortion coefficients outright. The pixel focal length comes either
        from the file directly, or from mm / (pixel pitch × capture area
        height) × video height. Guarded by ``distortion_coeffs.len() < 4``:
        a profile that carries its own coefficients (Canon/Sony per-frame
        ones do) wins over the file's.

    When ``lens_params`` supplies a pixel focal length, the calibration-
    resolution scaling below is **skipped** (``stretch_lens = false``) and the
    principal point is reset to the frame centre: the override is already in
    video pixels, so scaling it again would be wrong and the calibration's
    off-centre principal point no longer applies.

    Args:
        params: Compute parameters.
        timestamp_ms: Frame timestamp in ms.
        invert_asym_lens: Flip the vertical principal point of an asymmetric
            profile (upstream's ``invert_asym_lens``).

    Returns:
        ``(camera_matrix, distortion_coeffs, radial_distortion_limit,
        input_horizontal_stretch, input_vertical_stretch, focal_length)``.
    """
    ts_us = round(timestamp_ms * 1000.0)
    lens = params.lens

    interpolated = None
    if params.lens_positions and lens is not None:
        position = params.lens_positions.get_closest(ts_us, LENS_LOOKUP_MAX_DIFF_US)
        if position is not None:
            interpolated = lens.get_interpolated_profile_at(float(position))

    if interpolated is not None:
        source = interpolated
        camera_matrix = source.get_camera_matrix(
            (params.width, params.height), invert_asym_lens
        )
        distortion_coeffs = source.get_distortion_coeffs()
        radial_distortion_limit = float(source.radial_distortion_limit or 0.0)
        focal_length = source.focal_length
        # The *unpadded* count: `get_distortion_coeffs()` zero-pads to 12, and
        # the gate below is upstream's `lens.fisheye_params.distortion_coeffs
        # .len() < 4`, which counts what the profile actually carries.
        coeff_count = len(source.distortion_coeffs)
        calib_w = source.calib_dimension["w"] or params.width
        calib_h = source.calib_dimension["h"] or params.height
        h_stretch = source.input_horizontal_stretch
        v_stretch = source.input_vertical_stretch
    else:
        camera_matrix = params.camera_matrix.copy()
        distortion_coeffs = list(params.distortion_coeffs)
        radial_distortion_limit = params.radial_distortion_limit
        focal_length = params.focal_length
        coeff_count = (
            len(params.lens.distortion_coeffs)
            if params.lens is not None
            else len(params.distortion_coeffs)
        )
        calib_w = params.calib_width if params.calib_width > 0 else params.width
        calib_h = params.calib_height if params.calib_height > 0 else params.height
        h_stretch = params.input_horizontal_stretch
        v_stretch = params.input_vertical_stretch

    h_stretch = h_stretch if h_stretch > 0.01 else 1.0
    v_stretch = v_stretch if v_stretch > 0.01 else 1.0

    stretch_lens = True
    if params.lens_params and coeff_count < 4:
        entry = params.lens_params.get_closest(ts_us, LENS_LOOKUP_MAX_DIFF_US)
        if entry is not None:
            pixel_focal_length = entry.pixel_focal_length
            if pixel_focal_length is None and entry.focal_length is not None:
                focal_length = float(entry.focal_length)
                if entry.pixel_pitch is not None and entry.capture_area_size is not None:
                    pitch_mm = entry.pixel_pitch[1] / 1_000_000.0
                    pixel_focal_length = (
                        entry.focal_length
                        / (pitch_mm * entry.capture_area_size[1])
                        * params.height
                    )
            if pixel_focal_length is not None:
                camera_matrix[0, 0] = pixel_focal_length
                camera_matrix[1, 1] = pixel_focal_length
                camera_matrix[0, 2] = params.width / 2.0
                camera_matrix[1, 2] = params.height / 2.0
                stretch_lens = False
                if entry.focal_length is not None:
                    focal_length = float(entry.focal_length)

            if 0 < len(entry.distortion_coefficients) <= 12:
                for index, value in enumerate(entry.distortion_coefficients):
                    distortion_coeffs[index] = float(value)
                radial_distortion_limit = _radial_limit_for(
                    params.distortion_model_name, distortion_coeffs
                )

    if stretch_lens:
        # Scale camera matrix from calibration resolution to video resolution.
        if calib_w > 0 and calib_h > 0:
            ratio_x = (params.width / calib_w) * h_stretch
            ratio_y = (params.height / calib_h) * v_stretch
            camera_matrix[0, 0] *= ratio_x
            camera_matrix[1, 1] *= ratio_y
            camera_matrix[0, 2] *= ratio_x
            camera_matrix[1, 2] *= ratio_y

    if params.digital_zoom:
        camera_matrix[0, 0] *= params.digital_zoom
        camera_matrix[1, 1] *= params.digital_zoom

    return (
        camera_matrix,
        distortion_coeffs,
        radial_distortion_limit,
        h_stretch,
        v_stretch,
        focal_length,
    )


def _radial_limit_for(model_name: str, distortion_coeffs: list[float]) -> float:
    """Radial distortion limit for an overridden coefficient set.

    The models return None when the distortion is valid over the whole field
    of view, which counts as no limit.
    """
    from pygyroflow.stabilization.distortion_models import from_name

    try:
        result = from_name(model_name or "opencv_fisheye").radial_distortion_limit(
            distortion_coeffs
        )
    except Exception:
        return 0.0
    return 0.0 if result is None else float(result)


def _points_field(source, name, default=None):
    """Read a field from a decoded dict or from a dataclass.

    The mesh and IBIS records arrive one of two ways: built by the telemetry
    parser, or decoded straight out of a project file's CBOR, where this port
    keeps them as raw dicts because it does not model every nested struct.
    """
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _sample_spline(spline, position: float) -> np.ndarray:
    """A curve sample, or zeros outside the control points.

    Zero is what upstream's ``interpolate(...).unwrap_or_default()`` gives,
    and it is the right answer: no curve means no displacement to apply.
    """
    if spline is None:
        return np.zeros(3)
    value = spline.interpolate(position)
    if value is None:
        return np.zeros(3)
    return np.asarray(value, dtype=np.float64)


def _shift_per_point(params: ComputeParams, points, frame: int):
    """IBIS and OIS displacement for each point, from the gyro's stab data.

    Port of the ``shifts`` half of ``at_timestamp_for_points``. The splines are
    expressed in the sensor's own coordinates for the crop area the camera
    reported, so the row is remapped into that crop before the lookup and the
    result is scaled back into image pixels.

    Returns None when the frame has no stabilization data — upstream then
    leaves the shift at zero, which is also what ``unwrap_or_default()`` on a
    spline outside its range gives.
    """
    if not params.camera_stab_data:
        return None
    if frame >= len(params.camera_stab_data):
        return None
    stab = params.camera_stab_data[frame]
    if stab is None:
        return None

    crop_area = _points_field(stab, "crop_area")
    pixel_pitch = _points_field(stab, "pixel_pitch")
    if not crop_area or not pixel_pitch:
        return None

    ibis = as_catmull_rom(_points_field(stab, "ibis_spline"))
    ois = as_catmull_rom(_points_field(stab, "ois_spline"))
    if ibis is None and ois is None:
        return None
    offset = float(_points_field(stab, "offset", 0.0) or 0.0)

    is_scale = (
        params.width / float(crop_area[2]) / float(pixel_pitch[0]),
        params.height / float(crop_area[3]) / float(pixel_pitch[1]),
    )
    shifts = []
    for point in points:
        sensor_y = map_coord(
            float(point[1]), 0.0, float(params.height),
            float(crop_area[1]), float(crop_area[1]) + float(crop_area[3]),
        )
        s = _sample_spline(ibis, sensor_y + offset)
        o = _sample_spline(ois, sensor_y + offset)
        shifts.append((
            float(s[0] * is_scale[0]),
            float(s[1] * is_scale[1]),
            float(np.deg2rad(s[2] / 1000.0)),
            float(o[0] * is_scale[0]),
            float(o[1] * is_scale[1]),
        ))
    return shifts


def at_timestamp_for_points(
    params: ComputeParams,
    points,
    timestamp_ms: float,
    frame: int | None = None,
    use_fovs: bool = True,
):
    """Per-point transforms, for callers that sample specific pixels.

    Port of ``FrameTransform::at_timestamp_for_points``. The image path
    (:meth:`FrameTransform.at_timestamp`) computes one matrix per *row*; this
    computes one per *point*, at that point's own exposure instant, and adds
    the IBIS/OIS displacement and mesh correction that only make sense
    per-pixel.

    Returns ``(scaled_k, distortion_coeffs, new_k, rotations, shifts, mesh)``:
    the intrinsic matrix the points were captured through, the coefficients,
    the output matrix, and then three per-point extras that can each be None.
    """
    video_rotation = params.video_rotation
    if params.keyframes:
        value = params.keyframes.value_at_video_timestamp(KT.VideoRotation, timestamp_ms)
        if value is not None:
            video_rotation = value

    if frame is None:
        frame = frame_at_timestamp(timestamp_ms, params.scaled_fps)

    # The stretch values this returns are deliberately dropped: the points
    # path uses the lens's static ones below, exactly as upstream does.
    (camera_matrix, distortion_coeffs, _, _, _, _) = _get_lens_data_at_timestamp(
        params, timestamp_ms, params.framebuffer_inverted
    )

    fov = _get_fov(params, frame, use_fovs, timestamp_ms, False) * (
        _focal_length_fov_compensation(params, frame)
    )
    new_k = _get_new_k(params, camera_matrix, fov)

    mesh = None
    if params.mesh_correction and frame < len(params.mesh_correction):
        entry = params.mesh_correction[frame]
        # The first element is the *distorting* mesh — the one that maps the
        # ideal grid onto what the sensor actually recorded. A project file
        # decodes this as a two-element list; `mesh_correction` is not modelled
        # beyond that, so a list is the only shape that reaches here.
        if isinstance(entry, (list, tuple)) and entry:
            mesh = entry[0]

    frame_readout_time = _get_frame_readout_time(params, timestamp_ms)
    is_horizontal = params.frame_readout_direction.is_horizontal()
    rs_dim = params.width if is_horizontal else params.height
    row_readout_time = frame_readout_time / rs_dim if rs_dim > 0 else 0.0

    if params.per_frame_time_offsets and 0 <= frame < len(params.per_frame_time_offsets):
        timestamp_ms = timestamp_ms + params.per_frame_time_offsets[frame]

    start_ts = timestamp_ms - (frame_readout_time / 2.0)
    image_rotation = np.array(
        Rotation.from_euler("z", video_rotation * (math.pi / 180.0)).as_matrix(),
        dtype=np.float64,
    )

    org_keys = sorted(params.quaternions.keys()) if params.quaternions else None

    def lookup_quat(quats, ts_ms, keys=None):
        corrected = ts_ms - GyroSource.offset_at_timestamp(
            params.sync_offsets_adjusted, ts_ms
        )
        return _quat_at_timestamp(quats, corrected * 1000.0, keys)

    org_quat_inv = lookup_quat(params.quaternions, timestamp_ms, org_keys).inverse()
    smoothed_quat = lookup_quat(params.smoothed_quaternions, timestamp_ms)

    # One matrix per point when there is rolling shutter, one for all of them
    # otherwise — the point's row only matters if rows are exposed at
    # different times.
    points_iter = list(points) if abs(frame_readout_time) > 0.0 else [(0.0, 0.0)]
    rotations = []
    for point in points_iter:
        if abs(frame_readout_time) > 0.0:
            row = float(point[0]) if is_horizontal else float(point[1])
            quat_time = start_ts + row_readout_time * row
        else:
            quat_time = start_ts
        quat = smoothed_quat * org_quat_inv * lookup_quat(
            params.quaternions, quat_time, org_keys
        )
        r = image_rotation @ quat.to_rotation_matrix()
        r[0, 1] *= -1.0
        r[0, 2] *= -1.0
        r[1, 0] *= -1.0
        r[2, 0] *= -1.0
        if params.suppress_rotation:
            r = np.eye(3, dtype=np.float64)
        rotations.append(new_k @ r)

    shifts = _shift_per_point(params, points_iter, frame)
    if params.suppress_rotation and params.frame_readout_time == 0.0:
        shifts = None

    return camera_matrix, distortion_coeffs, new_k, rotations, shifts, mesh


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
        (
            camera_matrix,
            distortion_coeffs,
            radial_distortion_limit,
            input_horizontal_stretch,
            input_vertical_stretch,
            focal_length,
        ) = _get_lens_data_at_timestamp(params, timestamp_ms)

        # --- 3. FOV computation ---
        # The focal length compensation lands on the render fov only, and
        # before the optimal_fov block below — upstream order (focal length
        # smoothing, then the sharpest-FOV adjustment).
        fov = _get_fov(params, frame, True, timestamp_ms, False) * (
            _focal_length_fov_compensation(params, frame)
        )
        ui_fov = _get_fov(params, frame, True, timestamp_ms, True)

        # Upstream frame_transform.rs: a lens profile may carry the FOV it is
        # sharpest at. With per-frame fovs present the *UI* fov is divided by
        # it (the render fov already went through StabilizationParams.set_fovs
        # with the same factor); without them the render fov is scaled.
        if params.optimal_fov:
            if params.fovs:
                ui_fov /= params.optimal_fov
            else:
                fov *= params.optimal_fov

        scaled_k = camera_matrix.copy()
        new_k = _get_new_k(params, camera_matrix, fov)

        # --- 4. Rolling shutter parameters ---
        frame_readout_time = _get_frame_readout_time(
            params, timestamp_ms, can_invert=True
        )

        is_horizontal = params.frame_readout_direction.is_horizontal()
        rs_dim = params.width if is_horizontal else params.height
        row_readout_time = frame_readout_time / rs_dim if rs_dim > 0 else 0.0

        # Per-frame timestamp correction, applied from here on (upstream
        # shadows its `timestamp_ms` at exactly this point). Files that
        # provide it — Sony RTMD, some DJI/RED streams — carry a few ms of
        # per-frame jitter that otherwise lands straight on the gyro lookup.
        if params.per_frame_time_offsets:
            if 0 <= frame < len(params.per_frame_time_offsets):
                timestamp_ms = timestamp_ms + params.per_frame_time_offsets[frame]

        # Start of readout = center time - half readout time
        start_ts = timestamp_ms - (frame_readout_time / 2.0)

        # --- 5. Video rotation matrix ---
        rot_rad = video_rotation * (math.pi / 180.0)
        image_rotation = np.array(Rotation.from_euler("z", rot_rad).as_matrix(), dtype=np.float64)

        # --- 6. Quaternion lookups at frame center ---
        # Upstream GyroSource::quat_at_timestamp maps the video timestamp
        # DIRECTLY onto the quaternion key space (us), only clamping to
        # [first_key, last_key] — no first-key offset is added. The previous
        # `+ quat_keys[0]` shift misaligned every lookup by the stream's
        # lead-in (-5.414 ms on the DJI clip): a constant time error whose
        # residual is delta * domega/dt, ~3 deg/s of yaw jitter at walking
        # frequencies — exactly the excess measured against the official
        # export. Removed to match upstream.

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
            return _quat_at_timestamp(quats, corrected * 1000.0, keys)

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
        # Upstream's numbering, not the CPU path's: 8 is Lanczos4, which is
        # upstream's default. This used to be 2 with a "Bilinear" comment —
        # 2 means Bilinear *to the shader* (it is the tap count there), so
        # the GPU silently rendered every frame at the lowest-quality kernel
        # while the CPU path read the same 2 as Lanczos4.
        kernel_params.interpolation = int(Interpolation.Lanczos4)

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
        # From the per-timestamp lens, not the static params: a zoom lens'
        # stretch comes from whichever profile is current for this frame.
        kernel_params.input_vertical_stretch = float(input_vertical_stretch)
        kernel_params.input_horizontal_stretch = float(input_horizontal_stretch)
        kernel_params.background_margin = float(background_margin)
        kernel_params.background_margin_feather = float(background_feather)

        # 2D translation from adaptive zoom center. With an inverted
        # framebuffer (bottom-up row order) the vertical centre has to flip;
        # upstream does this right before packing translation2d
        # (frame_transform.rs), and the CPU and GPU samplers both rely on it.
        if params.framebuffer_inverted:
            adaptive_zoom_center_y = -adaptive_zoom_center_y

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

        # Report the smoothed focal length so the "Focal length: X mm" readout
        # tracks the curve the viewer actually sees, not the raw metadata.
        reported_focal_length = focal_length
        if params.focal_length_smoothing_enabled and frame < len(
            params.smoothed_focal_lengths
        ):
            smoothed_fl = params.smoothed_focal_lengths[frame]
            if smoothed_fl is not None:
                reported_focal_length = smoothed_fl

        return FrameTransform(
            matrices=matrices,
            kernel_params=kernel_params,
            fov=ui_fov,
            minimal_fov=params.minimal_fovs[frame] if frame < len(params.minimal_fovs) else 1.0,
            focal_length=reported_focal_length,
            distortion_model_name=params.distortion_model_name,
        )
