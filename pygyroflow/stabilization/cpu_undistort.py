"""CPU-based image undistortion and stabilization.

Port of Gyroflow's cpu_undistort.rs. Implements inverse mapping from output
pixels to input pixels via the rotation+distortion transform, with bilinear
interpolation sampling. Vectorized NumPy implementation.

Semantics mirrored from upstream ``undistort_coord`` + ``rotate_and_distort``:
  - Distortion is applied through the lens's distortion model (all 10 models
    supported via ``distortion_models.from_name``), not a hardcoded fisheye.
  - Rolling shutter: per-pixel matrix selection. A trial mapping with the
    center matrix estimates the source row (or column for horizontal RS),
    and each pixel then uses the matrix of its estimated source line —
    previously only the center matrix was used, discarding the per-row
    matrices computed by FrameTransform.
  - ``radial_distortion_limit``: points beyond the valid distortion range
    fall back to the background fill instead of sampling extrapolated garbage.
  - ``lens_correction_amount`` < 1 blends between corrected and uncorrected
    output positions.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from numpy.typing import NDArray

from pygyroflow.stabilization.distortion_models import from_name as model_from_name
from pygyroflow.stabilization.ewa import EWA_FILTERS, ewa_sample, map_jacobian
from pygyroflow.stabilization.frame_transform import FrameTransform
from pygyroflow.types.kernel_params import KernelParams

# The interpolation indices THIS module (and the CLI/GUI options that feed
# it) uses are zero-based: 0=Bilinear, 1=Bicubic, 2=Lanczos4, 3-6=EWA.
# Upstream Gyroflow — and therefore ``KernelParams.interpolation``, which is
# uploaded to the WGSL shader — numbers the same filters 2/4/8/10-13, where
# the value is also the kernel's tap count. The two collide on "2", so a
# value going to the GPU has to be translated rather than passed through.
CPU_TO_UPSTREAM_INTERPOLATION: dict[int, int] = {
    0: 2,    # Bilinear
    1: 4,    # Bicubic
    2: 8,    # Lanczos4
    3: 10,   # EWA RobidouxSharp
    4: 11,   # EWA Robidoux
    5: 12,   # EWA Mitchell
    6: 13,   # EWA Catmull-Rom
}

# KernelParamsFlags::HORIZONTAL_RS (1 << 4)
_HORIZONTAL_RS_FLAG = 16

_log = logging.getLogger(__name__)
_ewa_warned = False


def _warn_ewa_cost_once() -> None:
    """EWA is a per-tap gather in NumPy — say so before someone renders 4K.

    Measured on this machine (identity map with a 4% zoom-out, kernel radius
    3): 16-23 s per 1080p frame and 49-154 s per 4K frame, against ~2 s for
    Lanczos4 *including* the map build.  It is correct, but the CPU path is
    not where it belongs — upstream runs EWA in a fragment shader.
    """
    global _ewa_warned
    if not _ewa_warned:
        _ewa_warned = True
        _log.warning(
            "EWA interpolation on the CPU path is a NumPy per-tap gather and costs "
            "roughly an order of magnitude more than Lanczos4 (measured: 16-23 s per "
            "1080p frame, 49-154 s per 4K frame). Use it for stills or verification; "
            "render with lanczos4."
        )

# Cached output coordinate grids, keyed by (height, width). np.mgrid for a
# 1080p grid costs ~60ms; building it per frame is pure waste.
_GRID_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _coordinate_grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Cached float32 output grid.

    float32 throughout: the sampling map fed to cv2.remap is float32
    anyway, so float64 intermediates only double memory traffic (~1e-3 px
    coordinate error, invisible after bilinear sampling).
    """
    grid = _GRID_CACHE.get((h, w))
    if grid is None:
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        grid = (ys, xs)
        if len(_GRID_CACHE) > 8:
            _GRID_CACHE.clear()
        _GRID_CACHE[(h, w)] = grid
    # Copy: callers add per-frame translation in place.
    return grid[0].copy(), grid[1].copy()


def _apply_matrix(
    xs: NDArray,
    ys: NDArray,
    m: NDArray,
    kp: KernelParams,
) -> tuple[NDArray, NDArray, NDArray]:
    """Apply the 9-entry inverse transform matrix + 3D translation.

    ``m`` is a (..., 9+) array whose first 9 entries are the row-major 3x3
    matrix (shared scalar entries broadcast naturally).
    """
    tx3d = float(kp.translation3d[0])
    ty3d = float(kp.translation3d[1])
    tz3d = float(kp.translation3d[2])

    _x = xs * m[..., 0] + ys * m[..., 1] + m[..., 2] + tx3d
    _y = xs * m[..., 3] + ys * m[..., 4] + m[..., 5] + ty3d
    _w = xs * m[..., 6] + ys * m[..., 7] + m[..., 8] + tz3d
    return _x, _y, _w


def _rotate_and_distort(
    xs: NDArray,
    ys: NDArray,
    m: NDArray,
    kp: KernelParams,
    model,
    ibis_active: bool | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Vectorized port of upstream ``Stabilization::rotate_and_distort``.

    Args:
        xs, ys: Output pixel coordinates.
        m: Array with >= 14 matrix entries (shared or per-pixel).
        kp: Kernel parameters.
        model: Distortion model (``DistortionModelBase``).
        ibis_active: Precomputed "any IBIS entries set" flag. When None it
            is derived from ``m`` (O(m.size) — pass False explicitly for
            the per-pixel rolling-shutter case, derived once from the small
            (N, 14) matrix stack instead of the gathered per-pixel array).

    Returns:
        (src_x, src_y, valid) — source pixel coordinates and validity mask.
    """
    _x, _y, _w = _apply_matrix(xs, ys, m, kp)

    valid = _w > 0.0

    # Radial distortion limit: outside the valid range the model extrapolates
    # garbage — treat those pixels as invalid (background).
    r_limit_sq = float(kp.r_limit) * float(kp.r_limit)
    if r_limit_sq > 0.0:
        valid &= ~((_x * _x + _y * _y) > r_limit_sq * _w)

    # Light refraction correction
    lrc = kp.light_refraction_coefficient
    if lrc != 1.0 and lrc > 0.0:
        r_sq = _x * _x + _y * _y
        r = np.sqrt(r_sq) / np.where(_w != 0.0, _w, 1.0)
        mask = r > 0.0
        sin_theta = np.where(mask, r / np.sqrt(1.0 + r * r), 0.0)
        sin_theta_d = sin_theta * lrc
        ok = sin_theta_d < 1.0
        r_d = np.where(ok & mask, sin_theta_d / np.sqrt(np.maximum(1.0 - sin_theta_d * sin_theta_d, 1e-12)), r)
        scale_w = np.where((ok & mask) & (r_d != 0.0), r / np.where(r_d != 0.0, r_d, 1.0), 1.0)
        _w = _w * scale_w

    # Distortion via the lens's model (perspective divide happens inside).
    # Invalid pixels get a safe w to avoid 0-division warnings; they are
    # masked out below regardless.
    safe_w = np.where(valid & (_w != 0.0), _w, 1.0)
    xd, yd = model.distort_points(_x, _y, safe_w, kp)

    fx = float(kp.f[0])
    fy = float(kp.f[1])
    cx = float(kp.c[0])
    cy = float(kp.c[1])

    u = xd * fx
    v = yd * fy

    # IBIS/OIS compensation (entries [9:14] of the matrix row). Upstream
    # applies it on the focal-scaled coords before adding the principal
    # point, with m12/m13 as extra offsets. The active flag comes from the
    # small matrix stack (or the array itself for scalar-matrix callers).
    if ibis_active is None:
        ibis_active = bool(np.any(m[..., 9:14] != 0.0))
    if ibis_active:
        has_ibis = (
            (m[..., 9] != 0.0)
            | (m[..., 10] != 0.0)
            | (m[..., 11] != 0.0)
            | (m[..., 12] != 0.0)
            | (m[..., 13] != 0.0)
        )
        ang_rad = m[..., 11]
        cos_a = np.cos(-ang_rad)
        sin_a = np.sin(-ang_rad)
        u_adj = cos_a * u - sin_a * v - m[..., 9] + m[..., 12]
        v_adj = sin_a * u + cos_a * v - m[..., 10] + m[..., 13]
        u = np.where(has_ibis, u_adj, u)
        v = np.where(has_ibis, v_adj, v)

    u = u + cx
    v = v + cy

    # Input stretch
    hs = kp.input_horizontal_stretch
    vs = kp.input_vertical_stretch
    if hs > 0.001:
        u = u / hs
    if vs > 0.001:
        v = v / vs

    return u, v, valid


def _bilinear_sample_vectorized(
    frame: NDArray[np.float32],
    src_x: NDArray,
    src_y: NDArray,
    valid: NDArray[np.bool_],
    bg: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Vectorized bilinear sampling for an entire frame."""
    h, w = frame.shape[0], frame.shape[1]
    channels = frame.shape[2] if frame.ndim == 3 else 1

    flat_x = src_x.ravel()
    flat_y = src_y.ravel()
    flat_valid = valid.ravel()

    x0 = np.floor(flat_x).astype(np.int32)
    y0 = np.floor(flat_y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    fx = (flat_x - x0).astype(np.float32)
    fy = (flat_y - y0).astype(np.float32)

    # Clamp coordinates
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)

    # Bounds check (x1/y1 clamp to the edge, so the last row/column is
    # sampled with edge replication like upstream's coefficient-table sampler)
    in_bounds = (flat_x >= 0) & (flat_y >= 0) & (flat_x < w) & (flat_y < h) & flat_valid

    w00 = ((1.0 - fx) * (1.0 - fy))[:, np.newaxis]
    w10 = (fx * (1.0 - fy))[:, np.newaxis]
    w01 = ((1.0 - fx) * fy)[:, np.newaxis]
    w11 = (fx * fy)[:, np.newaxis]

    if channels > 1:
        # p_ij = pixel at (x0+i, y0+j): p10 is the +x neighbor, p01 the +y
        # neighbor — matching the w10 = fx*(1-fy) / w01 = (1-fx)*fy weights.
        p00 = frame[y0c, x0c].astype(np.float32)
        p10 = frame[y0c, x1c].astype(np.float32)
        p01 = frame[y1c, x0c].astype(np.float32)
        p11 = frame[y1c, x1c].astype(np.float32)
    else:
        p00 = frame[y0c, x0c].astype(np.float32)[:, np.newaxis]
        p10 = frame[y0c, x1c].astype(np.float32)[:, np.newaxis]
        p01 = frame[y1c, x0c].astype(np.float32)[:, np.newaxis]
        p11 = frame[y1c, x1c].astype(np.float32)[:, np.newaxis]

    result = w00 * p00 + w10 * p10 + w01 * p01 + w11 * p11

    # Apply background for out-of-bounds
    bg_row = bg[np.newaxis, :]
    result[~in_bounds] = bg_row

    return result


def cpu_undistort(
    frame: NDArray,
    transform: FrameTransform,
    interpolation: int = 2,
) -> NDArray:
    """Apply stabilization transform on CPU using vectorized NumPy.

    Args:
        frame: Input frame, shape (H, W, C), dtype uint8 or float32.
        transform: FrameTransform with matrices and kernel_params.
        interpolation: Upstream Gyroflow interpolation index —
            0 = Bilinear, 1 = Bicubic, 2 = Lanczos4 (upstream default),
            3-6 = EWA (RobidouxSharp/Robidoux/Mitchell/Catmull-Rom), which
            is implemented here rather than approximated with Lanczos4.

    Returns:
        Stabilized output frame, shape (output_H, output_W, C).
    """
    kp = transform.kernel_params
    matrices = np.asarray(transform.matrices, dtype=np.float64)
    model = model_from_name(transform.distortion_model_name)

    out_w = kp.output_width
    out_h = kp.output_height
    in_w = kp.width
    in_h = kp.height

    if out_w <= 0 or out_h <= 0:
        return np.array([], dtype=frame.dtype)

    channels = frame.shape[2] if frame.ndim == 3 else 1

    if frame.dtype == np.uint8:
        max_val = 255.0
    elif frame.dtype == np.uint16:
        max_val = 65535.0
    else:
        max_val = 0.0

    bg = np.array([kp.background[0], kp.background[1], kp.background[2], kp.background[3]],
                  dtype=np.float32)
    if max_val > 0:
        bg *= max_val
    bg_out = bg[:channels]

    # Output coordinate grid, offset by the 2D translation (adaptive zoom
    # center), mirroring upstream undistort_coord. The plain grid is cached
    # per size — it is identical for every frame.
    ys, xs = _coordinate_grid(out_h, out_w)
    xs = xs + float(kp.translation2d[0])
    ys = ys + float(kp.translation2d[1])

    # Lens correction blend: mix between fully corrected and uncorrected
    # output positions when lens_correction_amount < 1.
    lca = float(kp.lens_correction_amount)
    if lca < 1.0:
        factor = max(1.0 - lca, 0.001)
        fov_val = float(kp.fov) if kp.fov > 0.0 else 1.0
        out_cx = out_w / 2.0
        out_cy = out_h / 2.0
        out_fx = float(kp.f[0]) / fov_val / factor
        out_fy = float(kp.f[1]) / fov_val / factor
        nx = (xs - out_cx) / out_fx
        ny = (ys - out_cy) / out_fy
        ux, uy = model.undistort_points(nx, ny, kp)
        ok = ~(np.isnan(ux) | np.isnan(uy))
        new_x = ux * out_fx + out_cx
        new_y = uy * out_fy + out_cy
        xs = np.where(ok, new_x * (1.0 - lca) + xs * lca, xs)
        ys = np.where(ok, new_y * (1.0 - lca) + ys * lca, ys)

    # Matrix selection. Global shutter: the single matrix. Rolling shutter:
    # trial-map every pixel with the center matrix, estimate each pixel's
    # source line, then gather that line's matrix per pixel (upstream
    # ``undistort_coord`` semantics).
    matrix_count = len(matrices)
    horizontal = bool(kp.flags & _HORIZONTAL_RS_FLAG)
    # IBIS active flag from the small (N, 14) stack — checking the gathered
    # per-pixel array costs a full extra pass over ~100 MB.
    ibis_any = matrix_count > 0 and bool(np.any(matrices[:matrix_count, 9:14] != 0.0))

    if matrix_count > 1:
        center = matrices[matrix_count // 2]
        trial_x, trial_y, trial_valid = _rotate_and_distort(xs, ys, center, kp, model, ibis_active=ibis_any)

        if horizontal:
            est = np.rint(trial_x)
            fallback = xs
        else:
            est = np.rint(trial_y)
            fallback = ys
        est = np.where(trial_valid, est, fallback)
        idx_map = np.clip(est, 0, matrix_count - 1).astype(np.int64)

        m = matrices[idx_map]  # (H, W, 14) per-pixel matrix rows
        # (Row-band / run-length segmentation was tried here: evaluating
        # per-run with a scalar matrix to avoid the ~80 MB gather. It is
        # SLOWER in pure Python — per-call numpy overhead dominates when
        # rows split into many short runs. The vectorized per-pixel
        # gather is the optimum for this implementation.)
    else:
        m = matrices[0]

    src_x, src_y, valid = _rotate_and_distort(xs, ys, m, kp, model, ibis_active=ibis_any)

    # EWA needs the Jacobian of this map, and it has to be measured before the
    # background-mode extension below: a repeated or mirrored region is flat,
    # which would collapse the ellipse into its degenerate case.  Upstream
    # differentiates the mapping analytically-ish (forward differences at the
    # pixel); here the map is already materialised, so central differences of
    # the same map give the same thing for free.
    ewa_jac = None
    if int(interpolation) in EWA_FILTERS:
        ewa_jac = map_jacobian(src_x, src_y, valid)

    # Background modes (upstream semantics: repeat clamps to a 3px margin,
    # mirror reflects around it)
    bg_mode = kp.background_mode
    if bg_mode == 1:  # Edge repeat
        src_x = np.clip(src_x, 3.0, max(3.0, in_w - 3.0))
        src_y = np.clip(src_y, 3.0, max(3.0, in_h - 3.0))
    elif bg_mode == 2:  # Edge mirror
        rx = np.rint(src_x)
        ry = np.rint(src_y)
        width3 = in_w - 3.0
        height3 = in_h - 3.0
        src_x = np.where(rx > width3, width3 - (rx - width3), src_x)
        src_x = np.where(rx < 3.0, 6.0 - rx, src_x)
        src_y = np.where(ry > height3, height3 - (ry - height3), src_y)
        src_y = np.where(ry < 3.0, 6.0 - ry, src_y)

    # Sampling via cv2.remap (C implementation, ~80x faster than the numpy
    # gather). Semantic parity with the previous bilinear sampler:
    #  - the last row/column is sampled with edge replication: coordinates
    #    in [dim-1, dim) clamp to dim-1-eps so INTER_LINEAR returns the
    #    exact edge pixel;
    #  - invalid pixels (w <= 0, r_limit, non-converged undistort) and
    #    out-of-frame coordinates fall outside and take the constant
    #    border value (the background colour).
    import cv2

    map_x = src_x.astype(np.float32)
    map_y = src_y.astype(np.float32)

    fringe_x = (map_x >= in_w - 1) & (map_x < in_w)
    map_x[fringe_x] = np.float32(in_w - 1) - np.float32(1e-4)
    fringe_y = (map_y >= in_h - 1) & (map_y < in_h)
    map_y[fringe_y] = np.float32(in_h - 1) - np.float32(1e-4)

    sentinel = np.float32(-1e5)
    map_x = np.where(valid, map_x, sentinel)
    map_y = np.where(valid, map_y, sentinel)

    if frame.dtype == np.uint8:
        border = tuple(int(np.clip(round(b), 0, 255)) for b in bg_out)
    elif frame.dtype == np.uint16:
        border = tuple(int(np.clip(round(b), 0, 65535)) for b in bg_out)
    else:
        border = tuple(float(b) for b in bg_out)
    if channels == 1:
        border = border[0]

    if ewa_jac is not None:
        # EWA (indices 3-6) is not expressible as an OpenCV remap: the kernel
        # is stretched by the local Jacobian, so it needs its own gather.
        _warn_ewa_cost_once()
        return ewa_sample(frame, src_x, src_y, ewa_jac, valid, int(interpolation), border)

    interp_flags = {
        0: cv2.INTER_LINEAR,
        1: cv2.INTER_CUBIC,
        2: cv2.INTER_LANCZOS4,
    }
    interp_flag = interp_flags.get(int(interpolation), cv2.INTER_LANCZOS4)

    if interp_flag == cv2.INTER_LINEAR:
        output = cv2.remap(
            frame, map_x, map_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=border,
        )
    else:
        # Wider kernels (Bicubic ±2 px, Lanczos4 ±8 px) would bleed the
        # constant border colour into near-edge samples. Replicate the edge
        # instead, then paint the invalid / out-of-frame pixels with the
        # background colour afterwards — keeps the BORDER_CONSTANT
        # semantics of the bilinear path without the fringe.
        output = cv2.remap(
            frame, map_x, map_y, interp_flag,
            borderMode=cv2.BORDER_REPLICATE,
        )
        oob = (src_x < 0) | (src_x >= in_w) | (src_y < 0) | (src_y >= in_h)
        paint = ~valid | oob
        if paint.any():
            if channels == 1:
                output[paint] = border
            else:
                output[paint] = np.array(border, dtype=output.dtype)
    if channels == 1 and output.ndim == 2:
        output = output[:, :, np.newaxis]
    return output


# ----------------------------------------------------------------------
# The per-point path
# ----------------------------------------------------------------------
#
# Everything below answers a different question from `cpu_undistort` above: not
# "what does this output pixel sample?" but "where did this image point end
# up?". Callers that work in points — the autosync optical-flow sampler, the
# adaptive-zoom polygon, the STMap exporter — used to do their own, simpler
# version of this on distorted coordinates, which is why their results
# disagreed with the render.
#
# On precision: upstream runs this family in `f32` (the rotation is converted
# to a float matrix, and the distortion models take `(f32, f32)`). This port
# keeps the doubles the rest of its Python side uses. The algorithm and the
# branch structure are upstream's; the results agree to single precision, not
# bit for bit.

# What upstream returns for a point that will not converge, instead of
# raising: its callers drop those points and keep the rest.
_POINT_FAILURE = (-1000000.0, -1000000.0)


def _points_kernel_params(
    camera_matrix, distortion_coeffs, params, light_refraction_coefficient
) -> KernelParams:
    """The minimal KernelParams the distortion models need for one point.

    Deliberately not the full set: upstream builds a fresh one here too, and
    the models only read the intrinsics, the coefficients and the refraction
    factor.
    """
    import ctypes

    kernel_params = KernelParams()
    kernel_params.width = int(params.width)
    kernel_params.height = int(params.height)
    kernel_params.output_width = int(params.output_width)
    kernel_params.output_height = int(params.output_height)
    kernel_params.f = (ctypes.c_float * 2)(
        float(camera_matrix[0][0]), float(camera_matrix[1][1])
    )
    kernel_params.c = (ctypes.c_float * 2)(
        float(camera_matrix[0][2]), float(camera_matrix[1][2])
    )
    coeffs = [float(x) for x in distortion_coeffs]
    while len(coeffs) < 12:
        coeffs.append(0.0)
    kernel_params.k1 = (ctypes.c_float * 4)(*coeffs[0:4])
    kernel_params.k2 = (ctypes.c_float * 4)(*coeffs[4:8])
    kernel_params.k3 = (ctypes.c_float * 4)(*coeffs[8:12])
    kernel_params.light_refraction_coefficient = float(light_refraction_coefficient)
    return kernel_params


def _input_stretch(params) -> tuple[float, float]:
    """The stretch factors the points are expressed in.

    Read from the static lens, as upstream's ``undistort_points`` does — which
    is *not* what the image path uses. ``FrameTransform.at_timestamp`` takes
    the per-frame stretch that ``get_lens_data_at_timestamp`` returns; the
    points path drops it. The two differ only for a clip whose calibration
    changes mid-flight, and upstream prefers the static value here, so this
    does too rather than quietly reconciling them.
    """
    lens = params.lens
    if lens is None:
        return params.input_horizontal_stretch, params.input_vertical_stretch
    return (
        getattr(lens, "input_horizontal_stretch", 1.0) or 1.0,
        getattr(lens, "input_vertical_stretch", 1.0) or 1.0,
    )


def _apply_mesh(x: float, y: float, mesh, params) -> tuple[float, float]:
    """Focal-plane distortion, and the full mesh table on top of it.

    Both live in the same buffer and are gated by its first word: a positive
    header means "this many words of focal-plane table follow", and a value
    above 10 means a full mesh is present as well. The remapping in and out of
    the crop area is part of the format, not an implementation detail — the
    table is indexed by grid cell, not by pixel.
    """
    from pygyroflow.util import map_coord

    mesh_size = (mesh[3], mesh[4])
    origin = (mesh[5], mesh[6])
    crop_size = (mesh[7], mesh[8])

    if mesh[0] > 0.0 and mesh[int(mesh[0])] > 0.0:
        offset = int(mesh[0])  # first word = offset to the focal-plane table
        stabilization_grid = mesh_size[1] / 8.0

        x = map_coord(x, 0.0, float(params.width), origin[0], origin[0] + crop_size[0])
        y = map_coord(y, 0.0, float(params.height), origin[1], origin[1] + crop_size[1])

        index = int(min(max(math.floor(y / stabilization_grid), 0.0), 7.0))
        delta = y - stabilization_grid * index
        x += float(mesh[offset + 4 + index * 2 + 0]) * delta
        y += float(mesh[offset + 4 + index * 2 + 1]) * delta
        for j in range(index):
            x += float(mesh[offset + 4 + j * 2 + 0]) * stabilization_grid
            y += float(mesh[offset + 4 + j * 2 + 1]) * stabilization_grid

        x = map_coord(x, origin[0], origin[0] + crop_size[0], 0.0, float(params.width))
        y = map_coord(y, origin[1], origin[1] + crop_size[1], 0.0, float(params.height))

    if mesh[0] > 10.0:
        from pygyroflow.gyro_source.splines import interpolate_mesh

        x = map_coord(x, 0.0, float(params.width), origin[0], origin[0] + crop_size[0])
        y = map_coord(y, 0.0, float(params.height), origin[1], origin[1] + crop_size[1])

        new_x, new_y = interpolate_mesh(x, y, (mesh_size[0], mesh_size[1]), mesh)

        x = map_coord(new_x, origin[0], origin[0] + crop_size[0], 0.0, float(params.width))
        y = map_coord(new_y, origin[1], origin[1] + crop_size[1], 0.0, float(params.height))

    return x, y


def _partial_correction(
    pt, c, f, params, kernel_params, model, digital_lens, lens_correction_amount
):
    """Blend the corrected point back toward the uncorrected one.

    Port of the ``lens_correction_amount < 1`` branch: re-distort the corrected
    point and mix. This is what the "lens correction strength" slider does —
    at 1 the distortion is fully removed, at 0 the output keeps the original
    look — and the mixing happens in distorted coordinates, so it cannot be a
    linear blend of the two endpoints.
    """
    stretch_x, stretch_y = _input_stretch(params)
    out_c = [params.output_width / 2.0, params.output_height / 2.0]
    if stretch_x > 0.001:
        out_c[0] /= stretch_x
    if stretch_y > 0.001:
        out_c[1] /= stretch_y

    new_pt = ((pt[0] - out_c[0]) / f[0], (pt[1] - out_c[1]) / f[1])

    weight = 1.0
    refraction = kernel_params.light_refraction_coefficient
    if refraction != 1.0 and refraction > 0.0:
        radius = math.sqrt(new_pt[0] ** 2 + new_pt[1] ** 2) / weight
        sin_theta_d = (radius / math.sqrt(1.0 + radius * radius)) * refraction
        r_d = sin_theta_d / math.sqrt(1.0 - sin_theta_d * sin_theta_d)
        if r_d != 0.0:
            weight *= radius / r_d

    new_pt = model.distort_point(new_pt[0], new_pt[1], weight, kernel_params)
    new_pt = (new_pt[0] * f[0] + out_c[0], new_pt[1] * f[1] + out_c[1])

    if digital_lens is not None:
        new_pt = digital_lens.distort_point(new_pt[0], new_pt[1], 1.0, kernel_params)
        if digital_lens.id() in (
            "gopro_superview", "gopro6_superview", "gopro_hyperview"
        ):
            # Upstream's own comment says this is wrong but works. Kept as-is:
            # it is what the SuperView/HyperView look is calibrated against.
            size = (float(params.width), float(params.height))
            new_pt = (new_pt[0] / size[0] - 0.5, new_pt[1] / size[1] - 0.5)
            if digital_lens.id() in ("gopro_superview", "gopro6_superview"):
                new_pt = (new_pt[0] * 0.91, new_pt[1])
            else:
                new_pt = (new_pt[0] * 0.81, new_pt[1])
            new_pt = ((new_pt[0] + 0.5) * size[0], (new_pt[1] + 0.5) * size[1])

    amount = lens_correction_amount
    return (
        new_pt[0] * (1.0 - amount) + pt[0] * amount,
        new_pt[1] * (1.0 - amount) + pt[1] * amount,
    )


def undistort_points(
    distorted,
    camera_matrix,
    distortion_coeffs,
    rotation,
    p=None,
    rot_per_point=None,
    params=None,
    lens_correction_amount: float = 1.0,
    timestamp_ms: float = 0.0,
    shift_per_point=None,
    mesh=None,
) -> list[tuple[float, float]]:
    """Map image points through the lens into stabilized output coordinates.

    Port of ``cpu_undistort.rs::undistort_points``, which follows OpenCV's
    ``undistortPoints`` for fisheye and then adds everything Gyroflow needs on
    top: the digital lens, a focal-plane/mesh correction, the camera's own IBIS
    and OIS displacement, a per-point rotation for rows exposed at different
    times, light refraction, and a blend back toward the uncorrected position
    when the lens correction is dialled down.

    A point that does not converge comes back as ``(-1000000, -1000000)``.
    """
    from pygyroflow.keyframes.types import KeyframeType
    from pygyroflow.stabilization.distortion_models import from_name as model_from_name

    c = (float(camera_matrix[0][2]), float(camera_matrix[1][2]))
    f = (float(camera_matrix[0][0]), float(camera_matrix[1][1]))

    rr = np.asarray(rotation, dtype=np.float64)
    if p is not None:
        rr = np.asarray(p, dtype=np.float64) @ rr

    light_refraction_coefficient = float(params.light_refraction_coefficient)
    if params.keyframes:
        value = params.keyframes.value_at_video_timestamp(
            KeyframeType.LightRefractionCoeff, timestamp_ms
        )
        if value is not None:
            light_refraction_coefficient = float(value)

    kernel_params = _points_kernel_params(
        camera_matrix, distortion_coeffs, params, light_refraction_coefficient
    )

    model = model_from_name(params.distortion_model_name or "opencv_fisheye")
    digital_lens = params.digital_lens
    stretch_x, stretch_y = _input_stretch(params)
    result: list[tuple[float, float]] = []

    for index, point in enumerate(distorted):
        x = float(point[0])
        y = float(point[1])
        if stretch_x > 0.001:
            x *= stretch_x
        if stretch_y > 0.001:
            y *= stretch_y

        if digital_lens is not None:
            moved = digital_lens.undistort_point(x, y, kernel_params)
            if moved is not None:
                x, y = moved

        if mesh is not None:
            x, y = _apply_mesh(x, y, mesh, params)

        if shift_per_point is not None and index < len(shift_per_point):
            shift = shift_per_point[index]
            angle = shift[2]
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            x = x - c[0] - shift[3] + shift[0]
            y = y - c[1] - shift[4] + shift[1]
            x, y = cos_a * x - sin_a * y + c[0], sin_a * x + cos_a * y + c[1]

        # Normalised units, which is what the distortion models take.
        pw = ((x - c[0]) / f[0], (y - c[1]) / f[1])

        if rot_per_point is not None and index < len(rot_per_point):
            point_rotation = np.asarray(rot_per_point[index], dtype=np.float64)
        else:
            point_rotation = rr

        pt = model.undistort_point(pw[0], pw[1], kernel_params)
        if pt is None:
            result.append(_POINT_FAILURE)
            continue

        if light_refraction_coefficient != 1.0 and light_refraction_coefficient > 0.0:
            radius = math.sqrt(pt[0] ** 2 + pt[1] ** 2)
            if radius != 0.0:
                sin_theta_d = (radius / math.sqrt(1.0 + radius * radius)) / (
                    light_refraction_coefficient
                )
                r_d = sin_theta_d / math.sqrt(1.0 - sin_theta_d * sin_theta_d)
                factor = r_d / radius
                pt = (pt[0] * factor, pt[1] * factor)

        # Reproject through the point's rotation (which the caller has already
        # folded the output matrix into).
        projected = point_rotation @ np.array([pt[0], pt[1], 1.0], dtype=np.float64)
        if projected[2] == 0.0:
            result.append(_POINT_FAILURE)
            continue
        pt = (projected[0] / projected[2], projected[1] / projected[2])

        if lens_correction_amount < 1.0:
            pt = _partial_correction(
                pt, c, f, params, kernel_params, model, digital_lens,
                lens_correction_amount,
            )

        result.append((float(pt[0]), float(pt[1])))

    return result


def undistort_points_with_rolling_shutter(
    distorted, timestamp_ms: float, frame=None, params=None,
    lens_correction_amount: float = 1.0, use_fovs: bool = True,
) -> list[tuple[float, float]]:
    """Undistort points for one frame, rolling shutter included.

    Port of ``undistort_points_with_rolling_shutter``. This is the entry point
    most callers want: it asks :func:`at_timestamp_for_points
    <pygyroflow.stabilization.frame_transform.at_timestamp_for_points>` for the
    per-point rotations and then runs :func:`undistort_points` with them — each
    point is rotated by the orientation at *its own* row's exposure, which is
    the whole reason a fisheye and a rolling shutter interact.
    """
    from pygyroflow.stabilization.frame_transform import at_timestamp_for_points

    if not distorted:
        return []

    camera_matrix, coeffs, _new_k, rotations, shifts, mesh = at_timestamp_for_points(
        params, distorted, timestamp_ms, frame, use_fovs
    )
    return undistort_points(
        distorted, camera_matrix, coeffs, rotations[0],
        p=np.eye(3),
        rot_per_point=rotations,
        params=params,
        lens_correction_amount=lens_correction_amount,
        timestamp_ms=timestamp_ms,
        shift_per_point=shifts,
        mesh=mesh,
    )


def undistort_points_for_optical_flow(
    distorted, timestamp_us: int, params, points_dims: tuple[int, int]
) -> list[tuple[float, float]]:
    """Undistort points sampled for optical flow, in the flow's own scale.

    Port of ``undistort_points_for_optical_flow``. The difference from the
    render path is the *scale*: optical flow runs on a downscaled frame, so the
    points are expressed against ``points_dims`` while the calibration is
    against the full frame — the matrix is scaled to match, and no rotation and
    no stabilization is applied. This is the call that puts the flow's feature
    points into undistorted coordinates, which the offset search then assumes.
    """
    from pygyroflow.stabilization.frame_transform import _get_lens_data_at_timestamp

    image_dim_ratio = points_dims[0] / max(1, params.width)

    camera_matrix, coeffs, _, _, _, _ = _get_lens_data_at_timestamp(
        params, timestamp_us / 1000.0, False
    )
    scaled_k = np.asarray(camera_matrix, dtype=np.float64) * image_dim_ratio

    return undistort_points(
        distorted, scaled_k, coeffs, np.eye(3),
        p=None, rot_per_point=None, params=params,
        lens_correction_amount=1.0, timestamp_ms=timestamp_us / 1000.0,
        shift_per_point=None, mesh=None,
    )
