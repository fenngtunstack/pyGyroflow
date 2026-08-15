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

import numpy as np
from numpy.typing import NDArray

from pygyroflow.stabilization.frame_transform import FrameTransform
from pygyroflow.stabilization.distortion_models import from_name as model_from_name
from pygyroflow.types.kernel_params import KernelParams

# KernelParamsFlags::HORIZONTAL_RS (1 << 4)
_HORIZONTAL_RS_FLAG = 16

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
        interpolation: Ignored; always uses bilinear.

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
    else:
        m = matrices[0]

    src_x, src_y, valid = _rotate_and_distort(xs, ys, m, kp, model, ibis_active=ibis_any)

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

    output = cv2.remap(
        frame, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=border,
    )
    if channels == 1 and output.ndim == 2:
        output = output[:, :, np.newaxis]
    return output
