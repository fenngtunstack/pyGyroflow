"""CPU-based image undistortion and stabilization.

Port of Gyroflow's cpu_undistort.rs. Implements inverse mapping from output
pixels to input pixels via the rotation+distortion transform, with bilinear
interpolation sampling.

Vectorized NumPy implementation for performance.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pygyroflow.stabilization.frame_transform import FrameTransform
from pygyroflow.types.kernel_params import KernelParams


def _vectorized_rotate_distort(
    xs: NDArray[np.float32],
    ys: NDArray[np.float32],
    matrix: NDArray[np.float32],
    kp: KernelParams,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]]:
    """Vectorized inverse mapping: output coords -> input coords.

    Returns (src_x, src_y, valid_mask).
    """
    tx3d = float(kp.translation3d[0]) if hasattr(kp.translation3d, '__getitem__') else 0.0
    ty3d = float(kp.translation3d[1]) if hasattr(kp.translation3d, '__getitem__') else 0.0
    tz3d = float(kp.translation3d[2]) if hasattr(kp.translation3d, '__getitem__') else 0.0

    _x = xs * matrix[0] + ys * matrix[1] + matrix[2] + tx3d
    _y = xs * matrix[3] + ys * matrix[4] + matrix[5] + ty3d
    _w = xs * matrix[6] + ys * matrix[7] + matrix[8] + tz3d

    valid = _w > 0.0

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

    # Normalized coords
    safe_w = np.where(valid & (_w != 0.0), _w, 1.0)
    xn = np.where(valid, _x / safe_w, 0.0)
    yn = np.where(valid, _y / safe_w, 0.0)

    # Distortion coefficients
    k1 = float(kp.k1[0])
    k2 = float(kp.k1[1])
    k3 = float(kp.k1[2])
    k4 = float(kp.k1[3])

    r_sq = xn * xn + yn * yn
    r = np.sqrt(r_sq)
    theta = np.arctan(r)
    theta_d = theta * (1.0 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8)

    scale = np.where(r > 1e-8, theta_d / r, 1.0)

    fx = float(kp.f[0])
    fy = float(kp.f[1])
    cx = float(kp.c[0])
    cy = float(kp.c[1])

    u_dist = xn * scale * fx + cx
    v_dist = yn * scale * fy + cy

    # IBIS/OIS compensation
    if len(matrix) > 11 and (matrix[9] != 0.0 or matrix[10] != 0.0 or matrix[11] != 0.0):
        ang_rad = matrix[11]
        cos_a = np.cos(-ang_rad)
        sin_a = np.sin(-ang_rad)
        u_adj = cos_a * u_dist - sin_a * v_dist - matrix[9]
        v_adj = sin_a * u_dist + cos_a * v_dist - matrix[10]
        u_dist = u_adj
        v_dist = v_adj

    # Input stretch
    hs = kp.input_horizontal_stretch
    vs = kp.input_vertical_stretch
    if hs > 0.001:
        u_dist = u_dist / hs
    if vs > 0.001:
        v_dist = v_dist / vs

    return u_dist.astype(np.float32), v_dist.astype(np.float32), valid


def _bilinear_sample_vectorized(
    frame: NDArray[np.float32],
    src_x: NDArray[np.float32],
    src_y: NDArray[np.float32],
    valid: NDArray[np.bool_],
    bg: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Vectorized bilinear sampling for an entire frame."""
    h, w = frame.shape[0], frame.shape[1]
    channels = frame.shape[2] if frame.ndim == 3 else 1

    flat_x = src_x.ravel()
    flat_y = src_y.ravel()
    flat_valid = valid.ravel()
    n = len(flat_x)

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

    # Bounds check
    in_bounds = (flat_x >= 0) & (flat_y >= 0) & (flat_x < w - 1) & (flat_y < h - 1) & flat_valid

    w00 = ((1.0 - fx) * (1.0 - fy))[:, np.newaxis]
    w10 = (fx * (1.0 - fy))[:, np.newaxis]
    w01 = ((1.0 - fx) * fy)[:, np.newaxis]
    w11 = (fx * fy)[:, np.newaxis]

    if channels > 1:
        p00 = frame[y0c, x0c].astype(np.float32)
        p10 = frame[y1c, x0c].astype(np.float32)
        p01 = frame[y0c, x1c].astype(np.float32)
        p11 = frame[y1c, x1c].astype(np.float32)
    else:
        p00 = frame[y0c, x0c].astype(np.float32)[:, np.newaxis]
        p10 = frame[y1c, x0c].astype(np.float32)[:, np.newaxis]
        p01 = frame[y0c, x1c].astype(np.float32)[:, np.newaxis]
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
    matrices = transform.matrices

    out_w = kp.output_width
    out_h = kp.output_height
    in_w = kp.width
    in_h = kp.height

    if out_w <= 0 or out_h <= 0:
        return np.array([], dtype=frame.dtype)

    channels = frame.shape[2] if frame.ndim == 3 else 1

    if frame.dtype == np.uint8:
        input_float = frame.astype(np.float32)
        max_val = 255.0
    elif frame.dtype == np.uint16:
        input_float = frame.astype(np.float32)
        max_val = 65535.0
    else:
        input_float = frame.astype(np.float32)
        max_val = 0.0

    bg = np.array([kp.background[0], kp.background[1], kp.background[2], kp.background[3]],
                  dtype=np.float32)
    if max_val > 0:
        bg *= max_val
    bg_out = bg[:channels]

    # Build output coordinate grid
    ys, xs = np.mgrid[0:out_h, 0:out_w].astype(np.float32)

    # Use the single (global) or center matrix
    matrix_count = len(matrices)
    if matrix_count > 1:
        idx = matrix_count // 2
    else:
        idx = 0
    matrix = matrices[idx]

    src_x, src_y, valid = _vectorized_rotate_distort(xs, ys, matrix, kp)

    # Handle background modes
    bg_mode = kp.background_mode
    if bg_mode == 1:  # RepeatPixels: clamp
        src_x = np.clip(src_x, 0.0, in_w - 1.0)
        src_y = np.clip(src_y, 0.0, in_h - 1.0)
    elif bg_mode == 2:  # MirrorPixels
        margin = 3.0
        src_x = np.where(src_x < margin, margin + (margin - src_x), src_x)
        src_x = np.where(src_x > in_w - margin, (in_w - margin) - (src_x - (in_w - margin)), src_x)
        src_y = np.where(src_y < margin, margin + (margin - src_y), src_y)
        src_y = np.where(src_y > in_h - margin, (in_h - margin) - (src_y - (in_h - margin)), src_y)

    # Vectorized bilinear sampling
    output = _bilinear_sample_vectorized(input_float, src_x, src_y, valid, bg_out)
    output = output.reshape(out_h, out_w, channels)

    if frame.dtype == np.uint8:
        output = np.clip(output, 0, 255).astype(np.uint8)
    elif frame.dtype == np.uint16:
        output = np.clip(output, 0, 65535).astype(np.uint16)

    return output
