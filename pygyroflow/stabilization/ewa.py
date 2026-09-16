# -*- coding: utf-8 -*-
"""EWA (Elliptical Weighted Average) resampling.

Port of the EWA branch of Gyroflow's ``core/stabilization/cpu_undistort.rs``.
EWA is the right filter when the resampling *minifies*: the inverse map's
Jacobian says how much the source is compressed at each output pixel, and the
kernel is stretched by the same amount so it keeps averaging over a full
source footprint instead of point-sampling.  A fixed-radius kernel (Bicubic,
Lanczos) has no way to do that — it aliases wherever the stabilization zooms
out or the lens correction compresses the frame edge.

The four upstream options are Keys cubic filters (CubicBC) identified by their
(B, C) parameters; the kernel is evaluated by :func:`bc2` and the ellipse that
shapes it comes from the inverse map's Jacobian via :func:`clamped_ellipse`.

Implementation notes
--------------------
* Upstream differentiates the mapping with forward differences at ``eps =
  0.01``.  The map is already materialised here, so the Jacobian comes from
  central differences of the same map (:func:`map_jacobian`) — equal to second
  order and free.  It is taken from the *unclamped* map, matching upstream,
  which differentiates before the background-mode extension; a clamped
  (repeated or mirrored) region is flat and would collapse the ellipse into
  its degenerate case.
* The per-pixel sampling box varies in size, so the loop here runs over the
  largest box in the frame and masks off the taps each pixel does not want.
  Cost scales with the *worst* minification in the frame, which is why this is
  opt-in rather than the default.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Upstream's interpolation enum order: 3..6 are the EWA filters.
EWA_FILTERS: dict[int, tuple[float, float]] = {
    3: (0.2620145, 0.3689927),  # RobidouxSharp
    4: (0.3782157, 0.3108921),  # Robidoux
    5: (0.3333333, 0.3333333),  # Mitchell
    6: (0.0000000, 0.5000000),  # Catmull-Rom
}

EWA_NAMES: dict[int, str] = {
    3: "RobidouxSharp",
    4: "Robidoux",
    5: "Mitchell",
    6: "CatmullRom",
}

# Hard cap on the sampling half-box.  Upstream lets the box grow with the
# Jacobian without a limit; here every tap costs a gather of the whole plane,
# so the radius is capped (a 13x13 worst case) and the clipping reported.
MAX_RADIUS = 6


def cubic_bc_coeffs(b: float, c: float) -> tuple[np.ndarray, np.ndarray]:
    """Polynomial coefficients of the CubicBC kernel.

    Port of upstream's ``ewa_coeffs_p`` (|x| < 1 branch) and ``ewa_coeffs_q``
    (1 <= |x| < 2): the standard Keys cubic written out for parameters (B, C).
    """
    p = np.array([
        (6.0 - 2.0 * b) / 6.0,
        0.0,
        (-18.0 + 12.0 * b + 6.0 * c) / 6.0,
        (12.0 - 9.0 * b - 6.0 * c) / 6.0,
    ], dtype=np.float64)
    q = np.array([
        (8.0 * b + 24.0 * c) / 6.0,
        (-12.0 * b - 48.0 * c) / 6.0,
        (6.0 * b + 30.0 * c) / 6.0,
        (-1.0 * b - 6.0 * c) / 6.0,
    ], dtype=np.float64)
    return p, q


def bc2(x: npt.NDArray[np.float64], p: npt.NDArray[np.float64],
        q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """CubicBC kernel, vectorized.  Zero for |x| >= 2."""
    ax = np.abs(x)
    x2 = ax * ax
    return np.where(
        ax < 1.0,
        p[0] + p[1] * ax + p[2] * x2 + p[3] * x2 * ax,
        np.where(
            ax < 2.0,
            q[0] + q[1] * ax + q[2] * x2 + q[3] * x2 * ax,
            0.0,
        ),
    )


def affine_bbox(jac: tuple[np.ndarray, ...]) -> tuple[npt.NDArray, npt.NDArray]:
    """Half-extents of the box covering the unit circle under *jac*.

    Both footprints have to be covered, hence the ``max(..., 1.0)``: a
    minifying map has small derivatives, but the destination pixel still needs
    its own pixel's worth of samples.
    """
    jx, jy, jz, jw = jac
    return (
        2.0 * np.maximum(np.maximum(np.abs(jx + jy), np.abs(jx - jy)), 1.0),
        2.0 * np.maximum(np.maximum(np.abs(jz + jw), np.abs(jz - jw)), 1.0),
    )


def clamped_ellipse(jac: tuple[np.ndarray, ...]) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Minimum-area ellipse covering the unit circle in both images.

    Returns the quadratic form ``(a, b, c)`` such that the weight of a tap
    offset ``(fx, fy)`` from the sample centre is
    ``bc2(sqrt(fx^2*a + fx*fy*b + fy^2*c))``.
    """
    jx, jy, jz, jw = jac
    f0 = np.abs(jx * jw - jy * jz)
    f = np.maximum(f0 * f0, 0.1)
    a = (jz * jz + jw * jw) / f
    b = -2.0 * (jx * jz + jy * jw) / f
    c = (jx * jx + jy * jy) / f

    vx = c - a
    vy = -b
    lv = np.hypot(vx, vy)
    v0 = np.where(lv > 0.01, vx / np.where(lv > 0.0, lv, 1.0), 1.0)
    cc = np.sqrt(np.maximum(1.0 + v0, 0.0) / 2.0)
    s = np.sqrt(np.maximum(1.0 - v0, 0.0) / 2.0)

    a0 = a * cc * cc - b * cc * s + c * s * s
    c0 = a * s * s + b * cc * s + c * cc * cc
    bt1 = b * (cc * cc - s * s)
    bt2 = 2.0 * (a - c) * cc * s
    b0 = bt1 + bt2
    b0_alt = bt1 - bt2

    flip = np.abs(b0) > np.abs(b0_alt)
    s = np.where(flip, -s, s)
    b0 = np.where(flip, b0_alt, b0)

    a0 = np.minimum(a0, 1.0)
    c0 = np.minimum(c0, 1.0)
    sn = -s
    return (
        a0 * cc * cc - b0 * cc * sn + c0 * sn * sn,
        2.0 * a0 * cc * sn + b0 * cc * cc - b0 * sn * sn - 2.0 * c0 * cc * sn,
        a0 * sn * sn + b0 * cc * sn + c0 * cc * cc,
    )


def map_jacobian(
    map_x: npt.NDArray,
    map_y: npt.NDArray,
    valid: npt.NDArray | None = None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Numerical Jacobian of the destination -> source map.

    Returns ``(du/dx, du/dy, dv/dx, dv/dy)`` — the same four numbers upstream
    packs into its Jacobian vector.

    Pixels without a defined source (``valid`` False) carry placeholder
    coordinates, so their derivatives would be enormous and blow the sampling
    box up to the radius cap for the whole frame.  Those pixels, and the
    one-pixel halo whose differences they corrupt, are neutralised to the
    identity: they are painted with the background afterwards, so their
    Jacobian only has to stay finite.
    """
    mx = np.asarray(map_x, dtype=np.float64)
    my = np.asarray(map_y, dtype=np.float64)
    if valid is not None:
        ok = np.asarray(valid, dtype=bool)
        if not ok.all():
            mx = np.where(ok, mx, 0.0)
            my = np.where(ok, my, 0.0)

    du_dy, du_dx = np.gradient(mx)
    dv_dy, dv_dx = np.gradient(my)

    if valid is not None and not ok.all():
        halo = ~ok
        halo = halo | np.roll(halo, 1, axis=0) | np.roll(halo, -1, axis=0)
        halo = halo | np.roll(halo, 1, axis=1) | np.roll(halo, -1, axis=1)
        du_dx = np.where(halo, 1.0, du_dx)
        du_dy = np.where(halo, 0.0, du_dy)
        dv_dx = np.where(halo, 0.0, dv_dx)
        dv_dy = np.where(halo, 1.0, dv_dy)

    return du_dx, du_dy, dv_dx, dv_dy


def ewa_sample(
    frame: npt.NDArray,
    map_x: npt.NDArray,
    map_y: npt.NDArray,
    jac: tuple[np.ndarray, ...],
    valid: npt.NDArray,
    interpolation: int,
    border,
    chunk_rows: int = 192,
) -> npt.NDArray:
    """Resample *frame* with elliptical weighted averaging.

    Args:
        frame: (H, W, C) or (H, W) source frame (uint8/uint16/float32).
        map_x, map_y: destination -> source coordinate maps.
        jac: Jacobian from :func:`map_jacobian`, taken from the *unclamped* map.
        valid: destination pixels that have a defined source position.
        interpolation: upstream index, one of :data:`EWA_FILTERS`.
        border: background value, per channel for 3-D input.
        chunk_rows: rows per pass, bounding peak memory.

    Returns:
        Array shaped like *frame*, in its dtype.  Taps landing off the source
        frame are weighted with the border value — upstream's ``*bg``
        substitution, which lets the background bleed into the frame edge the
        way the reference does.
    """
    if interpolation not in EWA_FILTERS:
        raise ValueError(f"Not an EWA interpolation index: {interpolation}")

    b_param, c_param = EWA_FILTERS[interpolation]
    p, q = cubic_bc_coeffs(b_param, c_param)

    squeeze = frame.ndim == 2
    src = np.asarray(frame, dtype=np.float64)
    if squeeze:
        src = src[:, :, None]
    height, width, channels = src.shape

    map_x = np.asarray(map_x, dtype=np.float64)
    map_y = np.asarray(map_y, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)

    border_arr = np.asarray(border, dtype=np.float64).reshape(1, 1, channels)

    half_x, half_y = affine_bbox(jac)
    radius = int(np.ceil(float(max(np.max(half_x), np.max(half_y)))))
    if radius > MAX_RADIUS:
        half_x = np.minimum(half_x, float(MAX_RADIUS))
        half_y = np.minimum(half_y, float(MAX_RADIUS))
        radius = MAX_RADIUS

    out = np.empty((height, width, channels), dtype=np.float64)
    src_flat = src

    for row_start in range(0, height, chunk_rows):
        row_end = min(row_start + chunk_rows, height)
        sl = slice(row_start, row_end)

        mx = map_x[sl]
        my = map_y[sl]
        base_x = np.floor(mx).astype(np.int64)
        base_y = np.floor(my).astype(np.int64)
        frac_x = mx - base_x
        frac_y = my - base_y

        ea, eb, ec = (comp[sl] for comp in clamped_ellipse(jac))
        bx = half_x[sl]
        by = half_y[sl]

        acc = np.zeros(mx.shape + (channels,), dtype=np.float64)
        weight_sum = np.zeros(mx.shape, dtype=np.float64)

        # Tap offsets run from -radius to radius+1: the box is
        # [floor(uv - b), ceil(uv + b)], and with b <= radius the fractional
        # part pushes the upper end one past floor(uv) + radius.  The bbox test
        # below rejects whatever the widened range picks up in excess.
        for dy in range(-radius, radius + 2):
            fy = dy - frac_y
            fy2 = fy * fy
            tap_y = base_y + dy
            in_y = np.abs(fy) <= by
            if not np.any(in_y):
                continue

            for dx in range(-radius, radius + 2):
                tap_x = base_x + dx
                # The box test decides which taps the kernel wants.  Frame
                # bounds are deliberately NOT part of it: an off-frame tap
                # still carries its weight, substituting the background (see
                # on_frame below) — that is upstream's behaviour.
                inside = in_y & (np.abs(dx - frac_x) <= bx)
                if not np.any(inside):
                    continue

                fx = dx - frac_x
                dr = fx * fx * ea + fx * fy * eb + fy2 * ec
                k = bc2(np.sqrt(np.maximum(dr, 0.0)), p, q)
                k = np.where(inside, k, 0.0)
                if not np.any(k):
                    continue

                gathered = src_flat[
                    np.clip(tap_y, 0, height - 1), np.clip(tap_x, 0, width - 1)
                ]
                # Taps landing off the frame contribute the background with
                # their full weight (upstream substitutes *bg instead of
                # renormalising over the in-frame taps).
                on_frame = (tap_y >= 0) & (tap_y < height) & (tap_x >= 0) & (tap_x < width)
                contribution = np.where(on_frame[..., None], gathered, border_arr)
                acc += k[..., None] * contribution
                weight_sum += k

        safe = weight_sum > 0.0
        out[sl] = np.where(
            safe[..., None],
            acc / np.where(safe, weight_sum, 1.0)[..., None],
            np.broadcast_to(border_arr, mx.shape + (channels,)),
        )

    out[~valid] = border_arr[0, 0]
    if squeeze:
        out = out[:, :, 0]

    # Match the reference's float -> integer conversion (round half up), and
    # clamp: a weighted sum can overshoot the range because off-frame taps
    # carry the border value at full weight.
    if frame.dtype == np.uint8:
        out = np.clip(np.floor(out + 0.5), 0.0, 255.0)
    elif frame.dtype == np.uint16:
        out = np.clip(np.floor(out + 0.5), 0.0, 65535.0)
    return out.astype(frame.dtype, copy=False)
