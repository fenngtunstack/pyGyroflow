# -*- coding: utf-8 -*-
"""Phase comparison: official implied correction vs pipeline correction.

1. Registers input frames against the official export -> |R_official|(t).
2. Computes our pipeline's per-frame correction quaternion magnitude at
   the same timestamps.
3. Cross-correlates the two magnitude series to find the timing lag.

Usage:
    python tests/phase_vs_official.py <input> <official>
"""

from __future__ import annotations

import sys

import av
import cv2
import numpy as np

from pygyroflow.manager import StabilizationManager
from tests.reverse_official import frame_stream, register_rotation


def main() -> int:
    inp, off = sys.argv[1], sys.argv[2]
    K = np.array([[350.0, 0, 240.0], [0, 350.0, 135.0], [0, 0, 1.0]])

    # our pipeline correction (exact, no registration noise)
    m = StabilizationManager()
    m.load_video(inp)
    m.load_gyro_data(inp)
    m.smoothing.current().set_parameter("smoothness", 1.0)
    m.synchronize()
    m.gyro.clear_offsets()
    m.gyro.set_offset(0, 0.0)
    m.recompute_blocking()
    cp = m._build_compute_params()
    fps = m.params.fps

    ga = frame_stream(inp)
    gb = frame_stream(off)
    t_off, a_off = [], []
    t_our, a_our = [], []
    for i, (fa, fb) in enumerate(zip(ga, gb)):
        rod, n = register_rotation(fa, fb, K)
        if rod is not None:
            t_off.append(i / fps)
            a_off.append(np.degrees(np.linalg.norm(rod)))
        if i % 2 == 0:
            ts_ms = i * 1000.0 / fps
            tr = m.get_frame_transform(ts_ms, i, compute_params=cp)
            M = np.asarray(tr.matrices, dtype=np.float64)
            if M.ndim == 3:  # (rows, 1|N, 14) rs stack vs (N, 14) global
                row = M[M.shape[0] // 2, M.shape[1] // 2, :9]
            else:
                row = M[M.shape[0] // 2, :9]
            R3 = row.reshape(3, 3)
            # normalize out fov scaling via SVD (rotation-only part)
            U, _, Vt = np.linalg.svd(R3)
            R = U @ Vt
            ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
            t_our.append(i / fps)
            a_our.append(float(ang))
        if i >= 700:
            break

    t_off = np.array(t_off)
    a_off = np.array(a_off)
    t_our = np.array(t_our)
    a_our = np.array(a_our)
    print(f"official samples={len(t_off)}  ours={len(t_our)}")
    print(f"|R| med: official={np.median(a_off):.3f}  ours={np.median(a_our):.3f}")
    print(f"d|R|/frame med: official={np.median(np.abs(np.diff(a_off))):.4f}  "
          f"ours={np.median(np.abs(np.diff(a_our))):.4f}")

    # lag scan on the fast segment 6..23 s
    msk_o = (t_off >= 6) & (t_off <= 23)
    msk_u = (t_our >= 5.5) & (t_our <= 23.5)
    to, ao = t_off[msk_o], a_off[msk_o]
    tu, au = t_our[msk_u], a_our[msk_u]
    au_c = au - au.mean()
    denom = np.sqrt((au_c * au_c).sum())
    best = (-2, 0.0)
    for lag_ms in np.arange(-120.0, 120.0 + 1e-9, 4.0):
        a_shift = np.interp(to + lag_ms / 1000.0, tu, au)
        a_s = a_shift - a_shift.mean()
        c = float((a_s * ao).sum() / (np.sqrt((a_s * a_s).sum() * (ao * ao).sum()) + 1e-12))
        if c > best[0]:
            best = (c, lag_ms)
    print(f"\nlag scan (fast segment): best corr={best[0]:.4f} at lag={best[1]:+.1f} ms "
          f"(positive = official correction lags ours)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
