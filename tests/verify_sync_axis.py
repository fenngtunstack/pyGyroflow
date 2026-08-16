# -*- coding: utf-8 -*-
"""Axis-wise cross-correlation of visual vs gyro angular velocity.

Uses the PoseEstimator's per-frame-pair angular velocity vectors (not just
magnitude) and correlates each axis against the gyro quaternion derivative
over the high-motion window, where sync signal dominates.

Usage:
    python tests/verify_sync_axis.py <video> [t0 t1]
"""

from __future__ import annotations

import sys

import av
import cv2
import numpy as np

from pygyroflow.manager import StabilizationManager
from pygyroflow.synchronization import PoseEstimator


def visual_rotations(path: str, max_width: int = 480):
    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    fps = float(stream.average_rate or 30.0)
    tb = float(stream.time_base or 1)
    scale = min(1.0, max_width / max(1, stream.width))
    out_w = max(2, int(stream.width * scale) & ~1)
    out_h = max(2, int(stream.height * scale) & ~1)

    est = PoseEstimator()
    est.set_fps(fps)
    est.set_optical_flow_method(2)

    idx = 0
    for packet in container.demux(stream):
        if packet.dts is None:
            continue
        for frame in packet.decode():
            gray = frame.to_ndarray(format="gray")
            if scale < 1.0:
                gray = cv2.resize(gray, (out_w, out_h), interpolation=cv2.INTER_AREA)
            ts_us = (
                int(round(float(frame.pts) * tb * 1e6))
                if frame.pts is not None
                else int(round(idx * 1e6 / fps))
            )
            est.feed_frame(idx, ts_us, gray)
            idx += 1
    container.close()
    est.process_all()
    return est.get_visual_rotations(), fps


def main(video: str, t0: float, t1: float) -> int:
    m = StabilizationManager()
    m.load_video(video)
    m.load_gyro_data(video)

    rots, fps = visual_rotations(video)
    vt = np.array([r[0] / 1e6 for r in rots], dtype=np.float64)
    vw = np.array([r[1] for r in rots], dtype=np.float64)  # (N,3) deg/s

    oq = m.gyro.quaternions
    keys = sorted(oq.keys())
    gt = np.array(keys, dtype=np.float64) / 1e6
    qs = np.asarray([oq[k].quaternion() for k in keys], dtype=np.float64)
    # quaternion log-derivative -> axis-angle rate vector in deg/s
    dq = qs[1:] * qs[:-1]  # w,x,y,z multiply (relative rotation)
    w = np.clip(dq[:, 0], -1.0, 1.0)
    angle = 2.0 * np.degrees(np.arccos(np.abs(w)))
    axis = dq[:, 1:4] / np.maximum(np.linalg.norm(dq[:, 1:4], axis=1, keepdims=True), 1e-12)
    sign = np.where((dq[:, 0] < 0)[:, None], -1.0, 1.0)
    gw_full = axis * (angle[:, None] * sign)
    gdt = np.maximum(np.diff(gt), 1e-9)
    gw_full = gw_full / gdt[:, None]
    gt = gt[1:]

    # restrict to the high-motion window
    msk = (vt >= t0) & (vt <= t1)
    vt_w, vw_w = vt[msk], vw[msk]
    print(f"visual samples in window: {msk.sum()}, |w| med "
          f"{np.median(np.linalg.norm(vw_w, axis=1)):.1f} deg/s")

    best_off = None
    total_curve = []
    for off_ms in np.arange(-300.0, 300.0 + 1e-9, 5.0):
        g_at = np.stack(
            [np.interp(vt_w + off_ms / 1000.0, gt, gw_full[:, a]) for a in range(3)],
            axis=1,
        )
        # robust correlation: clip outlier rows
        a = np.clip(vw_w, -150, 150)
        b = np.clip(g_at, -150, 150)
        cc = 0.0
        for ax in range(3):
            sa, sb = a[:, ax] - a[:, ax].mean(), b[:, ax] - b[:, ax].mean()
            den = np.sqrt((sa * sa).sum() * (sb * sb).sum())
            if den > 0:
                cc += float((sa * sb).sum() / den)
        total_curve.append((off_ms, cc))
        if best_off is None or cc > best_off[1]:
            best_off = (off_ms, cc)
    for off_ms, cc in total_curve[::6]:
        bar = "#" * int(max(0, (cc - 0.5) * 60)) if cc > 0.5 else ""
        print(f"  offset {off_ms:7.1f} ms  corr3={cc:.4f}  {bar}")
    print(f"best: offset {best_off[0]:.1f} ms  corr3={best_off[1]:.4f}")
    return 0


if __name__ == "__main__":
    video = sys.argv[1]
    t0 = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
    t1 = float(sys.argv[3]) if len(sys.argv) > 3 else 23.0
    raise SystemExit(main(video, t0, t1))
