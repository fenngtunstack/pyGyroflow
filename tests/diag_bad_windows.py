# -*- coding: utf-8 -*-
"""Diagnose shaky segments reported by the user.

Per-1s-window stats for a clip:
  a) raw gyro |omega| (deg/s, from GyroSource IMU stream)
  b) smoothed-correction quaternion angular step (deg per second)
  c) IMU sample gaps and quaternion sign discontinuities
to see whether bad windows coincide with fast motion, data gaps, or
correction-quat anomalies.

Usage:
    python tests/diag_bad_windows.py <video> <label>
"""

from __future__ import annotations

import sys

import numpy as np

from pygyroflow.manager import StabilizationManager


def diag(video: str, label: str) -> None:
    m = StabilizationManager()
    m.load_video(video)
    m.load_gyro_data(video)
    m.synchronize()
    m.recompute_smoothing()

    gyro = m.gyro
    fps = m.params.fps
    offs = gyro.get_offsets()
    print(f"\n=== {label} ===")
    print(f"fps={fps:.3f}  duration={m.params.duration_ms/1000:.1f}s  "
          f"offsets={ {round(k/1e6, 2): round(v, 1) for k, v in offs.items()} }  "
          f"method={gyro.integration_method}")

    # raw imu angular velocity (absent for direct-quaternion sources like DJI)
    imus = gyro._get_imu_data()
    t = w = None
    if imus:
        t = np.array([x.timestamp_ms for x in imus], dtype=np.float64) / 1000.0
        g = np.asarray([x.gyro for x in imus], dtype=np.float64)
        w = np.linalg.norm(g, axis=-1)
        dt = np.diff(t)
        gaps = np.where(dt > 3.0 * np.median(dt))[0]
        print(f"imu samples={len(t)}  median_dt={np.median(dt)*1000:.2f}ms  "
              f"gaps>3x: {len(gaps)}")
        for gi in gaps[:12]:
            print(f"   gap at {t[gi]:.3f}s  dt={dt[gi]*1000:.1f}ms")

    # original (unsmoothed) orientation angular rate — the motion estimate
    oq = gyro.quaternions
    okeys = sorted(oq.keys())
    ots = np.array(okeys, dtype=np.float64)
    if ots.max() > 1e6:
        ots = ots / 1e6
    elif ots.max() > 1e4:
        ots = ots / 1e3
    oqs = np.asarray([oq[k].quaternion() for k in okeys], dtype=np.float64)
    odot = np.clip(np.abs(np.sum(oqs[1:] * oqs[:-1], axis=1)), 0.0, 1.0)
    ostep = np.degrees(2.0 * np.arccos(odot))
    odts = np.diff(ots)
    ow = np.where(odts > 0, ostep / np.maximum(odts, 1e-9), 0.0)  # deg/s
    if t is None:
        t, w = ots[1:], ow

    # smoothed correction quaternion per-sample step
    sq = gyro.smoothed_quaternions
    keys = sorted(sq.keys())
    ts = np.array(keys, dtype=np.float64)
    if ts.max() > 1e6:  # us -> s
        ts = ts / 1e6
    elif ts.max() > 1e4:  # ms -> s
        ts = ts / 1e3
    qs = np.asarray([sq[k].quaternion() for k in keys], dtype=np.float64)  # w,x,y,z
    raw_dot = np.sum(qs[1:] * qs[:-1], axis=1)
    dot = np.clip(np.abs(raw_dot), 0.0, 1.0)
    step_deg = np.degrees(2.0 * np.arccos(dot))
    dts = np.diff(ts)
    rate = np.where(dts > 0, step_deg / np.maximum(dts, 1e-9), 0.0)  # deg/s

    flips = np.where(raw_dot < 0)[0]
    print(f"correction quats: n={len(ts)}  sign-flips={len(flips)}")
    if len(flips):
        print(f"   flip times (s): {[round(float(ts[i+1]), 2) for i in flips[:15]]}")

    print("  t(s)   |w|med  |w|max   step/s_med  step/s_max")
    for w0 in np.arange(0.0, ts[-1] - 1.0, 1.0):
        left = "   ----     ----"
        if t is not None:
            msk = (t >= w0) & (t < w0 + 1.0)
            if msk.sum() > 3:
                left = f"{np.median(w[msk]):7.2f} {w[msk].max():8.2f}"
        right = "     ----      ----"
        idx = np.where((ts[1:] >= w0) & (ts[1:] < w0 + 1.0))[0]
        if len(idx) > 3:
            right = f"{np.median(rate[idx]):9.2f} {rate[idx].max():9.2f}"
        print(f"{w0:6.1f}  {left} {right}")


if __name__ == "__main__":
    diag(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "clip")
