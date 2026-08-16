# -*- coding: utf-8 -*-
"""Fine-grained sync-offset verification via cross-correlation.

Estimates visual angular velocity from the input video (optical flow +
essential matrix), gyro angular velocity from the parsed quaternions, then
scans offsets to maximize correlation. Compares the optimum with the
auto-sync result stored in GyroSource.offsets.
"""

from __future__ import annotations

import sys

import numpy as np

from pygyroflow.manager import StabilizationManager
from pygyroflow.synchronization import PoseEstimator
from tests.verify_stabilization import angular_speed_series


def gyro_speed_series(m) -> tuple[np.ndarray, np.ndarray]:
    oq = m.gyro.quaternions
    keys = sorted(oq.keys())
    ts = np.array(keys, dtype=np.float64) / 1e6
    qs = np.asarray([oq[k].quaternion() for k in keys], dtype=np.float64)
    dot = np.clip(np.abs(np.sum(qs[1:] * qs[:-1], axis=1)), 0.0, 1.0)
    step = np.degrees(2.0 * np.arccos(dot))
    dts = np.diff(ts)
    rate = np.where(dts > 0, step / np.maximum(dts, 1e-9), 0.0)
    return ts[1:], rate


def main(video: str) -> int:
    m = StabilizationManager()
    m.load_video(video)
    m.load_gyro_data(video)
    m.synchronize()
    offs = m.gyro.get_offsets()
    cur_off_ms = list(offs.values())[0] if offs else 0.0

    vt, vw = None, None
    speeds, fps = angular_speed_series(video, max_width=480)
    # rebuild timestamped series
    est_ts = np.arange(len(speeds), dtype=np.float64) / fps

    gt, gw = gyro_speed_series(m)

    # resample gyro speed onto video-frame timestamps for a set of offsets
    print(f"auto-sync offset: {cur_off_ms:.2f} ms")
    best = (-2.0, 0.0)
    rows = []
    for off_ms in np.arange(cur_off_ms - 120.0, cur_off_ms + 120.0 + 1e-9, 5.0):
        shifted = est_ts + off_ms / 1000.0
        g_at = np.interp(shifted, gt, gw)
        # correlation on the fast segment 6..23s only (where sync matters)
        msk = (est_ts >= 6.0) & (est_ts <= 23.0)
        a, b = vw if False else (speeds[msk], g_at[msk])
        a = np.clip(a, 0, 120)
        b = np.clip(b, 0, 120)
        c = float(np.corrcoef(a, b)[0, 1]) if len(a) > 10 else float("nan")
        rows.append((off_ms, c))
        if c == c and c > best[0]:
            best = (c, float(off_ms))
    for off_ms, c in rows[::4]:
        bar = "#" * int(max(0, (c - 0.5) * 100)) if c == c else ""
        print(f"  offset {off_ms:8.1f} ms  corr={c if c==c else 0:.4f}  {bar}")
    print(f"best corr={best[0]:.4f} at offset {best[1]:.1f} ms "
          f"(current {cur_off_ms:.1f} ms, delta {best[1]-cur_off_ms:+.1f} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
