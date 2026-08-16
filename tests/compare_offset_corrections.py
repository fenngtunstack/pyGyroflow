# -*- coding: utf-8 -*-
"""Compare per-frame transform quats for different sync offsets (no render).

If a small offset makes the correction trajectory rough, the bug is in
lookup/smoothing, not in the optical pipeline.
"""

from __future__ import annotations

import sys

import numpy as np

from pygyroflow.manager import StabilizationManager

VIDEO = "../DJI_20260507160359_0005_D.MP4"


def correction_series(offset_ms):
    m = StabilizationManager()
    m.load_video(VIDEO)
    m.load_gyro_data(VIDEO)
    m.smoothing.current().set_parameter("smoothness", 0.5)
    m.synchronize()
    if offset_ms is not None:
        m.gyro.clear_offsets()
        m.gyro.set_offset(0, offset_ms)
    m.recompute_blocking()
    cp = m._build_compute_params()

    fps = m.params.fps
    out = []
    for i in range(int(6.0 * fps), int(16.0 * fps)):
        ts_ms = i * 1000.0 / fps
        tr = m.get_frame_transform(ts_ms, i, compute_params=cp)
        out.append((ts_ms, np.asarray(tr.matrices, dtype=np.float64)))
    return out


def main() -> int:
    series = {}
    for off in (0.0, 20.0):
        rows = correction_series(off)
        mats = np.stack([r[1] for r in rows])  # (F, N, 14)
        series[off] = mats
        print(f"offset={off}ms  mats={mats.shape}  fov_col13 med={np.median(mats[:,0,13]) if mats.shape[-1]>13 else 'n/a'}")
    d = np.abs(series[20.0] - series[0.0])
    print(f"\n|max diff| per frame: med={np.median(d.max(axis=(1,2))):.6f}  max={d.max(axis=(1,2)).max():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
