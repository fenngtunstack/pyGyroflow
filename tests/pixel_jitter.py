# -*- coding: utf-8 -*-
"""Pixel-space jitter via phase correlation (what the eye actually sees).

For consecutive frame pairs, estimates the dominant global translation
with phase correlation and reports |d(translation)| statistics per
1-second window for ours vs official.
"""

from __future__ import annotations

import sys

import av
import cv2
import numpy as np


def displacement_series(path: str, max_width: int = 480):
    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    scale = min(1.0, max_width / max(1, stream.width))
    out_w = max(2, int(stream.width * scale) & ~1)
    out_h = max(2, int(stream.height * scale) & ~1)

    prev = None
    t = []
    disp = []
    fps = float(stream.average_rate or 30.0)
    i = 0
    for packet in container.demux(stream):
        if packet.dts is None:
            continue
        for frame in packet.decode():
            gray = frame.to_ndarray(format="gray")
            if scale < 1.0:
                gray = cv2.resize(gray, (out_w, out_h), interpolation=cv2.INTER_AREA)
            if prev is not None:
                (dx, dy), _ = cv2.phaseCorrelate(
                    np.float64(prev), np.float64(gray))
                disp.append((dx, dy))
                t.append(i / fps)
            prev = gray
            i += 1
    container.close()
    return np.array(t), np.array(disp)


def main() -> int:
    files = [
        ("ours    ", sys.argv[1] if len(sys.argv) > 1 else "../final2_dji_gpu.mp4"),
        ("official", "../DJI_20260507160359_0005_D_stabilized.mp4"),
        ("input   ", "../DJI_20260507160359_0005_D.MP4"),
    ]
    series = {}
    for name, path in files:
        t, d = displacement_series(path)
        series[name] = (t, d)

    print("  t(s)   jit_ours  jit_off   jit_input    (median |d(disp)| px/frame, 480w)")
    t0, _ = series["ours    "]
    for w0 in np.arange(8.0, 23.0, 1.0):
        row = f"{w0:6.1f}"
        for name, _ in files:
            t, d = series[name]
            m = (t >= w0) & (t < w0 + 1.0)
            dd = np.abs(np.diff(d[m], axis=0)) if m.sum() > 2 else np.zeros((0, 2))
            mag = np.linalg.norm(dd, axis=1) if len(dd) else np.array([0.0])
            row += f"  {np.median(mag):8.3f}"
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
