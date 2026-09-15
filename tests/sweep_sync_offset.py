# -*- coding: utf-8 -*-
"""Sync-offset sweep on a short, pts-preserving clip.

Renders the same clip at several gyro sync offsets and reports pixel jitter
(displacement magnitude) for each, so the optimal offset can be picked
empirically when auto-sync is unreliable (short clips / weak signal).

Key property this relies on: the offset is applied at quaternion LOOKUP time
(``ts -= offset_at_video_timestamp(ts)``), so one load + recompute serves the
whole sweep — each offset only costs one render.

The clip must keep the source file's timestamps (``ffmpeg -ss T0 -to T1 -i
src.mp4 -map 0:v:0 -c copy -copyts clip.mp4``); render() feeds each frame's
real pts to the transform lookup, which then hits the correct gyro window.
DJI note: ffmpeg cannot remux the djmd data stream (demuxer reports codec
none) — cut video only and load telemetry from the full source file.

Usage:
    python tests/sweep_sync_offset.py SRC.mp4 CLIP.mp4 [--offsets -40,-20,0,40]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from pixel_jitter import displacement_series  # noqa: E402

log = logging.getLogger(__name__)


def mag_mid_p90(path: str) -> tuple[float, float]:
    """(all-frames p90, mid-half p90) of displacement magnitude |(dx, dy)|."""
    t, d = displacement_series(path)
    m = np.hypot(d[:, 0], d[:, 1])
    mid = (t > t[len(t) // 4]) & (t < t[3 * len(t) // 4])
    return float(np.percentile(m, 90)), float(np.percentile(m[mid], 90))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", help="full file: telemetry + gyro timeline come from here")
    ap.add_argument("clip", help="pts-preserving video-only cut to render")
    ap.add_argument("--offsets", default="-40,-20,-8,0,8,20,40",
                    help="comma-separated offsets in ms (default: -40..40 step-varied)")
    ap.add_argument("--size", default="1920x1080", help="sweep render output size (WxH)")
    ap.add_argument("--outdir", default="/tmp/pyg_offset_sweep", help="where sweep renders go")
    ap.add_argument("--smoothness", type=float, default=0.5)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    offsets = [float(x) for x in args.offsets.split(",")]
    w, h = (int(x) for x in args.size.lower().split("x"))
    os.makedirs(args.outdir, exist_ok=True)

    from pygyroflow.manager import StabilizationManager

    mgr = StabilizationManager()
    mgr.load_video(args.source)
    mgr.smoothing.current().set_parameter("smoothness", args.smoothness)
    mgr.recompute_blocking()

    a, b = mag_mid_p90(args.clip)
    print(f"input   : all p90 {a:.3f} | mid p90 {b:.3f}")

    results: dict[float, float] = {}
    for off in offsets:
        out = os.path.join(args.outdir, f"sweep_{off:+g}.mp4")
        mgr.gyro.set_offset(0, off)
        mgr.set_output_size(w, h)  # after recompute — recompute overwrites it
        mgr.render(args.clip, out, {"codec": "H.264/AVC", "bitrate": 0, "use_gpu": False, "audio": False})
        a, b = mag_mid_p90(out)
        results[off] = b
        print(f"offset {off:+7g}ms : all p90 {a:.3f} | mid p90 {b:.3f}")

    best = min(results, key=results.get)
    print(f"BEST offset: {best:+g}ms (mid p90 {results[best]:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
