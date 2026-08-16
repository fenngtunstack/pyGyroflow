# -*- coding: utf-8 -*-
"""Offset/smoothness/readout sweep, mirroring the CLI render path exactly.

Usage:
    python tests/sweep_bad_segment.py <video> <tag> <offset_ms|auto> <smoothness> [readout_ms]
"""

from __future__ import annotations

import sys

from pygyroflow.manager import StabilizationManager


def render_config(video: str, out: str, offset_ms, smoothness: float, readout_ms=None, rdir=None) -> None:
    m = StabilizationManager()
    m.load_video(video)
    m.load_gyro_data(video)
    if readout_ms is not None:
        m.set_frame_readout_time(readout_ms)
    if rdir is not None:
        from pygyroflow.types.enums import ReadoutDirection
        m.params.frame_readout_direction = ReadoutDirection(rdir)
    m.smoothing.current().set_parameter("smoothness", smoothness)
    m.synchronize()
    if offset_ms is not None:
        m.gyro.clear_offsets()
        m.gyro.set_offset(0, offset_ms)
    m.recompute_blocking()
    m.render(video, out, {"audio": False, "use_gpu": True, "codec": "H.264/AVC"})


def main() -> int:
    video, tag, off_arg, sm_arg = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    readout = float(sys.argv[5]) if len(sys.argv) > 5 else None
    rdir = int(sys.argv[6]) if len(sys.argv) > 6 else None
    offset = None if off_arg == "auto" else float(off_arg)
    out = f"../sweep_{tag}.mp4"
    print(f">>> {out}: offset={off_arg}ms smoothness={sm_arg} readout={readout} dir={rdir}")
    render_config(video, out, offset, sm_arg, readout, rdir)

    from tests.verify_stabilization import angular_speed_series
    from tests.verify_windows import window_metrics

    speeds, fps = angular_speed_series(out, max_width=480)
    for t, mj, ms, n in window_metrics(speeds, fps):
        print(f"   {t:6.1f}  {mj:7.3f}  {ms:8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
