# -*- coding: utf-8 -*-
"""Dump the RS-sync cost landscape and per-window optima for a clip.

Reproduces manager.synchronize()'s exact track construction, then prints:
  - total cost at delays in [-400, +400] ms (1-D landscape)
  - optimum delay per 4-second window (drift check)

Usage:
    python tests/dump_rs_cost.py <video>
"""

from __future__ import annotations

import sys

import numpy as np

from pygyroflow.manager import StabilizationManager
from pygyroflow.synchronization import AutosyncProcess
from pygyroflow.synchronization.find_offset.rs_sync import RollingShutterSync


def main(video: str) -> int:
    m = StabilizationManager()
    m.load_video(video)
    m.load_gyro_data(video)

    frames = m._extract_gray_frames(video, int(sys.argv[2]) if len(sys.argv) > 2 else 200)
    height, width = frames[0][1].shape[:2]
    camera_matrix = m.lens.get_camera_matrix(size=(width, height))

    proc = AutosyncProcess(
        camera_matrix=camera_matrix,
        fps=m.params.fps,
        scaled_fps=m.params.get_scaled_fps(),
        of_method=2,
        pose_method=0,
        offset_method=2,
    )
    proc.pose_estimator.clear()
    proc._feed_and_estimate(frames, None)

    ordered = sorted(
        proc._pose_estimator.get_frame_results().values(),
        key=lambda f: f.frame_no,
    )
    print(f"pose frames: {len(ordered)}  quats: {len(m.gyro.quaternions)}")

    rs = RollingShutterSync(
        dict(m.gyro.quaternions),
        frame_readout_time_ms=m.params.frame_readout_time,
        fps=m.params.fps,
    )
    added = 0
    for a, b in zip(ordered, ordered[1:]):
        if a.prev_points is None or a.curr_points is None:
            continue
        if len(a.prev_points) < 2:
            continue
        rs.add_track_from_frames(
            a.timestamp_us,
            b.timestamp_us,
            a.prev_points,
            a.curr_points,
            height,
            camera_matrix=proc._pose_estimator_camera_matrix(),
        )
        added += 1
    print(f"tracks added: {added}")

    ts_all = [f.timestamp_us for f in ordered if f.timestamp_us > 0]
    t0, t1 = min(ts_all) / 1e6, max(ts_all) / 1e6

    print("\ndelay_landscape (total cost):")
    for delay_ms in np.arange(-400.0, 400.0 + 1e-9, 25.0):
        c = rs._compute_cost(delay_ms / 1000.0, t0, t1)
        bar = "#" * max(0, int(60 * (1.0 - c / 3.0)) if c < 3.0 else 0)
        print(f"  delay {delay_ms:7.1f} ms  cost={c:.5f}  {bar}")

    print("\nper-window optimum:")
    for w0 in np.arange(t0, t1 - 4.0, 4.0):
        best = (float("inf"), 0.0)
        for delay_ms in np.arange(-300.0, 300.0 + 1e-9, 5.0):
            c = rs._compute_cost(delay_ms / 1000.0, w0, w0 + 4.0)
            if c < best[0]:
                best = (c, delay_ms)
        print(f"  window {w0:6.2f}-{w0+4:6.2f}s  best delay {best[1]:7.1f} ms  cost {best[0]:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
