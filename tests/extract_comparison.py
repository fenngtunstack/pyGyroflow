"""Extract side-by-side comparison strips from input and stabilized videos.

Pulls frames at several timestamps from each video, stacks them
vertically (input on top, output below), and writes one PNG per
timestamp. Lets a human eyeball cropping, distortion correction, and
residual jitter edges.

Usage:
    python tests/extract_comparison.py input.mp4 output.mp4 out_dir [t1 t2 ...]
"""

from __future__ import annotations

import os
import sys

import av
import cv2
import numpy as np


def read_frame_at(path: str, target_ts_s: float):
    """Decode the frame closest to target_ts_s (None if video too short)."""
    c = av.open(path)
    s = c.streams.video[0]
    s.thread_type = "AUTO"
    dur = float(s.duration * s.time_base) if s.duration else 0.0
    if target_ts_s > dur:
        c.close()
        return None, dur
    # seek by seconds then take the next decodable frame
    c.seek(int(target_ts_s * 1_000_000))
    frame = None
    for packet in c.demux(s):
        for f in packet.decode():
            frame = f
            break
        if frame is not None:
            break
    img = frame.to_ndarray(format="rgb24") if frame is not None else None
    c.close()
    return img, dur


def main(argv: list[str]) -> int:
    if len(argv) < 4:
        print(__doc__)
        return 1
    inp, outp, out_dir = argv[1], argv[2], argv[3]
    times = [float(x) for x in argv[4:]] or [1.0, 5.0, 10.0]

    os.makedirs(out_dir, exist_ok=True)
    for t in times:
        a, dur_a = read_frame_at(inp, t)
        b, _ = read_frame_at(outp, t)
        if a is None or b is None:
            print(f"t={t}s out of range (input dur {dur_a:.1f}s), skipped")
            continue
        h = 360
        def resize(img):
            w = int(img.shape[1] * h / img.shape[0])
            return cv2.resize(img, (w, h))
        a, b = resize(a), resize(b)
        w = max(a.shape[1], b.shape[1])
        canvas = np.zeros((h * 2 + 8, w, 3), dtype=np.uint8)
        canvas[:h, : a.shape[1]] = a
        canvas[h + 8 :, : b.shape[1]] = b
        # red separator
        canvas[h : h + 8, :] = (60, 60, 220)
        name = os.path.join(out_dir, f"cmp_{t:06.1f}s.png")
        cv2.imwrite(name, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        print("wrote", name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
