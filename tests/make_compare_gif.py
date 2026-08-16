# -*- coding: utf-8 -*-
"""Build a 3-way comparison GIF: input vs official Gyroflow vs pyGyroFlow.

Crops the same 7-12s window from all three videos, stacks them vertically
with labels, and encodes an optimized palette GIF.
"""

from __future__ import annotations

import subprocess
import sys

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

T0, T1 = 7.0, 12.0
FPS = 15
W = 640          # per-panel width
MAX_H = 360      # per-panel height (1920x1080 -> 640x360)
FONT = r"C:\Windows\Fonts\arial.ttf"

PANELS = [
    ("INPUT (raw)", r"..\DJI_20260507160359_0005_D.MP4"),
    ("Gyroflow official", r"..\DJI_20260507160359_0005_D_stabilized.mp4"),
    ("pyGyroFlow (ours)", r"..\final3_dji.mp4"),
]

LABEL_BG = (0, 0, 0)
LABEL_FG = (255, 255, 255)


def frames_from(path: str):
    c = av.open(path)
    s = c.streams.video[0]
    s.thread_type = "AUTO"
    src_fps = float(s.average_rate or 30.0)
    step = src_fps / FPS  # sample ~FPS frames per second of source
    i = 0
    next_pick = 0.0
    for packet in c.demux(s):
        if packet.dts is None:
            continue
        for f in packet.decode():
            t = i / src_fps
            if T0 <= t < T1 and i >= next_pick:
                img = f.to_ndarray(format="rgb24")
                yield t, img
                next_pick = i + step
            i += 1
            if t >= T1:
                return
    c.close()


def main() -> int:
    font = ImageFont.truetype(FONT, 18)

    # collect synced frames: one per panel per pick index
    streams = [list(frames_from(p)) for _, p in PANELS]
    n = min(len(s) for s in streams)
    print(f"frames per panel: {[len(s) for s in streams]} -> using {n}")
    if n == 0:
        print("no frames")
        return 1

    bar_h = 26
    out_frames = []
    for k in range(n):
        panels = []
        for (label, _), st in zip(PANELS, streams):
            t, img = st[k]
            im = Image.fromarray(img).resize((W, MAX_H), Image.LANCZOS)
            d = ImageDraw.Draw(im)
            # label bar overlay at top-left
            tw = d.textlength(label, font=font)
            d.rectangle([0, 0, tw + 14, bar_h], fill=LABEL_BG)
            d.text((7, 4), label, font=font, fill=LABEL_FG)
            # timestamp bottom-right
            ts = f"{t:.2f}s"
            tw2 = d.textlength(ts, font=font)
            d.rectangle([W - tw2 - 14, MAX_H - bar_h + 2, W, MAX_H], fill=LABEL_BG)
            d.text((W - tw2 - 7, MAX_H - bar_h + 6), ts, font=font, fill=LABEL_FG)
            panels.append(im)
        canvas = Image.new("RGB", (W, MAX_H * len(panels) + 2 * (len(panels) - 1)), (30, 30, 30))
        for j, p in enumerate(panels):
            canvas.paste(p, (0, j * (MAX_H + 2)))
        out_frames.append(canvas)

    out = "../compare_dji_7s-12s.gif"
    # palette-optimized GIF via PIL adaptive palette, re-used across frames for
    # smaller size and no flicker
    pal_img = Image.new("P", (1, 1))
    out_frames[0].quantize(colors=256, method=Image.MEDIANCUT).palette and None
    first = out_frames[0].quantize(colors=256, method=Image.MEDIANCUT)
    first.save(
        out,
        save_all=True,
        append_images=[f.quantize(colors=256, method=Image.MEDIANCUT, palette=first) for f in out_frames[1:]],
        duration=int(1000 / FPS),
        loop=0,
        optimize=True,
    )
    print("saved", out, "frames:", len(out_frames))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
