# -*- coding: utf-8 -*-
"""Multi-platform 2-way comparison GIF: input vs stabilized, shakiest 3s window.

Generalizes make_compare_gif_gopro.py: works for any aspect ratio, picks the
shakiest window of the INPUT via phase correlation (pixel_jitter), stacks the
two panels vertically with labels + timestamps, and keeps the output small
enough for WeChat (<10MB). A manual window can be forced when the globally
shakiest segment is not gyro-correctable shake (e.g. OIS residual).

Usage:
    python tests/make_compare_gif_multi.py NAME IN.mp4 STAB.mp4 [-o out.gif]
    python tests/make_compare_gif_multi.py NAME IN.mp4 STAB.mp4 --window 9.0 11.5
"""

from __future__ import annotations

import argparse
import os
import sys

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
from pixel_jitter import displacement_series  # noqa: E402

FPS = 8
WIN_S = 3.0
COLORS = 192
BAR = 18
LABELS = ("INPUT (raw)", "pyGyroFlow")


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for cand in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ):
        if os.path.exists(cand):
            return ImageFont.truetype(cand, size)
    return ImageFont.load_default()


def shakiest_window(path: str, win: float = WIN_S) -> tuple[float, float]:
    """(t0, mean |disp|) of the input's worst fully-contained `win` window."""
    t, d = displacement_series(path)
    m = np.hypot(d[:, 0], d[:, 1])
    ends = np.searchsorted(t, t + win, side="right")
    csum = np.concatenate([[0.0], np.cumsum(m)])
    counts = np.maximum(ends - np.arange(len(m)), 1)
    means = (csum[ends] - csum[: len(m)]) / counts
    means[counts < 6] = -1.0          # too little data -> invalid start
    means[t + win > t[-1]] = -1.0     # window must fit inside the video
    k = int(np.argmax(means))
    return float(t[k]), float(means[k])


def frames_from(path: str, t0: float, t1: float):
    c = av.open(path)
    s = c.streams.video[0]
    s.thread_type = "AUTO"
    src_fps = float(s.average_rate or 30.0)
    step = src_fps / FPS
    i = 0
    next_pick = 0.0
    for packet in c.demux(s):
        if packet.dts is None:
            continue
        for f in packet.decode():
            t = i / src_fps
            if t0 <= t < t1 and i >= next_pick:
                yield t, f.to_ndarray(format="rgb24")
                next_pick = i + step
            i += 1
            if t >= t1:
                c.close()
                return
    c.close()


def build_gif(name: str, src: str, stab: str, out_path: str,
              window: tuple[float, float] | None, width: int) -> None:
    if window is not None:
        t0, t1 = window
        print(f"[{name}] manual window {t0:.2f}-{t1:.2f}s")
    else:
        t0, score = shakiest_window(src)
        t1 = t0 + WIN_S
        print(f"[{name}] shakiest window {t0:.2f}-{t1:.2f}s (mean |disp| {score:.2f}px)")

    font = load_font(13)
    streams = [list(frames_from(p, t0, t1)) for p in (src, stab)]
    n = min(len(s) for s in streams)
    if n < 8:
        print(f"[{name}] too few frames {[len(s) for s in streams]}, skip")
        return

    fh, fw = streams[0][0][1].shape[:2]
    ph = int(round(width * fh / fw)) & ~1

    raw = []
    for k in range(n):
        panels = []
        for label, st in zip(LABELS, streams):
            t, img = st[k]
            im = Image.fromarray(img).resize((width, ph), Image.LANCZOS)
            d = ImageDraw.Draw(im)
            tw = d.textlength(label, font=font)
            d.rectangle([0, 0, tw + 8, BAR], fill=(0, 0, 0))
            d.text((4, 2), label, font=font, fill=(255, 255, 255))
            ts = f"{t:.1f}s"
            tw2 = d.textlength(ts, font=font)
            d.rectangle([width - tw2 - 8, ph - BAR + 2, width, ph], fill=(0, 0, 0))
            d.text((width - tw2 - 4, ph - BAR + 3), ts, font=font, fill=(255, 255, 255))
            panels.append(im)
        canvas = Image.new("RGB", (width, ph * 2 + 2), (25, 25, 25))
        for j, p in enumerate(panels):
            canvas.paste(p, (0, j * (ph + 2)))
        raw.append(canvas)

    sample = raw[:: max(1, len(raw) // 8)]
    strip = Image.new("RGB", (width, sum(f.height for f in sample)))
    y = 0
    for f in sample:
        strip.paste(f, (0, y))
        y += f.height
    pal = strip.quantize(colors=COLORS, method=Image.MEDIANCUT)
    q = [f.quantize(colors=COLORS, method=Image.MEDIANCUT, palette=pal, dither=Image.Dither.NONE)
         for f in raw]

    q[0].save(out_path, save_all=True, append_images=q[1:],
              duration=int(1000 / FPS), loop=0, optimize=True)
    print(f"[{name}] saved {out_path} {os.path.getsize(out_path)/1e6:.2f} MB, {n} frames, {width}x{ph*2+2}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("name", help="short slug used in the default output filename")
    ap.add_argument("source", help="input (raw) video")
    ap.add_argument("stabilized", help="stabilized output video")
    ap.add_argument("-o", "--out", help="output .gif path (default: compare_<name>.gif)")
    ap.add_argument("--window", nargs=2, type=float, metavar=("T0", "T1"),
                    help="force [T0, T1) instead of the auto shakiest window")
    ap.add_argument("--width", type=int, default=424, help="panel width in px (default 424)")
    args = ap.parse_args()

    out = args.out or f"compare_{args.name}.gif"
    build_gif(args.name, args.source, args.stabilized, out,
              tuple(args.window) if args.window else None, args.width)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
