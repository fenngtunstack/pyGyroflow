# -*- coding: utf-8 -*-
"""2-way GoPro comparison GIF: input vs pyGyroFlow, shakiest 3s window."""

from __future__ import annotations

import av
from PIL import Image, ImageDraw, ImageFont

T0, T1 = 2.5, 5.5
FPS = 8
W = 424
FONT = r"C:\Windows\Fonts\arial.ttf"

PANELS = [
    ("INPUT (raw)", r"..\GX010045.MP4"),
    ("pyGyroFlow (ours)", r"..\final3_gopro.mp4"),
]


def frames_from(path: str):
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
            if T0 <= t < T1 and i >= next_pick:
                yield t, f.to_ndarray(format="rgb24")
                next_pick = i + step
            i += 1
            if t >= T1:
                return
    c.close()


def main() -> int:
    font = ImageFont.truetype(FONT, 14)
    streams = [list(frames_from(p)) for _, p in PANELS]
    n = min(len(s) for s in streams)
    print("frames:", [len(s) for s in streams], "->", n)

    ph = int(round(W * 1120 / 1280))  # GoPro 8:7 source aspect
    bar = 20
    raw = []
    for k in range(n):
        panels = []
        for (label, _), st in zip(PANELS, streams):
            t, img = st[k]
            im = Image.fromarray(img).resize((W, ph), Image.LANCZOS)
            d = ImageDraw.Draw(im)
            tw = d.textlength(label, font=font)
            d.rectangle([0, 0, tw + 8, bar], fill=(0, 0, 0))
            d.text((4, 2), label, font=font, fill=(255, 255, 255))
            ts = f"{t:.1f}s"
            tw2 = d.textlength(ts, font=font)
            d.rectangle([W - tw2 - 8, ph - bar + 2, W, ph], fill=(0, 0, 0))
            d.text((W - tw2 - 4, ph - bar + 4), ts, font=font, fill=(255, 255, 255))
            panels.append(im)
        canvas = Image.new("RGB", (W, ph * 2 + 2), (25, 25, 25))
        for j, p in enumerate(panels):
            canvas.paste(p, (0, j * (ph + 2)))
        raw.append(canvas)

    sample = raw[:: max(1, len(raw) // 8)]
    strip = Image.new("RGB", (W, sum(f.height for f in sample)))
    y = 0
    for f in sample:
        strip.paste(f, (0, y))
        y += f.height
    pal = strip.quantize(colors=192, method=Image.MEDIANCUT)
    q = [f.quantize(colors=192, method=Image.MEDIANCUT, palette=pal, dither=Image.Dither.NONE) for f in raw]

    out = "../compare_gopro_shakiest3s.gif"
    q[0].save(out, save_all=True, append_images=q[1:],
              duration=int(1000 / FPS), loop=0, optimize=True)
    import os
    print("saved", out, f"{os.path.getsize(out)/1e6:.2f} MB, {len(q)} frames, {W}x{ph*2+2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
