# -*- coding: utf-8 -*-
"""Reverse-engineer the official export's applied correction.

Registers input frames against the official stabilized output (ORB +
essential-matrix rotation) to estimate the per-frame rotation the official
Gyroflow applied, then compares it with the correction quaternions our
pipeline computes: angular-rate smoothness and relative phase.

Usage:
    python tests/reverse_official.py <input> <official> [ours_output]
"""

from __future__ import annotations

import sys

import av
import cv2
import numpy as np


def frame_stream(path: str, max_width: int = 480):
    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    for packet in container.demux(stream):
        if packet.dts is None:
            continue
        for frame in packet.decode():
            gray = frame.to_ndarray(format="gray")
            scale = min(1.0, max_width / max(1, gray.shape[1]))
            if scale < 1.0:
                gray = cv2.resize(gray, (int(gray.shape[1] * scale) & ~1,
                                         int(gray.shape[0] * scale) & ~1),
                                  interpolation=cv2.INTER_AREA)
            yield gray
    container.close()


def register_rotation(gray_a, gray_b, K):
    """Rotation (rodrigues vector, norm=deg) mapping content of a into b."""
    orb = cv2.ORB_create(nfeatures=2000)
    ka, da = orb.detectAndCompute(gray_a, None)
    kb, db = orb.detectAndCompute(gray_b, None)
    if da is None or db is None or len(ka) < 12 or len(kb) < 12:
        return None, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(da, db)
    if len(matches) < 12:
        return None, len(matches)
    matches = sorted(matches, key=lambda m: m.distance)[:150]
    pa = np.float32([ka[m.queryIdx].pt for m in matches])
    pb = np.float32([kb[m.trainIdx].pt for m in matches])
    E, mask = cv2.findEssentialMat(pa, pb, K, method=cv2.RANSAC,
                                   prob=0.999, threshold=2.0)
    if E is None or mask is None or mask.sum() < 10:
        return None, len(matches)
    _, R, t, _ = cv2.recoverPose(E, pa[mask.ravel() == 1], pb[mask.ravel() == 1], K)
    rod, _ = cv2.Rodrigues(R)
    return rod.ravel(), int(mask.sum())


def main() -> int:
    inp, off = sys.argv[1], sys.argv[2]
    K = np.array([[350.0, 0, 240.0], [0, 350.0, 135.0], [0, 0, 1.0]])

    ga = frame_stream(inp)
    gb = frame_stream(off)
    rows = []
    for i, (fa, fb) in enumerate(zip(ga, gb)):
        if i % 2:  # every other frame is enough
            continue
        rod, n = register_rotation(fa, fb, K)
        if rod is not None:
            ang = np.degrees(np.linalg.norm(rod))
            axis = rod / max(np.linalg.norm(rod), 1e-12)
            rows.append((i / 30.0, ang, axis * ang))
        if i >= 700:
            break
    t = np.array([r[0] for r in rows])
    ang = np.array([r[1] for r in rows])
    vec = np.array([r[2] for r in rows])

    print(f"registered {len(rows)} frame pairs (of ~750)")
    # smoothness of official's correction magnitude
    dang = np.abs(np.diff(ang))
    print(f"official correction |R|: med={np.median(ang):.3f} deg  "
          f"d|R|/frame med={np.median(dang):.4f}")

    if len(sys.argv) > 3:
        gc = frame_stream(sys.argv[3])
        ga2 = frame_stream(inp)
        rows2 = []
        for i, (fa, fc) in enumerate(zip(ga2, gc)):
            if i % 2:
                continue
            rod, n = register_rotation(fa, fc, K)
            if rod is not None:
                ang2 = np.degrees(np.linalg.norm(rod))
                rows2.append((i / 30.0, ang2))
            if i >= 700:
                break
        ang2 = np.array([r[1] for r in rows2])
        dang2 = np.abs(np.diff(ang2))
        print(f"ours     correction |R|: med={np.median(ang2):.3f} deg  "
              f"d|R|/frame med={np.median(dang2):.4f}")

    # per-window print
    print("\n  t(s)   off|R|  off_d  ")
    for w0 in np.arange(0, t[-1] - 1, 1.0):
        m = (t >= w0) & (t < w0 + 1.0)
        m2 = (t[:-1] >= w0) & (t[:-1] < w0 + 1.0)
        print(f"{w0:6.1f}  {np.median(ang[m]) if m.sum() else float('nan'):7.3f} "
              f"{np.median(dang[m2]) if m2.sum() else float('nan'):7.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
