"""Reverse-engineer the per-frame correction applied by a stabilized output.

For each sampled frame pair (input frame i, output frame i), estimate the
homography via feature matching + RANSAC, decompose it into rotation angle
and zoom (scale). This shows exactly what correction a reference stabilizer
(Gyroflow) applied per frame, to compare against our own output.

Usage: python tests/reverse_engineer.py <input> <output> [step]
"""
import sys

import av
import cv2
import numpy as np


def decode_frames(path, step, max_n=120):
    c = av.open(path)
    s = c.streams.video[0]
    s.thread_type = "AUTO"
    frames = []
    i = 0
    for p in c.demux(s):
        if p.dts is None:
            continue
        for f in p.decode():
            if i % step == 0:
                frames.append(f.to_ndarray(format="gray"))
            i += 1
            if len(frames) >= max_n:
                break
        if len(frames) >= max_n:
            break
    c.close()
    return frames


def frame_transform(a, b):
    """Estimate homography a->b via ECC on reduced frames (robust for
    near-pure rotation+scale)."""
    h, w = a.shape
    sz = (w // 3, h // 3)
    a2 = cv2.resize(a, sz, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    b2 = cv2.resize(b, sz, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    warp = np.eye(3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)
    try:
        _, warp = cv2.findTransformECC(a2, b2, warp, cv2.MOTION_HOMOGRAPHY, crit, None, 5)
    except cv2.error:
        return None
    return warp.astype(np.float64)


def decompose(H):
    """Rough decomposition of a homography: zoom = 1/scale, roll angle."""
    # Assume H ≈ s * K R K^-1 for image-center K: translation part absorbs c.
    # Simple metrics: determinant-based area scale + rotation of the upper 2x2
    # after removing scale.
    area_scale = np.linalg.det(H[:2, :2])
    zoom = 1.0 / np.sqrt(abs(area_scale))
    U, S, Vt = np.linalg.svd(H[:2, :2])
    R2 = U @ Vt
    angle = np.degrees(np.arctan2(R2[1, 0], R2[0, 0]))
    return zoom, angle, H[0, 2], H[1, 2]


def main():
    inp, out = sys.argv[1], sys.argv[2]
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    A = decode_frames(inp, step)
    B = decode_frames(out, step)
    n = min(len(A), len(B))
    print(f"frames: {n} (step {step})")
    print("idx    zoom     roll_deg   tx(px)    ty(px)")
    zooms, angles = [], []
    for i in range(n):
        H = frame_transform(A[i], B[i])
        if H is None:
            print(f"{i*step:4d}  ---")
            continue
        zoom, ang, tx, ty = decompose(H)
        zooms.append(zoom)
        angles.append(ang)
        print(f"{i*step:4d}  {zoom:6.3f}   {ang:+7.2f}   {tx:+7.1f}  {ty:+7.1f}")
    if zooms:
        z = np.array(zooms)
        a = np.array(angles)
        print(f"\nzoom: mean {z.mean():.3f} std {z.std():.3f} min {z.min():.3f} max {z.max():.3f}")
        print(f"roll: mean {a.mean():+.2f} std {a.std():.2f}")
        # per-frame zoom variation = breathing
        print(f"per-frame |dz|: median {np.median(np.abs(np.diff(z))):.5f}")


if __name__ == "__main__":
    main()
