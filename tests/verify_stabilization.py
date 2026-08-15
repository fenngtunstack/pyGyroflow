"""Stabilization quality verification on real videos (manual tool).

Estimates the per-frame-pair camera angular velocity of a video using the
project's own optical flow (DIS) + essential-matrix pose estimation, then
reports jitter metrics. Run on the raw input and on stabilized outputs and
compare: good stabilization keeps the slow intended motion but removes the
high-frequency component.

Metrics (omega = |angular velocity| in deg/s, one value per frame pair):
  mean_speed  mean |omega|              -- total motion (mostly unchanged)
  jitter      mean |omega[i+1]-omega[i]| -- high-frequency energy (should drop)
  jitter_rel  jitter / mean_speed        -- scale-free quality number

Usage:
    python tests/verify_stabilization.py video1.mp4 [video2.mp4 ...]
"""

from __future__ import annotations

import sys

import av
import cv2
import numpy as np

from pygyroflow.synchronization import PoseEstimator


def angular_speed_series(path: str, max_width: int = 480) -> tuple[np.ndarray, float]:
    """Return (|omega| per consecutive frame pair in deg/s, fps)."""
    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    fps = float(stream.average_rate or 30.0)
    tb = float(stream.time_base or 1)

    scale = min(1.0, max_width / max(1, stream.width))
    out_w = max(2, int(stream.width * scale) & ~1)
    out_h = max(2, int(stream.height * scale) & ~1)

    est = PoseEstimator()
    est.set_fps(fps)
    est.set_optical_flow_method(2)  # DIS

    idx = 0

    def emit(frame):
        nonlocal idx
        gray = frame.to_ndarray(format="gray")
        if scale < 1.0:
            gray = cv2.resize(gray, (out_w, out_h), interpolation=cv2.INTER_AREA)
        ts_us = (
            int(round(float(frame.pts) * tb * 1e6))
            if frame.pts is not None
            else int(round(idx * 1e6 / fps))
        )
        est.feed_frame(idx, ts_us, gray)
        idx += 1

    for packet in container.demux(stream):
        if packet.dts is None:
            continue
        for frame in packet.decode():
            emit(frame)
    for frame in stream.decode():
        emit(frame)
    container.close()

    est.process_all()
    rotations = est.get_visual_rotations()
    speeds = np.array(
        [float(np.linalg.norm(w)) for _ts, w in rotations], dtype=np.float64
    )
    return speeds, fps


def metrics(speeds: np.ndarray) -> dict:
    """Robust jitter metrics.

    Essential-matrix pose estimation degenerates on low-texture/blurred
    frames and can report rotations up to 180 deg; plain means are
    destroyed by those outliers, so medians and trimmed means are used.
    """
    if len(speeds) < 3:
        return {"frames": len(speeds)}
    lo, hi = np.percentile(speeds, [5, 95])
    trimmed = speeds[(speeds >= lo) & (speeds <= hi)]
    jit = np.abs(np.diff(speeds))
    jit_trim = jit[(jit >= np.percentile(jit, 5)) & (jit <= np.percentile(jit, 95))]
    return {
        "frames": len(speeds),
        "mean_speed": float(np.mean(trimmed)),
        "median_speed": float(np.median(speeds)),
        "jitter": float(np.mean(jit_trim)),
        "median_jitter": float(np.median(jit)),
        "std": float(np.std(trimmed)),
        "outliers": int(np.sum(speeds > 90.0)),
    }


def frame_diff_series(path: str, max_width: int = 480, n: int = 250) -> np.ndarray:
    """Mean absolute consecutive-frame gray difference (raw pixel motion).

    Immune to the pose-estimator outliers that pollute the angular-velocity
    metrics; a stabilized video must show a clear drop here.
    """
    import cv2

    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    scale = min(1.0, max_width / max(1, stream.width))
    w = int(stream.width * scale) & ~1
    h = int(stream.height * scale) & ~1

    diffs: list[float] = []
    prev = None
    idx = 0
    for packet in container.demux(stream):
        if packet.dts is None:
            continue
        for frame in packet.decode():
            g = frame.to_ndarray(format="gray")
            if scale < 1.0:
                g = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
            if prev is not None:
                diffs.append(float(np.mean(np.abs(g.astype(np.int32) - prev))))
            prev = g.astype(np.int32)
            idx += 1
            if idx >= n:
                break
        if idx >= n:
            break
    container.close()
    return np.array(diffs)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1

    results = []
    for path in argv[1:]:
        speeds, fps = angular_speed_series(path)
        m = metrics(speeds)
        fd = frame_diff_series(path)
        m["pixdiff"] = float(np.median(fd)) if len(fd) else 0.0
        results.append((path, m))
        print(
            f"{path.split('/')[-1]:44s} n={m.get('frames', 0):4d} "
            f"med|w|={m.get('median_speed', 0):7.2f} "
            f"medjit={m.get('median_jitter', 0):6.3f} "
            f"pixdiff={m['pixdiff']:6.2f} "
            f"outl={m.get('outliers', 0):3d}"
        )

    # Pairwise comparisons against the first file (assumed raw input)
    if len(results) >= 2:
        base = results[0][1]
        print()
        for path, m in results[1:]:
            if "jitter" not in m or "jitter" not in base:
                continue
            gain = base["jitter"] / max(m["jitter"], 1e-9)
            gain_med = base["median_jitter"] / max(m["median_jitter"], 1e-9)
            gain_px = base["pixdiff"] / max(m["pixdiff"], 1e-9)
            print(
                f"  vs input: {path.split('/')[-1]:40s} "
                f"jitter {gain:5.2f}x  med-jitter {gain_med:5.2f}x  "
                f"pixel-diff {gain_px:5.2f}x"
                f"  ({base['pixdiff']:.2f} -> {m['pixdiff']:.2f})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
