# -*- coding: utf-8 -*-
"""Per-axis residual jitter: ours vs official, camera-frame decomposition.

Feeds outputs through the pose estimator, keeps angular-velocity VECTORS,
rotates them into the camera frame via the gyro orientation stream, and
reports per-axis median jitter in the fast segment. Roll-heavy excess =>
horizon-lock gap; yaw-heavy => sync/RS gap.
"""

from __future__ import annotations

import sys

import numpy as np

from pygyroflow.manager import StabilizationManager

_CF_GT: np.ndarray | None = None
_CF_R: np.ndarray | None = None


def quat_to_rot(qwxyz):
    w, x, y, z = qwxyz[:, 0], qwxyz[:, 1], qwxyz[:, 2], qwxyz[:, 3]
    R = np.empty((len(qwxyz), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def load_orientation(video: str) -> None:
    m = StabilizationManager()
    m.load_video(video)
    m.load_gyro_data(video)
    oq = m.gyro.quaternions
    keys = sorted(oq.keys())
    globals()["_CF_GT"] = np.array(keys, dtype=np.float64) / 1e6
    globals()["_CF_R"] = quat_to_rot(
        np.array([oq[k].quaternion() for k in keys]))


def camera_frame_w(t, w):
    """Rotate world-frame angular velocity into the camera frame."""
    idx = np.clip(np.searchsorted(_CF_GT, t), 0, len(_CF_GT) - 1)
    Rc = _CF_R[idx]
    return np.einsum("nij,nj->ni", np.transpose(Rc, (0, 2, 1)), w)


def visual_series(path: str):
    import av
    import cv2
    from pygyroflow.synchronization import PoseEstimator

    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    fps = float(stream.average_rate or 30.0)
    tb = float(stream.time_base or 1)
    scale = min(1.0, 480 / max(1, stream.width))
    out_w = max(2, int(stream.width * scale) & ~1)
    out_h = max(2, int(stream.height * scale) & ~1)

    est = PoseEstimator()
    est.set_fps(fps)
    est.set_optical_flow_method(2)
    idx = 0
    for packet in container.demux(stream):
        if packet.dts is None:
            continue
        for frame in packet.decode():
            gray = frame.to_ndarray(format="gray")
            if scale < 1.0:
                gray = cv2.resize(gray, (out_w, out_h), interpolation=cv2.INTER_AREA)
            t_us = (int(round(float(frame.pts) * tb * 1e6))
                    if frame.pts is not None else int(round(idx * 1e6 / fps)))
            est.feed_frame(idx, t_us, gray)
            idx += 1
    container.close()
    est.process_all()
    rots = est.get_visual_rotations()
    t = np.array([r[0] / 1e6 for r in rots])
    w = np.array([r[1] for r in rots], dtype=np.float64)
    return t, w


def main() -> int:
    inp = "../DJI_20260507160359_0005_D.MP4"
    ours = sys.argv[1] if len(sys.argv) > 1 else "../final2_dji_gpu.mp4"
    official = "../DJI_20260507160359_0005_D_stabilized.mp4"

    load_orientation(inp)

    print("             pitch(x)  yaw(y)   roll(z)  [median |dw| per axis, deg/s, 8-22s]")
    for name, path in (("ours    ", ours), ("official", official), ("input   ", inp)):
        t, w = visual_series(path)
        wc = camera_frame_w(t, w)
        jit = np.abs(np.diff(wc, axis=0))
        msk = (t[:-1] >= 8.0) & (t[:-1] <= 22.0)
        cells = []
        for ax in range(3):
            col = jit[msk, ax]
            col = col[col < 45.0]
            cells.append(f"{np.median(col):8.3f}")
        print(f"{name} {' '.join(cells)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
