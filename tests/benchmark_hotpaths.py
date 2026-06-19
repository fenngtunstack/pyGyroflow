"""Micro-benchmarks for the hot paths flagged for vectorization.

Run manually (not collected by pytest):

    python tests/benchmark_hotpaths.py

Records baseline timings so optimization commits can show before/after.
Numbers are environment-dependent — use relative speedup, not absolute ms.
"""

from __future__ import annotations

import sys
import time

import numpy as np


def _time(fn, repeat: int = 3) -> float:
    """Median wall time (s) of fn over repeat runs."""
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]


def bench_frame_transform_global_shutter():
    """FrameTransform.at_timestamp, global shutter (num_rows=1)."""
    from pygyroflow.types.quaternion import Quat64
    from pygyroflow.stabilization import ComputeParams, FrameTransform

    n = 6000  # ~30s @ 200Hz IMU, like a real clip
    quats, smoothed = {}, {}
    for i in range(n):
        ts = i * 5000  # 200 Hz (us)
        ang = i * 0.0005
        quats[ts] = Quat64.from_euler_angles(0.0, 0.0, ang)
        smoothed[ts] = Quat64.from_euler_angles(0.0, 0.0, ang * 0.5)

    cp = ComputeParams(
        width=1920, height=1080, output_width=1920, output_height=1080,
        frame_count=900, scaled_fps=30.0, scaled_duration_ms=30000.0,
        quaternions=quats, smoothed_quaternions=smoothed,
        fovs=[1.0] * 900, fov_scale=1.0,
        camera_matrix=np.array([[1000., 0, 960], [0, 1000., 540], [0, 0, 1]]),
        distortion_coeffs=[0.0] * 12, frame_readout_time=0.0,
    )

    def run():
        for f in range(0, 900, 10):  # 90 frames
            FrameTransform.at_timestamp(cp, f * 1000.0 / 30.0, f)

    return _time(run)


def bench_frame_transform_rolling_shutter():
    """FrameTransform.at_timestamp WITH rolling shutter (the O(rows) loop)."""
    from pygyroflow.types.quaternion import Quat64
    from pygyroflow.stabilization import ComputeParams, FrameTransform

    n = 6000
    quats, smoothed = {}, {}
    for i in range(n):
        ts = i * 5000
        ang = i * 0.0005
        quats[ts] = Quat64.from_euler_angles(0.0, 0.0, ang)
        smoothed[ts] = Quat64.from_euler_angles(0.0, 0.0, ang * 0.5)

    cp = ComputeParams(
        width=1920, height=1080, output_width=1920, output_height=1080,
        frame_count=900, scaled_fps=30.0, scaled_duration_ms=30000.0,
        quaternions=quats, smoothed_quaternions=smoothed,
        fovs=[1.0] * 900, fov_scale=1.0,
        camera_matrix=np.array([[1000., 0, 960], [0, 1000., 540], [0, 0, 1]]),
        distortion_coeffs=[0.0] * 12,
        frame_readout_time=15.0,  # rolling shutter ON
    )

    def run():
        for f in range(0, 90, 10):  # 9 frames (RS is slow)
            FrameTransform.at_timestamp(cp, f * 1000.0 / 30.0, f)

    return _time(run)


def bench_default_algo_smoothing():
    """DefaultAlgo smoothing over a realistic clip."""
    from pygyroflow.types.quaternion import Quat64
    from pygyroflow.smoothing import Smoothing
    from pygyroflow.stabilization import ComputeParams

    n = 6000
    quats = {}
    for i in range(n):
        quats[i * 5000] = Quat64.from_euler_angles(0.0, 0.0, i * 0.0005)
    cp = ComputeParams(
        width=1920, height=1080, output_width=1920, output_height=1080,
        frame_count=900, scaled_fps=30.0, scaled_duration_ms=30000.0,
        quaternions=quats, smoothed_quaternions={},
        fovs=[1.0] * 900, fov_scale=1.0,
        camera_matrix=np.eye(3), distortion_coeffs=[0.0] * 12,
        frame_readout_time=0.0,
    )
    sm = Smoothing()

    def run():
        sm.smooth(quats, 30000.0, cp, org_quats=quats)

    return _time(run)


def bench_vqf():
    """VQF integration over a realistic clip."""
    from pygyroflow.imu_integration import VQFIntegrator
    from pygyroflow.types.time_types import TimeIMU

    n = 6000
    data = [
        TimeIMU(timestamp_ms=i * 5.0,
                gyro=np.array([0.1, -0.2, 0.05]),
                accl=np.array([0.0, 0.0, 9.8]),
                magn=None)
        for i in range(n)
    ]
    integ = VQFIntegrator()

    def run():
        integ.integrate(data, 30000.0)

    return _time(run)


def main():
    print("PyGyroFlow hot-path benchmarks (median of 3 runs)")
    print("=" * 60)
    benches = [
        ("frame_transform (global shutter, 90 frames)", bench_frame_transform_global_shutter),
        ("frame_transform (rolling shutter, 9 frames)", bench_frame_transform_rolling_shutter),
        ("default_algo smoothing (6000 samples)", bench_default_algo_smoothing),
        ("VQF integration (6000 samples)", bench_vqf),
    ]
    results = {}
    for name, fn in benches:
        try:
            t = fn()
            results[name] = t
            print(f"  {name:50s} {t*1000:8.1f} ms")
        except Exception as exc:
            print(f"  {name:50s} ERROR: {exc}")
            results[name] = None
    return results


if __name__ == "__main__":
    main()
