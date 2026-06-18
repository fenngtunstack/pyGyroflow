"""Golden reference data generator for PyGyroFlow.

This script generates reference data by running the same algorithms in Python
with known inputs, producing JSON files that can be used for regression testing.

For true Rust-vs-Python golden comparison, use the Rust workspace's
msgyro-imu-integration and msgyro-smoothing crates directly:

    cargo run -p msgyro-imu-integration --bin generate-golden-data

This Python script provides synthetic reference data for development testing.
"""

import json
import math
import os
from pathlib import Path

import numpy as np

GOLDEN_DIR = Path(__file__).parent


def generate_imu_data():
    """Generate synthetic IMU data: constant rotation around Z at 50 deg/s, 200Hz, 5 seconds."""
    n_samples = 1000  # 5 seconds * 200 Hz
    sample_rate = 200.0
    duration_ms = n_samples / sample_rate * 1000.0

    imu_data = []
    for i in range(n_samples):
        t_ms = i / sample_rate * 1000.0
        # Constant rotation around Z axis at 50 deg/s
        # With some gentle pitch oscillation
        gyro = [5.0 * math.sin(2 * math.pi * 0.5 * t_ms / 1000.0), 0.0, 50.0]
        # Gravity pointing down
        accl = [0.0, 0.0, 9.81]
        imu_data.append({
            "timestamp_ms": t_ms,
            "gyro": gyro,
            "accl": accl,
            "magn": None,
        })

    data = {
        "description": "Synthetic IMU data: Z-axis rotation 50 deg/s + gentle pitch oscillation 5 deg/s @ 200Hz for 5s",
        "n_samples": n_samples,
        "sample_rate": sample_rate,
        "duration_ms": duration_ms,
        "samples": imu_data,
    }
    path = GOLDEN_DIR / "imu_data.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Generated {path} ({len(imu_data)} samples)")
    return data


def generate_imu_integration_references(imu_data):
    """Generate reference quaternion outputs for all integration methods."""
    from pygyroflow.types.time_types import TimeIMU
    from pygyroflow.imu_integration import (
        SimpleGyroIntegrator, SimpleGyroAccelIntegrator,
        MahonyIntegrator, MadgwickIntegrator, ComplementaryIntegrator,
    )

    samples = imu_data["samples"]
    duration_ms = imu_data["duration_ms"]
    imu_list = [
        TimeIMU(
            timestamp_ms=s["timestamp_ms"],
            gyro=np.array(s["gyro"]) if s["gyro"] else None,
            accl=np.array(s["accl"]) if s["accl"] else None,
            magn=np.array(s["magn"]) if s["magn"] else None,
        )
        for s in samples
    ]

    integrators = {
        "simple_gyro": SimpleGyroIntegrator(),
        "simple_gyro_accel": SimpleGyroAccelIntegrator(),
        "mahony": MahonyIntegrator(),
        "madgwick": MadgwickIntegrator(),
        "complementary": ComplementaryIntegrator(),
    }

    for name, integrator in integrators.items():
        result = integrator.integrate(imu_list, duration_ms)
        # Convert to serializable format: {timestamp_us: [w, x, y, z]}
        quats_dict = {}
        for ts, q in result.items():
            quats_dict[str(ts)] = q.quaternion().tolist()

        data = {
            "description": f"IMU integration reference: {name}",
            "integrator": name,
            "n_samples": len(imu_list),
            "duration_ms": duration_ms,
            "quaternions": quats_dict,
        }
        path = GOLDEN_DIR / f"imu_{name}.json"
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Generated {path} ({len(result)} quaternions)")


def generate_smoothing_references():
    """Generate reference smoothing outputs."""
    from pygyroflow.types.quaternion import Quat64
    from pygyroflow.smoothing import DefaultAlgo, PlainSmoothing
    from pygyroflow.stabilization import ComputeParams

    # Create a quaternion sequence: smooth rotation with some jitter
    quats = {}
    np.random.seed(42)
    for i in range(200):
        t_us = i * 10000  # 100Hz
        base_angle = i * 0.02  # steady rotation
        jitter = np.random.randn() * 0.005  # small random perturbation
        q = Quat64.from_euler_angles(0.0, jitter, base_angle)
        quats[t_us] = q

    cp = ComputeParams(
        width=1920, height=1080, output_width=1920, output_height=1080,
        frame_count=200, scaled_fps=100.0, scaled_duration_ms=2000.0,
        quaternions=quats, fovs=[1.0]*200, fov_scale=1.0,
    )

    for algo_name, algo in [("default_0.3", DefaultAlgo()), ("default_0.7", DefaultAlgo()), ("plain", PlainSmoothing())]:
        if algo_name == "default_0.3":
            algo.set_parameter("smoothness", 0.3)
        elif algo_name == "default_0.7":
            algo.set_parameter("smoothness", 0.7)
        else:
            algo.set_parameter("time_constant", 0.3)

        result = algo.smooth(quats, 2000.0, cp)
        quats_dict = {str(ts): q.quaternion().tolist() for ts, q in result.items()}

        data = {
            "description": f"Smoothing reference: {algo_name}",
            "algorithm": algo_name,
            "n_input": len(quats),
            "duration_ms": 2000.0,
            "input_quaternions": {str(ts): q.quaternion().tolist() for ts, q in quats.items()},
            "output_quaternions": quats_dict,
        }
        path = GOLDEN_DIR / f"smoothing_{algo_name}.json"
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Generated {path}")


def generate_distortion_references():
    """Generate reference distortion/undistortion test points."""
    from pygyroflow.stabilization.distortion_models import from_name
    from pygyroflow.types.kernel_params import KernelParams

    models_and_coeffs = {
        "opencv_fisheye": {"k1": (-0.1, 0.02, 0.001, 0.0), "k2": (0.0, 0.0, 0.0, 0.0), "k3": (0.0, 0.0, 0.0, 0.0)},
        "opencv_standard": {"k1": (-0.05, 0.01, 0.0, 0.0), "k2": (0.001, -0.001, 0.0, 0.0), "k3": (0.0, 0.0, 0.0, 0.0)},
        "poly3": {"k1": (0.05, 0.0, 0.0, 0.0), "k2": (0.0, 0.0, 0.0, 0.0), "k3": (0.0, 0.0, 0.0, 0.0)},
        "poly5": {"k1": (0.03, -0.01, 0.0, 0.0), "k2": (0.0, 0.0, 0.0, 0.0), "k3": (0.0, 0.0, 0.0, 0.0)},
        "ptlens": {"k1": (0.02, -0.01, 0.005, 0.0), "k2": (0.0, 0.0, 0.0, 0.0), "k3": (0.0, 0.0, 0.0, 0.0)},
    }

    test_points_3d = [
        (0.1, 0.2, 1.0),
        (0.3, 0.4, 1.0),
        (-0.2, 0.3, 1.0),
        (0.0, 0.5, 1.0),
        (0.5, 0.0, 1.0),
    ]

    for model_name, coeffs in models_and_coeffs.items():
        model = from_name(model_name)
        kp = KernelParams()
        kp.k1 = coeffs["k1"]
        kp.k2 = coeffs["k2"]
        kp.k3 = coeffs["k3"]

        results = []
        for x, y, z in test_points_3d:
            dx, dy = model.distort_point(x, y, z, kp)
            ux, uy = model.undistort_point(dx, dy, kp)
            results.append({
                "input_3d": [x, y, z],
                "distorted": [dx, dy],
                "undistorted": [ux, uy],
                "roundtrip_error": float(np.sqrt((x - ux)**2 + (y - uy)**2)),
            })

        data = {
            "description": f"Distortion model reference: {model_name}",
            "model": model_name,
            "coefficients": coeffs,
            "test_points": results,
        }
        path = GOLDEN_DIR / f"distortion_{model_name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Generated {path}")


def generate_frame_transform_references():
    """Generate reference frame transform data."""
    from pygyroflow.types.quaternion import Quat64
    from pygyroflow.stabilization import ComputeParams, FrameTransform

    # Simple scenario: camera rotating slowly around Y axis
    quats = {}
    smoothed = {}
    for i in range(60):  # 1 second at 60fps
        t_us = i * 16667  # ~60fps
        angle = i * 0.005  # slow rotation
        quats[t_us] = Quat64.from_euler_angles(0.0, 0.0, angle)
        # Smoothed: half the rotation
        smoothed[t_us] = Quat64.from_euler_angles(0.0, 0.0, angle * 0.5)

    camera_matrix = np.array([
        [1000.0, 0.0, 960.0],
        [0.0, 1000.0, 540.0],
        [0.0, 0.0, 1.0],
    ])

    cp = ComputeParams(
        width=1920, height=1080, output_width=1920, output_height=1080,
        frame_count=60, scaled_fps=60.0, scaled_duration_ms=1000.0,
        quaternions=quats, smoothed_quaternions=smoothed,
        fovs=[1.0]*60, fov_scale=1.0,
        camera_matrix=camera_matrix,
        distortion_coeffs=[0.0]*12,
        frame_readout_time=0.0,
    )

    transforms = []
    for frame_idx in [0, 15, 30, 45, 59]:
        timestamp_ms = frame_idx * 1000.0 / 60.0
        ft = FrameTransform.at_timestamp(cp, timestamp_ms, frame_idx)
        transforms.append({
            "frame": frame_idx,
            "timestamp_ms": timestamp_ms,
            "matrix_count": ft.matrices.shape[0],
            "matrices_0": ft.matrices[0].tolist() if ft.matrices.shape[0] > 0 else [],
            "fov": ft.fov,
        })

    data = {
        "description": "Frame transform reference: slow Y-axis rotation",
        "transforms": transforms,
    }
    path = GOLDEN_DIR / "frame_transform.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Generated {path}")


if __name__ == "__main__":
    print("Generating golden reference data...")
    print(f"Output directory: {GOLDEN_DIR}")
    print()

    imu_data = generate_imu_data()
    print()
    generate_imu_integration_references(imu_data)
    print()
    generate_smoothing_references()
    print()
    generate_distortion_references()
    print()
    generate_frame_transform_references()
    print()
    print("Done! All golden reference data generated.")
