"""Tests for FrameTransform computation."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pygyroflow.types.quaternion import Quat64
from pygyroflow.stabilization import ComputeParams, FrameTransform


def _make_test_params():
    """Create a ComputeParams with simple test data."""
    quats = {}
    smoothed = {}
    for i in range(100):
        angle = i * 0.01
        q = Quat64.from_euler_angles(0.0, 0.0, angle)
        ts = i * 10000  # 10ms apart = 100fps
        quats[ts] = q
        smoothed[ts] = q  # Same = no stabilization needed

    return ComputeParams(
        width=1920, height=1080, output_width=1920, output_height=1080,
        frame_count=100, scaled_fps=100.0, scaled_duration_ms=1000.0,
        quaternions=quats, smoothed_quaternions=smoothed,
        fovs=[1.0] * 100, fov_scale=1.0,
        camera_matrix=np.array([
            [1000.0, 0.0, 960.0],
            [0.0, 1000.0, 540.0],
            [0.0, 0.0, 1.0],
        ]),
    )


class TestFrameTransform:
    def test_creation(self):
        from pygyroflow.types.kernel_params import KernelParams
        matrices = np.zeros((1, 14), dtype=np.float32)
        kp = KernelParams()
        ft = FrameTransform(matrices=matrices, kernel_params=kp, fov=1.0)
        assert ft.fov == 1.0
        assert ft.matrices.shape == (1, 14)

    def test_at_timestamp_global_shutter(self):
        """Global shutter produces a single row of matrices."""
        params = _make_test_params()
        ft = FrameTransform.at_timestamp(params, timestamp_ms=500.0, frame=50)

        # Should have matrices shape (1, 14) for global shutter
        assert ft.matrices.ndim == 2
        assert ft.matrices.shape[1] == 14

        # Kernel params should have reasonable values
        assert ft.kernel_params.width == 1920
        assert ft.kernel_params.height == 1080
        assert ft.kernel_params.fov > 0.0

    def test_at_timestamp_with_rolling_shutter(self):
        """Rolling shutter produces per-row matrices."""
        params = _make_test_params()
        params.frame_readout_time = 0.03  # 30ms readout

        ft = FrameTransform.at_timestamp(params, timestamp_ms=500.0, frame=50)
        # Rolling shutter: should have height rows
        assert ft.matrices.shape[0] == 1080
        assert ft.matrices.shape[1] == 14

    def test_no_distortion_produces_valid_transform(self):
        """When quaternions match (org == smoothed), transform matrix is well-formed."""
        params = _make_test_params()
        ft = FrameTransform.at_timestamp(params, timestamp_ms=500.0, frame=50)

        m = ft.matrices[0]
        # The 14-float row should not be all zeros
        assert np.any(m != 0), "Matrix row should not be all zeros"

        # Extract the 3x3 portion and verify it's not NaN/inf
        mat3x3 = np.array([
            [m[0], m[1], m[2]],
            [m[3], m[4], m[5]],
            [m[6], m[7], m[8]],
        ], dtype=np.float32)
        assert np.all(np.isfinite(mat3x3)), "Matrix should not contain NaN or Inf"
