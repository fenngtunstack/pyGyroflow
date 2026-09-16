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

class TestOptimalFov:
    """A lens profile's `optimal_fov` adjusts the UI fov (or the render fov).

    Upstream frame_transform.rs: with per-frame fovs present the UI fov is
    divided by it — the render fov already carries the same factor through
    `StabilizationParams.set_fovs`. Without per-frame fovs the render fov is
    multiplied. Neither branch existed here.
    """

    def test_ui_fov_is_divided_when_per_frame_fovs_exist(self):
        """The UI fov lands on FrameTransform.fov; the kernel gets `fov`."""
        base = _make_test_params()
        adj = _make_test_params()
        adj.optimal_fov = 1.25
        assert base.fovs  # the branch under test

        plain = FrameTransform.at_timestamp(base, timestamp_ms=500.0, frame=50)
        adjusted = FrameTransform.at_timestamp(adj, timestamp_ms=500.0, frame=50)
        assert adjusted.fov == pytest.approx(plain.fov / 1.25, rel=1e-6)
        # the render fov is untouched — set_fovs already applied the factor
        assert adjusted.kernel_params.fov == pytest.approx(
            plain.kernel_params.fov, rel=1e-6
        )

    def test_render_fov_is_multiplied_without_per_frame_fovs(self):
        base = _make_test_params()
        base.fovs = []
        adj = _make_test_params()
        adj.fovs = []
        adj.optimal_fov = 1.25

        plain = FrameTransform.at_timestamp(base, timestamp_ms=500.0, frame=50)
        adjusted = FrameTransform.at_timestamp(adj, timestamp_ms=500.0, frame=50)
        assert adjusted.kernel_params.fov == pytest.approx(
            plain.kernel_params.fov * 1.25, rel=1e-6
        )

    def test_absent_optimal_fov_is_a_no_op(self):
        base = _make_test_params()
        same = _make_test_params()
        a = FrameTransform.at_timestamp(base, timestamp_ms=500.0, frame=50)
        b = FrameTransform.at_timestamp(same, timestamp_ms=500.0, frame=50)
        assert a.kernel_params.fov == b.kernel_params.fov


class TestPerFrameTimeOffsets:
    """Per-frame timestamp corrections shift the rolling-shutter timing.

    Upstream adds `file_metadata.per_frame_time_offsets[frame]` to the video
    timestamp after the readout time is resolved, so the gyro lookup for each
    row starts from the corrected frame time.
    """

    def _transform(self, offsets):
        params = _make_test_params()
        params.frame_readout_time = 0.03
        params.per_frame_time_offsets = offsets
        return FrameTransform.at_timestamp(params, timestamp_ms=500.0, frame=50)

    def test_offsets_move_the_row_matrices(self):
        plain = self._transform([])
        shifted = self._transform([0.0] * 50 + [5.0] + [0.0] * 49)
        assert not np.allclose(plain.matrices, shifted.matrices)

    def test_out_of_range_frame_is_ignored(self):
        plain = self._transform([])
        short = self._transform([3.0])
        assert np.allclose(plain.matrices, short.matrices)

    def test_zero_offset_matches_no_offsets(self):
        plain = self._transform([])
        zeros = self._transform([0.0] * 100)
        assert np.allclose(plain.matrices, zeros.matrices)


class TestFramebufferInverted:
    """An inverted framebuffer flips the adaptive-zoom centre vertically."""

    def _transform(self, inverted, center_y=0.1):
        params = _make_test_params()
        params.adaptive_zoom_center_offset = (0.0, center_y)
        params.framebuffer_inverted = inverted
        return FrameTransform.at_timestamp(params, timestamp_ms=500.0, frame=50)

    def test_center_is_negated(self):
        upright = self._transform(False)
        flipped = self._transform(True)
        assert flipped.kernel_params.translation2d[1] == pytest.approx(
            -upright.kernel_params.translation2d[1], rel=1e-6
        )

    def test_horizontal_center_is_untouched(self):
        params = _make_test_params()
        params.adaptive_zoom_center_offset = (0.1, 0.1)
        upright = FrameTransform.at_timestamp(params, timestamp_ms=500.0, frame=50)
        params2 = _make_test_params()
        params2.adaptive_zoom_center_offset = (0.1, 0.1)
        params2.framebuffer_inverted = True
        flipped = FrameTransform.at_timestamp(params2, timestamp_ms=500.0, frame=50)
        assert flipped.kernel_params.translation2d[0] == pytest.approx(
            upright.kernel_params.translation2d[0], rel=1e-6
        )

    def test_zero_center_is_unaffected(self):
        assert self._transform(True, center_y=0.0).kernel_params.translation2d[1] == 0.0
