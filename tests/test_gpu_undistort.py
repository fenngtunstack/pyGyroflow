"""GPU undistort regression tests (real-hardware).

History: the GPU path went through several real bugs, each fixed and
verified on an Intel UHD 630 (Vulkan):
  1. ``_pack_to_u32`` bit-reinterpreted uint8 channels as float32
     (255 -> ~1.4e-43) — replaced by value-preserving f32 upload.
  2. The interpolation coefficients buffer defaulted to all-zeros
     (every bilinear weight 0) — now uploads the verbatim upstream
     488-entry COEFFS table.
  3. ``safe_area_rect`` stayed (0,0,0,0) while the shader's
     ``draw_safe_area`` dims pixels outside that rect by 0.5x — the
     frame transform now sets it to the full output frame.
  4. Spec-constant ``@id(100..103)`` annotations so pipeline constants
     (interpolation / channel counts / flags) actually bind.

With a real adapter the identity transform is BIT-EXACT vs cpu_undistort.
On machines without any adapter the test skips.
"""

from __future__ import annotations

import numpy as np
import pytest

from pygyroflow.types.quaternion import Quat64
from pygyroflow.stabilization import ComputeParams, FrameTransform, cpu_undistort
from pygyroflow.stabilization.distortion_models import from_name as dm_from_name


def _real_gpu_available() -> bool:
    """True if a hardware wgpu adapter can be requested."""
    try:
        from pygyroflow.gpu import WgpuBackend

        backend = WgpuBackend()  # no force_fallback: hardware only
        return bool(backend.available)
    except Exception:
        return False


def _patch_kernel_params(kp, out_w: int, channels: int) -> None:
    """Mirror manager.render()'s GPU kernel-param patching."""
    kp.output_stride = out_w * channels
    kp.max_pixel_value = 255.0
    kp.pixel_value_limit = 255.0
    kp.pix_element_count = channels
    kp.bytes_per_pixel = channels


pytestmark = pytest.mark.skipif(
    not _real_gpu_available(),
    reason="No hardware wgpu adapter available",
)


class TestGpuUndistort:
    def test_gpu_matches_cpu_uint8_identity(self):
        """GPU undistort of a uint8 identity frame == cpu_undistort."""
        from pygyroflow.gpu import WgpuBackend

        size = 64
        channels = 3
        frame = np.random.default_rng(11).integers(0, 255, (size, size, channels), dtype=np.uint8)

        quats = {}
        for i in range(30):
            quats[i * 10000] = Quat64.from_euler_angles(0.0, 0.0, 0.0)
        cp = ComputeParams(
            width=size, height=size, output_width=size, output_height=size,
            frame_count=30, scaled_fps=100.0, scaled_duration_ms=300.0,
            quaternions=dict(quats), smoothed_quaternions=dict(quats),
            fovs=[1.0] * 30, fov_scale=1.0,
            camera_matrix=np.array([
                [size / 2.0, 0.0, size / 2.0],
                [0.0, size / 2.0, size / 2.0],
                [0.0, 0.0, 1.0],
            ]),
            distortion_coeffs=[0.0] * 12,
            frame_readout_time=0.0,
        )
        transform = FrameTransform.at_timestamp(cp, timestamp_ms=100.0, frame=10)
        cpu_out = cpu_undistort(frame, transform)

        backend = WgpuBackend()
        kp = transform.kernel_params
        _patch_kernel_params(kp, size, channels)
        model = dm_from_name(transform.distortion_model_name)
        gpu_out = backend.undistort_frame(
            input_frame=frame,
            kernel_params=kp,
            matrices=transform.matrices,
            distortion_model_wgsl=model.wgsl_functions(),
        )

        assert gpu_out.shape == cpu_out.shape
        assert np.array_equal(gpu_out, cpu_out), (
            f"GPU vs CPU max diff {np.abs(gpu_out.astype(int) - cpu_out.astype(int)).max()}"
        )

    def test_gpu_fisheye_distortion_matches_cpu(self):
        """Nonzero fisheye coefficients: GPU vs CPU within rounding."""
        from pygyroflow.gpu import WgpuBackend

        size = 64
        channels = 3
        frame = np.random.default_rng(3).integers(0, 255, (size, size, channels), dtype=np.uint8)

        quats = {}
        for i in range(30):
            quats[i * 10000] = Quat64.from_euler_angles(0.0, 0.0, 0.0)
        cp = ComputeParams(
            width=size, height=size, output_width=size, output_height=size,
            frame_count=30, scaled_fps=100.0, scaled_duration_ms=300.0,
            quaternions=dict(quats), smoothed_quaternions=dict(quats),
            fovs=[1.0] * 30, fov_scale=1.0,
            camera_matrix=np.array([
                [size / 2.0, 0.0, size / 2.0],
                [0.0, size / 2.0, size / 2.0],
                [0.0, 0.0, 1.0],
            ]),
            distortion_coeffs=[0.15, -0.05, 0.02, -0.003] + [0.0] * 8,
            frame_readout_time=0.0,
        )
        transform = FrameTransform.at_timestamp(cp, timestamp_ms=100.0, frame=10)
        cpu_out = cpu_undistort(frame, transform)

        backend = WgpuBackend()
        kp = transform.kernel_params
        _patch_kernel_params(kp, size, channels)
        model = dm_from_name(transform.distortion_model_name)
        gpu_out = backend.undistort_frame(
            input_frame=frame,
            kernel_params=kp,
            matrices=transform.matrices,
            distortion_model_wgsl=model.wgsl_functions(),
        )

        d = np.abs(gpu_out.astype(int) - cpu_out.astype(int))
        # fisheye Newton iterations run iteratively in WGSL vs our vectorized
        # numpy loop — allow small rounding differences
        assert d.max() <= 2, f"GPU vs CPU fisheye max diff {d.max()}"
