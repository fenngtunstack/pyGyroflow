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
from pygyroflow.stabilization.cpu_undistort import CPU_TO_UPSTREAM_INTERPOLATION
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
        # The two paths number their kernels differently; translate, or the
        # comparison is between two different filters.
        cpu_interp = 0
        transform.kernel_params.interpolation = CPU_TO_UPSTREAM_INTERPOLATION[cpu_interp]
        cpu_out = cpu_undistort(frame, transform, interpolation=cpu_interp)

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
        # The two paths number their kernels differently; translate, or the
        # comparison is between two different filters.
        cpu_interp = 0
        transform.kernel_params.interpolation = CPU_TO_UPSTREAM_INTERPOLATION[cpu_interp]
        cpu_out = cpu_undistort(frame, transform, interpolation=cpu_interp)

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

class TestGpuInterpolation:
    """The interpolation kernel actually reaches the shader.

    Two independent faults made `--interpolation` a no-op on the GPU:
    `FrameTransform` wrote a hardcoded CPU-convention `2` into
    `KernelParams.interpolation` (the shader reads 2 as *bilinear*, so every
    GPU render was bilinear), and the pipeline cache keyed on the shader
    source alone, so the first kernel built was reused for the whole process
    no matter what constant was asked for.
    """

    SIZE = 96
    CHANNELS = 3

    @staticmethod
    def _scene():
        """A textured frame under a fractionally-sampling transform."""
        import cv2

        from pygyroflow.types.quaternion import Quat64

        size = TestGpuInterpolation.SIZE
        rng = np.random.default_rng(7)
        frame = cv2.GaussianBlur(
            rng.integers(0, 255, (size, size, 3), dtype=np.uint8), (3, 3), 0
        )
        quats = {
            i * 10000: Quat64.from_euler_angles(
                np.deg2rad(1.7), np.deg2rad(0.9), np.deg2rad(0.4)
            )
            for i in range(30)
        }
        cp = ComputeParams(
            width=size, height=size, output_width=size, output_height=size,
            frame_count=30, scaled_fps=100.0, scaled_duration_ms=300.0,
            fovs=[1.0] * 30, fov_scale=1.0,
            camera_matrix=np.array([
                [size / 2.0, 0.0, size / 2.0],
                [0.0, size / 2.0, size / 2.0],
                [0.0, 0.0, 1.0],
            ]),
            distortion_coeffs=[0.15, -0.05, 0.02, -0.003] + [0.0] * 8,
            frame_readout_time=0.0,
            quaternions=dict(quats),
            smoothed_quaternions={k: Quat64.identity() for k in quats},
        )
        return frame, cp

    @classmethod
    def _gpu_frame(cls, backend, frame, cp, upstream_interp):
        size, channels = cls.SIZE, cls.CHANNELS
        transform = FrameTransform.at_timestamp(cp, timestamp_ms=100.0, frame=10)
        kp = transform.kernel_params
        _patch_kernel_params(kp, size, channels)
        kp.interpolation = upstream_interp
        model = dm_from_name(transform.distortion_model_name)
        return backend.undistort_frame(
            input_frame=frame,
            kernel_params=kp,
            matrices=transform.matrices,
            distortion_model_wgsl=model.wgsl_functions(),
        )

    @staticmethod
    def _sharpness(img):
        import cv2

        return cv2.Laplacian(
            cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.CV_64F
        ).var()

    def test_default_kernel_is_lanczos4(self):
        """Upstream's default, not the CPU path's 2."""
        from pygyroflow.types.enums import Interpolation

        frame, cp = self._scene()
        kp = FrameTransform.at_timestamp(cp, timestamp_ms=100.0, frame=10).kernel_params
        assert kp.interpolation == int(Interpolation.Lanczos4) == 8

    def test_constant_reaches_the_shader(self):
        """Wider kernels must produce wider kernels, not the same frame."""
        from pygyroflow.gpu import WgpuBackend

        frame, cp = self._scene()
        backend = WgpuBackend()
        bilinear = self._gpu_frame(backend, frame, cp, 2)
        lanczos = self._gpu_frame(backend, frame, cp, 8)
        assert np.abs(bilinear.astype(int) - lanczos.astype(int)).max() > 5

    def test_wider_kernel_is_sharper(self):
        from pygyroflow.gpu import WgpuBackend

        frame, cp = self._scene()
        backend = WgpuBackend()
        bilinear = self._gpu_frame(backend, frame, cp, 2)
        lanczos = self._gpu_frame(backend, frame, cp, 8)
        assert self._sharpness(lanczos) > self._sharpness(bilinear) * 1.2

    @pytest.mark.parametrize("cpu_interp", [0, 1, 2])
    def test_each_kernel_matches_its_cpu_counterpart(self, cpu_interp):
        """Translated through the documented mapping, GPU == CPU.

        Before the fix bicubic and Lanczos4 were off by ~20 levels, because
        the GPU sat on bilinear no matter what was requested.
        """
        from pygyroflow.gpu import WgpuBackend

        frame, cp = self._scene()
        upstream = CPU_TO_UPSTREAM_INTERPOLATION[cpu_interp]
        gpu_out = self._gpu_frame(WgpuBackend(), frame, cp, upstream)
        transform = FrameTransform.at_timestamp(cp, timestamp_ms=100.0, frame=10)
        cpu_out = cpu_undistort(frame, transform, interpolation=cpu_interp)
        assert np.abs(gpu_out.astype(int) - cpu_out.astype(int)).max() <= 3
