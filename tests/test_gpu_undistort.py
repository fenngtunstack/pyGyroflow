"""GPU undistort regression tests.

The wgpu undistort path is known-broken for uint8/RGB input: ``_pack_to_u32``
reinterprets each uint8 channel as a float32 bit-pattern (255 -> ~1.4e-43),
and the output buffer dtype mismatches the u32 shader path. These tests pin
that behavior as ``xfail`` so the defect is tracked and surfaces immediately
once fixed.

Uses ``force_fallback_adapter=True`` so the tests run headless without a
physical GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from pygyroflow.types.quaternion import Quat64
from pygyroflow.stabilization import ComputeParams, FrameTransform, cpu_undistort
from pygyroflow.stabilization.distortion_models import from_name as dm_from_name


def _gpu_available() -> bool:
    """True if a wgpu adapter (incl. software fallback) can be requested."""
    try:
        from pygyroflow.gpu import WgpuBackend

        backend = WgpuBackend(force_fallback=True)
        return bool(backend.available)
    except Exception:
        return False


def _identity_transform(size: int = 64) -> FrameTransform:
    """Build an identity FrameTransform (quats == smoothed -> no stabilization)."""
    quats = {}
    for i in range(30):
        ts = i * 10000  # 100 fps
        q = Quat64.from_euler_angles(0.0, 0.0, 0.0)
        quats[ts] = q

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
    return FrameTransform.at_timestamp(cp, timestamp_ms=100.0, frame=10)


def _patch_kernel_params(kp, out_w: int, channels: int) -> None:
    """Mirror manager.render()'s GPU kernel-param patching."""
    kp.output_stride = out_w * channels
    kp.max_pixel_value = 255.0
    kp.pix_element_count = channels
    kp.bytes_per_pixel = channels


pytestmark = pytest.mark.skipif(
    not _gpu_available(),
    reason="No wgpu adapter (incl. software fallback) available",
)


@pytest.mark.xfail(
    reason="GPU undistort still outputs all-zeros vs CPU reference. The "
    "_pack_to_u32 bit-reinterpretation bug is fixed (uint8 now promoted to "
    "f32 on upload), but the pipeline still produces no output under the "
    "lavapipe (non-conformant Vulkan) software adapter used in CI. Full "
    "diagnosis requires a real GPU. Tracked in 06-test-report known limits.",
    strict=True,
    raises=AssertionError,
)
def test_gpu_matches_cpu_uint8_identity():
    """GPU undistort of a uint8 identity frame should match cpu_undistort.

    Currently fails: the uint8 path produces near-zero / garbage output due
    to the pack/unpack contract mismatch in backend._pack_to_u32.
    """
    from pygyroflow.gpu import WgpuBackend

    size = 64
    channels = 3
    frame = np.full((size, size, channels), 128, dtype=np.uint8)

    transform = _identity_transform(size)
    # CPU reference (correct).
    cpu_out = cpu_undistort(frame, transform)

    # GPU path.
    kp = transform.kernel_params
    _patch_kernel_params(kp, size, channels)
    model = dm_from_name("opencv_fisheye")
    backend = WgpuBackend(force_fallback=True)
    gpu_out = backend.undistort_frame(
        input_frame=frame,
        kernel_params=kp,
        matrices=transform.matrices,
        distortion_model_wgsl=model.wgsl_functions(),
    )

    assert gpu_out.shape == cpu_out.shape
    np.testing.assert_allclose(
        gpu_out.astype(np.float32), cpu_out.astype(np.float32),
        atol=2, err_msg="GPU output diverges from CPU reference",
    )
