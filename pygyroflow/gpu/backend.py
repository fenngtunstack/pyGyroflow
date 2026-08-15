"""GPU undistortion backend using wgpu-py.

Provides WgpuBackend which wraps the full GPU pipeline for frame stabilization:
shader compilation, buffer management, compute pass dispatch, and result readback.
Falls back to CPU when wgpu is not available.
"""

from __future__ import annotations

import ctypes
import logging
from typing import Optional

import numpy as np

from pygyroflow.types.errors import GPUError
from pygyroflow.types.kernel_params import KernelParams
from pygyroflow.gpu.shader_builder import build_undistort_shader

log = logging.getLogger(__name__)

# WGSL shader override constants mapped to KernelParams fields.
# These correspond to @id(100)..@id(103) in the WGSL template.
_PIPELINE_CONSTANTS_KEYS = (
    "interpolation",
    "pix_element_count",
    "bytes_per_pixel",
    "flags",
)


class WgpuBackend:
    """GPU undistortion backend using wgpu-py.

    Loads Gyroflow's WGSL shader and dispatches compute passes for frame
    stabilization.  Initialises lazily so importing this module never fails
    when wgpu is not installed.
    """

    def __init__(self, force_fallback: bool = False) -> None:
        """Initialize the wgpu backend.

        Args:
            force_fallback: If True, request a software/CPU fallback adapter
                (``force_fallback_adapter=True``). Used by headless tests
                without a physical GPU.
        """
        self._device: object | None = None
        self._queue: object | None = None
        self._pipeline_cache: dict[str, object] = {}
        self._initialized: bool = False
        self._available: Optional[bool] = None
        self._force_fallback: bool = force_fallback

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        """Lazy initialization of the wgpu device.

        Called automatically before GPU operations.  Sets ``_initialized``
        on success or raises :class:`~pygyroflow.types.errors.GPUError`.
        """
        if self._initialized:
            return

        try:
            import wgpu  # type: ignore[import-untyped]

            adapter = wgpu.gpu.request_adapter_sync(
                power_preference="high-performance",
                force_fallback_adapter=self._force_fallback,
            )
            if adapter is None:
                raise GPUError("No suitable GPU adapter found")

            self._device = adapter.request_device_sync()
            self._queue = self._device.queue
            self._initialized = True
            self._available = True

            info = getattr(adapter, "request_adapter_info", lambda: {})()
            log.info("wgpu device initialized: %s", info)
        except ImportError:
            self._available = False
            raise GPUError(
                "wgpu-py is not installed. Install with: pip install wgpu"
            )
        except Exception as exc:
            self._available = False
            raise GPUError(f"Failed to initialize wgpu: {exc}") from exc

    @property
    def available(self) -> bool:
        """Return True if a wgpu device was successfully initialised."""
        if self._available is None:
            try:
                self._ensure_initialized()
            except GPUError:
                pass
        return self._available is True

    # ------------------------------------------------------------------
    # Pipeline creation
    # ------------------------------------------------------------------

    def _get_or_create_pipeline(
        self,
        shader_code: str,
        pipeline_constants: dict[str, float],
    ) -> object:
        """Return a cached compute pipeline or create a new one.

        Caching is based on a hash of the shader source so the pipeline
        is reused across frames when the distortion model does not change.
        """
        import hashlib

        key = hashlib.sha256(shader_code.encode()).hexdigest()
        if key in self._pipeline_cache:
            return self._pipeline_cache[key]

        import wgpu  # type: ignore[import-untyped]

        shader_module = self._device.create_shader_module(code=shader_code)

        # Build bind group layout for the compute pipeline (buffer path).
        # Bindings mirror wgpu_undistort.wgsl:
        #   0: uniform   KernelParams
        #   1: storage-r matrices
        #   2: storage-r coeffs
        #   3: storage-r mesh_data
        #   4: storage-r drawing
        #   5: storage-r input_buffer
        #   6: storage-rw output_buffer
        bgl = self._device.create_bind_group_layout(
            entries=[
                {
                    "binding": i,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {
                        "type": wgpu.BufferBindingType.uniform
                        if i == 0
                        else (
                            wgpu.BufferBindingType.read_only_storage
                            if i != 6
                            else wgpu.BufferBindingType.storage
                        ),
                    },
                }
                for i in range(7)
            ]
        )

        pipeline_layout = self._device.create_pipeline_layout(
            bind_group_layouts=[bgl]
        )

        pipeline = self._device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={
                "module": shader_module,
                "entry_point": "undistort_compute",
                "constants": pipeline_constants,
            },
        )

        self._pipeline_cache[key] = pipeline
        return pipeline

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def undistort_frame(
        self,
        input_frame: np.ndarray,
        kernel_params: KernelParams,
        matrices: np.ndarray,
        distortion_model_wgsl: str,
        digital_lens_wgsl: str = "",
        coeffs: np.ndarray | None = None,
        drawing_bytes: bytes = b"",
        mesh_data: np.ndarray | None = None,
    ) -> np.ndarray:
        """Process a single frame through the GPU compute pipeline.

        Args:
            input_frame: Input image, shape (H, W) or (H, W, C), dtype uint8/uint16/float32.
            kernel_params: Populated KernelParams struct (320 bytes).
            matrices: Transform matrices, shape (matrix_count, 14), dtype float32.
            distortion_model_wgsl: WGSL source for distort_point / undistort_point.
            digital_lens_wgsl: WGSL source for digital lens, or empty.
            coeffs: Pre-computed interpolation coefficients (484 floats).
            drawing_bytes: Drawing overlay data.
            mesh_data: Mesh correction data, flat float32 array.

        Returns:
            Stabilised output frame with same dtype and channel count.

        Raises:
            GPUError: If the GPU pipeline fails or wgpu is not available.
        """
        self._ensure_initialized()

        if input_frame.ndim == 2:
            # Greyscale -- treat as (H, W, 1)
            h, w = input_frame.shape
            channels = 1
        elif input_frame.ndim == 3:
            h, w, channels = input_frame.shape
        else:
            raise ValueError(
                f"input_frame must be 2D or 3D, got {input_frame.ndim}D"
            )

        import wgpu  # type: ignore[import-untyped]

        # Determine scalar type for the shader. All integer dtypes are
        # promoted to f32 on upload (the shader's f32 path reads values
        # directly via f32(input_buffer[...])); the prior u32 path packed
        # each uint8 into a float32 bit-pattern via view(), which corrupted
        # values (255 -> 3.57e-43). f32 upload + clip-on-readback is correct.
        scalar_type = "f32"
        padded = input_frame.astype(np.float32)

        out_h = kernel_params.output_height
        out_w = kernel_params.output_width

        # Build the shader.
        shader_code = build_undistort_shader(
            distortion_model_wgsl=distortion_model_wgsl,
            digital_lens_wgsl=digital_lens_wgsl,
            use_buffer_input=True,
            scalar_type=scalar_type,
        )

        # Pipeline constants matching WGSL @id overrides.
        constants = {}
        for i, name in enumerate(_PIPELINE_CONSTANTS_KEYS):
            constants[str(100 + i)] = float(getattr(kernel_params, name))

        pipeline = self._get_or_create_pipeline(shader_code, constants)

        # Create GPU buffers.
        in_bytes = padded.tobytes()
        out_bytes = np.zeros((out_h, out_w, max(channels, 1)), dtype=np.float32).tobytes()
        params_bytes = kernel_params.to_bytes()
        mat_bytes = matrices.astype(np.float32).tobytes()

        buf_input = self._device.create_buffer_with_data(
            data=in_bytes,
            usage=wgpu.BufferUsage.STORAGE,
        )
        buf_output = self._device.create_buffer_with_data(
            data=out_bytes,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        buf_params = self._device.create_buffer_with_data(
            data=params_bytes,
            usage=wgpu.BufferUsage.UNIFORM,
        )
        buf_matrices = self._device.create_buffer_with_data(
            data=mat_bytes,
            usage=wgpu.BufferUsage.STORAGE,
        )

        # Coefficients buffer (interpolation lookup table).
        if coeffs is None:
            from pygyroflow.gpu.coeffs_table import COEFFS
            coeffs = COEFFS
        buf_coeffs = self._device.create_buffer_with_data(
            data=coeffs.astype(np.float32).tobytes(),
            usage=wgpu.BufferUsage.STORAGE,
        )

        # Mesh data buffer.
        if mesh_data is None:
            mesh_data = np.zeros(1024, dtype=np.float32)
        buf_mesh = self._device.create_buffer_with_data(
            data=mesh_data.astype(np.float32).tobytes(),
            usage=wgpu.BufferUsage.STORAGE,
        )

        # Drawing buffer (must be at least 16 bytes even if disabled).
        if not drawing_bytes or len(drawing_bytes) < 16:
            drawing_bytes = b"\x00" * 16
        buf_drawing = self._device.create_buffer_with_data(
            data=drawing_bytes,
            usage=wgpu.BufferUsage.STORAGE,
        )

        # Staging buffer for CPU readback.
        staging = self._device.create_buffer(
            size=len(out_bytes),
            usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST,
        )

        # Create bind group.
        bind_group = self._device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": buf_params}},
                {"binding": 1, "resource": {"buffer": buf_matrices}},
                {"binding": 2, "resource": {"buffer": buf_coeffs}},
                {"binding": 3, "resource": {"buffer": buf_mesh}},
                {"binding": 4, "resource": {"buffer": buf_drawing}},
                {"binding": 5, "resource": {"buffer": buf_input}},
                {"binding": 6, "resource": {"buffer": buf_output}},
            ],
        )

        # Dispatch compute pass.
        command_encoder = self._device.create_command_encoder()
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group, [], None, None)
        workgroups_x = (out_w + 7) // 8
        workgroups_y = (out_h + 7) // 8
        compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1)
        compute_pass.end()

        # Copy output to staging buffer for readback.
        command_encoder.copy_buffer_to_buffer(
            buf_output, 0, staging, 0, len(out_bytes)
        )
        self._queue.submit([command_encoder.finish()])

        # Map and read back.
        staging.map_sync(mode=wgpu.MapMode.READ)
        data = staging.read_mapped()
        staging.unmap()
        result = np.frombuffer(data, dtype=np.float32).reshape(
            out_h, out_w, max(channels, 1)
        )

        # Cast back to original dtype.
        if input_frame.dtype == np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)
        elif input_frame.dtype == np.uint16:
            result = np.clip(result, 0, 65535).astype(np.uint16)

        # Squeeze channel dim if input was greyscale.
        if input_frame.ndim == 2:
            result = result[:, :, 0]

        return result

    @staticmethod
    def undistort_frame_cpu_fallback(
        input_frame: np.ndarray,
        transform,
    ) -> np.ndarray:
        """CPU fallback when GPU is not available.

        ``transform`` is a stabilization.FrameTransform (carries both the
        matrices and the kernel params — the old signature passed them
        separately and crashed with TypeError on call).
        """
        from pygyroflow.stabilization.cpu_undistort import cpu_undistort

        return cpu_undistort(input_frame, transform)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
# (removed) _pack_to_u32: see note at the f32 upload path above.
