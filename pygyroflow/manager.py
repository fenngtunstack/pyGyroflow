"""StabilizationManager — the central orchestrator for the entire pipeline.

Port of Gyroflow's src/core/lib.rs StabilizationManager struct.
Ties together gyro source, lens profile, smoothing, keyframes, adaptive zoom,
and rendering into a single cohesive workflow.

The Python version simplifies the threading model (no Arc<RwLock>) since
Python's GIL provides sufficient synchronization for the intended use cases
(single-threaded CLI, notebook, or GUI-driven processing).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from pygyroflow.gyro_source import GyroSource, FileMetadata
from pygyroflow.lens import LensProfile, LensProfileDatabase
from pygyroflow.smoothing import Smoothing
from pygyroflow.keyframes import KeyframeManager
from pygyroflow.stabilization import ComputeParams, FrameTransform
from pygyroflow.stabilization_params import StabilizationParams
from pygyroflow.types.errors import GyroflowError, TelemetryParseError, VideoIOError

log = logging.getLogger(__name__)


@dataclass
class InputFile:
    """Reference to the input video file."""

    url: str = ""
    project_file_url: str | None = None
    image_sequence_fps: float = 0.0
    image_sequence_start: int = 0
    preset_name: str | None = None
    preset_output_size: tuple[int, int] | None = None


class StabilizationManager:
    """Central orchestrator for video stabilization.

    Coordinates:
      - GyroSource: raw IMU data and quaternions
      - LensProfile: camera calibration
      - LensProfileDatabase: profile search and loading
      - Smoothing: algorithm selection and execution
      - KeyframeManager: keyframe-based animation
      - StabilizationParams: user settings
      - Adaptive zoom: per-frame FOV computation
      - Rendering: GPU/CPU undistortion and video output

    Typical workflow::

        mgr = StabilizationManager()
        mgr.load_video("input.mp4")
        mgr.load_lens_profile("GoPro Hero11 Black 16:9")
        mgr.recompute_blocking()
        mgr.render("input.mp4", "output.mp4", {"codec": "H.265/HEVC"})
    """

    def __init__(self) -> None:
        self.gyro: GyroSource = GyroSource()
        self.lens: LensProfile = LensProfile()
        self.smoothing: Smoothing = Smoothing()
        self.params: StabilizationParams = StabilizationParams()
        self.keyframes: KeyframeManager = KeyframeManager()
        self.lens_db: LensProfileDatabase = LensProfileDatabase()
        self.input_file: InputFile = InputFile()

        self._compute_id: int = 0
        self._smoothing_checksum: int = 0
        self._zooming_checksum: int = 0
        self._gpu_backend: Any = None

    # ------------------------------------------------------------------
    # Video loading
    # ------------------------------------------------------------------

    def init_from_video_data(
        self,
        duration_ms: float,
        fps: float,
        frame_count: int,
        video_size: tuple[int, int],
    ) -> None:
        """Initialize from video metadata.

        Args:
            duration_ms: Video duration in milliseconds.
            fps: Video frame rate.
            frame_count: Total number of frames.
            video_size: (width, height) in pixels.
        """
        self.params.fps = fps
        self.params.frame_count = frame_count
        self.params.duration_ms = duration_ms
        self.params.size = video_size

        # Short videos use Complementary filter for stability
        if duration_ms < 10000.0:
            self.gyro.integration_method = 1  # Complementary

        self.keyframes.clear()

    def load_video(self, path: str) -> dict:
        """Load a video file and extract telemetry.

        Opens the video, reads metadata, extracts gyro data,
        and auto-detects lens profile if possible.

        Args:
            path: Path to the video file.

        Returns:
            Dict with video metadata (width, height, fps, duration_ms, frame_count).

        Raises:
            VideoIOError: If the video cannot be opened.
            TelemetryParseError: If telemetry parsing fails.
        """
        import os

        if not os.path.isfile(path):
            raise VideoIOError(f"File not found: {path}")

        # Try to get video metadata via PyAV
        video_info = self._get_video_info(path)

        if video_info["width"] <= 0 or video_info["height"] <= 0 or video_info["duration_ms"] <= 0:
            raise VideoIOError(f"Invalid video metadata: {video_info}")

        width = video_info["width"]
        height = video_info["height"]
        fps = video_info["fps"]
        duration_ms = video_info["duration_ms"]
        frame_count = video_info["frame_count"]

        self.init_from_video_data(duration_ms, fps, frame_count, (width, height))

        # Load gyro data from telemetry
        self.load_gyro_data(path, is_video=True, index=0)

        # Try to auto-load lens profile
        self._try_auto_load_lens_profile()

        # Set sizes
        self.set_size(width, height)
        output_width = width
        output_height = height
        if self.lens.output_dimension is not None:
            output_width = self.lens.output_dimension["w"]
            output_height = self.lens.output_dimension["h"]
        self.set_output_size(output_width, output_height)

        self.input_file.url = path
        return video_info

    def load_gyro_data(
        self,
        path: str,
        is_video: bool = True,
        index: int = 0,
    ) -> None:
        """Load gyro data from a file.

        Args:
            path: Path to the video or gyro data file.
            is_video: Whether the path is a video file (vs raw gyro data).
            index: Sample index for multi-stream files.
        """
        self.gyro.init_from_params(self.params.get_scaled_duration_ms())
        self.gyro.clear()
        self.gyro.file_url = path

        self._invalidate_smoothing()
        self._invalidate_zooming()

        try:
            from pygyroflow.telemetry import parse_telemetry_file

            md = parse_telemetry_file(
                path,
                sample_index=index if index > 0 else None,
                video_size=self.params.size,
                fps=self.params.fps,
            )
        except (TelemetryParseError, ImportError) as exc:
            log.warning("Failed to parse telemetry from %s: %s", path, exc)
            md = FileMetadata()

        # Detect GoPro and disable rolling shutter (already applied)
        if md.detected_source and md.detected_source.startswith("GoPro"):
            md.frame_readout_time = None

        if is_video:
            # Apply frame readout direction from telemetry
            self.params.frame_readout_direction = md.frame_readout_direction
            self.params.frame_readout_time = md.frame_readout_time or 0.0

            # Try to load lens profile from telemetry
            if md.lens_profile is not None:
                self._load_lens_from_metadata(md.lens_profile, path)

            # Override FPS if telemetry provides it
            if md.frame_rate is not None:
                md_fps = md.frame_rate
                fps = self.params.fps
                if abs(md_fps - fps) > 1.0:
                    self.override_video_fps(md_fps, recompute=False)

        self.gyro.load_from_telemetry(md)

    def load_lens_profile(self, name_or_path: str) -> None:
        """Load a lens profile by name or path.

        Args:
            name_or_path: Profile name (e.g. "GoPro Hero11") or file path.
        """
        # Ensure database is loaded
        if not self.lens_db.loaded:
            self.lens_db.load_all()

        # Try lookup by name
        profile = self.lens_db.get_by_name(name_or_path)
        if profile is None:
            profile = self.lens_db.find(name_or_path)

        if profile is not None:
            self.lens = profile
        else:
            # Try loading as JSON file
            import json
            import os

            if os.path.isfile(name_or_path):
                try:
                    with open(name_or_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.lens = LensProfile.from_json(data)
                    self.lens.path_to_file = name_or_path
                except Exception as exc:
                    raise GyroflowError(f"Failed to load lens profile: {exc}")
            else:
                raise GyroflowError(f"Lens profile not found: {name_or_path}")

        # Resolve interpolations
        self.lens.resolve_interpolations(self.lens_db)

        # Check if swap is needed for portrait/landscape
        w, h = self.params.size
        lens_w = self.lens.calib_dimension["w"]
        lens_h = self.lens.calib_dimension["h"]
        if w == lens_h and h == lens_w:
            self.lens = self.lens.swapped()

        self._invalidate_zooming()

    # ------------------------------------------------------------------
    # Size management
    # ------------------------------------------------------------------

    def set_size(self, width: int, height: int) -> None:
        """Set input video size."""
        self.params.size = (width, height)

    def set_output_size(self, width: int, height: int) -> bool:
        """Set output size, adjusted to fit within input bounds.

        Returns True if the size was changed.
        """
        if width <= 0 or height <= 0:
            return False

        r = abs(self.params.video_rotation)
        ow, oh = self.params.size
        if r == 90.0 or r == 270.0:
            ow, oh = oh, ow

        wp = float(width)
        hp = float(height)
        sw = ow / wp
        sh = oh / hp
        scale = min(sw, sh)

        nw = round(wp * scale)
        nh = round(hp * scale)

        # Ensure even dimensions
        nw -= nw % 2
        nh -= nh % 2

        output_size = (nw, nh)
        if self.params.output_size != output_size:
            self.params.output_size = output_size
            return True
        return False

    # ------------------------------------------------------------------
    # Recomputation pipeline
    # ------------------------------------------------------------------

    def recompute_smoothing(self) -> None:
        """Recompute smoothed quaternions."""
        cp = self._build_compute_params()
        self.gyro.smoothed_quaternions = self.smoothing.smooth(
            self.gyro.quaternions,
            self.gyro.duration_ms,
            cp,
            org_quats=self.gyro.quaternions,
        )

    def recompute_adaptive_zoom(self) -> None:
        """Recompute adaptive zoom FOVs."""
        from pygyroflow.zooming import calculate_fovs, ZoomMethod

        cp = self._build_compute_params()
        frames = self.params.frame_count
        fps = self.params.get_scaled_fps()

        if frames <= 0 or fps <= 0:
            return

        timestamps = [(i, i * 1000.0 / fps) for i in range(frames)]

        method = ZoomMethod(self.params.adaptive_zoom_method)
        fovs, minimal_fovs = calculate_fovs(cp, timestamps, method)

        lens_fov_adj = self.lens.optimal_fov or 1.0
        self.params.set_fovs(fovs, lens_fov_adj)
        self.params.minimal_fovs = minimal_fovs

    def recompute_undistortion(self) -> None:
        """Prepare GPU/CPU undistortion pipeline for rendering."""
        # GPU backend is initialized lazily on first use in render().
        # Per-frame undistortion is handled by FrameTransform + WgpuBackend.undistort_frame().
        pass

    def recompute_blocking(self) -> None:
        """Run full recomputation: smoothing -> zoom -> undistortion."""
        self.recompute_smoothing()
        self.recompute_adaptive_zoom()
        self.recompute_undistortion()

    # ------------------------------------------------------------------
    # Frame transform
    # ------------------------------------------------------------------

    def get_frame_transform(self, timestamp_ms: float, frame: int) -> FrameTransform:
        """Get stabilization transform for a single frame.

        Args:
            timestamp_ms: Frame center timestamp in milliseconds.
            frame: Frame index for FOV/IBIS lookup.

        Returns:
            FrameTransform with matrices and kernel params.
        """
        cp = self._build_compute_params()
        return FrameTransform.at_timestamp(cp, timestamp_ms, frame)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, input_path: str, output_path: str, options: dict | None = None) -> None:
        """Render stabilized video.

        Args:
            input_path: Input video file path.
            output_path: Output video file path.
            options: Dict with render options:
                - codec: "H.264/AVC", "H.265/HEVC", "ProRes" (default: "H.265/HEVC")
                - bitrate: Bitrate in Mbps (0 = auto)
                - use_gpu: Whether to use GPU acceleration (default: False).
                  The GPU (wgpu) undistort path is known-broken for the
                  common uint8/RGB input (pack/unpack contract mismatch,
                  see tests/test_gpu_undistort.py xfail); CPU is used by
                  default. Opt in only for experimentation.
        """
        options = options or {}

        from pygyroflow.rendering import FfmpegProcessor
        from pygyroflow.stabilization import cpu_undistort

        codec = options.get("codec", "H.265/HEVC")
        bitrate = options.get("bitrate", 0)
        use_gpu = options.get("use_gpu", False)

        proc = FfmpegProcessor()
        info = proc.open_input(input_path)

        width = info.get("width", self.params.size[0])
        height = info.get("height", self.params.size[1])
        fps = info.get("fps", self.params.fps)

        out_w = self.params.output_size[0] or width
        out_h = self.params.output_size[1] or height

        proc.create_output(output_path, out_w, out_h, fps, codec=codec, bitrate=bitrate)

        gpu_backend = None
        distortion_model = None

        if use_gpu:
            import warnings
            warnings.warn(
                "GPU (wgpu) undistort path is known-broken for uint8/RGB input "
                "(pack/unpack contract mismatch) and is untested. Output may be "
                "corrupt. Use CPU (use_gpu=False) for correct results.",
                stacklevel=2,
            )
            try:
                from pygyroflow.gpu import WgpuBackend
                from pygyroflow.stabilization.distortion_models import from_name as dm_from_name

                gpu_backend = WgpuBackend()
                if gpu_backend.available:
                    model_name = self.lens.distortion_model or "opencv_fisheye"
                    distortion_model = dm_from_name(model_name)
                    log.info("GPU acceleration enabled (wgpu)")
                else:
                    gpu_backend = None
                    log.info("GPU not available, using CPU fallback")
            except Exception as exc:
                gpu_backend = None
                log.warning("GPU init failed, falling back to CPU: %s", exc)

        def stabilize_frame(frame_data, timestamp_ms, frame_idx):
            timestamp_ms = frame_idx * 1000.0 / fps if fps > 0 else 0.0
            transform = self.get_frame_transform(timestamp_ms, frame_idx)

            if gpu_backend is not None:
                # Patch kernel params for GPU shader requirements
                kp = transform.kernel_params
                channels = frame_data.shape[2] if frame_data.ndim == 3 else 1
                kp.output_stride = out_w * channels
                kp.max_pixel_value = 255.0
                kp.pix_element_count = channels
                kp.bytes_per_pixel = channels

                wgsl = distortion_model.wgsl_functions()
                return gpu_backend.undistort_frame(
                    input_frame=frame_data,
                    kernel_params=kp,
                    matrices=transform.matrices,
                    distortion_model_wgsl=wgsl,
                )

            return cpu_undistort(frame_data, transform)

        proc.process_frames(stabilize_frame)
        proc.close()

    # ------------------------------------------------------------------
    # Parameter setters (mirroring Rust's StabilizationManager API)
    # ------------------------------------------------------------------

    def set_video_rotation(self, v: float) -> None:
        self.params.video_rotation = v
        self._invalidate_smoothing()

    def set_trim_ranges(self, ranges: list[tuple[float, float]]) -> None:
        self.params.trim_ranges = ranges if ranges != [(0.0, 1.0)] else []
        self._invalidate_smoothing()

    def set_frame_readout_time(self, v: float) -> None:
        self.params.frame_readout_time = v

    def set_adaptive_zoom(self, v: float) -> None:
        self.params.adaptive_zoom_window = v
        self._invalidate_zooming()

    def set_fov(self, v: float) -> None:
        self.params.fov = v

    def set_lens_correction_amount(self, v: float) -> None:
        self.params.lens_correction_amount = v
        self._invalidate_zooming()

    def set_smoothing_method(self, index: int) -> list[dict]:
        """Set the smoothing algorithm by index."""
        self.smoothing.set_current(index)
        self._invalidate_smoothing()
        return self.smoothing.current().get_parameters_json()

    def set_smoothing_param(self, name: str, value: float) -> None:
        self.smoothing.current().set_parameter(name, value)
        self._invalidate_smoothing()

    def override_video_fps(self, fps: float, recompute: bool = True) -> None:
        """Override video FPS with a scaling factor."""
        if abs(fps - self.params.fps) > 0.001:
            self.params.fps_scale = fps / self.params.fps
        else:
            self.params.fps_scale = None

        self.gyro.init_from_params(self.params.get_scaled_duration_ms())
        self.keyframes.timestamp_scale = self.params.fps_scale

        if recompute:
            self._invalidate_smoothing()

    def clear(self) -> None:
        """Reset all state."""
        self.params.clear()
        self._invalidate_smoothing()
        self.input_file = InputFile()
        self.gyro = GyroSource()
        self.keyframes.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_compute_params(self) -> ComputeParams:
        """Build a ComputeParams snapshot from current state."""
        lens = self.lens
        w, h = self.params.size
        ow, oh = self.params.output_size

        # Get camera matrix from lens
        camera_matrix = lens.get_camera_matrix(size=(w, h))
        distortion_coeffs = lens.get_distortion_coeffs()

        return ComputeParams(
            width=w,
            height=h,
            output_width=ow,
            output_height=oh,
            frame_count=self.params.frame_count,
            video_rotation=self.params.video_rotation,
            scaled_fps=self.params.get_scaled_fps(),
            scaled_duration_ms=self.params.get_scaled_duration_ms(),
            quaternions=dict(self.gyro.quaternions),
            smoothed_quaternions=dict(self.gyro.smoothed_quaternions),
            fovs=list(self.params.fovs),
            minimal_fovs=list(self.params.minimal_fovs),
            fov_scale=self.params.fov,
            fov_overview=self.params.fov_overview,
            show_safe_area=self.params.show_safe_area,
            max_zoom=self.params.max_zoom,
            max_zoom_iterations=self.params.max_zoom_iterations,
            camera_matrix=camera_matrix.copy(),
            distortion_coeffs=distortion_coeffs,
            distortion_model_name=lens.distortion_model or "opencv_fisheye",
            lens_correction_amount=self.params.lens_correction_amount,
            light_refraction_coefficient=self.params.light_refraction_coefficient,
            frame_readout_time=self.params.frame_readout_time,
            frame_readout_direction=self.params.frame_readout_direction,
            background=self.params.background.copy(),
            background_mode=self.params.background_mode,
            background_margin=self.params.background_margin,
            background_margin_feather=self.params.background_margin_feather,
            adaptive_zoom_window=self.params.adaptive_zoom_window,
            adaptive_zoom_center_offset=self.params.adaptive_zoom_center_offset,
            adaptive_zoom_method=self.params.adaptive_zoom_method,
            additional_rotation=self.params.additional_rotation,
            additional_translation=self.params.additional_translation,
            video_speed=self.params.video_speed,
            video_speed_affects_smoothing=self.params.video_speed_affects_smoothing,
            video_speed_affects_zooming=self.params.video_speed_affects_zooming,
            framebuffer_inverted=self.params.framebuffer_inverted,
            trim_ranges=list(self.params.trim_ranges),
            calib_width=lens.calib_dimension["w"],
            calib_height=lens.calib_dimension["h"],
            input_horizontal_stretch=lens.input_horizontal_stretch if lens.input_horizontal_stretch > 0.01 else 1.0,
            input_vertical_stretch=lens.input_vertical_stretch if lens.input_vertical_stretch > 0.01 else 1.0,
            focal_length=lens.focal_length,
        )

    def _invalidate_smoothing(self) -> None:
        """Mark smoothing as needing recomputation."""
        self._compute_id += 1
        self._smoothing_checksum = 0
        self._invalidate_zooming()

    def _invalidate_zooming(self) -> None:
        """Mark zooming as needing recomputation."""
        self._compute_id += 1
        self._zooming_checksum = 0

    def _try_auto_load_lens_profile(self) -> None:
        """Try to auto-load a lens profile based on camera identifier."""
        if not self.lens_db.loaded:
            self.lens_db.load_all()

        # Would need camera_identifier from telemetry for full auto-load
        # For now, this is a no-op placeholder

    def _load_lens_from_metadata(self, lens_data: Any, source_path: str) -> None:
        """Load lens profile from telemetry metadata."""
        if isinstance(lens_data, str) and lens_data:
            # It's a profile name/identifier
            profile = self.lens_db.find(lens_data)
            if profile is not None:
                self.lens = profile
        elif isinstance(lens_data, dict):
            # It's inline JSON data
            self.lens = LensProfile.from_json(lens_data)
            self.lens.path_to_file = source_path
            self.lens.resolve_interpolations(self.lens_db)

    def _get_video_info(self, path: str) -> dict:
        """Extract video metadata using PyAV."""
        try:
            import av

            container = av.open(path)
            stream = container.streams.video[0]

            fps = float(stream.average_rate)
            duration_s = float(stream.duration * stream.time_base) if stream.duration else 0.0
            if duration_s <= 0:
                duration_s = float(container.duration) / 1_000_000 if container.duration else 0.0

            width = stream.codec_context.width
            height = stream.codec_context.height
            frame_count = stream.frames
            if frame_count <= 0 and fps > 0 and duration_s > 0:
                frame_count = int(duration_s * fps)

            container.close()

            return {
                "width": width,
                "height": height,
                "fps": fps,
                "duration_ms": duration_s * 1000.0,
                "frame_count": frame_count,
            }
        except ImportError:
            raise VideoIOError("PyAV (av) is required for video loading")
        except Exception as exc:
            raise VideoIOError(f"Failed to open video {path}: {exc}")
