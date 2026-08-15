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

import numpy as np

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
        self._radial_limit_cache: dict[tuple[str, tuple[float, ...]], float] = {}

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

        # GoPro rolling-shutter readout (SROT) is kept: upstream Gyroflow
        # uses it for RS correction. The previous unconditional
        # `frame_readout_time = None` here disabled RS on all GoPro files.

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
        smoothed = self.smoothing.smooth(
            self.gyro.quaternions,
            self.gyro.duration_ms,
            cp,
            org_quats=self.gyro.quaternions,
        )

        # Upstream gyro_source.rs recompute_smoothness(): after smoothing,
        # store the CORRECTION quaternion sm^-1 * org, not the smoothed
        # orientation itself. FrameTransform composes it with the org
        # lookups (smoothed * org_c^-1 * org_row = sm^-1 * org_row), which
        # is the rotation that maps the raw frame back onto the smoothed
        # path. Without this step the stabilization is ineffective.
        org = self.gyro.quaternions
        self.gyro.smoothed_quaternions = {
            ts: q.inverse() * org[ts]
            for ts, q in smoothed.items()
            if ts in org
        }

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
    # Synchronization
    # ------------------------------------------------------------------

    def synchronize(
        self,
        input_path: str | None = None,
        sample_count: int = 200,
        search_range_ms: float = 500.0,
        use_rs: bool = True,
        progress_callback: Any = None,
    ) -> float | None:
        """Auto-synchronize the gyro timeline to the video via optical flow.

        Samples grayscale frames from the video (subsampled and downscaled
        for speed), estimates camera rotation between consecutive frames,
        and searches for the time offset that best matches the gyro data.
        With ``use_rs`` (and quaternion data available) this runs the
        rolling-shutter-aware per-point quaternion search; otherwise a 1-D
        angular-velocity cross-correlation is used.

        The result is stored on the gyro source (``gyro.set_offset``) and
        thereby takes effect in ``get_frame_transform`` lookups.

        Args:
            input_path: Video to analyze (defaults to the loaded file).
            sample_count: Approximate number of frames to analyze.
            search_range_ms: Offset search window in ms.
            use_rs: Prefer the rolling-shutter-aware search.
            progress_callback: Optional Callable[[float], None].

        Returns:
            Offset in ms (``visual = gyro + offset``), or None on failure.
        """
        path = input_path or self.input_file.url
        if not path:
            raise GyroflowError("No input video loaded for synchronization")
        if not self.gyro.quaternions:
            log.warning("No gyro quaternions; skipping auto-sync")
            return None

        frames = self._extract_gray_frames(path, sample_count)
        if len(frames) < 10:
            log.warning("Only %d frames extracted; skipping auto-sync", len(frames))
            return None

        gyro_data = self._gyro_angular_velocity()
        if not gyro_data:
            log.warning("No usable angular-velocity signal; skipping auto-sync")
            return None

        from pygyroflow.synchronization import AutosyncProcess

        height, width = frames[0][1].shape[:2]
        scale = width / max(1, self.params.size[0])
        camera_matrix = self.lens.get_camera_matrix(size=(width, height))

        method = 2 if use_rs else 1
        proc = AutosyncProcess(
            camera_matrix=camera_matrix,
            fps=self.params.fps,
            scaled_fps=self.params.get_scaled_fps(),
            of_method=2,  # DIS: fastest detector
            pose_method=0,
            offset_method=method,
        )

        log.info(
            "Auto-sync: analyzing %d frames (%dx%d, %.2fx scale)",
            len(frames), width, height, scale,
        )
        offset = proc.run(
            frames,
            gyro_data,
            search_range_ms=search_range_ms,
            sample_count=None,  # already subsampled during extraction
            progress_callback=progress_callback,
            quaternions=dict(self.gyro.quaternions) if use_rs else None,
            frame_readout_time_ms=self.params.frame_readout_time,
        )

        if offset is None or not np.isfinite(offset):
            log.warning("Auto-sync failed to find an offset")
            return None

        self.gyro.set_offset(0, float(offset))
        log.info("Auto-sync offset: %.2f ms", offset)
        return float(offset)

    def _extract_gray_frames(
        self,
        path: str,
        sample_count: int,
        max_width: int = 480,
    ) -> list[tuple[int, Any]]:
        """Decode a subsampled, downscaled sequence of grayscale frames.

        Returns [(timestamp_us, gray_u8), ...] with real container pts.
        """
        import av
        import cv2

        frames: list[tuple[int, Any]] = []

        container = av.open(path)
        try:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            fps = float(stream.average_rate or 30.0)
            frame_count = stream.frames or int(fps * 60.0)
            every = max(1, frame_count // max(1, sample_count))

            scale = min(1.0, max_width / max(1, stream.width))
            out_w = max(2, int(stream.width * scale) & ~1)
            out_h = max(2, int(stream.height * scale) & ~1)
            tb = float(stream.time_base or 1)

            def emit(frame, idx: int):
                if idx % every != 0:
                    return
                gray = frame.to_ndarray(format="gray")
                if scale < 1.0:
                    gray = cv2.resize(gray, (out_w, out_h), interpolation=cv2.INTER_AREA)
                ts_us = (
                    int(round(float(frame.pts) * tb * 1e6))
                    if frame.pts is not None
                    else int(round(idx * 1e6 / fps))
                )
                frames.append((ts_us, gray))

            idx = 0
            for packet in container.demux(stream):
                if packet.dts is None:
                    continue
                for frame in packet.decode():
                    emit(frame, idx)
                    idx += 1
            # Flush the threaded decoder (trailing frames).
            for frame in stream.decode():
                emit(frame, idx)
                idx += 1
        finally:
            container.close()

        return frames

    def _gyro_angular_velocity(self) -> list[tuple[int, Any]]:
        """Angular-velocity signal [(timestamp_us, deg/s 3-vector), ...].

        Uses raw IMU samples when available; otherwise derives angular
        velocity from consecutive quaternion differences.
        """
        raw = [imu for imu in self.gyro.raw_imu if imu.gyro is not None]
        if len(raw) >= 3:
            return [
                (int(round(imu.timestamp_ms * 1000.0)), np.asarray(imu.gyro, dtype=np.float64))
                for imu in raw
            ]

        keys = sorted(self.gyro.quaternions.keys())
        if len(keys) < 3:
            return []

        result: list[tuple[int, Any]] = []
        for i in range(len(keys) - 1):
            t0, t1 = keys[i], keys[i + 1]
            dt_s = max((t1 - t0) / 1e6, 1e-9)
            q0 = self.gyro.quaternions[t0]
            q1 = self.gyro.quaternions[t1]
            dq = q1 * q0.inverse()
            w, x, y, z = dq.quaternion()
            ang = 2.0 * np.arccos(np.clip(abs(w), 0.0, 1.0))
            axis = np.array([x, y, z], dtype=np.float64)
            norm = np.linalg.norm(axis)
            omega = np.zeros(3) if norm < 1e-12 else axis / norm * (ang / dt_s)
            result.append((int(t0), np.rad2deg(omega)))
        return result

    # All 48 axis orientations (permutations × signs), upstream rs_sync.rs.
    _POSSIBLE_ORIENTATIONS = (
        "YxZ", "Xyz", "XZy", "Zxy", "zyX", "yxZ", "ZXY", "zYx", "ZYX", "yXz",
        "YZX", "XyZ", "Yzx", "zXy", "YXz", "xyz", "yZx", "XYZ", "zxy", "xYz",
        "XYz", "zxY", "zXY", "xZy", "zyx", "xyZ", "Yxz", "xzy", "yZX", "yzX",
        "ZYx", "xYZ", "zYX", "ZxY", "yzx", "xZY", "Xzy", "XzY", "YzX", "Zyx",
        "XZY", "yxz", "xzY", "ZyX", "YXZ", "yXZ", "YZx", "ZXy",
    )

    def guess_imu_orientation(
        self,
        input_path: str | None = None,
        sample_count: int = 100,
        progress_callback: Any = None,
    ) -> str | None:
        """Guess the IMU axis orientation from footage (upstream guess_orient).

        For files without an embedded orientation (upstream telemetry-parser
        returns None on modern GoPros), upstream Gyroflow tries all 48 axis
        permutations: re-integrate the raw gyro with each orientation and
        pick the one whose rolling-shutter sync cost against the optical
        flow is minimal. Requires raw IMU data — files with pre-integrated
        quaternions (DJI, GoPro CORI) already carry an absolute
        orientation and are skipped.

        Returns the winning orientation string (also applied to the gyro
        source), or None when guessing is not applicable / inconclusive.
        """
        md = self.gyro.file_metadata
        if md is None or not getattr(md, "raw_imu", None):
            log.info("Orientation guess: no raw IMU data, skipping")
            return None
        if getattr(md, "quaternions", None):
            log.info("Orientation guess: pre-integrated quaternions present, skipping")
            return None
        if self.gyro.imu_transforms.imu_orientation not in (None, "XYZ"):
            log.info(
                "Orientation guess: file already specifies '%s', skipping",
                self.gyro.imu_transforms.imu_orientation,
            )
            return None

        path = input_path or self.input_file.url
        if not path:
            raise GyroflowError("No input video loaded for orientation guessing")

        frames = self._extract_gray_frames(path, sample_count)
        if len(frames) < 10:
            log.warning("Orientation guess: only %d frames, skipping", len(frames))
            return None

        height, width = frames[0][1].shape[:2]
        camera_matrix = self.lens.get_camera_matrix(size=(width, height))

        from pygyroflow.synchronization import AutosyncProcess
        from pygyroflow.synchronization.find_offset.rs_sync import RollingShutterSync

        proc = AutosyncProcess(
            camera_matrix=camera_matrix,
            fps=self.params.fps,
            scaled_fps=self.params.get_scaled_fps(),
            of_method=2,
            pose_method=0,
            offset_method=2,
        )
        estimator = proc.run_pose_only(frames)
        if estimator is None:
            return None

        # Build the optical-flow tracks once; they are orientation-invariant.
        ordered = sorted(estimator.get_frame_results().values(), key=lambda f: f.frame_no)
        template = RollingShutterSync(
            {}, frame_readout_time_ms=self.params.frame_readout_time, fps=self.params.fps
        )
        K = camera_matrix if not np.allclose(camera_matrix, np.eye(3)) else None
        added = 0
        for a, b in zip(ordered, ordered[1:]):
            if a.prev_points is None or a.curr_points is None or len(a.prev_points) < 2:
                continue
            template.add_track_from_frames(
                a.timestamp_us, b.timestamp_us,
                a.prev_points, a.curr_points, float(height), camera_matrix=K,
            )
            added += 1
        if added < 3:
            log.warning("Orientation guess: only %d usable tracks, skipping", added)
            return None

        ts_all = [f.timestamp_us for f in ordered if f.timestamp_us > 0]
        windows = []
        n_win = min(6, max(1, len(ts_all) // 2 - 1))
        for k in range(n_win):
            i = int((k + 0.5) / n_win * (len(ts_all) - 2))
            windows.append((ts_all[i] / 1e6, ts_all[i + 2] / 1e6))

        orig_method = self.gyro.integration_method
        best: tuple[float, str] | None = None
        for oi, orient in enumerate(self._POSSIBLE_ORIENTATIONS):
            self.gyro.imu_transforms.imu_orientation = orient
            self.gyro.integrate()
            rs = RollingShutterSync(
                self.gyro.quaternions,
                frame_readout_time_ms=self.params.frame_readout_time,
                fps=self.params.fps,
            )
            rs.tracks = template.tracks
            try:
                cost = sum(rs._compute_cost(0.0, lo, hi) for lo, hi in windows)
            except Exception:
                cost = float("inf")
            if best is None or cost < best[0]:
                best = (cost, orient)
            if progress_callback:
                progress_callback((oi + 1) / len(self._POSSIBLE_ORIENTATIONS))

        # Restore/apply the winner and recompute downstream state.
        self.gyro.imu_transforms.imu_orientation = best[1]
        self.gyro.integration_method = orig_method
        self.gyro.integrate()
        self._invalidate_smoothing()
        log.info("Orientation guess: best '%s' (cost %.4f)", best[1], best[0])
        return best[1]

    # ------------------------------------------------------------------
    # Frame transform
    # ------------------------------------------------------------------

    def get_frame_transform(
        self,
        timestamp_ms: float,
        frame: int,
        compute_params: "ComputeParams | None" = None,
    ) -> FrameTransform:
        """Get stabilization transform for a single frame.

        Args:
            timestamp_ms: Frame center timestamp in milliseconds.
            frame: Frame index for FOV/IBIS lookup.
            compute_params: Pre-built snapshot for render loops. Passing it
                skips the per-frame rebuild (which copies the whole
                quaternion dicts — ~25k entries × per frame on DJI).

        Returns:
            FrameTransform with matrices and kernel params.
        """
        cp = compute_params if compute_params is not None else self._build_compute_params()
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
                - audio: Copy audio streams to the output (default: True)
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

        # Audio streams must be added before the first packet is muxed (the
        # container header is written on first mux); packets are copied
        # after the video pass.
        if options.get("audio", True):
            try:
                proc.prepare_audio()
            except Exception:
                log.warning("Audio preparation failed; output will be silent", exc_info=True)

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

        # Render-loop ComputeParams is built ONCE (the per-frame rebuild
        # copied the full quaternion dicts — ~25k entries on DJI per frame).
        render_cp = self._build_compute_params()

        def stabilize_frame(frame_data, timestamp_ms, frame_idx):
            # Use the real decode timestamp (pts) from the demuxer — covers
            # variable frame rate videos. FfmpegProcessor already falls back
            # to frame-index timing when pts is missing.
            transform = self.get_frame_transform(timestamp_ms, frame_idx, compute_params=render_cp)

            if gpu_backend is not None:
                # Patch kernel params for GPU shader requirements
                kp = transform.kernel_params
                channels = frame_data.shape[2] if frame_data.ndim == 3 else 1
                kp.output_stride = out_w * channels
                kp.max_pixel_value = 255.0
                kp.pixel_value_limit = 255.0
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

        # Copy audio packets through the streams prepared before the video
        # pass. Previously the output was always silent.
        if options.get("audio", True):
            try:
                proc.copy_audio()
            except Exception:
                log.warning("Audio copy failed; output will be silent", exc_info=True)

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

        # Camera diagonal FOV per frame from the lens intrinsics (mirrors
        # upstream ComputeParams::calculate_camera_fovs, consumed by
        # DefaultAlgo's velocity normalization: fov_ratio = dfov / 120).
        w_px = float(w)
        h_px = float(h)
        diag_px = (w_px * w_px + h_px * h_px) ** 0.5
        fy = camera_matrix[1, 1] if camera_matrix[1, 1] > 0 else 1.0
        diagonal_fov = 2.0 * float(np.arctan(diag_px / (2.0 * fy))) * 180.0 / np.pi

        # Radial distortion limit from the lens's distortion model (mirrors
        # upstream lens_profile.rs: DistortionModel::from_name(...).radial_distortion_limit(&coeffs))
        radial_limit = self._get_radial_distortion_limit(
            lens.distortion_model or "opencv_fisheye", distortion_coeffs
        )

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
            keyframes=self.keyframes,
            sync_offsets_adjusted=dict(self.gyro.offsets_adjusted),
            fovs=list(self.params.fovs),
            minimal_fovs=list(self.params.minimal_fovs),
            camera_diagonal_fovs=[diagonal_fov],
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
            radial_distortion_limit=radial_limit,
        )

    def _get_radial_distortion_limit(self, model_name: str, coeffs: list[float]) -> float:
        """Compute (and cache) the radial distortion limit for a lens.

        Returns 0.0 when the model reports no limit (distortion is valid
        over the whole field of view).
        """
        cache_key = (model_name, tuple(coeffs))
        if cache_key not in self._radial_limit_cache:
            from pygyroflow.stabilization.distortion_models import from_name as dm_from_name

            limit = dm_from_name(model_name).radial_distortion_limit(coeffs)
            self._radial_limit_cache[cache_key] = float(limit) if limit is not None else 0.0
        return self._radial_limit_cache[cache_key]

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
        """Auto-load a lens profile matching the detected camera model.

        Uses the telemetry's detected_source ("GoPro HERO12 Black",
        "DJI Osmo Nano", ...) and the video aspect ratio: the database
        search ranks same-aspect calibrations first, mirroring Gyroflow's
        automatic profile suggestion. Skipped when telemetry already
        provided a lens profile (e.g. DJI's embedded one).
        """
        # A telemetry-provided lens (non-zero calib dimension) wins.
        if self.lens.calib_dimension.get("w", 0) > 0:
            return

        source = getattr(self.gyro.file_metadata, "detected_source", None) if self.gyro.file_metadata else None
        if not source:
            return
        # Guard against placeholder sources: a bare "Unknown" would
        # literally match profiles whose name contains the word.
        words = source.split()
        if len(words) < 2:
            return
        brand = words[0].lower()
        known_brands = {prof.camera_brand.strip().lower() for _k, prof in self.lens_db.profiles if prof.camera_brand}
        if brand not in known_brands:
            return

        if not self.lens_db.loaded:
            self.lens_db.load_all()
        if not len(self.lens_db):
            return

        w, h = self.params.size
        aspect = (w / h) if w > 0 and h > 0 else None

        results = self.lens_db.search(source, aspect_ratio=aspect, limit=50)
        if not results:
            log.info("Auto lens: no match for '%s'", source)
            return

        best = results[0]
        log.info(
            "Auto lens: matched '%s' for '%s' (calib %sx%s)",
            best.get_display_name(), source,
            best.calib_dimension.get("w"), best.calib_dimension.get("h"),
        )
        try:
            self.load_lens_profile_by_object(best)
        except Exception as exc:
            log.warning("Auto lens load failed: %s", exc)

    def load_lens_profile_by_object(self, profile: LensProfile) -> None:
        """Adopt an already-loaded LensProfile (auto-match path)."""
        self.lens = profile
        self.lens.resolve_interpolations(self.lens_db)

        # Swap for portrait/landscape like the name-based loader
        w, h = self.params.size
        lens_w = self.lens.calib_dimension["w"]
        lens_h = self.lens.calib_dimension["h"]
        if w == lens_h and h == lens_w:
            self.lens = self.lens.swapped()

        self._invalidate_zooming()

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
