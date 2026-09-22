"""StabilizationManager — the central orchestrator for the entire pipeline.

Port of Gyroflow's src/core/lib.rs StabilizationManager struct.
Ties together gyro source, lens profile, smoothing, keyframes, adaptive zoom,
and rendering into a single cohesive workflow.

The Python version simplifies the threading model (no Arc<RwLock>) since
Python's GIL provides sufficient synchronization for the intended use cases
(single-threaded CLI, notebook, or GUI-driven processing).
"""

from __future__ import annotations

import datetime
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pygyroflow.gyro_source import GyroSource, FileMetadata
from pygyroflow.lens import LensProfile, LensProfileDatabase
from pygyroflow.smoothing import Smoothing, get_max_angles
from pygyroflow.keyframes import KeyframeManager, KeyframeType
from pygyroflow.stabilization import ComputeParams, FrameTransform
from pygyroflow.stabilization_params import StabilizationParams
from pygyroflow.types.enums import ReadoutDirection
from pygyroflow.types.errors import GyroflowError, TelemetryParseError, VideoIOError
from pygyroflow.util import ClosestMap

log = logging.getLogger(__name__)


def _keyframed_or(keyframes, key, timestamp_ms: float, default: float) -> float:
    """Keyframed value at *timestamp_ms*, falling back to *default*.

    Upstream's ``value_at_gyro_timestamp(...).unwrap_or(param)``: a keyframe
    overrides the parameter, absence of one leaves it alone.
    """
    if keyframes is None:
        return default
    value = keyframes.value_at_gyro_timestamp(key, timestamp_ms)
    return default if value is None else float(value)


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

        # Note: no short-clip integrator demotion — a 9 s Hero6 clip A/B
        # (offset fixed at the measured optimum) showed VQF at least as
        # stable as Complementary (p90 5.72 vs 6.70, mid-p90 9.03 vs 9.05),
        # and upstream Gyroflow always defaults to VQF.

        self.keyframes.clear()

    def load_video(self, path: str, fps: float | None = None) -> dict:
        """Load a video (or image sequence) and extract telemetry.

        Opens the input, reads metadata, extracts gyro data, and auto-detects
        a lens profile if possible.

        Args:
            path: Video file, or an image sequence (directory, printf pattern
                such as ``shots/frame_%04d.exr``, or a single frame).
            fps: Frame rate to assume for an image sequence.  Sequences carry
                no rate of their own; without this FFmpeg assumes 25 fps and
                the gyro timeline runs at the wrong speed.  Ignored for video.

        Returns:
            Dict with video metadata (width, height, fps, duration_ms,
            frame_count, image_sequence).

        Raises:
            VideoIOError: If the input cannot be opened.
            TelemetryParseError: If telemetry parsing fails.
        """
        import os

        from pygyroflow.rendering.image_sequence import looks_like_image_sequence

        if not os.path.isfile(path) and not looks_like_image_sequence(path):
            # A directory or printf pattern is not a file; only image
            # sequences get a second chance.
            raise VideoIOError(f"File not found: {path}")

        # Try to get video metadata via PyAV
        video_info = self._get_video_info(path, fps=fps)

        if video_info["width"] <= 0 or video_info["height"] <= 0 or video_info["duration_ms"] <= 0:
            raise VideoIOError(f"Invalid video metadata: {video_info}")

        width = video_info["width"]
        height = video_info["height"]
        fps = video_info["fps"]
        duration_ms = video_info["duration_ms"]
        frame_count = video_info["frame_count"]

        self.init_from_video_data(duration_ms, fps, frame_count, (width, height))


        sequence = video_info.get("image_sequence")

        # Container display rotation. A phone or action camera stores it in
        # the tkhd matrix and the player applies it at playback; ignoring it
        # renders portrait footage on its side. Upstream reads it with
        # av_display_rotation_get and folds it in as (360 - rotation) % 360
        # (render_queue.rs). Image sequences and stills have no container.
        if sequence is None:
            from pygyroflow.rendering.container_metadata import (
                read_display_rotation,
            )

            rotation = read_display_rotation(path)
            if rotation:
                self.params.video_rotation = (360.0 - rotation) % 360.0
                log.info(
                    "Container display rotation %.1f deg -> video_rotation %.1f",
                    rotation, self.params.video_rotation,
                )
        if sequence is not None:
            self.input_file.image_sequence_fps = fps
            self.input_file.image_sequence_start = sequence.start_number
            log.info(
                "Image sequence: %d frame(s) from %s starting at %d, %.2f fps",
                sequence.frame_count, sequence.pattern,
                sequence.start_number, fps,
            )
            if not sequence.is_sequence:
                log.info("Single still image input — output will be a one-frame video")
            # Frames carry no telemetry; a separate gyro source must be
            # supplied by the caller (manager.load_gyro_data with a real
            # telemetry file), otherwise only lens correction applies.
        else:
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

    # ------------------------------------------------------------------
    # .gyroflow project files
    # ------------------------------------------------------------------

    def load_project(self, path: str) -> None:
        """Adopt a `.gyroflow` project.

        Applies the settings this port models and keeps the rest — the raw
        sections round-trip untouched through :meth:`save_project`, so a file
        written by a newer Gyroflow is not silently reduced.
        """
        from pygyroflow.project import GyroflowProject, resolve_videofile

        proj = GyroflowProject.load(path)
        self.project = proj
        self.input_file.project_file_url = path
        if proj.videofile:
            self.input_file.url = resolve_videofile(
                proj.videofile, path, proj.image_sequence_start
            )
        if proj.image_sequence_fps:
            self.input_file.image_sequence_fps = proj.image_sequence_fps
        if proj.image_sequence_start:
            self.input_file.image_sequence_start = proj.image_sequence_start

        info = proj.video_info
        if info.width and info.height:
            self.params.size = (info.width, info.height)
        if info.fps:
            self.params.fps = info.fps
        if info.num_frames:
            self.params.frame_count = info.num_frames
        if info.duration_ms:
            self.params.duration_ms = info.duration_ms
        if info.fps_scale is not None:
            self.params.fps_scale = info.fps_scale
        self.params.video_rotation = info.rotation

        if proj.calibration_data:
            self._apply_project_calibration(proj.calibration_data)

        stab = proj.stabilization
        if stab:
            self._apply_project_stabilization(stab)

        gyro_src = proj.gyro_source
        # Embedded motion first, transforms after. `load_from_telemetry`
        # calls `GyroSource.clear()`, which resets the IMU transform — so
        # applying the project's rotation/lpf before loading the data would
        # throw it away again.
        self.gyro.init_from_params(self.params.duration_ms)
        if self._load_project_motion(proj):
            log.info(
                "Project carries its own gyro: %d quaternion(s), %d IMU sample(s)",
                len(self.gyro.quaternions), len(self.gyro.raw_imu),
            )
        if gyro_src:
            self._apply_project_gyro_source(gyro_src)
            # load_from_telemetry set integration_method from what it loaded;
            # the project's record of how it was integrated comes after.
            if len(self.gyro.quaternions) and "integration_method" in gyro_src:
                self.gyro.integration_method = int(gyro_src["integration_method"])

        self.gyro.set_offsets({int(k): float(v) for k, v in proj.offsets.items()})
        # Project load replaces the offset curve wholesale — the keyframe
        # manager's mirror has to follow (upstream's set_offset wrappers do
        # this on every mutation).
        self.keyframes.update_gyro(self.gyro.get_offsets())

        # float32 ndarray, not a tuple: `_build_compute_params` copies it with
        # `.copy()`, which a tuple does not have.
        self.params.background = np.asarray(
            proj.background_color, dtype=np.float32
        )
        self.params.background_mode = proj.background_mode
        self.params.background_margin = proj.background_margin
        self.params.background_margin_feather = proj.background_margin_feather
        self.params.light_refraction_coefficient = proj.light_refraction_coefficient

        duration = proj.video_info.duration_ms or self.params.duration_ms
        if proj.trim_ranges_ms:
            # A negative end is relative to the clip's end, not a wrap-around;
            # import_gyroflow_data does the same fold (lib.rs).
            self.params.trim_ranges = [
                (
                    (a / duration, (duration + b if b < 0.0 else b) / duration)
                    if duration
                    else (0.0, 1.0)
                )
                for a, b in proj.trim_ranges_ms
            ]
        elif proj.trim_start or proj.trim_end != 1.0:
            self.params.trim_ranges = [(proj.trim_start, proj.trim_end)]
        else:
            self.params.trim_ranges = []

        log.info(
            "Loaded project %s (version %d, %d field(s))",
            path, proj.version, len(proj.to_dict()),
        )

    def apply_preset(self, source: str | dict) -> None:
        """Apply a preset onto the already-loaded clip.

        A Gyroflow preset is a `.gyroflow` whose `videofile` is empty —
        ``detect_types`` in cli.rs routes exactly that shape to the preset
        list. `source` may be such a path, an inline JSON object as a string,
        or an already-parsed dict (upstream's `--preset` accepts a file or
        the content directly, and rewrites single quotes to double).

        Deliberately *not* applied: `video_info`, and any `offsets`. A preset
        is meant to be reused across clips, so letting it set the clip's
        dimensions or its resolved sync offsets would transplant one clip's
        facts onto another. The render reads the real dimensions from the
        decoded file anyway, which is why upstream gets away with copying
        them.
        """
        import json

        from pygyroflow.project import GyroflowProject

        if isinstance(source, dict):
            data = source
        else:
            text = str(source)
            if not text.lstrip().startswith("{"):
                with open(text, encoding="utf-8") as handle:
                    text = handle.read()
            # Strict first; the single-quoted rewrite is only a fallback, so
            # an apostrophe inside a string value survives.
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = json.loads(text.replace("'", '"'))

        proj = GyroflowProject.from_dict(data)
        if proj.calibration_data:
            self._apply_project_calibration(proj.calibration_data)
        if proj.stabilization:
            self._apply_project_stabilization(proj.stabilization)
        if proj.gyro_source:
            self._apply_project_gyro_source(proj.gyro_source)

        if proj.synchronization:
            # Upstream lands a preset's sync section in the lens profile's
            # sync_settings (render_queue.rs::update_sync_settings), merged
            # over whatever the profile already carried.
            merged = dict(self.lens.sync_settings or {})
            merged.update(proj.synchronization)
            self.lens.sync_settings = merged

        if proj.output:
            project = getattr(self, "project", None)
            if project is None:
                project = self.project = GyroflowProject()
            project.output.update(proj.output)

        log.info(
            "Applied preset: %s",
            ", ".join(
                name
                for name, section in (
                    ("calibration_data", proj.calibration_data),
                    ("stabilization", proj.stabilization),
                    ("gyro_source", proj.gyro_source),
                    ("synchronization", proj.synchronization),
                    ("output", proj.output),
                )
                if section
            )
            or "no recognised sections",
        )

    def _load_project_motion(self, proj) -> bool:
        """Load a project's embedded gyro into the gyro source.

        Returns True when something was loaded. This mirrors the branch order
        of ``import_gyroflow_data`` (lib.rs), which is worth spelling out
        because the two payload shapes are easy to conflate:

        1. The project counts as carrying data if its gyro file is a *different*
           file from the video, or if the decoded ``file_metadata`` has motion.
        2. Its ``gyro_source`` blobs are bincode, and hold either a full IMU
           stream (``raw_imu``) or pre-integrated ``quaternions``.
        3. ``file_metadata`` is CBOR and holds the whole metadata struct. It is
           the *fallback*: upstream uses the bincode ``raw_imu`` when that is
           non-empty and only reaches for ``file_metadata`` when it is not.

        That last ordering is what a real Gyroflow 1.6.3 export needs. Those
        write ``gyro_source.raw_imu``/``quaternions`` as ``null`` and keep
        everything in ``file_metadata`` plus ``integrated_quaternions``, so a
        loader that only knows the bincode branch finds no gyro at all in a
        project that has 25185 samples of it.

        Not implemented: upstream's final fallback, loading the gyro data out
        of a separate ``gyro_source.filepath`` file. This port has no sidecar
        loader, so a project in that shape is reported instead of silently
        coming back empty.
        """
        from pygyroflow.gyro_source import FileMetadata
        from pygyroflow.types.quaternion import Quat64
        from pygyroflow.types.time_types import TimeIMU

        gyro_src = proj.gyro_source or {}
        built_in = None
        if isinstance(gyro_src.get("file_metadata"), str):
            try:
                built_in = proj.read_blob("file_metadata")
            except Exception:
                log.warning("Could not decode file_metadata", exc_info=True)

        gyro_path = gyro_src.get("filepath") or ""
        gyro_is_another_file = bool(gyro_path) and gyro_path != (proj.videofile or "")
        bincode_blobs_present = any(
            gyro_src.get(name)
            for name in ("quaternions", "raw_imu", "gravity_vectors", "image_orientations")
        )
        # Upstream's gate is the first two clauses. The third is this port's:
        # our own writer has produced projects that carry a `quaternions` blob
        # without a `file_metadata` (upstream always writes both for a non-Simple
        # export, so it has no equivalent shape). Without it such a project
        # loads as empty.
        if not (gyro_is_another_file or bincode_blobs_present
                or (built_in is not None and built_in.has_motion())):
            return False

        quats = proj.read_blob("quaternions")
        raw_imu = proj.read_blob("raw_imu")
        gravity = proj.read_blob("gravity_vectors")
        orientations = proj.read_blob("image_orientations")

        def as_quats(blob):
            if not blob:
                return {}
            return {
                int(ts): Quat64.from_quaternion(np.asarray(q, dtype=np.float64))
                for ts, q in blob.items()
            }

        metadata = FileMetadata(
            imu_orientation=gyro_src.get("imu_orientation"),
            raw_imu=[
                TimeIMU(
                    timestamp_ms=sample[0],
                    gyro=sample[1],
                    accl=sample[2],
                    magn=sample[3],
                )
                for sample in (raw_imu or [])
            ],
            quaternions=as_quats(quats),
            gravity_vectors=(
                {
                    int(ts): np.asarray(v, dtype=np.float64)
                    for ts, v in gravity.items()
                }
                if gravity
                else None
            ),
            image_orientations=as_quats(orientations) or None,
            detected_source=gyro_src.get("detected_source"),
        )

        # Upstream only uses these blobs when `raw_imu` is non-empty, and
        # otherwise falls through to re-reading the video's telemetry. This
        # port uses whatever the blobs carry — a project that embedded
        # `quaternions` said what it wanted, and the alternative here would be
        # to go and parse a clip the caller may not even have loaded.
        if (metadata.raw_imu or metadata.quaternions
                or metadata.gravity_vectors or metadata.image_orientations):
            self.gyro.load_from_telemetry(metadata)
            return True

        if built_in is not None:
            self.gyro.load_from_telemetry(built_in)
            return True

        if gyro_is_another_file:
            log.info(
                "Project points at a separate gyro file (%s); this port has no "
                "sidecar loader, so no gyro data was read",
                gyro_path,
            )
        return False

    def output_options(self) -> dict:
        """The project's `output` section, for a caller building render options.

        Returns an empty dict when no project has been loaded or created, so
        `render` options can be merged over it unconditionally.
        """
        project = getattr(self, "project", None)
        return dict(project.output) if project is not None else {}

    def _apply_project_calibration(self, calibration_data: dict) -> None:
        """Build a LensProfile from the project's calibration_data."""
        from pygyroflow.lens import LensProfile

        try:
            self.lens = LensProfile.from_json(dict(calibration_data))
        except Exception:
            log.warning(
                "Could not load the project's calibration data", exc_info=True
            )

    def _apply_project_gyro_source(self, gyro_src: dict) -> None:
        """Apply the `gyro_source` section's IMU transforms."""
        t = self.gyro.imu_transforms
        if "lpf" in gyro_src:
            t.imu_lpf = float(gyro_src["lpf"] or 0.0)
        if "mf" in gyro_src:
            t.imu_mf = int(gyro_src["mf"] or 0)
        if gyro_src.get("rotation"):
            t.imu_rotation_angles = tuple(float(v) for v in gyro_src["rotation"])
        if gyro_src.get("acc_rotation"):
            t.acc_rotation_angles = tuple(
                float(v) for v in gyro_src["acc_rotation"]
            )
        if gyro_src.get("imu_orientation"):
            t.imu_orientation = str(gyro_src["imu_orientation"])
        if gyro_src.get("gyro_bias"):
            t.gyro_bias = [float(v) for v in gyro_src["gyro_bias"]]
        if "integration_method" in gyro_src:
            self.gyro.integration_method = int(gyro_src["integration_method"])
        if "sample_index" in gyro_src and gyro_src["sample_index"] is not None:
            self.gyro.file_load_options.sample_index = int(gyro_src["sample_index"])

    @staticmethod
    def _readout_direction_from_project(value, fallback: ReadoutDirection) -> ReadoutDirection:
        """Parse ``stabilization.frame_readout_direction`` from a project.

        Real Gyroflow 1.6.3 exports write the variant **name**
        (``"TopToBottom"``), because serde serializes a unit enum as its
        variant name; the field is not an integer in any file this port has
        seen. Upstream accepts both — ``as_i64`` first, then ``as_str`` — so
        both are accepted here too, and anything else keeps the previous value
        rather than failing the load.
        """
        if isinstance(value, str):
            try:
                return ReadoutDirection[value]
            except KeyError:
                log.warning("Unknown frame_readout_direction %r in project", value)
                return fallback
        try:
            return ReadoutDirection(int(value))
        except (ValueError, TypeError):
            log.warning("Unknown frame_readout_direction %r in project", value)
            return fallback

    def _apply_project_stabilization(self, stab: dict) -> None:
        """Apply the `stabilization` section to params and the smoothing algo."""
        p = self.params

        def get(key: str, default=None):
            value = stab.get(key, default)
            return default if value is None else value

        # Keyframes first: the smoothing-parameter keys below may be keyframed,
        # and upstream's ComputeParams serde carries the KeyframeManager inside
        # this same section (serialize shape: keyframes.rs:76-83).
        if isinstance(stab.get("keyframes"), dict):
            self.keyframes.deserialize(stab["keyframes"])

        p.fov = float(get("fov", p.fov))
        # The sign is kept, not dropped: upstream reads a negative readout time
        # as "this sensor reads bottom to top" and nothing else carries that.
        # Every consumer inside frame_transform takes abs() itself.
        readout_time = float(get("frame_readout_time", p.frame_readout_time))
        p.frame_readout_time = readout_time
        if readout_time < 0.0:
            p.frame_readout_direction = ReadoutDirection.BottomToTop
        if "frame_readout_direction" in stab:
            p.frame_readout_direction = self._readout_direction_from_project(
                stab["frame_readout_direction"], p.frame_readout_direction
            )
        p.adaptive_zoom_window = float(get("adaptive_zoom_window", p.adaptive_zoom_window))
        if stab.get("adaptive_zoom_center_offset"):
            p.adaptive_zoom_center_offset = tuple(
                float(v) for v in stab["adaptive_zoom_center_offset"]
            )
        if "adaptive_zoom_method" in stab:
            p.adaptive_zoom_method = int(stab["adaptive_zoom_method"])
        if stab.get("additional_rotation"):
            p.additional_rotation = tuple(float(v) for v in stab["additional_rotation"])
        if stab.get("additional_translation"):
            p.additional_translation = tuple(
                float(v) for v in stab["additional_translation"]
            )
        p.lens_correction_amount = float(
            get("lens_correction_amount", p.lens_correction_amount)
        )
        if "max_zoom" in stab:
            p.max_zoom = float(stab["max_zoom"]) if stab["max_zoom"] is not None else None
        if "max_zoom_iterations" in stab:
            p.max_zoom_iterations = int(stab["max_zoom_iterations"])
        for flag in (
            "video_speed_affects_smoothing",
            "video_speed_affects_zooming",
            "video_speed_affects_zooming_limit",
        ):
            if flag in stab:
                setattr(p, flag, bool(stab[flag]))
        if "video_speed" in stab:
            p.video_speed = float(stab["video_speed"] or 1.0)

        amount = float(stab.get("horizon_lock_amount") or 0.0)
        roll = float(stab.get("horizon_lock_roll") or 0.0)
        pitch_enabled = bool(stab.get("horizon_lock_pitch_enabled") or False)
        pitch = float(stab.get("horizon_lock_pitch") or 0.0)
        self.smoothing.horizon_lock.set_horizon(
            lock_percent=amount, roll=roll,
            lock_pitch=pitch_enabled, pitch=pitch,
        )

        method = stab.get("method")
        if method:
            names = self.smoothing.get_names()
            if method in names:
                self.smoothing.set_current(names.index(method))
            else:
                log.warning("Unknown smoothing method %r in project", method)
        for entry in stab.get("smoothing_params") or []:
            if isinstance(entry, dict) and "name" in entry:
                self.smoothing.current().set_parameter(
                    entry["name"], float(entry["value"])
                )

        if bool(get("use_gravity_vectors", False)):
            self.gyro.use_gravity_vectors = True
        if "horizon_lock_integration_method" in stab:
            self.gyro.horizon_lock_integration_method = int(
                stab["horizon_lock_integration_method"]
            )

    def save_project(self, path: str, project_type: str | None = None) -> None:
        """Write the current state out as a `.gyroflow` project.

        Sections the loader kept verbatim are written back unchanged; the
        ones this port owns are rebuilt from live state.

        `project_type` mirrors upstream's `GyroflowProjectType`:

        ``None``
            Keep whatever payloads the project already carried. The safe
            default for a load-modify-save round trip.
        ``"simple"``
            Drop every embedded motion payload, leaving the settings only —
            what ``export_gyroflow_data(Simple)`` produces.
        ``"with_gyro_data"``
            Embed the metadata as a CBOR ``file_metadata`` blob, so the
            project can be re-loaded without the original clip's telemetry.
        ``"with_processed_data"``
            The above plus the caches a plugin reads —
            ``integrated_quaternions``, ``smoothed_quaternions``,
            ``adaptive_zoom_fovs``, the two ``synced_imu_timestamps*``
            timelines and the focal length curves.

        Upstream never writes the ``gyro_source.quaternions``/``raw_imu``/
        ``gravity_vectors`` blobs in any mode; they are legacy fields it only
        reads (``import_gyroflow_data`` even removes them after loading), so
        modes 2 and 3 drop any old copy rather than leave one to disagree with
        the metadata written beside it.

        `app_version` and `date` are stamped from the writer, as upstream's
        `export_gyroflow_data` does — they describe who produced the file,
        so carrying the loaded file's values over would be a false claim.

        Returns:
            The path actually written. A ``"simple"`` export of a project
            whose ``videofile`` is empty (i.e. a preset) gets a ``.gyroflow``
            name; otherwise `path` is used as given.
        """
        import pygyroflow
        from pygyroflow.project import PROJECT_VERSION, GyroflowProject

        if project_type not in (None, "simple", "with_gyro_data", "with_processed_data"):
            raise ValueError(f"Unknown project_type: {project_type!r}")

        proj = getattr(self, "project", None) or GyroflowProject()
        self.project = proj

        proj.version = PROJECT_VERSION
        proj.app_version = pygyroflow.__version__
        proj.date = datetime.date.today().isoformat()
        if not proj.videofile:
            proj.videofile = self.input_file.url
        if self.lens is not None:
            try:
                proj.calibration_data = self.lens.get_json_value()
            except Exception:
                log.warning("Could not serialise the lens into the project", exc_info=True)

        p = self.params
        # Only overwrite the sections the caller actually owns; a project
        # loaded from disk keeps its extra keys.
        proj.video_info = type(proj.video_info).from_dict(
            {
                **proj.video_info.to_dict(),
                "width": p.size[0],
                "height": p.size[1],
                "rotation": p.video_rotation,
                "num_frames": p.frame_count,
                "fps": p.fps,
                "duration_ms": p.duration_ms,
                "fps_scale": p.fps_scale,
                "vfr_fps": p.get_scaled_fps(),
                "vfr_duration_ms": p.get_scaled_duration_ms(),
                "created_at": p.video_created_at,
            }
        )
        proj.stabilization = {
            **proj.stabilization,
            "keyframes": self.keyframes.serialize(),
            "fov": p.fov,
            "method": self.smoothing.current().get_name(),
            "smoothing_params": [
                {"name": e["name"], "value": e["value"]}
                for e in self.smoothing.current().get_parameters_json()
                if isinstance(e, dict) and "name" in e and "value" in e
            ],
            "frame_readout_time": abs(p.frame_readout_time),
            # The variant name, not its number: serde writes a unit enum that
            # way and a real export shows `"TopToBottom"`. The reader accepts
            # both, but a file we write should look like one Gyroflow writes.
            "frame_readout_direction": p.frame_readout_direction.name,
            "adaptive_zoom_window": p.adaptive_zoom_window,
            "adaptive_zoom_center_offset": list(p.adaptive_zoom_center_offset),
            "adaptive_zoom_method": p.adaptive_zoom_method,
            "additional_rotation": list(p.additional_rotation),
            "additional_translation": list(p.additional_translation),
            "lens_correction_amount": p.lens_correction_amount,
            "horizon_lock_amount": (
                self.smoothing.horizon_lock.horizonlockpercent
                if self.smoothing.horizon_lock.lock_enabled else 0.0
            ),
            "horizon_lock_roll": self.smoothing.horizon_lock.horizonroll,
            "horizon_lock_pitch_enabled": self.smoothing.horizon_lock.lock_pitch,
            "horizon_lock_pitch": self.smoothing.horizon_lock.horizonpitch,
            "use_gravity_vectors": self.gyro.use_gravity_vectors,
            "horizon_lock_integration_method": self.gyro.horizon_lock_integration_method,
            "video_speed": p.video_speed,
            "video_speed_affects_smoothing": p.video_speed_affects_smoothing,
            "video_speed_affects_zooming": p.video_speed_affects_zooming,
            "video_speed_affects_zooming_limit": p.video_speed_affects_zooming_limit,
            "max_zoom": p.max_zoom,
            "max_zoom_iterations": p.max_zoom_iterations,
            # Added in project version 4.
            "frame_offset": p.frame_offset,
            "focal_length_smoothing_enabled": p.focal_length_smoothing_enabled,
            "focal_length_smoothing_strength": p.focal_length_smoothing_strength,
        }
        t = self.gyro.imu_transforms
        proj.gyro_source = {
            **proj.gyro_source,
            "filepath": self.gyro.file_url or self.input_file.url,
            "lpf": t.imu_lpf,
            "mf": t.imu_mf,
            "rotation": list(t.imu_rotation_angles) if t.imu_rotation_angles else None,
            "acc_rotation": list(t.acc_rotation_angles) if t.acc_rotation_angles else None,
            "imu_orientation": t.imu_orientation,
            "gyro_bias": list(t.gyro_bias) if t.gyro_bias else None,
            "integration_method": self.gyro.integration_method,
            # Added in project version 4.
            "sample_index": self.gyro.file_load_options.sample_index,
            "detected_source": getattr(self.gyro.file_metadata, "detected_source", None),
        }
        proj.background_color = list(p.background)
        proj.background_mode = int(p.background_mode)
        proj.background_margin = p.background_margin
        proj.background_margin_feather = p.background_margin_feather
        proj.light_refraction_coefficient = p.light_refraction_coefficient
        proj.offsets = {int(k): float(v) for k, v in self.gyro.get_offsets().items()}
        if p.trim_ranges:
            proj.trim_ranges_ms = [
                [a * p.duration_ms, b * p.duration_ms] for a, b in p.trim_ranges
            ]
            proj.trim_start, proj.trim_end = p.trim_ranges[0][0], p.trim_ranges[-1][1]
        else:
            proj.trim_ranges_ms = []
            proj.trim_start, proj.trim_end = 0.0, 1.0

        if project_type == "simple":
            proj.strip_motion_payloads()
        elif project_type in ("with_gyro_data", "with_processed_data"):
            processed = project_type == "with_processed_data"
            self._embed_motion_payloads(proj, keep_processed=processed)
            if processed:
                self._embed_processed_payloads(proj)

        proj.save(path)
        self.input_file.project_file_url = path
        log.info("Saved project %s", path)

    def _project_file_metadata(self):
        """The FileMetadata to embed in a project, from live state.

        Upstream writes `compress_to_base91_cbor(&*file_metadata)` — the gyro
        source's own metadata, whose `quaternions` field holds what the project
        should be able to re-load. Here that is `self.gyro.quaternions`, the
        integrated set: it is what a real export carries (in
        `DJI_20260507160359_0005_D.gyroflow` the two maps are identical), and
        it is what makes the file self-sufficient, since `has_motion()` is what
        a loader keys on.
        """
        from pygyroflow.gyro_source import FileMetadata

        source = self.gyro.file_metadata
        return FileMetadata(
            imu_orientation=source.imu_orientation,
            raw_imu=list(source.raw_imu),
            quaternions=dict(self.gyro.quaternions) or dict(source.quaternions),
            gravity_vectors=source.gravity_vectors,
            image_orientations=source.image_orientations,
            detected_source=source.detected_source,
            frame_readout_time=source.frame_readout_time,
            frame_readout_direction=source.frame_readout_direction,
            frame_rate=source.frame_rate,
            camera_identifier=source.camera_identifier,
            lens_profile=source.lens_profile,
            lens_positions=dict(source.lens_positions),
            lens_params=dict(source.lens_params),
            digital_zoom=source.digital_zoom,
            has_accurate_timestamps=source.has_accurate_timestamps,
            additional_data=dict(source.additional_data or {}),
            per_frame_time_offsets=list(source.per_frame_time_offsets),
            camera_stab_data=list(source.camera_stab_data),
            mesh_correction=list(source.mesh_correction),
        )

    def _embed_motion_payloads(self, proj, keep_processed: bool = False) -> None:
        """Write the `file_metadata` blob (modes 2 and 3).

        Upstream's non-Simple export does exactly this one insert; the
        `gyro_source.quaternions`/`raw_imu`/`gravity_vectors` blobs are
        *legacy* fields it only ever reads, so any old copy of them is dropped
        rather than left to disagree with the metadata beside it.

        The processed caches are dropped as well unless *keep_processed*: they
        are what separates mode 3 from mode 2, and the loader keeps unmodelled
        sections verbatim, so a mode-2 export of a project loaded from a mode-3
        file would otherwise carry them along.
        """
        from pygyroflow.project import _BINCODE_READERS, _CBOR_READERS

        metadata = self._project_file_metadata()
        proj.write_blob("file_metadata", metadata)
        for name in _BINCODE_READERS:
            proj.gyro_source.pop(name, None)
        if not keep_processed:
            for name in _CBOR_READERS:
                if name != "file_metadata":
                    proj.gyro_source.pop(name, None)

    def _embed_processed_payloads(self, proj) -> None:
        """Write the caches a plugin reads (mode 3).

        ``synced_imu_timestamps`` is the gyro timeline rebased onto the video's
        by the sync offsets, so a plugin can line the two up without redoing
        the sync. The ``_with_per_frame_offset`` twin subtracts the per-frame
        timestamp correction back off, which is the timeline the samples were
        actually captured on. Both are computed the way lib.rs does, frame
        index and ``ceil`` included.
        """
        params = self.params
        metadata = self._project_file_metadata()
        # `params.fovs` verbatim, empty included: recomputing the adaptive zoom
        # is the caller's job, and writing a made-up single value in its place
        # would be worse than writing none.
        fovs = list(params.fovs or [])

        synced = []
        synced_final = []
        readout_half = abs(params.frame_readout_time) / 2.0
        scaled_fps = params.get_scaled_fps()
        for timestamp_us in sorted(self.gyro.quaternions):
            timestamp_ms = timestamp_us / 1000.0
            timestamp_ms += self.gyro.offset_at_gyro_timestamp(timestamp_ms)
            synced.append(timestamp_ms)

            # `ceil` and the frame index, straight from lib.rs. A frame index
            # past the end has no offset to subtract, so it contributes none.
            frame = math.ceil((timestamp_ms - readout_half) * scaled_fps / 1000.0)
            correction = (
                metadata.per_frame_time_offsets[frame]
                if 0 <= frame < len(metadata.per_frame_time_offsets)
                else 0.0
            )
            synced_final.append(timestamp_ms - correction)

        proj.write_blob("integrated_quaternions", dict(self.gyro.quaternions))
        proj.write_blob("smoothed_quaternions", dict(self.gyro.smoothed_quaternions))
        proj.write_blob("adaptive_zoom_fovs", fovs)
        proj.write_blob("synced_imu_timestamps", synced)
        proj.write_blob(
            "synced_imu_timestamps_with_per_frame_offset", synced_final
        )
        for name, curve in (
            ("focal_lengths", params.focal_lengths),
            ("smoothed_focal_lengths", params.smoothed_focal_lengths),
        ):
            if curve:
                proj.write_blob(name, [v for v in curve])

    def recompute_smoothing(self) -> None:
        """Recompute smoothed quaternions.

        Mirrors upstream ``GyroSource::recompute_smoothness``, which does all
        of this in one place and in this order:

        1. multiply every original orientation by the additional rotation
           (keyframed, else ``params.additional_rotation``);
        2. lock the horizon **on the originals**, before smoothing;
        3. smooth;
        4. record the maximum angles over the trim range;
        5. store the correction quaternion ``smoothed⁻¹ * org``.

        Steps 2 and 5 used to be split across this method and
        ``Smoothing.smooth``, which applied the lock twice (once after
        smoothing there, again here) — and the additional rotation was never
        applied at all, so the GUI's horizon-roll control did nothing.
        """
        from pygyroflow.types.quaternion import Quat64

        cp = self._build_compute_params()
        org = self.gyro.quaternions

        # 1. Additional rotation, applied to a copy of the originals.
        keyframes = getattr(cp, "keyframes", None)
        base = cp.additional_rotation
        rotated: dict[int, Quat64] = {}
        for ts, q in org.items():
            ts_ms = ts / 1000.0
            angles = [
                _keyframed_or(
                    keyframes, kf_type, ts_ms, float(base[i])
                )
                for i, kf_type in enumerate(
                    (
                        KeyframeType.AdditionalRotationX,
                        KeyframeType.AdditionalRotationY,
                        KeyframeType.AdditionalRotationZ,
                    )
                )
            ]
            rot = Quat64.from_euler_angles(
                np.deg2rad(angles[1]),  # roll
                np.deg2rad(angles[0]),  # pitch
                np.deg2rad(angles[2]),  # yaw
            )
            rotated[ts] = q * rot

        # 2. Lock the horizon on the originals, then smooth (upstream's
        #    `if true` branch; the reverse order is upstream's dead branch).
        self._lock_horizon(rotated, org, cp)

        smoothed = self.smoothing.smooth(rotated, self.gyro.duration_ms, cp)

        # 3. Max angles over the trim range (upstream feeds these to the UI).
        try:
            self.gyro.max_angles = get_max_angles(org, smoothed, cp)
        except Exception:
            log.warning("Could not compute max angles", exc_info=True)

        # 4. Store the CORRECTION quaternion sm⁻¹ * org, not the smoothed
        #    orientation itself. FrameTransform composes it with the org
        #    lookups (smoothed * org_c⁻¹ * org_row = sm⁻¹ * org_row), which
        #    is the rotation that maps the raw frame back onto the smoothed
        #    path. Without this step the stabilization is ineffective.
        self.gyro.smoothed_quaternions = {
            ts: q.inverse() * org[ts]
            for ts, q in smoothed.items()
            if ts in org
        }

    def _lock_horizon(
        self,
        quats: dict,
        org_quats: dict,
        cp,
    ) -> None:
        """Apply the horizon lock in place, when it is enabled or keyframed.

        Quaternion mode by default; the gravity-vector mode needs an
        accelerometer series and stays opt-in (``--horizon-gravity``).
        """
        lock = self.smoothing.horizon_lock
        if not (lock.lock_enabled or cp.keyframes.is_keyframed(
            KeyframeType.LockHorizonAmount
        )):
            return

        grav = None
        use_grav = False
        if self.gyro.use_gravity_vectors:
            accl = [imu for imu in self.gyro.raw_imu if imu.accl is not None]
            if len(accl) >= 3:
                grav = {}
                for imu in accl:
                    v = np.asarray(imu.accl, dtype=np.float64)
                    n = np.linalg.norm(v)
                    if n > 1e-6:
                        # accelerometer at rest reads "up" (~+g); the gravity
                        # mode compares against +Y in sensor frame
                        grav[int(round(imu.timestamp_ms * 1000.0))] = v / n
                use_grav = len(grav) >= 3
                if not use_grav:
                    grav = None

        lock.lock(quats, org_quats=org_quats, grav=grav, use_grav=use_grav,
                  compute_params=cp)

    def recompute_adaptive_zoom(self) -> None:
        """Recompute adaptive zoom FOVs, then enforce the max-zoom limit."""
        self._smoothing_fov_limit_per_frame = []
        self._recompute_zoom_once()
        self._apply_max_zoom_limit()

    def _recompute_zoom_once(self) -> list[float]:
        """One adaptive-zoom pass; returns the FOVs it produced."""
        from pygyroflow.zooming import calculate_fovs, ZoomMethod

        cp = self._build_compute_params()
        frames = self.params.frame_count
        fps = self.params.get_scaled_fps()

        if frames <= 0 or fps <= 0:
            return []

        timestamps = [(i, i * 1000.0 / fps) for i in range(frames)]

        method = ZoomMethod.from_index(self.params.adaptive_zoom_method)
        fovs, minimal_fovs = calculate_fovs(cp, timestamps, method)

        lens_fov_adj = self.lens.optimal_fov or 1.0
        self.params.set_fovs(fovs, lens_fov_adj)
        self.params.minimal_fovs = minimal_fovs
        return list(fovs)

    def _apply_max_zoom_limit(self) -> None:
        """Relax smoothing where the required crop would exceed max zoom.

        Mirrors upstream's loop in ``StabilizationManager::recompute_adaptive_zoom``:
        when a frame needs more crop than the zoom limit allows, smoothing is
        relaxed for that frame instead (through
        ``ComputeParams.smoothing_fov_limit_per_frame``) and both stages are
        re-run, up to ``max_zoom_iterations`` times with progressively looser
        thresholds. Without it ``max_zoom`` was carried around and never used:
        the crop was whatever the zoom estimator asked for.
        """
        max_zoom_param = self.params.max_zoom or 0.0
        max_zoom_iters = int(self.params.max_zoom_iterations or 0)
        frames = self.params.frame_count
        fps = self.params.get_scaled_fps()
        if frames <= 0 or fps <= 0:
            return

        keyframes = self.keyframes
        keyed = keyframes.get_keyframes(KeyframeType.MaxZoom) if keyframes else None
        max_zoom_max = (
            max((kf.value for kf in keyed.values()), default=max_zoom_param)
            if keyed
            else max_zoom_param
        )
        if max_zoom_max <= 50.0 or max_zoom_iters <= 0:
            return

        out_w = self.params.output_size[0] or self.params.size[0]
        scaling_factor = self.params.size[0] / max(1, out_w)
        thresholds = (0.95, 0.9, 0.85, 0.8)

        limit = [1.0] * len(self.params.fovs)
        for iteration in range(max_zoom_iters):
            any_above_limit = False
            for i, fov in enumerate(self.params.fovs):
                ts = i * 1000.0 / fps
                zoom_limit = (
                    _keyframed_or(keyframes, KeyframeType.MaxZoom, ts, max_zoom_param)
                    / 100.0
                )
                if self.params.video_speed_affects_zooming_limit and (
                    self.params.video_speed != 1.0
                    or keyframes.is_keyframed(KeyframeType.VideoSpeed)
                ):
                    vid_speed = abs(
                        _keyframed_or(
                            keyframes, KeyframeType.VideoSpeed, ts,
                            self.params.video_speed,
                        )
                    )
                    zoom_limit *= min(1.0 + (vid_speed - 1.0) / 4.0, 1.8)

                fov_limit = 1.0 / (zoom_limit * scaling_factor) if zoom_limit else 0.0
                if fov_limit and fov < fov_limit:
                    any_above_limit = True
                    limit[i] *= min(
                        fov / fov_limit,
                        thresholds[min(iteration, len(thresholds) - 1)],
                    )

            if not any_above_limit:
                if iteration == 0:
                    limit = []  # never any conflict: leave smoothing alone
                break

            self._smoothing_fov_limit_per_frame = limit
            self.recompute_smoothing()
            self._recompute_zoom_once()

        self._smoothing_fov_limit_per_frame = limit

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

    def set_gyro_offset(self, timestamp_us: int, offset_ms: float) -> None:
        """Set a sync offset and mirror it into the keyframe manager.

        Upstream's ``GyroFlow::set_offset`` (lib.rs:1066-1070): the gyro
        source is mutated, then ``keyframes.update_gyro`` re-mirrors the
        offset curve — without that mirror, keyframed values queried on the
        gyro timeline (all the smoothing-path lookups) keep reading the
        pre-sync timeline.
        """
        self.gyro.set_offset(timestamp_us, offset_ms)
        self.keyframes.update_gyro(self.gyro.get_offsets())
        self._invalidate_zooming()

    def remove_gyro_offset(self, timestamp_us: int) -> None:
        """Remove a sync offset and mirror it into the keyframe manager."""
        self.gyro.remove_offset(timestamp_us)
        self.keyframes.update_gyro(self.gyro.get_offsets())
        self._invalidate_zooming()

    def clear_gyro_offsets(self) -> None:
        """Clear all sync offsets and mirror that into the keyframe manager."""
        self.gyro.clear_offsets()
        self.keyframes.update_gyro(self.gyro.get_offsets())
        self._invalidate_zooming()

    def synchronize(
        self,
        input_path: str | None = None,
        sample_count: int = 1000,
        search_range_ms: float = 500.0,
        use_rs: bool = True,
        progress_callback: Any = None,
        of_method: int = 2,
        offset_method: int | None = None,
        pose_method: int = 0,
    ) -> float | None:
        """Auto-synchronize the gyro timeline to the video via optical flow.

        Samples grayscale frames from the video (downscaled for speed),
        estimates camera rotation between consecutive frames, and searches
        for the time offset that best matches the gyro data. With ``use_rs``
        (and quaternion data available) this runs the rolling-shutter-aware
        per-point quaternion search; otherwise a 1-D angular-velocity
        cross-correlation is used.

        ``sample_count`` defaults to 1000: the RS-aware cost needs
        adjacent-frame optical-flow tracks. With the previous default of
        200, tracks spanned 4-frame baselines (~133 ms); during fast
        motion the correspondences break and the cost landscape flattens,
        producing wrong offsets (DJI walking clip: +179 ms found where the
        true offset is 0 - the landscape with per-frame tracks has a clean
        minimum at 0).

        The result is stored on the gyro source (``gyro.set_offset``) and
        thereby takes effect in ``get_frame_transform`` lookups.

        Args:
            input_path: Video to analyze (defaults to the loaded file).
            sample_count: Approximate number of frames to analyze.
            search_range_ms: Offset search window in ms.
            use_rs: Prefer the rolling-shutter-aware search.
            progress_callback: Optional Callable[[float], None].
            of_method: Optical-flow method index.
            offset_method: Offset-search method index; ``None`` picks 2 (RS)
                when ``use_rs`` else 1 (cross-correlation).
            pose_method: Pose-estimation method index. The faithful
                ``PoseFindEssentialMat`` (0) demands inter-frame parallax —
                a rotation-only synthetic scene gives it zero usable pairs —
                so a caller syncing such footage selects 2 instead.

        Returns:
            Offset in ms (``visual = gyro + offset``), or None on failure.
        """
        path = input_path or self.input_file.url
        if not path:
            raise GyroflowError("No input video loaded for synchronization")
        if not self.gyro.quaternions:
            log.warning("No gyro quaternions; skipping auto-sync")
            return None

        # Multi-point refinement needs >=10 frames per 1 s window. The frame
        # cap is a MEMORY bound: 7 GB machines OOM when a long 4K decode and a
        # ~3000-frame retained list stack on top of the loaded gyro data.
        want = sample_count
        if self.params.duration_ms >= 12_000:
            want = int(max(sample_count, min(1500, self.params.duration_ms / 1000.0 * 10)))
        frames = self._extract_gray_frames(path, want)
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
            of_method=of_method,
            pose_method=pose_method,
            offset_method=method if offset_method is None else offset_method,
            compute_params=self._build_sync_compute_params(),
        )

        log.info(
            "Auto-sync: analyzing %d frames (%dx%d, %.2fx scale)",
            len(frames), width, height, scale,
        )

        def _run(search_ms: float, prior_ms: float) -> float | None:
            return proc.run(
                frames,
                gyro_data,
                search_range_ms=search_ms,
                sample_count=None,  # already subsampled during extraction
                progress_callback=progress_callback,
                quaternions=dict(self.gyro.quaternions) if use_rs else None,
                frame_readout_time_ms=self.params.frame_readout_time,
                initial_offset_ms=prior_ms,
            )

        # DJI prior: try a narrow window around +8 ms first — a small search
        # space cannot accommodate the large flat-landscape mislocks — and
        # fall back to the full window when the narrow pass yields nothing.
        offset = None
        prior = (
            self._DJI_SYNC_PRIOR_MS
            if (self.gyro.file_metadata.detected_source or "").startswith("DJI")
            else None
        )
        if prior is not None:
            log.info("Auto-sync: DJI prior, narrow search +/-%.0f ms around %.0f ms",
                     self._DJI_SYNC_PRIOR_WINDOW_MS / 2, prior)
            offset = _run(self._DJI_SYNC_PRIOR_WINDOW_MS, prior)
            if offset is not None and np.isfinite(offset):
                # the RS fine refinement can wander outside the coarse grid;
                # an escapee is a flat-landscape artifact, not signal — the
                # measured prior (two independent DJI samples) is more
                # trustworthy than a wandering refinement
                if abs(offset - prior) > self._DJI_SYNC_PRIOR_WINDOW_MS / 2:
                    log.warning(
                        "Auto-sync: narrow refinement escaped the prior window "
                        "(%.1f ms); clamping to the %.0f ms prior",
                        offset, prior,
                    )
                    offset = prior
            if offset is None or not np.isfinite(offset):
                log.info("Auto-sync: narrow prior search failed; widening to +/-%.0f ms",
                         search_range_ms / 2)
                offset = None
        if offset is None:
            offset = _run(search_range_ms, 0.0)

        if offset is None or not np.isfinite(offset):
            log.warning("Auto-sync failed to find an offset")
            if prior is not None:
                offset = prior  # better than 0: every measured DJI sample sits at +8
                log.warning("Auto-sync: falling back to the DJI prior %.0f ms", prior)
            else:
                return None

        self.set_gyro_offset(0, float(offset))
        log.info("Auto-sync offset: %.2f ms", offset)

        # Multi-point refinement (upstream auto_sync_points / max_sync_points):
        # estimate the offset independently on rotation-rich 1 s windows and
        # keep a piecewise-linear offset curve to absorb gyro/video clock drift.
        points = self._refine_sync_points(
            frames, gyro_data, float(offset),
            quaternions=dict(self.gyro.quaternions) if use_rs else None,
            frame_readout_time_ms=self.params.frame_readout_time,
        )
        if len(points) >= 2:
            # set_offsets is the gyro source's bulk replace; mirror the curve
            # into the keyframe manager the same way set_gyro_offset does.
            self.gyro.set_offsets(points)
            self.keyframes.update_gyro(self.gyro.get_offsets())
            self._invalidate_zooming()
            vals = [points[k] for k in sorted(points)]
            log.info(
                "Auto-sync: %d sync points, %.1f..%.1f ms (span %.1f ms)",
                len(points), vals[0], vals[-1], vals[-1] - vals[0],
            )
        return float(offset)

    # Multi-point sync refinement (upstream parity: max_sync_points=5,
    # time_per_syncpoint=1 s, per-point offsets -> linear drift model).
    _SYNC_POINT_COUNT = 5
    _SYNC_POINT_WINDOW_MS = 500.0
    _SYNC_POINT_MAX_DEVIATION_MS = 40.0
    _SYNC_POINT_SEARCH_MS = 120.0
    _SYNC_POINT_MIN_DRIFT_MS = 15.0
    _SYNC_POINT_MAX_RESIDUAL_MS = 10.0
    # DJI cameras show a consistent ~+8 ms telemetry-vs-video offset
    # (Osmo Nano autosync lock 7.92 ms, Avata offset sweep optimum +8 ms).
    # A narrow window around the prior also guards against the FPV
    # full-range mislocks (Avata full clip: 105 ms on a flat landscape).
    _DJI_SYNC_PRIOR_MS = 8.0
    _DJI_SYNC_PRIOR_WINDOW_MS = 40.0

    @staticmethod
    def _valid_sync_points(
        points: dict[int, float],
        global_offset: float,
        max_deviation_ms: float = 40.0,
    ) -> dict[int, float]:
        """Global point at t=0 plus inliers within max_deviation of it.

        A per-point estimate that lands far from the global offset is an
        estimator mislock, not drift — drop it.
        """
        valid = {0: global_offset}
        for ts_us, off in sorted(points.items()):
            if ts_us <= 0 or not math.isfinite(off):
                continue
            if abs(off - global_offset) <= max_deviation_ms:
                valid[ts_us] = off
            else:
                log.info(
                    "Auto-sync: dropping sync point at %.2fs "
                    "(offset %.1f ms deviates > %.0f ms)",
                    ts_us / 1e6, off, max_deviation_ms,
                )
        return valid

    @staticmethod
    def _drift_significant(
        points: dict[int, float],
        min_total_drift_ms: float = 15.0,
        max_residual_ms: float = 10.0,
    ) -> bool:
        """True when the per-point offsets form a credible linear drift.

        Multi-point offsets only beat a constant when the data shows
        SYSTEMATIC drift: enough total slope AND small fit residuals.
        Quantisation noise (GoPro gpmd ±8-10 ms) with one stray point
        otherwise fakes a ramp that is worse than staying constant.
        """
        keys = sorted(points)
        if len(keys) < 3:
            return False
        t = np.asarray(keys, dtype=np.float64)
        v = np.asarray([points[k] for k in keys], dtype=np.float64)
        if t[-1] <= t[0]:
            return False
        slope, intercept = np.polyfit(t, v, 1)
        residuals = v - (slope * t + intercept)
        total_drift = abs(slope) * (t[-1] - t[0])
        return bool(
            total_drift >= min_total_drift_ms
            and np.abs(residuals).max() <= max_residual_ms
        )

    def _refine_sync_points(
        self,
        frames: list,
        gyro_data: list,
        global_offset: float,
        quaternions: dict | None,
        frame_readout_time_ms: float,
    ) -> dict[int, float]:
        """Estimate per-sync-point offsets on rotation-rich windows.

        Returns the validated offset map (>=2 entries when multi-point
        estimation succeeded and survived outlier rejection).
        """
        if self.params.duration_ms < 12_000 or len(frames) < 60:
            return {}

        from pygyroflow.synchronization import AutosyncProcess
        from pygyroflow.synchronization.optimsync import OptimSync

        try:
            ts_ms = np.array([t / 1000.0 for t, _ in gyro_data], dtype=np.float64)
            w = np.array([g for _, g in gyro_data], dtype=np.float64)
            points_ms, _rank, _step = OptimSync(ts_ms, w).run(
                target_sync_points=self._SYNC_POINT_COUNT,
                trim_ranges_s=[(0.0, self.params.duration_ms / 1000.0)],
            )
        except Exception:
            log.warning("Auto-sync: OptimSync point selection failed", exc_info=True)
            points_ms = []
        if len(points_ms) < 2:
            # Upstream (render_queue.rs:1451-1453): when the optimal-point
            # selection comes back empty, the sync points fall back to a
            # uniform spread — chunk centres of max_sync_points over the
            # clip — instead of abandoning multi-point refinement.
            chunks = self.params.duration_ms / self._SYNC_POINT_COUNT
            start = chunks / 2.0
            points_ms = [
                start + i * chunks for i in range(self._SYNC_POINT_COUNT)
            ]
            log.info(
                "Auto-sync: no optimal sync points; falling back to %d "
                "uniform points", self._SYNC_POINT_COUNT,
            )

        height, width = frames[0][1].shape[:2]
        camera_matrix = self.lens.get_camera_matrix(size=(width, height))
        win_us = int(self._SYNC_POINT_WINDOW_MS * 1000)
        method = 2 if quaternions is not None else 1
        candidates: dict[int, float] = {}
        # Built once for the whole pass: every window in it shares a lens.
        sync_params = self._build_sync_compute_params()

        # Per-point search window (B-17): upstream's default search_size is
        # 5 s (cli.rs:623) and a lens profile's sync_settings can override
        # it — in *seconds*, which render_queue.rs:1470 scales to ms. The
        # port honours the profile override but keeps a narrower default
        # than upstream's 5000 ms: the serial Python offset searches cost
        # ~40x their rayon-parallel originals, and the refinement pass runs
        # one search per sync point.
        search_ms = self._SYNC_POINT_SEARCH_MS
        if isinstance(self.lens.sync_settings, dict):
            try:
                profile_ms = float(self.lens.sync_settings.get("search_size", 0.0)) * 1000.0
                if profile_ms > 0.0:
                    search_ms = profile_ms
            except (TypeError, ValueError):
                pass

        for p_ms in points_ms:
            center_us = int(p_ms * 1000.0)
            window = [f for f in frames if center_us - win_us <= f[0] <= center_us + win_us]
            if len(window) < 10:
                continue
            try:
                proc = AutosyncProcess(
                    camera_matrix=camera_matrix,
                    fps=self.params.fps,
                    scaled_fps=self.params.get_scaled_fps(),
                    of_method=2,
                    pose_method=0,
                    offset_method=method,
                    compute_params=sync_params,
                )
                off = proc.run(
                    window,
                    gyro_data,
                    search_range_ms=search_ms,
                    sample_count=None,
                    quaternions=quaternions,
                    frame_readout_time_ms=frame_readout_time_ms,
                )
            except Exception:
                log.warning(
                    "Auto-sync: sync-point estimation failed at %.2fs", p_ms, exc_info=True
                )
                continue
            if off is not None and np.isfinite(off):
                candidates[center_us] = float(off)

        valid = self._valid_sync_points(
            candidates, global_offset, self._SYNC_POINT_MAX_DEVIATION_MS
        )
        if len(valid) < 3:
            return {}  # two points cannot distinguish drift from noise
        if not self._drift_significant(
            valid, self._SYNC_POINT_MIN_DRIFT_MS, self._SYNC_POINT_MAX_RESIDUAL_MS
        ):
            log.info(
                "Auto-sync: per-point offsets show no credible drift; "
                "keeping constant %.1f ms", global_offset,
            )
            return {}
        return valid

    def _extract_gray_frames(
        self,
        path: str,
        sample_count: int,
        max_width: int = 480,
    ) -> list[tuple[int, Any]]:
        """Decode a subsampled, downscaled sequence of grayscale frames.

        Accepts an image sequence as well as a video, so auto-sync works for
        sequence input that pairs a separate gyro source.

        Returns [(timestamp_us, gray_u8), ...] with real container pts.
        """
        import av
        import cv2

        from pygyroflow.rendering.image_sequence import (
            FFMPEG_DEFAULT_FPS,
            format_options,
            looks_like_image_sequence,
            resolve_image_sequence,
        )

        frames: list[tuple[int, Any]] = []

        sequence = resolve_image_sequence(path) if looks_like_image_sequence(path) else None
        seq_fps = self.input_file.image_sequence_fps or None
        if sequence is not None and not seq_fps:
            log.warning(
                "Image sequence has no frame rate; assuming FFmpeg's default "
                "%.0f fps for auto-sync", FFMPEG_DEFAULT_FPS,
            )
        container = (
            av.open(
                sequence.pattern,
                format="image2" if sequence.is_sequence else None,
                options=format_options(sequence, seq_fps),
            )
            if sequence is not None
            else av.open(path)
        )
        try:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            fps = float(stream.average_rate or 30.0)
            frame_count = stream.frames or int(fps * 60.0)
            if sequence is not None:
                frame_count = sequence.frame_count
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
        sync_params = self._build_sync_compute_params()
        template = RollingShutterSync(
            {}, frame_readout_time_ms=self.params.frame_readout_time, fps=self.params.fps,
            scaled_fps=self.params.get_scaled_fps(), compute_params=sync_params,
        )
        K = camera_matrix if not np.allclose(camera_matrix, np.eye(3)) else None
        added = 0
        for a, b in zip(ordered, ordered[1:]):
            if a.prev_points is None or a.curr_points is None or len(a.prev_points) < 2:
                continue
            template.add_track_from_frames(
                a.timestamp_us, b.timestamp_us,
                a.prev_points, a.curr_points, float(height), camera_matrix=K,
                compute_params=sync_params, points_dims=(width, height),
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
                scaled_fps=self.params.get_scaled_fps(),
                compute_params=sync_params,
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

    def render(
        self,
        input_path: str,
        output_path: str,
        options: dict | None = None,
        *,
        trim_ranges: list[tuple[float, float]] | None = None,
        progress_callback=None,
    ) -> None:
        """Render stabilized video.

        Args:
            input_path: Input video file path.
            output_path: Output video file path.
            options: Dict with render options:
                - codec: "H.264/AVC", "H.265/HEVC", "ProRes" (default: "H.265/HEVC")
                - bitrate: Bitrate in Mbps (0 = auto)
                - audio: Copy audio streams to the output (default: True)
                - export_trims_separately: write one file per trim range
                  instead of concatenating them (default: False). Each range
                  gets a "-NNN" suffix before the extension, as upstream
                  does. Needs more than one range to do anything.
                - pad_with_black / preserve_other_tracks: keep the whole clip
                  and ignore trim ranges, matching upstream's gate on those
                  two flags.
                - use_gpu: Whether to use GPU acceleration (default: False).
                  The GPU (wgpu) undistort path is numerically verified:
                  identity bit-exact vs CPU bilinear (also under lavapipe).
                  Speedup depends on the Vulkan implementation: real
                  hardware (Intel UHD 630) measured ~6.5x per frame
                  (78 vs 505 ms at 1280x1120), end-to-end ~2.4x; a
                  lavapipe software Vulkan yields only ~2.2x. CPU remains
                  the default; opt in for speed.
            trim_ranges: Override ``params.trim_ranges``. Internal — the
                per-range export path uses it to render one range at a time.
        """
        options = options or {}

        if options.get("export_trims_separately") and len(self.params.trim_ranges) > 1:
            from pygyroflow.rendering.ffmpeg_processor import output_path_for_range

            for index, rng in enumerate(self.params.trim_ranges):
                self.render(
                    input_path,
                    output_path_for_range(output_path, index),
                    {**options, "export_trims_separately": False},
                    trim_ranges=[rng],
                )
            return

        if trim_ranges is None:
            # Upstream only feeds ranges_ms to the processor when neither
            # pad_with_black nor preserve_other_tracks is set; with either
            # one the output keeps the full length and the ranges only drive
            # the keyframe/zoom window.
            if options.get("pad_with_black") or options.get("preserve_other_tracks"):
                trim_ranges = []
            else:
                trim_ranges = self.params.trim_ranges

        from pygyroflow.rendering import FfmpegProcessor
        from pygyroflow.rendering.ffmpeg_processor import normalise_ranges
        from pygyroflow.stabilization import cpu_undistort
        from pygyroflow.stabilization.cpu_undistort import (
            CPU_TO_UPSTREAM_INTERPOLATION,
        )
        from pygyroflow.types.enums import Interpolation

        ranges_ms = normalise_ranges(trim_ranges, self.params.duration_ms)

        codec = options.get("codec", "H.265/HEVC")
        bitrate = options.get("bitrate", 0)
        use_gpu = options.get("use_gpu", False)
        # Upstream Gyroflow interpolation indices: 0 Bilinear / 1 Bicubic /
        # 2 Lanczos4 (their default) / 3-6 EWA (implemented, but ~10x the CPU
        # cost of Lanczos4 — see pygyroflow/stabilization/ewa.py).
        interp_index = int(options.get("interpolation", 2))

        proc = FfmpegProcessor()
        # Image sequences carry no frame rate: the one resolved at load time
        # (--fps) must be repeated here or the frames would be re-timed.
        info = proc.open_input(
            input_path,
            fps=getattr(self.input_file, "image_sequence_fps", 0.0) or None,
        )

        width = info.get("width", self.params.size[0])
        height = info.get("height", self.params.size[1])
        fps = info.get("fps", self.params.fps)

        out_w = self.params.output_size[0] or width
        out_h = self.params.output_size[1] or height

        proc.create_output(output_path, out_w, out_h, fps, codec=codec, bitrate=bitrate)

        # --- Frame-rate scaling and video speed (C-05) ---
        # These are different things doing different jobs, and conflating
        # them is the easy mistake. `fps_scale` compresses the timeline the
        # frames are *looked up* on — a 240 fps recording written into a
        # 60 fps container — so it changes which transform a frame gets but
        # not how many frames there are: upstream divides `timestamp_us` by
        # the scale before every lookup. `video_speed` changes the frame
        # *count*, which upstream drives through `rate_control`.
        #
        # This runs before the output streams exist, so a speed-changed
        # render never declares an audio track it will not fill (upstream
        # clears the audio codec the same way, for the same reason).
        fps_scale = self.params.fps_scale
        video_speed = self.params.video_speed
        speed_keyframed = self.keyframes.is_keyframed(KeyframeType.VideoSpeed)

        def speed_at(timestamp_ms: float) -> float:
            return _keyframed_or(
                self.keyframes, KeyframeType.VideoSpeed, timestamp_ms, video_speed
            )

        if video_speed != 1.0 or speed_keyframed:
            speed = speed_at
            log.info(
                "Video speed %.3f%s: frames will be %s",
                video_speed,
                " (keyframed)" if speed_keyframed else "",
                "dropped" if video_speed > 1.0 else "duplicated",
            )
            if options.get("audio", True):
                log.warning(
                    "Video speed is not 1.0; dropping audio (upstream does "
                    "the same)"
                )
                options = {**options, "audio": False}
        else:
            speed = None
        if fps_scale:
            log.info("fps_scale %.4f: frame lookups use timestamp / scale", fps_scale)

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
            # GPU path verified on hardware (see tests/test_gpu_undistort.py):
            # identity bit-exact vs CPU, ~6.5x per-frame speedup.
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
            lookup_ms = timestamp_ms / fps_scale if fps_scale else timestamp_ms
            transform = self.get_frame_transform(lookup_ms, frame_idx, compute_params=render_cp)

            if gpu_backend is not None:
                # Patch kernel params for GPU shader requirements
                kp = transform.kernel_params
                channels = frame_data.shape[2] if frame_data.ndim == 3 else 1
                # The shader counts taps: its `interpolation` is 2/4/8/10-13,
                # while `interp_index` above is the CPU path's 0/1/2/3-6. The
                # two disagree on "2", so translate instead of passing it on
                # (which silently downgraded every GPU render to bilinear).
                kp.interpolation = CPU_TO_UPSTREAM_INTERPOLATION.get(
                    interp_index, int(Interpolation.Lanczos4)
                )
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

            return cpu_undistort(frame_data, transform, interpolation=interp_index)

        proc.process_frames(
            stabilize_frame, ranges_ms=ranges_ms, speed=speed,
            progress_callback=progress_callback,
        )

        # Copy audio packets through the streams prepared before the video
        # pass. Previously the output was always silent. The same ranges
        # have to be applied or a trimmed render would carry the audio of
        # the parts it dropped.
        if options.get("audio", True):
            try:
                proc.copy_audio(ranges_ms=ranges_ms)
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

    # ------------------------------------------------------------------
    # Remaining upstream setter surface (core/lib.rs). Each mirrors its
    # upstream counterpart's invalidation: what a setter touches decides
    # whether smoothing, zooming or neither must be recomputed.
    # ------------------------------------------------------------------

    def set_stab_enabled(self, v: bool) -> None:
        self.params.stab_enabled = bool(v)
        self._invalidate_smoothing()

    def set_video_speed(self, v: float) -> None:
        self.params.video_speed = float(v)
        self._invalidate_smoothing()

    def set_max_zoom(self, v: float | None) -> None:
        self.params.max_zoom = None if v is None else float(v)
        self._invalidate_zooming()

    def set_frame_offset(self, v: int) -> None:
        self.params.frame_offset = int(v)
        self._invalidate_smoothing()

    def set_frame_readout_direction(self, v) -> None:
        self.params.frame_readout_direction = v
        self._invalidate_smoothing()

    def set_additional_rotation(self, x: float, y: float, z: float) -> None:
        self.params.additional_rotation = (float(x), float(y), float(z))
        self._invalidate_smoothing()

    def set_additional_rotation_x(self, v: float) -> None:
        r = list(self.params.additional_rotation)
        r[0] = float(v)
        self.set_additional_rotation(*r)

    def set_additional_rotation_y(self, v: float) -> None:
        r = list(self.params.additional_rotation)
        r[1] = float(v)
        self.set_additional_rotation(*r)

    def set_additional_rotation_z(self, v: float) -> None:
        r = list(self.params.additional_rotation)
        r[2] = float(v)
        self.set_additional_rotation(*r)

    def set_additional_translation(self, x: float, y: float, z: float) -> None:
        self.params.additional_translation = (float(x), float(y), float(z))
        self._invalidate_smoothing()

    def set_additional_translation_x(self, v: float) -> None:
        t = list(self.params.additional_translation)
        t[0] = float(v)
        self.set_additional_translation(*t)

    def set_additional_translation_y(self, v: float) -> None:
        t = list(self.params.additional_translation)
        t[1] = float(v)
        self.set_additional_translation(*t)

    def set_additional_translation_z(self, v: float) -> None:
        t = list(self.params.additional_translation)
        t[2] = float(v)
        self.set_additional_translation(*t)

    def set_input_horizontal_stretch(self, v: float) -> None:
        self.params.input_horizontal_stretch = float(v)
        self._invalidate_zooming()

    def set_input_vertical_stretch(self, v: float) -> None:
        self.params.input_vertical_stretch = float(v)
        self._invalidate_zooming()

    def set_light_refraction_coefficient(self, v: float) -> None:
        self.params.light_refraction_coefficient = float(v)
        self._invalidate_zooming()

    def set_background_mode(self, v) -> None:
        self.params.background_mode = v
        self._invalidate_zooming()

    def set_background_margin(self, v: float) -> None:
        self.params.background_margin = float(v)
        self._invalidate_zooming()

    def set_background_margin_feather(self, v: float) -> None:
        self.params.background_margin_feather = float(v)
        self._invalidate_zooming()

    def set_background_color(self, color) -> None:
        self.params.background = np.asarray(color, dtype=np.float32)
        self._invalidate_zooming()

    def set_horizon_lock(self, percent: float, roll: float = 0.0,
                         pitch: float = 0.0) -> None:
        lock = self.smoothing.horizon_lock
        lock.horizonlockpercent = float(percent)
        lock.horizonroll = float(roll)
        lock.horizonpitch = float(pitch)
        lock.lock_enabled = abs(float(percent)) > 0.01
        self._invalidate_smoothing()

    def set_digital_lens_name(self, name: str | None) -> None:
        """Digital lens lives on the lens profile (upstream lib.rs:1030)."""
        self.lens.digital_lens = name
        self._invalidate_zooming()

    def set_digital_lens_param(self, index: int, value: float) -> None:
        """The four digital-lens coefficients, on the lens profile like
        upstream (lib.rs:1038-1043), defaulting to zeros on first write."""
        current = self.lens.digital_lens_params
        params = list(current) if current is not None else [0.0] * 4
        if 0 <= index < len(params):
            params[index] = float(value)
        self.lens.digital_lens_params = params
        self._invalidate_zooming()

    def set_zooming_method(self, v: int) -> None:
        self.params.adaptive_zoom_method = int(v)
        self._invalidate_zooming()

    def set_of_method(self, v: int) -> None:
        """The optical-flow method for the *next* sync run (upstream keeps
        it on SyncParams; the port has no persistent estimator until a
        sync starts)."""
        self._of_method = int(v)

    def set_show_detected_features(self, v: bool) -> None:
        self.params.show_detected_features = bool(v)

    def set_show_optical_flow(self, v: bool) -> None:
        self.params.show_optical_flow = bool(v)

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

    def _build_sync_compute_params(self) -> "ComputeParams | None":
        """The ``ComputeParams`` the pose estimators run against during sync.

        Port of upstream's ``autosync.rs:86-89``: a full snapshot from the
        manager, then two overrides. The keyframes go because sync is finding
        an offset, not rendering — an animated zoom or FOV would move the crop
        under it. ``lens_correction_amount`` is pinned to 1.0 because the pose
        estimators want the fully corrected image; this is the field upstream
        sets, and the one its ``Camera::delta`` does *not* read (it passes a
        literal 1.0), which is what made the estimator's lens handling easy to
        lose in the port.

        Returns None when there is no usable lens. The estimators accept that
        and fall back to a pinhole camera — the behaviour they had before this
        parameter existed — so a manager without a lens profile can still sync.
        The check is up front rather than a caught exception around the whole
        build: only the absent camera matrix is tolerated, not whatever else
        the snapshot might raise on.
        """
        lens = self.lens
        if lens is None:
            return None
        w, h = self.params.size
        if lens.get_camera_matrix(size=(w, h)) is None:
            log.warning(
                "Auto-sync: no camera matrix for %dx%d; pose estimation runs on a "
                "pinhole camera without lens distortion",
                w,
                h,
            )
            return None

        params = self._build_compute_params()
        params.keyframes.clear()
        params.lens_correction_amount = 1.0
        return params

    def _build_compute_params(self) -> ComputeParams:
        """Build a ComputeParams snapshot from current state."""
        lens = self.lens
        w, h = self.params.size
        ow, oh = self.params.output_size

        # Get camera matrix from lens
        camera_matrix = lens.get_camera_matrix(size=(w, h))
        distortion_coeffs = lens.get_distortion_coeffs()

        # Radial distortion limit from the lens's distortion model (mirrors
        # upstream lens_profile.rs: DistortionModel::from_name(...).radial_distortion_limit(&coeffs))
        radial_limit = self._get_radial_distortion_limit(
            lens.distortion_model or "opencv_fisheye", distortion_coeffs
        )

        metadata = self.gyro.file_metadata
        from pygyroflow.stabilization.distortion_models import (
            from_name as lens_model_from_name,
        )

        cp = ComputeParams(
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
            camera_diagonal_fovs=[],  # filled by calculate_camera_fovs below
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
            video_speed_affects_zooming_limit=self.params.video_speed_affects_zooming_limit,
            smoothing_fov_limit_per_frame=list(
                getattr(self, "_smoothing_fov_limit_per_frame", None) or []
            ),
            framebuffer_inverted=self.params.framebuffer_inverted,
            trim_ranges=list(self.params.trim_ranges),
            calib_width=lens.calib_dimension["w"],
            calib_height=lens.calib_dimension["h"],
            input_horizontal_stretch=lens.input_horizontal_stretch if lens.input_horizontal_stretch > 0.01 else 1.0,
            input_vertical_stretch=lens.input_vertical_stretch if lens.input_vertical_stretch > 0.01 else 1.0,
            focal_length=lens.focal_length,
            # Per-timestamp lens data. Both maps are empty for a fixed-focal-
            # length clip, which leaves get_lens_data_at_timestamp on exactly
            # the static path it had before.
            lens=lens,
            lens_positions=ClosestMap(metadata.lens_positions),
            lens_params=ClosestMap(metadata.lens_params),
            digital_zoom=metadata.digital_zoom,
            radial_distortion_limit=radial_limit,
            focal_length_smoothing_strength=self.params.focal_length_smoothing_strength,
            # Only the points path reads these three; the image path takes its
            # mesh and IBIS data through the kernel parameters instead.
            mesh_correction=list(metadata.mesh_correction or []),
            camera_stab_data=list(metadata.camera_stab_data or []),
            digital_lens=(
                lens_model_from_name(lens.digital_lens) if lens.digital_lens else None
            ),
            digital_lens_params=(
                list(lens.digital_lens_params) if lens.digital_lens_params else None
            ),
            optimal_fov=lens.optimal_fov,
            per_frame_time_offsets=list(
                getattr(self.gyro.file_metadata, "per_frame_time_offsets", None) or []
            ),
        )
        # One FOV per frame only when the calibration actually moves; see
        # ComputeParams.calculate_camera_fovs.
        cp.calculate_camera_fovs()
        # Run FL smoothing unconditionally, so `cp.focal_lengths` always holds
        # the dequantized curve by the time anything reads the compute params
        # (upstream lib.rs: run it before the zooming checksum is compared).
        self._apply_focal_length_smoothing(cp)
        return cp

    @staticmethod
    def extract_focal_lengths(cp: ComputeParams) -> list[float | None]:
        """Per-frame focal length from the lens metadata, ``None`` where absent.

        Port of upstream ``Gyroflow::extract_focal_lengths``. The per-frame
        values come from ``lens_params`` at the frame's own timestamp, with the
        same 100 ms lookup window the lens data uses — a zoom lens is exactly
        the case where a static profile focal length is wrong.
        """
        from pygyroflow.stabilization.frame_transform import LENS_LOOKUP_MAX_DIFF_US
        from pygyroflow.util import timestamp_at_frame

        if not cp.lens_params:
            return []

        focal_lengths: list[float | None] = []
        for frame in range(cp.frame_count):
            timestamp_ms = timestamp_at_frame(frame, cp.scaled_fps)
            entry = cp.lens_params.get_closest(
                round(timestamp_ms * 1000.0), LENS_LOOKUP_MAX_DIFF_US
            )
            if entry is not None and entry.focal_length is not None:
                focal_lengths.append(float(entry.focal_length))
            else:
                focal_lengths.append(None)
        return focal_lengths

    def _apply_focal_length_smoothing(self, cp: ComputeParams) -> None:
        """Fill the focal length caches on *cp* (port of lib.rs
        ``apply_focal_length_smoothing``).

        Two curves, both frame-indexed:

        * ``focal_lengths`` — the raw metadata run through a short Gaussian.
          Dequantization, not smoothing: the raw curve is the denominator of
          the compensation ratio, so its stairs would show up in the output.
        * ``smoothed_focal_lengths`` — that dequantized curve through the
          velocity-adaptive filter, and the curve the output is made to track.

        The single UI knob (`focal_length_smoothing_strength`, 0..1) drives all
        three filter dials so the slider feels monotonic: more strength means a
        longer stationary time constant, a *higher* velocity threshold (the
        filter resists opening up), and a longer fast-zoom time constant, so
        real zoom edges round off instead of snapping to the raw shape.
        """
        from pygyroflow.smoothing.focal_length import (
            smooth_focal_lengths_adaptive,
            smooth_focal_lengths_gaussian,
        )

        params = self.params
        raw = self.extract_focal_lengths(cp)
        active = bool(params.focal_length_smoothing_enabled) and bool(raw)

        dequantized: list[float | None] = []
        smoothed: list[float | None] = []
        if active:
            # `math.floor(x + 0.5)` is Rust's round-half-away-from-zero; the
            # `.max(5)` below makes the half-integer cases equal anyway, but
            # the two languages should not be left to differ by accident.
            window = max(math.floor(cp.scaled_fps * 0.5 + 0.5), 5)
            dequantized = smooth_focal_lengths_gaussian(raw, 1.0, window)

            s = min(max(params.focal_length_smoothing_strength, 0.0), 1.0)
            smoothed = smooth_focal_lengths_adaptive(
                dequantized,
                cp.scaled_fps,
                0.1 * 300.0**s,          # stationary time constant: 0.1 .. 30 s
                0.05 + 0.35 * s * s,     # fast-zoom time constant: 0.05 .. 0.40 s
                0.3 + 7.7 * s**1.5,      # velocity threshold: 0.3 .. 8.0
            )

        # Rendering side: only populated when smoothing is active, so
        # frame_transform's compensation is a clean 1.0 otherwise.
        if active:
            cp.focal_lengths = dequantized
            cp.smoothed_focal_lengths = list(smoothed)
            cp.focal_length_smoothing_enabled = True
        else:
            cp.focal_lengths = []
            cp.smoothed_focal_lengths = []
            cp.focal_length_smoothing_enabled = False

        # Chart side: expose the raw curve whenever per-frame data exists, so a
        # timeline toggle works with smoothing off. The smoothed curve is only
        # meaningful when it was actually computed.
        params.focal_lengths = raw
        params.smoothed_focal_lengths = smoothed

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

        Mirrors upstream's two-step autoload (controller.rs):
        1. Build a CameraIdentifier from the telemetry's camera tags
           (GoPro EISA/VFOV, Sony lens info, Insta360 FOV) + resolution +
           fps, and look the profile up by its exact database identifier
           (e.g. "gopro-hero6black-wide-2704x2028@29970-no-eis").
        2. Fall back to a database search re-ranked by calibration-size
           equality, fps closeness and default-FOV preference — the text
           search itself only ranks by aspect ratio + alphabeticals, which
           picks arbitrary fps/resolution variants on tagless files
           (Hero5/6 write no EISA/VFOV tags).

        Skipped when telemetry already provided a lens profile (e.g. DJI's
        embedded one).
        """
        # A telemetry-provided lens (non-zero calib dimension) wins.
        if self.lens.calib_dimension.get("w", 0) > 0:
            return

        md = self.gyro.file_metadata
        source = getattr(md, "detected_source", None) if md else None
        if not source:
            return
        # Guard against placeholder sources: a bare "Unknown" would
        # literally match profiles whose name contains the word.
        words = source.split()
        if len(words) < 2:
            return

        # Load the database BEFORE consulting it: the brand guard used to
        # run first, iterating an empty (not yet loaded) profiles list and
        # silently rejecting every source -- GoPro never auto-matched a
        # lens profile.
        if not self.lens_db.loaded:
            self.lens_db.load_all()
        if not len(self.lens_db):
            return
        brand = words[0].lower()
        known_brands = {prof.camera_brand.strip().lower() for _k, prof in self.lens_db.profiles if prof.camera_brand}
        if brand not in known_brands:
            return

        w, h = self.params.size
        fps = self.params.fps

        # Step 1: exact identifier lookup from telemetry camera tags
        camera_tags = (md.additional_data or {}).get("camera_tags") if md else None
        if camera_tags:
            from pygyroflow.camera.identifier import CameraIdentifier

            ident = CameraIdentifier.from_metadata(
                brand=words[0],
                model=" ".join(words[1:]),
                video_width=w,
                video_height=h,
                fps=fps,
                samples=[{"tag_map": camera_tags}],
            )
            for candidate_id in (ident.get_identifier_for_autoload(), ident.identifier):
                if not candidate_id:
                    continue
                profile = self.lens_db.find(candidate_id)
                if profile is not None and profile.calib_dimension.get("w", 0) > 0:
                    log.info(
                        "Auto lens: exact identifier '%s' -> '%s'",
                        candidate_id, profile.get_display_name(),
                    )
                    try:
                        self.load_lens_profile_by_object(profile)
                        self._apply_lens_readout_time()
                        return
                    except Exception as exc:
                        log.warning("Auto lens load failed: %s", exc)

            # Near miss (e.g. zoom lens calibrated at 14.60 mm while the
            # camera reports 14): same brand-model-resolution-fps, nearest
            # focal length.
            profile = self._find_identifier_near_miss(ident, w, h, fps)
            if profile is not None:
                log.info(
                    "Auto lens: near-identifier match for '%s' -> '%s'",
                    ident.identifier, profile.get_display_name(),
                )
                try:
                    self.load_lens_profile_by_object(profile)
                    self._apply_lens_readout_time()
                    return
                except Exception as exc:
                    log.warning("Auto lens load failed: %s", exc)

        # Step 2: aspect-ratio filtered search, re-ranked for the actual
        # resolution / frame rate / default FOV.
        aspect = (w / h) if w > 0 and h > 0 else None
        results = self.lens_db.search(source, aspect_ratio=aspect, limit=50)
        if not results:
            log.info("Auto lens: no match for '%s'", source)
            return

        best = min(results, key=lambda p: self._lens_match_penalty(p, w, h, fps))
        log.info(
            "Auto lens: matched '%s' for '%s' (calib %sx%s)",
            best.get_display_name(), source,
            best.calib_dimension.get("w"), best.calib_dimension.get("h"),
        )
        try:
            self.load_lens_profile_by_object(best)
            self._apply_lens_readout_time()
        except Exception as exc:
            log.warning("Auto lens load failed: %s", exc)

    def _find_identifier_near_miss(
        self, ident: Any, w: int, h: int, fps: float
    ) -> LensProfile | None:
        """Find a DB profile for the same brand-model-size-fps at the
        nearest focal length.

        Zoom lenses are calibrated at their true focal length (e.g. a
        "14 mm" setting measures 14.60 mm), so the exact identifier built
        from the camera-reported focal length can miss while a perfectly
        good calibration exists. Hash- or name-based lens_info tokens are
        skipped (no focal number to compare).
        """
        import re

        fps_int = round(fps * 1000)
        prefix = f"{ident.brand}-{ident.model}-".lower().replace(" ", "")
        size_fps = f"-{w}x{h}@{fps_int}"
        best: tuple[float, LensProfile] | None = None
        try:
            own_focal = float(ident.focal_length or 0.0)
        except (TypeError, ValueError):
            own_focal = 0.0
        if own_focal <= 0.0:
            return None

        for key, prof in self.lens_db.profiles:
            if not key or not key.lower().startswith(prefix) or not key.lower().endswith(size_fps):
                continue
            token = key[len(prefix): -len(size_fps)]
            m = re.match(r"^(\d+(?:\.\d+)?)\s*mm$", token, re.IGNORECASE)
            if not m:
                continue
            diff = abs(float(m.group(1)) - own_focal)
            if best is None or diff < best[0]:
                best = (diff, prof)
        if best is not None and best[0] <= 1.0:  # within 1 mm
            return best[1]
        return None

    @staticmethod
    def _lens_match_penalty(profile: LensProfile, w: int, h: int, fps: float) -> tuple:
        """Ranking key for lens auto-match among same-brand candidates.

        Ordered by: aspect-ratio match (the search's own first criterion —
        a 4:3 calibration must never win over an 8:7 one for 8:7 footage),
        exact calibration size (then swapped), fps closeness, default-FOV
        preference (Wide is GoPro's default; Linear/Super are opt-in
        settings), NO-EIS over EIS variants (a clip we must stabilize most
        likely has EIS off), then display name.
        """
        cw = profile.calib_dimension.get("w", 0)
        ch = profile.calib_dimension.get("h", 0)
        if w > 0 and h > 0 and cw > 0 and ch > 0:
            aspect_rank = 0 if abs((cw / ch) - (w / h)) < 0.01 else 1
        else:
            aspect_rank = 1
        if (cw, ch) == (w, h):
            size_rank = 0
        elif (ch, cw) == (w, h):
            size_rank = 1
        else:
            size_rank = 2
        prof_fps = profile.fps or 0.0
        fps_diff = abs(prof_fps - fps) if prof_fps > 0 else 999.0
        name = profile.get_display_name().lower()
        fov_pref = {"wide": 0, "super": 1, "hyper": 2, "linear": 3, "narrow": 4, "medium": 5, "max": 6}
        fov_rank = min(
            (rank for token, rank in fov_pref.items() if token in name),
            default=7,
        )
        eis_rank = 0 if "no-eis" in name else 1
        return (aspect_rank, size_rank, fps_diff, fov_rank, eis_rank, name)

    def _apply_lens_readout_time(self) -> None:
        """Apply the loaded lens profile's rolling-shutter readout time.

        Mirrors upstream lib.rs: after autoload, a lens-provided
        frame_readout_time (e.g. GoPro 11.11 ms) overrides params so RS
        correction engages even when the file itself carries no SROT tag.
        """
        from pygyroflow.types.enums import ReadoutDirection

        fr = self.lens.frame_readout_time
        if fr is not None and fr != 0.0:
            self.params.frame_readout_time = abs(fr)
            self.params.frame_readout_direction = (
                ReadoutDirection.BottomToTop if fr < 0.0 else ReadoutDirection.TopToBottom
            )

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

    def _get_video_info(self, path: str, fps: float | None = None) -> dict:
        """Extract video metadata using PyAV.

        *path* may be an image sequence (directory, printf pattern or single
        frame).  Those carry no frame rate, so *fps* overrides the container's
        — FFmpeg would otherwise assume 25 fps and put the gyro timeline at
        the wrong speed.  The frame count comes from the files on disk, since
        the image2 demuxer reports a duration but no frame count.
        """
        from pygyroflow.rendering.image_sequence import (
            format_options,
            looks_like_image_sequence,
            resolve_image_sequence,
        )

        sequence = resolve_image_sequence(path) if looks_like_image_sequence(path) else None
        try:
            import av

            if sequence is not None:
                container = av.open(
                    sequence.pattern,
                    format="image2" if sequence.is_sequence else None,
                    options=format_options(sequence, fps),
                )
            else:
                container = av.open(path)
            stream = container.streams.video[0]

            container_fps = float(stream.average_rate)
            # A caller-supplied rate only speaks for image sequences — a video
            # container knows its own rate, and overriding it would put the
            # gyro timeline at the wrong speed.
            effective_fps = float(fps) if (fps and sequence is not None) else container_fps
            duration_s = float(stream.duration * stream.time_base) if stream.duration else 0.0
            if duration_s <= 0:
                duration_s = float(container.duration) / 1_000_000 if container.duration else 0.0

            width = stream.codec_context.width
            height = stream.codec_context.height
            frame_count = stream.frames
            if sequence is not None:
                # image2 reports frames=0 but knows the duration; the on-disk
                # count is the authoritative one (and the only one available
                # for a single still, where duration is None).
                frame_count = sequence.frame_count
                duration_s = frame_count / effective_fps if effective_fps > 0 else duration_s
            elif frame_count <= 0 and container_fps > 0 and duration_s > 0:
                frame_count = int(duration_s * container_fps)

            container.close()

            return {
                "width": width,
                "height": height,
                "fps": effective_fps,
                "duration_ms": duration_s * 1000.0,
                "frame_count": frame_count,
                "image_sequence": sequence,
            }
        except ImportError:
            raise VideoIOError("PyAV (av) is required for video loading")
        except VideoIOError:
            raise
        except Exception as exc:
            raise VideoIOError(f"Failed to open video {path}: {exc}")
