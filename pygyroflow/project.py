"""``.gyroflow`` project files — the JSON container Gyroflow saves alongside a clip.

A project records everything the pipeline needs that the video itself does not
carry: the gyro transform, the calibration, every stabilization parameter, the
resolved sync offsets, the trim and the output settings. Upstream writes it
with ``StabilizationManager::export_gyroflow_data`` (lib.rs) and reads it with
``import_gyroflow_data``; this module mirrors that schema.

Three things are deliberately literal:

* Sections that this port does not model are kept as raw dicts and written
  back untouched. A project written by a newer Gyroflow round-trips through
  here without losing fields.
* Large per-frame payloads are base91 + zlib envelopes. They are decoded on
  demand rather than eagerly, and the *inner* codec depends on the field:
  the ``gyro_source`` family is bincode-legacy, the ``WithProcessedData``
  caches are CBOR. See ``pygyroflow.util`` for the details.
* The version is a plain field, not normalised on load. ``project_version``
  changes how upstream interprets a RED clip's gyro timestamps, so a
  JSON-level round-trip keeps whatever the file said. Writing a *new* file
  (:meth:`~pygyroflow.manager.StabilizationManager.save_project`) stamps
  ``PROJECT_VERSION``, matching ``export_gyroflow_data``, which always emits
  the current one regardless of what it loaded.

``PROJECT_VERSION`` tracks ``export_gyroflow_data``, which currently writes
4. The reference files in the test-video directory are version 2 and stay
version 2 through a :class:`GyroflowProject` round-trip.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from pygyroflow.util import (
    compress_to_base91,
    decode_cbor_f64_list,
    decode_cbor_quat_map,
    decode_imu_list,
    decode_quat_map,
    decode_vec3_map,
    decompress_from_base91,
    encode_cbor_f64_list,
    encode_cbor_quat_map,
    encode_f64_list,
    encode_quat_map,
)

log = logging.getLogger(__name__)

PROJECT_TITLE = "Gyroflow data file"
PROJECT_VERSION = 4

# ``gyro_source`` blobs: base91 + zlib + bincode legacy (util.rs
# ``compress_to_base91``). These are the ones a "simple" export carries.
_BINCODE_READERS = {
    "quaternions": decode_quat_map,
    "image_orientations": decode_quat_map,
    "gravity_vectors": decode_vec3_map,
    "raw_imu": decode_imu_list,
}

# ``WithProcessedData`` blobs: base91 + zlib + CBOR (util.rs
# ``compress_to_base91_cbor``). These are the caches exported for plugins.
_CBOR_READERS = {
    "integrated_quaternions": decode_cbor_quat_map,
    "smoothed_quaternions": decode_cbor_quat_map,
    "adaptive_zoom_fovs": decode_cbor_f64_list,
    "synced_imu_timestamps": decode_cbor_f64_list,
    "synced_imu_timestamps_with_per_frame_offset": decode_cbor_f64_list,
    "focal_lengths": decode_cbor_f64_list,
    "smoothed_focal_lengths": decode_cbor_f64_list,
}

# Which encoder :meth:`GyroflowProject.write_blob` uses for each name.
_BLOB_WRITERS = {
    "quaternions": encode_quat_map,
    "integrated_quaternions": encode_cbor_quat_map,
    "smoothed_quaternions": encode_cbor_quat_map,
    "adaptive_zoom_fovs": encode_cbor_f64_list,
    "synced_imu_timestamps": encode_cbor_f64_list,
    "synced_imu_timestamps_with_per_frame_offset": encode_cbor_f64_list,
    "focal_lengths": encode_cbor_f64_list,
    "smoothed_focal_lengths": encode_cbor_f64_list,
}

# Readable but not writable here: their entry layouts are not the quaternion
# map's (``gravity_vectors`` is 32 bytes an entry, ``raw_imu`` is variable
# length), so the generic dict/list fallback in write_blob would silently
# produce a payload Gyroflow cannot read. Refuse instead.
_READ_ONLY_BLOBS = {"image_orientations", "gravity_vectors", "raw_imu"}


def _jsonable(value: Any) -> Any:
    """Recursively convert to something ``json`` can write.

    The parameter structs are numpy-backed, so a plain ``params.fov`` is a
    ``float32`` and ``json.dump`` refuses it. Upstream's serde widens f32 to
    f64, which is what this does too, and it unwraps numpy arrays to lists.
    """
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        return float(value)
    item = getattr(value, "item", None)  # numpy scalar
    if item is not None and getattr(value, "shape", None) == ():
        return item()
    tolist = getattr(value, "tolist", None)  # numpy array
    if tolist is not None:
        return _jsonable(tolist())
    return value


@dataclass
class ProjectVideoInfo:
    width: int = 0
    height: int = 0
    rotation: float = 0.0
    num_frames: int = 0
    fps: float = 0.0
    duration_ms: float = 0.0
    fps_scale: float | None = None
    vfr_fps: float = 0.0
    vfr_duration_ms: float = 0.0
    created_at: int | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ProjectVideoInfo:
        data = data or {}
        created_at = data.get("created_at")
        return cls(
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            rotation=float(data.get("rotation", 0.0) or 0.0),
            num_frames=int(data.get("num_frames", 0)),
            fps=float(data.get("fps", 0.0) or 0.0),
            duration_ms=float(data.get("duration_ms", 0.0) or 0.0),
            fps_scale=data.get("fps_scale"),
            vfr_fps=float(data.get("vfr_fps", 0.0) or 0.0),
            vfr_duration_ms=float(data.get("vfr_duration_ms", 0.0) or 0.0),
            created_at=None if created_at is None else int(created_at),
            raw=dict(data),
        )

    def to_dict(self) -> dict[str, Any]:
        out = dict(self.raw)
        out.update(
            width=self.width, height=self.height, rotation=self.rotation,
            num_frames=self.num_frames, fps=self.fps, duration_ms=self.duration_ms,
            fps_scale=self.fps_scale, vfr_fps=self.vfr_fps,
            vfr_duration_ms=self.vfr_duration_ms,
        )
        # Only version 4 files carry this; adding a null to an older file's
        # video_info would be a change the round-trip contract forbids.
        if self.created_at is not None or "created_at" in self.raw:
            out["created_at"] = self.created_at
        return out


@dataclass
class GyroflowProject:
    """A parsed `.gyroflow` file.

    `calibration_data`, `stabilization`, `gyro_source`, `output`,
    `synchronization` and `keyframes` are kept as plain dicts — that is what
    they are in the file, and keeping them raw means a save/load round-trip
    does not silently drop fields this port does not model yet.
    """

    videofile: str = ""
    version: int = PROJECT_VERSION
    app_version: str = ""
    date: str = ""
    title: str = PROJECT_TITLE

    calibration_data: dict[str, Any] = field(default_factory=dict)
    video_info: ProjectVideoInfo = field(default_factory=ProjectVideoInfo)
    stabilization: dict[str, Any] = field(default_factory=dict)
    gyro_source: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    synchronization: dict[str, Any] = field(default_factory=dict)
    keyframes: dict[str, Any] = field(default_factory=dict)

    background_color: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    background_mode: int = 0
    background_margin: float = 0.0
    background_margin_feather: float = 0.0
    light_refraction_coefficient: float = 1.0

    trim_start: float = 0.0
    trim_end: float = 1.0
    trim_ranges_ms: list[list[float]] = field(default_factory=list)

    image_sequence_start: int = 0
    image_sequence_fps: float = 0.0

    # timestamp_us -> offset_ms
    offsets: dict[int, float] = field(default_factory=dict)

    # Anything the schema grows that this port does not know about.
    unknown: dict[str, Any] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GyroflowProject:
        known = {
            "title", "version", "app_version", "videofile", "date",
            "calibration_data", "video_info", "stabilization", "gyro_source",
            "output", "synchronization", "keyframes", "background_color",
            "background_mode", "background_margin", "background_margin_feather",
            "light_refraction_coefficient", "trim_start", "trim_end",
            "trim_ranges_ms", "image_sequence_start", "image_sequence_fps",
            "offsets",
        }
        offsets = {int(k): float(v) for k, v in (data.get("offsets") or {}).items()}
        return cls(
            videofile=data.get("videofile", "") or "",
            version=int(data.get("version", PROJECT_VERSION)),
            app_version=data.get("app_version", "") or "",
            date=data.get("date", "") or "",
            title=data.get("title", PROJECT_TITLE) or PROJECT_TITLE,
            calibration_data=dict(data.get("calibration_data") or {}),
            video_info=ProjectVideoInfo.from_dict(data.get("video_info")),
            stabilization=dict(data.get("stabilization") or {}),
            gyro_source=dict(data.get("gyro_source") or {}),
            output=dict(data.get("output") or {}),
            synchronization=dict(data.get("synchronization") or {}),
            keyframes=dict(data.get("keyframes") or {}),
            background_color=list(data.get("background_color") or [0.0] * 4),
            background_mode=int(data.get("background_mode", 0) or 0),
            background_margin=float(data.get("background_margin", 0.0) or 0.0),
            background_margin_feather=float(
                data.get("background_margin_feather", 0.0) or 0.0
            ),
            light_refraction_coefficient=float(
                data.get("light_refraction_coefficient", 1.0) or 1.0
            ),
            trim_start=float(data.get("trim_start", 0.0) or 0.0),
            trim_end=float(data.get("trim_end", 1.0) or 1.0),
            trim_ranges_ms=[
                [float(a), float(b)] for a, b in (data.get("trim_ranges_ms") or [])
            ],
            image_sequence_start=int(data.get("image_sequence_start", 0) or 0),
            image_sequence_fps=float(data.get("image_sequence_fps", 0.0) or 0.0),
            offsets=offsets,
            unknown={k: v for k, v in data.items() if k not in known},
        )

    @classmethod
    def load(cls, path: str) -> GyroflowProject:
        """Read a `.gyroflow` file."""
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = dict(self.unknown)
        data.update(
            {
                "title": self.title,
                "version": self.version,
                "app_version": self.app_version,
                "videofile": self.videofile,
                "calibration_data": self.calibration_data,
                "date": self.date,
                "background_color": list(self.background_color),
                "background_mode": self.background_mode,
                "background_margin": self.background_margin,
                "background_margin_feather": self.background_margin_feather,
                "light_refraction_coefficient": self.light_refraction_coefficient,
                "video_info": self.video_info.to_dict(),
                "stabilization": self.stabilization,
                "gyro_source": self.gyro_source,
                "output": self.output,
                "synchronization": self.synchronization,
                "keyframes": self.keyframes,
                "offsets": {str(k): v for k, v in sorted(self.offsets.items())},
                "trim_start": self.trim_start,
                "trim_end": self.trim_end,
            }
        )
        if self.trim_ranges_ms:
            data["trim_ranges_ms"] = [list(r) for r in self.trim_ranges_ms]
        if self.image_sequence_start:
            data["image_sequence_start"] = self.image_sequence_start
        if self.image_sequence_fps:
            data["image_sequence_fps"] = self.image_sequence_fps
        return data

    def save(self, path: str) -> None:
        """Write the project out, pretty-printed like upstream does.

        ``allow_nan=False``: JSON has no NaN or Infinity, and Python's default
        of emitting the bare tokens produces a file no other parser — Gyroflow
        included — can read. Failing here is better than writing one.
        """
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(self.to_dict()), handle, indent=2, allow_nan=False)

    # ------------------------------------------------------------------
    # Embedded payloads
    # ------------------------------------------------------------------

    def read_blob(self, name: str, section: str = "gyro_source") -> Any | None:
        """Decode a base91+zlib payload from *section*, or None if absent.

        The reader is chosen by field name, because upstream uses bincode for
        the ``gyro_source`` blobs and CBOR for the processed-data caches. An
        unrecognised name still gets its envelope unwrapped and the payload
        bytes back, so a caller can reach a field this port does not model.
        """
        holder = getattr(self, section, None)
        if not isinstance(holder, dict):
            return None
        text = holder.get(name)
        if not isinstance(text, str) or not text:
            return None
        try:
            raw = decompress_from_base91(text)
        except Exception:
            log.warning("Could not decode %s.%s payload", section, name, exc_info=True)
            return None
        reader = _BINCODE_READERS.get(name) or _CBOR_READERS.get(name)
        if reader is None:
            return raw
        try:
            return reader(raw)
        except Exception:
            log.warning("Malformed %s payload", name, exc_info=True)
            return None

    def write_blob(self, name: str, value: Any, section: str = "gyro_source") -> None:
        """Pack *value* into a base91+zlib blob in *section*.

        Known field names pick their own codec; anything else is written with
        the bincode one, which is what the ``gyro_source`` section uses.
        """
        writer = _BLOB_WRITERS.get(name)
        if writer is not None:
            raw = writer(value)
        elif name in _READ_ONLY_BLOBS:
            raise ValueError(
                f"{name!r} has a different entry layout than the quaternion "
                "map; this port can read it but not write it"
            )
        elif isinstance(value, dict):
            raw = encode_quat_map(value)
        elif isinstance(value, (list, tuple)):
            raw = encode_f64_list(list(value))
        elif isinstance(value, bytes):
            raw = value
        else:
            raise TypeError(f"Unsupported payload type: {type(value).__name__}")
        holder = getattr(self, section, None)
        if not isinstance(holder, dict):
            raise TypeError(f"{section} is not a dict section")
        holder[name] = compress_to_base91(raw)
