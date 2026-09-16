"""CBOR (de)serialization of ``FileMetadata``.

Port of the serde impls behind ``gyro_source/file_metadata.rs``,
``camera_identifier.rs`` and telemetry-parser's ``IMUData``, as they appear
inside a ``.gyroflow`` project's ``gyro_source.file_metadata`` blob: a
base91 + zlib + CBOR payload holding the whole metadata struct.

This is the payload that lets a project stand on its own. A ``WithGyroData``
or ``WithProcessedData`` project carries its IMU here, so it can be stabilized
without the original clip's telemetry — and, for a real Gyroflow 1.6.3 export,
*only* here: ``gyro_source.raw_imu``/``quaternions`` are left ``null`` and the
motion lives in this blob plus ``integrated_quaternions``.

Layout notes that are easy to get wrong:

* A struct is a CBOR **map with text keys in declaration order**. So
  ``LensParams`` is an 8-key map, ``IMUData`` a 4-key map, and a tuple such as
  ``(u32, u32)`` is an **array**. The bincode *side* of this same struct is a
  bare sequence with no keys at all (``util.decode_imu_list``) — the two codecs
  disagree on purpose, and one layout's reader cannot be reused for the other.
* ``ReadoutDirection`` serializes as its variant **name** (``"TopToBottom"``),
  not its discriminant, because the derive has no rename attribute.
* A ``serde_json::Value`` field (``lens_profile``, ``additional_data``) goes
  out in the key order it holds — **not** sorted. ``serde_json::Map`` is a
  ``BTreeMap`` by default and would sort, but a real 1.6.3 export has
  ``lens_profile`` starting with ``calibrated_by``, so this build has
  ``preserve_order`` on and the profile keeps the order it was embedded with.
* Floats go out in the narrowest form that holds them, and ``f32`` fields are
  rounded to single precision *before* that (a value stored as ``1.3`` is
  written as the single ``1.3f32``, not as the double ``1.3``).

The byte layouts here are checked against ciborium itself — the crate
``ciborium`` is what writes these files — via
``tests/golden/cbor_file_metadata.json``, and end to end against a real
export. See ``tests/golden/generate_file_metadata_reference.py``.
"""

from __future__ import annotations

import struct
from typing import Any

import cbor2
import numpy as np

from pygyroflow.gyro_source.file_metadata import FileMetadata, LensParams
from pygyroflow.types.enums import ReadoutDirection
from pygyroflow.types.quaternion import Quat64
from pygyroflow.types.time_types import TimeIMU
from pygyroflow.util import _cbor_f64, _cbor_head, _cbor_int

__all__ = [
    "encode_file_metadata",
    "decode_file_metadata",
    "encode_imu_list",
    "encode_lens_params",
    "encode_lens_params_map",
    "encode_lens_positions",
    "encode_gravity_vectors",
    "encode_optional_quat_map",
    "encode_camera_identifier",
    "encode_camera_stab_data",
    "encode_mesh_correction",
    "encode_json_value",
    "encode_per_frame_offsets",
]

# Declaration order, from the Rust struct. Both the encoder and the decoder
# walk it, so the two cannot drift apart silently.
_FILE_METADATA_FIELDS = (
    "imu_orientation",
    "raw_imu",
    "quaternions",
    "gravity_vectors",
    "image_orientations",
    "detected_source",
    "frame_readout_time",
    "frame_readout_direction",
    "frame_rate",
    "camera_identifier",
    "lens_profile",
    "lens_positions",
    "lens_params",
    "digital_zoom",
    "has_accurate_timestamps",
    "additional_data",
    "per_frame_time_offsets",
    "camera_stab_data",
    "mesh_correction",
)

_LENS_PARAMS_FIELDS = (
    "focal_length",
    "pixel_pitch",
    "sensor_size_px",
    "capture_area_origin",
    "capture_area_size",
    "pixel_focal_length",
    "distortion_coefficients",
    "focus_distance",
)

_CAMERA_IDENTIFIER_FIELDS = (
    "brand",
    "model",
    "lens_model",
    "lens_info",
    "focal_length",
    "camera_setting",
    "fps",
    "video_width",
    "video_height",
    "additional",
    "identifier",
)

_IMU_FIELDS = ("timestamp_ms", "gyro", "accl", "magn")


# ----------------------------------------------------------------------
# Value writers
# ----------------------------------------------------------------------


def _text(value: str) -> bytes:
    encoded = str(value).encode("utf-8")
    return _cbor_head(3, len(encoded)) + encoded


def _array(items: list[bytes]) -> bytes:
    return _cbor_head(4, len(items)) + b"".join(items)


def _map(entries: list[tuple[bytes, bytes]]) -> bytes:
    """A map from already-encoded key/value pairs, in the given order.

    Order is the caller's business: structs pass declaration order, JSON
    objects pass the order their dict holds.
    """
    return _cbor_head(5, len(entries)) + b"".join(k + v for k, v in entries)


def _struct(entries: list[tuple[str, bytes]]) -> bytes:
    return _map([(_text(name), value) for name, value in entries])


def _f32(value: float) -> float:
    """Round to single precision, as an ``f32`` field does on the way out.

    Not cosmetic: ``1.3`` and ``1.3f32`` are different doubles, and the
    narrower one is what lands in the file.
    """
    return float(struct.unpack(">f", struct.pack(">f", float(value)))[0])


def _get(source: Any, name: str, default: Any = None) -> Any:
    """Field lookup that works on a CBOR-decoded dict and on a dataclass.

    Lets a value that came out of a decode go straight back in without the
    port having to model every nested struct.
    """
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _opt_float(value: float | None, narrow: bool = False) -> bytes:
    if value is None:
        return b"\xf6"
    return _cbor_f64(_f32(value) if narrow else float(value))


def _opt_int_tuple(value, width: int = 2) -> bytes:
    """``Option<(u32, u32)>`` — a tuple is an array, not a map."""
    if value is None:
        return b"\xf6"
    items = [_cbor_int(int(v)) for v in value]
    if len(items) != width:
        raise ValueError(f"expected a {width}-tuple, got {value!r}")
    return _array(items)


def _opt_f32_tuple(value, width: int = 2) -> bytes:
    """``Option<(f32, f32)>``, or a bare ``(f32, f32, f32, f32)`` at width 4."""
    if value is None:
        return b"\xf6"
    items = [_cbor_f64(_f32(v)) for v in value]
    if len(items) != width:
        raise ValueError(f"expected a {width}-tuple, got {value!r}")
    return _array(items)


def _json_value(value: Any) -> bytes:
    """A ``serde_json::Value``.

    Maps keep the key order they are given, and that is a fact about real
    files rather than a style choice: ``lens_profile`` in a Gyroflow 1.6.3
    export starts with ``calibrated_by``, not with the alphabetically first
    ``calib_dimension``. So this build of Gyroflow has ``serde_json``'s
    ``preserve_order`` on (feature-unified in from somewhere in its
    dependency graph), and the profile goes out in the order it was embedded.
    Sorting here would produce a valid file that is not the same file.
    """
    if value is None:
        return b"\xf6"
    if value is True:
        return b"\xf5"
    if value is False:
        return b"\xf4"
    if isinstance(value, str):
        return _text(value)
    if isinstance(value, (int, np.integer)):
        return _cbor_int(int(value))
    if isinstance(value, (float, np.floating)):
        return _cbor_f64(float(value))
    if isinstance(value, dict):
        return _map([(_text(k), _json_value(v)) for k, v in value.items()])
    if isinstance(value, (list, tuple)):
        return _array([_json_value(v) for v in value])
    if isinstance(value, np.ndarray):
        return _array([_json_value(v) for v in value.tolist()])
    raise TypeError(f"cannot encode {type(value).__name__} as CBOR JSON value")


def _quat(value) -> bytes:
    """A quaternion as a 4-element array.

    ``Quat64`` or a plain sequence, the same two shapes ``util``'s quaternion
    map accepts.
    """
    getter = getattr(value, "quaternion", None)
    components = getter() if getter is not None else value
    return _array([_cbor_f64(float(c)) for c in components])


def _vec3(value) -> bytes:
    components = np.asarray(value, dtype=np.float64).ravel()
    if components.size != 3:
        raise ValueError(f"expected a 3-vector, got {components.size} components")
    return _array([_cbor_f64(float(c)) for c in components])


def _quat_map(quaternions) -> bytes:
    if quaternions is None:
        return b"\xf6"
    return _map([(_cbor_int(int(ts)), _quat(q)) for ts, q in sorted(quaternions.items())])


def _vec3_map(vectors) -> bytes:
    if vectors is None:
        return b"\xf6"
    return _map([(_cbor_int(int(ts)), _vec3(v)) for ts, v in sorted(vectors.items())])


# ----------------------------------------------------------------------
# Field encoders
# ----------------------------------------------------------------------


def encode_imu(sample: TimeIMU) -> bytes:
    """One ``IMUData``: a 4-key map, each channel null or a 3-array."""
    return _struct(
        [
            ("timestamp_ms", _cbor_f64(float(sample.timestamp_ms))),
            ("gyro", b"\xf6" if sample.gyro is None else _vec3(sample.gyro)),
            ("accl", b"\xf6" if sample.accl is None else _vec3(sample.accl)),
            ("magn", b"\xf6" if sample.magn is None else _vec3(sample.magn)),
        ]
    )


def encode_imu_list(samples) -> bytes:
    return _array([encode_imu(sample) for sample in samples or []])


def encode_lens_params(params: LensParams) -> bytes:
    """One ``LensParams``: 8 keys in declaration order, always all present."""
    return _struct(
        [
            ("focal_length", _opt_float(_get(params, "focal_length"), narrow=True)),
            ("pixel_pitch", _opt_int_tuple(_get(params, "pixel_pitch"))),
            ("sensor_size_px", _opt_int_tuple(_get(params, "sensor_size_px"))),
            ("capture_area_origin", _opt_f32_tuple(_get(params, "capture_area_origin"))),
            ("capture_area_size", _opt_f32_tuple(_get(params, "capture_area_size"))),
            (
                "pixel_focal_length",
                _opt_float(_get(params, "pixel_focal_length"), narrow=True),
            ),
            (
                "distortion_coefficients",
                _array(
                    [
                        _cbor_f64(float(c))
                        for c in _get(params, "distortion_coefficients", []) or []
                    ]
                ),
            ),
            ("focus_distance", _opt_float(_get(params, "focus_distance"), narrow=True)),
        ]
    )


def encode_lens_params_map(mapping) -> bytes:
    return _map(
        [(_cbor_int(int(ts)), encode_lens_params(p)) for ts, p in sorted(mapping.items())]
    )


def encode_lens_positions(mapping) -> bytes:
    return _map(
        [(_cbor_int(int(ts)), _cbor_f64(float(v))) for ts, v in sorted(mapping.items())]
    )


def encode_camera_identifier(identifier) -> bytes:
    if identifier is None:
        return b"\xf6"
    entries = []
    for name in _CAMERA_IDENTIFIER_FIELDS:
        value = _get(identifier, name)
        if name == "focal_length":
            entries.append((name, _opt_float(value)))
        elif name in ("fps", "video_width", "video_height"):
            entries.append((name, _cbor_int(int(value or 0))))
        else:
            entries.append((name, _text(value or "")))
    return _struct(entries)


def encode_catmull_rom(spline) -> bytes:
    """``CatmullRom<T>`` is a single private ``points`` field of ``(f64, T)``."""
    points = _get(spline, "points", spline) or []
    return _struct(
        [
            (
                "points",
                _array([_array([_cbor_f64(float(p)), _vec3(v)]) for p, v in points]),
            )
        ]
    )


def encode_camera_stab_data(entry) -> bytes:
    if entry is None:
        return b"\xf6"
    return _struct(
        [
            ("offset", _cbor_f64(float(_get(entry, "offset", 0.0)))),
            ("sensor_size", _opt_int_tuple(_get(entry, "sensor_size"))),
            ("crop_area", _opt_f32_tuple(_get(entry, "crop_area"), width=4)),
            ("pixel_pitch", _opt_int_tuple(_get(entry, "pixel_pitch"))),
            ("ibis_spline", encode_catmull_rom(_get(entry, "ibis_spline"))),
            ("ois_spline", encode_catmull_rom(_get(entry, "ois_spline"))),
        ]
    )


def encode_mesh_correction(entries) -> bytes:
    """``Vec<(Vec<f64>, Vec<f32>)>`` — a 2-array per entry."""
    return _array(
        [
            _array(
                [
                    _array([_cbor_f64(float(v)) for v in distances]),
                    _array([_cbor_f64(_f32(v)) for v in weights]),
                ]
            )
            for distances, weights in (entries or [])
        ]
    )


def encode_gravity_vectors(vectors) -> bytes:
    """``Option<TimeVec>`` — the field, not the bare map."""
    return _vec3_map(vectors)


def encode_optional_quat_map(quaternions) -> bytes:
    """``Option<TimeQuat>`` — the field, not the bare map."""
    return _quat_map(quaternions)


def encode_json_value(value) -> bytes:
    """A bare ``serde_json::Value`` field payload."""
    return _json_value(value)


def encode_per_frame_offsets(values) -> bytes:
    return _array([_cbor_f64(float(v)) for v in values or []])


def encode_file_metadata(meta: FileMetadata) -> bytes:
    """Serialize *meta* to the CBOR payload a project file holds.

    Byte-identical to what ciborium produces for the same struct, with one
    caveat: a ``Quat64`` normalizes on construction, so quaternion components
    that came out of a decode come back with up to ~2e-16 of drift. Feed the
    encoder raw ``(w, x, y, z)`` sequences to reproduce a file exactly.
    """
    direction = ReadoutDirection(meta.frame_readout_direction)
    values = {
        "imu_orientation": (
            _text(meta.imu_orientation) if meta.imu_orientation is not None else b"\xf6"
        ),
        "raw_imu": encode_imu_list(meta.raw_imu or []),
        "quaternions": _quat_map(meta.quaternions),
        "gravity_vectors": _vec3_map(meta.gravity_vectors),
        "image_orientations": _quat_map(meta.image_orientations),
        "detected_source": (
            _text(meta.detected_source) if meta.detected_source is not None else b"\xf6"
        ),
        "frame_readout_time": _opt_float(meta.frame_readout_time),
        # The derive has no rename, so the variant NAME goes out, not the value.
        "frame_readout_direction": _text(direction.name),
        "frame_rate": _opt_float(meta.frame_rate),
        "camera_identifier": encode_camera_identifier(meta.camera_identifier),
        "lens_profile": (
            _json_value(meta.lens_profile) if meta.lens_profile is not None else b"\xf6"
        ),
        "lens_positions": encode_lens_positions(meta.lens_positions or {}),
        "lens_params": encode_lens_params_map(meta.lens_params or {}),
        "digital_zoom": _opt_float(meta.digital_zoom),
        "has_accurate_timestamps": b"\xf5" if meta.has_accurate_timestamps else b"\xf4",
        # Upstream's field is a `serde_json::Value` whose default is `Null`, so
        # an unset one goes out as null. This port models it as a plain dict
        # (the telemetry parsers fill it in), which has no separate "unset"
        # state — an empty dict is the same thing here, and it encodes as null
        # so that a file we write matches one Gyroflow would write. Both read
        # back as `{}`, so the distinction is unobservable downstream; it does
        # mean a payload holding a literal empty object will not re-encode to
        # the same bytes.
        "additional_data": (
            _json_value(meta.additional_data) if meta.additional_data else b"\xf6"
        ),
        "per_frame_time_offsets": encode_per_frame_offsets(meta.per_frame_time_offsets),
        "camera_stab_data": _array(
            [encode_camera_stab_data(e) for e in meta.camera_stab_data or []]
        ),
        "mesh_correction": encode_mesh_correction(meta.mesh_correction),
    }
    return _struct([(name, values[name]) for name in _FILE_METADATA_FIELDS])


# ----------------------------------------------------------------------
# Readers
# ----------------------------------------------------------------------


def decode_imu(entry: dict) -> TimeIMU:
    """One ``IMUData`` map -> a ``TimeIMU`` sample."""

    def channel(name):
        value = entry.get(name) if isinstance(entry, dict) else None
        if value is None:
            return None
        return np.asarray(value, dtype=np.float64)

    return TimeIMU(
        timestamp_ms=float(entry.get("timestamp_ms", 0.0)),
        gyro=channel("gyro"),
        accl=channel("accl"),
        magn=channel("magn"),
    )


def decode_lens_params(entry: dict) -> LensParams:
    def opt_float(name):
        value = entry.get(name)
        return None if value is None else float(value)

    def opt_tuple(name):
        value = entry.get(name)
        return None if value is None else tuple(value)

    return LensParams(
        focal_length=opt_float("focal_length"),
        pixel_pitch=opt_tuple("pixel_pitch"),
        sensor_size_px=opt_tuple("sensor_size_px"),
        capture_area_origin=opt_tuple("capture_area_origin"),
        capture_area_size=opt_tuple("capture_area_size"),
        pixel_focal_length=opt_float("pixel_focal_length"),
        distortion_coefficients=[
            float(c) for c in entry.get("distortion_coefficients") or []
        ],
        focus_distance=opt_float("focus_distance"),
    )


def _to_quat_map(value) -> dict[int, Quat64]:
    if not value:
        return {}
    return {
        int(ts): Quat64.from_quaternion(np.asarray(q, dtype=np.float64))
        for ts, q in value.items()
    }


def _to_vec3_map(value) -> dict[int, np.ndarray] | None:
    if not value:
        return None
    return {int(ts): np.asarray(v, dtype=np.float64) for ts, v in value.items()}


def decode_file_metadata(raw: bytes) -> FileMetadata:
    """Parse a project's ``file_metadata`` payload.

    Uses ``cbor2``, which accepts every float width ``ciborium`` might have
    written. Field keys are looked up by name rather than positionally, so a
    file from a version that added or reordered fields still reads as far as
    it can — the struct carries ``#[serde(default)]`` upstream for the same
    reason.

    An empty-but-present map comes back as ``None`` for the two ``Option``
    map fields (``gravity_vectors``, ``image_orientations``): the difference
    is unobservable downstream, where an empty map and no map behave alike,
    but it means such a file will not re-encode byte for byte.
    """
    loaded = cbor2.loads(raw)
    if not isinstance(loaded, dict):
        raise ValueError(
            f"file_metadata should be a CBOR map, got {type(loaded).__name__}"
        )

    name = loaded.get("frame_readout_direction")
    if name is None:
        readout = ReadoutDirection.TopToBottom
    else:
        try:
            readout = ReadoutDirection[name]
        except KeyError:
            # A direction this port does not model: keep the default rather
            # than refusing the whole file.
            readout = ReadoutDirection.TopToBottom

    def opt_float(field):
        value = loaded.get(field)
        return None if value is None else float(value)

    camera_identifier = loaded.get("camera_identifier")

    return FileMetadata(
        imu_orientation=loaded.get("imu_orientation"),
        raw_imu=[decode_imu(entry) for entry in loaded.get("raw_imu") or []],
        quaternions=_to_quat_map(loaded.get("quaternions")),
        gravity_vectors=_to_vec3_map(loaded.get("gravity_vectors")),
        image_orientations=_to_quat_map(loaded.get("image_orientations")) or None,
        detected_source=loaded.get("detected_source"),
        frame_readout_time=opt_float("frame_readout_time"),
        frame_readout_direction=readout,
        frame_rate=opt_float("frame_rate"),
        camera_identifier=(
            _build_camera_identifier(camera_identifier)
            if camera_identifier is not None
            else None
        ),
        lens_profile=loaded.get("lens_profile"),
        lens_positions={
            int(ts): float(v) for ts, v in (loaded.get("lens_positions") or {}).items()
        },
        lens_params={
            int(ts): decode_lens_params(entry)
            for ts, entry in (loaded.get("lens_params") or {}).items()
        },
        digital_zoom=opt_float("digital_zoom"),
        has_accurate_timestamps=bool(loaded.get("has_accurate_timestamps", False)),
        additional_data=loaded.get("additional_data") or {},
        per_frame_time_offsets=[
            float(v) for v in loaded.get("per_frame_time_offsets") or []
        ],
        camera_stab_data=loaded.get("camera_stab_data") or [],
        mesh_correction=loaded.get("mesh_correction") or [],
    )


def _build_camera_identifier(entry: dict):
    """Rebuild a ``CameraIdentifier`` from its CBOR map."""
    from pygyroflow.camera.identifier import CameraIdentifier

    kwargs = {}
    for name in _CAMERA_IDENTIFIER_FIELDS:
        value = entry.get(name)
        if name in ("fps", "video_width", "video_height"):
            kwargs[name] = int(value or 0)
        elif name == "focal_length":
            kwargs[name] = None if value is None else float(value)
        else:
            kwargs[name] = value or ""
    return CameraIdentifier(**kwargs)
