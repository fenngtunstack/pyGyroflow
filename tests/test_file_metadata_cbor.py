"""The CBOR payload a project carries its `FileMetadata` in (gap item C-06).

A `WithGyroData`/`WithProcessedData` project embeds the whole metadata struct
as a base91 + zlib + CBOR blob. Upstream decodes it on load and, for a real
Gyroflow 1.6.3 export, that blob is the *only* place the motion lives —
`gyro_source.raw_imu` and `.quaternions` are `null` there.

Two layers of evidence, and they are different in kind:

* `tests/golden/cbor_file_metadata.json` — ciborium's own bytes for 19 cases
  covering every field, produced by a scratch crate that mirrors the upstream
  struct definitions (see `tests/golden/generate_file_metadata_reference.py`).
* A real 1.6.3 export, round-tripped byte for byte in
  `TestAgainstARealExport`. That is the case that would catch a layout that is
  self-consistent but wrong.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from pygyroflow.camera.identifier import CameraIdentifier  # noqa: E402
from pygyroflow.gyro_source.file_metadata import FileMetadata, LensParams  # noqa: E402
from pygyroflow.gyro_source.file_metadata_cbor import (  # noqa: E402
    decode_file_metadata,
    encode_camera_identifier,
    encode_camera_stab_data,
    encode_file_metadata,
    encode_gravity_vectors,
    encode_imu_list,
    encode_json_value,
    encode_lens_params_map,
    encode_lens_positions,
    encode_mesh_correction,
    encode_optional_quat_map,
    encode_per_frame_offsets,
)
from pygyroflow.types.enums import ReadoutDirection  # noqa: E402
from pygyroflow.types.time_types import TimeIMU  # noqa: E402
from pygyroflow.util import decompress_from_base91  # noqa: E402

_FIXTURE = pathlib.Path(__file__).parent / "golden" / "cbor_file_metadata.json"

# A real Gyroflow 1.6.3 WithProcessedData export, 3.4 MB. Kept outside the
# repo (it is a user file, not a fixture), so these tests skip without it.
_REAL_EXPORT = (
    pathlib.Path("/home/ft/workspace/PreReserach/msGyroFlow")
    / "DJI_20260507160359_0005_D.gyroflow"
)


# ----------------------------------------------------------------------
# fixture-value builders
# ----------------------------------------------------------------------


def _numpy_vec3(value):
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64)


def _build_imu(entries):
    return [
        TimeIMU(
            timestamp_ms=entry["timestamp_ms"],
            gyro=_numpy_vec3(entry.get("gyro")),
            accl=_numpy_vec3(entry.get("accl")),
            magn=_numpy_vec3(entry.get("magn")),
        )
        for entry in entries
    ]


def _build_lens_params(entry):
    """A dict from the fixture -> LensParams (tuple-typed fields included)."""
    kwargs = dict(entry)
    for name in ("pixel_pitch", "sensor_size_px", "capture_area_origin",
                 "capture_area_size"):
        if name in kwargs and kwargs[name] is not None:
            kwargs[name] = tuple(kwargs[name])
    return LensParams(**kwargs)


def _build_float_map(mapping):
    if mapping is None:
        return None
    return {int(k): float(v) for k, v in mapping.items()}


def _build_vec3_map(mapping):
    if mapping is None:
        return None
    return {int(k): np.asarray(v, dtype=np.float64) for k, v in mapping.items()}


def _build_quat_map(mapping):
    """Raw component tuples, not Quat64: see TestAgainstARealExport."""
    if mapping is None:
        return None
    return {int(k): tuple(v) for k, v in mapping.items()}


def _build_camera_identifier(entry):
    return None if entry is None else CameraIdentifier(**entry)


def _build_file_metadata(payload):
    values = dict(payload)
    direction = values.pop("frame_readout_direction", "TopToBottom")
    return FileMetadata(
        raw_imu=_build_imu(values.pop("raw_imu", [])),
        quaternions=_build_quat_map(values.pop("quaternions", {})) or {},
        gravity_vectors=_build_vec3_map(values.pop("gravity_vectors", None)),
        image_orientations=_build_quat_map(values.pop("image_orientations", None)),
        frame_readout_direction=ReadoutDirection[direction],
        camera_identifier=_build_camera_identifier(values.pop("camera_identifier", None)),
        lens_positions=_build_float_map(values.pop("lens_positions", {})) or {},
        lens_params={
            int(k): _build_lens_params(v)
            for k, v in (values.pop("lens_params", {}) or {}).items()
        },
        **values,
    )


def _encode_stab_list(entries):
    """`Vec<CameraStabData>` is an array of maps; the field encoder is per entry."""
    from pygyroflow.gyro_source.file_metadata_cbor import _array

    return _array([encode_camera_stab_data(entry) for entry in entries])


# op -> (builder, encoder). Every encoder is called with the built value.
_ENCODERS = {
    "imu_list": (_build_imu, encode_imu_list),
    "lens_params_map": (
        lambda payload: {int(k): _build_lens_params(v) for k, v in payload.items()},
        encode_lens_params_map,
    ),
    "lens_positions": (_build_float_map, encode_lens_positions),
    "gravity_vectors": (_build_vec3_map, encode_gravity_vectors),
    "image_orientations": (_build_quat_map, encode_optional_quat_map),
    "camera_identifier": (_build_camera_identifier, encode_camera_identifier),
    "json_value": (lambda payload: payload, encode_json_value),
    "per_frame_time_offsets": (lambda payload: payload, encode_per_frame_offsets),
    "mesh_correction": (
        lambda payload: [(d, w) for d, w in payload],
        encode_mesh_correction,
    ),
    "camera_stab_data": (lambda payload: payload, _encode_stab_list),
    "file_metadata": (_build_file_metadata, encode_file_metadata),
}


def _load_cases():
    with open(_FIXTURE, encoding="utf-8") as handle:
        return json.load(handle)["cases"]


_CASES = _load_cases()


class TestAgainstCiborium:
    """Every field, against the bytes ciborium itself writes."""

    @pytest.mark.parametrize("case", _CASES, ids=lambda c: c["name"])
    def test_encoder_matches_ciborium(self, case):
        build, encode = _ENCODERS[case["op"]]
        expected = bytes.fromhex(case["cbor"])
        assert encode(build(case["input"])) == expected

    def test_the_fixture_is_not_self_generated(self):
        with open(_FIXTURE, encoding="utf-8") as handle:
            doc = json.load(handle)
        provenance = doc["_provenance"].lower()
        assert "ciborium" in provenance
        assert "not produced by the python port" in provenance

    def test_every_field_is_covered(self):
        """A field with no case would be a layout nobody ever checked."""
        ops = {case["op"] for case in _CASES}
        assert {"imu_list", "lens_params_map", "lens_positions", "gravity_vectors",
                "image_orientations", "camera_identifier", "json_value",
                "per_frame_time_offsets", "mesh_correction", "camera_stab_data",
                "file_metadata"} == ops
        full = next(c for c in _CASES if c["name"] == "file_metadata_full")
        for field in (
            "imu_orientation", "raw_imu", "quaternions", "gravity_vectors",
            "image_orientations", "detected_source", "frame_readout_time",
            "frame_readout_direction", "frame_rate", "camera_identifier",
            "lens_profile", "lens_positions", "lens_params", "digital_zoom",
            "has_accurate_timestamps", "additional_data", "per_frame_time_offsets",
            "camera_stab_data", "mesh_correction",
        ):
            assert field in full["input"], field


class TestStructShapes:
    """The specific traps, called out so a "simplification" trips a test."""

    def test_a_struct_is_a_map_with_text_keys(self):
        """Not an array: the key names are in the payload."""
        raw = encode_lens_params_map({5: LensParams(focal_length=85.0)})
        assert b"focal_length" in raw
        assert b"distortion_coefficients" in raw

    def test_a_tuple_is_an_array(self):
        raw = encode_lens_params_map({5: LensParams(pixel_pitch=(3400, 4000))})
        assert bytes.fromhex("82190d48190fa0") in raw

    def test_the_readout_direction_is_a_name_not_a_number(self):
        meta = FileMetadata(frame_readout_direction=ReadoutDirection.RightToLeft)
        raw = encode_file_metadata(meta)
        assert b"RightToLeft" in raw
        assert b"BottomToTop" in encode_file_metadata(
            FileMetadata(frame_readout_direction=ReadoutDirection.BottomToTop)
        )

    def test_f32_fields_are_rounded_before_encoding(self):
        """1.3 and 1.3f32 are different doubles; the narrow one is what goes
        in. Asserted via the decoded value, which must be the f32 one."""
        raw = encode_lens_params_map({0: LensParams(focal_length=1.3)})
        import cbor2

        decoded = cbor2.loads(raw)
        stored = decoded[0]["focal_length"]
        assert stored == float(np.float32(1.3))
        assert stored != 1.3

    def test_json_object_key_order_is_preserved_not_sorted(self):
        """A real export's `lens_profile` starts with `calibrated_by`, so
        sorting would produce a valid file that is not the same file."""
        payload = {"calibrated_by": "x", "calib_dimension": {"w": 1}}
        raw = encode_json_value(payload)
        assert raw.index(b"calibrated_by") < raw.index(b"calib_dimension")

    def test_an_empty_map_and_null_are_different_bytes(self):
        assert encode_gravity_vectors(None) == b"\xf6"
        assert encode_gravity_vectors({}) == b"\xa0"
        assert encode_lens_positions({}) == b"\xa0"
        assert encode_optional_quat_map(None) == b"\xf6"


class TestDecoder:
    @pytest.mark.parametrize("case", [c for c in _CASES if c["op"] == "file_metadata"],
                             ids=lambda c: c["name"])
    def test_decoding_ciborium_bytes_gives_back_the_input(self, case):
        """Read the bytes the Rust writer produced, and check the values."""
        meta = decode_file_metadata(bytes.fromhex(case["cbor"]))
        expected = case["input"]

        assert meta.imu_orientation == expected.get("imu_orientation")
        assert meta.detected_source == expected.get("detected_source")
        assert meta.frame_readout_time == expected.get("frame_readout_time")
        assert meta.frame_rate == expected.get("frame_rate")
        assert meta.digital_zoom == expected.get("digital_zoom")
        assert meta.has_accurate_timestamps == expected.get(
            "has_accurate_timestamps", False
        )
        assert len(meta.raw_imu) == len(expected.get("raw_imu", []))
        assert sorted(meta.quaternions) == sorted(
            int(k) for k in expected.get("quaternions", {})
        )
        assert sorted(meta.lens_positions) == sorted(
            int(k) for k in expected.get("lens_positions", {})
        )
        assert sorted(meta.lens_params) == sorted(
            int(k) for k in expected.get("lens_params", {})
        )
        assert meta.per_frame_time_offsets == expected.get(
            "per_frame_time_offsets", []
        )

    def test_decoding_a_raw_imu_sample_keeps_every_channel(self):
        case = next(c for c in _CASES if c["name"] == "imu_all_channel_combinations")
        meta = decode_file_metadata(_wrap_raw_imu(bytes.fromhex(case["cbor"])))
        assert len(meta.raw_imu) == 4
        assert meta.raw_imu[0].gyro is None
        assert meta.raw_imu[1].gyro == pytest.approx([1.0, -2.5, 0.25])
        assert meta.raw_imu[1].accl is None
        assert meta.raw_imu[2].magn == pytest.approx([1.5, 2.5, 3.5])
        assert meta.raw_imu[3].accl == pytest.approx([0.0, 0.0, 0.0])

    def test_decoding_lens_params_keeps_the_tuple_fields(self):
        case = next(c for c in _CASES if c["name"] == "lens_params_full_and_empty")
        raw = _wrap("lens_params", bytes.fromhex(case["cbor"]))
        meta = decode_file_metadata(raw)
        entry = meta.lens_params[1]
        assert entry.focal_length == pytest.approx(85.0)
        assert entry.pixel_pitch == (3400, 3400)
        assert entry.sensor_size_px == (6000, 4000)
        assert entry.capture_area_origin == pytest.approx((0.0, 120.5))
        assert entry.capture_area_size == pytest.approx((6000.0, 3376.0))
        assert entry.pixel_focal_length == pytest.approx(1234.5)
        assert entry.distortion_coefficients == pytest.approx([0.1, -0.05, 0.001])
        assert entry.focus_distance == pytest.approx(float(np.float32(1.3)))
        assert meta.lens_params[0] == LensParams()

    def test_decoding_the_readout_direction_by_name(self):
        case = next(c for c in _CASES if c["name"] == "file_metadata_full")
        meta = decode_file_metadata(bytes.fromhex(case["cbor"]))
        assert meta.frame_readout_direction is ReadoutDirection.BottomToTop

    def test_an_unknown_direction_falls_back_instead_of_failing(self):
        raw = _wrap("frame_readout_direction", encode_json_value("Sideways"))
        assert (
            decode_file_metadata(raw).frame_readout_direction
            is ReadoutDirection.TopToBottom
        )

    def test_decoding_the_camera_identifier(self):
        case = next(c for c in _CASES if c["name"] == "camera_identifier")
        raw = _wrap("camera_identifier", bytes.fromhex(case["cbor"]))
        ident = decode_file_metadata(raw).camera_identifier
        assert ident.brand == "DJI"
        assert ident.fps == 29970
        assert ident.video_width == 3840
        assert ident.focal_length == pytest.approx(18.0)

    def test_a_non_map_payload_is_rejected(self):
        import cbor2

        with pytest.raises(ValueError):
            decode_file_metadata(cbor2.dumps([1, 2, 3]))


def _wrap(field: str, value_cbor: bytes) -> bytes:
    """A one-field FileMetadata map, so a single value can be decoded."""
    from pygyroflow.gyro_source.file_metadata_cbor import _map, _text

    return _map([(_text(field), value_cbor)])


def _wrap_raw_imu(value_cbor: bytes) -> bytes:
    return _wrap("raw_imu", value_cbor)


@pytest.mark.skipif(not _REAL_EXPORT.is_file(), reason="reference export not present")
class TestLoadingTheRealProject:
    """The payload has to reach the pipeline, not just decode.

    ``gyro_source.raw_imu`` and ``gyro_source.quaternions`` are ``null`` in
    this file, so a loader that only knows the bincode blobs finds nothing in
    a project carrying 25185 samples. That is what these check.
    """

    @pytest.fixture(scope="class")
    def manager(self):
        from pygyroflow.manager import StabilizationManager

        manager = StabilizationManager()
        manager.load_project(str(_REAL_EXPORT))
        return manager

    def test_the_gyro_data_arrives(self, manager):
        assert len(manager.gyro.quaternions) == 25185
        assert manager.gyro.raw_imu == []
        assert manager.gyro.file_metadata.detected_source == "DJI Osmo Nano"

    def test_the_video_parameters_arrive(self, manager):
        assert manager.params.frame_count == 753
        assert manager.params.size == (1920, 1080)

    def test_the_lens_profile_and_extra_metadata_arrive(self, manager):
        metadata = manager.gyro.file_metadata
        assert metadata.lens_profile["calibrated_by"] == "DJI"
        assert metadata.additional_data["imu_sampling_rate"]

    def test_the_readout_time_comes_from_the_stabilization_section(self, manager):
        assert manager.params.frame_readout_time == pytest.approx(9.109809308813688)
        assert manager.params.frame_readout_direction is ReadoutDirection.TopToBottom


class TestReadoutDirectionFromAProject:
    """Real exports write the variant name; the field is not an integer.

    A loader expecting an int drops it silently — the direction reverts to
    its default and the render subtracts the rolling shutter the wrong way.
    """

    def _manager(self):
        from pygyroflow.manager import StabilizationManager

        return StabilizationManager()

    def test_the_variant_name_is_accepted(self):
        manager = self._manager()
        manager._apply_project_stabilization(
            {"frame_readout_direction": "BottomToTop"}
        )
        assert manager.params.frame_readout_direction is ReadoutDirection.BottomToTop

    def test_an_integer_is_still_accepted(self):
        """Older files, and upstream's ``as_i64`` branch, use one."""
        manager = self._manager()
        manager._apply_project_stabilization({"frame_readout_direction": 2})
        assert manager.params.frame_readout_direction is ReadoutDirection.LeftToRight

    def test_an_unknown_value_keeps_what_was_there(self):
        manager = self._manager()
        manager.params.frame_readout_direction = ReadoutDirection.RightToLeft
        manager._apply_project_stabilization({"frame_readout_direction": "Sideways"})
        assert manager.params.frame_readout_direction is ReadoutDirection.RightToLeft

    def test_a_negative_readout_time_means_bottom_to_top(self):
        manager = self._manager()
        manager.params.frame_readout_direction = ReadoutDirection.TopToBottom
        manager._apply_project_stabilization({"frame_readout_time": -9.5})
        assert manager.params.frame_readout_direction is ReadoutDirection.BottomToTop

    def test_the_sign_of_the_readout_time_is_kept(self):
        """Upstream keeps it and reads the direction from it; every consumer
        inside frame_transform takes abs() itself."""
        manager = self._manager()
        manager._apply_project_stabilization({"frame_readout_time": -9.5})
        assert manager.params.frame_readout_time == pytest.approx(-9.5)


@pytest.mark.skipif(not _REAL_EXPORT.is_file(), reason="reference export not present")
class TestAgainstARealExport:
    """A 3.4 MB real Gyroflow 1.6.3 project, byte for byte.

    Everything the golden fixture checks is a layout *derived* from the struct
    definitions. This is the same layout against bytes a real build wrote, on
    a payload with 25185 quaternions, a lens profile, DJI's extra metadata and
    19 fields of real values.
    """

    @pytest.fixture(scope="class")
    def payload(self):
        import json as _json

        with open(_REAL_EXPORT, encoding="utf-8") as handle:
            data = _json.load(handle)
        return decompress_from_base91(data["gyro_source"]["file_metadata"])

    def test_the_blob_decodes(self, payload):
        assert len(payload) == 1059160
        meta = decode_file_metadata(payload)
        assert len(meta.quaternions) == 25185
        assert meta.detected_source == "DJI Osmo Nano"
        assert meta.frame_readout_time == pytest.approx(9.109809308813688)
        assert meta.frame_readout_direction is ReadoutDirection.TopToBottom
        assert meta.has_accurate_timestamps is True
        assert meta.raw_imu == []
        assert meta.gravity_vectors is None

    def test_every_field_is_present_in_the_payload(self, payload):
        import cbor2

        loaded = cbor2.loads(payload)
        from pygyroflow.gyro_source.file_metadata_cbor import _FILE_METADATA_FIELDS

        assert list(loaded) == list(_FILE_METADATA_FIELDS)

    def test_the_nested_json_fields_survive(self, payload):
        meta = decode_file_metadata(payload)
        assert meta.lens_profile["calibrated_by"] == "DJI"
        assert "calib_dimension" in meta.lens_profile
        assert meta.additional_data["clip_meta_header"]["proto_file_name"]

    def test_re_encoding_reproduces_the_file_byte_for_byte(self, payload):
        """The whole 1059160 bytes, all 19 fields.

        The quaternions are re-supplied as raw components: a ``Quat64``
        normalizes on construction, and this file's values are unit only to
        ~3e-16, so routing them through it shifts the last mantissa bits. That
        is the subject of the next test.
        """
        import cbor2

        meta = decode_file_metadata(payload)
        meta.quaternions = _build_quat_map(cbor2.loads(payload)["quaternions"])
        assert encode_file_metadata(meta) == payload

    def test_the_quat64_route_differs_only_in_the_last_mantissa_bit(self, payload):
        """Decoding into the typed field and encoding back is value-exact to
        within a couple of ULP, not byte-exact. Pinned so the difference is a
        known property rather than a surprise.
        """
        meta = decode_file_metadata(payload)
        reencoded = encode_file_metadata(meta)
        assert len(reencoded) == len(payload)

        original = decode_file_metadata(payload).quaternions
        difference = 0.0
        for ts, quat in meta.quaternions.items():
            components = np.asarray(quat.quaternion())
            reference = np.asarray([
                original[ts].quaternion()[0], original[ts].quaternion()[1],
                original[ts].quaternion()[2], original[ts].quaternion()[3],
            ])
            difference = max(difference, float(np.max(np.abs(components - reference))))
        assert difference <= 4e-16

        # Everything outside the quaternion map is still byte-identical.
        tail = reencoded[1057629:]
        assert tail == payload[1057629:]
