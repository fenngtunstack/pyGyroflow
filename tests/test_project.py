"""`.gyroflow` project files: the JSON container, its payload codecs, and the
manager's load/save bridge.

Two classes of check live here.

The first is self-contained, and leans on the *other* decoder already in this
repo (`tests/decode_gyroflow_project.py`). That one was written independently
and validated against real Gyroflow output, so agreement between it and
`pygyroflow.util` is stronger evidence than either agreeing with itself.

The second class runs against the real reference projects in the shared
test-video directory. Those tests skip when the directory is absent, so the
suite still passes on a clean checkout. One of them is a full
`WithProcessedData` export from real Gyroflow 1.6.3, which is what the CBOR
encoders are checked against byte for byte — before it was found, that layout
was only pinned against the CBOR specification.
"""

from __future__ import annotations

import base64
import json
import pathlib
import random
import struct
import sys
import zlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from decode_gyroflow_project import (  # noqa: E402  (repo-local reference decoder)
    b91decode,
    decode_field,
    parse_f64_list_bincode,
    parse_quat_list_bincode,
)

from pygyroflow.project import (  # noqa: E402
    PROJECT_TITLE,
    PROJECT_VERSION,
    GyroflowProject,
    ProjectVideoInfo,
)
from pygyroflow.util import (  # noqa: E402
    base91_decode,
    base91_encode,
    compress_to_base91,
    decode_cbor_f64_list,
    decode_cbor_quat_map,
    decode_f64_list,
    decode_imu_list,
    decode_quat_map,
    decode_vec3_map,
    decompress_from_base91,
    encode_cbor_f64_list,
    encode_cbor_quat_map,
    encode_f64_list,
    encode_quat_map,
)

_REFERENCE_DIR = pathlib.Path("/home/ft/workspace/testvideos")
_REFERENCE_PROJECTS = [
    _REFERENCE_DIR / "extra-03-GoPro-Hero-6.gyroflow",
    _REFERENCE_DIR / "extra-09-GoPro-Hero5-Session.gyroflow",
]
# A real Gyroflow 1.6.3 `WithProcessedData` export: 25185 IMU samples, all the
# CBOR caches, and a 1 MB `file_metadata` blob. The two projects above carry
# `null` for every motion payload, so only this one can check the encoders
# against bytes Gyroflow actually wrote.
_PROCESSED_PROJECT = (
    pathlib.Path("/home/ft/workspace/PreReserach/msGyroFlow")
    / "DJI_20260507160359_0005_D.gyroflow"
)
_have_references = all(p.is_file() for p in _REFERENCE_PROJECTS)
_have_processed_project = _PROCESSED_PROJECT.is_file()
requires_references = pytest.mark.skipif(
    not _have_references, reason="reference .gyroflow files not present"
)
requires_processed_project = pytest.mark.skipif(
    not _have_processed_project,
    reason="reference WithProcessedData project not present",
)


def _reference_data(path):
    return json.loads(pathlib.Path(path).read_text())


# ----------------------------------------------------------------------
# base91
# ----------------------------------------------------------------------


class TestBase91:
    """The alphabet and the 13/14-bit split, against the repo's other decoder."""

    @pytest.mark.parametrize("n", [0, 1, 2, 3, 7, 8, 64, 257, 4096])
    def test_round_trip(self, n):
        data = bytes(random.Random(n).randrange(256) for _ in range(n))
        assert base91_decode(base91_encode(data)) == data

    def test_matches_reference_decoder(self):
        data = bytes(random.Random(7).randrange(256) for _ in range(1000))
        assert b91decode(base91_encode(data)) == data

    def test_alphabet_is_the_documented_one(self):
        """Pins the alphabet: a wrong table still round-trips with itself."""
        expected = (
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
            "!#$%&()*+,./:;<=>?@[]^_`{|}~\""
        )
        assert len(expected) == 91
        encoded = "".join(base91_encode(bytes([b])) for b in range(256))
        assert set(encoded) <= set(expected)
        # Every symbol of the alphabet is reachable, so a swapped pair is
        # detectable rather than hidden behind an unused character.
        assert set(encoded) == set(expected)

    def test_encode_is_not_base64(self):
        """The alphabets overlap, so assert on a case where they diverge."""
        data = b"\x00\x01\x02\x03"
        assert base91_encode(data) != base64.b64encode(data).decode()
        assert base91_decode(base91_encode(data)) == data

    def test_characters_outside_the_alphabet_are_skipped(self):
        """Faithful to the ``base91`` crate and to the reference decoder here:
        ``slice_decode`` drops anything it does not recognise rather than
        failing, so this must not raise either."""
        assert base91_decode("AAA\x01AAA") == base91_decode("AAAAAA")


class TestZlibEnvelope:
    def test_round_trip(self):
        data = bytes(random.Random(3).randrange(256) for _ in range(5000))
        assert decompress_from_base91(compress_to_base91(data)) == data

    def test_empty_is_empty(self):
        assert decompress_from_base91("") == b""

    def test_is_actually_compressed(self):
        """A repetitive payload must come out shorter than bare base91 of the
        raw bytes — otherwise zlib is not in the path at all."""
        data = b"gyro" * 5000
        assert len(compress_to_base91(data)) < len(base91_encode(data))

    def test_matches_reference_decode_field(self):
        """The reference decoder's whole pipeline, byte for byte."""
        raw = bytes(random.Random(11).randrange(256) for _ in range(4096))
        assert decode_field("q:" + compress_to_base91(raw)) == raw


# ----------------------------------------------------------------------
# bincode payloads (the gyro_source family)
# ----------------------------------------------------------------------


class TestBincodePayloads:
    def test_quat_map_round_trip(self):
        quats = {
            1_000_000: (0.1, 0.2, 0.3, 0.9),
            33_333_000: (-0.5, 0.5, -0.5, 0.5),
        }
        out = decode_quat_map(encode_quat_map(quats))
        assert set(out) == set(quats)
        for ts, q in quats.items():
            assert out[ts] == pytest.approx(q)

    def test_quat_map_layout_is_bincode_legacy(self):
        """u64 count, then (i64, f64 x4) per entry, little-endian, no varint."""
        raw = encode_quat_map({5: (1.0, 2.0, 3.0, 4.0)})
        assert len(raw) == 8 + 8 + 32
        count, ts, *values = struct.unpack("<Qqdddd", raw)
        assert (count, ts) == (1, 5)
        assert values == [1.0, 2.0, 3.0, 4.0]

    def test_quat_map_is_sorted_by_timestamp(self):
        raw = encode_quat_map({9: (0.0, 0.0, 0.0, 1.0), 2: (1.0, 0.0, 0.0, 0.0)})
        assert [row[0] for row in parse_quat_list_bincode(raw)] == [2, 9]

    def test_quat_map_matches_reference_decoder(self):
        quats = {i * 10_000: (i * 0.01, 0.0, 0.0, 1.0) for i in range(50)}
        parsed = parse_quat_list_bincode(encode_quat_map(quats))
        assert len(parsed) == 50
        for ts, w, x, y, z in parsed:
            assert (w, x, y, z) == pytest.approx(quats[ts])

    def test_quat_map_accepts_quat64_objects(self):
        from pygyroflow.types.quaternion import Quat64

        quat = Quat64.identity()
        out = decode_quat_map(encode_quat_map({1: quat}))
        assert out[1] == pytest.approx(quat.quaternion())

    def test_f64_list_round_trip(self):
        values = [0.0, 1.5, -2.25, 1e-9, 1e9]
        assert decode_f64_list(encode_f64_list(values)) == pytest.approx(values)

    def test_f64_list_matches_reference_decoder(self):
        values = [i * 0.125 for i in range(16)]
        assert parse_f64_list_bincode(encode_f64_list(values)) == pytest.approx(
            values
        )

    def test_vec3_map_is_32_bytes_an_entry(self):
        """``gravity_vectors`` is ``BTreeMap<i64, Vector3<f64>>``, not the
        quaternion map — a distinction the length check enforces."""
        raw = struct.pack("<Q", 1) + struct.pack("<qddd", 7, 1.0, 2.0, 3.0)
        assert decode_vec3_map(raw) == {7: (1.0, 2.0, 3.0)}
        with pytest.raises(ValueError):
            decode_quat_map(raw)  # 40-byte entries; 32 is a mismatch

    def test_imu_list_reads_optional_channels(self):
        """``Vec<TimeIMU>``: the channels are Options, so entries vary in
        length and have to be walked rather than sliced."""
        raw = struct.pack("<Q", 2)
        raw += struct.pack("<d", 12.5) + b"\x01" + struct.pack("<3d", 1, 2, 3) + b"\x00\x00"
        raw += struct.pack("<d", 17.5) + b"\x00\x01" + struct.pack("<3d", 0, 9.8, 0) + b"\x00"
        assert decode_imu_list(raw) == [
            (12.5, (1.0, 2.0, 3.0), None, None),
            (17.5, None, (0.0, 9.8, 0.0), None),
        ]

    @pytest.mark.parametrize(
        "decoder",
        [decode_quat_map, decode_f64_list, decode_vec3_map],
    )
    def test_truncated_payload_raises(self, decoder):
        """A short payload must not be read as a shorter map: the header is
        checked against the length rather than trusted."""
        with pytest.raises(ValueError):
            decoder(struct.pack("<Q", 5) + b"\x00" * 10)


# ----------------------------------------------------------------------
# CBOR payloads (the WithProcessedData family)
# ----------------------------------------------------------------------


class TestCborPayloads:
    """The quaternion layout comes from nalgebra's serde impls (``Unit`` ->
    ``Quaternion`` -> ``Vector4`` is a plain 4-element sequence), so the map
    is ``{int: [4 floats]}``.

    The float *width* is not free choice. ciborium writes the shortest form a
    value survives exactly — half before single before double — so the encoder
    has to as well or the bytes differ. Read off a real Gyroflow 1.6.3 export
    (see TestAgainstARealGyroflowProject); the cases below pin the rule
    including the half-before-single order, which is the non-obvious part.
    """

    def test_quat_map_byte_layout(self):
        raw = encode_cbor_quat_map({1: (0.0, 0.0, 0.0, 1.0)})
        expected = (
            b"\xa1"                       # map, 1 entry
            b"\x01"                       # key 1
            b"\x84"                       # array, 4 items
            + (b"\xf9" + struct.pack(">e", 0.0)) * 3
            + b"\xf9" + struct.pack(">e", 1.0)
        )
        assert raw == expected
        assert len(raw) == 15

    def test_f64_list_byte_layout(self):
        """0.5 and -2.0 are both exact halves, so both go out as ``0xf9``."""
        raw = encode_cbor_f64_list([0.5, -2.0])
        assert raw == (
            b"\x82"
            + b"\xf9" + struct.pack(">e", 0.5)
            + b"\xf9" + struct.pack(">e", -2.0)
        )

    def test_integers_and_halves_take_the_half_form(self):
        raw = encode_cbor_f64_list([1.0, 0.0, -0.0])
        assert len(raw) == 1 + 3 * 3
        assert raw.count(b"\xf9") == 3

    @pytest.mark.parametrize(
        "value,prefix,code",
        [
            (66.5, b"\xf9", ">e"),        # in the reference file as a half
            (2738.0, b"\xf9", ">e"),
            (578.375, b"\xfa", ">f"),     # single-exact but not half-exact
            (0.1, b"\xfb", ">d"),         # exact in neither
            (1e300, b"\xfb", ">d"),       # beyond single range
        ],
    )
    def test_the_shortest_exact_form_wins(self, value, prefix, code):
        raw = encode_cbor_f64_list([value])
        assert raw == b"\x81" + prefix + struct.pack(code, value)

    def test_a_half_wins_over_a_single_when_both_are_exact(self):
        """The order that the reference file settled: 2738.0 is single-exact
        *and* half-exact, and Gyroflow wrote the half."""
        assert encode_cbor_f64_list([2738.0])[1:2] == b"\xf9"

    def test_a_failed_narrowing_falls_through_to_the_next_width(self):
        """No exception escapes: an out-of-range value just lands on f64."""
        assert encode_cbor_f64_list([1e300])[1:2] == b"\xfb"
        assert encode_cbor_f64_list([-1e300])[1:2] == b"\xfb"

    def test_nan_is_not_shortened(self):
        """The exactness test is a comparison, and NaN never compares equal —
        so it stays in the widest form instead of silently changing bits."""
        raw = encode_cbor_f64_list([float("nan")])
        assert raw[1:2] == b"\xfb"

    def test_keys_are_sorted(self):
        raw = encode_cbor_quat_map({9: (0.0, 0.0, 0.0, 1.0), 2: (1.0, 0.0, 0.0, 0.0)})
        assert raw[:2] == b"\xa2\x02"

    def test_round_trip(self):
        quats = {i * 1000: (0.1, 0.2, 0.3, 0.9) for i in range(30)}
        assert decode_cbor_quat_map(encode_cbor_quat_map(quats)) == pytest.approx(
            quats, abs=1e-12
        )
        values = [i * 0.25 for i in range(30)]
        assert decode_cbor_f64_list(encode_cbor_f64_list(values)) == pytest.approx(
            values
        )

    def test_reading_a_short_float_form(self):
        """Reading must accept what cbor2 itself writes, because Gyroflow
        files in the wild were not necessarily written by our encoder."""
        import cbor2

        assert decode_cbor_f64_list(cbor2.dumps([1.0, 0.5])) == [1.0, 0.5]
        assert decode_cbor_quat_map(cbor2.dumps({3: [1.0, 0.0, 0.0, 0.0]})) == {
            3: (1.0, 0.0, 0.0, 0.0)
        }

    def test_wrong_container_raises(self):
        import cbor2

        with pytest.raises(ValueError):
            decode_cbor_f64_list(cbor2.dumps({"a": 1}))
        with pytest.raises(ValueError):
            decode_cbor_quat_map(cbor2.dumps([1, 2]))


def _cbor_head(buf, k):
    """Parse one CBOR head; returns (major, argument, next_index)."""
    first = buf[k]
    major, ai = first >> 5, first & 0x1F
    if ai < 24:
        return major, ai, k + 1
    if ai == 24:
        return major, buf[k + 1], k + 2
    if ai == 25:
        return major, struct.unpack_from(">H", buf, k + 1)[0], k + 3
    if ai == 26:
        return major, struct.unpack_from(">I", buf, k + 1)[0], k + 5
    if ai == 27:
        return major, struct.unpack_from(">Q", buf, k + 1)[0], k + 9
    raise AssertionError(f"unsupported additional info {ai}")


def _cbor_float_widths(raw, is_map):
    """The float header widths of a CBOR payload, in order, by walking it.

    Substring counting cannot answer this: ``0xf9`` is a perfectly ordinary
    byte inside an 8-byte float's payload, so ``raw.count(b"\\xf9")`` is
    meaningless. Widths are 3/5/9 bytes after the header byte.
    """
    widths = []
    major, count, k = _cbor_head(raw, 0)
    assert major == (5 if is_map else 4), major

    def read_floats(idx, how_many):
        for _ in range(how_many):
            assert raw[idx] >> 5 == 7, hex(raw[idx])
            width = raw[idx] & 0x1F
            assert width in (25, 26, 27), hex(raw[idx])
            widths.append(width)
            idx += {25: 3, 26: 5, 27: 9}[width]
        return idx

    for _ in range(count):
        if is_map:
            _, _, k = _cbor_head(raw, k)          # key: any integer
            array_major, items, k = _cbor_head(raw, k)
            assert array_major == 4, array_major
            k = read_floats(k, items)
        else:
            k = read_floats(k, count)
            break
    assert k == len(raw), (k, len(raw))
    return widths


def _shortest_float_width(value):
    """The narrowest CBOR float form holding *value* exactly (25/26/27)."""
    for width, code in ((25, ">e"), (26, ">f")):
        if struct.unpack(code, struct.pack(code, value))[0] == value:
            return width
    return 27


# ----------------------------------------------------------------------
# The CBOR encoders against real Gyroflow output
# ----------------------------------------------------------------------


@requires_processed_project
class TestAgainstARealGyroflowProject:
    """Decode a real export's payloads, re-encode, require the same bytes.

    This is the only external check the *write* side of these codecs has. A
    round trip through our own decoder would pass even if encoder and decoder
    shared a wrong assumption; Gyroflow's own output cannot.

    What it caught: every float was written 8 bytes wide. ciborium narrows to
    half or single precision whenever the value survives it exactly, which is
    most of a timestamp list — 934 bytes out of 225734 in the smaller one.
    Decoding was never affected (any width parses), so the bug was invisible
    until real bytes were compared.
    """

    @pytest.fixture(scope="class")
    def payloads(self):
        data = _reference_data(_PROCESSED_PROJECT)
        return data["gyro_source"]

    @pytest.mark.parametrize(
        "name,decoder,encoder",
        [
            ("adaptive_zoom_fovs", decode_cbor_f64_list, encode_cbor_f64_list),
            ("synced_imu_timestamps", decode_cbor_f64_list, encode_cbor_f64_list),
            (
                "synced_imu_timestamps_with_per_frame_offset",
                decode_cbor_f64_list,
                encode_cbor_f64_list,
            ),
            ("integrated_quaternions", decode_cbor_quat_map, encode_cbor_quat_map),
        ],
    )
    def test_payload_round_trips_byte_for_byte(
        self, payloads, name, decoder, encoder
    ):
        raw = decompress_from_base91(payloads[name])
        assert encoder(decoder(raw)) == raw

    def test_the_values_survive_the_round_trip(self, payloads):
        raw = decompress_from_base91(payloads["synced_imu_timestamps"])
        values = decode_cbor_f64_list(raw)
        assert len(values) == 25185
        assert decode_cbor_f64_list(encode_cbor_f64_list(values)) == values

    def test_the_real_export_uses_narrow_floats(self, payloads):
        """If this ever stops holding, the width fix has been undone — the
        byte-for-byte test would still pass on a file that avoided the
        narrow forms by luck."""
        raw = decompress_from_base91(payloads["synced_imu_timestamps"])
        widths = _cbor_float_widths(raw, is_map=False)
        counts = {w: widths.count(w) for w in (25, 26, 27)}
        assert counts == {25: 9, 26: 220, 27: 24956}

    def test_the_quaternion_map_is_all_doubles(self, payloads):
        """The counterpart, from the same file: not one half or single
        appears here, because no quaternion component in this clip is
        exactly representable in one. So the encoder cannot be "always
        narrow" either."""
        raw = decompress_from_base91(payloads["integrated_quaternions"])
        widths = _cbor_float_widths(raw, is_map=True)
        assert len(widths) == 4 * 25185
        assert set(widths) == {27}

    def test_gyroflows_width_is_the_shortest_exact_form_for_every_value(
        self, payloads
    ):
        """The rule, checked against the file rather than against itself.

        This is what makes ``_cbor_f64`` correct rather than merely
        self-consistent: for all 25185 values Gyroflow chose the narrowest
        form that holds the value, with half preferred over single where both
        are exact.
        """
        raw = decompress_from_base91(payloads["synced_imu_timestamps"])
        widths = _cbor_float_widths(raw, is_map=False)
        values = decode_cbor_f64_list(raw)
        assert len(widths) == len(values) == 25185
        assert all(
            width == _shortest_float_width(value)
            for width, value in zip(widths, values)
        )

    def test_narrow_values_reencode_as_halves(self):
        """The values read out of the reference as ``0xf9``, and the rule
        that puts them back there."""
        assert encode_cbor_f64_list([66.5])[1:2] == b"\xf9"
        assert encode_cbor_f64_list([2738.0])[1:2] == b"\xf9"
        assert encode_cbor_f64_list([578.375])[1:2] == b"\xfa"


# ----------------------------------------------------------------------
# Project model
# ----------------------------------------------------------------------


class TestProjectVideoInfo:
    def test_defaults(self):
        info = ProjectVideoInfo()
        assert (info.width, info.height, info.num_frames) == (0, 0, 0)

    def test_keeps_unmodelled_keys(self):
        """Forward compatibility: an unknown key must survive from_dict/to_dict."""
        info = ProjectVideoInfo.from_dict({"width": 16, "height": 8, "future": 42})
        assert info.to_dict()["future"] == 42

    def test_tolerates_null_numbers(self):
        """Gyroflow writes `fps_scale: null`; it must not become a TypeError."""
        info = ProjectVideoInfo.from_dict(
            {"fps_scale": None, "duration_ms": None, "fps": None}
        )
        assert info.fps == 0.0
        assert info.fps_scale is None

    def test_created_at_is_not_invented(self):
        """A version-2 file has no `created_at`; adding a null would be a
        change the round-trip contract forbids."""
        assert "created_at" not in ProjectVideoInfo.from_dict({}).to_dict()
        info = ProjectVideoInfo.from_dict({"created_at": 1234})
        assert info.created_at == 1234
        assert info.to_dict()["created_at"] == 1234


class TestProjectFromDict:
    def test_unknown_top_level_section_survives(self):
        data = {"title": "t", "brand_new_section": {"a": 1}}
        proj = GyroflowProject.from_dict(data)
        assert proj.unknown == {"brand_new_section": {"a": 1}}
        assert proj.to_dict()["brand_new_section"] == {"a": 1}

    def test_offset_keys_become_ints(self):
        """JSON object keys are strings; the gyro keys them by int."""
        proj = GyroflowProject.from_dict({"offsets": {"16950998": 49.5}})
        assert proj.offsets == {16950998: 49.5}
        assert proj.to_dict()["offsets"] == {"16950998": 49.5}

    def test_empty_document_gets_defaults(self):
        proj = GyroflowProject.from_dict({})
        assert proj.title == PROJECT_TITLE
        assert proj.version == PROJECT_VERSION

    def test_sections_are_copied_not_aliased(self):
        """A save must not mutate the dict the caller still holds."""
        source = {"stabilization": {"fov": 1.0}}
        proj = GyroflowProject.from_dict(source)
        proj.stabilization["fov"] = 0.5
        assert source["stabilization"]["fov"] == 1.0

    def test_version_is_kept_as_written(self):
        """`project_version` changes how upstream reads a RED clip's gyro
        timestamps, so a JSON round-trip must not renumber the file."""
        assert GyroflowProject.from_dict({"version": 2}).to_dict()["version"] == 2
        assert PROJECT_VERSION != 2


class TestBlobs:
    def test_quaternions_use_bincode(self):
        proj = GyroflowProject()
        proj.write_blob("quaternions", {5: (1.0, 2.0, 3.0, 4.0)})
        text = proj.gyro_source["quaternions"]
        assert isinstance(text, str)
        # Unwrap with the repo's independent base91 + zlib pair, then parse
        # with its independent bincode parser.
        assert parse_quat_list_bincode(zlib.decompress(b91decode(text)))[0] == (
            5, 1.0, 2.0, 3.0, 4.0
        )
        assert proj.read_blob("quaternions") == {5: (1.0, 2.0, 3.0, 4.0)}

    def test_processed_caches_use_cbor(self):
        """`integrated_quaternions` is a CBOR map, not the bincode layout —
        reading it as bincode would give garbage, so assert on the bytes."""
        proj = GyroflowProject()
        proj.write_blob("integrated_quaternions", {5: (1.0, 2.0, 3.0, 4.0)})
        raw = decompress_from_base91(proj.gyro_source["integrated_quaternions"])
        assert raw[0] == 0xA1  # CBOR map, one entry
        assert raw[1] == 0x05  # integer key, not a u64 count
        assert proj.read_blob("integrated_quaternions") == {
            5: (1.0, 2.0, 3.0, 4.0)
        }

    def test_round_trip_f64_list(self):
        proj = GyroflowProject()
        proj.write_blob("adaptive_zoom_fovs", [0.9, 1.0, 1.1])
        assert proj.read_blob("adaptive_zoom_fovs") == pytest.approx([0.9, 1.0, 1.1])

    def test_read_only_blobs_refuse_to_be_written(self):
        """`gravity_vectors` entries are 32 bytes, not 40; the generic dict
        fallback would emit a payload Gyroflow cannot read."""
        proj = GyroflowProject()
        with pytest.raises(ValueError):
            proj.write_blob("gravity_vectors", {1: (1.0, 2.0, 3.0)})
        with pytest.raises(ValueError):
            proj.write_blob("raw_imu", [(1.0, None, None, None)])

    def test_gravity_vectors_can_still_be_read(self):
        proj = GyroflowProject()
        proj.gyro_source["gravity_vectors"] = compress_to_base91(
            struct.pack("<Q", 1) + struct.pack("<qddd", 7, 1.0, 2.0, 3.0)
        )
        assert proj.read_blob("gravity_vectors") == {7: (1.0, 2.0, 3.0)}

    def test_missing_blob_is_none(self):
        assert GyroflowProject().read_blob("quaternions") is None

    def test_null_blob_is_none(self):
        """Gyroflow writes `"quaternions": null` when there is nothing to
        embed; that is absence, not a malformed payload."""
        proj = GyroflowProject()
        proj.gyro_source["quaternions"] = None
        assert proj.read_blob("quaternions") is None

    def test_unknown_blob_returns_raw_bytes(self):
        """A payload this port does not model is still reachable."""
        proj = GyroflowProject()
        proj.write_blob("brand_new_payload", b"\xde\xad\xbe\xef")
        assert proj.read_blob("brand_new_payload") == b"\xde\xad\xbe\xef"

    def test_corrupt_blob_returns_none(self):
        """A truncated payload must not take the whole load down."""
        proj = GyroflowProject()
        proj.gyro_source["quaternions"] = "!!!! not base91 !!!!"
        assert proj.read_blob("quaternions") is None

    def test_unknown_section_is_none(self):
        assert GyroflowProject().read_blob("quaternions", section="nope") is None

    def test_write_to_unknown_section_raises(self):
        with pytest.raises(TypeError):
            GyroflowProject().write_blob("quaternions", {}, section="nope")

    def test_unsupported_payload_type_raises(self):
        with pytest.raises(TypeError):
            GyroflowProject().write_blob("brand_new_payload", 12345)


# ----------------------------------------------------------------------
# Reference files
# ----------------------------------------------------------------------


# What the two reference files actually say. They are different enough — a
# HERO6 4:3 clip at 2704x2028 and a HERO5 Session 16:9 one at 3840x2160 — that
# asserting shared numbers would only hide which file was read.
_REFERENCE_EXPECTATIONS = {
    "extra-03-GoPro-Hero-6": {
        "size": (2704, 2028),
        "num_frames": 1019,
        "duration_ms": 34001.0,
        "identifier": "gopro-hero6black-wide-2704x2028@29970-no-eis",
        "max_sync_points": 3,
    },
    "extra-09-GoPro-Hero5-Session": {
        "size": (3840, 2160),
        "num_frames": 2955,
        "duration_ms": 98599.0,
        # A HERO5 Session with no published calibration: the identifier is
        # genuinely empty, which is not the same as "failed to parse".
        "identifier": "",
        "max_sync_points": 10,
    },
}


def _expect(path):
    return _REFERENCE_EXPECTATIONS[pathlib.Path(path).stem]


@requires_references
class TestReferenceFiles:
    @pytest.mark.parametrize("path", _REFERENCE_PROJECTS, ids=lambda p: p.stem)
    def test_loads_header(self, path):
        proj = GyroflowProject.load(str(path))
        assert proj.title == PROJECT_TITLE
        assert proj.videofile.endswith(".MP4")
        assert proj.app_version
        assert proj.date

    @pytest.mark.parametrize("path", _REFERENCE_PROJECTS, ids=lambda p: p.stem)
    def test_loads_video_info(self, path):
        info = GyroflowProject.load(str(path)).video_info
        expected = _expect(path)
        assert (info.width, info.height) == expected["size"]
        assert info.num_frames == expected["num_frames"]
        assert info.duration_ms == pytest.approx(expected["duration_ms"])
        assert info.fps == pytest.approx(29.97)

    @pytest.mark.parametrize("path", _REFERENCE_PROJECTS, ids=lambda p: p.stem)
    def test_loads_calibration(self, path):
        cal = GyroflowProject.load(str(path)).calibration_data
        assert cal["camera_brand"] == "GoPro"
        assert cal["identifier"] == _expect(path)["identifier"]
        assert cal["calib_dimension"]["w"] > 0

    @pytest.mark.parametrize("path", _REFERENCE_PROJECTS, ids=lambda p: p.stem)
    def test_loads_offsets_and_sync(self, path):
        proj = GyroflowProject.load(str(path))
        assert proj.offsets
        assert all(isinstance(k, int) for k in proj.offsets)
        assert all(isinstance(v, float) for v in proj.offsets.values())
        assert proj.synchronization["offset_method"] == 2
        assert (
            proj.synchronization["max_sync_points"]
            == _expect(path)["max_sync_points"]
        )

    @pytest.mark.parametrize("path", _REFERENCE_PROJECTS, ids=lambda p: p.stem)
    def test_full_document_round_trip(self, path):
        """Every key of the original, with every value, survives the model.

        This is the property that makes the format safe to round-trip: a
        field this port does not model must come back identical rather than
        be dropped.
        """
        original = _reference_data(path)
        written = GyroflowProject.from_dict(original).to_dict()

        missing = [k for k in original if k not in written]
        changed = {
            k: (original[k], written[k])
            for k in original
            if k in written and original[k] != written[k]
        }
        assert missing == []
        assert changed == {}

    @pytest.mark.parametrize("path", _REFERENCE_PROJECTS, ids=lambda p: p.stem)
    def test_save_load_round_trip(self, path, tmp_path):
        """The same property through the filesystem, not just the dicts."""
        original = _reference_data(path)
        out = tmp_path / "rt.gyroflow"
        GyroflowProject.from_dict(original).save(str(out))

        reloaded = GyroflowProject.load(str(out))
        assert reloaded.to_dict() == GyroflowProject.from_dict(original).to_dict()
        assert reloaded.offsets == GyroflowProject.from_dict(original).offsets

    @pytest.mark.parametrize("path", _REFERENCE_PROJECTS, ids=lambda p: p.stem)
    def test_writes_plain_json(self, path, tmp_path):
        out = tmp_path / "rt.gyroflow"
        GyroflowProject.load(str(path)).save(str(out))
        text = out.read_text(encoding="utf-8")
        assert "NaN" not in text
        assert json.loads(text)["title"] == PROJECT_TITLE

    @pytest.mark.parametrize("path", _REFERENCE_PROJECTS, ids=lambda p: p.stem)
    def test_absent_payload_is_reported_absent(self, path):
        """These files carry no embedded gyro — `quaternions` is null — and
        that has to read as "nothing here", not as a decode failure."""
        proj = GyroflowProject.load(str(path))
        assert proj.gyro_source.get("quaternions") is None
        assert proj.read_blob("quaternions") is None


@requires_references
class TestCalibrationRoundTrip:
    """`LensProfile.get_json_value` must reproduce the file's calibration_data.

    The reverse direction is what makes `save_project` useful: a project
    written by this port has to be readable by the real Gyroflow.
    """

    @pytest.mark.parametrize("path", _REFERENCE_PROJECTS, ids=lambda p: p.stem)
    def test_round_trip(self, path):
        from pygyroflow.lens import LensProfile

        original = _reference_data(path)["calibration_data"]
        written = LensProfile.from_json(dict(original)).get_json_value()

        missing = [k for k in original if k not in written]
        changed = {
            k: (original[k], written[k])
            for k in original
            if k in written and original[k] != written[k]
        }
        assert missing == []
        assert changed == {}

    @pytest.mark.parametrize("path", _REFERENCE_PROJECTS, ids=lambda p: p.stem)
    def test_identifier_survives(self, path):
        from pygyroflow.lens import LensProfile

        original = _reference_data(path)["calibration_data"]
        written = LensProfile.from_json(dict(original)).get_json_value()
        assert written["identifier"] == original["identifier"]
        assert written["distortion_model_id"] == original["distortion_model_id"]


# ----------------------------------------------------------------------
# Manager bridge
# ----------------------------------------------------------------------


def _new_manager():
    from pygyroflow.manager import StabilizationManager

    mgr = StabilizationManager()
    mgr.init_from_video_data(34001.0, 29.97, 1019, (2704, 2028))
    mgr.set_size(2704, 2028)
    return mgr


@requires_references
class TestManagerLoad:
    @pytest.fixture
    def manager(self):
        return _new_manager()

    def test_applies_video_info(self, manager):
        manager.load_project(str(_REFERENCE_PROJECTS[0]))
        assert manager.params.size == (2704, 2028)
        assert manager.params.fps == pytest.approx(29.97)
        assert manager.params.frame_count == 1019
        assert manager.params.duration_ms == pytest.approx(34001.0)

    def test_applies_calibration(self, manager):
        manager.load_project(str(_REFERENCE_PROJECTS[0]))
        assert manager.lens is not None
        assert manager.lens.camera_matrix[0][0] == pytest.approx(1186.54, abs=0.01)

    def test_applies_readout_time(self, manager):
        manager.load_project(str(_REFERENCE_PROJECTS[0]))
        assert abs(manager.params.frame_readout_time) == pytest.approx(
            11.1111, abs=1e-4
        )

    def test_applies_offsets(self, manager):
        manager.load_project(str(_REFERENCE_PROJECTS[0]))
        offsets = manager.gyro.get_offsets()
        assert len(offsets) == 3
        assert offsets[16950998] == pytest.approx(49.3182, abs=1e-4)

    def test_applies_smoothing_params(self, manager):
        manager.load_project(str(_REFERENCE_PROJECTS[0]))
        assert manager.smoothing.current().get_name() == "Default"
        assert manager.smoothing.current().get_parameter(
            "smoothness"
        ) == pytest.approx(0.4957877358490566)

    def test_applies_adaptive_zoom_window(self, manager):
        manager.load_project(str(_REFERENCE_PROJECTS[0]))
        assert manager.params.adaptive_zoom_window == pytest.approx(4.0)

    def test_keeps_the_raw_project_on_the_manager(self, manager):
        """The unmodelled sections stay reachable for a later save."""
        manager.load_project(str(_REFERENCE_PROJECTS[0]))
        assert manager.project.synchronization["offset_method"] == 2
        assert manager.input_file.project_file_url == str(_REFERENCE_PROJECTS[0])

    def test_trim_range_uses_the_duration(self, manager):
        manager.load_project(str(_REFERENCE_PROJECTS[0]))
        assert manager.params.trim_ranges == []

    def test_trim_range_with_negative_end(self, manager, tmp_path):
        """A negative end is measured back from the clip's end."""
        original = _reference_data(_REFERENCE_PROJECTS[0])
        original["trim_ranges_ms"] = [[0.0, -5000.0]]
        path = tmp_path / "trim.gyroflow"
        path.write_text(json.dumps(original))

        manager.load_project(str(path))
        assert manager.params.trim_ranges == pytest.approx(
            [(0.0, (34001.0 - 5000.0) / 34001.0)]
        )

    def test_unknown_smoothing_method_is_ignored(self, manager, tmp_path):
        """A method from a newer Gyroflow must not raise."""
        original = _reference_data(_REFERENCE_PROJECTS[0])
        original["stabilization"]["method"] = "SomeFutureAlgo"
        path = tmp_path / "future.gyroflow"
        path.write_text(json.dumps(original))

        manager.load_project(str(path))
        assert manager.smoothing.current().get_name() == "Default"

    def test_unknown_field_does_not_break_loading(self, manager, tmp_path):
        original = _reference_data(_REFERENCE_PROJECTS[0])
        original["stabilization"]["invented_slider"] = 0.25
        original["a_whole_new_section"] = {"x": 1}
        path = tmp_path / "extended.gyroflow"
        path.write_text(json.dumps(original))

        manager.load_project(str(path))
        assert manager.params.size == (2704, 2028)


@requires_references
class TestManagerSave:
    @pytest.fixture
    def manager(self):
        mgr = _new_manager()
        mgr.load_project(str(_REFERENCE_PROJECTS[0]))
        return mgr

    def test_writes_live_state(self, manager, tmp_path):
        manager.params.fov = 0.8
        manager.params.adaptive_zoom_window = -1.0
        out = tmp_path / "saved.gyroflow"
        manager.save_project(str(out))

        data = json.loads(out.read_text())
        assert data["stabilization"]["fov"] == pytest.approx(0.8)
        assert data["stabilization"]["adaptive_zoom_window"] == pytest.approx(-1.0)
        assert data["version"] == PROJECT_VERSION

    def test_writes_the_version_4_fields(self, manager, tmp_path):
        """A reader that trusts the version number needs the fields that
        version implies to actually be there."""
        out = tmp_path / "saved.gyroflow"
        manager.save_project(str(out))

        data = json.loads(out.read_text())
        for key in (
            "frame_offset",
            "focal_length_smoothing_enabled",
            "focal_length_smoothing_strength",
        ):
            assert key in data["stabilization"]
        assert data["video_info"]["created_at"] == manager.params.video_created_at

    def test_stamps_the_writer_not_the_loaded_file(self, manager, tmp_path):
        """`app_version` and `date` describe who wrote the file. Carrying the
        loaded file's values over would claim Gyroflow wrote this."""
        import datetime

        import pygyroflow

        original = _reference_data(_REFERENCE_PROJECTS[0])
        assert manager.project.app_version == original["app_version"]

        out = tmp_path / "saved.gyroflow"
        manager.save_project(str(out))

        data = json.loads(out.read_text())
        assert data["app_version"] == pygyroflow.__version__
        assert data["app_version"] != original["app_version"]
        assert data["date"] == datetime.date.today().isoformat()

    def test_preserves_sections_it_does_not_model(self, manager, tmp_path):
        """`output` and anything unknown are the caller's; save must not
        clobber them."""
        manager.project.output["bitrate"] = 63
        manager.project.unknown["a_whole_new_section"] = {"x": 1}
        out = tmp_path / "saved.gyroflow"
        manager.save_project(str(out))

        data = json.loads(out.read_text())
        assert data["output"]["bitrate"] == 63
        assert data["a_whole_new_section"] == {"x": 1}

    def test_keeps_the_embedded_gyro_payloads(self, manager, tmp_path):
        """The quaternion blob is expensive to recompute; a save must not
        drop it just because the loader kept it raw."""
        manager.project.gyro_source["quaternions"] = compress_to_base91(
            encode_quat_map({5: (1.0, 0.0, 0.0, 0.0)})
        )
        out = tmp_path / "saved.gyroflow"
        manager.save_project(str(out))

        reloaded = GyroflowProject.load(str(out))
        assert reloaded.read_blob("quaternions") == {5: (1.0, 0.0, 0.0, 0.0)}

    def test_round_trips_through_the_manager(self, manager, tmp_path):
        manager.params.fov = 0.75
        out = tmp_path / "saved.gyroflow"
        manager.save_project(str(out))

        other = _new_manager()
        other.load_project(str(out))
        assert other.params.fov == pytest.approx(0.75)
        assert other.params.size == (2704, 2028)
        assert abs(other.params.frame_readout_time) == pytest.approx(11.1111, abs=1e-4)
        assert len(other.gyro.get_offsets()) == 3
        assert other.smoothing.current().get_parameter(
            "smoothness"
        ) == pytest.approx(0.4957877358490566)

    def test_saves_without_a_loaded_project(self, tmp_path):
        """A session that never loaded a project still writes a valid file."""
        from pygyroflow.manager import StabilizationManager

        mgr = StabilizationManager()
        mgr.init_from_video_data(1000.0, 30.0, 30, (640, 480))
        mgr.set_size(640, 480)
        out = tmp_path / "fresh.gyroflow"
        mgr.save_project(str(out))

        data = json.loads(out.read_text())
        assert data["title"] == PROJECT_TITLE
        assert data["video_info"]["width"] == 640
        assert data["app_version"]
        assert data["version"] == PROJECT_VERSION

    def test_refuses_to_write_non_finite_numbers(self, manager, tmp_path):
        """JSON has no NaN; emitting the bare token would produce a file no
        other parser can read."""
        manager.params.fov = float("nan")
        with pytest.raises(ValueError):
            manager.save_project(str(tmp_path / "saved.gyroflow"))


class TestProjectFileShape:
    def test_save_produces_utf8_json(self, tmp_path):
        proj = GyroflowProject(videofile="E:/clips/测试 视频.mp4")
        out = tmp_path / "u.gyroflow"
        proj.save(str(out))
        assert json.loads(out.read_text(encoding="utf-8"))["videofile"] == (
            "E:/clips/测试 视频.mp4"
        )

    def test_new_project_stamps_the_current_version(self):
        assert GyroflowProject().to_dict()["version"] == PROJECT_VERSION

    def test_trim_ranges_ms_is_omitted_when_empty(self):
        """Matches upstream, which only writes the key when there is a range
        (and the GUI's legacy trim_start/trim_end otherwise)."""
        data = GyroflowProject().to_dict()
        assert "trim_ranges_ms" not in data
        assert data["trim_start"] == 0.0
        assert data["trim_end"] == 1.0

    def test_zlib_is_lossless_for_the_envelope(self):
        """A guard on the compression level: level 9 and the decoder in the
        Rust reference have to agree on the stream."""
        raw = bytes(random.Random(5).randrange(256) for _ in range(1000))
        text = compress_to_base91(raw)
        assert zlib.decompress(b91decode(text)) == raw
