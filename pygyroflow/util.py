"""Shared utility functions.

Port of Gyroflow's timestamp_at_frame / frame_at_timestamp (lib.rs) and the
base91 + zlib + bincode/cbor codecs util.rs uses to pack the large blobs in
a .gyroflow project file.

The codecs have to match byte for byte: a project file written here has to
load in Gyroflow and vice versa. The base91 alphabet and the bit packing
follow the reference implementation the Rust ``base91`` crate uses, and the
"bincode legacy" configuration is what ``bincode::config::legacy()`` selects.

Which codec applies is decided by *where* in the file a blob lives, because
upstream uses two different ones (util.rs):

``gyro_source`` -- ``compress_to_base91`` / ``decompress_from_base91``,
bincode-legacy. That is ``Vec<TimeIMU>`` for ``raw_imu``, ``BTreeMap<i64,
Quat64>`` for ``quaternions`` and ``image_orientations``, and ``BTreeMap<i64,
Vector3<f64>>`` for ``gravity_vectors``.

``WithProcessedData`` exports -- ``compress_to_base91_cbor`` /
``decompress_from_base91_cbor``, CBOR. Those are the caches a plugin reads:
``integrated_quaternions``, ``smoothed_quaternions``, ``adaptive_zoom_fovs``,
``synced_imu_timestamps``, ``focal_lengths`` and friends.

Both wrap the payload in the same zlib-best + base91 envelope; only the
payload encoding differs.

Upstream writes the bare base91 with no marker in front — the ``q:`` prefix
that ``tests/decode_gyroflow_project.py`` strips is a defensive leftover, not
something ``compress_to_base91`` emits. Do not add one here: a real encoder
output can start with ``q:`` by chance, and a reader that strips it would
corrupt the stream.
"""

from __future__ import annotations

import bisect
import struct
import zlib
from typing import Any

import cbor2

# Upstream's `MapClosest` stands in for "no such neighbour" with this key
# rather than an Option, so an absent side ends up at a distance of about
# 99999. It only matters when `max_diff` is larger than that, which none of
# upstream's callers use — but reproducing it keeps the two implementations
# identical rather than merely equivalent where they are called.
_MISSING_KEY = -99999


# Reference base91 alphabet (the Rust `base91` crate uses the same one).
_B91_ALPHABET = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    "!#$%&()*+,./:;<=>?@[]^_`{|}~\""
)
_B91_DECODE = {c: i for i, c in enumerate(_B91_ALPHABET)}


def base91_encode(data: bytes) -> str:
    """Encode *data* with the reference base91 packing."""
    out: list[str] = []
    bit_buffer = 0
    bit_count = 0
    for byte in data:
        bit_buffer |= byte << bit_count
        bit_count += 8
        if bit_count > 13:
            value = bit_buffer & 8191
            if value > 88:
                bit_buffer >>= 13
                bit_count -= 13
            else:
                value = bit_buffer & 16383
                bit_buffer >>= 14
                bit_count -= 14
            out.append(_B91_ALPHABET[value % 91])
            out.append(_B91_ALPHABET[value // 91])
    if bit_count:
        out.append(_B91_ALPHABET[bit_buffer % 91])
        if bit_count > 7 or bit_buffer > 90:
            out.append(_B91_ALPHABET[bit_buffer // 91])
    return "".join(out)


def base91_decode(text: str) -> bytes:
    """Decode base91 *text* back to bytes."""
    out = bytearray()
    bit_buffer = 0
    bit_count = 0
    value = -1
    for char in text:
        code = _B91_DECODE.get(char)
        if code is None:
            continue
        if value < 0:
            value = code
        else:
            value += code * 91
            bit_buffer |= value << bit_count
            bit_count += 13 if (value & 8191) > 88 else 14
            while bit_count >= 8:
                out.append(bit_buffer & 255)
                bit_buffer >>= 8
                bit_count -= 8
            value = -1
    if value >= 0:
        out.append((bit_buffer | value << bit_count) & 255)
    return bytes(out)


def compress_to_base91(data: bytes) -> str:
    """bincode-legacy payload -> zlib(best) -> base91, as util.rs does."""
    return base91_encode(zlib.compress(bytes(data), 9))


def decompress_from_base91(text: str) -> bytes:
    """Inverse of :func:`compress_to_base91`."""
    if not text:
        return b""
    return zlib.decompress(base91_decode(text))


def _quat_xyzw(value: Any) -> tuple[float, float, float, float]:
    """Accept a Quat64, a numpy array or a plain sequence of four."""
    getter = getattr(value, "quaternion", None)
    if getter is not None:
        w, x, y, z = getter()
        return (float(w), float(x), float(y), float(z))
    w, x, y, z = value
    return (float(w), float(x), float(y), float(z))


def encode_quat_map(quaternions: dict[int, Any]) -> bytes:
    """Bincode-legacy bytes for a timestamp -> quaternion map.

    Rust serializes ``BTreeMap<i64, Quat64>`` as a u64 entry count followed by
    each ``(i64, f64, f64, f64, f64)``. This is the layout ``.gyroflow`` files
    use for ``gyro_source.quaternions``; the ``integrated_quaternions`` cache
    uses CBOR instead (:func:`encode_cbor_quat_map`).

    Values may be ``Quat64`` instances or plain ``(w, x, y, z)`` sequences.
    """
    keys = sorted(quaternions)
    out = bytearray(struct.pack("<Q", len(keys)))
    for ts in keys:
        out += struct.pack("<qdddd", int(ts), *_quat_xyzw(quaternions[ts]))
    return bytes(out)


def decode_quat_map(data: bytes) -> dict[int, tuple[float, float, float, float]]:
    """Inverse of :func:`encode_quat_map`; values are (w, x, y, z).

    The length is checked against the header rather than trusted: a corrupt
    or truncated payload must raise instead of quietly yielding a short map.
    """
    (count,) = struct.unpack_from("<Q", data, 0)
    if len(data) != 8 + 40 * count:
        raise ValueError(
            f"quaternion payload is {len(data)} bytes, expected "
            f"{8 + 40 * count} for {count} entries"
        )
    result: dict[int, tuple[float, float, float, float]] = {}
    offset = 8
    for _ in range(count):
        ts, w, x, y, z = struct.unpack_from("<qdddd", data, offset)
        result[ts] = (w, x, y, z)
        offset += 40
    return result


def encode_f64_list(values: list[float]) -> bytes:
    """Bincode-legacy bytes for a ``Vec<f64>``."""
    out = bytearray(struct.pack("<Q", len(values)))
    for value in values:
        out += struct.pack("<d", float(value))
    return bytes(out)


def decode_f64_list(data: bytes) -> list[float]:
    """Inverse of :func:`encode_f64_list`."""
    (count,) = struct.unpack_from("<Q", data, 0)
    if len(data) != 8 + 8 * count:
        raise ValueError(
            f"f64 payload is {len(data)} bytes, expected {8 + 8 * count} "
            f"for {count} values"
        )
    return list(struct.unpack_from(f"<{count}d", data, 8))


def decode_vec3_map(data: bytes) -> dict[int, tuple[float, float, float]]:
    """``BTreeMap<i64, Vector3<f64>>``, the layout of ``gravity_vectors``.

    ``Vector3<f64>`` is a serde sequence of three, so unlike the quaternion
    map each entry is 8 + 24 bytes.
    """
    (count,) = struct.unpack_from("<Q", data, 0)
    if len(data) != 8 + 32 * count:
        raise ValueError(
            f"vec3 payload is {len(data)} bytes, expected {8 + 32 * count} "
            f"for {count} entries"
        )
    result: dict[int, tuple[float, float, float]] = {}
    offset = 8
    for _ in range(count):
        ts, x, y, z = struct.unpack_from("<qddd", data, offset)
        result[ts] = (x, y, z)
        offset += 32
    return result


def decode_imu_list(data: bytes) -> list[tuple[float, Any, Any, Any]]:
    """``Vec<TimeIMU>``, the layout of ``gyro_source.raw_imu``.

    ``TimeIMU`` is telemetry-parser's ``IMUData``: a ``f64`` timestamp in
    milliseconds and three ``Option<[f64; 3]>`` channels. Bincode legacy
    writes an ``Option`` as a one-byte tag, so the entries are variable
    length and have to be walked rather than sliced.

    Returns ``(timestamp_ms, gyro, accl, magn)`` tuples; each channel is a
    3-tuple or ``None``.
    """
    (count,) = struct.unpack_from("<Q", data, 0)
    offset = 8

    def read_vec3():
        nonlocal offset
        (present,) = struct.unpack_from("<B", data, offset)
        offset += 1
        if not present:
            return None
        values = struct.unpack_from("<3d", data, offset)
        offset += 24
        return values

    result = []
    for _ in range(count):
        (timestamp_ms,) = struct.unpack_from("<d", data, offset)
        offset += 8
        gyro = read_vec3()
        accl = read_vec3()
        magn = read_vec3()
        result.append((timestamp_ms, gyro, accl, magn))
    return result


# ----------------------------------------------------------------------
# CBOR payloads (the `WithProcessedData` caches)
# ----------------------------------------------------------------------


def _cbor_head(major: int, argument: int) -> bytes:
    """CBOR head for *major* type with the shortest legal argument."""
    if argument < 24:
        return bytes([(major << 5) | argument])
    if argument < 1 << 8:
        return bytes([(major << 5) | 24, argument])
    if argument < 1 << 16:
        return bytes([(major << 5) | 25]) + struct.pack(">H", argument)
    if argument < 1 << 32:
        return bytes([(major << 5) | 26]) + struct.pack(">I", argument)
    return bytes([(major << 5) | 27]) + struct.pack(">Q", argument)


def _cbor_int(value: int) -> bytes:
    value = int(value)
    return (
        _cbor_head(0, value)
        if value >= 0
        else _cbor_head(1, -1 - value)
    )


def _cbor_f64(value: float) -> bytes:
    """CBOR float in the shortest form that holds *value* exactly.

    ``ciborium`` does not always write 8-byte floats: a value representable
    exactly in half or single precision is written in that, 7 or 4 bytes
    shorter. This is not cosmetic — it is the difference between a file
    byte-identical to Gyroflow's and one that merely decodes the same.

    The preference order is half *before* single, which is counter-intuitive
    and was read off a real file rather than assumed: real Gyroflow 1.6.3
    exports contain values like 66.5 and 2738.0 written as ``0xf9`` halves,
    even though they are also single-exact. Verified against every float in
    that project — 753/753 and 25185/25185 for the two timestamp lists
    reproduce, while the reverse order gets 9 of them wrong.

    NaN falls through to the 8-byte form: the round-trip comparison that
    decides "exact" is false for NaN, so it is never shortened. No other
    value goes the wrong way, since a failed narrowing raises or compares
    unequal and lands on the next width.
    """
    value = float(value)
    for prefix, code in ((b"\xf9", ">e"), (b"\xfa", ">f")):
        try:
            packed = struct.pack(code, value)
        except (OverflowError, struct.error):
            continue
        if struct.unpack(code, packed)[0] == value:
            return prefix + packed
    return b"\xfb" + struct.pack(">d", value)


def encode_cbor_quat_map(quaternions: dict[int, Any]) -> bytes:
    """CBOR ``BTreeMap<i64, Quat64>``, as ``integrated_quaternions`` stores it.

    ``Quat64`` is nalgebra's ``UnitQuaternion<f64>``, which serializes through
    ``Unit`` -> ``Quaternion`` -> ``Vector4`` as a plain 4-element array
    (nalgebra base/unit.rs, geometry/quaternion.rs). So each value is a CBOR
    array of four floats, and the whole thing is a map.
    """
    keys = sorted(quaternions)
    out = bytearray(_cbor_head(5, len(keys)))
    for ts in keys:
        out += _cbor_int(ts)
        out += _cbor_head(4, 4)
        for component in _quat_xyzw(quaternions[ts]):
            out += _cbor_f64(component)
    return bytes(out)


def decode_cbor_quat_map(data: bytes) -> dict[int, tuple[float, float, float, float]]:
    """Inverse of :func:`encode_cbor_quat_map`; values are (w, x, y, z).

    Decoding goes through cbor2, which accepts any float width — so this
    reads a file Gyroflow wrote even though :func:`_cbor_f64` on the write
    side always picks the 8-byte form.
    """
    loaded = cbor2.loads(data)
    if not isinstance(loaded, dict):
        raise ValueError(f"expected a CBOR map, got {type(loaded).__name__}")
    return {
        int(ts): (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        for ts, q in loaded.items()
    }


def encode_cbor_f64_list(values: list[float]) -> bytes:
    """CBOR ``Vec<f64>``, as ``adaptive_zoom_fovs`` and friends store it."""
    out = bytearray(_cbor_head(4, len(values)))
    for value in values:
        out += _cbor_f64(value)
    return bytes(out)


def decode_cbor_f64_list(data: bytes) -> list[float]:
    """Inverse of :func:`encode_cbor_f64_list`."""
    loaded = cbor2.loads(data)
    if not isinstance(loaded, list):
        raise ValueError(f"expected a CBOR array, got {type(loaded).__name__}")
    return [float(v) for v in loaded]


def encode_cbor_optional_f64_list(values: list[float | None]) -> bytes:
    """CBOR ``Vec<Option<f64>>``, as the focal length curves store it.

    A frame whose focal length the camera never reported is a ``null`` in the
    array, not a zero — the two are different things here, and a curve padded
    with zeros would divide the compensation ratio by zero-length optics.
    """
    out = bytearray(_cbor_head(4, len(values)))
    for value in values:
        out += b"\xf6" if value is None else _cbor_f64(float(value))
    return bytes(out)


def decode_cbor_optional_f64_list(data: bytes) -> list[float | None]:
    """Inverse of :func:`encode_cbor_optional_f64_list`."""
    loaded = cbor2.loads(data)
    if not isinstance(loaded, list):
        raise ValueError(f"expected a CBOR array, got {type(loaded).__name__}")
    return [None if v is None else float(v) for v in loaded]


def compress_to_base91_cbor(data: bytes) -> str:
    """CBOR bytes -> zlib(best) -> base91 (util.rs:compress_to_base91_cbor).

    Only the payload differs from :func:`compress_to_base91`; the envelope is
    the same zlib + base91 pair.
    """
    return compress_to_base91(data)


def decompress_from_base91_cbor(text: str) -> Any:
    """Inverse of :func:`compress_to_base91_cbor`.

    Returns whatever the CBOR holds; shaping it is the caller's job.
    """
    return cbor2.loads(decompress_from_base91(text))


class ClosestMap:
    """Nearest-key lookup with a distance cap (Rust's ``MapClosest``).

    Port of ``util.rs::MapClosest``. Its semantics are fussier than "nearest
    key", and the differences decide real behaviour where it is used — the
    per-frame lens data, where a wrong answer is a wrong calibration:

    * An exact key wins outright.
    * Otherwise the **strictly** closer neighbour wins. Two neighbours exactly
      equidistant from the key give **None**: upstream drops that lookup and
      the caller falls back to the static value, rather than picking a side.
    * The cap is strict too — a neighbour exactly ``max_diff`` away does not
      count.
    * A missing neighbour on one side is at the sentinel distance above, not
      at infinity, which is a difference only for caps above ~100000.

    The keys are sorted once on construction. The callers walk a whole clip,
    so this is one lookup per frame against a map that also has roughly one
    entry per frame; re-sorting per call would be quadratic.
    """

    def __init__(self, mapping: dict[int, Any] | None = None) -> None:
        self._mapping = mapping if mapping is not None else {}
        self._keys = sorted(self._mapping)

    def __len__(self) -> int:
        return len(self._keys)

    def __bool__(self) -> bool:
        return bool(self._keys)

    def get_closest(self, key: int, max_diff: int) -> Any | None:
        """Value nearest *key* within *max_diff*, or None."""
        if not self._keys:
            return None
        if key in self._mapping:
            return self._mapping[key]

        index = bisect.bisect_left(self._keys, key)
        below = self._keys[index - 1] if index > 0 else None
        above = self._keys[index] if index < len(self._keys) else None

        above_diff = abs(key - (above if above is not None else _MISSING_KEY))
        below_diff = abs(key - (below if below is not None else _MISSING_KEY))

        if above is not None and above_diff < max_diff and above_diff < below_diff:
            return self._mapping[above]
        if below is not None and below_diff < max_diff and below_diff < above_diff:
            return self._mapping[below]
        return None


def timestamp_at_frame(frame: int, fps: float) -> float:
    """Convert frame index to timestamp in milliseconds.

    Args:
        frame: Zero-based frame index.
        fps: Video frame rate.

    Returns:
        Timestamp in milliseconds.
    """
    return frame * 1000.0 / fps if fps > 0 else 0.0


def frame_at_timestamp(timestamp_ms: float, fps: float) -> int:
    """Convert timestamp to frame index.

    Args:
        timestamp_ms: Timestamp in milliseconds.
        fps: Video frame rate.

    Returns:
        Zero-based frame index (rounded).
    """
    return round(timestamp_ms * fps / 1000.0) if fps > 0 else 0
