"""Camera identifier — Sony distortion-hash serialization.

The hash is a CRC32 over a JSON string built by ``serde_json`` upstream
(``camera_identifier.rs``). It is the key Sony lens profiles are looked up
by, so any byte of difference means the profile never auto-loads.

Two things diverge by default:

* ``serde_json`` serializes compactly (``{"a":1}``); Python's ``json.dumps``
  inserts ``", "`` / ``": "``. That alone changed every hash.
* the numbers keep their Rust types — ``coeff_scale`` is an f32 widened to
  f64 (0.001 becomes 0.0010000000474974513), and the rest are integers.

The expected values below come from a Rust reference program built with the
same crates (``serde_json`` with ``preserve_order`` + ``crc32fast``).
"""

import json
import zlib

import pytest

from pygyroflow.camera.identifier import _sony_distortion_hash, _widen_f32

# Rust: {focal_length_nm: 24_000_000u32, effective_sensor_height_nm: 13_000_000u32,
#        unk1: 1u8, coeff_scale: 0.001f32, coeffs: [100,200,300]u16}
# serde_json::json!({...}).to_string() ==
#   {"unk1":[24000000,13000000],"unk2":1,"unk3":0.0010000000474974513,"unk4":[100,200,300]}
# crc32 == 906280fe
RUST_VECTOR = {
    "focal_length_nm": 24_000_000,
    "effective_sensor_height_nm": 13_000_000,
    "unk1": 1,
    "coeff_scale": 0.001,
    "coeffs": [100, 200, 300],
}
RUST_CRC = "906280fe"


class TestSonyDistortionHash:
    def test_matches_the_rust_reference(self):
        assert _sony_distortion_hash(RUST_VECTOR) == RUST_CRC

    def test_is_lowercase_hex_without_padding(self):
        digest = _sony_distortion_hash(RUST_VECTOR)
        assert digest == digest.lower()
        assert len(digest) <= 8
        assert digest.strip("0123456789abcdef") == ""

    def test_whitespace_would_change_the_hash(self):
        """The old `json.dumps` default separators broke every lookup."""
        payload = {
            "unk1": [24_000_000, 13_000_000],
            "unk2": 1,
            "unk3": _widen_f32(0.001),
            "unk4": [100, 200, 300],
        }
        spaced = json.dumps(payload)  # ", " and ": " — Python's default
        compact = json.dumps(payload, separators=(",", ":"))
        assert spaced != compact
        crc = lambda s: format(zlib.crc32(s.encode()) & 0xFFFFFFFF, "x")  # noqa: E731
        assert crc(spaced) != RUST_CRC
        assert crc(compact) == RUST_CRC

    def test_float32_widening_matters(self):
        """The hashed string carries the f32-widened double, not a plain 0.001.

        `_sony_distortion_hash` widens internally, so a naive 0.001 still
        hashes correctly; this pins down that the widening is what the CRC
        depends on, by hashing the un-widened string directly.
        """
        def build(scale):
            return json.dumps(
                {
                    "unk1": [24_000_000, 13_000_000],
                    "unk2": 1,
                    "unk3": scale,
                    "unk4": [100, 200, 300],
                },
                separators=(",", ":"),
            )

        crc = lambda s: format(  # noqa: E731
            zlib.crc32(s.encode()) & 0xFFFFFFFF, "x"
        )
        assert build(0.001) != build(_widen_f32(0.001))
        assert crc(build(0.001)) != RUST_CRC
        assert crc(build(_widen_f32(0.001))) == RUST_CRC

    def test_integer_inputs_are_tolerated(self):
        """Parsed telemetry may hand these over as floats."""
        as_floats = {
            "focal_length_nm": 24_000_000.0,
            "effective_sensor_height_nm": 13_000_000.0,
            "unk1": 1.0,
            "coeff_scale": 0.001,
            "coeffs": [100.0, 200.0, 300.0],
        }
        assert _sony_distortion_hash(as_floats) == RUST_CRC

    def test_different_geometry_gives_a_different_hash(self):
        other = dict(RUST_VECTOR, focal_length_nm=35_000_000)
        assert _sony_distortion_hash(other) != RUST_CRC

    def test_is_deterministic(self):
        assert _sony_distortion_hash(RUST_VECTOR) == _sony_distortion_hash(RUST_VECTOR)


class TestWidenF32:
    @pytest.mark.parametrize("value", [0.001, 1.0, 0.5, 3.14, -2.5])
    def test_round_trip_is_idempotent(self, value):
        once = _widen_f32(value)
        assert _widen_f32(once) == once

    def test_exact_values_survive(self):
        assert _widen_f32(0.5) == 0.5
        assert _widen_f32(2.0) == 2.0

    def test_inexact_values_widen(self):
        assert _widen_f32(0.001) == pytest.approx(0.001, rel=1e-6)
        assert _widen_f32(0.001) != 0.001


class TestSonyExtractionUsesTheHash:
    """End-to-end through the Sony extractor."""

    @staticmethod
    def _identify(samples):
        from pygyroflow.camera.identifier import CameraIdentifier

        return CameraIdentifier.from_metadata(
            brand="Sony", model="ILCE-7M4", samples=samples
        )

    def test_lens_info_falls_back_to_the_distortion_hash(self):
        samples = [
            {"tag_map": {"LensDistortion": {"Data": dict(RUST_VECTOR)}}}
        ]
        assert self._identify(samples).lens_info == RUST_CRC

    def test_focal_length_wins_over_the_hash(self):
        samples = [
            {
                "tag_map": {
                    "Lens": {"FocalLength": 24.0},
                    "LensDistortion": {"Data": dict(RUST_VECTOR)},
                }
            }
        ]
        info = self._identify(samples).lens_info
        assert info != RUST_CRC
        assert "24" in info
