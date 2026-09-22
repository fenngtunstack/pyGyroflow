"""Lens-profile checksums on load (C-17 piece)."""

from __future__ import annotations

import json
import zlib

from pygyroflow.lens.database import LensProfileDatabase, _assign_profile_checksum
from pygyroflow.lens.profile import LensProfile


def _profile(**over):
    data = {
        "name": "t",
        "identifier": "TestCam 4k",
        "calib_dimension": {"w": 1920, "h": 1080},
        "fisheye_params": {
            "camera_matrix": [[1000.0, 0.0, 960.0],
                              [0.0, 1000.5, 540.0]],
            "distortion_coeffs": [-0.1, 0.02, 0.0, 0.0],
        },
    }
    data.update(over)
    return LensProfile.from_json(data)


class TestJsonProfileChecksum:
    def test_the_format_string_matches_upstream(self):
        """``{identifier}|{w}{h}|{fx:.8}{fy:.8}|{cx:.8}{cy:.8}|{4x:.8}``
        hashed with crc32, hex-encoded 8 wide (rs:112-130)."""
        prof = _profile()
        _assign_profile_checksum(prof)
        expected_payload = (
            "TestCam 4k|19201080"
            "|1000.000000001000.50000000"
            "|960.00000000540.00000000"
            "|-0.100000000.020000000.000000000.00000000"
        )
        want = format(
            zlib.crc32(expected_payload.encode()) & 0xFFFFFFFF, "08x"
        )
        assert prof.checksum == want

    def test_eight_decimal_formatting(self):
        """Rust ``{:.8}`` keeps trailing zeros; Python ``:.8f`` matches."""
        prof = _profile(fisheye_params={
            "camera_matrix": [[1.5, 0.0, 2.25], [0.0, 1.5, 3.75]],
            "distortion_coeffs": [0.0, 0.0],
        })
        _assign_profile_checksum(prof)
        payload_hashed_again = (
            "TestCam 4k|19201080"
            "|1.500000001.50000000"
            "|2.250000003.75000000"
            "|0.000000000.000000000.000000000.00000000"
        )
        want = format(
            zlib.crc32(payload_hashed_again.encode()) & 0xFFFFFFFF, "08x"
        )
        assert prof.checksum == want

    def test_missing_coefficients_default_to_zero(self):
        """The helper pads to four; a 1-coeff profile and a 3-coeff profile
        that agree on the shared prefix hash identically."""
        one = _profile(fisheye_params={
            "camera_matrix": [[10.0, 0.0, 1.0], [0.0, 10.0, 2.0]],
            "distortion_coeffs": [-0.1],
        })
        _assign_profile_checksum(one)
        three = _profile(fisheye_params={
            "camera_matrix": [[10.0, 0.0, 1.0], [0.0, 10.0, 2.0]],
            "distortion_coeffs": [-0.1, 0.0, 0.0],
        })
        _assign_profile_checksum(three)
        assert one.checksum == three.checksum

    def test_no_matrix_leaves_checksum_unset(self):
        prof = _profile(fisheye_params={})
        _assign_profile_checksum(prof)
        assert prof.checksum is None


class TestGyroflowPresetChecksum:
    def test_is_crc32_of_the_path(self):
        db = LensProfileDatabase()
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            preset = os.path.join(tmp, "my-preset.gyroflow")
            with open(preset, "w") as fh:
                fh.write("{}")
            db.load_from_directory(tmp)
        assert len(db) == 1
        prof = db.profiles[0][1]
        assert prof.checksum == format(
            zlib.crc32(preset.encode()) & 0xFFFFFFFF, "08x"
        )


class TestSearchUsesChecksum:
    def test_favourite_boost_matches_by_checksum(self, tmp_path, monkeypatch):
        db = LensProfileDatabase()
        import os
        import tempfile

        data = {
            "name": "Favcam Lens",
            "camera_brand": "Favcam",
            "camera_model": "X100",
            "calib_dimension": {"w": 1920, "h": 1080},
            "fisheye_params": {
                "camera_matrix": [[500.0, 0.0, 960.0], [0.0, 500.0, 540.0]],
                "distortion_coeffs": [0.0],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "favcam.json")
            with open(path, "w") as fh:
                json.dump(data, fh)
            db.load_from_directory(tmp)

        prof = db.profiles[0][1]
        assert prof.checksum is not None
        results = db.search("Favcam", favorites={prof.checksum})
        assert results and results[0] is prof
