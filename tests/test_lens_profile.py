"""Tests for LensProfile and LensProfileDatabase."""

import json
import os
import tempfile

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pygyroflow.lens import LensProfile, LensProfileDatabase


class TestLensProfile:
    def test_default_creation(self):
        p = LensProfile()
        assert p.name == ""
        assert p.camera_brand == ""
        assert len(p.camera_matrix) == 0
        assert len(p.distortion_coeffs) == 0

    def test_from_json_basic(self):
        data = {
            "name": "Test Camera",
            "camera_brand": "TestBrand",
            "camera_model": "TestModel",
            "calib_dimension": {"w": 1920, "h": 1080},
            "fps": 30.0,
            "camera_matrix": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]],
            "distortion_coeffs": [-0.1, 0.02, 0.001, 0.0],
            "distortion_model": "opencv_fisheye",
            "global_shutter": True,
        }
        p = LensProfile.from_json(data)
        assert p.name == "Test Camera"
        assert p.camera_brand == "TestBrand"
        assert p.calib_dimension == {"w": 1920, "h": 1080}
        assert p.fps == 30.0
        assert len(p.camera_matrix) == 3
        assert p.distortion_coeffs[0] == -0.1
        assert p.global_shutter is True

    def test_get_camera_matrix(self):
        p = LensProfile()
        p.calib_dimension = {"w": 1920, "h": 1080}
        p.camera_matrix = [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]]
        mat = p.get_camera_matrix()
        assert mat.shape == (3, 3)
        assert_allclose(mat[0, 0], 1000.0)

    def test_get_camera_matrix_default(self):
        p = LensProfile()
        p.calib_dimension = {"w": 1920, "h": 1080}
        mat = p.get_camera_matrix()
        assert mat.shape == (3, 3)
        assert mat[0, 0] > 0  # Should have a valid focal length

    def test_get_distortion_coeffs_padded(self):
        p = LensProfile()
        p.distortion_coeffs = [-0.1, 0.02]
        coeffs = p.get_distortion_coeffs()
        assert len(coeffs) == 12
        assert coeffs[0] == -0.1
        assert coeffs[1] == 0.02
        assert coeffs[2] == 0.0

    def test_get_aspect_ratio(self):
        p = LensProfile()
        p.calib_dimension = {"w": 1920, "h": 1080}
        assert p.get_aspect_ratio() == "16:9"

    def test_get_size_str(self):
        p = LensProfile()
        p.calib_dimension = {"w": 1920, "h": 1080}
        assert p.get_size_str() == "1080p"

    def test_swapped(self):
        p = LensProfile()
        p.calib_dimension = {"w": 1920, "h": 1080}
        p.orig_dimension = {"w": 1920, "h": 1080}
        p.camera_matrix = [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]]

        swapped = p.swapped()
        assert swapped.calib_dimension["w"] == 1080
        assert swapped.calib_dimension["h"] == 1920

    def test_from_json_with_fisheye_params(self):
        data = {
            "name": "Fisheye Test",
            "fisheye_params": {
                "camera_matrix": [[800, 0, 640], [0, 800, 360], [0, 0, 1]],
                "distortion_coeffs": [-0.3, 0.1, 0.0, 0.0],
                "RMS_error": 0.5,
            },
        }
        p = LensProfile.from_json(data)
        assert p.camera_matrix[0][0] == 800
        assert p.distortion_coeffs[0] == -0.3
        assert p.rms_error == 0.5


class TestLensProfileDatabase:
    def test_empty_database(self):
        db = LensProfileDatabase()
        assert len(db) == 0
        assert not db.loaded

    def test_empty_search_returns_empty(self):
        db = LensProfileDatabase()
        results = db.search("GoPro")
        assert results == []

    def test_load_from_json_file(self):
        profile_data = {
            "name": "Test Profile",
            "camera_brand": "TestBrand",
            "camera_model": "TestModel",
            "calib_dimension": {"w": 1920, "h": 1080},
            "fps": 30.0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_profile.json")
            with open(filepath, "w") as f:
                json.dump(profile_data, f)

            db = LensProfileDatabase()
            db.load_from_directory(tmpdir)

            assert len(db) > 0
            assert db.loaded

    def test_get_by_name(self):
        profile_data = {
            "name": "UniqueTestName",
            "calib_dimension": {"w": 1920, "h": 1080},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.json")
            with open(filepath, "w") as f:
                json.dump(profile_data, f)

            db = LensProfileDatabase()
            db.load_from_directory(tmpdir)

            result = db.get_by_name("UniqueTestName")
            assert result is not None
            assert result.name == "UniqueTestName"

    def test_search_finds_match(self):
        profile_data = {
            "name": "TestBrand TestModel 1080p",
            "camera_brand": "TestBrand",
            "camera_model": "TestModel",
            "calib_dimension": {"w": 1920, "h": 1080},
            "calibrated_by": "tester",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.json")
            with open(filepath, "w") as f:
                json.dump(profile_data, f)

            db = LensProfileDatabase()
            db.load_from_directory(tmpdir)

            results = db.search("TestBrand")
            assert len(results) > 0
