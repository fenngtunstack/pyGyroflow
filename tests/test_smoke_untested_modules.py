"""Smoke tests for modules that previously had zero test coverage.

These verify import-ability, construction, and basic API surface — not full
correctness. Goal: catch broken imports / vanished symbols early (several of
these modules were silently broken, e.g. stmap imported a non-existent
``_rotate_and_distort``).
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# synchronization
# ---------------------------------------------------------------------------

class TestSynchronizationSmoke:
    def test_public_api_imports(self):
        from pygyroflow.synchronization import (
            find_offset_rs_sync, find_offset_visual_features,
            PoseEstimator, AutosyncProcess, OptimSync,
            create_detector, OF_METHOD_MAP,
        )
        assert callable(find_offset_rs_sync)
        assert callable(find_offset_visual_features)
        assert isinstance(OF_METHOD_MAP, dict)

    def test_find_offset_rs_sync_empty_returns_none(self):
        from pygyroflow.synchronization import find_offset_rs_sync
        # Empty inputs should return None gracefully (no crash).
        result = find_offset_rs_sync([], [])
        assert result is None

    def test_detector_creation(self):
        from pygyroflow.synchronization import create_detector, OF_METHOD_MAP
        for name in OF_METHOD_MAP:
            det = create_detector(name)
            assert det is not None


# ---------------------------------------------------------------------------
# telemetry
# ---------------------------------------------------------------------------

class TestTelemetrySmoke:
    def test_parse_telemetry_file_missing_raises(self):
        from pygyroflow.telemetry import parse_telemetry_file
        from pygyroflow.types.errors import TelemetryParseError
        with pytest.raises(TelemetryParseError):
            parse_telemetry_file("/nonexistent/path.mp4")

    def test_parse_telemetry_file_bad_extension(self, tmp_path):
        from pygyroflow.telemetry import parse_telemetry_file
        from pygyroflow.types.errors import TelemetryParseError
        bad = tmp_path / "data.txt"
        bad.write_bytes(b"\x00")
        with pytest.raises(TelemetryParseError):
            parse_telemetry_file(str(bad))


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

class TestCalibrationSmoke:
    def test_lens_calibrator_constructs(self):
        from pygyroflow.calibration import LensCalibrator
        cal = LensCalibrator()
        assert cal is not None


# ---------------------------------------------------------------------------
# camera
# ---------------------------------------------------------------------------

class TestCameraSmoke:
    def test_camera_identifier_constructs(self):
        from pygyroflow.camera import CameraIdentifier
        ci = CameraIdentifier()
        assert ci is not None


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

class TestCliSmoke:
    def test_cli_main_importable(self):
        from pygyroflow.cli.main import main
        assert callable(main)


# ---------------------------------------------------------------------------
# stmap — KNOWN BROKEN IMPORT (tracked)
# ---------------------------------------------------------------------------

class TestStmapSmoke:
    @pytest.mark.xfail(
        reason="stmap/exporter.py imports _rotate_and_distort from "
        "cpu_undistort, but that function was renamed to "
        "_vectorized_rotate_distort during a refactor; the scalar version "
        "stmap expects no longer exists. Module is unimportable until fixed.",
        strict=True,
        raises=ImportError,
    )
    def test_stmap_imports(self):
        # Should succeed once the broken import is fixed.
        from pygyroflow.stmap import STMapExporter, STMapFormat  # noqa: F401
