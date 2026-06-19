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
# stmap — was broken import (fixed: switched to _vectorized_rotate_distort)
# ---------------------------------------------------------------------------

class TestStmapSmoke:
    def test_stmap_imports(self):
        from pygyroflow.stmap import STMapExporter, STMapFormat
        assert STMapExporter is not None
        assert {f.value for f in STMapFormat} == {"exr", "npz", "png16"}

    def test_stmap_undistort_map(self):
        """STMapExporter produces a finite (H, W, 2) undistort map.

        Previously this module was unimportable (broken _rotate_and_distort
        reference); the vectorized rewrite must still produce valid output.
        """
        import warnings

        import numpy as np

        from pygyroflow.stmap import STMapExporter
        from pygyroflow.stabilization import ComputeParams
        from pygyroflow.types.quaternion import Quat64

        q = Quat64.from_euler_angles(0.0, 0.0, 0.0)
        cp = ComputeParams(
            width=16, height=16, output_width=16, output_height=16,
            frame_count=1, scaled_fps=30.0, scaled_duration_ms=1000.0,
            quaternions={0: q}, smoothed_quaternions={0: q},
            fovs=[1.0], fov_scale=1.0,
            camera_matrix=np.array([[8.0, 0, 8], [0, 8, 8], [0, 0, 1]]),
            distortion_coeffs=[0.0] * 12, frame_readout_time=0.0,
        )
        exp = STMapExporter(cp)
        # cpu_undistort emits a benign divide-by-zero RuntimeWarning at r=0;
        # the result is masked correctly, suppress to keep test output clean.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = exp.compute_undistort_map(timestamp_ms=0.0, frame=0)
        assert m.shape == (16, 16, 2)
        assert m.dtype == np.float32
        assert np.isfinite(m).all()
