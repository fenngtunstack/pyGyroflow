"""Tests for StabilizationManager."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pygyroflow.manager import StabilizationManager, InputFile


class TestStabilizationManagerInit:
    def test_initialization(self):
        mgr = StabilizationManager()
        assert mgr.gyro is not None
        assert mgr.lens is not None
        assert mgr.smoothing is not None
        assert mgr.keyframes is not None
        assert mgr.lens_db is not None
        assert mgr.input_file is not None

    def test_default_parameters(self):
        mgr = StabilizationManager()
        assert mgr.gyro.duration_ms == 0.0
        assert len(mgr.gyro.quaternions) == 0


class TestStabilizationManagerInitFromVideo:
    def test_init_from_video_data(self):
        mgr = StabilizationManager()
        mgr.init_from_video_data(
            duration_ms=10000.0,
            fps=30.0,
            frame_count=300,
            video_size=(1920, 1080),
        )
        assert mgr.params.fps == 30.0
        assert mgr.params.frame_count == 300
        assert mgr.params.duration_ms == 10000.0
        assert mgr.params.size == (1920, 1080)

    def test_short_video_uses_complementary(self):
        mgr = StabilizationManager()
        mgr.init_from_video_data(
            duration_ms=5000.0,  # < 10s
            fps=30.0,
            frame_count=150,
            video_size=(1920, 1080),
        )
        assert mgr.gyro.integration_method == 1  # Complementary

    def test_long_video_keeps_default(self):
        mgr = StabilizationManager()
        mgr.init_from_video_data(
            duration_ms=20000.0,  # > 10s
            fps=30.0,
            frame_count=600,
            video_size=(1920, 1080),
        )
        assert mgr.gyro.integration_method == 2  # VQF default


class TestInputFile:
    def test_default(self):
        f = InputFile()
        assert f.url == ""
        assert f.project_file_url is None
        assert f.image_sequence_fps == 0.0
