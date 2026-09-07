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


class TestLensAutoMatchRanking:
    """_lens_match_penalty ordering for the tagless-file fallback path."""

    @staticmethod
    def _profile(lens_model: str = "", note: str = "", calib=(0, 0), fps=0.0):
        from pygyroflow.lens import LensProfile
        p = LensProfile()
        p.camera_brand = "GoPro"
        p.camera_model = "HERO6 Black"
        p.lens_model = lens_model
        p.note = note
        p.calib_dimension = {"w": calib[0], "h": calib[1]}
        p.fps = fps
        return p

    def test_aspect_beats_alphabetical(self):
        # 8:7 footage must never pick a 4:3 profile, whatever its name
        p_43 = self._profile("Wide", "NO-EIS", calib=(4000, 3000), fps=29.97)
        p_87 = self._profile("Wide", "NO-EIS", calib=(3840, 3360), fps=29.97)
        key = StabilizationManager._lens_match_penalty
        assert key(p_87, 1280, 1120, 29.97) < key(p_43, 1280, 1120, 29.97)

    def test_exact_size_then_fps_then_default_fov(self):
        key = StabilizationManager._lens_match_penalty
        w, h, fps = 2704, 2028, 29.97
        exact = self._profile("Wide", "NO-EIS", calib=(2704, 2028), fps=29.97)
        wrong_fps = self._profile("Wide", "NO-EIS", calib=(2704, 2028), fps=23.98)
        wrong_size = self._profile("Wide", "NO-EIS", calib=(1920, 1440), fps=29.97)
        linear = self._profile("Linear", "NO-EIS", calib=(2704, 2028), fps=29.97)
        eis = self._profile("Wide", "EIS-Y", calib=(2704, 2028), fps=29.97)
        # exact size + fps + Wide + NO-EIS wins over every other variant
        for other in (wrong_fps, wrong_size, linear, eis):
            assert key(exact, w, h, fps) < key(other, w, h, fps)

    def test_apply_lens_readout_time(self):
        from pygyroflow.lens import LensProfile
        from pygyroflow.types.enums import ReadoutDirection
        mgr = StabilizationManager()
        assert mgr.params.frame_readout_time == 0.0
        mgr.lens = LensProfile()
        mgr.lens.frame_readout_time = 11.1111
        mgr._apply_lens_readout_time()
        assert mgr.params.frame_readout_time == pytest.approx(11.1111)
        assert mgr.params.frame_readout_direction == ReadoutDirection.TopToBottom
        # direction stays a ReadoutDirection (str values crash frame_transform)
        assert isinstance(mgr.params.frame_readout_direction, ReadoutDirection)
        mgr.lens.frame_readout_time = -15.0
        mgr._apply_lens_readout_time()
        assert mgr.params.frame_readout_time == pytest.approx(15.0)
        assert mgr.params.frame_readout_direction == ReadoutDirection.BottomToTop
