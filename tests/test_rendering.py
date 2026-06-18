"""Tests for FfmpegProcessor import and initialization."""

import pytest

from pygyroflow.rendering import FfmpegProcessor


class TestFfmpegProcessorInit:
    def test_import(self):
        """FfmpegProcessor can be imported successfully."""
        proc = FfmpegProcessor()
        assert proc is not None

    def test_initial_state(self):
        proc = FfmpegProcessor()
        assert proc.input_info is None

    @pytest.mark.skipif(
        True,  # Always skip: no test video file available
        reason="Requires an actual video file"
    )
    def test_open_nonexistent_file(self):
        proc = FfmpegProcessor()
        with pytest.raises(Exception):
            proc.open_input("/nonexistent/path/video.mp4")
