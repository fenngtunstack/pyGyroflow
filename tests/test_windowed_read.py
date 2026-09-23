"""A-04: size-tiered detection windows and the mmap-backed parse view."""

from __future__ import annotations

import mmap
import os

import pytest

from pygyroflow.telemetry.parser import (
    _detect_buffer,
    _detect_window_for,
    _read_all,
)

GIB = 1024 * 1024 * 1024
MIB = 1024 * 1024


class TestTieredWindow:
    @pytest.mark.parametrize(
        "size_gib,expected_mib",
        [(1, 5), (5, 5), (6, 50), (31, 180), (61, 220), (101, 500)],
    )
    def test_the_tier_table(self, size_gib, expected_mib):
        # Verbatim from lib.rs:73-84 (boundaries are exclusive: >5 GiB etc).
        assert _detect_window_for(size_gib * GIB) == expected_mib * MIB

    @pytest.mark.parametrize(
        "size_gib,expected_mib",
        [(6, 50), (31, 180), (61, 220), (101, 500)],
    )
    def test_buffer_length_on_sparse_files(self, tmp_path, size_gib, expected_mib):
        """The tiers are exercised against real (sparse) file sizes, not by
        mocking getsize: truncate() creates the size without the data."""
        path = tmp_path / "big.mp4"
        with open(path, "wb") as f:
            f.truncate(size_gib * GIB)
        assert len(_detect_buffer(str(path))) == 2 * expected_mib * MIB

    def test_head_and_tail_content_survive_the_window(self, tmp_path):
        path = tmp_path / "big.mp4"
        with open(path, "wb") as f:
            f.write(b"HEADMARK")
            f.truncate(6 * GIB)
            f.seek(-8, os.SEEK_END)
            f.write(b"TAILMARK")
        buf = _detect_buffer(str(path))
        assert buf[:8] == b"HEADMARK"
        assert buf[-8:] == b"TAILMARK"

    def test_small_file_is_read_whole(self, tmp_path):
        path = tmp_path / "s.mp4"
        path.write_bytes(b"whole-thing")
        assert _detect_buffer(str(path)) == b"whole-thing"


class TestMmapParseView:
    def test_no_whole_file_copy(self, tmp_path):
        path = tmp_path / "f.bin"
        path.write_bytes(b"x" * 4096)
        assert isinstance(_read_all(str(path)), mmap.mmap)

    def test_empty_file_degrades_to_empty_bytes(self, tmp_path):
        path = tmp_path / "e.bin"
        path.write_bytes(b"")
        assert _read_all(str(path)) == b""

    def test_the_view_supports_the_parser_contract(self, tmp_path):
        """Everything the four parsers do to the full-file buffer: slice,
        find, len, int-index, negative slice."""
        path = tmp_path / "f.bin"
        path.write_bytes(b"ABCD" + b"\x00" * 100 + b"SROT" + b"tail")
        view = _read_all(str(path))
        assert view[:4] == b"ABCD"
        assert view.find(b"SROT") == 104
        assert len(view) == 112
        assert view[0] == 0x41
        assert view[-4:] == b"tail"

    def test_parse_telemetry_file_runs_on_the_view(self, tmp_path):
        """End to end: a non-video mp4 goes through detection (which now
        mmaps the full-file Sony fallback) and comes back Unknown instead
        of loading the file."""
        path = tmp_path / "clip.mp4"
        path.write_bytes(b"\x00" * (10 * MIB))
        from pygyroflow.telemetry.parser import detect_telemetry_format

        assert detect_telemetry_format(str(path)) == "Unknown"
