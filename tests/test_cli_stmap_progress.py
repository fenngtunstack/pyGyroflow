"""CLI `--export-stmap` and `--stdout-progress` (C-11 pieces)."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

av = pytest.importorskip("av", reason="av required for CLI media tests")


def _tiny_clip(path, width=64, height=48, frames=10):
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=30)
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    for i in range(frames):
        img = np.zeros((height, width, 3), np.uint8)
        img[:, :, 0] = (i * 23) % 256
        img[height // 4: height // 2, width // 4: width // 2, 1] = 200
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        frame.pts = i
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    return str(path)


class TestExportStmap:
    def test_mode_one_writes_only_undistort(self, tmp_path):
        from pygyroflow.cli.main import main

        clip = _tiny_clip(tmp_path / "in.mp4")
        out = tmp_path / "out.mp4"
        main(["--no-autosync", "--export-stmap", "1", "-o", str(out), clip])
        assert (tmp_path / "out-undistort.exr").exists()
        assert not (tmp_path / "out-redistort.exr").exists()
        assert not out.exists()  # nothing rendered

    def test_mode_two_writes_both(self, tmp_path):
        from pygyroflow.cli.main import main

        clip = _tiny_clip(tmp_path / "in.mp4")
        out = tmp_path / "out.mp4"
        main(["--no-autosync", "--export-stmap", "2", "-o", str(out), clip])
        assert (tmp_path / "out-undistort.exr").exists()
        assert (tmp_path / "out-redistort.exr").exists()

    def test_output_refuses_to_overwrite(self, tmp_path):
        from pygyroflow.cli.main import main

        clip = _tiny_clip(tmp_path / "in.mp4")
        out = tmp_path / "out.mp4"
        existing = tmp_path / "out-undistort.exr"
        existing.write_bytes(b"precious")
        with pytest.raises(SystemExit):
            main(["--no-autosync", "--export-stmap", "1",
                  "-o", str(out), clip])
        assert existing.read_bytes() == b"precious"


class TestStdoutProgress:
    def test_render_emits_json_progress(self, tmp_path, capsys, monkeypatch):
        from pygyroflow.cli import main as cli_main
        from pygyroflow.manager import StabilizationManager

        clip = _tiny_clip(tmp_path / "in.mp4")
        calls = []

        def fake_render(self, target, output, options, progress_callback=None):
            calls.append(progress_callback)
            if progress_callback:
                progress_callback(0.25)
                progress_callback(0.9)

        monkeypatch.setattr(StabilizationManager, "render", fake_render)
        cli_main.main(["--no-autosync", "--stdout-progress",
                       "-o", str(tmp_path / "out.mp4"), clip])
        assert calls and calls[0] is not None
        lines = [
            json.loads(line)
            for line in capsys.readouterr().out.splitlines()
            if line.startswith("{")
        ]
        fractions = [entry["fraction"] for entry in lines if entry["type"] == "progress"]
        assert fractions == [0.25, 0.9]
        assert lines[-1]["type"] == "done"

    def test_without_the_flag_no_callback(self, tmp_path, monkeypatch):
        from pygyroflow.cli import main as cli_main
        from pygyroflow.manager import StabilizationManager

        clip = _tiny_clip(tmp_path / "in.mp4")
        calls = []

        def fake_render(self, target, output, options, progress_callback=None):
            calls.append(progress_callback)

        monkeypatch.setattr(StabilizationManager, "render", fake_render)
        cli_main.main(["--no-autosync", "-o", str(tmp_path / "out.mp4"), clip])
        assert calls == [None]
