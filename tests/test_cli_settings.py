"""Persistent CLI defaults via the settings file (C-10)."""

from __future__ import annotations

import json

import numpy as np
import pytest

av = pytest.importorskip("av", reason="av required for CLI media tests")

from pygyroflow.cli.main import main  # noqa: E402


def _tiny_clip(path, width=64, height=48, frames=8):
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=30)
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    for i in range(frames):
        img = np.zeros((height, width, 3), np.uint8)
        img[:, :, 0] = (i * 29) % 256
        img[height // 4: height // 2, width // 4: width // 2, 1] = 210
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        frame.pts = i
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    return str(path)


class TestSettingsDefaults:
    def test_output_params_come_from_the_settings_file(
            self, tmp_path, monkeypatch):
        """A settings file carrying 'output_params' shapes the render the
        same way an explicit -p does."""
        import pygyroflow.settings as settings_mod
        from pygyroflow.manager import StabilizationManager

        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({
            "output_params": {"bitrate": 7.0},
        }))

        real_load = settings_mod.Settings.load

        def load_with_file(self, path=None):
            return real_load(self, str(settings_file))

        monkeypatch.setattr(settings_mod.Settings, "load", load_with_file)

        clip = _tiny_clip(tmp_path / "in.mp4")
        seen = {}

        def fake_render(self, target, output, options,
                        progress_callback=None):
            seen.update(options)

        monkeypatch.setattr(StabilizationManager, "render", fake_render)
        main(["--no-autosync", "-o", str(tmp_path / "out.mp4"), clip])
        assert seen.get("bitrate") == pytest.approx(7.0)

    def test_explicit_flag_beats_the_file(self, tmp_path, monkeypatch):
        import pygyroflow.settings as settings_mod
        from pygyroflow.manager import StabilizationManager

        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({
            "output_params": {"bitrate": 7.0},
        }))
        real_load = settings_mod.Settings.load
        monkeypatch.setattr(
            settings_mod.Settings, "load",
            lambda self, path=None: real_load(self, str(settings_file)),
        )

        clip = _tiny_clip(tmp_path / "in.mp4")
        seen = {}

        def fake_render(self, target, output, options,
                        progress_callback=None):
            seen.update(options)

        monkeypatch.setattr(StabilizationManager, "render", fake_render)
        main(["--no-autosync", "-o", str(tmp_path / "out.mp4"), clip,
              "-p", "{'bitrate': 12.0}"])
        assert seen.get("bitrate") == pytest.approx(12.0)

    def test_a_missing_or_broken_settings_file_is_silent(self, tmp_path,
                                                          monkeypatch):
        """No settings file (or an unreadable one) must not crash the CLI —
        the defaults simply do not apply."""
        import pygyroflow.settings as settings_mod
        monkeypatch.setattr(
            settings_mod.Settings, "load",
            lambda self, path=None: (_ for _ in ()).throw(
                OSError("unreadable")),
        )
        clip = _tiny_clip(tmp_path / "in.mp4")
        with pytest.raises(SystemExit):
            try:
                main(["--no-autosync", "--help", clip])
            except SystemExit as exc:
                # --help exits 0; reaching parse means no crash on load
                assert exc.code in (0, None)
                raise

    def test_sync_params_from_the_file_reach_the_search(self, tmp_path,
                                                         monkeypatch):
        import pygyroflow.settings as settings_mod
        from pygyroflow.synchronization import AutosyncProcess

        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({
            "sync_params": {"of_method": 0},
        }))
        real_load = settings_mod.Settings.load
        monkeypatch.setattr(
            settings_mod.Settings, "load",
            lambda self, path=None: real_load(self, str(settings_file)),
        )

        # The settings blob reaches mgr.synchronize as of_method; with no
        # gyro in the clip the sync logs and skips, so spy the manager
        # method rather than the estimator.
        from pygyroflow.manager import StabilizationManager

        seen = {}

        def fake_sync(self, **kwargs):
            seen.update(kwargs)
            return None

        monkeypatch.setattr(StabilizationManager, "synchronize", fake_sync)
        clip = _tiny_clip(tmp_path / "in.mp4")
        main(["--autosync", "-o", str(tmp_path / "out.mp4"), clip])
        assert seen.get("of_method") == 0
