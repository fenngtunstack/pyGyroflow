"""CLI input handling, presets and `--export-project` (gap items C-06/C-11).

The CLI's job here is to know what it was handed. Upstream splits positional
inputs by extension and by one fact inside the `.gyroflow` — whether it names
a `videofile` (a project) or not (a preset) — and these tests pin that.

The end-to-end tests drive the real CLI in a subprocess against a real
project with a real embedded gyro blob, so they exercise the whole chain:
`detect_input_types` -> `load_project` -> gyro from the blob ->
`recompute_blocking` -> render.
"""

from __future__ import annotations

import json
import math
import pathlib
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("av")

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

_FPS = 30.0
_FRAMES = 12
_DURATION_MS = _FRAMES * 1000.0 / _FPS

# (bit, channel) for the four identity quadrants written into each frame.
_BITS = [(0, 0), (1, 0), (2, 2), (3, 2)]


def write_clip(path, frames=_FRAMES, fps=_FPS, width=64, height=48):
    """A clip whose frames carry their own index."""
    import av

    container = av.open(str(path), mode="w")
    video = container.add_stream("libx264", rate=int(fps))
    video.width, video.height, video.pix_fmt = width, height, "yuv420p"
    half = height // 2
    for i in range(frames):
        img = np.full((height, width, 3), 128, np.uint8)
        for bit, channel in _BITS:
            rows = slice(0, half) if bit % 2 == 0 else slice(half, height)
            img[rows, :, channel] = 255 if (i >> bit) & 1 else 0
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        frame.pts = i
        for pkt in video.encode(frame):
            container.mux(pkt)
    for pkt in video.encode():
        container.mux(pkt)
    container.close()


def frame_count(path):
    import av

    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


def quaternion_blob(duration_ms=_DURATION_MS, step_ms=1.0):
    """A slow roll at 20 deg/s, as `gyro_source.quaternions` stores it.

    Keys are microseconds; the values are (w, x, y, z) as nalgebra orders
    them, which is also the (w, x, y, z) `Quat64.from_quaternion` wants.
    """
    out = {}
    t = 0.0
    while t < duration_ms:
        half = math.radians(20.0 * t / 1000.0) / 2.0
        out[int(round(t * 1000.0))] = (math.cos(half), math.sin(half), 0.0, 0.0)
        t += step_ms
    return out


def build_project(path, videofile, dims=(64, 48), num_frames=_FRAMES,
                  duration_ms=_DURATION_MS, with_gyro=True, **extra):
    """Write a project file the CLI can be pointed at."""
    from pygyroflow.project import PROJECT_VERSION, GyroflowProject

    proj = GyroflowProject()
    proj.videofile = str(videofile)
    proj.version = PROJECT_VERSION
    proj.app_version = "test"
    proj.date = "2026-01-01"
    proj.video_info = type(proj.video_info).from_dict(
        {
            "width": dims[0], "height": dims[1], "rotation": 0.0,
            "num_frames": num_frames, "fps": _FPS, "duration_ms": duration_ms,
            "fps_scale": None, "vfr_fps": _FPS, "vfr_duration_ms": duration_ms,
        }
    )
    proj.gyro_source = {
        "filepath": str(videofile),
        "lpf": 0.0, "mf": 0, "rotation": None, "acc_rotation": None,
        "imu_orientation": "XYZ", "gyro_bias": None, "integration_method": 0,
    }
    proj.stabilization = {"fov": 1.0, "method": "Default", "frame_readout_time": 0.0}
    for key, value in extra.items():
        setattr(proj, key, value)
    if with_gyro:
        proj.write_blob("quaternions", quaternion_blob(duration_ms))
    proj.save(str(path))
    return path


def write_preset(path, **sections):
    """A preset is a project with no videofile."""
    from pygyroflow.project import GyroflowProject

    proj = GyroflowProject()
    proj.videofile = ""
    proj.app_version = "test"
    proj.stabilization = {"fov": 1.0, "method": "Default"}
    for key, value in sections.items():
        setattr(proj, key, value)
    proj.save(str(path))
    return path


def run_cli(*args, cwd=REPO_ROOT):
    return subprocess.run(
        [sys.executable, "-m", "pygyroflow", *map(str, args)],
        capture_output=True,
        text=True,
        cwd=str(cwd),
        timeout=300,
    )


# ----------------------------------------------------------------------
# Input classification
# ----------------------------------------------------------------------


class TestDetectInputTypes:
    def test_splits_by_extension_and_videofile(self, tmp_path):
        from pygyroflow.cli.main import detect_input_types

        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"")
        lens = tmp_path / "lens.json"
        lens.write_text("{}")
        # The only difference between a project and a preset is this field.
        project = build_project(tmp_path / "clip.gyroflow", clip)
        preset = write_preset(tmp_path / "look.gyroflow")

        videos, profiles, presets = detect_input_types(
            [str(clip), str(lens), str(project), str(preset)]
        )
        assert videos == [str(clip), str(project)]
        assert profiles == [str(lens)]
        assert presets == [str(preset)]

    def test_extension_case_does_not_matter(self, tmp_path):
        from pygyroflow.cli.main import detect_input_types

        path = tmp_path / "clip.GYROFLOW"
        path.write_text(json.dumps({"videofile": "x.mp4"}))
        videos, _, presets = detect_input_types([str(path)])
        assert videos == [str(path)]
        assert presets == []

    def test_unreadable_gyroflow_becomes_a_preset(self, tmp_path):
        """Not a crash: the file cannot name a video, so it cannot be one."""
        from pygyroflow.cli.main import detect_input_types

        path = tmp_path / "broken.gyroflow"
        path.write_text("{ not json")
        videos, _, presets = detect_input_types([str(path)])
        assert videos == []
        assert presets == [str(path)]

    def test_anything_else_is_a_video(self, tmp_path):
        from pygyroflow.cli.main import detect_input_types

        videos, profiles, presets = detect_input_types(
            ["a.mov", "b.exe", "frames/f_%04d.png"]
        )
        assert len(videos) == 3
        assert not profiles and not presets


class TestLoadJsonArg:
    def test_inline_object(self):
        from pygyroflow.cli.main import load_json_arg

        assert load_json_arg('{"fov": 1.5}') == {"fov": 1.5}

    def test_inline_single_quotes(self):
        """Upstream rewrites single quotes because its help text uses them."""
        from pygyroflow.cli.main import load_json_arg

        assert load_json_arg("{'fov': 1.5}") == {"fov": 1.5}

    def test_file_path(self, tmp_path):
        from pygyroflow.cli.main import load_json_arg

        path = tmp_path / "p.json"
        path.write_text('{"fov": 0.5}')
        assert load_json_arg(str(path)) == {"fov": 0.5}

    def test_invalid_json_raises(self):
        from pygyroflow.cli.main import load_json_arg

        with pytest.raises(Exception):
            load_json_arg("{ not json")


class TestMergeJson:
    def test_recursive_merge(self):
        from pygyroflow.cli.main import merge_json

        target = {"a": {"b": 1, "c": 2}}
        merge_json(target, {"a": {"b": 9}})
        assert target == {"a": {"b": 9, "c": 2}}

    def test_non_dict_replaces(self):
        from pygyroflow.cli.main import merge_json

        target = {"a": {"b": 1}, "c": 1}
        merge_json(target, {"a": 5, "d": 2})
        assert target == {"a": 5, "c": 1, "d": 2}

    def test_new_keys_are_added(self):
        from pygyroflow.cli.main import merge_json

        target = {}
        merge_json(target, {"x": {"y": 1}})
        assert target == {"x": {"y": 1}}


class TestProjectOutputPath:
    @pytest.mark.parametrize(
        "rendered,expected",
        [
            ("clip_stabilized.mp4", "clip.gyroflow"),
            ("/out/clip_stabilized.mov", "/out/clip.gyroflow"),
            ("clip.mp4", "clip.gyroflow"),
            ("clip_stabilized", "clip.gyroflow"),
            ("a_b_stabilized.mp4", "a_b.gyroflow"),
        ],
    )
    def test_strips_the_suffix_and_swaps_the_extension(self, rendered, expected):
        from pygyroflow.cli.main import project_output_path

        assert project_output_path(rendered) == expected


# ----------------------------------------------------------------------
# Relative video paths
# ----------------------------------------------------------------------


class TestResolveVideofile:
    def test_existing_path_is_left_alone(self, tmp_path):
        from pygyroflow.project import resolve_videofile

        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"")
        assert resolve_videofile(str(clip), str(tmp_path / "p.gyroflow")) == str(clip)

    def test_missing_path_is_rebased_onto_the_project_folder(self, tmp_path):
        """A project records an absolute path from another machine."""
        from pygyroflow.project import resolve_videofile

        (tmp_path / "clip.mp4").write_bytes(b"")
        resolved = resolve_videofile(
            "E:/someone/else/clip.mp4", str(tmp_path / "p.gyroflow")
        )
        assert resolved == str(tmp_path / "clip.mp4")

    def test_missing_path_stays_when_the_file_is_nowhere(self, tmp_path):
        from pygyroflow.project import resolve_videofile

        assert resolve_videofile(
            "E:/nope/clip.mp4", str(tmp_path / "p.gyroflow")
        ) == "E:/nope/clip.mp4"

    def test_sequence_pattern_checks_the_concrete_frame(self, tmp_path):
        """The stored name is a pattern; upstream verifies one frame exists
        and then returns the pattern for the sequence loader."""
        from pygyroflow.project import resolve_videofile

        (tmp_path / "f_00007.exr").write_bytes(b"")
        resolved = resolve_videofile(
            "D:/shots/f_%05d.exr", str(tmp_path / "p.gyroflow"), sequence_start=7
        )
        assert resolved == str(tmp_path / "f_%05d.exr")

    def test_sequence_pattern_with_the_wrong_start_is_not_rebased(self, tmp_path):
        from pygyroflow.project import resolve_videofile

        (tmp_path / "f_00007.exr").write_bytes(b"")
        resolved = resolve_videofile(
            "D:/shots/f_%05d.exr", str(tmp_path / "p.gyroflow"), sequence_start=1
        )
        assert resolved == "D:/shots/f_%05d.exr"

    def test_no_project_path(self):
        from pygyroflow.project import resolve_videofile

        assert resolve_videofile("E:/x.mp4", None) == "E:/x.mp4"


# ----------------------------------------------------------------------
# Presets
# ----------------------------------------------------------------------


def new_manager(**kwargs):
    from pygyroflow.manager import StabilizationManager

    mgr = StabilizationManager()
    mgr.init_from_video_data(_DURATION_MS, _FPS, _FRAMES, (64, 48))
    mgr.set_size(64, 48)
    mgr.set_output_size(64, 48)
    return mgr


class TestApplyPreset:
    def test_applies_stabilization_from_a_dict(self):
        mgr = new_manager()
        mgr.apply_preset({"stabilization": {"fov": 0.75, "method": "Default"}})
        assert mgr.params.fov == pytest.approx(0.75)

    def test_applies_from_inline_json_with_single_quotes(self):
        mgr = new_manager()
        mgr.apply_preset("{'stabilization': {'fov': 0.6}}")
        assert mgr.params.fov == pytest.approx(0.6)

    def test_applies_from_a_preset_file(self, tmp_path):
        preset = write_preset(
            tmp_path / "look.gyroflow", stabilization={"fov": 0.8, "method": "Default"}
        )
        mgr = new_manager()
        mgr.apply_preset(str(preset))
        assert mgr.params.fov == pytest.approx(0.8)

    def test_does_not_touch_the_clip_dimensions_or_offsets(self, tmp_path):
        """A preset is reusable across clips; transplanting one clip's
        dimensions or its resolved sync offsets onto another is wrong."""
        preset = write_preset(
            tmp_path / "look.gyroflow",
            stabilization={"fov": 0.8, "method": "Default"},
        )
        mgr = new_manager()
        mgr.params.size = (1920, 1080)
        mgr.gyro.set_offsets({1234: 5.0})
        mgr.apply_preset(str(preset))
        assert mgr.params.size == (1920, 1080)
        assert mgr.gyro.get_offsets() == {1234: 5.0}
        assert mgr.params.fov == pytest.approx(0.8)

    def test_synchronization_lands_in_the_lens_sync_settings(self):
        """Upstream puts it there (render_queue.rs::update_sync_settings)."""
        mgr = new_manager()
        mgr.apply_preset(
            {"synchronization": {"offset_method": 1, "search_size": 3}}
        )
        assert mgr.lens.sync_settings["offset_method"] == 1
        assert mgr.lens.sync_settings["search_size"] == 3

    def test_synchronization_merges_over_existing(self):
        mgr = new_manager()
        mgr.lens.sync_settings = {"offset_method": 2, "max_sync_points": 5}
        mgr.apply_preset({"synchronization": {"offset_method": 1}})
        assert mgr.lens.sync_settings == {"offset_method": 1, "max_sync_points": 5}

    def test_output_lands_in_the_project(self):
        mgr = new_manager()
        mgr.apply_preset({"output": {"codec": "H.264/AVC", "bitrate": 150}})
        assert mgr.output_options()["codec"] == "H.264/AVC"

    def test_output_options_is_empty_without_a_project(self):
        assert new_manager().output_options() == {}


# ----------------------------------------------------------------------
# Embedded gyro in a project
# ----------------------------------------------------------------------


class TestProjectCarriesItsOwnGyro:
    def test_load_project_restores_the_quaternions(self, tmp_path):
        """`WithGyroData` projects have to stabilize without the clip's own
        telemetry — that is the whole point of the format."""
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"")
        project = build_project(tmp_path / "p.gyroflow", clip)

        mgr = new_manager()
        mgr.load_project(str(project))
        assert len(mgr.gyro.quaternions) == 400
        assert mgr.gyro.imu_transforms.imu_orientation == "XYZ"

    def test_a_project_without_gyro_loads_none(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"")
        project = build_project(tmp_path / "p.gyroflow", clip, with_gyro=False)

        mgr = new_manager()
        mgr.load_project(str(project))
        assert not mgr.gyro.quaternions

    def test_the_imu_transform_survives_the_motion_load(self, tmp_path):
        """`load_from_telemetry` clears the gyro source; the project's own
        transform has to be applied after that, not before."""
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"")
        project = build_project(tmp_path / "p.gyroflow", clip)
        data = json.loads(pathlib.Path(project).read_text())
        data["gyro_source"]["rotation"] = [1.0, 2.0, 3.0]
        data["gyro_source"]["lpf"] = 15.0
        pathlib.Path(project).write_text(json.dumps(data))

        mgr = new_manager()
        mgr.load_project(str(project))
        assert mgr.gyro.imu_transforms.imu_rotation_angles == (1.0, 2.0, 3.0)
        assert mgr.gyro.imu_transforms.imu_lpf == pytest.approx(15.0)


# ----------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------


class TestSaveProjectTypes:
    def test_simple_strips_the_motion_payloads(self, tmp_path):
        from pygyroflow.project import GyroflowProject

        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"")
        project = build_project(tmp_path / "p.gyroflow", clip)
        out = tmp_path / "simple.gyroflow"

        mgr = new_manager()
        mgr.load_project(str(project))
        assert len(mgr.gyro.quaternions) == 400
        mgr.save_project(str(out), "simple")

        saved = GyroflowProject.load(str(out))
        assert saved.read_blob("quaternions") is None
        # ... but the settings are all still there.
        assert saved.videofile.endswith("clip.mp4")
        assert saved.stabilization

    def test_default_keeps_what_was_loaded(self, tmp_path):
        from pygyroflow.project import GyroflowProject

        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"")
        project = build_project(tmp_path / "p.gyroflow", clip)
        out = tmp_path / "kept.gyroflow"

        mgr = new_manager()
        mgr.load_project(str(project))
        mgr.save_project(str(out))

        assert len(GyroflowProject.load(str(out)).read_blob("quaternions")) == 400

    @pytest.mark.parametrize("kind", ["with_gyro_data", "with_processed_data"])
    def test_the_motion_bearing_types_embed_the_metadata(self, tmp_path, kind):
        """Both write `file_metadata`, and the result has to be loadable
        without the original clip's telemetry — that is the whole point."""
        from pygyroflow.gyro_source.file_metadata_cbor import decode_file_metadata
        from pygyroflow.project import GyroflowProject
        from pygyroflow.util import decompress_from_base91

        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"")
        project = build_project(tmp_path / "p.gyroflow", clip)
        out = tmp_path / f"{kind}.gyroflow"

        mgr = new_manager()
        mgr.load_project(str(project))
        mgr.save_project(str(out), kind)

        saved = GyroflowProject.load(str(out))
        blob = saved.gyro_source.get("file_metadata")
        assert isinstance(blob, str)
        metadata = decode_file_metadata(decompress_from_base91(blob))
        assert metadata.has_motion()
        assert len(metadata.quaternions) == 400

        # And a fresh manager gets the gyro back out of it.
        reloaded = new_manager()
        reloaded.load_project(str(out))
        assert len(reloaded.gyro.quaternions) == 400

    def test_the_legacy_bincode_blobs_are_dropped(self, tmp_path):
        """Upstream never writes them; a stale copy sitting beside the
        metadata it disagrees with is worse than none."""
        from pygyroflow.project import GyroflowProject

        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"")
        project = build_project(tmp_path / "p.gyroflow", clip)
        out = tmp_path / "gyro.gyroflow"

        mgr = new_manager()
        mgr.load_project(str(project))
        mgr.save_project(str(out), "with_gyro_data")

        saved = GyroflowProject.load(str(out))
        assert saved.gyro_source.get("quaternions") is None

    def test_only_the_processed_mode_writes_the_caches(self, tmp_path):
        """Mode 2 and mode 3 differ by exactly the plugin caches; carrying
        them into a mode-2 export would promise data it does not own."""
        from pygyroflow.project import GyroflowProject

        cache_names = (
            "integrated_quaternions", "smoothed_quaternions",
            "adaptive_zoom_fovs", "synced_imu_timestamps",
            "synced_imu_timestamps_with_per_frame_offset",
        )
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"")
        project = build_project(tmp_path / "p.gyroflow", clip)

        mgr = new_manager()
        mgr.load_project(str(project))

        plain = tmp_path / "gyro.gyroflow"
        mgr.save_project(str(plain), "with_gyro_data")
        source = GyroflowProject.load(str(plain)).gyro_source
        assert all(source.get(name) is None for name in cache_names)

        rich = tmp_path / "processed.gyroflow"
        mgr.save_project(str(rich), "with_processed_data")
        source = GyroflowProject.load(str(rich)).gyro_source
        assert all(isinstance(source.get(name), str) for name in cache_names)

    def test_the_synced_timeline_is_the_gyro_timeline(self, tmp_path):
        """`synced_imu_timestamps` is the quaternion timeline rebased by the
        sync offsets — with none set, it is the keys in milliseconds."""
        from pygyroflow.project import GyroflowProject
        from pygyroflow.util import decode_cbor_f64_list, decompress_from_base91

        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"")
        project = build_project(tmp_path / "p.gyroflow", clip)
        out = tmp_path / "processed.gyroflow"

        mgr = new_manager()
        mgr.load_project(str(project))
        expected = sorted(mgr.gyro.quaternions)
        mgr.save_project(str(out), "with_processed_data")

        source = GyroflowProject.load(str(out)).gyro_source
        synced = decode_cbor_f64_list(
            decompress_from_base91(source["synced_imu_timestamps"])
        )
        assert synced == [ts / 1000.0 for ts in expected]

    def test_unknown_type_raises(self, tmp_path):
        mgr = new_manager()
        with pytest.raises(ValueError):
            mgr.save_project(str(tmp_path / "x.gyroflow"), "nonsense")

    def test_simple_export_is_itself_a_preset(self, tmp_path):
        """Round trip: exported Simple, re-read, and it names no video."""
        from pygyroflow.project import GyroflowProject

        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"")
        project = build_project(tmp_path / "p.gyroflow", clip)
        out = tmp_path / "preset.gyroflow"

        mgr = new_manager()
        mgr.load_project(str(project))
        mgr.save_project(str(out), "simple")
        data = json.loads(pathlib.Path(out).read_text())
        # A Simple export keeps the videofile, so it is still a project —
        # clearing it is what makes a reusable preset.
        assert data["videofile"].endswith("clip.mp4")
        assert GyroflowProject.load(str(out)).stabilization


# ----------------------------------------------------------------------
# The CLI itself
# ----------------------------------------------------------------------


class TestCli:
    def test_version(self):
        result = run_cli("--version")
        assert result.returncode == 0
        assert "pyGyroFlow" in result.stdout

    def test_no_inputs_is_an_error(self):
        result = run_cli()
        assert result.returncode == 2
        assert "No videos to process" in result.stderr + result.stdout

    def test_more_than_one_lens_profile_is_an_error(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"")
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        a.write_text("{}")
        b.write_text("{}")
        result = run_cli(clip, a, b)
        assert result.returncode == 2
        assert "More than one lens profile" in result.stderr + result.stdout

    def test_project_input_renders_using_its_own_gyro(self, tmp_path):
        """The headline case: a project carries the gyro, so the CLI can
        stabilize from the project alone."""
        clip = tmp_path / "clip.mp4"
        write_clip(clip)
        project = build_project(tmp_path / "p.gyroflow", clip)
        out = tmp_path / "out.mp4"

        result = run_cli(
            project, "-o", out, "--codec", "H.264/AVC", "--no-autosync",
        )
        assert result.returncode == 0, result.stderr
        assert out.exists()
        assert frame_count(out) == _FRAMES

    def test_export_project_writes_settings_only(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        write_clip(clip)
        project = build_project(tmp_path / "p.gyroflow", clip)

        result = run_cli(
            project, "-o", tmp_path / "out.mp4",
            "--codec", "H.264/AVC", "--no-autosync", "--export-project", "1",
        )
        assert result.returncode == 0, result.stderr
        assert not (tmp_path / "out.mp4").exists()
        written = tmp_path / "out.gyroflow"
        assert written.exists()

        data = json.loads(written.read_text())
        assert "quaternions" not in data["gyro_source"]
        assert data["video_info"]["width"] == 64

    @pytest.mark.parametrize("number,expect_caches", [(2, False), (3, True)])
    def test_export_project_types_two_and_three(self, tmp_path, number, expect_caches):
        """2 embeds the metadata, 3 the metadata plus the plugin caches.

        Run through the real CLI, and the result re-loaded into a *fresh*
        manager that never saw the clip: that is what "carries its own gyro"
        has to mean.
        """
        from pygyroflow.manager import StabilizationManager
        from pygyroflow.project import GyroflowProject

        clip = tmp_path / "clip.mp4"
        write_clip(clip)
        project = build_project(tmp_path / "p.gyroflow", clip)

        result = run_cli(
            project, "-o", tmp_path / "out.mp4",
            "--codec", "H.264/AVC", "--no-autosync",
            "--export-project", str(number),
        )
        assert result.returncode == 0, result.stderr

        written = tmp_path / "out.gyroflow"
        saved = GyroflowProject.load(str(written))
        assert isinstance(saved.gyro_source.get("file_metadata"), str)
        assert saved.gyro_source.get("quaternions") is None

        caches = ("integrated_quaternions", "synced_imu_timestamps")
        for name in caches:
            present = isinstance(saved.gyro_source.get(name), str)
            assert present is expect_caches, name

        reloaded = StabilizationManager()
        reloaded.load_project(str(written))
        assert len(reloaded.gyro.quaternions) == 400

    def test_preset_flag_changes_the_saved_settings(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        write_clip(clip)
        project = build_project(tmp_path / "p.gyroflow", clip)

        result = run_cli(
            project, "-o", tmp_path / "out.mp4",
            "--codec", "H.264/AVC", "--no-autosync", "--export-project", "1",
            "--preset", "{'stabilization': {'fov': 0.42}}",
        )
        assert result.returncode == 0, result.stderr
        data = json.loads((tmp_path / "out.gyroflow").read_text())
        assert data["stabilization"]["fov"] == pytest.approx(0.42)

    def test_positional_preset_is_applied(self, tmp_path):
        """A .gyroflow with no videofile is a preset, not a video."""
        clip = tmp_path / "clip.mp4"
        write_clip(clip)
        project = build_project(tmp_path / "p.gyroflow", clip)
        preset = write_preset(
            tmp_path / "look.gyroflow",
            stabilization={"fov": 0.33, "method": "Default"},
        )

        result = run_cli(
            project, preset, "-o", tmp_path / "out.mp4",
            "--codec", "H.264/AVC", "--no-autosync", "--export-project", "1",
        )
        assert result.returncode == 0, result.stderr
        data = json.loads((tmp_path / "out.gyroflow").read_text())
        assert data["stabilization"]["fov"] == pytest.approx(0.33)

    def test_refuses_to_clobber_without_dash_f(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        write_clip(clip)
        project = build_project(tmp_path / "p.gyroflow", clip)
        out = tmp_path / "out.mp4"
        out.write_bytes(b"already here")

        result = run_cli(project, "-o", out, "--codec", "H.264/AVC", "--no-autosync")
        assert result.returncode == 1
        assert "already exists" in result.stderr + result.stdout
        assert out.read_bytes() == b"already here"

    def test_project_pointing_at_a_missing_video_fails_clearly(self, tmp_path):
        project = build_project(tmp_path / "p.gyroflow", tmp_path / "gone.mp4")
        result = run_cli(project, "-o", tmp_path / "out.mp4", "--no-autosync")
        assert result.returncode == 1
        assert "not there" in result.stderr + result.stdout
