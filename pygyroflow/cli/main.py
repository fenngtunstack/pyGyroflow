"""PyGyroFlow CLI — command-line interface for video stabilization.

Usage:
    pygyroflow input.mp4 [-o output.mp4] [options]
    pygyroflow clip.gyroflow [-o output.mp4]
    pygyroflow clip.mp4 preset.gyroflow

Positional inputs are sorted by type the way Gyroflow's CLI does
(``detect_types``, cli.rs): ``.json`` is a lens profile, a ``.gyroflow``
that names a ``videofile`` is a project to stabilize, a ``.gyroflow``
without one is a preset, and anything else is a video.

Options:
    --codec       Output codec (H.264/AVC, H.265/HEVC, ProRes)
    --bitrate     Output bitrate in Mbps (0 = auto)
    --lens        Lens profile name or path
    --preset      Preset to apply (a .gyroflow, or inline JSON)
    --export-project N  Write a project file instead of rendering
    --export-stmap N    Write ST-Maps instead of rendering (1 = undistort,
                        2 = + redistort; EXR next to the video)
    --stdout-progress   Log one JSON progress line per second on stdout
    --smoothness  Smoothness factor 0-1 (default: 0.5)
    --gpu         Enable GPU acceleration (experimental)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from argparse import BooleanOptionalAction

log = logging.getLogger(__name__)

# Upstream's default output suffix; `-t/--suffix` overrides it.
DEFAULT_SUFFIX = "_stabilized"

# ``--export-project N`` -> the type name ``save_project`` takes. Upstream's
# ``GyroflowProjectType`` in numeric order (cli.rs parses the same numbers).
# 0 is "do not export", which never reaches the lookup.
_PROJECT_TYPE_BY_NUMBER = {
    1: "simple",
    2: "with_gyro_data",
    3: "with_processed_data",
}

# Keys of a `synchronization` section this port's `synchronize()` understands.
# The rest still land in `lens.sync_settings` — where upstream keeps them, and
# where a project save preserves them — they just do not drive the search yet.
_SYNC_KEYS = ("of_method", "offset_method")


def detect_input_types(paths):
    """Sort positional inputs into (videos, lens_profiles, presets).

    Port of cli.rs's ``detect_types``. The distinction that matters is inside
    ``.gyroflow``: a project names a video, a preset does not. Both are the
    same file format — ``export_gyroflow_data`` always writes ``videofile``,
    and a preset is what you get when that comes out empty.
    """
    videos, lens_profiles, presets = [], [], []
    for path in paths:
        lowered = path.lower()
        if lowered.endswith(".json"):
            lens_profiles.append(path)
        elif lowered.endswith(".gyroflow"):
            videofile = ""
            try:
                with open(path, encoding="utf-8") as handle:
                    videofile = json.load(handle).get("videofile") or ""
            except Exception:
                log.warning("Could not read %s; treating it as a preset", path)
            (videos if videofile else presets).append(path)
        else:
            videos.append(path)
    return videos, lens_profiles, presets


def load_json_arg(text):
    """Parse an inline JSON object, or a path to a file holding one.

    Upstream rewrites every single quote to a double one first
    (``preset.replace('\\'', '"')``) because its own help text shows
    single-quoted examples. Doing that unconditionally would corrupt any
    string value containing an apostrophe, so this parses strictly first and
    only falls back to the rewrite.
    """
    if text.strip().startswith("{"):
        return parse_relaxed(text)
    with open(text, encoding="utf-8") as handle:
        return parse_relaxed(handle.read())


def parse_relaxed(text):
    """JSON, or the single-quoted variant Gyroflow's docs use."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(text.replace("'", '"'))


def merge_json(target, extra):
    """Recursively merge *extra* into *target* (upstream's ``merge_json``)."""
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge_json(target[key], value)
        else:
            target[key] = value


def project_output_path(rendered_path, suffix=DEFAULT_SUFFIX):
    """Where ``--export-project`` writes, given the path the render would use.

    Upstream strips the default suffix off the render filename and swaps the
    extension for ``.gyroflow`` (render_queue.rs).
    """
    folder, name = os.path.split(rendered_path)
    stem, dot, _ = name.rpartition(".")
    if not dot:
        stem, folder = name, ""
    if suffix and stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return os.path.join(folder, stem + ".gyroflow")


def main(argv=None) -> None:
    """CLI entry point. *argv* defaults to ``sys.argv[1:]`` (test hook)."""
    parser = argparse.ArgumentParser(
        prog="pygyroflow",
        description="Video stabilization using PyGyroFlow",
    )
    parser.add_argument(
        "input",
        nargs="*",
        help="Input files: videos, .gyroflow projects, lens profile JSONs, "
             "and .gyroflow presets (a project with no videofile)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path",
    )
    parser.add_argument(
        "--codec",
        default="H.265/HEVC",
        choices=[
            "H.264/AVC", "H.265/HEVC", "ProRes",
            "PNG Sequence", "EXR Sequence",
        ],
        help="Output codec (default: H.265/HEVC). The sequence options write "
             "one file per frame and need a printf pattern in -o "
             "(e.g. 'out/frame_%%05d.png'); EXR is written as 32-bit float",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Frame rate for image sequence input (EXR/PNG/...) — sequences "
             "carry no rate of their own, and without this FFmpeg assumes "
             "25 fps. Ignored for video input",
    )
    parser.add_argument(
        "--gyro",
        metavar="FILE",
        help="Separate telemetry source (a video with embedded gyro, or a "
             "gyro data file). Image sequences have no telemetry of their "
             "own, so this is required for them to be stabilized",
    )
    parser.add_argument(
        "--bitrate",
        type=float,
        default=0,
        help="Output bitrate in Mbps (0 = auto)",
    )
    parser.add_argument(
        "--lens",
        help="Lens profile name or path to JSON file",
    )
    parser.add_argument(
        "-t", "--suffix",
        default=DEFAULT_SUFFIX,
        help=f"Suffix for auto-generated output names (default: {DEFAULT_SUFFIX})",
    )
    parser.add_argument(
        "-f", "--overwrite",
        action="store_true",
        help="Overwrite the output if it already exists (default: refuse)",
    )
    parser.add_argument(
        "--preset",
        action="append",
        metavar="FILE_OR_JSON",
        default=[],
        help="Preset to apply: a path, or inline JSON such as "
             "\"{'stabilization': {'fov': 1.5}}\". Repeatable, applied in "
             "order after any preset given as a positional input",
    )
    # Persistent defaults (C-10): the same keys upstream's settings.json
    # drives — a plain `-p`/`-s` JSON blob in the file replaces having to
    # repeat it on every invocation. Explicit flags win over the file.
    try:
        from pygyroflow.settings import Settings
        _settings = Settings()
        _settings.load()
    except Exception:
        _settings = None

    def _settings_json(key: str):
        if _settings is None:
            return None
        value = _settings.get(key)
        return json.dumps(value) if isinstance(value, dict) else None

    parser.add_argument(
        "-p", "--output-params",
        metavar="JSON",
        help="Render options to merge in, e.g. "
             "\"{'codec': 'H.265/HEVC', 'bitrate': 150, 'audio': true}\". "
             "Falls back to the 'output_params' key in the settings file",
    )
    parser.add_argument(
        "-s", "--sync-params",
        metavar="JSON",
        help="Synchronization options to merge in, e.g. "
             "\"{'offset_method': 2, 'of_method': 2}\". Falls back to the "
             "'sync_params' key in the settings file",
    )
    parser.add_argument(
        "--export-project",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        metavar="N",
        help="Write a project file instead of rendering. 1 = settings only "
             "(a reusable preset); 2 = the above plus the metadata, so it "
             "loads without the clip's telemetry; 3 = the above plus the "
             "caches a plugin reads. Matches upstream's GyroflowProjectType",
    )
    parser.add_argument(
        "--export-stmap",
        type=int,
        default=0,
        choices=[0, 1, 2],
        metavar="N",
        help="Write ST-Maps instead of rendering. 1 = undistort map only; "
             "2 = both undistort and redistort (two files). Output goes "
             "next to the video with -undistort/-redistort suffixes",
    )
    parser.add_argument(
        "--stdout-progress",
        action="store_true",
        help="Emit one JSON progress line per second on stdout (for "
             "wrappers); regular logs stay on stderr",
    )
    parser.add_argument(
        "--smoothness",
        type=float,
        default=0.5,
        help="Smoothness factor 0-1 (default: 0.5)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU acceleration (verified on hw: ~2.4x faster end-to-end)",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Drop audio instead of copying it to the output",
    )
    parser.add_argument(
        "--autosync",
        action=BooleanOptionalAction,
        default=True,
        help="Auto-sync gyro to video via optical flow before stabilizing "
             "(default: on; guards reject low-confidence results and fall "
             "back to zero offset. Use --no-autosync to skip)",
    )
    parser.add_argument(
        "--horizon-lock",
        type=float,
        default=0.0,
        metavar="PERCENT",
        help="Horizon lock strength 0-100 (default 0=off); keeps the "
             "horizon level — most useful for FPV/drone footage",
    )
    parser.add_argument(
        "--horizon-gravity",
        action="store_true",
        help="Horizon lock from accelerometer gravity vectors (needs IMU "
             "accl data; falls back to quaternion mode without it)",
    )
    parser.add_argument(
        "--interpolation",
        default="lanczos4",
        choices=["bilinear", "bicubic", "lanczos4",
                 "ewa-robidoux-sharp", "ewa-robidoux", "ewa-mitchell", "ewa-catmull-rom"],
        help="Resampling interpolation (default lanczos4, matches upstream "
             "Gyroflow). The ewa-* filters stretch the kernel with the local "
             "Jacobian, which is what you want when the stabilization "
             "minifies — but on the CPU path each is a per-tap NumPy gather "
             "and costs ~10x Lanczos4 (measured 16-23 s per 1080p frame)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Calibrate a lens from chessboard footage instead of stabilizing: "
             "detect the board across the input (video or image sequence) and "
             "write a lens profile JSON that --lens can load",
    )
    parser.add_argument(
        "--cols", type=int, default=14,
        help="Chessboard inner corners per row (default 14)",
    )
    parser.add_argument(
        "--rows", type=int, default=8,
        help="Chessboard inner corners per column (default 8)",
    )
    parser.add_argument(
        "--square-size", type=float, default=1.0,
        help="Chessboard square size in any unit (default 1.0; only the ratio matters)",
    )
    parser.add_argument(
        "--calib-model", default="opencv_fisheye",
        choices=["opencv_fisheye", "poly3", "poly5", "ptlens"],
        help="Distortion model for the calibrated profile (default opencv_fisheye, "
             "which is what upstream's calibrator produces)",
    )
    parser.add_argument(
        "--calib-every", type=int, default=10,
        help="Run detection on every N-th frame (default 10; detection is the "
             "expensive part and neighbouring frames are near-duplicates)",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the version and exit",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args(argv)

    # Persistent defaults: an explicit flag beats the settings file.
    if args.output_params is None:
        args.output_params = _settings_json("output_params")
    if args.sync_params is None:
        args.sync_params = _settings_json("sync_params")

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.version:
        import pygyroflow

        print(f"pyGyroFlow {pygyroflow.__version__}")
        return

    from pygyroflow.manager import StabilizationManager
    from pygyroflow.rendering.image_sequence import (
        FFMPEG_DEFAULT_FPS,
        looks_like_image_sequence,
        sequence_output_stem,
    )

    if args.calibrate:
        sys.exit(_run_calibration(args))

    videos, lens_profiles, presets = detect_input_types(args.input)
    # `--preset` appends to the positional presets, matching upstream, which
    # pushes the flag's value onto the same list `detect_types` filled.
    for text in args.preset:
        try:
            presets.append(load_json_arg(text))
        except Exception as exc:
            log.error("Could not read --preset %s: %s", text, exc)
            sys.exit(2)

    if len(lens_profiles) > 1:
        log.error("More than one lens profile given: %s", lens_profiles)
        sys.exit(2)
    if not videos:
        log.error(
            "No videos to process. Inputs were: %s",
            ", ".join(args.input) or "(none)",
        )
        sys.exit(2)

    log.info("Videos: %s", videos)
    if lens_profiles:
        log.info("Lens profiles: %s", lens_profiles)
    if presets:
        log.info("Presets: %s", presets)

    for path in videos:
        log.info("Processing: %s", path)

        mgr = StabilizationManager()

        try:
            # A project input carries its own settings. It is loaded twice on
            # purpose: once to find out which video it refers to, then again
            # after load_video, which re-derives size/fps/frame count from the
            # file and would otherwise wipe what the project restored.
            is_project = path.lower().endswith(".gyroflow")
            if is_project:
                mgr.load_project(path)
                target = mgr.input_file.url or ""
                if not target or not os.path.exists(target):
                    log.error(
                        "Project %s points at a video that is not there: %s",
                        path, mgr.input_file.url or "(none)",
                    )
                    sys.exit(1)
                log.info("Project %s -> video %s", path, target)
            else:
                target = path

            # Load video (or image sequence)
            if looks_like_image_sequence(target) and not args.fps:
                log.warning(
                    "Image sequence input without --fps: assuming FFmpeg's "
                    "default %.0f fps. Gyro timing will be wrong if the "
                    "footage was shot at another rate.", FFMPEG_DEFAULT_FPS,
                )
            info = mgr.load_video(target, fps=args.fps)
            log.info(
                "%s: %dx%d @ %.2f fps, %.1f ms, %d frames",
                "Sequence" if info.get("image_sequence") else "Video",
                info["width"], info["height"],
                info["fps"], info["duration_ms"], info["frame_count"],
            )

            if is_project:
                mgr.load_project(path)

            for preset in presets:
                try:
                    mgr.apply_preset(preset)
                except Exception as exc:
                    log.error("Could not apply preset %s: %s", preset, exc)
                    sys.exit(1)

            # Optional separate telemetry source — an image sequence carries
            # none, so this is the only way to stabilize one.
            if args.gyro:
                mgr.load_gyro_data(args.gyro, is_video=True)
                log.info("Gyro loaded from separate source: %s", args.gyro)

            # Warn loudly if no gyro data was extracted — stabilization would
            # otherwise run on empty input and produce a "successful" but
            # un-stabilized video. This is the loud guard for the silent
            # telemetry-degradation path in manager.load_gyro_data.
            if not mgr.gyro.quaternions:
                log.warning(
                    "No gyro/IMU data found in %s. Output will NOT be "
                    "stabilized (only lens correction applies). "
                    "Use --gyro FILE for image sequences.", target,
                )

            # Load lens profile: --lens wins over a positional profile; with
            # neither, whatever the project or preset set stays.
            lens = args.lens or (lens_profiles[0] if lens_profiles else None)
            if lens:
                try:
                    mgr.load_lens_profile(lens)
                    log.info("Lens profile: %s", mgr.lens.get_display_name())
                except Exception as exc:
                    log.error("Failed to load lens profile '%s': %s", lens, exc)
                    sys.exit(1)

            # Configure smoothing
            mgr.smoothing.current().set_parameter("smoothness", args.smoothness)
            if args.horizon_lock > 0.0:
                mgr.smoothing.horizon_lock.set_horizon(
                    lock_percent=min(100.0, args.horizon_lock),
                    roll=0.0,
                    lock_pitch=False,
                    pitch=0.0,
                )
                if args.horizon_gravity:
                    mgr.gyro.set_use_gravity_vectors(True)
                log.info("Horizon lock: %.0f%%", min(100.0, args.horizon_lock))

            # Sync parameters, in increasing order of precedence: the lens
            # profile's own sync_settings, then a preset's synchronization
            # section (which apply_preset merged into that same dict), then
            # --sync-params.
            sync_kwargs = {}
            for source, name in (
                (mgr.lens.sync_settings, "lens profile" if not presets else "lens profile/preset"),
                (load_json_arg(args.sync_params) if args.sync_params else None, "--sync-params"),
            ):
                if not isinstance(source, dict):
                    continue
                for key in _SYNC_KEYS:
                    if key in source:
                        sync_kwargs[key] = int(source[key])
                        log.info("Sync %s from %s: %s", key, name, source[key])

            # Auto-sync gyro timeline to video (optical flow based)
            if args.autosync:
                try:
                    offset = mgr.synchronize(**sync_kwargs)
                except Exception as exc:
                    log.warning("Auto-sync failed: %s", exc)
                    offset = None
                if offset is None:
                    log.warning("Continuing without sync offset")

            # Run stabilization pipeline
            log.info("Computing stabilization...")
            mgr.recompute_blocking()

            # Determine output path
            is_sequence_out = args.codec in ("PNG Sequence", "EXR Sequence")
            if args.output:
                output = args.output
            elif is_sequence_out:
                extension = "png" if args.codec == "PNG Sequence" else "exr"
                directory = os.path.join(
                    os.path.dirname(os.path.abspath(target)),
                    sequence_output_stem(target) + args.suffix,
                )
                os.makedirs(directory, exist_ok=True)
                output = os.path.join(directory, f"frame_%05d.{extension}")
            elif info.get("image_sequence"):
                output = sequence_output_stem(target) + args.suffix + ".mp4"
            else:
                base = target.rsplit(".", 1)
                output = (
                    base[0] + args.suffix + "." + (base[1] if len(base) > 1 else "mp4")
                )

            if args.export_project:
                # Nothing is rendered, so the printf-pattern rule does not
                # apply: the project lands where the video would have gone.
                project_path = project_output_path(output, args.suffix)
                refuse_to_overwrite(project_path, args.overwrite)
                mgr.save_project(
                    project_path, _PROJECT_TYPE_BY_NUMBER[args.export_project]
                )
                log.info("Wrote project: %s", project_path)
                continue

            if args.export_stmap:
                # Like --export-project: no render, the maps land where
                # the video would have gone. EXR per upstream's stmap.rs.
                from pygyroflow.stmap.exporter import STMapExporter

                stem = output.rpartition(".")[0] or output
                base = stem
                paths = []
                if args.export_stmap >= 1:
                    paths.append(("undistort", f"{base}-undistort.exr"))
                if args.export_stmap >= 2:
                    paths.append(("distort", f"{base}-redistort.exr"))
                exporter = STMapExporter(mgr._build_compute_params())
                for map_type, path in paths:
                    refuse_to_overwrite(path, args.overwrite)
                    exporter.export(path, map_type=map_type)
                    log.info("Wrote ST-Map: %s", path)
                continue

            refuse_to_overwrite(output, args.overwrite)

            render_options = {
                "codec": args.codec,
                "bitrate": args.bitrate,
                "use_gpu": args.gpu,
                "audio": not args.no_audio,
                "interpolation": {
                    "bilinear": 0,
                    "bicubic": 1,
                    "lanczos4": 2,
                    "ewa-robidoux-sharp": 3,
                    "ewa-robidoux": 4,
                    "ewa-mitchell": 5,
                    "ewa-catmull-rom": 6,
                }[args.interpolation],
            }
            # A preset's `output` section and then --output-params override the
            # defaults above — upstream's order (setup_defaults, then the
            # preset, then the flag).
            merge_json(render_options, mgr.output_options())
            if args.output_params:
                merge_json(render_options, load_json_arg(args.output_params))

            # Render
            log.info("Rendering to: %s", output)
            progress_cb = None
            if args.stdout_progress:
                # One JSON line per callback, timestamped by wall clock so
                # a wrapper can pace itself even when frames come in
                # bursts. Regular logs stay wherever logging sends them.
                started = time.time()

                def progress_cb(fraction: float) -> None:
                    print(
                        json.dumps({
                            "type": "progress",
                            "fraction": round(max(0.0, min(1.0, float(fraction))), 4),
                            "elapsed_s": round(time.time() - started, 1),
                        }),
                        flush=True,
                    )

            mgr.render(target, output, render_options,
                       progress_callback=progress_cb)
            if progress_cb is not None:
                print(json.dumps({"type": "done", "output": output}), flush=True)
            log.info("Done: %s", output)

        except SystemExit:
            raise
        except Exception as exc:
            log.error("Failed to process %s: %s", path, exc)
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)


def refuse_to_overwrite(path: str, overwrite: bool) -> None:
    """Upstream writes a .tmp and renames under `-f`; refusing to clobber is
    the same guarantee for a port that writes in place."""
    if not overwrite and os.path.exists(path):
        log.error("Output %s already exists (use -f to overwrite)", path)
        sys.exit(1)


def _run_calibration(args) -> int:
    """Chessboard-calibrate a lens and write a lens profile JSON.

    Returns a process exit code.  Shares the input handling with the normal
    path, so a video, an image sequence or a single still all work.
    """
    import json
    import os

    from pygyroflow.calibration import LensCalibrator
    from pygyroflow.calibration.calibrator import SUPPORTED_MODELS, iter_gray_frames
    from pygyroflow.rendering.image_sequence import (
        FFMPEG_DEFAULT_FPS,
        looks_like_image_sequence,
        sequence_output_stem,
    )

    if len(args.input) != 1:
        log.error("Calibration takes exactly one input, got %d", len(args.input))
        return 2
    path = args.input[0]

    if args.calib_model not in SUPPORTED_MODELS:
        log.error("Unsupported --calib-model '%s' (choose from %s)",
                  args.calib_model, ", ".join(SUPPORTED_MODELS))
        return 2

    if looks_like_image_sequence(path) and not args.fps:
        log.warning(
            "Image sequence input without --fps: assuming FFmpeg's default %.0f fps",
            FFMPEG_DEFAULT_FPS,
        )

    calibrator = LensCalibrator(
        columns=args.cols,
        rows=args.rows,
        square_size=args.square_size,
        distortion_model=args.calib_model,
    )

    log.info("Scanning %s for a %dx%d chessboard (every %d frames)",
             path, args.cols, args.rows, args.calib_every)
    try:
        frames = iter_gray_frames(path, fps=args.fps, every_n=args.calib_every)
        calibrator.feed_frames(frames, every_n=1)
    except FileNotFoundError:
        log.error("No such file, and not an image sequence: %s", path)
        return 1
    except Exception as exc:
        log.error("Failed to read frames from %s: %s", path, exc)
        return 1

    log.info("Chessboard detected in %d frame(s), %d accepted for calibration",
             calibrator.num_detected, calibrator.num_candidates)
    if calibrator.num_candidates < 2:
        log.error(
            "Not enough usable frames (%d). Move the board around the frame — "
            "calibrating only from the centre leaves the wider distortion "
            "coefficients undetermined.", calibrator.num_candidates,
        )
        return 1

    try:
        result = calibrator.calibrate()
    except ValueError as exc:
        log.error("Calibration refused: %s", exc)
        return 1
    except RuntimeError as exc:
        log.error("Calibration failed: %s", exc)
        return 1

    profile = calibrator.to_lens_profile_dict()
    profile["name"] = f"{os.path.basename(os.path.abspath(path))} calibration"
    profile["date"] = time.strftime("%Y-%m-%d")

    output = args.output or (sequence_output_stem(path) + "_lens_profile.json")
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(profile, handle, indent=2)

    K = result.camera_matrix
    log.info(
        "Calibrated from %d frame(s): fx=%.2f fy=%.2f cx=%.2f cy=%.2f, RMS=%.3f px",
        len(result.used_frames), K[0, 0], K[1, 1], K[0, 2], K[1, 2], result.rms,
    )
    if result.model_fit_error is not None:
        log.warning(
            "Model '%s' reproduces the calibrated fisheye curve to %.2f%% over the "
            "image radius — check the frame edges before trusting it",
            args.calib_model, result.model_fit_error * 100.0,
        )
    log.info("Wrote lens profile: %s", output)
    return 0


if __name__ == "__main__":
    main()
