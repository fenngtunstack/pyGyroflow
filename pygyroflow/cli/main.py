"""PyGyroFlow CLI — command-line interface for video stabilization.

Usage:
    pygyroflow input.mp4 [-o output.mp4] [options]

Options:
    --codec       Output codec (H.264/AVC, H.265/HEVC, ProRes)
    --bitrate     Output bitrate in Mbps (0 = auto)
    --lens        Lens profile name or path
    --smoothness  Smoothness factor 0-1 (default: 0.5)
    --no-gpu      (removed) GPU is disabled by default
    --gpu         Enable GPU acceleration (known-broken, experimental)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from argparse import BooleanOptionalAction

log = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="pygyroflow",
        description="Video stabilization using PyGyroFlow",
    )
    parser.add_argument(
        "input",
        nargs="+",
        help="Input video file(s)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path",
    )
    parser.add_argument(
        "--codec",
        default="H.265/HEVC",
        choices=["H.264/AVC", "H.265/HEVC", "ProRes"],
        help="Output codec (default: H.265/HEVC)",
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
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from pygyroflow.manager import StabilizationManager
    from pygyroflow.rendering.image_sequence import (
        FFMPEG_DEFAULT_FPS,
        looks_like_image_sequence,
        sequence_output_stem,
    )

    if args.calibrate:
        sys.exit(_run_calibration(args))

    for path in args.input:
        log.info("Processing: %s", path)

        mgr = StabilizationManager()

        try:
            # Load video (or image sequence)
            if looks_like_image_sequence(path) and not args.fps:
                log.warning(
                    "Image sequence input without --fps: assuming FFmpeg's "
                    "default %.0f fps. Gyro timing will be wrong if the "
                    "footage was shot at another rate.", FFMPEG_DEFAULT_FPS,
                )
            info = mgr.load_video(path, fps=args.fps)
            log.info(
                "%s: %dx%d @ %.2f fps, %.1f ms, %d frames",
                "Sequence" if info.get("image_sequence") else "Video",
                info["width"], info["height"],
                info["fps"], info["duration_ms"], info["frame_count"],
            )

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
                    "Use --gyro FILE for image sequences.", path,
                )

            # Load lens profile
            if args.lens:
                try:
                    mgr.load_lens_profile(args.lens)
                    log.info("Lens profile: %s", mgr.lens.get_display_name())
                except Exception as exc:
                    log.error("Failed to load lens profile '%s': %s", args.lens, exc)
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

            # Auto-sync gyro timeline to video (optical flow based)
            if args.autosync:
                try:
                    offset = mgr.synchronize()
                except Exception as exc:
                    log.warning("Auto-sync failed: %s", exc)
                    offset = None
                if offset is None:
                    log.warning("Continuing without sync offset")

            # Run stabilization pipeline
            log.info("Computing stabilization...")
            mgr.recompute_blocking()

            # Determine output path
            if args.output:
                output = args.output
            elif info.get("image_sequence"):
                output = sequence_output_stem(path) + "_stabilized.mp4"
            else:
                base = path.rsplit(".", 1)
                output = base[0] + "_stabilized." + (base[1] if len(base) > 1 else "mp4")

            # Render
            log.info("Rendering to: %s", output)
            mgr.render(
                path,
                output,
                {
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
                },
            )
            log.info("Done: %s", output)

        except Exception as exc:
            log.error("Failed to process %s: %s", path, exc)
            if args.verbose:
                import traceback
                traceback.print_exc()
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
