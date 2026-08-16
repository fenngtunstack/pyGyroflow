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
        action="store_true",
        help="Auto-sync gyro to video via optical flow before stabilizing",
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

    for path in args.input:
        log.info("Processing: %s", path)

        mgr = StabilizationManager()

        try:
            # Load video
            info = mgr.load_video(path)
            log.info(
                "Video: %dx%d @ %.2f fps, %.1f ms, %d frames",
                info["width"], info["height"],
                info["fps"], info["duration_ms"], info["frame_count"],
            )

            # Warn loudly if no gyro data was extracted — stabilization would
            # otherwise run on empty input and produce a "successful" but
            # un-stabilized video. This is the loud guard for the silent
            # telemetry-degradation path in manager.load_gyro_data.
            if not mgr.gyro.quaternions:
                log.warning(
                    "No gyro/IMU data found in %s. Output will NOT be "
                    "stabilized (only lens correction applies).", path,
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
                },
            )
            log.info("Done: %s", output)

        except Exception as exc:
            log.error("Failed to process %s: %s", path, exc)
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
