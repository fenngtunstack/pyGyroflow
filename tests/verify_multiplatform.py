"""Multi-platform CLI stabilization verification matrix.

Checks the full load -> parse -> lens-autoload -> recompute chain for
every brand the CLI claims to support (GoPro / DJI / Sony / Insta360),
printing one line per file:

    parse counts (raw IMU / quats), detected source, auto-matched lens,
    rolling-shutter readout, correction magnitude.

Files are optional: a row is skipped when its local path doesn't exist.
Add your own files via PYGYROFLOW_VERIFY_FILES="a.mp4,b.mp4".

Usage:
    python tests/verify_multiplatform.py
    python tests/verify_multiplatform.py --jitter OUT.mp4 IN.mp4   # pixel jitter only
"""

from __future__ import annotations

import os
import sys

import numpy as np


def _approx_eq(a: float, b: float, tol: float = 1e-5) -> bool:
    return abs(a - b) <= tol

DEFAULT_FILES = [
    # (label, path)
    ("GoPro Hero6 4k", "/home/ft/workspace/testvideos/extra-04-GoPro-Hero-6.MP4"),
    ("GoPro Hero8 ble", "/home/ft/workspace/testvideos/gpmf-05-hero6+ble.mp4"),
    ("GoPro Hero10", "/home/ft/workspace/testvideos/extra-07-GoPro-Hero-10.MP4"),
    ("GoPro HERO12 8:7", "/home/ft/workspace/PreReserach/msGyroFlow/GX010045.MP4"),
    ("DJI Osmo Nano", "/home/ft/workspace/PreReserach/msGyroFlow/DJI_20260507160359_0005_D.MP4"),
    ("Sony RX100M7", "/home/ft/workspace/PreReserach/msGyroFlow/sony-ois-only.MP4"),
    ("Sony a7sIII", "/home/ft/workspace/testvideos/issue-44-06-C0837-a7s3-tamron28-200-at-28mm.MP4"),
    ("Insta360 OneR", "/home/ft/workspace/testvideos/insta360-oner-preview.insv"),
]


def verify_parse(label: str, path: str) -> None:
    from pygyroflow.manager import StabilizationManager

    mgr = StabilizationManager()
    info = mgr.load_video(path)
    mgr.smoothing.current().set_parameter("smoothness", 0.5)
    mgr.recompute_blocking()

    sq = mgr.gyro.smoothed_quaternions
    ks = sorted(sq)
    if ks:
        corrs = np.degrees([2 * sq[k].angle() for k in ks[:: max(1, len(ks) // 200)]])
        corr = f"median |corr| {np.median(corrs):5.2f} deg (p90 {np.percentile(corrs, 90):5.2f})"
    else:
        corr = "NO GYRO DATA"

    lens = mgr.lens.get_display_name() if mgr.lens.calib_dimension.get("w") else "(none)"
    print(
        f"{label:<16} {info['width']:5d}x{info['height']:<5d}@{info['fps']:.2f} "
        f"raw={len(mgr.gyro.raw_imu):5d} quat={len(mgr.gyro.quaternions):5d} "
        f"{(mgr.gyro.file_metadata.detected_source or '?'):<18} "
        f"rs={mgr.params.frame_readout_time:6.2f}ms lens={lens[:52]} | {corr}"
    )


def main() -> None:
    if len(sys.argv) >= 4 and sys.argv[1] == "--jitter":
        sys.path.insert(0, os.path.dirname(__file__))
        from pixel_jitter import displacement_series

        for label, path in [("input", sys.argv[3]), ("output", sys.argv[2])]:
            t, disp = displacement_series(path)
            disp = np.array(disp)
            t = np.array(t)
            mid = (t > t[len(t) // 4]) & (t < t[3 * len(t) // 4])
            print(
                f"{label:<8} all: median {np.median(disp):.3f} p90 {np.percentile(disp, 90):.3f} "
                f"| mid: median {np.median(disp[mid]):.3f} p90 {np.percentile(disp[mid], 90):.3f}"
            )
        return

    files = list(DEFAULT_FILES)
    extra = os.environ.get("PYGYROFLOW_VERIFY_FILES")
    if extra:
        files += [("extra", p) for p in extra.split(",") if p]

    print(f"{'platform':<16} {'video':<16} {'raw/quat':<9} {'source':<15} {'rs':<8} lens | correction")
    for label, path in files:
        if not os.path.isfile(path):
            print(f"{label:<16} SKIP (no local file: {path})")
            continue
        try:
            verify_parse(label, path)
        except Exception as exc:  # noqa: BLE001 - report, don't crash the matrix
            print(f"{label:<16} FAIL: {exc}")

    # Insta360 fallback: when no real .insv is registered above, verify via
    # the synthetic file whose expected output was cross-validated against
    # upstream telemetry-parser (see TestInsta360Parsing).
    if not any("insta360-oner" in p for _l, p in files):
        try:
            sys.path.insert(0, os.path.dirname(__file__))
            import tempfile

            from test_e2e import build_synthetic_insta360_file

            with tempfile.NamedTemporaryFile(suffix=".insv", delete=False) as f:
                f.write(build_synthetic_insta360_file(raw_gyro=True, with_offsets_index=True))
                synthetic = f.name
            from pygyroflow.telemetry import parse_telemetry_file

            md = parse_telemetry_file(synthetic, fps=30.0)
            ok = (
                md.detected_source == "Insta360 OneR"
                and len(md.raw_imu) == 50
                and _approx_eq(md.raw_imu[0].gyro[0], 225.830989)
                and md.lens_profile is not None
            )
            os.unlink(synthetic)
            print(
                f"{'Insta360 OneR':<16} {'OK' if ok else 'FAIL'} (synthetic cross-check vs upstream telemetry-parser; "
                "add real footage via PYGYROFLOW_VERIFY_FILES)"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{'Insta360 OneR':<16} FAIL: {exc}")


if __name__ == "__main__":
    main()
