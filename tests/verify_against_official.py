# -*- coding: utf-8 -*-
"""Frame-by-frame verification against the official Gyroflow render.

Uses the official project file (.gyroflow, JSON) for HERO6 — which carries
the exact reproduction parameters (lens calibration, smoothing params,
3 sync offsets with real clock drift, output size) — re-renders the same
input through pyGyroFlow with those parameters, and compares every frame
against the official stabilized output (PSNR + mean abs diff).

Also reports the sync-offset comparison: our auto-sync vs the official
offsets, which quantifies the remaining sync gap separately from the
pipeline gap.

Usage:
    python tests/verify_against_official.py            # full check
    python tests/verify_against_official.py --sync-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np

log = logging.getLogger("verify_official")

PROJECT = "/home/ft/workspace/testvideos/extra-03-GoPro-Hero-6.gyroflow"
OFFICIAL_MP4 = "/home/ft/workspace/testvideos/extra-02-GoPro-Hero-6_stabilized.mp4"
INPUT_MP4 = "/home/ft/workspace/testvideos/extra-04-GoPro-Hero-6.MP4"
OUR_RENDER = "/tmp/verify_official_ours.mp4"


def load_project_params() -> dict:
    d = json.load(open(PROJECT, encoding="utf-8"))
    return d


def configure_manager(d: dict, offsets: dict[int, float] | None):
    from pygyroflow.manager import StabilizationManager
    from pygyroflow.lens import LensProfile

    mgr = StabilizationManager()
    # integration before load (parse-time)
    mgr.load_video(INPUT_MP4)

    # lens from the project's calibration block via the standard parser
    cal = d["calibration_data"]
    lens = LensProfile.from_json(cal)
    mgr.lens = lens

    stab = d["stabilization"]
    for sp in stab.get("smoothing_params", []):
        try:
            mgr.smoothing.current().set_parameter(sp["name"], sp["value"])
        except Exception as exc:  # unknown param -> report, continue
            log.warning("smoothing param %s not supported: %s", sp["name"], exc)
    mgr.set_adaptive_zoom(float(stab.get("adaptive_zoom_window", 4.0)))
    mgr.set_lens_correction_amount(float(stab.get("lens_correction_amount", 1.0)))
    mgr.set_frame_readout_time(float(stab.get("frame_readout_time") or 0.0))

    # sync offsets from the project (keys are us timestamps) unless overridden
    if offsets is None:
        offsets = {int(k): float(v) for k, v in d.get("offsets", {}).items()}
    if offsets:
        mgr.gyro.set_offsets(offsets)
    return mgr


def render_ours(d: dict, offsets: dict[int, float] | None) -> None:
    mgr = configure_manager(d, offsets)
    mgr.recompute_blocking()
    out = d.get("output", {})
    w = int(out.get("output_width") or mgr.params.size[0])
    h = int(out.get("output_height") or mgr.params.size[1])
    mgr.set_output_size(w, h)
    log.info("Rendering ours at %dx%d ...", w, h)
    mgr.render(INPUT_MP4, OUR_RENDER,
               {"codec": "H.264/AVC", "bitrate": 0, "use_gpu": False, "audio": False})


def frame_psnr_series(a_path: str, b_path: str) -> tuple[np.ndarray, np.ndarray]:
    import av
    import cv2

    ca, cb = av.open(a_path), av.open(b_path)
    sa, sb = ca.streams.video[0], cb.streams.video[0]
    psnrs, mads = [], []
    ita, itb = ca.demux(sa), cb.demux(sb)
    fa = fb = None
    while True:
        if fa is None:
            for pkt in ita:
                frames = pkt.decode()
                if frames:
                    fa = frames[0].to_ndarray(format="gray")
                    break
        if fb is None:
            for pkt in itb:
                frames = pkt.decode()
                if frames:
                    fb = frames[0].to_ndarray(format="gray")
                    break
        if fa is None or fb is None:
            break
        if fa.shape != fb.shape:
            fb = cv2.resize(fb, (fa.shape[1], fa.shape[0]), interpolation=cv2.INTER_AREA)
        mse = np.mean((fa.astype(np.float64) - fb.astype(np.float64)) ** 2)
        psnrs.append(10.0 * np.log10(255.0 ** 2 / max(mse, 1e-9)))
        mads.append(np.mean(np.abs(fa.astype(np.float64) - fb.astype(np.float64))))
        fa = fb = None
    ca.close(); cb.close()
    return np.array(psnrs), np.array(mads)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync-only", action="store_true", help="skip the full render")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    d = load_project_params()
    official_offsets = {int(k): float(v) for k, v in d.get("offsets", {}).items()}

    print("=== official sync offsets (3-point, real drift) ===")
    for k in sorted(official_offsets):
        print(f"  t={k/1e6:7.2f}s  offset={official_offsets[k]:8.3f} ms")

    # our auto-sync on the same file for comparison
    from pygyroflow.manager import StabilizationManager
    m = StabilizationManager()
    m.load_video(INPUT_MP4)
    m.smoothing.current().set_parameter("smoothness", 0.5)
    ours = m.synchronize()
    print(f"our auto-sync global offset: {ours:.2f} ms "
          f"(official points span {min(official_offsets.values()):.2f}..{max(official_offsets.values()):.2f})")

    if args.sync_only:
        return 0

    render_ours(d, offsets=None)  # use the official offsets for the render
    psnrs, mads = frame_psnr_series(OUR_RENDER, OFFICIAL_MP4)
    print(f"=== frame-by-frame vs official ({len(psnrs)} frames) ===")
    print(f"PSNR  median {np.median(psnrs):6.2f} dB | p5 {np.percentile(psnrs,5):6.2f} | min {psnrs.min():6.2f}")
    print(f"MAD   median {np.median(mads):6.2f}    | p95 {np.percentile(mads,95):6.2f}")
    print("(PSNR >30dB ≈ visually near-identical; 25-30 ≈ same stabilization, "
          "resampling/encode differences)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
