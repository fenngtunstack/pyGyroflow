"""Windowed stabilization quality analysis (manual tool).

Splits a video into 1-second windows and reports per-window jitter
(median |d(omega)| between consecutive frame pairs, deg/s per frame step).
Locates segments the user reports as "still shaky" (e.g. ~8s +3s).

Usage:
    python tests/verify_windows.py video1.mp4 [video2.mp4 ...]
"""

from __future__ import annotations

import sys

import numpy as np

from tests.verify_stabilization import angular_speed_series


def window_metrics(speeds: np.ndarray, fps: float, win_s: float = 1.0):
    """Yield (t_start_s, median_jitter, median_speed, n) per window."""
    step = max(1, int(round(win_s * fps)))
    for start in range(0, len(speeds), step):
        chunk = speeds[start : start + step]
        if len(chunk) < 8:
            continue
        jit = np.abs(np.diff(chunk))
        # drop essential-matrix outliers (>=90 deg/s jumps) from both stats
        ok = chunk < 90.0
        yield (
            start / fps,
            float(np.median(jit[jit < 45.0])) if np.any(jit < 45.0) else float("nan"),
            float(np.median(chunk[ok])) if np.any(ok) else float("nan"),
            len(chunk),
        )


def main(argv: list[str]) -> int:
    for path in argv[1:]:
        speeds, fps = angular_speed_series(path)
        print(f"\n=== {path}  (fps={fps:.2f}, frames={len(speeds)}) ===")
        print("  t(s)   medjit  medspeed")
        rows = list(window_metrics(speeds, fps))
        # global median for context
        alljit = [r[1] for r in rows if r[1] == r[1]]
        gmed = float(np.median(alljit)) if alljit else 0.0
        for t, mj, ms, n in rows:
            flag = "  <-- BAD" if mj == mj and mj > 3 * gmed and mj > 0.5 else ""
            print(f"{t:6.1f}  {mj:7.3f}  {ms:8.2f}{flag}")
        print(f"  (global window-median medjit = {gmed:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
