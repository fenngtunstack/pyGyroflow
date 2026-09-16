"""Regenerate tests/golden/focal_length_smoothing.json.

Unlike generate_references.py in this directory, the expected values here are
NOT produced by the Python port — that would be circular. They come from the
upstream Rust implementation, executed verbatim:

1. Copy ``opensource/gyroflow/src/core/smoothing/focal_length.rs`` into a
   scratch crate's ``src/focal_length.rs``, unchanged.
2. Add a ``main.rs`` that reads ``<op> <args> | <values>`` lines on stdin and
   prints the result as a JSON array at 17 significant digits (which
   round-trips an f64 exactly), plus a workspace ``focal_length`` module
   declaration.
3. Feed it the case list below, then fold the output into the JSON.

The scratch crate is deliberately not committed: it is ~30 lines of glue and
would be a second copy of upstream code in the tree. What matters is that the
function bodies were never retyped — a transcription slip would hide a port
bug, which is exactly what this fixture exists to catch.

Usage (from the repo root)::

    python tests/golden/generate_focal_length_reference.py

Prints the Rust harness to stdout and writes the command script it expects on
stdin. Build and run the harness yourself; the upstream file is GPL and is not
redistributed here.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

GOLDEN = Path(__file__).parent / "focal_length_smoothing.json"

UPSTREAM = "opensource/gyroflow/src/core/smoothing/focal_length.rs"

MAIN_RS = r'''
mod focal_length;
use std::io::{self, Read};

fn parse_vals(line: &str) -> Vec<Option<f64>> {
    line.split_whitespace()
        .map(|t| if t == "x" { None } else { Some(t.parse::<f64>().unwrap()) })
        .collect()
}

fn dump(v: &[Option<f64>]) -> String {
    let parts: Vec<String> = v.iter().map(|x| match x {
        Some(f) => format!("{:.17e}", f),
        None => "null".to_string(),
    }).collect();
    format!("[{}]", parts.join(","))
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    for line in input.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        let mut it = line.splitn(2, '|');
        let head = it.next().unwrap().trim();
        let vals = parse_vals(it.next().unwrap_or("").trim());
        let a: Vec<&str> = head.split_whitespace().collect();
        let out = match a[0] {
            "gaussian" => focal_length::smooth_focal_lengths_gaussian(
                &vals, a[1].parse().unwrap(), a[2].parse().unwrap()),
            "adaptive" => focal_length::smooth_focal_lengths_adaptive(
                &vals, a[1].parse().unwrap(), a[2].parse().unwrap(),
                a[3].parse().unwrap(), a[4].parse().unwrap()),
            other => panic!("unknown op {other}"),
        };
        println!("{}", dump(&out));
    }
}
'''


def stair(n: int, start: float = 18.0, step: float = 1.0, per: int = 7) -> list[float]:
    """Camera-quantized focal length: holds, then steps."""
    return [start + step * (i // per) for i in range(n)]


def ramp(n: int, start: float = 18.0, end: float = 60.0) -> list[float]:
    """A deliberate zoom move."""
    return [start + (end - start) * i / (n - 1) for i in range(n)]


def build_cases() -> list[tuple]:
    cases: list[tuple] = []

    def gauss(name, vals, strength, window):
        cases.append(("gaussian", name, (strength, window), vals))

    def adapt(name, vals, fps, rate):
        cases.append(("adaptive", name, (fps,) + tuple(rate), vals))

    # The single-knob mapping upstream derives from `strength` (lib.rs).
    mid = (1.732, 0.1375, 3.0224)  # strength 0.5
    lo = (0.1, 0.05, 0.3)          # strength 0.0
    hi = (30.0, 0.40, 8.0)         # strength 1.0

    gauss("g_stairs_w5", stair(40), 1.0, 5)
    gauss("g_stairs_w9_half", stair(40), 0.5, 9)
    gauss("g_even_window", stair(20), 1.0, 4)
    gauss("g_zero_strength", stair(20), 0.0, 5)
    gauss("g_window_gt_n", stair(6), 1.0, 21)
    gauss("g_empty", [], 1.0, 5)
    gauss("g_gaps", [18.0, None, None, 18.5, 19.0, None, 19.0, 19.5, None, 20.0], 1.0, 5)
    gauss("g_all_none", [None] * 8, 1.0, 5)
    gauss("g_ramp", ramp(30), 1.0, 15)
    gauss("g_single", [24.0], 1.0, 5)

    rng = random.Random(7)
    jitter = [v + rng.uniform(-0.4, 0.4) for v in stair(60, step=0.5, per=12)]
    adapt("a_stairs_jitter", jitter, 30.0, mid)
    adapt("a_zoom_ramp", ramp(60), 30.0, mid)
    adapt("a_strength_0", jitter, 30.0, lo)
    adapt("a_strength_1", jitter, 30.0, hi)
    gaps = list(jitter)
    gaps[10:16] = [None] * 6
    gaps[0] = None
    gaps[-1] = None
    gaps[30] = None
    adapt("a_gaps", gaps, 30.0, mid)
    adapt("a_all_none", [None] * 12, 30.0, mid)
    adapt("a_one_value", [None] * 5 + [35.0] + [None] * 5, 30.0, mid)
    adapt("a_len1", [20.0], 30.0, mid)
    adapt("a_fps0", jitter, 0.0, mid)
    adapt("a_leading_gap", [None, None, None] + stair(17, step=2.0, per=6), 60.0, hi)
    adapt("a_flat", [20.0] * 25, 25.0, mid)
    adapt("a_big_step", [18.0] * 10 + [70.0] + [70.0] * 10, 30.0, mid)
    return cases


def command_script(cases) -> str:
    lines = []
    for kind, name, args, vals in cases:
        joined = " ".join(f"{x:.17g}" for x in args)
        data = " ".join("x" if v is None else repr(float(v)) for v in vals)
        lines.append(f"# {kind} {name}\n{kind} {joined} | {data}")
    return "\n".join(lines) + "\n"


def main() -> int:
    cases = build_cases()
    if "--emit-harness" in sys.argv:
        print("// src/focal_length.rs : copy " + UPSTREAM + " verbatim")
        print(MAIN_RS)
        return 0

    script = Path("focal_length_cases.txt")
    script.write_text(command_script(cases), encoding="utf-8")
    print(f"wrote {script} ({len(cases)} cases)")
    print("run:  flref < focal_length_cases.txt > focal_length_rust.json")
    print(f"then fold into {GOLDEN} and delete the scratch files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
