"""Regenerate tests/golden/undistort_points.json.

The expected values are upstream Rust output, not the Python port's.

The subject is ``cpu_undistort.rs::undistort_points`` — the function every
points-path caller funnels through (the FOV polygon, ``almeida``, the optical
flow estimators). It is the one part of the D-06 family that had no external
reference: the port's own tests are structural, which catches "the stages are
wired in the wrong order" but not "the arithmetic is upstream's".

Unlike the other references in this directory this one needs more than two
files, so the staging is scripted rather than copy-pasted. Nothing derived from
upstream is committed: ``--stage`` pulls it out of the read-only
``opensource/gyroflow/`` tree on demand, so there is no second copy to drift.

    python tests/golden/generate_undistort_points_reference.py --stage /tmp/upref
    cd /tmp/upref && cargo build --release --offline
    python tests/golden/generate_undistort_points_reference.py --fill /tmp/upref

``--fill`` runs the binary against the fixture's inputs and writes back the
outputs; ``--check`` runs it against the committed outputs and reports
differences instead. Both need the binary built first.

What ``--stage`` copies, and why each is verbatim:

* ``src/cpu_undistort_points.rs`` — lines 649-803 of ``cpu_undistort.rs``, i.e.
  the ``// Ported from OpenCV`` comment through the end of the file. Slice
  boundaries are found by content, not by hard-coded line numbers.
* ``src/distortion_models/`` — the whole directory. The model dispatch is part
  of what is being tested, and the models are small.
* ``src/splines.rs`` — the mesh branch calls into ``BivariateSpline``.

The shims that stand in for the rest of the crate live in the driver, not
here; see its module comment for what each one drops.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

GOLDEN_DIR = Path(__file__).parent
REPO = GOLDEN_DIR.parent.parent
UPSTREAM = REPO.parent / "opensource" / "gyroflow" / "src" / "core"
FIXTURE = GOLDEN_DIR / "undistort_points.json"

DRIVER = GOLDEN_DIR / "undistort_points_reference_main.rs"
CARGO = GOLDEN_DIR / "undistort_points_reference.Cargo.toml"

CPU_UNDISTORT = UPSTREAM / "stabilization" / "cpu_undistort.rs"
MODELS_DIR = UPSTREAM / "stabilization" / "distortion_models"
SPLINES = UPSTREAM / "gyro_source" / "splines.rs"

# The slice runs from this comment to the end of the file. Matching on content
# rather than a line number means an upstream edit above it cannot silently
# shift the window onto a different function.
SLICE_START = "// Ported from OpenCV:"

# ---------------------------------------------------------------------------
# Case inputs.
#
# The matrices live here, computed in Python, so the skeleton is reproducible
# and reviewable; only the two output arrays come from the Rust. Each case name
# says which branch of the copied function it is pinning.
# ---------------------------------------------------------------------------

W, H = 1920, 1080
FX, FY, CX, CY = 1000.0, 1000.0, 960.0, 540.0

IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _k(fx=FX, fy=FY, cx=CX, cy=CY):
    return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]


def _rot(yaw_deg, pitch_deg, roll_deg):
    """A small rotation matrix, written out so the fixture is self-contained."""
    import math

    y, p, r = (math.radians(a) for a in (yaw_deg, pitch_deg, roll_deg))
    cy_, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    return [
        [cy_ * cp, cy_ * sp * sr - sy * cr, cy_ * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy_ * cr, sy * sp * cr - cy_ * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _border(n=24):
    """Points around a 1920x1080 frame, the shape the FOV polygon samples."""
    return [
        [round(x, 6), round(y, 6)]
        for x, y in (
            [(W * i / n, 0.0) for i in range(n)]
            + [(W, H * i / n) for i in range(n)]
            + [(W * (n - i) / n, H) for i in range(n)]
            + [(0.0, H * (n - i) / n) for i in range(n)]
        )
    ]


def _focal_plane_mesh(offset=11, slope_x=0.8, slope_y=0.3, length=120):
    """A mesh buffer with a focal-plane table at *offset*.

    Layout (see `gyro_source/sony.rs`): 9 header words, then the mesh
    coefficients, then the focal-plane pairs. Word 0 is the offset to the
    table; words 1-2 are the grid dimensions; 3-4 the mesh size; 5-6 the
    origin; 7-8 the crop size.
    """
    mesh = [0.0] * length
    mesh[0] = float(offset)
    mesh[1], mesh[2] = 3.0, 3.0
    mesh[3], mesh[4] = float(W), float(H)
    mesh[5], mesh[6] = 0.0, 0.0
    mesh[7], mesh[8] = float(W), float(H)
    mesh[offset] = 1.0  # non-zero: the table is present
    for index in range(8):
        mesh[offset + 4 + index * 2 + 0] = slope_x
        mesh[offset + 4 + index * 2 + 1] = slope_y
    return mesh


def _full_mesh(length=120):
    """A mesh with `mesh[0] > 10`, so both mesh branches run.

    The coefficients are small but non-zero, so the spline solve produces a
    non-trivial displacement rather than silently returning the input.
    """
    mesh = _focal_plane_mesh(length=length)
    base = 9 + 3 * 3 * 2
    for index in range(base, base + 3 * 4 * 3 * 2):
        mesh[index] = 0.02 * ((index % 7) - 3)
    return mesh


def _cases():
    """(name, inputs, ...) — see `skeleton()` for how they are laid out."""
    fisheye = [-0.1, 0.02, 0.001, 0.0] + [0.0] * 8
    standard = [0.1, -0.05, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    pts = [(CX, CY), (CX + FX, CY), (CX, CY + FY), (300.0, 200.0), (1700.0, 900.0)]

    return [
        # -- the basics ----------------------------------------------------
        {
            "name": "identity_no_distortion",
            "params": {},
            "distorted": pts,
            "camera_matrix": _k(),
            "distortion_coeffs": [0.0] * 12,
            "rotation": IDENTITY,
        },
        {
            "name": "fisheye_distortion",
            "params": {"distortion_model": "opencv_fisheye"},
            "distorted": pts,
            "camera_matrix": _k(),
            "distortion_coeffs": fisheye,
            "rotation": IDENTITY,
        },
        {
            "name": "opencv_standard_distortion",
            "params": {"distortion_model": "opencv_standard"},
            "distorted": pts,
            "camera_matrix": _k(),
            "distortion_coeffs": standard,
            "rotation": IDENTITY,
        },
        {
            "name": "poly3_distortion",
            "params": {"distortion_model": "poly3"},
            "distorted": pts,
            "camera_matrix": _k(),
            "distortion_coeffs": [-0.05] + [0.0] * 11,
            "rotation": IDENTITY,
        },
        {
            "name": "ptlens_distortion",
            "params": {"distortion_model": "ptlens"},
            "distorted": pts,
            "camera_matrix": _k(),
            "distortion_coeffs": [-0.08, 0.03, -0.01] + [0.0] * 9,
            "rotation": IDENTITY,
        },
        # -- the rotation arguments ----------------------------------------
        {
            "name": "rotation_present",
            "params": {},
            "distorted": pts,
            "camera_matrix": _k(),
            "distortion_coeffs": [0.0] * 12,
            "rotation": _rot(2.0, -1.5, 0.7),
        },
        {
            "name": "p_is_premultiplied",
            "params": {},
            "distorted": pts,
            "camera_matrix": _k(),
            "distortion_coeffs": [0.0] * 12,
            "rotation": _rot(1.0, 0.5, 0.0),
            # A non-uniform p survives the perspective divide, unlike c*I.
            "p": _k(1400.0, 1000.0, 960.0, 540.0),
        },
        {
            "name": "uniform_p_cancels",
            "params": {},
            "distorted": pts,
            "camera_matrix": _k(),
            "distortion_coeffs": [0.0] * 12,
            "rotation": _rot(1.0, 0.5, 0.0),
            "p": [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        },
        {
            "name": "rot_per_point_overrides",
            "params": {},
            "distorted": [(CX + FX, CY), (CX + FX, CY), (CX + FX, CY)],
            "camera_matrix": _k(),
            "distortion_coeffs": [0.0] * 12,
            "rotation": _k(),
            "rot_per_point": [_k(), _rot(3.0, 0.0, 0.0), _rot(0.0, 0.0, 3.0)],
        },
        # -- lens correction strength --------------------------------------
        {
            "name": "lens_correction_full",
            "params": {},
            "distorted": pts,
            "camera_matrix": _k(),
            "distortion_coeffs": fisheye,
            "rotation": IDENTITY,
            "lens_correction_amount": 1.0,
        },
        {
            "name": "lens_correction_half",
            "params": {},
            "distorted": pts,
            "camera_matrix": _k(),
            "distortion_coeffs": fisheye,
            "rotation": IDENTITY,
            "lens_correction_amount": 0.5,
        },
        {
            "name": "lens_correction_none",
            "params": {},
            "distorted": pts,
            "camera_matrix": _k(),
            "distortion_coeffs": fisheye,
            "rotation": IDENTITY,
            "lens_correction_amount": 0.0,
        },
        # -- light refraction ----------------------------------------------
        {
            "name": "light_refraction_from_params",
            "params": {"light_refraction_coefficient": 1.5},
            "distorted": pts,
            "camera_matrix": _k(),
            "distortion_coeffs": fisheye,
            "rotation": IDENTITY,
        },
        {
            "name": "light_refraction_from_keyframe",
            "params": {
                "light_refraction_coefficient": 1.5,
                "light_refraction_keyframes": [[0.0, 1.2], [100.0, 0.8]],
            },
            "distorted": pts,
            "camera_matrix": _k(),
            "distortion_coeffs": fisheye,
            "rotation": IDENTITY,
            "timestamp_ms": 100.0,
        },
        # -- input stretch -------------------------------------------------
        {
            "name": "input_stretch",
            "params": {
                "input_horizontal_stretch": 0.75,
                "input_vertical_stretch": 1.25,
            },
            "distorted": pts,
            "camera_matrix": _k(),
            "distortion_coeffs": fisheye,
            "rotation": IDENTITY,
            "lens_correction_amount": 0.4,
            # The (960, 1540) probe sits at tan(theta) ~ -4.9, within a few
            # hundredths of a radian of the tangent pole, where d(tan)/d(theta)
            # is ~25. The fisheye inversion resolves theta to about 1e-5 in
            # f32 against the port's f64, and the pole multiplies that by 25 —
            # so this probe measures tan's conditioning, not the stretch. The
            # other four probes stay at the default bound.
            "tolerance": 1e-3,
        },
        # -- IBIS ----------------------------------------------------------
        {
            "name": "shift_per_point",
            "params": {},
            "distorted": pts,
            "camera_matrix": _k(),
            "distortion_coeffs": fisheye,
            "rotation": IDENTITY,
            "shift_per_point": [
                [1.5, -2.0, 0.0, 0.5, 0.5],
                [-1.0, 3.0, 0.002, -1.0, 2.0],
                [0.0, 0.0, -0.0015, 0.0, 0.0],
                [4.0, 4.0, 0.01, 1.0, -1.0],
                [-3.5, 0.25, 0.0, 0.0, 0.0],
            ],
        },
        # -- the two mesh branches -----------------------------------------
        {
            "name": "mesh_focal_plane_only",
            "params": {},
            "distorted": pts,
            # offset 9 keeps `mesh[0]` under 10, so only the focal-plane
            # table runs; `mesh[9]` is its presence flag.
            "camera_matrix": _k(),
            "distortion_coeffs": fisheye,
            "rotation": IDENTITY,
            "mesh": _focal_plane_mesh(offset=9),
        },
        {
            "name": "mesh_full",
            "params": {},
            "distorted": pts,
            "camera_matrix": _k(),
            "distortion_coeffs": fisheye,
            "rotation": IDENTITY,
            "mesh": _full_mesh(),
        },
        # -- digital lenses ------------------------------------------------
        {
            "name": "digital_lens_superview",
            "params": {"digital_lens": "gopro_superview"},
            "distorted": pts,
            "camera_matrix": _k(),
            "distortion_coeffs": [0.0] * 12,
            "rotation": IDENTITY,
            # amount < 1 is what reaches the 0.91 x correction.
            "lens_correction_amount": 0.3,
        },
        {
            "name": "digital_lens_hyperview",
            "params": {"digital_lens": "gopro_hyperview"},
            "distorted": pts,
            "camera_matrix": _k(),
            "distortion_coeffs": [0.0] * 12,
            "rotation": IDENTITY,
            "lens_correction_amount": 0.3,
        },
        {
            "name": "digital_lens_superview_full_correction",
            "params": {"digital_lens": "gopro_superview"},
            "distorted": pts,
            "camera_matrix": _k(),
            "distortion_coeffs": fisheye,
            "rotation": IDENTITY,
            "lens_correction_amount": 1.0,
        },
        # -- the sentinel ---------------------------------------------------
        {
            "name": "non_converging_point_returns_sentinel",
            "params": {},
            "distorted": [(CX, CY), (CX + FX, CY), (CX + 100.0, CY), (CX + 2 * FX, CY)],
            "camera_matrix": _k(),
            "distortion_coeffs": [1000000.0] + [0.0] * 11,
            "rotation": IDENTITY,
        },
        # -- geometry the FOV path actually produces -----------------------
        {
            "name": "output_size_differs",
            "params": {"output_width": 960, "output_height": 540},
            "distorted": pts,
            "camera_matrix": _k(),
            "distortion_coeffs": fisheye,
            "rotation": _rot(0.5, 0.5, 0.5),
        },
        {
            "name": "frame_border_polygon",
            "params": {"distortion_model": "opencv_fisheye"},
            "distorted": _border(),
            "camera_matrix": _k(),
            "distortion_coeffs": fisheye,
            "rotation": _rot(1.0, 2.0, 0.0),
            "lens_correction_amount": 1.0,
        },
        {
            "name": "frame_border_polygon_no_correction",
            "params": {"distortion_model": "opencv_fisheye"},
            "distorted": _border(),
            "camera_matrix": _k(),
            "distortion_coeffs": fisheye,
            "rotation": _rot(1.0, 2.0, 0.0),
            "lens_correction_amount": 0.0,
        },
    ]


PARAM_DEFAULTS = {
    "width": W,
    "height": H,
    "output_width": W,
    "output_height": H,
    "light_refraction_coefficient": 1.0,
    "input_horizontal_stretch": 1.0,
    "input_vertical_stretch": 1.0,
    "distortion_model": "opencv_fisheye",
    "digital_lens": None,
    "light_refraction_keyframes": [],
}


def skeleton() -> dict:
    """The fixture's inputs, with `expected` left empty for the Rust to fill."""
    cases = []
    for case in _cases():
        params = dict(PARAM_DEFAULTS)
        params.update(case["params"])
        entry = {
            "name": case["name"],
            "params": params,
            "distorted": case["distorted"],
            "camera_matrix": case["camera_matrix"],
            "distortion_coeffs": case["distortion_coeffs"],
            "rotation": case["rotation"],
            "expected": [],
        }
        for key in (
            "p",
            "rot_per_point",
            "lens_correction_amount",
            "timestamp_ms",
            "shift_per_point",
            "mesh",
            "tolerance",
        ):
            if key in case:
                entry[key] = case[key]
        cases.append(entry)
    return {
        "description": (
            "undistort_points reference: upstream cpu_undistort.rs run verbatim"
        ),
        "_provenance": (
            "Generated by tests/golden/generate_undistort_points_reference.py. The "
            "'expected' values are upstream Rust output, not the Python port's. "
            "Upstream computes in f32 and the port in f64, so compare with a "
            "relative tolerance (the reference binary's --check uses 1e-5 over a "
            "1.0 floor)."
        ),
        "cases": cases,
    }


def _slice_upstream() -> str:
    text = CPU_UNDISTORT.read_text(encoding="utf-8")
    marker = text.index(SLICE_START)
    # Back up to the start of the line the marker sits on.
    start = text.rindex("\n", 0, marker) + 1
    body = text[start:]
    if "pub fn undistort_points(" not in body:
        raise SystemExit("slice does not contain undistort_points; upstream moved?")
    top_level = body.count("\npub fn ")
    if top_level != 1:
        raise SystemExit(
            f"slice holds {top_level} top-level function(s), expected only "
            "undistort_points; it is no longer the tail of the file"
        )
    return body


def stage(target: Path) -> int:
    src = target / "src"
    if target.exists():
        shutil.rmtree(target)
    (src / "distortion_models").mkdir(parents=True)

    shutil.copyfile(CARGO, target / "Cargo.toml")
    shutil.copyfile(DRIVER, src / "main.rs")
    (src / "cpu_undistort_points.rs").write_text(_slice_upstream(), encoding="utf-8")
    shutil.copyfile(SPLINES, src / "splines.rs")
    # Every file, not just the .rs ones: each model `include_str!`s its own
    # `.cl` and `.wgsl` shader source, and a missing one is a compile error.
    for model in sorted(MODELS_DIR.iterdir()):
        if model.is_file():
            shutil.copyfile(model, src / "distortion_models" / model.name)

    body = (src / "cpu_undistort_points.rs").read_text(encoding="utf-8")
    staged = list((src / "distortion_models").iterdir())
    print(f"staged {target}")
    print(f"  cpu_undistort_points.rs: {len(body.splitlines())} lines")
    print(f"  distortion_models/: {len(staged)} files")
    print(f"build with: cd {target} && cargo build --release --offline")
    return 0


def run_binary(target: Path, check: bool) -> int:
    binary = target / "target" / "release" / "upref"
    if not binary.is_file():
        print(f"no reference binary at {binary}", file=sys.stderr)
        print("run --stage then build it", file=sys.stderr)
        return 2
    command = [str(binary), str(FIXTURE)]
    if check:
        command.append("--check")
    result = subprocess.run(command, capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        print(f"fixture: {FIXTURE}")
        if FIXTURE.is_file():
            with open(FIXTURE, encoding="utf-8") as handle:
                cases = json.load(handle)["cases"]
            print(f"{len(cases)} case(s), {sum(len(c['distorted']) for c in cases)} point(s)")
        return 0

    if "--skeleton" in args:
        with open(FIXTURE, "w", encoding="utf-8") as handle:
            json.dump(skeleton(), handle, indent=2)
            handle.write("\n")
        print(f"wrote inputs to {FIXTURE} (expected values still empty)")
        return 0

    if "--stage" in args:
        index = args.index("--stage")
        return stage(Path(args[index + 1]))

    if "--fill" in args:
        index = args.index("--fill")
        return run_binary(Path(args[index + 1]), check=False)

    if "--check" in args:
        index = args.index("--check")
        return run_binary(Path(args[index + 1]), check=True)

    print("expected one of --skeleton / --stage / --fill / --check", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
