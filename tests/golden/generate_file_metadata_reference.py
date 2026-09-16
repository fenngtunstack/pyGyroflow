"""Regenerate tests/golden/cbor_file_metadata.json.

The expected bytes are **ciborium's own output** — the crate Gyroflow uses to
write these payloads — not the Python port's. So this file does not generate
the fixture by running Python; it drives the reference program and keeps it in
sync.

How the reference is built (see ``file_metadata_reference.rs`` next to this
file, and ``file_metadata_reference.Cargo.toml``):

    mkdir -p /tmp/fmref/src
    cp file_metadata_reference.Cargo.toml /tmp/fmref/Cargo.toml
    cp file_metadata_reference.rs        /tmp/fmref/src/main.rs
    cd /tmp/fmref && cargo build --release --offline
    /tmp/fmref/target/release/fmref <repo>/tests/golden/cbor_file_metadata.json

The program reads the fixture's ``input`` values, encodes each with ciborium,
and writes the bytes back into the case as ``cbor``. ``--check`` reports
differences instead of writing, which is what this script runs:

    python tests/golden/generate_file_metadata_reference.py --check /tmp/fmref

The struct definitions in the reference are copied field-for-field from
``opensource/gyroflow/src/core/gyro_source/file_metadata.rs``,
``core/camera_identifier.rs`` and vendor/telemetry-parser's ``util.rs``, with
only these substitutions (nalgebra is not available in a scratch crate):

    Quat64                   -> [f64; 4]
    Vector3<f64>             -> [f64; 3]
    CatmullRom<Vector3<f64>> -> CatmullRom<[f64; 3]>

``serde_json`` is built with ``preserve_order`` — matching what a real export
shows, where a lens profile is *not* alphabetically ordered.

Not committed: the Cargo project itself. It is two files and one command, and
vendoring a second copy of upstream's structs under a different name would
invite someone to edit the wrong one.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

GOLDEN_DIR = Path(__file__).parent
FIXTURE = GOLDEN_DIR / "cbor_file_metadata.json"
REFERENCE_SOURCE = GOLDEN_DIR / "file_metadata_reference.rs"
REFERENCE_CARGO = GOLDEN_DIR / "file_metadata_reference.Cargo.toml"


def main() -> int:
    args = sys.argv[1:]
    check = "--check" in args
    rest = [a for a in args if a != "--check"]

    if not rest:
        print(__doc__)
        print(f"fixture:   {FIXTURE}")
        print(f"reference: {REFERENCE_SOURCE}")
        print(f"cargo:     {REFERENCE_CARGO}")
        with open(FIXTURE, encoding="utf-8") as handle:
            cases = json.load(handle)["cases"]
        print(f"{len(cases)} case(s) in the fixture")
        return 0

    binary = Path(rest[0]) / "target" / "release" / "fmref"
    if not binary.is_file():
        print(f"no reference binary at {binary}", file=sys.stderr)
        print("build it as described above", file=sys.stderr)
        return 2

    command = [str(binary), str(FIXTURE)]
    if check:
        command.append("--check")
    result = subprocess.run(command, capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
