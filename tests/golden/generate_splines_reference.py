"""Regenerate tests/golden/splines.json.

The expected values are upstream Rust output, not the Python port's — see
``splines_reference.rs`` (a verbatim copy of
``opensource/gyroflow/src/core/gyro_source/splines.rs``) and
``splines_reference_main.rs`` (the driver). To reproduce:

    mkdir -p /tmp/splref/src
    cp tests/golden/splines_reference.Cargo.toml /tmp/splref/Cargo.toml
    cp tests/golden/splines_reference.rs       /tmp/splref/src/splines.rs
    cp tests/golden/splines_reference_main.rs  /tmp/splref/src/main.rs
    cd /tmp/splref && cargo build --release --offline
    /tmp/splref/target/release/splref <repo>/tests/golden/splines.json

The driver reads each case's ``input``, evaluates it with the real
implementation and writes the results back as ``expected``. ``--check``
reports differences instead of writing:

    python tests/golden/generate_splines_reference.py --check /tmp/splref

The mesh buffers are stored in the fixture verbatim rather than regenerated
from a formula: both sides would have to agree on the formula, and a shared
mistake there would be invisible.

`--check` compares with a 1e-12 relative tolerance rather than for equality,
because the Rust reference is not bit-reproducible across builds: recompiling
the same source shifts a handful of these values by one ULP (a debug and a
release build disagree too). The Python port reproduces the committed
fixture's values exactly; a *regenerated* fixture may differ in the last bit.

Not committed: the Cargo project. Two files and one command, and a second copy
of upstream's spline code under another name invites editing the wrong one.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

GOLDEN_DIR = Path(__file__).parent
FIXTURE = GOLDEN_DIR / "splines.json"


def main() -> int:
    args = sys.argv[1:]
    check = "--check" in args
    rest = [a for a in args if a != "--check"]

    if not rest:
        print(__doc__)
        print(f"fixture:   {FIXTURE}")
        with open(FIXTURE, encoding="utf-8") as handle:
            cases = json.load(handle)["cases"]
        print(f"{len(cases)} case(s) in the fixture")
        return 0

    binary = Path(rest[0]) / "target" / "release" / "splref"
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
