"""``python -m pygyroflow`` — mirrors the ``pygyroflow`` console script.

The project's own docs advertise ``python -m pygyroflow`` as the way to run
the CLI, but without this the package is not executable and it fails with
"No module named pygyroflow.__main__".
"""

from __future__ import annotations

from pygyroflow.cli.main import main

if __name__ == "__main__":
    main()
