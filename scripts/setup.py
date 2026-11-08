#!/usr/bin/env python3
"""Choose the Linux or Windows setup script for the current OS.

Run from the repository root:

    python scripts/setup.py
    python scripts/setup.py --check
    python scripts/setup.py --host 0.0.0.0

Linux users can run ``scripts/setup_linux.py`` and Windows users
``scripts/setup_win.py`` directly.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    scripts = Path(__file__).resolve().parent
    if os.name == "nt":
        target = scripts / "setup_win.py"
    elif sys.platform.startswith("linux"):
        target = scripts / "setup_linux.py"
    else:
        raise SystemExit("[setup] 只支持 Linux 和 Windows。")
    try:
        return subprocess.run(
            [sys.executable, str(target), *sys.argv[1:]],
            check=False,
        ).returncode
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
