#!/usr/bin/env python3
"""Linux one-shot setup for the MaiForge Web UI.

Run from the repository root:

    python3 scripts/setup_linux.py
    python3 scripts/setup_linux.py --check          # detect only
    python3 scripts/setup_linux.py --host 0.0.0.0   # expose to the LAN

The shared implementation lives in ``setup_common.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from setup_common import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(platform="linux", script_path=Path(__file__).resolve()))
