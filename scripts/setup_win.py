#!/usr/bin/env python3
"""Windows one-shot setup for the MaiForge Web UI.

Run from the repository root (PowerShell or cmd):

    python scripts\\setup_win.py
    python scripts\\setup_win.py --check          # detect only
    python scripts\\setup_win.py --host 0.0.0.0   # expose to the LAN

The shared implementation lives in ``setup_common.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from setup_common import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(platform="win", script_path=Path(__file__).resolve()))
