#!/usr/bin/env python3
"""Run the synthetic click-track demo.

    PYTHONPATH=src .venv/bin/python scripts/make_demo.py [out_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from mainote.demo import run_demo


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out/demo")
    summary = run_demo(out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
