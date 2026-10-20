"""Runs the full analysis: raw SECOM data -> figures, results.json and Power BI CSVs.

Thin wrapper around ``secom run`` for use without installing the package.
From the repo root:

    python scripts/00_fetch_data.py
    python scripts/run_analysis.py [--no-figures] [--figure-format svg] ...

The numbers in reports/ENGINEERING_REPORT.md come from reports/results.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from secom.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", "--root", str(ROOT), *sys.argv[1:]]))
