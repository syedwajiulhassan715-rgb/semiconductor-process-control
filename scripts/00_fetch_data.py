"""Download the UCI SECOM dataset into data/raw/.

Thin wrapper around ``secom fetch`` for use without installing the package.

Source: McCann, M. & Johnston, A. (2008). SECOM [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C54305
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from secom.cli import main

if __name__ == "__main__":
    sys.exit(main(["fetch", "--root", str(ROOT)]))
