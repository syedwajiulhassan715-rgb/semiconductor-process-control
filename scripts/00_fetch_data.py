"""Download the UCI SECOM dataset into data/raw/.

Source: McCann, M. & Johnston, A. (2008). SECOM [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C54305
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom"
FILES = ("secom.data", "secom_labels.data", "secom.names")
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = RAW / name
        if dest.exists():
            print(f"skip  {name} (already present)")
            continue
        print(f"fetch {name} ...", end=" ", flush=True)
        urllib.request.urlretrieve(f"{BASE}/{name}", dest)
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()[:16]
        print(f"{dest.stat().st_size:,} bytes  sha256:{digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
