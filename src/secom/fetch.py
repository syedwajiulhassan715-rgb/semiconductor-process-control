"""Download the UCI SECOM dataset.

Source: McCann, M. & Johnston, A. (2008). SECOM [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C54305
"""

from __future__ import annotations

import hashlib
import urllib.request
from collections.abc import Callable
from pathlib import Path

BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom"
FILES = ("secom.data", "secom_labels.data", "secom.names")


def fetch(raw_dir: str | Path, *, log: Callable[[str], None] = print) -> list[Path]:
    """Download any missing SECOM files into ``raw_dir``. Returns all file paths.

    Files already present are skipped, so this is safe to rerun.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in FILES:
        dest = raw_dir / name
        paths.append(dest)
        if dest.exists():
            log(f"skip  {name} (already present)")
            continue
        urllib.request.urlretrieve(f"{BASE_URL}/{name}", dest)
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()[:16]
        log(f"fetch {name}: {dest.stat().st_size:,} bytes  sha256:{digest}")
    return paths
