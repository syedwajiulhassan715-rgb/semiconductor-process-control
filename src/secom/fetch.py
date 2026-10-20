"""Download the raw datasets.

SECOM: McCann, M. & Johnston, A. (2008). SECOM [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C54305

Tennessee Eastman (validation only): the Braatz training and testing files
used in Chiang, Russell & Braatz (2001), mirrored at
github.com/camaramm/tennessee-eastman-profBraatz. Only the files the
validation uses are fetched: normal-operation training data (d00.dat) and the
22 testing files (d00_te.dat ... d21_te.dat). They are never committed.
"""

from __future__ import annotations

import hashlib
import urllib.request
from collections.abc import Callable
from pathlib import Path

BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom"
FILES = ("secom.data", "secom_labels.data", "secom.names")

TEP_BASE_URL = "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master"
TEP_FILES = ("d00.dat", *(f"d{k:02d}_te.dat" for k in range(22)), "LICENSE")

DATASETS = ("secom", "tep")


def _download(
    base_url: str, names: tuple[str, ...], dest_dir: Path, log: Callable[[str], None]
) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        dest = dest_dir / name
        paths.append(dest)
        if dest.exists():
            log(f"skip  {name} (already present)")
            continue
        urllib.request.urlretrieve(f"{base_url}/{name}", dest)
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()[:16]
        log(f"fetch {name}: {dest.stat().st_size:,} bytes  sha256:{digest}")
    return paths


def fetch(raw_dir: str | Path, *, log: Callable[[str], None] = print) -> list[Path]:
    """Download any missing SECOM files into ``raw_dir``. Returns all file paths.

    Files already present are skipped, so this is safe to rerun.
    """
    return _download(BASE_URL, FILES, Path(raw_dir), log)


def fetch_tep(raw_dir: str | Path, *, log: Callable[[str], None] = print) -> list[Path]:
    """Download any missing Tennessee Eastman files into ``raw_dir/tep``.

    ``raw_dir/tep/SHA256SUMS`` records the digest of every file, so a changed
    mirror shows up as a diff rather than silently different results.
    """
    tep_dir = Path(raw_dir) / "tep"
    paths = _download(TEP_BASE_URL, TEP_FILES, tep_dir, log)
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in paths]
    (tep_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths
