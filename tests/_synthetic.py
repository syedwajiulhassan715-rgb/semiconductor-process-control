"""Small synthetic stand-in for the raw SECOM files, for fast unit tests.

The real data is 1567 runs x 590 sensors; this writes a few hundred runs over
ISO weeks 30-42 of 2008 with the same file format, a degraded pre-baseline
period (weeks <= 34) and sensors that exercise every audit disposition.
``secom.io`` checks the shape against ``N_RUNS`` / ``N_SENSORS``, so callers
patch those to :data:`N_RUNS` / :data:`N_SENSORS` (see :func:`patch_shape`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

N_RUNS = 300
N_SENSORS = 20
SEED = 11

#: Small enough to keep every stage well under a second.
FAST_CONFIG: dict = {
    "cusum": {"n_sim": 200, "max_run": 2000},
    "sensors": {"watchlist_regime_top": 5, "watchlist_max": 8},
    "model": {
        "cv_folds": 3,
        "n_perm": 3,
        "rolling_origin_fracs": [0.6, 0.7],
        "triage_top_k_pct": [10, 50],
    },
    "changepoint": {"min_segment": 20, "n_perm": 30, "n_boot": 30, "max_changepoints": 2},
    "figures": {"format": "svg", "dpi": 50},
}


def make_raw(n_runs: int = N_RUNS, n_sensors: int = N_SENSORS, seed: int = SEED):
    """Return ``(values, labels, timestamps)`` shaped like the raw files."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2008-07-21 06:00:00")  # Monday of ISO week 30
    end = pd.Timestamp("2008-10-17 18:00:00")  # Friday of ISO week 42
    offsets = np.sort(rng.uniform(0, (end - start).total_seconds(), n_runs))
    ts = start + pd.to_timedelta(np.round(offsets), unit="s")
    week = ts.isocalendar().week.to_numpy()
    p_fail = np.where(week <= 34, 0.18, 0.06)
    fail = rng.random(n_runs) < p_fail
    # guarantee failures in the baseline and in the forward test window
    fail[[int(0.5 * n_runs), int(0.8 * n_runs), int(0.9 * n_runs), int(0.95 * n_runs)]] = True

    x = rng.normal(0.0, 1.0, (n_runs, n_sensors))
    t = np.linspace(0, 1, n_runs)
    x[:, 1] += 3 * t  # drifts with run order
    x[:, 2] += np.where(week <= 34, 2.0, 0.0)  # moves with the yield regime
    x[:, 3] += 1.5 * fail  # discriminates failures
    x[:, 15] = 4.2  # constant
    x[:, 16] = x[:, 0]  # exact duplicate of S001
    x[rng.random(n_runs) < 0.75, 17] = np.nan  # sparse
    x[rng.random(n_runs) < 0.05, 18] = np.nan  # sporadic gaps
    x[:5, 19] = np.nan  # leading gap: seeded with a look-ahead fill
    x[fail & (rng.random(n_runs) < 0.5), 4] = np.nan  # informative missingness
    labels = np.where(fail, 1, -1)
    return x, labels, ts


def write_raw(raw_dir: Path, **kwargs) -> Path:
    """Write ``secom.data`` and ``secom_labels.data`` into ``raw_dir``."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    x, labels, ts = make_raw(**kwargs)
    np.savetxt(raw_dir / "secom.data", x, fmt="%.6g", delimiter=" ")
    lines = [f'{lab} "{t:%d/%m/%Y %H:%M:%S}"' for lab, t in zip(labels, ts, strict=True)]
    (raw_dir / "secom_labels.data").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return raw_dir


def patch_shape(
    monkeypatch: pytest.MonkeyPatch, n_runs: int = N_RUNS, n_sensors: int = N_SENSORS
) -> None:
    """Make ``secom.io.load_raw`` accept the synthetic shape."""
    import secom.io

    monkeypatch.setattr(secom.io, "N_RUNS", n_runs)
    monkeypatch.setattr(secom.io, "N_SENSORS", n_sensors)


def write_config(path: Path, **overrides) -> Path:
    """Write :data:`FAST_CONFIG` (plus ``overrides``) as YAML."""
    data = {**FAST_CONFIG, **overrides}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path
