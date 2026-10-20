"""Apply the audit results and impute the remaining gaps.

Imputation only looks at earlier runs (no centred windows or whole-file medians)
so it could actually be run live. See DECISIONS.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Trailing window (in runs) for the rolling median. About half a shift at the
#: median 37 min between runs.
IMPUTE_WINDOW = 25

#: Shift start/end hours. Anything else is shift C.
SHIFT_BOUNDS = ((6, 14, "A"), (14, 22, "B"))


def _shift_label(hour: int) -> str:
    for lo, hi, name in SHIFT_BOUNDS:
        if lo <= hour < hi:
            return name
    return "C"


def add_time_context(runs: pd.DataFrame) -> pd.DataFrame:
    """Add date/week/shift columns and minutes_since_prev.

    Runs aren't evenly spaced (median 37 min, longest gap 2 days), which is why
    the gap column is there.
    """
    out = runs.copy()
    ts = out["timestamp"]
    iso = ts.dt.isocalendar()
    out["date"] = ts.dt.normalize()
    out["iso_year"] = iso["year"].astype(int)
    out["iso_week"] = iso["week"].astype(int)
    out["year_week"] = out["iso_year"].astype(str) + "-W" + out["iso_week"].astype(str).str.zfill(2)
    out["hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.day_name()
    out["shift"] = out["hour"].map(_shift_label)
    out["minutes_since_prev"] = ts.diff().dt.total_seconds().div(60.0)
    out["run_index"] = np.arange(1, len(out) + 1)
    return out


def causal_impute(
    frame: pd.DataFrame,
    sensors: list[str],
    *,
    window: int = IMPUTE_WINDOW,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill gaps using only earlier runs. Returns (filled frame, log).

    For each missing value, try in order:

    1. median of the previous `window` runs
    2. median of all previous runs
    3. first observed value (only for gaps at the very start, so this is the one
       case that looks ahead; logged as filled_seed_lookahead)
    """
    out = frame.copy()
    rows = []
    for s in sensors:
        col = out[s]
        missing = col.isna()
        if not missing.any():
            continue
        rolling = col.rolling(window=window, min_periods=1).median().shift(1)
        expanding = col.expanding(min_periods=1).median().shift(1)
        filled = col.fillna(rolling)
        n_roll = int(missing.sum() - filled.isna().sum())
        filled = filled.fillna(expanding)
        n_exp = int(missing.sum() - n_roll - filled.isna().sum())
        n_seed = int(filled.isna().sum())
        if n_seed:
            first = col.dropna()
            filled = filled.fillna(first.iloc[0] if len(first) else 0.0)
        out[s] = filled
        rows.append(
            {
                "sensor": s,
                "n_missing": int(missing.sum()),
                "filled_rolling": n_roll,
                "filled_expanding": n_exp,
                "filled_seed_lookahead": n_seed,
            }
        )
    return out, pd.DataFrame(rows)


def build_clean(
    runs: pd.DataFrame, audit: pd.DataFrame, *, window: int = IMPUTE_WINDOW
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Produce the analysis-ready frame.

    Returns ``(clean_frame, retained_sensor_ids, impute_log)``.
    """
    retained = [s for s in audit.index if audit.at[s, "retained"]]
    keep_cols = ["run_id", "timestamp", "is_fail", *retained]
    frame = runs[keep_cols].copy()
    frame, impute_log = causal_impute(frame, retained, window=window)
    frame = add_time_context(frame)
    if frame[retained].isna().any().any():
        raise ValueError("imputation left missing values in retained sensors")
    return frame, retained, impute_log


def standardise(frame: pd.DataFrame, sensors: list[str], baseline: np.ndarray) -> pd.DataFrame:
    """Robust z-scores (median / MAD) against a baseline window."""
    base = frame.loc[baseline, sensors]
    med = base.median()
    mad = (base - med).abs().median() * 1.4826
    mad = mad.replace(0.0, np.nan)
    return (frame[sensors] - med) / mad
