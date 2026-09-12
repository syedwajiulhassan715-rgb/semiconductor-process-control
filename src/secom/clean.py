"""Cleaning: apply the audit dispositions and impute the surviving gaps.

Imputation here is deliberately **causal** -- every imputed value is computed
from runs that had already happened at that point in the sequence. A centred
rolling statistic, or a global median computed over the whole file, would give
slightly better-looking imputations and would also leak information backwards in
time. Since the stated purpose of this dataset is deciding whether sensors can
support *live* monitoring, an imputation that could not be executed in real time
would invalidate the answer. See DECISIONS.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Trailing window, in runs, for the causal rolling median used to impute gaps.
#: At the median inter-arrival of 37 minutes this spans roughly half a shift,
#: long enough to be stable and short enough to track drift over the 89-day
#: campaign.
IMPUTE_WINDOW = 25

#: Shift boundaries (hour of day) for a conventional 3-shift fab schedule.
SHIFT_BOUNDS = ((6, 14, "A"), (14, 22, "B"))


def _shift_label(hour: int) -> str:
    for lo, hi, name in SHIFT_BOUNDS:
        if lo <= hour < hi:
            return name
    return "C"


def add_time_context(runs: pd.DataFrame) -> pd.DataFrame:
    """Attach calendar and cadence columns used for subgrouping and in Power BI.

    ``minutes_since_prev`` matters: the line does not run on a fixed cadence
    (median 37 min, 90th percentile 186 min, longest gap 2 days), so any
    per-run chart is unequally spaced in time and gaps are themselves a process
    event worth seeing.
    """
    out = runs.copy()
    ts = out["timestamp"]
    iso = ts.dt.isocalendar()
    out["date"] = ts.dt.normalize()
    out["iso_year"] = iso["year"].astype(int)
    out["iso_week"] = iso["week"].astype(int)
    out["year_week"] = (
        out["iso_year"].astype(str) + "-W" + out["iso_week"].astype(str).str.zfill(2)
    )
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
    """Fill gaps using only prior runs; return the filled frame and a log.

    The cascade for each missing cell, in order:

    1. trailing rolling median over the previous ``window`` runs;
    2. expanding median over all previous runs, for early gaps where the
       rolling window has nothing in it yet;
    3. the first observed value, for a sensor whose very first runs are missing
       and which therefore has no history at all.

    Step 3 is the only step that looks forward. It is unavoidable -- there is no
    prior information by construction -- and the log records exactly how many
    cells took that path so the exposure is quantified rather than hidden.
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


def standardise(
    frame: pd.DataFrame, sensors: list[str], baseline: np.ndarray
) -> pd.DataFrame:
    """Robust z-scores against a baseline window (median / MAD-sigma).

    Median and MAD rather than mean and SD so that the excursion being looked
    for does not set the scale it is measured against.
    """
    base = frame.loc[baseline, sensors]
    med = base.median()
    mad = (base - med).abs().median() * 1.4826
    mad = mad.replace(0.0, np.nan)
    return (frame[sensors] - med) / mad
