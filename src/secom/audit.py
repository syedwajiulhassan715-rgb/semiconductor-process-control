"""Per-sensor data-quality audit and the retention decision it drives.

The SECOM documentation advertises 590 features. It does not mention that a
large fraction of those columns carry no usable information: some are frozen at
a single value, and many are byte-for-byte duplicates of another column. Any
analysis that treats 590 as the effective dimensionality is overstating both the
information content and the severity of its own multiple-comparison problem.

This module measures those defects rather than assuming them, and emits an
auditable ledger: one row per raw sensor, with a disposition and the reason.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from scipy import stats as sps

from .stats import benjamini_hochberg

#: Missingness above which a sensor is dropped rather than imputed.
#: SECOM missingness is bimodal: 506 sensors sit at or below 3.3% and the next
#: group starts at 16.6%, with the largest gap in the whole distribution between
#: 17.4% and 45.6%. Any threshold in [0.175, 0.456] drops exactly the same 32
#: sensors, so this choice is insensitive over a wide band.
MAX_MISSING_FRAC = 0.20

#: Absolute Pearson r at or above which two surviving sensors are redundant.
NEAR_DUPLICATE_R = 0.999

DISPOSITIONS = (
    "drop_constant",
    "drop_duplicate",
    "drop_sparse",
    "drop_near_duplicate",
    "retain",
)


def _column_hash(col: np.ndarray) -> str:
    """Hash a float column so that NaN positions participate in the identity."""
    sentinel = np.where(np.isnan(col), -9.87654321e30, col)
    return hashlib.blake2b(sentinel.tobytes(), digest_size=16).hexdigest()


def _robust_outlier_frac(present: np.ndarray, k: float = 5.0) -> float:
    """Fraction of observations beyond median +/- k * (1.4826 * MAD).

    MAD-based rather than SD-based: with tails this heavy, an SD-based screen is
    inflated by the very points it is meant to find.
    """
    if len(present) < 10:
        return np.nan
    med = np.median(present)
    mad = np.median(np.abs(present - med)) * 1.4826
    if mad == 0:
        # More than half the observations are identical, so there is no scale to
        # measure against. Returning 0.0 here would hide the opposite situation:
        # a sensor pinned at one value except for a handful of excursions, where
        # every differing point is unambiguously an outlier.
        return float(np.mean(present != med))
    return float(np.mean(np.abs(present - med) > k * mad))


def audit_sensors(
    frame: pd.DataFrame,
    sensors: list[str],
    *,
    max_missing_frac: float = MAX_MISSING_FRAC,
    near_duplicate_r: float = NEAR_DUPLICATE_R,
) -> pd.DataFrame:
    """Profile every sensor and assign a retention disposition.

    The cascade is ordered and each sensor receives exactly one reason, so the
    counts by disposition sum to the raw sensor count:

    1. ``drop_constant``       -- fewer than two distinct observed values.
    2. ``drop_duplicate``      -- identical values *and* identical NaN pattern to
       an earlier surviving sensor.
    3. ``drop_sparse``         -- missing fraction above ``max_missing_frac``.
    4. ``drop_near_duplicate`` -- absolute correlation at or above
       ``near_duplicate_r`` against an earlier surviving sensor.
    5. ``retain``              -- everything else.

    Constants are removed first because a frozen column is trivially a duplicate
    of every other frozen column, and attributing it to duplication would hide
    the more specific defect.
    """
    values = frame[sensors]
    n_rows = len(values)

    rows: list[dict] = []
    for s in sensors:
        col = values[s].to_numpy(dtype=float)
        present = col[~np.isnan(col)]
        n = len(present)
        rows.append(
            {
                "sensor": s,
                "n_present": n,
                "missing_frac": 1.0 - n / n_rows,
                "n_unique": int(np.unique(present).size),
                "mean": float(present.mean()) if n else np.nan,
                "sd": float(present.std(ddof=1)) if n > 1 else np.nan,
                "min": float(present.min()) if n else np.nan,
                "max": float(present.max()) if n else np.nan,
                "iqr": float(np.percentile(present, 75) - np.percentile(present, 25))
                if n
                else np.nan,
                "skew": float(sps.skew(present))
                if n > 2 and np.ptp(present) > 0
                else np.nan,
                "excess_kurtosis": float(sps.kurtosis(present))
                if n > 3 and np.ptp(present) > 0
                else np.nan,
                "robust_outlier_frac": _robust_outlier_frac(present),
                "_hash": _column_hash(col),
            }
        )
    audit = pd.DataFrame(rows).set_index("sensor")
    audit["disposition"] = ""
    audit["duplicate_of"] = pd.Series(index=audit.index, dtype="object")

    # 1. constants
    audit.loc[audit["n_unique"] <= 1, "disposition"] = "drop_constant"

    # 2. exact duplicates, keeping the lowest-numbered representative
    seen: dict[str, str] = {}
    for s in sensors:
        if audit.at[s, "disposition"]:
            continue
        h = audit.at[s, "_hash"]
        if h in seen:
            audit.at[s, "disposition"] = "drop_duplicate"
            audit.at[s, "duplicate_of"] = seen[h]
        else:
            seen[h] = s

    # 3. sparse
    sparse = (audit["disposition"] == "") & (audit["missing_frac"] > max_missing_frac)
    audit.loc[sparse, "disposition"] = "drop_sparse"

    # 4. near-duplicates among survivors
    survivors = [s for s in sensors if audit.at[s, "disposition"] == ""]
    if survivors:
        corr = values[survivors].corr(method="pearson", min_periods=30).to_numpy()
        corr = np.nan_to_num(corr, nan=0.0)
        kept: list[int] = []
        for j, s in enumerate(survivors):
            hit = next((k for k in kept if abs(corr[j, k]) >= near_duplicate_r), None)
            if hit is None:
                kept.append(j)
            else:
                audit.at[s, "disposition"] = "drop_near_duplicate"
                audit.at[s, "duplicate_of"] = survivors[hit]

    audit.loc[audit["disposition"] == "", "disposition"] = "retain"
    audit["retained"] = audit["disposition"] == "retain"
    return audit.drop(columns="_hash")


def missingness_vs_outcome(
    frame: pd.DataFrame, sensors: list[str], is_fail: np.ndarray
) -> pd.DataFrame:
    """Test whether *whether a sensor reported* is associated with the outcome.

    Missingness is only safely imputable if it is not itself informative. If a
    sensor tends to go missing on failing runs, imputation silently destroys
    signal and the missingness should be carried as its own indicator instead.
    Fisher's exact test is used because several sensors have very few missing
    runs, where the chi-square approximation is unreliable.
    """
    is_fail = np.asarray(is_fail, dtype=bool)
    rows = []
    for s in sensors:
        miss = frame[s].isna().to_numpy()
        if miss.sum() == 0 or miss.sum() == len(miss):
            continue
        table = np.array(
            [
                [int((miss & is_fail).sum()), int((miss & ~is_fail).sum())],
                [int((~miss & is_fail).sum()), int((~miss & ~is_fail).sum())],
            ]
        )
        odds, p = sps.fisher_exact(table)
        rows.append(
            {
                "sensor": s,
                "n_missing": int(miss.sum()),
                "fail_rate_when_missing": table[0, 0] / max(table[0].sum(), 1),
                "fail_rate_when_present": table[1, 0] / max(table[1].sum(), 1),
                "odds_ratio": float(odds),
                "p_fisher": float(p),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_fisher"] = benjamini_hochberg(out["p_fisher"].to_numpy())
    return out.sort_values("q_fisher", kind="stable").reset_index(drop=True)
