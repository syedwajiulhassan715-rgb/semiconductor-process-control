"""Multiple testing correction and effect sizes.

590 tests at alpha=0.05 gives ~30 false positives by chance, so sensor screens
report BH q-values along with an effect size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values. NaN p-values stay NaN and don't count towards m."""
    p = np.asarray(pvalues, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return q
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(p[idx], kind="stable")]
    ranked = p[order] * m / np.arange(1, m + 1)
    # Step-up: enforce monotonicity from the largest p downwards.
    q[order] = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    return q


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised mean difference (a - b) with pooled SD, Hedges-uncorrected."""
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled_var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if pooled_var <= 0:
        return np.nan
    return float((a.mean() - b.mean()) / np.sqrt(pooled_var))


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta, P(a > b) - P(a < b). Computed from the Mann-Whitney U."""
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return np.nan
    u = sps.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic").statistic
    return float(2.0 * u / (na * nb) - 1.0)


@dataclass(frozen=True)
class ShiftTest:
    """Result of comparing one sensor between two groups."""

    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    delta_mean: float
    cohens_d: float
    cliffs_delta: float
    p_welch: float
    p_mannwhitney: float


def compare_groups(a: np.ndarray, b: np.ndarray) -> ShiftTest:
    """Compare one sensor between groups a and b (Welch t-test and Mann-Whitney).

    The screens sort on Mann-Whitney since a lot of these sensors aren't normal.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    na, nb = len(a), len(b)
    if na < 3 or nb < 3 or (np.ptp(a) == 0 and np.ptp(b) == 0):
        return ShiftTest(na, nb, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    p_w = float(sps.ttest_ind(a, b, equal_var=False).pvalue)
    p_mw = float(sps.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    return ShiftTest(
        n_a=na,
        n_b=nb,
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        delta_mean=float(a.mean() - b.mean()),
        cohens_d=cohens_d(a, b),
        cliffs_delta=cliffs_delta(a, b),
        p_welch=p_w,
        p_mannwhitney=p_mw,
    )


def screen_sensors(
    frame: pd.DataFrame,
    sensors: list[str],
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> pd.DataFrame:
    """Run compare_groups on every sensor and add q_mannwhitney / q_welch.

    Sorted by q_mannwhitney.
    """
    rows = []
    for s in sensors:
        col = frame[s].to_numpy(dtype=float)
        res = compare_groups(col[mask_a], col[mask_b])
        rows.append({"sensor": s, **res.__dict__})
    out = pd.DataFrame(rows)
    out["q_mannwhitney"] = benjamini_hochberg(out["p_mannwhitney"].to_numpy())
    out["q_welch"] = benjamini_hochberg(out["p_welch"].to_numpy())
    return out.sort_values("q_mannwhitney", kind="stable").reset_index(drop=True)
