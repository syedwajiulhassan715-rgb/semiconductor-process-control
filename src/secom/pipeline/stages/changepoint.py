"""Locate the yield change at run grain, and test for further change points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ...changepoint import (
    ChangePointFit,
    binary_segmentation,
    bootstrap_location_ci,
    fit_single,
    permutation_null,
    pvalue_from_null,
    segments,
)
from ...config import register_section
from ..context import PipelineContext
from ..plotting import ACCENT, GOOD, INK, MUTED

NAME = "changepoint"
REQUIRES = ("clean", "is_fail", "masks")
PRODUCES = ("changepoint_fit",)


@register_section("changepoint")
@dataclass(frozen=True)
class ChangepointConfig:
    #: Shortest segment either side of a split, in runs.
    min_segment: int = 30
    #: Permutations per test (plain and block). p-values bottom out at 1/(n+1).
    n_perm: int = 2000
    #: Parametric-bootstrap replicates for the location interval.
    n_boot: int = 1000
    ci_level: float = 0.95
    #: Binary segmentation stops after this many accepted change points.
    max_changepoints: int = 5


def _when(clean: pd.DataFrame, i: int) -> dict[str, Any]:
    """Run number, timestamp and ISO week of zero-based row ``i``."""
    row = clean.iloc[i]
    return {
        "run": int(i + 1),
        "timestamp": str(row["timestamp"]),
        "year_week": str(row["year_week"]),
    }


def run(ctx: PipelineContext) -> None:
    cc = ctx.config.section(ChangepointConfig)
    alpha = ctx.config.alpha
    clean = ctx["clean"]
    x = ctx["is_fail"].astype(float)
    blocks = pd.factorize(clean["date"])[0]
    rng = np.random.default_rng(ctx.config.seed)

    fit = fit_single(x, cc.min_segment)
    null_plain = permutation_null(x, min_segment=cc.min_segment, n_perm=cc.n_perm, rng=rng)
    null_block = permutation_null(
        x, min_segment=cc.min_segment, n_perm=cc.n_perm, rng=rng, blocks=blocks
    )
    lo, hi, _ = bootstrap_location_ci(
        fit, n_boot=cc.n_boot, rng=rng, level=cc.ci_level, min_segment=cc.min_segment
    )
    cps, tests = binary_segmentation(
        x,
        blocks,
        min_segment=cc.min_segment,
        alpha=alpha,
        n_perm=cc.n_perm,
        rng=rng,
        max_changepoints=cc.max_changepoints,
    )

    # Phase I starts at the first run of the first baseline week (D7).
    boundary = int(np.flatnonzero(ctx["masks"]["base"])[0])
    ts = clean["timestamp"]
    hours = float((ts.iloc[boundary] - ts.iloc[fit.k]) / pd.Timedelta(hours=1))
    inside = lo <= boundary <= hi

    ctx.results["changepoint"] = {
        "n_runs": len(x),
        "min_segment": cc.min_segment,
        "n_perm": cc.n_perm,
        "n_boot": cc.n_boot,
        "ci_level": cc.ci_level,
        "alpha": alpha,
        "permutation_block": "calendar day",
        "n_blocks": int(blocks.max() + 1),
        "estimate": {
            "first_run_after": _when(clean, fit.k),
            "last_run_before": _when(clean, fit.k - 1),
            "n_before": fit.n_before,
            "n_after": fit.n_after,
            "fails_before": int(x[: fit.k].sum()),
            "fails_after": int(x[fit.k :].sum()),
            "rate_before": fit.rate_before,
            "rate_after": fit.rate_after,
            "rate_ratio": fit.rate_before / fit.rate_after,
            "lr_stat": fit.lr_stat,
            "p_perm": pvalue_from_null(fit.lr_stat, null_plain),
            "p_block": pvalue_from_null(fit.lr_stat, null_block),
            "p_min_attainable": 1.0 / (1 + cc.n_perm),
            "block_null_95th": float(np.quantile(null_block, 0.95)),
            "plain_null_95th": float(np.quantile(null_plain, 0.95)),
        },
        "ci": {"first_run_lo": _when(clean, lo), "first_run_hi": _when(clean, hi)},
        "baseline_boundary": {
            "first_run": _when(clean, boundary),
            "inside_ci": bool(inside),
            "runs_after_estimate": int(boundary - fit.k),
            "hours_after_estimate": hours,
            # Worst case within the interval: baseline runs that would precede the change.
            "baseline_runs_before_ci_hi": int(max(0, hi - boundary)),
        },
        "agrees_with_baseline": bool(inside),
        "segmentation": {
            "rule": (
                f"split a segment while its block-permutation p < {alpha}; "
                f"segments shorter than {2 * cc.min_segment} runs are not tested"
            ),
            "change_first_runs": [int(c + 1) for c in cps],
            "segments": [
                {
                    "first_run": _when(clean, a),
                    "last_run": _when(clean, b - 1),
                    "n": int(b - a),
                    "fails": int(f),
                    "rate": float(r),
                }
                for a, b, f, r in segments(x, cps)
            ],
            "tests": [
                {
                    "first_run": int(t.start + 1),
                    "last_run": int(t.stop),
                    "split_first_run": int(t.k + 1),
                    "lr_stat": t.lr_stat,
                    "p_perm": t.p_perm,
                    "p_block": t.p_block,
                    "accepted": t.accepted,
                }
                for t in tests
            ],
        },
    }
    ctx.artifacts["changepoint_fit"] = fit
    est = ctx.results["changepoint"]["estimate"]["first_run_after"]
    ctx.log(
        f"change at run {est['run']} ({est['timestamp']}, {est['year_week']}), "
        f"CI runs {lo + 1}-{hi + 1}; W{ctx.config.regimes.baseline_weeks[0]} boundary "
        f"{'inside' if inside else 'OUTSIDE'} the interval; {len(cps)} change point(s) kept"
    )
    if ctx.figures_enabled:
        _figure(ctx, clean, x, fit, lo, hi, boundary, cps, null_block)


def _figure(
    ctx: PipelineContext,
    clean: pd.DataFrame,
    x: np.ndarray,
    fit: ChangePointFit,
    lo: int,
    hi: int,
    boundary: int,
    cps: list[int],
    null_block: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    t = clean["timestamp"]
    null_ll = float(fit.profile[fit.k] - fit.lr_stat / 2)
    lr = 2 * (fit.profile - null_ll)
    level = ctx.results["changepoint"]["ci_level"]
    start_week = clean["year_week"].iloc[boundary]

    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(8.2, 4.6), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]}
    )
    a1.plot(t, lr, color=INK, lw=1.0)
    a1.axvspan(
        t.iloc[lo], t.iloc[hi], color=ACCENT, alpha=0.12, label=f"{level:.0%} bootstrap interval"
    )
    a1.axvline(
        t.iloc[fit.k],
        color=ACCENT,
        lw=1.6,
        label=f"estimate: run {fit.k + 1}, {t.iloc[fit.k]:%d %b %H:%M}",
    )
    a1.axvline(
        t.iloc[boundary],
        color=GOOD,
        lw=1.2,
        ls="--",
        label=f"Phase I start: run {boundary + 1} ({start_week})",
    )
    a1.axhline(
        np.quantile(null_block, 0.95),
        color=MUTED,
        lw=1,
        ls=":",
        label="95th pct of max LR under day-block permutation",
    )
    a1.set_ylabel("LR statistic of split")
    a1.set_ylim(0, None)
    a1.set_title("Where does the failure rate change? Profile likelihood over every split")
    a1.legend(fontsize=6.8, frameon=False, loc="upper right")
    a1.grid(axis="y", lw=0.6)

    roll = pd.Series(x).rolling(50, min_periods=10, center=True).mean()
    a2.plot(t, roll, color=MUTED, lw=0.8, label="50-run rolling fail rate")
    for a, b, _, r in segments(x, cps):
        a2.hlines(r, t.iloc[a], t.iloc[b - 1], color=INK, lw=2.0)
    a2.plot([], [], color=INK, lw=2.0, label="segment rate (binary segmentation)")
    fails = np.flatnonzero(x)
    a2.vlines(t.iloc[fails], 0, 0.012, color=ACCENT, lw=0.6, label="failed run")
    a2.axvline(t.iloc[fit.k], color=ACCENT, lw=1.6)
    a2.axvline(t.iloc[boundary], color=GOOD, lw=1.2, ls="--")
    a2.set_ylabel("fail rate")
    a2.set_ylim(0, None)
    a2.legend(fontsize=6.8, frameon=False, loc="upper right")
    a2.grid(axis="y", lw=0.6)
    fig.autofmt_xdate()
    fig.tight_layout()
    ctx.save_figure(fig, "fig12_changepoint")
