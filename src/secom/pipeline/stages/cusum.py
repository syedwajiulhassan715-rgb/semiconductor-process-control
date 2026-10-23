"""Bernoulli CUSUMs designed on the Phase I rate, with thresholds set by simulation.

The upper CUSUM watches for a rise in the fail rate (the deployed yield
monitor). The lower CUSUM, designed for p1 = lower_ratio * p0, watches for a
fall: it is the improvement detector the report recommends over the g-chart.
A second lower CUSUM, designed on the degraded pre-baseline rate, replays the
campaign to show when the real week-34 improvement would have been confirmed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from ...spc import (
    BernoulliCusumResult,
    bernoulli_cusum,
    bernoulli_cusum_lower,
    calibrate_cusum_h,
)
from ..context import PipelineContext

NAME = "cusum"
REQUIRES = ("clean", "is_fail", "masks")
PRODUCES = ("cusum_all", "cusum_design", "cusum_lower_all", "cusum_lower_design")

CusumFn = Callable[..., BernoulliCusumResult]


def _window_detection(
    fn: CusumFn,
    clean: pd.DataFrame,
    is_fail: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    p0: float,
    p1: float,
    h: float,
) -> dict[str, dict[str, Any]]:
    """Run the chart fresh on each regime window and record its first signal."""
    detect: dict[str, dict[str, Any]] = {}
    for label, mask in (
        ("pre_baseline", masks["pre"]),
        ("phase_ii", masks["post"]),
        ("baseline_self_check", masks["base"]),
    ):
        sub = fn(is_fail[mask], p0=p0, p1=p1, h=h)
        idx = sub.first_signal_index
        ts = clean.loc[mask, "timestamp"].to_numpy()
        episodes = int((sub.table["signal"].astype(int).diff() == 1).sum()) + int(
            bool(sub.table["signal"].iloc[0])
        )
        detect[label] = {
            "n_runs": int(mask.sum()),
            "n_fail": int(is_fail[mask].sum()),
            "rate": float(is_fail[mask].mean()),
            "runs_to_signal": None if idx is None else int(idx + 1),
            "days_to_signal": None
            if idx is None
            else float((ts[idx] - ts[0]) / np.timedelta64(1, "D")),
            "signal_date": None if idx is None else str(pd.Timestamp(ts[idx]).date()),
            "alarm_episodes": episodes,
        }
    return detect


def _signal_record(res: BernoulliCusumResult, clean: pd.DataFrame) -> dict[str, Any]:
    """Every alarm of a chart run over the whole campaign, as 1-based run numbers."""
    sig = res.table["signal"].to_numpy()
    idx = np.flatnonzero(sig)
    ts = clean["timestamp"].to_numpy()
    weeks = clean["year_week"].to_numpy()
    return {
        "n_signals": int(len(idx)),
        "signal_runs": [int(i + 1) for i in idx],
        "signal_timestamps": [str(pd.Timestamp(ts[i])) for i in idx],
        "signal_weeks": [str(weeks[i]) for i in idx],
        "max_statistic": float(res.table["cusum"].max()),
    }


def run(ctx: PipelineContext) -> None:
    cfg = ctx.config
    cc = cfg.cusum
    clean = ctx["clean"]
    is_fail = ctx["is_fail"]
    masks = ctx["masks"]

    def calibrate(p0: float, p1: float) -> tuple[float, float]:
        return calibrate_cusum_h(
            p0, p1, target_arl0=cc.target_arl0, n_sim=cc.n_sim, max_run=cc.max_run, seed=cfg.seed
        )

    # Upper (degradation) CUSUM: the deployed yield monitor.
    p0 = float(is_fail[masks["base"]].mean())
    p1 = cc.shift_ratio * p0
    h, arl0 = calibrate(p0, p1)
    cusum_all = bernoulli_cusum(is_fail, p0=p0, p1=p1, h=h)
    detect = _window_detection(bernoulli_cusum, clean, is_fail, masks, p0=p0, p1=p1, h=h)

    # Lower (improvement) CUSUM on the same Phase I rate.
    lp1 = cc.lower_ratio * p0
    lh, larl0 = calibrate(p0, lp1)
    lower_all = bernoulli_cusum_lower(is_fail, p0=p0, p1=lp1, h=lh)
    lower_detect = _window_detection(
        bernoulli_cusum_lower, clean, is_fail, masks, p0=p0, p1=lp1, h=lh
    )

    # Retrospective: a lower CUSUM designed on the degraded rate of weeks <= 34,
    # run from run 1. Its first signal is when the improvement would have been
    # confirmed had that rate been the reference. Illustrative only: the
    # reference rate is estimated from the same data it is run on.
    rp0 = float(is_fail[masks["pre"]].mean())
    rp1 = cc.lower_ratio * rp0
    rh, rarl0 = calibrate(rp0, rp1)
    retro = bernoulli_cusum_lower(is_fail, p0=rp0, p1=rp1, h=rh)
    retro_rec = _signal_record(retro, clean)
    first = retro.first_signal_index

    ctx.results["cusum"] = {
        "p0": p0,
        "p1": p1,
        "h": float(h),
        "arl0_runs": float(arl0),
        "baseline_weeks": list(cfg.regimes.baseline_weeks),
        "detection": detect,
        "lower": {
            "direction": "improvement (fail rate falls)",
            "p0": p0,
            "p1": lp1,
            "lower_ratio": cc.lower_ratio,
            "h": float(lh),
            "arl0_runs": float(larl0),
            "weights": {
                "fail": float(np.log(lp1 / p0)),
                "pass": float(np.log((1 - lp1) / (1 - p0))),
            },
            "detection": lower_detect,
            "all_runs": _signal_record(lower_all, clean),
        },
        "lower_retrospective": {
            "note": (
                "lower CUSUM designed on the weeks <= 34 rate and run from run 1; "
                "illustrates when the improvement would have been confirmed; the "
                "reference rate is estimated from the same data"
            ),
            "p0": rp0,
            "p1": rp1,
            "h": float(rh),
            "arl0_runs": float(rarl0),
            "first_signal_run": None if first is None else int(first + 1),
            "first_signal_timestamp": None
            if first is None
            else str(pd.Timestamp(clean["timestamp"].iloc[first])),
            "first_signal_week": None if first is None else str(clean["year_week"].iloc[first]),
            **retro_rec,
        },
    }
    ctx.artifacts["cusum_all"] = cusum_all
    ctx.artifacts["cusum_design"] = {"p0": p0, "p1": p1, "h": float(h), "arl0": float(arl0)}
    ctx.artifacts["cusum_lower_all"] = lower_all
    ctx.artifacts["cusum_lower_design"] = {
        "p0": p0,
        "p1": lp1,
        "h": float(lh),
        "arl0": float(larl0),
        "lower_ratio": cc.lower_ratio,
    }
    ctx.log(
        f"h={h:.3f}, ARL0={arl0:.0f}; Phase II signal in "
        f"{detect['phase_ii']['runs_to_signal']} runs; lower h={lh:.3f}, "
        f"{lower_all.table['signal'].sum()} improvement alarms; retrospective lower "
        f"chart first signals at run {retro_rec['signal_runs'][:1]}"
    )
