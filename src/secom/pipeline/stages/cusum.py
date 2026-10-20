"""Bernoulli CUSUM designed on the Phase I rate, with its threshold set by simulation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ...spc import bernoulli_cusum, calibrate_cusum_h
from ..context import PipelineContext

NAME = "cusum"
REQUIRES = ("clean", "is_fail", "masks")
PRODUCES = ("cusum_all", "cusum_design")


def run(ctx: PipelineContext) -> None:
    cfg = ctx.config
    clean = ctx["clean"]
    is_fail = ctx["is_fail"]
    masks = ctx["masks"]

    p0 = float(is_fail[masks["base"]].mean())
    p1 = cfg.cusum.shift_ratio * p0
    h, arl0 = calibrate_cusum_h(
        p0,
        p1,
        target_arl0=cfg.cusum.target_arl0,
        n_sim=cfg.cusum.n_sim,
        max_run=cfg.cusum.max_run,
        seed=cfg.seed,
    )
    cusum_all = bernoulli_cusum(is_fail, p0=p0, p1=p1, h=h)

    detect: dict[str, dict[str, Any]] = {}
    for label, mask in (
        ("pre_baseline", masks["pre"]),
        ("phase_ii", masks["post"]),
        ("baseline_self_check", masks["base"]),
    ):
        sub = bernoulli_cusum(is_fail[mask], p0=p0, p1=p1, h=h)
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
    ctx.results["cusum"] = {
        "p0": p0,
        "p1": p1,
        "h": float(h),
        "arl0_runs": float(arl0),
        "baseline_weeks": list(cfg.regimes.baseline_weeks),
        "detection": detect,
    }
    ctx.artifacts["cusum_all"] = cusum_all
    ctx.artifacts["cusum_design"] = {"p0": p0, "p1": p1, "h": float(h), "arl0": float(arl0)}
    ctx.log(
        f"h={h:.3f}, ARL0={arl0:.0f}; Phase II signal in "
        f"{detect['phase_ii']['runs_to_signal']} runs"
    )
