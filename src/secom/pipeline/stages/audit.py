"""Give every raw sensor a disposition, and test whether missingness is informative."""

from __future__ import annotations

from ...audit import audit_sensors, missingness_vs_outcome
from ..context import PipelineContext

NAME = "audit"
REQUIRES = ("raw",)
PRODUCES = ("audit", "missingness")


def run(ctx: PipelineContext) -> None:
    cfg = ctx.config
    data = ctx["raw"]
    runs = data.runs
    audit = audit_sensors(
        runs,
        data.sensors,
        max_missing_frac=cfg.audit.max_missing_frac,
        near_duplicate_r=cfg.audit.near_duplicate_r,
    )
    miss = missingness_vs_outcome(runs, data.sensors, runs["is_fail"].to_numpy())
    ctx.results["audit"] = {
        "disposition_counts": audit["disposition"].value_counts().to_dict(),
        "n_retained": int(audit["retained"].sum()),
        "missingness_tests": int(len(miss)),
        "missingness_sig_q05": int((miss["q_fisher"] < cfg.alpha).sum()),
        "missingness_raw_p05": int((miss["p_fisher"] < cfg.alpha).sum()),
        "missingness_min_q": float(miss["q_fisher"].min()),
        "missingness_expected_by_chance": round(cfg.alpha * len(miss), 1),
    }
    ctx.artifacts["audit"] = audit
    ctx.artifacts["missingness"] = miss
    ctx.log(f"retained {audit['retained'].sum()} of {len(audit)}")
