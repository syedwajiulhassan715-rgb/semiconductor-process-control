"""Drop audited sensors, impute causally, and define the yield regimes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...clean import build_clean
from ..context import PipelineContext

NAME = "clean"
REQUIRES = ("raw", "audit")
PRODUCES = ("clean", "retained", "impute_log", "is_fail", "masks", "phase")

PHASE_LABELS = ("0 - pre-baseline", "1 - Phase I baseline", "2 - Phase II monitored")


def run(ctx: PipelineContext) -> None:
    cfg = ctx.config
    clean, retained, impute_log = build_clean(
        ctx["raw"].runs, ctx["audit"], window=cfg.clean.impute_window
    )
    clean = clean.reset_index(drop=True)
    n_cells = len(clean) * len(retained)
    ctx.results["cleaning"] = {
        "n_retained": len(retained),
        "cells_imputed": int(impute_log["n_missing"].sum()),
        "cells_total": int(n_cells),
        "imputed_pct": float(100 * impute_log["n_missing"].sum() / n_cells),
        "lookahead_cells": int(impute_log["filled_seed_lookahead"].sum()),
        "lookahead_pct": float(100 * impute_log["filled_seed_lookahead"].sum() / n_cells),
    }

    wk = clean["iso_week"].to_numpy()
    lo, hi = cfg.regimes.baseline_weeks
    masks = {
        "pre": wk <= cfg.regimes.pre_baseline_max_week,
        "base": (wk >= lo) & (wk <= hi),
        "post": wk > hi,
    }
    phase = pd.Series(
        np.select([masks["pre"], masks["base"]], list(PHASE_LABELS[:2]), default=PHASE_LABELS[2]),
        index=clean.index,
    )

    ctx.artifacts.update(
        clean=clean,
        retained=retained,
        impute_log=impute_log,
        is_fail=clean["is_fail"].to_numpy(),
        masks=masks,
        phase=phase,
    )
    ctx.log(f"{ctx.results['cleaning']['imputed_pct']:.2f}% of cells imputed")
