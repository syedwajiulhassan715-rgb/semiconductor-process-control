"""Drift, regime and failure screens, the time-confound gate, and the watchlist."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

from ...spc import individuals_limits
from ...stats import benjamini_hochberg, screen_sensors
from ..context import PipelineContext

NAME = "sensor_screen"
REQUIRES = ("clean", "retained", "is_fail", "masks")
PRODUCES = (
    "drift",
    "regime",
    "fail_all",
    "fail_stable",
    "confound",
    "watchlist",
    "limits",
)


def run(ctx: PipelineContext) -> None:
    cfg = ctx.config
    alpha = cfg.alpha
    clean = ctx["clean"]
    retained = ctx["retained"]
    is_fail = ctx["is_fail"]
    m_pre, m_base = ctx["masks"]["pre"], ctx["masks"]["base"]

    rho = np.empty(len(retained))
    pv = np.empty(len(retained))
    for i, s in enumerate(retained):
        rho[i], pv[i] = sps.spearmanr(clean["run_index"], clean[s])
    drift = pd.DataFrame(
        {"sensor_id": retained, "drift_rho": rho, "drift_q": benjamini_hochberg(pv)}
    )

    regime = screen_sensors(clean, retained, m_pre, m_base)
    fail_all = screen_sensors(clean, retained, is_fail, ~is_fail)
    fail_stable = screen_sensors(clean, retained, is_fail & m_base, ~is_fail & m_base)

    stable_sig = fail_stable["q_mannwhitney"] < alpha
    results: dict[str, Any] = {
        "n_tested": len(retained),
        "drift_sig": int((drift["drift_q"] < alpha).sum()),
        "drift_pct": float(100 * (drift["drift_q"] < alpha).mean()),
        "regime_sig": int((regime["q_mannwhitney"] < alpha).sum()),
        "regime_sig_large_effect": int(
            (
                (regime["q_mannwhitney"] < alpha)
                & (regime["cliffs_delta"].abs() > cfg.sensors.large_effect_delta)
            ).sum()
        ),
        "fail_all_sig": int((fail_all["q_mannwhitney"] < alpha).sum()),
        "fail_stable_sig": int(stable_sig.sum()),
        "fail_stable_raw_p05": int((fail_stable["p_mannwhitney"] < alpha).sum()),
        "fail_stable_expected_by_chance": round(alpha * len(retained), 1),
        "fail_stable_survivors": fail_stable.loc[stable_sig, "sensor"].tolist(),
    }

    # do the all-runs hits still hold up inside the stable weeks?
    hits = fail_all.loc[fail_all["q_mannwhitney"] < alpha, "sensor"].tolist()
    stable_p = fail_stable.set_index("sensor")["p_mannwhitney"]
    confound = pd.DataFrame(
        {
            "sensor": hits,
            "drift_rho": drift.set_index("sensor_id").loc[hits, "drift_rho"].to_numpy(),
            "p_within_stable": stable_p.loc[hits].to_numpy(),
        }
    )
    results["hits_surviving_confound_check"] = confound.loc[
        confound["p_within_stable"] < alpha, "sensor"
    ].tolist()
    ctx.results["sensors"] = results

    # watchlist: tier 1/2 sensors + top regime movers, capped
    watch = list(
        dict.fromkeys(
            fail_stable.loc[stable_sig, "sensor"].tolist()
            + hits
            + regime.head(cfg.sensors.watchlist_regime_top)["sensor"].tolist()
        )
    )[: cfg.sensors.watchlist_max]

    lim_rows = []
    for s in retained:
        base_vals = clean.loc[m_base, s].to_numpy()
        cl = individuals_limits(base_vals)
        med = float(np.median(base_vals))
        mad = float(np.median(np.abs(base_vals - med)) * 1.4826) or np.nan
        lim_rows.append(
            {
                "sensor_id": s,
                "baseline_center": cl.center,
                "baseline_sigma": cl.sigma,
                "baseline_lcl": cl.lcl,
                "baseline_ucl": cl.ucl,
                "baseline_n_excluded": cl.n_excluded,
                "baseline_median": med,
                "baseline_mad": mad,
            }
        )

    ctx.artifacts.update(
        drift=drift,
        regime=regime,
        fail_all=fail_all,
        fail_stable=fail_stable,
        confound=confound,
        watchlist=watch,
        limits=pd.DataFrame(lim_rows),
    )
    ctx.log(
        f"{results['drift_pct']:.0f}% drift; {results['fail_all_sig']} raw hits -> "
        f"{len(results['hits_surviving_confound_check'])} survive confound check"
    )
