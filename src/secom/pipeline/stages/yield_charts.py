"""Weekly p-chart (naive, Laney, Phase I-centred), the regime test, and the g-chart."""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

from ...spc import g_chart, p_chart
from ..context import PipelineContext

NAME = "yield_charts"
REQUIRES = ("clean", "is_fail", "masks")
PRODUCES = ("weekly", "naive", "laney", "laney_p1", "gchart")


def run(ctx: PipelineContext) -> None:
    clean = ctx["clean"]
    is_fail = ctx["is_fail"]
    m_pre = ctx["masks"]["pre"]

    wkly = clean.groupby("year_week")["is_fail"].agg(n="size", d="sum").reset_index()
    d, n, labels = wkly["d"].to_numpy(), wkly["n"].to_numpy(), wkly["year_week"].to_numpy()
    naive = p_chart(d, n, labels=labels, force_type="p")
    laney = p_chart(d, n, labels=labels)
    # laney is centred on all weeks. The Power BI chart is centred on the
    # Phase I weeks only, so compute that version too to check the DAX against.
    base_weeks = (
        clean.groupby("year_week")["iso_week"]
        .first()
        .reindex(wkly["year_week"])
        .between(*ctx.config.regimes.baseline_weeks)
        .to_numpy()
    )
    laney_p1 = p_chart(d, n, labels=labels, baseline=base_weeks)
    p_bar = laney.p_bar
    n_for_lcl = 9 * (1 - p_bar) / p_bar

    tab = np.array(
        [
            [int(is_fail[m_pre].sum()), int((~is_fail[m_pre]).sum())],
            [int(is_fail[~m_pre].sum()), int((~is_fail[~m_pre]).sum())],
        ]
    )
    odds, p_fisher = sps.fisher_exact(tab)
    ci = sps.contingency.odds_ratio(tab).confidence_interval(0.95)

    gch = g_chart(np.flatnonzero(is_fail), n_total=len(clean))

    ctx.results["yield"] = {
        "weekly_subgroups": int(len(wkly)),
        "p_bar": float(p_bar),
        "sigma_z": float(laney.sigma_z),
        "dispersion_chi2_p": float(laney.dispersion_p),
        "chart_type": laney.chart_type,
        "naive_violations": int(naive.table["violation"].sum()),
        "naive_violation_weeks": list(wkly["year_week"][naive.table["violation"].to_numpy()]),
        "laney_violations": int(laney.table["violation"].sum()),
        "n_needed_for_nonzero_lcl": float(n_for_lcl),
        "weeks_with_lcl_pinned_at_zero": int((wkly["n"] < n_for_lcl).sum()),
        "regime_pre_rate": float(is_fail[m_pre].mean()),
        "regime_post_rate": float(is_fail[~m_pre].mean()),
        "regime_odds_ratio": float(odds),
        "regime_or_ci": [float(ci.low), float(ci.high)],
        "regime_fisher_p": float(p_fisher),
        "sigma_z_phase1_centred": float(laney_p1.sigma_z),
        "p_bar_phase1": float(laney_p1.p_bar),
        "phase1_centred_violations": int(laney_p1.table["violation"].sum()),
        "phase1_centred_violation_weeks": list(
            wkly["year_week"][laney_p1.table["violation"].to_numpy()]
        ),
        "g_chart_center": float(gch.center),
        "g_chart_violations": int(gch.table["violation"].sum()),
    }
    ctx.artifacts.update(weekly=wkly, naive=naive, laney=laney, laney_p1=laney_p1, gchart=gch)
    ctx.log(
        f"sigma_z={laney.sigma_z:.3f}, naive flags {naive.table['violation'].sum()}, "
        f"Laney flags {laney.table['violation'].sum()}"
    )
