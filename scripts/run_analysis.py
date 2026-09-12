"""End-to-end analysis: raw SECOM in, Power BI star schema and figures out.

Run from the repository root::

    python scripts/00_fetch_data.py
    python scripts/run_analysis.py

Every figure in reports/ and every number quoted in reports/ENGINEERING_REPORT.md
is produced by this script and written to reports/results.json, so any claim in
the write-up can be traced to a computed value rather than a remembered one.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from secom.audit import audit_sensors, missingness_vs_outcome  # noqa: E402
from secom.clean import build_clean  # noqa: E402
from secom.export import (  # noqa: E402
    build_dim_date,
    build_dim_sensor,
    build_fact_measurement,
    build_fact_run,
    write_all,
)
from secom.io import load_raw  # noqa: E402
from secom.model import (  # noqa: E402
    evaluate_forward,
    evaluate_shuffled,
    make_models,
)
from secom.spc import (  # noqa: E402
    bernoulli_cusum,
    calibrate_cusum_h,
    g_chart,
    individuals_limits,
    p_chart,
)
from secom.stats import benjamini_hochberg, screen_sensors  # noqa: E402

warnings.filterwarnings("ignore")

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
POWERBI_DATA = ROOT / "data" / "powerbi"
FIGS = ROOT / "reports" / "figures"

#: Phase I baseline: the longest contiguous run of weeks whose standardised
#: weekly failure proportions all sit within +/- 2.4 sigma of their own mean.
BASELINE_WEEKS = (35, 39)
PRE_BASELINE_MAX_WEEK = 34
TRAIN_FRAC = 0.70
SEED = 20080719

INK = "#1b1b1b"
ACCENT = "#b3412c"
MUTED = "#8a8a8a"
GOOD = "#2f6f4e"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "font.size": 9,
            "axes.edgecolor": "#cccccc",
            "axes.labelcolor": INK,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": "#ebebeb",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def main() -> int:
    _style()
    for d in (PROCESSED, POWERBI_DATA, FIGS):
        d.mkdir(parents=True, exist_ok=True)
    R: dict = {}

    # ---------------------------------------------------------------- ingest
    data = load_raw(RAW)
    runs = data.runs
    R["dataset"] = {
        "n_runs": int(len(runs)),
        "n_sensors_raw": len(data.sensors),
        "n_fail": int(runs["is_fail"].sum()),
        "fail_rate": float(runs["is_fail"].mean()),
        "start": str(runs["timestamp"].min()),
        "end": str(runs["timestamp"].max()),
        "span_days": int((runs["timestamp"].max() - runs["timestamp"].min()).days),
        "cell_missing_rate": float(runs[data.sensors].isna().to_numpy().mean()),
    }
    print(f"[1/9] loaded {len(runs)} runs x {len(data.sensors)} sensors")

    # ----------------------------------------------------------------- audit
    audit = audit_sensors(runs, data.sensors)
    miss = missingness_vs_outcome(runs, data.sensors, runs["is_fail"].to_numpy())
    R["audit"] = {
        "disposition_counts": audit["disposition"].value_counts().to_dict(),
        "n_retained": int(audit["retained"].sum()),
        "missingness_tests": int(len(miss)),
        "missingness_sig_q05": int((miss["q_fisher"] < 0.05).sum()),
        "missingness_raw_p05": int((miss["p_fisher"] < 0.05).sum()),
        "missingness_min_q": float(miss["q_fisher"].min()),
        "missingness_expected_by_chance": round(0.05 * len(miss), 1),
    }
    print(f"[2/9] audit: retained {audit['retained'].sum()} of {len(audit)}")

    # ----------------------------------------------------------------- clean
    clean, retained, impute_log = build_clean(runs, audit)
    clean = clean.reset_index(drop=True)
    n_cells = len(clean) * len(retained)
    R["cleaning"] = {
        "n_retained": len(retained),
        "cells_imputed": int(impute_log["n_missing"].sum()),
        "cells_total": int(n_cells),
        "imputed_pct": float(100 * impute_log["n_missing"].sum() / n_cells),
        "lookahead_cells": int(impute_log["filled_seed_lookahead"].sum()),
        "lookahead_pct": float(
            100 * impute_log["filled_seed_lookahead"].sum() / n_cells
        ),
    }
    print(f"[3/9] cleaned: {R['cleaning']['imputed_pct']:.2f}% of cells imputed")

    is_fail = clean["is_fail"].to_numpy()
    wk = clean["iso_week"].to_numpy()
    m_pre = wk <= PRE_BASELINE_MAX_WEEK
    m_base = (wk >= BASELINE_WEEKS[0]) & (wk <= BASELINE_WEEKS[1])
    m_post = wk > BASELINE_WEEKS[1]

    # ------------------------------------------------------------ yield SPC
    wkly = clean.groupby("year_week")["is_fail"].agg(n="size", d="sum").reset_index()
    naive = p_chart(
        wkly["d"].to_numpy(), wkly["n"].to_numpy(),
        labels=wkly["year_week"].to_numpy(), force_type="p",
    )
    laney = p_chart(
        wkly["d"].to_numpy(), wkly["n"].to_numpy(),
        labels=wkly["year_week"].to_numpy(),
    )
    # Retrospective Phase I stability assessment ("was this process ever in
    # control?") centres on all weeks. The monitoring chart the Power BI model
    # reproduces is a different question -- "is this week consistent with the
    # baseline?" -- and must centre on the Phase I window only. Both are
    # exported so the DAX has an unambiguous reference to be checked against.
    base_weeks = (
        clean.groupby("year_week")["iso_week"].first().reindex(wkly["year_week"])
        .between(*BASELINE_WEEKS).to_numpy()
    )
    laney_p1 = p_chart(
        wkly["d"].to_numpy(), wkly["n"].to_numpy(),
        labels=wkly["year_week"].to_numpy(),
        baseline=base_weeks,
    )
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

    fails_idx = np.flatnonzero(is_fail)
    gch = g_chart(fails_idx, n_total=len(clean))

    R["yield"] = {
        "weekly_subgroups": int(len(wkly)),
        "p_bar": float(p_bar),
        "sigma_z": float(laney.sigma_z),
        "dispersion_chi2_p": float(laney.dispersion_p),
        "chart_type": laney.chart_type,
        "naive_violations": int(naive.table["violation"].sum()),
        "naive_violation_weeks": list(
            wkly["year_week"][naive.table["violation"].to_numpy()]
        ),
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
    print(f"[4/9] yield: sigma_z={laney.sigma_z:.3f}, naive flags "
          f"{naive.table['violation'].sum()}, Laney flags {laney.table['violation'].sum()}")

    # ---------------------------------------------------------------- CUSUM
    p0 = float(is_fail[m_base].mean())
    p1 = 2 * p0
    h, arl0 = calibrate_cusum_h(p0, p1, target_arl0=500, n_sim=3000, max_run=8000, seed=SEED)
    cusum_all = bernoulli_cusum(is_fail, p0=p0, p1=p1, h=h)

    detect = {}
    for label, mask in (("pre_baseline", m_pre), ("phase_ii", m_post), ("baseline_self_check", m_base)):
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
    R["cusum"] = {
        "p0": p0, "p1": p1, "h": float(h), "arl0_runs": float(arl0),
        "baseline_weeks": list(BASELINE_WEEKS), "detection": detect,
    }
    print(f"[5/9] CUSUM: h={h:.3f}, ARL0={arl0:.0f}; Phase II signal in "
          f"{detect['phase_ii']['runs_to_signal']} runs")

    # ----------------------------------------------------- sensor evidence
    rho = np.empty(len(retained)); pv = np.empty(len(retained))
    for i, s in enumerate(retained):
        rho[i], pv[i] = sps.spearmanr(clean["run_index"], clean[s])
    drift = pd.DataFrame(
        {"sensor_id": retained, "drift_rho": rho, "drift_q": benjamini_hochberg(pv)}
    )

    regime = screen_sensors(clean, retained, m_pre, m_base)
    fail_all = screen_sensors(clean, retained, is_fail, ~is_fail)
    fail_stable = screen_sensors(clean, retained, is_fail & m_base, ~is_fail & m_base)

    R["sensors"] = {
        "n_tested": len(retained),
        "drift_sig": int((drift["drift_q"] < 0.05).sum()),
        "drift_pct": float(100 * (drift["drift_q"] < 0.05).mean()),
        "regime_sig": int((regime["q_mannwhitney"] < 0.05).sum()),
        "regime_sig_large_effect": int(
            ((regime["q_mannwhitney"] < 0.05) & (regime["cliffs_delta"].abs() > 0.5)).sum()
        ),
        "fail_all_sig": int((fail_all["q_mannwhitney"] < 0.05).sum()),
        "fail_stable_sig": int((fail_stable["q_mannwhitney"] < 0.05).sum()),
        "fail_stable_raw_p05": int((fail_stable["p_mannwhitney"] < 0.05).sum()),
        "fail_stable_expected_by_chance": round(0.05 * len(retained), 1),
        "fail_stable_survivors": fail_stable.loc[
            fail_stable["q_mannwhitney"] < 0.05, "sensor"
        ].tolist(),
    }

    # Confound check on the whole-dataset hits.
    hits = fail_all.loc[fail_all["q_mannwhitney"] < 0.05, "sensor"].tolist()
    stable_p = fail_stable.set_index("sensor")["p_mannwhitney"]
    confound = pd.DataFrame(
        {
            "sensor": hits,
            "drift_rho": drift.set_index("sensor_id").loc[hits, "drift_rho"].to_numpy(),
            "p_within_stable": stable_p.loc[hits].to_numpy(),
        }
    )
    R["sensors"]["hits_surviving_confound_check"] = confound.loc[
        confound["p_within_stable"] < 0.05, "sensor"
    ].tolist()
    print(f"[6/9] sensors: {R['sensors']['drift_pct']:.0f}% drift; "
          f"{R['sensors']['fail_all_sig']} raw hits -> "
          f"{len(R['sensors']['hits_surviving_confound_check'])} survive confound check")

    # Watchlist: everything with any tier-1/2 evidence, plus top regime movers.
    watch = list(
        dict.fromkeys(
            fail_stable.loc[fail_stable["q_mannwhitney"] < 0.05, "sensor"].tolist()
            + hits
            + regime.head(25)["sensor"].tolist()
        )
    )[:40]

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
    limits = pd.DataFrame(lim_rows)

    # ---------------------------------------------------------------- model
    X, y = clean[retained], is_fail
    model_rows = [r.as_row() for r in evaluate_shuffled(X, y, make_models(seed=SEED))]
    model_rows += [r.as_row() for r in evaluate_forward(X, y, make_models(seed=SEED), train_frac=TRAIN_FRAC)]
    models_df = pd.DataFrame(model_rows)

    rolling = []
    for frac in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85):
        r = evaluate_forward(X, y, {"logistic_l2": make_models(SEED)["logistic_l2"]}, train_frac=frac)[0]
        rolling.append(
            {"train_frac": frac, "n_test": r.n_test, "n_test_fail": r.n_test_fail,
             "BER": r.ber, "ROC_AUC": r.roc_auc, "PR_AUC": r.pr_auc}
        )
    rolling_df = pd.DataFrame(rolling)

    rng = np.random.default_rng(SEED)
    null = np.array(
        [
            evaluate_forward(X, rng.permutation(y), {"m": make_models(SEED)["logistic_l2"]}, train_frac=TRAIN_FRAC)[0].roc_auc
            for _ in range(60)
        ]
    )
    obs_auc = float(models_df.loc[
        (models_df["model"] == "logistic_l2")
        & (models_df["protocol"] == "forward_chronological"), "ROC_AUC"].iloc[0])

    cut = int(len(clean) * TRAIN_FRAC)
    mdl = make_models(SEED)["logistic_l2"]
    mdl.fit(X.iloc[:cut], y[:cut])
    scores_te = mdl.predict_proba(X.iloc[cut:])[:, 1]
    y_te = y[cut:]
    triage = []
    for k in (5, 10, 15, 20, 25, 30, 40, 50):
        nf = max(1, int(len(y_te) * k / 100))
        thr = np.sort(scores_te)[-nf]
        flagged = scores_te >= thr
        cap = int((flagged & y_te).sum())
        triage.append(
            {"top_k_pct": k, "n_flagged": int(flagged.sum()), "captured": cap,
             "capture_pct": float(100 * cap / y_te.sum()),
             "lift": float((cap / flagged.sum()) / y_te.mean())}
        )
    triage_df = pd.DataFrame(triage)

    R["model"] = {
        "results": model_rows,
        "rolling_origin": rolling,
        "rolling_mean_auc": float(rolling_df["ROC_AUC"].mean()),
        "rolling_sd_auc": float(rolling_df["ROC_AUC"].std()),
        "permutation": {
            "observed_auc": obs_auc,
            "null_mean": float(null.mean()),
            "null_sd": float(null.std()),
            "p_value": float((null >= obs_auc).mean()),
            "n_perm": int(len(null)),
        },
        "triage": triage,
        "best_lift": float(triage_df["lift"].max()),
    }
    shuffled_ber = float(models_df.loc[
        (models_df["model"] == "logistic_l2")
        & (models_df["protocol"].str.contains("shuffled")), "BER"].iloc[0])
    forward_ber = float(models_df.loc[
        (models_df["model"] == "logistic_l2")
        & (models_df["protocol"] == "forward_chronological"), "BER"].iloc[0])
    R["model"]["ber_shuffled"] = shuffled_ber
    R["model"]["ber_forward"] = forward_ber
    print(f"[7/9] model: BER shuffled {shuffled_ber:.3f} -> forward {forward_ber:.3f}; "
          f"AUC {obs_auc:.3f} vs null {null.mean():.3f}")

    # -------------------------------------------------------------- figures
    _figures(clean, wkly, naive, laney, cusum_all, m_pre, m_base, m_post,
             audit, regime, fail_all, fail_stable, confound, rolling_df,
             null, obs_auc, triage_df, models_df, R)
    print("[8/9] figures written")

    # -------------------------------------------------------------- exports
    phase = pd.Series(
        np.select([m_pre, m_base], ["0 - pre-baseline", "1 - Phase I baseline"],
                  default="2 - Phase II monitored"),
        index=clean.index,
    )
    dim_sensor = build_dim_sensor(
        audit, drift=drift, regime=regime, fail_all=fail_all,
        fail_stable=fail_stable, limits=limits, watchlist=watch,
    )
    fact_run = build_fact_run(clean, cusum=cusum_all.table, model_score=None, phase=phase)
    fact_run.loc[fact_run.index[cut:], "model_score"] = scores_te
    yield_week = laney.table.rename(columns={"subgroup": "year_week"}).copy()
    yield_week["naive_ucl"] = naive.table["ucl"].to_numpy()
    yield_week["naive_violation"] = naive.table["violation"].to_numpy()
    # Phase I-centred monitoring limits: this is what the DAX must reproduce.
    yield_week["center_phase1"] = laney_p1.table["center"].to_numpy()
    yield_week["lcl_phase1"] = laney_p1.table["lcl"].to_numpy()
    yield_week["ucl_phase1"] = laney_p1.table["ucl"].to_numpy()
    yield_week["violation_phase1"] = laney_p1.table["violation"].to_numpy()

    tables = {
        "dim_date": build_dim_date(clean),
        "dim_sensor": dim_sensor,
        "fact_run": fact_run,
        "fact_measurement": build_fact_measurement(clean, watch, limits),
        "fact_yield_week": yield_week,
        "fact_gchart": gch.table,
    }
    manifest = write_all(POWERBI_DATA, tables)
    clean.to_csv(PROCESSED / "secom_clean_wide.csv", index=False,
                 date_format="%Y-%m-%dT%H:%M:%S")
    audit.to_csv(PROCESSED / "sensor_audit.csv")
    models_df.to_csv(PROCESSED / "model_results.csv", index=False)
    confound.to_csv(PROCESSED / "confound_check.csv", index=False)
    impute_log.to_csv(PROCESSED / "imputation_log.csv", index=False)

    R["watchlist"] = watch
    R["exports"] = manifest.to_dict("records")
    (ROOT / "reports" / "results.json").write_text(
        json.dumps(R, indent=2, default=str), encoding="utf-8"
    )
    print("[9/9] exports written to data/powerbi/ and data/processed/")
    print(manifest.to_string(index=False))
    return 0


def _figures(clean, wkly, naive, laney, cusum_all, m_pre, m_base, m_post,
             audit, regime, fail_all, fail_stable, confound, rolling_df,
             null, obs_auc, triage_df, models_df, R) -> None:
    # Fig 1: weekly p-chart, naive vs Laney
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    x = np.arange(len(wkly))
    ax.fill_between(x, naive.table["lcl"], naive.table["ucl"], color=ACCENT, alpha=0.10,
                    step="mid", label="naive p-chart limits")
    ax.step(x, laney.table["ucl"], where="mid", color=INK, lw=1.2, ls="--",
            label=f"Laney p' UCL (σ_z={laney.sigma_z:.2f})")
    ax.step(x, naive.table["ucl"], where="mid", color=ACCENT, lw=1.0, alpha=0.7)
    ax.axhline(laney.p_bar, color=MUTED, lw=1, label=f"p̄ = {laney.p_bar:.3f}")
    ax.plot(x, wkly["d"] / wkly["n"], "o-", color=INK, ms=4.5, lw=1.3, label="weekly fail rate")
    viol = naive.table["violation"].to_numpy()
    ax.plot(x[viol], (wkly["d"] / wkly["n"])[viol], "o", ms=10, mfc="none",
            mec=ACCENT, mew=1.8, label="flagged by naive chart only")
    ax.set_xticks(x); ax.set_xticklabels(wkly["year_week"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("failure rate"); ax.set_ylim(0, None)
    ax.set_title("Weekly failure rate: overdispersion removes the only alarm")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.grid(axis="y", lw=0.6)
    fig.tight_layout(); fig.savefig(FIGS / "fig01_yield_pchart.png"); plt.close(fig)

    # Fig 2: CUSUM
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    t = clean["timestamp"]
    ax.plot(t, cusum_all.table["cusum"], color=INK, lw=0.9)
    ax.axhline(cusum_all.h, color=ACCENT, lw=1.2, ls="--",
               label=f"decision interval h = {cusum_all.h:.2f}  (ARL₀ ≈ {R['cusum']['arl0_runs']:.0f} runs)")
    for mask, colour, lab in ((m_pre, ACCENT, "pre-baseline"), (m_base, GOOD, "Phase I baseline"),
                              (m_post, "#3a5f9e", "Phase II monitored")):
        ax.axvspan(t[mask].min(), t[mask].max(), color=colour, alpha=0.07, label=lab)
    sig = cusum_all.table["signal"].to_numpy()
    ax.plot(t[sig], cusum_all.table["cusum"][sig], "v", color=ACCENT, ms=5,
            label=f"alarm ({sig.sum()} runs)")
    det = R["cusum"]["detection"]
    for mask, key, lab in ((m_pre, "pre_baseline", "W29–34"),
                           (m_base, "baseline_self_check", "W35–39"),
                           (m_post, "phase_ii", "W40–42")):
        centre = t[mask].iloc[int(mask.sum() // 2)]
        ax.text(centre, 2.60,
                f"{lab}\n{det[key]['rate']:.1%} fail · {det[key]['alarm_episodes']} alarms",
                ha="center", va="bottom", fontsize=6.8, color=MUTED)
    ax.set_ylim(0, 3.25)
    ax.set_ylabel("CUSUM statistic"); ax.set_title(
        "Bernoulli CUSUM on run-level pass/fail — signals what the p-chart cannot")
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left"); ax.grid(axis="y", lw=0.6)
    fig.autofmt_xdate()
    fig.tight_layout(); fig.savefig(FIGS / "fig02_cusum.png"); plt.close(fig)

    # Fig 3: evidence funnel
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    stages = ["raw sensors", "survive audit", "shift with\nyield regime",
              "associate with\nfailure (all runs)", "survive time-\nconfound check"]
    counts = [R["dataset"]["n_sensors_raw"], R["audit"]["n_retained"],
              R["sensors"]["regime_sig"], R["sensors"]["fail_all_sig"],
              len(R["sensors"]["hits_surviving_confound_check"])]
    bars = ax.bar(stages, counts, color=[MUTED, MUTED, "#c8a57d", ACCENT, GOOD], width=0.62)
    for b, c in zip(bars, counts, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, c + 8, str(c), ha="center", fontweight="bold")
    ax.set_yscale("symlog"); ax.set_ylabel("sensors (log scale)")
    ax.set_title("Evidence funnel: 590 candidate signals reduce to one defensible finding")
    ax.tick_params(axis="x", labelsize=7.5); ax.grid(axis="y", lw=0.6)
    fig.tight_layout(); fig.savefig(FIGS / "fig03_evidence_funnel.png"); plt.close(fig)

    # Fig 4: the surviving sensor
    surv = R["sensors"]["hits_surviving_confound_check"]
    if surv:
        s = surv[0]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.3),
                                     gridspec_kw={"width_ratios": [2.1, 1]})
        v = clean[s].to_numpy(); f = clean["is_fail"].to_numpy()
        a1.plot(clean["timestamp"], v, lw=0.6, color=MUTED, zorder=1)
        a1.scatter(clean["timestamp"][f], v[f], s=16, color=ACCENT, zorder=3, label="fail")
        a1.set_title(f"{s} over the campaign"); a1.set_ylabel(f"{s} value")
        a1.legend(fontsize=7, frameon=False); a1.grid(axis="y", lw=0.6)
        groups, labels = [], []
        for lab, m in (("W29–34", m_pre), ("W35–39", m_base), ("W40–42", m_post)):
            groups += [v[m & ~f], v[m & f]]; labels += [f"{lab}\npass", f"{lab}\nfail"]
        bp = a2.boxplot(groups, tick_labels=labels, showfliers=False, patch_artist=True, widths=0.6)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(ACCENT if i % 2 else "#dddddd"); patch.set_edgecolor(INK)
        for med in bp["medians"]: med.set_color(INK)
        a2.tick_params(axis="x", labelsize=6.5); a2.grid(axis="y", lw=0.6)
        a2.set_title("separation by regime")
        fig.tight_layout(); fig.savefig(FIGS / "fig04_surviving_sensor.png"); plt.close(fig)

    # Fig 5: protocol effect
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.2))
    sub = models_df[models_df["model"] == "logistic_l2"]
    a1.bar(["shuffled\n10-fold CV", "forward\nchronological"],
           [sub[sub.protocol.str.contains('shuffled')]["ROC_AUC"].iloc[0],
            sub[sub.protocol == "forward_chronological"]["ROC_AUC"].iloc[0]],
           color=[ACCENT, INK], width=0.5)
    a1.axhline(0.5, color=MUTED, ls="--", lw=1); a1.text(1.45, 0.51, "chance", fontsize=7, color=MUTED)
    a1.set_ylim(0.4, 0.8); a1.set_ylabel("ROC AUC")
    a1.set_title("Shuffling the split buys apparent skill"); a1.grid(axis="y", lw=0.6)
    a2.hist(null, bins=16, color="#dddddd", edgecolor=MUTED,
            label=f"permuted labels (n={len(null)})")
    a2.axvline(obs_auc, color=ACCENT, lw=2, label=f"observed = {obs_auc:.3f}")
    a2.set_xlabel("forward-split ROC AUC"); a2.set_title("…but the residual signal is real")
    a2.legend(fontsize=7, frameon=False); a2.grid(axis="y", lw=0.6)
    fig.tight_layout(); fig.savefig(FIGS / "fig05_protocol.png"); plt.close(fig)

    # Fig 6: triage capture curve
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(triage_df["top_k_pct"], triage_df["capture_pct"], "o-", color=INK, label="model triage")
    ax.plot([0, 100], [0, 100], ls="--", color=MUTED, lw=1, label="random review")
    ax.set_xlabel("% of runs sent for engineering review")
    ax.set_ylabel("% of failures captured")
    ax.set_title("Review effort vs failures caught — barely better than random")
    ax.legend(fontsize=7, frameon=False); ax.grid(lw=0.6)
    fig.tight_layout(); fig.savefig(FIGS / "fig06_triage.png"); plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
