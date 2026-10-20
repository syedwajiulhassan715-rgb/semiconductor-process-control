"""PCA multivariate SPC (T² and SPE) on the retained sensors, fitted on Phase I.

Two limit schemes are reported. The theoretical F / Jackson-Mudholkar limits
assume multivariate normality; the primary scheme instead calibrates each limit
as the (1 - alpha) quantile of out-of-fold baseline scores, because the
theoretical limits turn out to be far too tight on these heavy-tailed sensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

from ...config import register_section
from ...mspc import (
    MSPCScores,
    PCAModel,
    contribution_share,
    cross_validated_scores,
    empirical_limit,
    fit_pca_model,
    score,
)
from ..context import PipelineContext
from ..plotting import ACCENT, BLUE, GOOD, INK, LIGHT, MUTED, SAND

NAME = "multivariate"
REQUIRES = (
    "audit",
    "clean",
    "retained",
    "is_fail",
    "masks",
    "drift",
    "regime",
    "cusum_design",
)
PRODUCES = ("mspc_model", "mspc_scores", "mspc_alarms")


@register_section("mspc")
@dataclass(frozen=True)
class MSPCConfig:
    #: cumulative baseline variance the retained components must explain
    variance_target: float = 0.80
    #: per-chart false-alarm probability. Not 0.0027: with out-of-fold baseline
    #: kurtosis in the hundreds, the 99.73% quantile of 717 runs rests on
    #: about two points and cannot be estimated (see D27).
    alpha: float = 0.01
    #: contiguous time blocks used to score the baseline out of fold
    cv_folds: int = 5
    #: variance targets for the component-count sensitivity check
    sensitivity_targets: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    #: alternative alphas for the empirical limits, as a robustness check
    alpha_sensitivity: tuple[float, ...] = (0.05, 0.02, 0.01, 0.005)
    #: sensors listed in the contribution ranking
    top_n: int = 15
    #: |drift rho| at or above which a sensor counts as drifting in fig08
    drift_highlight: float = 0.3


REGIMES = (("pre", "pre_baseline"), ("base", "phase_i_baseline"), ("post", "phase_ii"))
CHARTS = ("t2", "spe", "either")


@dataclass(frozen=True)
class Scheme:
    """Monitoring statistics, limits and alarms under one limit scheme."""

    t2: np.ndarray
    spe: np.ndarray
    t2_limit: np.ndarray  # per run, so the baseline can use its own limit
    spe_limit: float
    alarms: dict[str, np.ndarray]


def _scheme(t2: np.ndarray, spe: np.ndarray, t2_limit: np.ndarray, spe_limit: float) -> Scheme:
    t2_alarm = t2 > t2_limit
    spe_alarm = spe > spe_limit
    return Scheme(
        t2,
        spe,
        t2_limit,
        spe_limit,
        {"t2": t2_alarm, "spe": spe_alarm, "either": t2_alarm | spe_alarm},
    )


def _theoretical(model: PCAModel, sc: MSPCScores, base: np.ndarray) -> Scheme:
    """Textbook limits; baseline points are in-sample and use the Beta T² limit."""
    t2_lim = np.where(base, model.t2_limit_phase1, model.t2_limit)
    return _scheme(sc.t2, sc.spe, t2_lim, model.spe_limit)


def _empirical(
    sc: MSPCScores, oof_t2: np.ndarray, oof_spe: np.ndarray, base: np.ndarray, alpha: float
) -> Scheme:
    """Limits from out-of-fold baseline quantiles; baseline rows scored out of fold."""
    t2 = sc.t2.copy()
    spe = sc.spe.copy()
    t2[base], spe[base] = oof_t2, oof_spe
    t2_lim = np.full(len(t2), empirical_limit(oof_t2, alpha))
    return _scheme(t2, spe, t2_lim, empirical_limit(oof_spe, alpha))


def _fisher(alarm: np.ndarray, fail: np.ndarray) -> dict[str, Any]:
    """Fail rate among alarmed vs quiet runs, and Fisher's exact test."""
    table = np.array(
        [
            [int((alarm & fail).sum()), int((alarm & ~fail).sum())],
            [int((~alarm & fail).sum()), int((~alarm & ~fail).sum())],
        ]
    )
    odds, p = sps.fisher_exact(table)
    n_alarm, n_quiet = int(table[0].sum()), int(table[1].sum())
    return {
        "n_alarmed": n_alarm,
        "n_quiet": n_quiet,
        "fail_rate_alarmed": float(table[0, 0] / n_alarm) if n_alarm else None,
        "fail_rate_quiet": float(table[1, 0] / n_quiet) if n_quiet else None,
        "odds_ratio": float(odds) if np.isfinite(odds) else None,
        "p_fisher": float(p),
    }


def _first_signal(alarm: np.ndarray, ts: np.ndarray) -> dict[str, Any]:
    idx = int(np.argmax(alarm)) if alarm.any() else None
    return {
        "runs_to_signal": None if idx is None else idx + 1,
        "days_to_signal": None
        if idx is None
        else float((ts[idx] - ts[0]) / np.timedelta64(1, "D")),
    }


def _regimes(
    scheme: Scheme, masks: dict[str, np.ndarray], fail: np.ndarray, ts: np.ndarray
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, label in REGIMES:
        m = masks[key]
        a = scheme.alarms
        out[label] = {
            "n_runs": int(m.sum()),
            "fail_rate": float(fail[m].mean()),
            **{f"{c}_alarm_rate": float(a[c][m].mean()) for c in CHARTS},
            "first_signal": {c: _first_signal(a[c][m], ts[m]) for c in CHARTS},
            "alarm_vs_failure": _fisher(a["either"][m], fail[m]),
        }
    return out


def _contributions(
    sc: MSPCScores,
    scheme: Scheme,
    rows_ok: np.ndarray,
    sensors: list[str],
    drift: pd.Series,
    regime_delta: pd.Series,
    kurtosis: pd.Series,
    top_n: int,
) -> tuple[dict[str, Any], dict[str, pd.Series]]:
    out: dict[str, Any] = {}
    shares: dict[str, pd.Series] = {}
    for stat, c in (("spe", sc.spe_contrib), ("t2", sc.t2_contrib)):
        rows = scheme.alarms[stat] & rows_ok
        share = pd.Series(contribution_share(c, rows), index=sensors)
        # per-run largest single-sensor share, to tell one-sensor outlier
        # events from broad multivariate departures
        cr = np.clip(c[rows], 0.0, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            max_share = cr.max(axis=1) / cr.sum(axis=1) if len(cr) else np.array([])
        shares[stat] = share
        ranked = share.sort_values(ascending=False)
        rho_drift = sps.spearmanr(share, drift.loc[sensors])
        rho_regime = sps.spearmanr(share, regime_delta.loc[sensors])
        top = ranked.head(top_n)
        out[stat] = {
            "n_alarmed_runs_outside_baseline": int(rows.sum()),
            "uniform_share": 1.0 / len(sensors),
            "top": [
                {
                    "sensor": s,
                    "share": float(v),
                    "abs_drift_rho": float(drift[s]),
                    "abs_regime_cliffs_delta": float(regime_delta[s]),
                    "excess_kurtosis": float(kurtosis[s]),
                }
                for s, v in zip(top.index.astype(str), top.to_numpy(), strict=True)
            ],
            "median_kurtosis_top": float(kurtosis.loc[top.index].median()),
            "median_kurtosis_all": float(kurtosis.loc[sensors].median()),
            "median_max_single_sensor_share": float(np.median(max_share))
            if len(max_share)
            else None,
            "frac_runs_one_sensor_over_half": float(np.mean(max_share > 0.5))
            if len(max_share)
            else None,
            "top_share_total": float(top.sum()),
            "spearman_share_vs_abs_drift": float(rho_drift.statistic),
            "spearman_share_vs_abs_drift_p": float(rho_drift.pvalue),
            "spearman_share_vs_abs_regime_delta": float(rho_regime.statistic),
            "spearman_share_vs_abs_regime_delta_p": float(rho_regime.pvalue),
            "S060_rank": list(ranked.index).index("S060") + 1 if "S060" in ranked.index else None,
            "S060_share": float(share["S060"]) if "S060" in share.index else None,
        }
    return out, shares


def run(ctx: PipelineContext) -> None:
    cfg = ctx.config.section(MSPCConfig)
    clean: pd.DataFrame = ctx["clean"]
    retained: list[str] = ctx["retained"]
    fail: np.ndarray = ctx["is_fail"]
    masks: dict[str, np.ndarray] = ctx["masks"]
    base = masks["base"]
    ts = clean["timestamp"].to_numpy()
    x_all = clean[retained].to_numpy(dtype=float)

    def build(
        target: float,
    ) -> tuple[PCAModel, MSPCScores, Scheme, Scheme, tuple[np.ndarray, np.ndarray]]:
        model = fit_pca_model(x_all[base], retained, variance_target=target, alpha=cfg.alpha)
        x = x_all[:, [retained.index(s) for s in model.sensors]]
        sc = score(model, x)
        oof = cross_validated_scores(
            x[base], model.sensors, n_components=model.n_components, n_folds=cfg.cv_folds
        )
        theo = _theoretical(model, sc, base)
        emp = _empirical(sc, oof[0], oof[1], base, cfg.alpha)
        return model, sc, theo, emp, oof

    model, sc, theo, emp, emp_oof = build(cfg.variance_target)
    regimes = {
        "empirical_cv": _regimes(emp, masks, fail, ts),
        "theoretical": _regimes(theo, masks, fail, ts),
    }

    drift = ctx["drift"].set_index("sensor_id")["drift_rho"].abs()
    regime_delta = ctx["regime"].set_index("sensor")["cliffs_delta"].abs()
    kurtosis = ctx["audit"]["excess_kurtosis"]
    contrib, shares = _contributions(
        sc, emp, ~base, model.sensors, drift, regime_delta, kurtosis, cfg.top_n
    )

    run_index = clean["run_index"].to_numpy()
    trend = {
        stat: {
            "spearman_vs_run_index_all": float(sps.spearmanr(run_index, v).statistic),
            "spearman_vs_run_index_baseline": float(
                sps.spearmanr(run_index[base], v[base]).statistic
            ),
        }
        for stat, v in (("t2", emp.t2), ("spe", emp.spe))
    }

    sensitivity = []
    for target in cfg.sensitivity_targets:
        m2, _, theo2, emp2, _ = build(target)
        row: dict[str, Any] = {"variance_target": target, "n_components": m2.n_components}
        for key, label in REGIMES:
            row[f"either_alarm_rate_{label}"] = float(emp2.alarms["either"][masks[key]].mean())
        row["theoretical_baseline_alarm_rate"] = float(theo2.alarms["either"][base].mean())
        for key, label in REGIMES:
            fr = _fisher(emp2.alarms["either"][masks[key]], fail[masks[key]])
            row[f"fisher_or_{label}"] = fr["odds_ratio"]
            row[f"fisher_p_{label}"] = fr["p_fisher"]
        sensitivity.append(row)

    alpha_sensitivity = []
    for a in cfg.alpha_sensitivity:
        e = _empirical(sc, emp_oof[0], emp_oof[1], base, a)
        row = {"alpha": a, "t2_limit": float(e.t2_limit[0]), "spe_limit": e.spe_limit}
        for key, label in REGIMES:
            mk = masks[key]
            fr = _fisher(e.alarms["either"][mk], fail[mk])
            row[f"either_alarm_rate_{label}"] = float(e.alarms["either"][mk].mean())
            row[f"fisher_or_{label}"] = fr["odds_ratio"]
            row[f"fisher_p_{label}"] = fr["p_fisher"]
        alpha_sensitivity.append(row)

    # How far does each regime's statistic sit from the out-of-fold baseline?
    shift: dict[str, Any] = {}
    for key, label in (("pre", "pre_baseline"), ("post", "phase_ii")):
        shift[label] = {}
        for stat, v in (("t2", emp.t2), ("spe", emp.spe)):
            a_, b_ = v[masks[key]], v[base]
            u = sps.mannwhitneyu(a_, b_, alternative="two-sided")
            shift[label][stat] = {
                "median": float(np.median(a_)),
                "baseline_median": float(np.median(b_)),
                "cliffs_delta": float(2 * u.statistic / (len(a_) * len(b_)) - 1),
                "p_mannwhitney": float(u.pvalue),
            }

    oof_t2, oof_spe = emp_oof
    oof_tail = {
        stat: {
            "excess_kurtosis": float(sps.kurtosis(v)),
            "max": float(v.max()),
            "quantiles": {str(q): float(np.quantile(v, q)) for q in (0.5, 0.9, 0.95, 0.99, 0.9973)},
            "n_above_limit": int((v > lim).sum()),
        }
        for stat, v, lim in (
            ("t2", oof_t2, float(emp.t2_limit[0])),
            ("spe", oof_spe, emp.spe_limit),
        )
    }

    cusum_pre = ctx.results.get("cusum", {}).get("detection", {}).get("pre_baseline", {})
    lam = model.eigenvalues
    ctx.results["multivariate"] = {
        "model": {
            "n_sensors": len(model.sensors),
            "dropped_constant_in_baseline": model.dropped_constant,
            "n_baseline": model.n_baseline,
            "baseline_rank": int(np.sum(lam > lam[0] * 1e-12)),
            "smallest_eigenvalue": float(lam[-1]),
            "variance_target": cfg.variance_target,
            "n_components": model.n_components,
            "explained": model.explained,
            "first_component_explained": float(lam[0] / lam.sum()),
            "kaiser_components": int(np.sum(lam > 1)),
            "retained_condition_number": model.condition_number,
            "alpha": model.alpha,
            "cv_folds": cfg.cv_folds,
        },
        "limits": {
            "theoretical": {
                "t2_phase2_F": model.t2_limit,
                "t2_phase1_beta": model.t2_limit_phase1,
                "spe": model.spe_limit,
                "spe_method": model.spe_limit_method,
            },
            "empirical_cv": {"t2": float(emp.t2_limit[0]), "spe": emp.spe_limit},
            "out_of_fold_tail": oof_tail,
        },
        "nominal_alarm_rate": {"per_chart": cfg.alpha, "either_max": 2 * cfg.alpha},
        "regimes": regimes,
        "shift_vs_baseline": shift,
        "cusum_pre_baseline_runs_to_signal": cusum_pre.get("runs_to_signal"),
        "alarm_vs_failure_all_runs": _fisher(emp.alarms["either"], fail),
        "trend": trend,
        "contributions": contrib,
        "sensitivity": sensitivity,
        "alpha_sensitivity": alpha_sensitivity,
    }
    ctx.artifacts["mspc_model"] = model
    ctx.artifacts["mspc_scores"] = sc
    ctx.artifacts["mspc_alarms"] = emp.alarms

    r = regimes["empirical_cv"]
    ctx.log(
        f"A={model.n_components} ({model.explained:.0%}); either-alarm rate "
        f"pre {r['pre_baseline']['either_alarm_rate']:.1%}, "
        f"base {r['phase_i_baseline']['either_alarm_rate']:.1%}, "
        f"post {r['phase_ii']['either_alarm_rate']:.1%}"
    )

    if ctx.figures_enabled:
        _figures(ctx, clean, model, theo, emp, masks, fail, shares, sensitivity, drift)


def _shade(ax: Any, t: pd.Series, masks: dict[str, np.ndarray]) -> None:
    for key, colour, lab in (
        ("pre", ACCENT, "pre-baseline"),
        ("base", GOOD, "Phase I baseline"),
        ("post", BLUE, "Phase II monitored"),
    ):
        ax.axvspan(t[masks[key]].min(), t[masks[key]].max(), color=colour, alpha=0.07, label=lab)


def _figures(
    ctx: PipelineContext,
    clean: pd.DataFrame,
    model: PCAModel,
    theo: Scheme,
    emp: Scheme,
    masks: dict[str, np.ndarray],
    fail: np.ndarray,
    shares: dict[str, pd.Series],
    sensitivity: list[dict[str, Any]],
    drift: pd.Series,
) -> None:
    import matplotlib.pyplot as plt

    cfg = ctx.config.section(MSPCConfig)
    res = ctx.results["multivariate"]
    rg = res["regimes"]["empirical_cv"]
    rg_theo = res["regimes"]["theoretical"]["phase_i_baseline"]
    t = clean["timestamp"]

    # Fig 7: T² and SPE across the campaign, both limit schemes
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.2, 5.2), sharex=True)
    for ax, stat, lim, lim_theo, lab in (
        (a1, emp.t2, float(emp.t2_limit[0]), model.t2_limit, "Hotelling T²"),
        (a2, emp.spe, emp.spe_limit, model.spe_limit, "SPE (Q)"),
    ):
        _shade(ax, t, masks)
        ax.plot(t, stat, color=MUTED, lw=0.6, zorder=2)
        ax.scatter(t[fail], stat[fail], s=10, color=ACCENT, zorder=3, label="failed run")
        ax.axhline(lim, color=INK, lw=1.2, ls="--", label=f"cross-validated limit ({lim:.0f})")
        ax.axhline(lim_theo, color=INK, lw=0.8, ls=":", label=f"theoretical limit ({lim_theo:.0f})")
        ax.set_yscale("log")
        ax.set_ylabel(lab)
        ax.grid(axis="y", lw=0.6)
    a1.set_title(
        f"PCA monitor, {len(model.sensors)} sensors, A = {model.n_components}: calibrated "
        f"limits flag {rg['pre_baseline']['either_alarm_rate']:.0%} of degraded-period runs"
    )
    a1.legend(fontsize=6.5, frameon=False, ncol=3, loc="upper right")
    for key, label in REGIMES:
        centre = t[masks[key]].iloc[int(masks[key].sum() // 2)]
        a2.text(
            centre,
            0.04,
            f"{rg[label]['either_alarm_rate']:.0%} alarm · {rg[label]['fail_rate']:.1%} fail",
            transform=a2.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7,
            color=INK,
        )
    a2.text(
        0.995,
        0.97,
        f"theoretical limits: {rg_theo['either_alarm_rate']:.0%} of baseline runs alarm "
        f"(nominal ≤ {2 * model.alpha:.2%})",
        transform=a2.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        color=MUTED,
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    ctx.save_figure(fig, "fig07_mspc_t2_spe")

    # Fig 8: top SPE contributors outside the baseline, coloured by drift
    share = shares["spe"].sort_values(ascending=False).head(cfg.top_n)[::-1]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    hi = cfg.drift_highlight
    colours = [
        ACCENT if drift[s] >= hi else SAND if drift[s] >= hi / 3 else LIGHT for s in share.index
    ]
    ax.barh(share.index, 100 * share.to_numpy(), color=colours, edgecolor=INK, lw=0.4)
    for y, s in enumerate(share.index):
        ax.text(
            100 * share[s] + 0.03,
            y,
            f"|ρ_drift| = {drift[s]:.2f}",
            va="center",
            fontsize=6.5,
            color=MUTED,
        )
    spe_c = res["contributions"]["spe"]
    ax.axvline(100 * spe_c["uniform_share"], color=MUTED, ls="--", lw=1, label="equal share")
    ax.set_xlabel("mean share of SPE in alarmed runs outside the baseline (%)")
    ax.set_title(
        f"SPE alarms: top {cfg.top_n} sensors carry {spe_c['top_share_total']:.0%}; "
        f"share vs |drift| ρ = {spe_c['spearman_share_vs_abs_drift']:.2f}"
    )
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.grid(axis="x", lw=0.6)
    fig.tight_layout()
    ctx.save_figure(fig, "fig08_mspc_contributions")

    # Fig 9: sensitivity to the number of components
    sens = pd.DataFrame(sensitivity)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for label, colour, name in (
        ("pre_baseline", ACCENT, "pre-baseline"),
        ("phase_i_baseline", GOOD, "Phase I baseline (out of fold)"),
        ("phase_ii", BLUE, "Phase II"),
    ):
        a1.plot(
            sens["n_components"],
            100 * sens[f"either_alarm_rate_{label}"],
            "o-",
            color=colour,
            label=name,
        )
    a1.plot(
        sens["n_components"],
        100 * sens["theoretical_baseline_alarm_rate"],
        "o:",
        color=MUTED,
        label="baseline, theoretical limits",
    )
    a1.set_xlabel("components retained (A)")
    a1.set_ylabel("runs alarming, T² or SPE (%)")
    a1.set_title("Alarm rate by regime")
    a1.legend(fontsize=6.5, frameon=False)
    a1.grid(lw=0.6)
    for label, colour, name in (
        ("pre_baseline", ACCENT, "pre-baseline"),
        ("phase_i_baseline", GOOD, "Phase I baseline"),
        ("phase_ii", BLUE, "Phase II"),
    ):
        a2.plot(sens["n_components"], sens[f"fisher_p_{label}"], "o-", color=colour, label=name)
    a2.axhline(0.05, color=MUTED, ls="--", lw=1, label="p = 0.05")
    a2.set_yscale("log")
    a2.set_xlabel("components retained (A)")
    a2.set_ylabel("Fisher p, alarm vs failure")
    a2.set_title("Do alarmed runs fail more often, within each regime?")
    a2.legend(fontsize=7, frameon=False)
    a2.grid(lw=0.6)
    fig.tight_layout()
    ctx.save_figure(fig, "fig09_mspc_sensitivity")
