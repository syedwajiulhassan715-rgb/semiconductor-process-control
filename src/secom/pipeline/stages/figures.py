"""The six core report figures, fig01-fig06.

New stages draw and save their own figures; this stage only owns these six.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..context import PipelineContext
from ..plotting import ACCENT, BLUE, GOOD, INK, LIGHT, MUTED, SAND

NAME = "figures"
REQUIRES = (
    "clean",
    "masks",
    "weekly",
    "naive",
    "laney",
    "cusum_all",
    "models_df",
    "perm_null",
    "obs_auc",
    "triage_df",
)
PRODUCES: tuple[str, ...] = ()


def run(ctx: PipelineContext) -> None:
    _yield_pchart(ctx)
    _cusum(ctx)
    _evidence_funnel(ctx)
    _surviving_sensor(ctx)
    _protocol(ctx)
    _triage(ctx)
    ctx.log(f"written to {ctx.paths.figures}")


def _yield_pchart(ctx: PipelineContext) -> None:
    wkly, naive, laney = ctx["weekly"], ctx["naive"], ctx["laney"]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    x = np.arange(len(wkly))
    ax.fill_between(
        x,
        naive.table["lcl"],
        naive.table["ucl"],
        color=ACCENT,
        alpha=0.10,
        step="mid",
        label="naive p-chart limits",
    )
    ax.step(
        x,
        laney.table["ucl"],
        where="mid",
        color=INK,
        lw=1.2,
        ls="--",
        label=f"Laney p' UCL (σ_z={laney.sigma_z:.2f})",
    )
    ax.step(x, naive.table["ucl"], where="mid", color=ACCENT, lw=1.0, alpha=0.7)
    ax.axhline(laney.p_bar, color=MUTED, lw=1, label=f"p̄ = {laney.p_bar:.3f}")
    ax.plot(x, wkly["d"] / wkly["n"], "o-", color=INK, ms=4.5, lw=1.3, label="weekly fail rate")
    viol = naive.table["violation"].to_numpy()
    ax.plot(
        x[viol],
        (wkly["d"] / wkly["n"])[viol],
        "o",
        ms=10,
        mfc="none",
        mec=ACCENT,
        mew=1.8,
        label="flagged by naive chart only",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(wkly["year_week"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("failure rate")
    ax.set_ylim(0, None)
    ax.set_title("Weekly failure rate: overdispersion removes the only alarm")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.grid(axis="y", lw=0.6)
    fig.tight_layout()
    ctx.save_figure(fig, "fig01_yield_pchart")


def _cusum(ctx: PipelineContext) -> None:
    clean, cusum_all, masks = ctx["clean"], ctx["cusum_all"], ctx["masks"]
    R = ctx.results
    arl0 = R["cusum"]["arl0_runs"]
    h_label = f"decision interval h = {cusum_all.h:.2f}  (ARL₀ ≈ {arl0:.0f} runs)"
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    t = clean["timestamp"]
    ax.plot(t, cusum_all.table["cusum"], color=INK, lw=0.9)
    ax.axhline(cusum_all.h, color=ACCENT, lw=1.2, ls="--", label=h_label)
    for mask, colour, lab in (
        (masks["pre"], ACCENT, "pre-baseline"),
        (masks["base"], GOOD, "Phase I baseline"),
        (masks["post"], BLUE, "Phase II monitored"),
    ):
        ax.axvspan(t[mask].min(), t[mask].max(), color=colour, alpha=0.07, label=lab)
    sig = cusum_all.table["signal"].to_numpy()
    ax.plot(
        t[sig],
        cusum_all.table["cusum"][sig],
        "v",
        color=ACCENT,
        ms=5,
        label=f"alarm ({sig.sum()} runs)",
    )
    det = R["cusum"]["detection"]
    for mask, key, lab in (
        (masks["pre"], "pre_baseline", "W29–34"),
        (masks["base"], "baseline_self_check", "W35–39"),
        (masks["post"], "phase_ii", "W40–42"),
    ):
        centre = t[mask].iloc[int(mask.sum() // 2)]
        ax.text(
            centre,
            2.60,
            f"{lab}\n{det[key]['rate']:.1%} fail · {det[key]['alarm_episodes']} alarms",
            ha="center",
            va="bottom",
            fontsize=6.8,
            color=MUTED,
        )
    ax.set_ylim(0, 3.25)
    ax.set_ylabel("CUSUM statistic")
    ax.set_title("Bernoulli CUSUM on run-level pass/fail — signals what the p-chart cannot")
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
    ax.grid(axis="y", lw=0.6)
    fig.autofmt_xdate()
    fig.tight_layout()
    ctx.save_figure(fig, "fig02_cusum")


def _evidence_funnel(ctx: PipelineContext) -> None:
    R = ctx.results
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    stages = [
        "raw sensors",
        "survive audit",
        "shift with\nyield regime",
        "associate with\nfailure (all runs)",
        "survive time-\nconfound check",
    ]
    counts = [
        R["dataset"]["n_sensors_raw"],
        R["audit"]["n_retained"],
        R["sensors"]["regime_sig"],
        R["sensors"]["fail_all_sig"],
        len(R["sensors"]["hits_surviving_confound_check"]),
    ]
    bars = ax.bar(stages, counts, color=[MUTED, MUTED, SAND, ACCENT, GOOD], width=0.62)
    for b, c in zip(bars, counts, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, c + 8, str(c), ha="center", fontweight="bold")
    ax.set_yscale("symlog")
    ax.set_ylabel("sensors (log scale)")
    ax.set_title("Evidence funnel: 590 candidate signals reduce to one defensible finding")
    ax.tick_params(axis="x", labelsize=7.5)
    ax.grid(axis="y", lw=0.6)
    fig.tight_layout()
    ctx.save_figure(fig, "fig03_evidence_funnel")


def _surviving_sensor(ctx: PipelineContext) -> None:
    surv = ctx.results["sensors"]["hits_surviving_confound_check"]
    if not surv:
        return
    clean, masks = ctx["clean"], ctx["masks"]
    s = surv[0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.3), gridspec_kw={"width_ratios": [2.1, 1]})
    v = clean[s].to_numpy()
    f = clean["is_fail"].to_numpy()
    a1.plot(clean["timestamp"], v, lw=0.6, color=MUTED, zorder=1)
    a1.scatter(clean["timestamp"][f], v[f], s=16, color=ACCENT, zorder=3, label="fail")
    a1.set_title(f"{s} over the campaign")
    a1.set_ylabel(f"{s} value")
    a1.legend(fontsize=7, frameon=False)
    a1.grid(axis="y", lw=0.6)
    groups, labels = [], []
    for lab, m in (("W29–34", masks["pre"]), ("W35–39", masks["base"]), ("W40–42", masks["post"])):
        groups += [v[m & ~f], v[m & f]]
        labels += [f"{lab}\npass", f"{lab}\nfail"]
    bp = a2.boxplot(groups, tick_labels=labels, showfliers=False, patch_artist=True, widths=0.6)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(ACCENT if i % 2 else LIGHT)
        patch.set_edgecolor(INK)
    for med in bp["medians"]:
        med.set_color(INK)
    a2.tick_params(axis="x", labelsize=6.5)
    a2.grid(axis="y", lw=0.6)
    a2.set_title("separation by regime")
    fig.tight_layout()
    ctx.save_figure(fig, "fig04_surviving_sensor")


def _protocol(ctx: PipelineContext) -> None:
    models_df, null, obs_auc = ctx["models_df"], ctx["perm_null"], ctx["obs_auc"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.2))
    sub = models_df[models_df["model"] == "logistic_l2"]
    a1.bar(
        ["shuffled\n10-fold CV", "forward\nchronological"],
        [
            sub[sub.protocol.str.contains("shuffled")]["ROC_AUC"].iloc[0],
            sub[sub.protocol == "forward_chronological"]["ROC_AUC"].iloc[0],
        ],
        color=[ACCENT, INK],
        width=0.5,
    )
    a1.axhline(0.5, color=MUTED, ls="--", lw=1)
    a1.text(1.45, 0.51, "chance", fontsize=7, color=MUTED)
    a1.set_ylim(0.4, 0.8)
    a1.set_ylabel("ROC AUC")
    a1.set_title("Shuffling the split buys apparent skill")
    a1.grid(axis="y", lw=0.6)
    a2.hist(null, bins=16, color=LIGHT, edgecolor=MUTED, label=f"permuted labels (n={len(null)})")
    a2.axvline(obs_auc, color=ACCENT, lw=2, label=f"observed = {obs_auc:.3f}")
    a2.set_xlabel("forward-split ROC AUC")
    a2.set_title("…but the residual signal is real")
    a2.legend(fontsize=7, frameon=False)
    a2.grid(axis="y", lw=0.6)
    fig.tight_layout()
    ctx.save_figure(fig, "fig05_protocol")


def _triage(ctx: PipelineContext) -> None:
    triage_df = ctx["triage_df"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(triage_df["top_k_pct"], triage_df["capture_pct"], "o-", color=INK, label="model triage")
    ax.plot([0, 100], [0, 100], ls="--", color=MUTED, lw=1, label="random review")
    ax.set_xlabel("% of runs sent for engineering review")
    ax.set_ylabel("% of failures captured")
    ax.set_title("Review effort vs failures caught — barely better than random")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(lw=0.6)
    fig.tight_layout()
    ctx.save_figure(fig, "fig06_triage")
