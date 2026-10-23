"""Detection power of every yield scheme against simulated step shifts in the fail rate."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ...config import register_section
from ...power import (
    SCHEMES,
    MonitorDesign,
    calibrate_h_for_false_alarm,
    exact_shewhart_detection,
    g_chart_first_signal,
    g_limits,
    side_for_ratio,
    simulate_detection,
)
from ...spc import calibrate_cusum_h
from ..context import PipelineContext
from ..plotting import ACCENT, BLUE, GOOD, MUTED, SAND

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

NAME = "shift_simulation"
REQUIRES = (
    "is_fail",
    "masks",
    "weekly",
    "laney",
    "laney_p1",
    "cusum_design",
    "cusum_lower_design",
)
PRODUCES = ("power_table", "monitor_design")


@register_section("shift_simulation")
@dataclass(frozen=True)
class ShiftSimulationConfig:
    #: Post-shift fail rate as a multiple of the Phase I rate p0 (1 = in control).
    shift_ratios: tuple[float, ...] = (0.25, 0.33, 0.5, 0.67, 1.0, 1.5, 2.0, 3.0, 4.0)
    #: Also simulate the ratios actually observed before and after Phase I.
    include_observed_ratios: bool = True
    #: Runs within which detection is scored; the last is the campaign length.
    horizons: tuple[int, ...] = (120, 400, 1567)
    #: Simulated streams per shift ratio.
    n_sim: int = 4000
    #: The lower (improvement) CUSUM is designed for p1 = improvement_ratio * p0.
    improvement_ratio: float = 0.5
    #: Shewhart limit width in sigmas.
    k: float = 3.0
    #: g-chart false-alarm probability, matched to a 3-sigma chart (D11).
    g_alpha: float = 0.0027


#: Schemes drawn in fig10/fig11, per panel.
_DEGRADATION = ("p_chart", "laney_p_prime", "cusum_upper", "cusum_upper_matched")
_IMPROVEMENT = ("p_chart", "laney_p_prime", "g_chart", "cusum_lower", "cusum_lower_matched")
_STYLE = {
    "p_chart": (ACCENT, "-", "o"),
    "laney_p_prime": (SAND, "-", "s"),
    "g_chart": (BLUE, "-", "D"),
    "cusum_upper": (GOOD, "-", "^"),
    "cusum_upper_matched": (GOOD, "--", "^"),
    "cusum_lower": (GOOD, "-", "v"),
    "cusum_lower_matched": (GOOD, "--", "v"),
}


def _design(ctx: PipelineContext, sc: ShiftSimulationConfig) -> tuple[MonitorDesign, dict]:
    cfg = ctx.config
    d = ctx["cusum_design"]
    p0 = float(d["p0"])
    sizes = tuple(int(v) for v in ctx["weekly"]["n"])
    horizon = int(max(sc.horizons))
    lower_p1 = sc.improvement_ratio * p0
    deployed_lower = ctx["cusum_lower_design"]
    if np.isclose(deployed_lower["p1"], lower_p1):
        # the cusum stage already calibrated this design on the real data
        lower_h, lower_arl0 = float(deployed_lower["h"]), float(deployed_lower["arl0"])
    else:
        lower_h, lower_arl0 = calibrate_cusum_h(
            p0,
            lower_p1,
            target_arl0=cfg.cusum.target_arl0,
            n_sim=cfg.cusum.n_sim,
            max_run=cfg.cusum.max_run,
            seed=cfg.seed,
        )
    g_lcl, g_ucl = g_limits(p0, sc.g_alpha)

    # False-alarm budgets the matched CUSUMs are held to, over the campaign.
    target_upper = exact_shewhart_detection(p0, np.array(sizes), horizon, p0=p0, k=sc.k)
    rng = np.random.default_rng(cfg.seed + 3)
    g_first = g_chart_first_signal(rng.random((sc.n_sim, horizon)) < p0, lcl=g_lcl, ucl=g_ucl)
    target_lower = float(np.mean((g_first > 0) & (g_first <= horizon)))
    upper_matched_h = calibrate_h_for_false_alarm(
        p0, float(d["p1"]), horizon=horizon, target=target_upper, n_sim=sc.n_sim, seed=cfg.seed + 1
    )
    lower_matched_h = calibrate_h_for_false_alarm(
        p0, lower_p1, horizon=horizon, target=target_lower, n_sim=sc.n_sim, seed=cfg.seed + 2
    )

    design = MonitorDesign(
        p0=p0,
        subgroup_sizes=sizes,
        sigma_z=float(ctx["laney"].sigma_z),
        sigma_z_phase1=float(ctx["laney_p1"].sigma_z),
        cusum_upper_p1=float(d["p1"]),
        cusum_upper_h=float(d["h"]),
        cusum_lower_p1=lower_p1,
        cusum_lower_h=float(lower_h),
        g_lcl=g_lcl,
        g_ucl=g_ucl,
        cusum_upper_matched_h=upper_matched_h,
        cusum_lower_matched_h=lower_matched_h,
        k=sc.k,
    )
    n_lcl = 9 * (1 - p0) / p0
    summary = {
        "p0": p0,
        "weekly_sizes": list(sizes),
        "max_weekly_n": int(max(sizes)),
        "n_needed_for_nonzero_lcl_at_p0": float(n_lcl),
        "weeks_with_lcl_zero_at_p0": int(sum(n < n_lcl for n in sizes)),
        "k": sc.k,
        "sigma_z": design.sigma_z,
        "sigma_z_phase1": design.sigma_z_phase1,
        "g_alpha": sc.g_alpha,
        "g_lcl": g_lcl,
        "g_ucl": g_ucl,
        "cusum_upper": {
            "p1": design.cusum_upper_p1,
            "h": design.cusum_upper_h,
            "arl0_runs": float(d["arl0"]),
        },
        "cusum_lower": {"p1": lower_p1, "h": design.cusum_lower_h, "arl0_runs": float(lower_arl0)},
        "cusum_upper_matched": {
            "h": upper_matched_h,
            "matched_to": "p_chart",
            "target_false_alarm": target_upper,
        },
        "cusum_lower_matched": {
            "h": lower_matched_h,
            "matched_to": "g_chart",
            "target_false_alarm": target_lower,
        },
    }
    return design, summary


def _row(table: pd.DataFrame, scheme: str, ratio: float) -> pd.Series:
    sel = table[(table["scheme"] == scheme) & np.isclose(table["ratio"], ratio)]
    return sel.iloc[0]


def run(ctx: PipelineContext) -> None:
    sc = ctx.config.section(ShiftSimulationConfig)
    is_fail, masks = ctx["is_fail"], ctx["masks"]
    design, summary = _design(ctx, sc)
    p0 = design.p0

    observed = {
        "pre_baseline": float(is_fail[masks["pre"]].mean() / p0),
        "phase_ii": float(is_fail[masks["post"]].mean() / p0),
    }
    ratios = list(sc.shift_ratios)
    if sc.include_observed_ratios:
        ratios += [r for r in observed.values() if not np.isclose(ratios, r).any()]
    ratios = sorted(ratios)
    horizons = tuple(sorted(sc.horizons))
    horizon = horizons[-1]

    table = simulate_detection(
        design, tuple(ratios), horizons=horizons, n_sim=sc.n_sim, seed=ctx.config.seed
    )

    # Every Shewhart detection probability has an exact binomial value.
    scales = {
        "p_chart": 1.0,
        "laney_p_prime": design.sigma_z,
        "laney_p_prime_phase1_sigma": design.sigma_z_phase1,
    }
    sizes = np.array(design.subgroup_sizes)
    diffs = [
        abs(
            float(_row(table, scheme, r)[f"p_detect_{h}"])
            - exact_shewhart_detection(
                r * p0, sizes, h, p0=p0, k=sc.k, scale=scale, side=side_for_ratio(r)
            )
        )
        for scheme, scale in scales.items()
        for r in ratios
        for h in horizons
    ]

    def pick(ratio: float, schemes: tuple[str, ...], h: int) -> dict[str, Any]:
        return {
            s: {
                f"p_detect_{h}": float(_row(table, s, ratio)[f"p_detect_{h}"]),
                "median_run_length": _none(_row(table, s, ratio)["median_run_length"]),
            }
            for s in schemes
        }

    shewhart = ("p_chart", "laney_p_prime", "laney_p_prime_phase1_sigma")
    improvement_power = table[(table["ratio"] < 1) & table["scheme"].isin(shewhart)]
    ctx.results["shift_simulation"] = {
        "convention": (
            "zero-state step shift at run 1; weekly subgroups cycle through the "
            "observed weekly sizes from a random start week; detection probability "
            "is P(first signal in the direction of the shift within h runs); at "
            "ratio 1 any signal counts, as a false alarm"
        ),
        "n_sim": int(sc.n_sim),
        "seed": int(ctx.config.seed),
        "horizons": list(horizons),
        "ratios": ratios,
        "observed_ratios": observed,
        "design": summary,
        "schemes": {k: {"label": v[0], "detects": v[1]} for k, v in SCHEMES.items()},
        "exact_check": {
            "n_compared": len(diffs),
            "max_abs_diff": float(max(diffs)),
            "mc_se_max": float(0.5 / np.sqrt(sc.n_sim)),
        },
        "headline": {
            "shewhart_improvement_max_p_detect": float(
                improvement_power[[f"p_detect_{h}" for h in horizons]].to_numpy().max()
            ),
            "in_control": pick(1.0, tuple(SCHEMES), horizon),
            "degradation_2x": pick(2.0, tuple(SCHEMES), 400),
            "degradation_observed_pre": pick(observed["pre_baseline"], _DEGRADATION, 120),
            "improvement_half": pick(0.5, _IMPROVEMENT, horizon),
        },
        "rows": [{k: _none(v) for k, v in rec.items()} for rec in table.to_dict(orient="records")],
    }
    ctx.artifacts["power_table"] = table
    ctx.artifacts["monitor_design"] = design
    h2 = ctx.results["shift_simulation"]["headline"]["degradation_2x"]
    ctx.log(
        f"2x shift, P(detect in 400 runs): p-chart {h2['p_chart']['p_detect_400']:.2f}, "
        f"CUSUM {h2['cusum_upper']['p_detect_400']:.2f}, matched CUSUM "
        f"{h2['cusum_upper_matched']['p_detect_400']:.2f}; exact check "
        f"max |diff| {max(diffs):.4f}"
    )
    if ctx.figures_enabled:
        _figures(ctx, table, observed, horizons, tuple(sc.shift_ratios))


def _none(v: Any) -> Any:
    """NaN/None -> None so results.json holds null rather than NaN."""
    if v is None:
        return None
    if isinstance(v, float | np.floating) and not np.isfinite(v):
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def _panel_lines(
    ax: Axes,
    table: pd.DataFrame,
    schemes: tuple[str, ...],
    ratios_mask: Callable[[pd.Series], pd.Series],
    ycol: str,
) -> None:
    for s in schemes:
        sub = table[(table["scheme"] == s) & ratios_mask(table["ratio"])].sort_values("ratio")
        colour, ls, marker = _STYLE[s]
        ax.plot(
            sub["ratio"],
            sub[ycol].astype(float),
            ls=ls,
            marker=marker,
            color=colour,
            ms=4,
            lw=1.3,
            label=SCHEMES[s][0],
        )


def _style_axis(ax: Axes, ticks: list[float], xlim: tuple[float, float]) -> None:
    import matplotlib.pyplot as plt

    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("post-shift fail rate ÷ Phase I rate")
    ax.axvline(1.0, color=MUTED, lw=0.8)


def _mark_observed(ax: Axes, observed: dict[str, float], *, upright: bool = False) -> None:
    """Dotted lines at the observed ratios; labels rotated, or upright in a top band."""
    for key, lab, ha in (("phase_ii", "W40–42", "left"), ("pre_baseline", "W29–34", "right")):
        r = observed[key]
        ax.axvline(r, color=MUTED, lw=0.8, ls=":")
        if upright:
            ax.text(
                r * (1.02 if ha == "left" else 1 / 1.02),
                0.985,
                f"{lab}\nobserved",
                transform=ax.get_xaxis_transform(),
                fontsize=6.2,
                color=MUTED,
                ha=ha,
                va="top",
            )
        else:
            ax.text(
                r * 1.02,
                0.99,
                f"{lab} observed",
                transform=ax.get_xaxis_transform(),
                fontsize=6.2,
                color=MUTED,
                ha="left",
                va="top",
                rotation=90,
            )


def _shared_legend(fig: Figure, axes: tuple[Axes, ...], bottom: float) -> None:
    seen: dict[str, Any] = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            seen.setdefault(label, handle)
    fig.legend(
        list(seen.values()),
        list(seen.keys()),
        loc="lower center",
        bbox_to_anchor=(0.5, bottom),
        ncol=4,
        fontsize=6.6,
        frameon=False,
    )


def _figures(
    ctx: PipelineContext,
    table: pd.DataFrame,
    observed: dict[str, float],
    horizons: tuple[int, ...],
    grid: tuple[float, ...],
) -> None:
    import matplotlib.pyplot as plt

    horizon = horizons[-1]
    r_max, r_min = float(table["ratio"].max()), float(table["ratio"].min())
    deg_ticks = [r for r in grid if r >= 1]
    imp_ticks = [r for r in grid if r <= 1]
    deg_lim, imp_lim = (0.93, r_max * 1.08), (r_min * 0.92, 1.07)
    never = "p-chart and Laney p′ cannot signal:\nLCL = 0 in every observed week"

    # Fig 10: how long until each scheme signals
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.9), sharey=True)
    _panel_lines(a1, table, _DEGRADATION, lambda r: r >= 1, "median_run_length")
    _panel_lines(a2, table, _IMPROVEMENT, lambda r: r <= 1, "median_run_length")
    _style_axis(a1, deg_ticks, deg_lim)
    _style_axis(a2, imp_ticks, imp_lim)
    _mark_observed(a1, observed)
    a2.text(0.03, 0.97, never, transform=a2.transAxes, fontsize=6.8, color=ACCENT, va="top")
    for ax, title in ((a1, "Degradation (fail rate rises)"), (a2, "Improvement (fail rate falls)")):
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(lw=0.6, which="both")
    a1.set_ylim(10, 2.5 * horizon)
    a1.set_ylabel("median runs to first signal")
    _shared_legend(fig, (a1, a2), bottom=0.07)
    fig.text(
        0.5,
        0.015,
        f"Each point: {table['n_sim'].iloc[0]:,} simulated streams. Missing points: fewer "
        f"than half of streams signal within {horizon:,} runs (the campaign length).",
        ha="center",
        fontsize=6.6,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    ctx.save_figure(fig, "fig10_detection_delay")

    # Fig 11: probability of catching the shift within about one Phase II window
    h = 400 if 400 in horizons else horizons[len(horizons) // 2]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.7), sharey=True)
    _panel_lines(a1, table, _DEGRADATION, lambda r: r >= 1, f"p_detect_{h}")
    _panel_lines(a2, table, _IMPROVEMENT, lambda r: r <= 1, f"p_detect_{h}")
    _style_axis(a1, deg_ticks, deg_lim)
    _style_axis(a2, imp_ticks, imp_lim)
    _mark_observed(a1, observed, upright=True)
    a2.text(0.05, 0.58, never, transform=a2.transAxes, fontsize=6.8, color=ACCENT, va="top")
    for ax, title in ((a1, "Degradation"), (a2, "Improvement")):
        ax.set_title(title)
        ax.grid(lw=0.6)
    a1.set_ylim(-0.03, 1.25)
    a1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    a1.set_ylabel(f"P(signal in the shift direction\nwithin {h} runs)")
    _shared_legend(fig, (a1, a2), bottom=0.07)
    fig.text(
        0.5,
        0.015,
        "At ratio 1 (in control) any signal counts, so the value there is the false-alarm "
        f"probability within {h} runs.",
        ha="center",
        fontsize=6.6,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    ctx.save_figure(fig, "fig11_detection_probability")
