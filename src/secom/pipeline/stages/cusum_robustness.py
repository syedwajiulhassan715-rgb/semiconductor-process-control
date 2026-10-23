"""How far the CUSUM's design assumptions can be trusted on this line.

Two questions, both about the false-alarm cadence the report quotes:

1. **p0.** The deployed chart is designed on the Phase I rate (3.07%), but
   after the change point the process runs at 4.27% (§8). This re-designs the
   CUSUM on other p0 values with the same logic (p1 = shift_ratio * p0, h
   calibrated to the same ARL0) and reports, for each design, the alarms it
   raises on the real data and its in-control ARL at the baseline and the
   post-change rate, in runs and in days at the line's run rate.
2. **Overdispersion.** Every simulation so far treats runs as independent
   Bernoulli trials, but the weekly rates are overdispersed (sigma_z = 1.46).
   This adds a beta-binomial day or week random effect calibrated so that
   simulated weekly sigma_z matches the observed one, verifies the calibration
   on fresh streams, and re-measures ARL0, false-alarm probability and
   detection power under it, against the independent case on common random
   numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ...config import register_section
from ...power import (
    NO_SIGNAL,
    SCHEMES,
    Calendar,
    Dispersion,
    MonitorDesign,
    calibrate_dispersion,
    cusum_first_signal,
    n_blocks_needed,
    overdispersed_fails,
    simulate_detection,
    subgroup_sigma_z,
)
from ...spc import bernoulli_cusum, calibrate_cusum_h, p_chart
from ..context import PipelineContext
from ..plotting import ACCENT, BLUE, GOOD, INK, MUTED, SAND
from .shift_simulation import ShiftSimulationConfig

if TYPE_CHECKING:
    from matplotlib.axes import Axes

NAME = "cusum_robustness"
REQUIRES = (
    "clean",
    "is_fail",
    "masks",
    "weekly",
    "laney",
    "cusum_design",
    "cusum_lower_design",
    "monitor_design",
    "changepoint_fit",
)
PRODUCES = ("robustness_detection",)


@register_section("cusum_robustness")
@dataclass(frozen=True)
class CusumRobustnessConfig:
    #: Design rates p0 for the sensitivity sweep. The Phase I, post-change
    #: and Phase II rates are always added.
    p0_sweep: tuple[float, ...] = (0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06)
    #: Simulated in-control streams per ARL estimate.
    arl_n_sim: int = 1500
    #: Length of each ARL stream, in runs; longer run lengths are censored.
    arl_max_run: int = 8000
    #: Levels of the beta-binomial random effect to simulate (day and/or week).
    dispersion_levels: tuple[str, ...] = ("week", "day")
    #: Streams used to solve for the intraclass correlation rho.
    calibration_n_sim: int = 2000
    #: Fresh streams used to verify the calibrated sigma_z.
    verify_n_sim: int = 2000
    #: Shift ratios re-simulated under overdispersion (observed ratios are added).
    ratios: tuple[float, ...] = (0.5, 1.0, 2.0)


def _none(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float | np.floating):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    return v


def _calendar(clean: pd.DataFrame, weekly: pd.DataFrame) -> Calendar:
    """Runs per calendar day and each day's week, in campaign order."""
    day = clean["timestamp"].dt.floor("D")
    g = clean.groupby(day, sort=True).agg(n=("is_fail", "size"), week=("year_week", "first"))
    widx = {w: i for i, w in enumerate(weekly["year_week"])}
    return Calendar(tuple(int(v) for v in g["n"]), tuple(widx[w] for w in g["week"]))


def _run_lengths(fails: np.ndarray, *, p0: float, p1: float, h: float) -> tuple[float, float]:
    """(ARL, censored fraction) of a zero-state CUSUM over the given streams.

    Streams that never signal count at their full length, so with censoring
    the ARL is a lower bound, as in :func:`spc.calibrate_cusum_h`.
    """
    first = cusum_first_signal(fails, p0=p0, p1=p1, h=h)
    censored = first == NO_SIGNAL
    rl = np.where(censored, fails.shape[1], first)
    return float(rl.mean()), float(censored.mean())


def _count_signals(x: np.ndarray, *, p0: float, p1: float, h: float) -> int:
    return int(bernoulli_cusum(x, p0=p0, p1=p1, h=h).table["signal"].sum())


def run(ctx: PipelineContext) -> None:
    cfg = ctx.config
    rc = cfg.section(CusumRobustnessConfig)
    sc = cfg.section(ShiftSimulationConfig)
    clean, is_fail, masks = ctx["clean"], ctx["is_fail"], ctx["masks"]
    design: MonitorDesign = ctx["monitor_design"]
    upper = ctx["cusum_design"]
    fit = ctx["changepoint_fit"]

    span_days = float((clean["timestamp"].max() - clean["timestamp"].min()) / pd.Timedelta(days=1))
    runs_per_day = len(clean) / span_days
    horizon = int(max(sc.horizons))
    p_base = float(upper["p0"])
    p_post = float(fit.rate_after)
    p_phase2 = float(is_fail[masks["post"]].mean())
    true_rates = {"baseline": p_base, "post_change": p_post}

    # Common random numbers for every in-control ARL estimate.
    rng = np.random.default_rng(cfg.seed + 11)
    arl_u = rng.random((rc.arl_n_sim, rc.arl_max_run), dtype=np.float32)
    arl_off = rng.integers(0, len(design.subgroup_sizes), rc.arl_n_sim)

    p0_block = _p0_reestimate(
        ctx, rc, arl_u, true_rates, runs_per_day, p_base=p_base, p_post=p_post, p_phase2=p_phase2
    )
    red = p0_block["headline"]["redesigned"]
    od_block, det_table = _overdispersion(
        ctx,
        rc,
        sc,
        design,
        (red["p0"], red["p1"], red["h"]),
        arl_u,
        arl_off,
        true_rates,
        runs_per_day,
        horizon,
    )

    # When the retrospective lower CUSUM (designed on the degraded rate)
    # confirms the improvement, measured from the estimated change point.
    retro = ctx.results["cusum"]["lower_retrospective"]
    after = [r for r in retro["signal_runs"] if r >= fit.k + 1]
    ts = clean["timestamp"]
    retro_block: dict[str, Any] = {
        "change_point_run": int(fit.k + 1),
        "first_signal_at_or_after_change": after[0] if after else None,
        "runs_after_change": after[0] - (fit.k + 1) if after else None,
        "hours_after_change": float(
            (ts.iloc[after[0] - 1] - ts.iloc[fit.k]) / pd.Timedelta(hours=1)
        )
        if after
        else None,
        "signals_before_change": [r for r in retro["signal_runs"] if r < fit.k + 1],
    }

    ctx.results["cusum_robustness"] = {
        "runs_per_day": runs_per_day,
        "campaign_days": span_days,
        "p0_reestimate": p0_block,
        "overdispersion": od_block,
        "lower_retrospective": retro_block,
    }
    ctx.artifacts["robustness_detection"] = det_table

    d = p0_block["headline"]
    od = od_block["in_control_headline"]
    dep_post = d["deployed_at_post_change"]
    ctx.log(
        f"deployed CUSUM at the post-change rate: ARL {dep_post['arl_runs']:.0f} "
        f"runs ({dep_post['arl_days']:.1f} days); redesigned on "
        f"p0={p_post:.4f}: h={d['redesigned']['h']:.3f}, "
        f"{d['redesigned']['alarms']['w35_onward']} alarms from W35 vs "
        f"{d['deployed']['alarms']['w35_onward']}; overdispersed ARL0 "
        + ", ".join(f"{k} {v['arl_runs']:.0f}" for k, v in od.items())
    )
    if ctx.figures_enabled:
        _figure(ctx, p0_block, od_block)


def _p0_reestimate(
    ctx: PipelineContext,
    rc: CusumRobustnessConfig,
    arl_u: np.ndarray,
    true_rates: dict[str, float],
    runs_per_day: float,
    *,
    p_base: float,
    p_post: float,
    p_phase2: float,
) -> dict[str, Any]:
    cfg = ctx.config
    cc = cfg.cusum
    is_fail, masks = ctx["is_fail"], ctx["masks"]
    fit = ctx["changepoint_fit"]
    upper = ctx["cusum_design"]
    windows = {
        "baseline_self_check": masks["base"],
        "phase_ii": masks["post"],
        "w35_onward": masks["base"] | masks["post"],
        "post_change_segment": np.arange(len(is_fail)) >= fit.k,
    }
    named = {"baseline": p_base, "post_change": p_post, "phase_ii": p_phase2}
    p0s = sorted({*(float(v) for v in rc.p0_sweep), *named.values()})
    truth = {k: arl_u < np.float32(v) for k, v in true_rates.items()}

    designs = []
    for p0 in p0s:
        p1 = cc.shift_ratio * p0
        if np.isclose(p0, p_base):
            h, arl0 = float(upper["h"]), float(upper["arl0"])  # the deployed design
        else:
            h, arl0 = calibrate_cusum_h(
                p0,
                p1,
                target_arl0=cc.target_arl0,
                n_sim=cc.n_sim,
                max_run=cc.max_run,
                seed=cfg.seed,
            )
        row: dict[str, Any] = {
            "label": next((k for k, v in named.items() if np.isclose(v, p0)), None),
            "p0": p0,
            "p1": p1,
            "h": h,
            "arl0_design_runs": arl0,
        }
        for tname, fails in truth.items():
            arl, cens = _run_lengths(fails, p0=p0, p1=p1, h=h)
            row[f"arl_at_{tname}_rate"] = {
                "true_rate": true_rates[tname],
                "arl_runs": arl,
                "arl_days": arl / runs_per_day,
                "censored_frac": cens,
            }
        row["alarms"] = {
            k: _count_signals(is_fail[m], p0=p0, p1=p1, h=h) for k, m in windows.items()
        }
        pre = bernoulli_cusum(is_fail[masks["pre"]], p0=p0, p1=p1, h=h).first_signal_index
        row["pre_baseline_runs_to_signal"] = None if pre is None else int(pre + 1)
        n_w35 = int(windows["w35_onward"].sum())
        row["expected_alarms_w35_onward_if_in_control_at_post_change_rate"] = (
            n_w35 / row["arl_at_post_change_rate"]["arl_runs"]
        )
        designs.append(row)

    def by_label(label: str) -> dict[str, Any]:
        return next(r for r in designs if r["label"] == label)

    dep, red = by_label("baseline"), by_label("post_change")
    return {
        "design_logic": (
            f"p1 = {cc.shift_ratio:g} * p0; h calibrated by simulation to ARL0 = "
            f"{cc.target_arl0:g} runs at p0 ({cc.n_sim} paths x {cc.max_run} runs, seed "
            f"{cfg.seed}); alarms counted with reset on signal, each window charted "
            "from a fresh start"
        ),
        "rates": {
            "baseline": p_base,
            "post_change": p_post,
            "phase_ii": p_phase2,
            "post_change_source": "changepoint segment after the estimated change",
        },
        "windows": {k: int(m.sum()) for k, m in windows.items()},
        "arl_simulation": {
            "n_sim": rc.arl_n_sim,
            "max_run": rc.arl_max_run,
            "seed": cfg.seed + 11,
            "note": "independent Bernoulli streams; censored run lengths count as max_run",
        },
        "designs": designs,
        "headline": {
            "deployed": dep,
            "redesigned": red,
            "deployed_at_post_change": dep["arl_at_post_change_rate"],
            "redesigned_at_post_change": red["arl_at_post_change_rate"],
        },
    }


def _overdispersion(
    ctx: PipelineContext,
    rc: CusumRobustnessConfig,
    sc: ShiftSimulationConfig,
    design: MonitorDesign,
    post_design: tuple[float, float, float],
    arl_u: np.ndarray,
    arl_off: np.ndarray,
    true_rates: dict[str, float],
    runs_per_day: float,
    horizon: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    cfg = ctx.config
    clean, weekly, laney = ctx["clean"], ctx["weekly"], ctx["laney"]
    masks = ctx["masks"]
    cal = _calendar(clean, weekly)
    if cal.week_sizes != tuple(design.subgroup_sizes):
        raise RuntimeError("calendar weeks do not match the weekly subgroups")
    target = float(laney.sigma_z)
    p0 = design.p0

    # Observed dispersion, for the record and to tell the two levels apart.
    w35 = clean[masks["base"] | masks["post"]]
    wk35 = w35.groupby("year_week")["is_fail"].agg(n="size", d="sum")
    days = clean.groupby(clean["timestamp"].dt.floor("D"))["is_fail"].agg(n="size", d="sum")
    observed = {
        "weekly_sigma_z_all_weeks": target,
        "weekly_sigma_z_w35_onward": float(
            p_chart(wk35["d"].to_numpy(), wk35["n"].to_numpy()).sigma_z
        ),
        "daily_sigma_z_all_days": float(
            p_chart(days["d"].to_numpy(), days["n"].to_numpy()).sigma_z
        ),
        "n_days": len(cal.day_sizes),
        "n_weeks": cal.n_weeks,
    }

    # Verification streams: one campaign each in the observed order.
    vr = np.random.default_rng(cfg.seed + 21)
    v_u = vr.random((rc.verify_n_sim, horizon))
    v_off = np.zeros(rc.verify_n_sim, dtype=np.int64)
    v_blocks = {
        lv: vr.random((rc.verify_n_sim, n_blocks_needed(cal, horizon, lv)))
        for lv in ("day", "week")
    }
    wsizes = np.asarray(cal.week_sizes)
    dsizes = np.asarray(cal.day_sizes)

    def verify(disp: Dispersion | None) -> dict[str, float]:
        if disp is None:
            fails = v_u < p0
        else:
            fails = overdispersed_fails(v_u, v_blocks[disp.level], v_off, p0, disp)
        sz = subgroup_sigma_z(fails, wsizes, v_off)
        sz = sz[np.isfinite(sz)]
        dz = subgroup_sigma_z(fails, dsizes, v_off)
        dz = dz[np.isfinite(dz)]
        return {
            "n_sim": int(len(sz)),
            "mean_sigma_z": float(sz.mean()),
            "mc_se": float(sz.std(ddof=1) / np.sqrt(len(sz))),
            "q05": float(np.quantile(sz, 0.05)),
            "median": float(np.median(sz)),
            "q95": float(np.quantile(sz, 0.95)),
            "p_at_least_observed": float(np.mean(sz >= target)),
            "mean_daily_sigma_z": float(dz.mean()),
            "p_daily_at_least_observed": float(np.mean(dz >= observed["daily_sigma_z_all_days"])),
        }

    levels: dict[str, Any] = {}
    dispersions: dict[str, Dispersion | None] = {"independent": None}
    for i, lv in enumerate(rc.dispersion_levels):
        rho, achieved = calibrate_dispersion(
            target,
            p=p0,
            calendar=cal,
            level=lv,
            n_sim=rc.calibration_n_sim,
            seed=cfg.seed + 31 + i,
        )
        disp = Dispersion(rho, lv, cal)
        dispersions[lv] = disp
        sd = float(np.sqrt(p0 * (1 - p0) * rho))
        levels[lv] = {
            "rho": rho,
            "calibration_mean_sigma_z": achieved,
            "block_rate_sd_at_p0": sd,
            "block_rate_cv_at_p0": sd / p0,
            "verification": verify(disp),
        }
    independent_check = verify(None)

    # In-control run lengths (ARL0) under each condition.
    arl_blocks = {
        lv: np.random.default_rng(cfg.seed + 41).random(
            (rc.arl_n_sim, n_blocks_needed(cal, rc.arl_max_run, lv))
        )
        for lv in rc.dispersion_levels
    }
    cusums: dict[str, tuple[float, float, float]] = {
        "cusum_upper": (p0, design.cusum_upper_p1, design.cusum_upper_h),
        "cusum_lower": (p0, design.cusum_lower_p1, design.cusum_lower_h),
    }
    in_control: list[dict[str, Any]] = []
    stream_cache: dict[tuple[str, str], np.ndarray] = {}

    def streams(cond: str, rate_name: str) -> np.ndarray:
        key = (cond, rate_name)
        if key not in stream_cache:
            disp = dispersions[cond]
            rate = true_rates[rate_name]
            if disp is None:
                stream_cache[key] = arl_u < np.float32(rate)
            else:
                stream_cache[key] = overdispersed_fails(
                    arl_u, arl_blocks[cond], arl_off, rate, disp
                )
        return stream_cache[key]

    cases = [
        ("cusum_upper", "baseline"),
        ("cusum_lower", "baseline"),
        ("cusum_upper", "post_change"),
        ("cusum_upper_post_change_design", "post_change"),
    ]
    cusums["cusum_upper_post_change_design"] = post_design
    for cond in dispersions:
        for scheme, rate_name in cases:
            q0, q1, h = cusums[scheme]
            fails = streams(cond, rate_name)
            first = cusum_first_signal(fails, p0=q0, p1=q1, h=h)
            censored = first == NO_SIGNAL
            rl = np.where(censored, fails.shape[1], first)
            in_control.append(
                {
                    "scheme": scheme,
                    "design_p0": q0,
                    "h": h,
                    "true_rate_label": rate_name,
                    "true_rate": true_rates[rate_name],
                    "condition": cond,
                    "arl_runs": float(rl.mean()),
                    "arl_days": float(rl.mean() / runs_per_day),
                    "median_run_length": float(np.median(rl)),
                    "censored_frac": float(censored.mean()),
                    f"p_false_alarm_{horizon}": float(np.mean(~censored & (first <= horizon))),
                }
            )
    stream_cache.clear()

    # Detection power under each condition, on common random numbers.
    p_obs = {
        "pre_baseline": float(ctx["is_fail"][masks["pre"]].mean() / p0),
        "phase_ii": float(ctx["is_fail"][masks["post"]].mean() / p0),
    }
    ratios = sorted({*(float(r) for r in rc.ratios), *p_obs.values()})
    horizons = tuple(sorted(sc.horizons))
    tables = []
    for cond, maybe_disp in dispersions.items():
        t = simulate_detection(
            design,
            tuple(ratios),
            horizons=horizons,
            n_sim=sc.n_sim,
            seed=cfg.seed,
            dispersion=maybe_disp,
        )
        t.insert(0, "condition", cond)
        tables.append(t)
    det = pd.concat(tables, ignore_index=True)

    def det_value(scheme: str, ratio: float, cond: str, col: str) -> float | None:
        sel = det[
            (det["condition"] == cond) & (det["scheme"] == scheme) & np.isclose(det["ratio"], ratio)
        ]
        return _none(sel[col].iloc[0]) if len(sel) else None

    conds = list(dispersions)
    detection_headline = {
        f"false_alarm_{horizon}": {
            s: {c: det_value(s, 1.0, c, f"p_detect_{horizon}") for c in conds} for s in SCHEMES
        },
        "degradation_2x_p_detect_400": {
            s: {c: det_value(s, 2.0, c, "p_detect_400") for c in conds}
            for s in ("p_chart", "laney_p_prime", "cusum_upper", "cusum_upper_matched")
        },
        "degradation_2x_median_run_length": {
            s: {c: det_value(s, 2.0, c, "median_run_length") for c in conds}
            for s in ("p_chart", "laney_p_prime", "cusum_upper", "cusum_upper_matched")
        },
        f"improvement_half_p_detect_{horizon}": {
            s: {c: det_value(s, 0.5, c, f"p_detect_{horizon}") for c in conds}
            for s in ("g_chart", "cusum_lower", "cusum_lower_matched")
        },
        "improvement_half_median_run_length": {
            s: {c: det_value(s, 0.5, c, "median_run_length") for c in conds}
            for s in ("g_chart", "cusum_lower", "cusum_lower_matched")
        },
    }

    def ic_row(scheme: str, rate: str, cond: str) -> dict[str, Any]:
        return next(
            r
            for r in in_control
            if r["scheme"] == scheme and r["true_rate_label"] == rate and r["condition"] == cond
        )

    block = {
        "model": (
            "beta-binomial random effect: every calendar day (or ISO week) of the "
            "observed campaign layout draws its own fail probability from a Beta "
            "with mean p and intraclass correlation rho; runs within it are "
            "Bernoulli at that probability"
        ),
        "calibration": (
            "rho solved (Brent) so the mean Laney sigma_z of simulated in-control "
            "campaigns at p0, charted on the observed weekly subgroups and centred "
            "on their own rate, equals the observed all-weeks sigma_z; verified on "
            "fresh streams"
        ),
        "target_sigma_z": target,
        "calibration_rate": p0,
        "observed": observed,
        "independent_reference": independent_check,
        "levels": levels,
        "in_control": in_control,
        "in_control_headline": {
            cond: {
                "arl_runs": ic_row("cusum_upper", "baseline", cond)["arl_runs"],
                "arl_days": ic_row("cusum_upper", "baseline", cond)["arl_days"],
                f"p_false_alarm_{horizon}": ic_row("cusum_upper", "baseline", cond)[
                    f"p_false_alarm_{horizon}"
                ],
            }
            for cond in dispersions
        },
        "detection_headline": detection_headline,
        "detection_settings": {
            "n_sim": int(sc.n_sim),
            "seed": int(cfg.seed),
            "horizons": list(horizons),
            "ratios": ratios,
            "note": (
                "same seed, stream count and start weeks as shift_simulation, so the "
                "independent rows reproduce its table and every condition shares "
                "common random numbers"
            ),
        },
        "detection": [
            {k: _none(v) for k, v in rec.items()} for rec in det.to_dict(orient="records")
        ],
    }
    return block, det


def _figure(
    ctx: PipelineContext,
    p0_block: dict[str, Any],
    od_block: dict[str, Any],
) -> None:
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})
    _p0_panel(a1, p0_block)
    _od_panel(a2, od_block)
    fig.tight_layout()
    ctx.save_figure(fig, "fig16_cusum_robustness")


def _p0_panel(ax: Axes, block: dict[str, Any]) -> None:
    rows = block["designs"]
    x = np.array([r["p0"] for r in rows]) * 100
    for key, colour, lab in (
        ("arl_at_baseline_rate", GOOD, "process at the Phase I rate"),
        ("arl_at_post_change_rate", ACCENT, "process at the post-change rate"),
    ):
        y = [r[key]["arl_days"] for r in rows]
        ax.plot(x, y, "o-", color=colour, ms=3.5, lw=1.2, label=lab)
    rates = block["rates"]
    for k, lab in (("baseline", "Phase I p₀"), ("post_change", "post-change")):
        ax.axvline(rates[k] * 100, color=MUTED, lw=0.8, ls=":")
        ax.text(
            rates[k] * 100,
            0.98,
            f" {lab}\n {rates[k]:.2%}",
            transform=ax.get_xaxis_transform(),
            fontsize=6.4,
            color=MUTED,
            va="top",
        )
    ax.set_yscale("log")
    ax.set_xlabel("design p₀ of the upper CUSUM (%)")
    ax.set_ylabel("in-control ARL (days)")
    ax.set_title("False-alarm cadence depends on the design p₀")
    ax.legend(fontsize=6.8, frameon=False, loc="lower right")
    ax.grid(lw=0.6, which="both")


def _od_panel(ax: Axes, block: dict[str, Any]) -> None:
    conds = ["independent", *block["levels"].keys()]
    labels = {
        "cusum_upper": "upper, Phase I design",
        "cusum_lower": "lower, Phase I design",
    }
    colours = {"independent": INK, "week": SAND, "day": BLUE}
    width = 0.8 / len(conds)
    for j, cond in enumerate(conds):
        vals = []
        for scheme in labels:
            row = next(
                r
                for r in block["in_control"]
                if r["scheme"] == scheme
                and r["true_rate_label"] == "baseline"
                and r["condition"] == cond
            )
            vals.append(row["arl_days"])
        xs = np.arange(len(labels)) + (j - (len(conds) - 1) / 2) * width
        bars = ax.bar(xs, vals, width=width * 0.92, color=colours.get(cond, MUTED), label=cond)
        for b, v in zip(bars, vals, strict=True):
            ax.text(
                b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=6.4
            )
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(list(labels.values()), fontsize=7.5)
    ax.set_ylabel("in-control ARL (days)")
    ax.set_title(f"Overdispersion (σ_z = {block['target_sigma_z']:.2f}) shortens ARL₀")
    ax.legend(title="runs simulated as", fontsize=6.8, title_fontsize=6.8, frameon=False)
    ax.grid(axis="y", lw=0.6)
