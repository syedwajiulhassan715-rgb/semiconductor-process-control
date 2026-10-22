"""Ground-truth validation of the monitoring code on the Tennessee Eastman benchmark.

Run with ``secom validate tep``. It is deliberately not a stage of the SECOM
pipeline: the SECOM analysis must never need this second dataset.

What is checked, all with this repo's own library code:

* :mod:`secom.mspc` PCA T² and SPE, trained on ``d00.dat`` (normal
  operation), with the textbook F / Jackson-Mudholkar limits and with the
  cross-validated empirical limits the SECOM multivariate stage uses.
* A univariate scheme: one Shewhart chart per variable, limits widened by a
  Šidák correction so the family of 52 charts false-alarms at ``alpha`` per
  sample. Sigma is the overall training SD. The moving-range sigma of
  :func:`secom.spc.individuals_limits` (used for the SECOM sensor charts) is
  also fitted, but only its calibration is reported: TEP's variables are
  strongly autocorrelated, the moving range then underestimates sigma, and
  those limits alarm on almost every normal sample (see D39).

Per fault: detection rate after the onset, false-alarm rate before it, and the
detection delay. The calibration of the textbook limits is then compared
between TEP and SECOM, which is the question the SECOM multivariate section
left open: are its badly calibrated textbook limits a bug or a property of
the data?
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sps

from . import tep
from .mspc import (
    PCAModel,
    cross_validated_scores,
    empirical_limit,
    fit_pca_model,
    n_components_for_variance,
    score,
)
from .spc import individuals_limits, moving_range_sigma

CHARTS = ("t2", "spe", "univariate")


@dataclass(frozen=True)
class TEPValidationConfig:
    #: per-chart false-alarm probability, matching the published PCA benchmark
    #: and the SECOM multivariate stage
    alpha: float = 0.01
    #: retained components. 11 is the setting commonly attributed to the
    #: Chiang et al. (2001) PCA benchmark on these files (see D38)
    n_components: int = 11
    #: alternative component counts for the sensitivity check; the SECOM
    #: variance rule (80%) is added automatically
    sensitivity_components: tuple[int, ...] = (5, 9, 11, 15, 20)
    #: contiguous time blocks for the out-of-fold scores behind the empirical limits
    cv_folds: int = 5
    #: consecutive alarms required for the "persistent" detection delay
    delay_run_length: int = 6
    #: SECOM variance target, for the calibration comparison
    secom_variance_target: float = 0.80


# --------------------------------------------------------------------------
# Charts


@dataclass(frozen=True)
class Monitors:
    """The fitted monitors and their fixed limits."""

    model: PCAModel
    cv_t2_limit: float
    cv_spe_limit: float
    oof_t2: np.ndarray
    oof_spe: np.ndarray
    uni_center: np.ndarray
    uni_lcl: np.ndarray
    uni_ucl: np.ndarray
    uni_k: float
    #: the same family of charts with moving-range sigma, for calibration only
    mr_lcl: np.ndarray
    mr_ucl: np.ndarray
    mr_sigma_ratio: np.ndarray  # MR sigma / SD per variable


def fit_monitors(train: np.ndarray, cfg: TEPValidationConfig) -> Monitors:
    """Fit PCA (textbook and cross-validated limits) and the univariate charts on ``train``."""
    names = list(tep.VARIABLES)
    model = fit_pca_model(train, names, n_components=cfg.n_components, alpha=cfg.alpha)
    oof_t2, oof_spe = cross_validated_scores(
        train, names, n_components=cfg.n_components, n_folds=cfg.cv_folds
    )
    k = tep.sidak_k(cfg.alpha, train.shape[1])
    center = train.mean(axis=0)
    sd = train.std(axis=0, ddof=1)
    mr = [individuals_limits(train[:, j], k=k) for j in range(train.shape[1])]
    return Monitors(
        model=model,
        cv_t2_limit=empirical_limit(oof_t2, cfg.alpha),
        cv_spe_limit=empirical_limit(oof_spe, cfg.alpha),
        oof_t2=oof_t2,
        oof_spe=oof_spe,
        uni_center=center,
        uni_lcl=center - k * sd,
        uni_ucl=center + k * sd,
        uni_k=k,
        mr_lcl=np.array([lim.lcl for lim in mr]),
        mr_ucl=np.array([lim.ucl for lim in mr]),
        mr_sigma_ratio=np.array([lim.sigma for lim in mr]) / sd,
    )


def alarms(mon: Monitors, x: np.ndarray, *, limits: str = "textbook") -> dict[str, np.ndarray]:
    """Per-sample alarms for each chart on ``x`` (new observations).

    ``limits`` is ``"textbook"`` (F for T², Jackson-Mudholkar for SPE) or
    ``"cv"`` (cross-validated empirical quantiles). The univariate chart is
    the same under both.
    """
    sc = score(mon.model, x)
    if limits == "textbook":
        t2_lim, spe_lim = mon.model.t2_limit, mon.model.spe_limit
    elif limits == "cv":
        t2_lim, spe_lim = mon.cv_t2_limit, mon.cv_spe_limit
    else:
        raise ValueError(f"unknown limits {limits!r}")
    uni = np.asarray(((x < mon.uni_lcl) | (x > mon.uni_ucl)).any(axis=1))
    return {"t2": sc.t2 > t2_lim, "spe": sc.spe > spe_lim, "univariate": uni}


def lag1_autocorrelation(x: np.ndarray) -> np.ndarray:
    """Lag-1 autocorrelation of each column."""
    z = np.asarray(x, dtype=float)
    z = z - z.mean(axis=0)
    return np.asarray((z[:-1] * z[1:]).sum(axis=0) / (z**2).sum(axis=0))


# --------------------------------------------------------------------------
# Metrics


def _fault_metrics(a: np.ndarray, run_length: int) -> dict[str, Any]:
    first = tep.detection_delay(a, run_length=1)
    persistent = tep.detection_delay(a, run_length=run_length)
    return {
        "detection_rate": tep.detection_rate(a),
        "false_alarm_rate_pre_fault": tep.false_alarm_rate(a),
        "delay_first_samples": first,
        "delay_first_minutes": None if first is None else first * tep.SAMPLE_MINUTES,
        "delay_persistent_samples": persistent,
        "delay_persistent_minutes": None if persistent is None else persistent * tep.SAMPLE_MINUTES,
    }


def _rates(a: dict[str, np.ndarray], rows: slice | np.ndarray = slice(None)) -> dict[str, float]:
    out = {c: float(v[rows].mean()) for c, v in a.items()}
    if "t2" in a and "spe" in a:
        out["either_pca"] = float((a["t2"][rows] | a["spe"][rows]).mean())
    return out


def _benchmark_comparison(faults: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for i, chart in enumerate(("t2", "spe")):
        ours = np.array([f["textbook"][chart]["detection_rate"] for f in faults])
        ref = np.array([1 - tep.BENCHMARK_PCA_MISSED[f["fault"]][i] for f in faults])
        diff = ours - ref
        out[chart] = {
            "mean_abs_diff": float(np.mean(np.abs(diff))),
            "max_abs_diff": float(np.max(np.abs(diff))),
            "max_abs_diff_fault": int(faults[int(np.argmax(np.abs(diff)))]["fault"]),
            "n_within_0_05": int(np.sum(np.abs(diff) <= 0.05)),
            "mean_diff": float(np.mean(diff)),
            "pearson_r": float(np.corrcoef(ours, ref)[0, 1]),
        }
    return out


# --------------------------------------------------------------------------
# Calibration: textbook limits on TEP versus SECOM


def _calibration(model: PCAModel, x_train: np.ndarray, oof: tuple[np.ndarray, np.ndarray]) -> dict:
    """Textbook-limit alarm rates in-sample (Beta T²) and out of fold (F T²)."""
    ins = score(model, x_train)
    t2_in = ins.t2 > model.t2_limit_phase1
    spe_in = ins.spe > model.spe_limit
    t2_oof = oof[0] > model.t2_limit
    spe_oof = oof[1] > model.spe_limit
    return {
        "n_components": model.n_components,
        "explained": model.explained,
        "in_sample": {
            "t2": float(t2_in.mean()),
            "spe": float(spe_in.mean()),
            "either": float((t2_in | spe_in).mean()),
        },
        "out_of_fold": {
            "t2": float(t2_oof.mean()),
            "spe": float(spe_oof.mean()),
            "either": float((t2_oof | spe_oof).mean()),
        },
    }


def secom_calibration(root: Path, cfg: TEPValidationConfig) -> dict[str, Any]:
    """The same calibration check on the SECOM Phase I baseline.

    Runs the SECOM pipeline up to ``clean`` in memory (nothing is written),
    then fits the PCA model exactly as the multivariate stage does.
    """
    from .config import load_config
    from .pipeline import Paths, PipelineContext, run_pipeline

    cfg_path = root / "config" / "default.yaml"
    ctx = PipelineContext(
        config=load_config(cfg_path if cfg_path.exists() else None),
        paths=Paths.from_root(root),
        log=lambda _msg: None,
    )
    run_pipeline(ctx, only=["clean"], skip_figures=True)
    clean, retained, base = ctx["clean"], ctx["retained"], ctx["masks"]["base"]
    x = clean[retained].to_numpy(dtype=float)[base]
    model = fit_pca_model(x, retained, variance_target=cfg.secom_variance_target, alpha=cfg.alpha)
    x = x[:, [retained.index(s) for s in model.sensors]]
    oof = cross_validated_scores(
        x, model.sensors, n_components=model.n_components, n_folds=cfg.cv_folds
    )
    out = _calibration(model, x, oof)
    out["n_baseline"] = int(x.shape[0])
    out["n_sensors"] = len(model.sensors)
    out["median_excess_kurtosis"] = float(
        np.median(_excess_kurtosis((x - model.mean) / model.scale))
    )
    # The moving-range sigma behind the SECOM per-sensor Phase I limits is
    # only unbiased for independent runs; record how far that holds here.
    sd = x.std(axis=0, ddof=1)
    mr = np.array([moving_range_sigma(x[:, j]) for j in range(x.shape[1])])
    out["baseline_lag1_autocorrelation_median"] = float(np.median(lag1_autocorrelation(x)))
    out["baseline_mr_over_sd_sigma_median"] = float(np.median(mr / sd))
    return out


def _excess_kurtosis(z: np.ndarray) -> np.ndarray:
    return np.asarray(sps.kurtosis(z, axis=0))


# --------------------------------------------------------------------------
# Driver


def run_tep_validation(
    root: str | Path,
    cfg: TEPValidationConfig | None = None,
    *,
    with_secom: bool = True,
    log: Callable[[str], None] = print,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute every TEP validation result. Returns ``(results, figure_data)``."""
    cfg = cfg or TEPValidationConfig()
    root = Path(root).resolve()
    tep_dir = root / "data" / "raw" / "tep"
    data = tep.load_tep(tep_dir)
    mon = fit_monitors(data.train, cfg)
    model = mon.model
    log(
        f"PCA: A={model.n_components} ({model.explained:.1%}), T2 limit {model.t2_limit:.2f}, "
        f"SPE limit {model.spe_limit:.2f} ({model.spe_limit_method}); "
        f"univariate k={mon.uni_k:.3f}"
    )

    normal = {lim: alarms(mon, data.test[0], limits=lim) for lim in ("textbook", "cv")}
    faults: list[dict[str, Any]] = []
    traces: dict[int, dict[str, np.ndarray]] = {}
    pre_pooled: dict[str, list[np.ndarray]] = {lim: [] for lim in ("textbook", "cv")}
    for k in tep.FAULTS:
        x = data.test[k]
        row: dict[str, Any] = {"fault": k, "description": tep.FAULT_DESCRIPTIONS[k]}
        for lim in ("textbook", "cv"):
            a = alarms(mon, x, limits=lim)
            row[lim] = {c: _fault_metrics(a[c], cfg.delay_run_length) for c in CHARTS}
            pre_pooled[lim].append(np.column_stack([a[c][: tep.FAULT_ONSET] for c in CHARTS]))
        miss_t2, miss_spe = tep.BENCHMARK_PCA_MISSED[k]
        row["benchmark_pca"] = {
            "t2_detection_rate": 1 - miss_t2,
            "spe_detection_rate": 1 - miss_spe,
        }
        faults.append(row)
        sc = score(model, x)
        traces[k] = {"t2": sc.t2, "spe": sc.spe}

    pre = {
        lim: {c: float(np.vstack(v)[:, i].mean()) for i, c in enumerate(CHARTS)}
        for lim, v in pre_pooled.items()
    }

    names = list(tep.VARIABLES)
    a80 = n_components_for_variance(model.eigenvalues, cfg.secom_variance_target)
    sensitivity = []
    for a_ in sorted({*cfg.sensitivity_components, a80}):
        m = fit_pca_model(data.train, names, n_components=a_, alpha=cfg.alpha)
        s0 = score(m, data.test[0])
        det_t2, det_spe = [], []
        for k in tep.FAULTS:
            if k in tep.UNDETECTABLE_BY_PCA:
                continue
            s = score(m, data.test[k])
            det_t2.append(tep.detection_rate(s.t2 > m.t2_limit))
            det_spe.append(tep.detection_rate(s.spe > m.spe_limit))
        sensitivity.append(
            {
                "n_components": a_,
                "explained": m.explained,
                "secom_variance_rule": a_ == a80,
                "d00_te_far_t2": float((s0.t2 > m.t2_limit).mean()),
                "d00_te_far_spe": float((s0.spe > m.spe_limit).mean()),
                "mean_detection_t2_detectable_faults": float(np.mean(det_t2)),
                "mean_detection_spe_detectable_faults": float(np.mean(det_spe)),
            }
        )

    x0 = data.test[0]
    mr_alarm = (x0 < mon.mr_lcl) | (x0 > mon.mr_ucl)
    sd_alarm = (x0 < mon.uni_lcl) | (x0 > mon.uni_ucl)
    rho1 = lag1_autocorrelation(data.train)
    univariate = {
        "per_chart_alpha": 1 - (1 - cfg.alpha) ** (1 / tep.N_VARS),
        "k": mon.uni_k,
        "d00_te_family_far_sd_sigma": float(sd_alarm.any(axis=1).mean()),
        "d00_te_family_far_mr_sigma": float(mr_alarm.any(axis=1).mean()),
        "d00_te_per_chart_far_sd_sigma_mean": float(sd_alarm.mean()),
        "d00_te_per_chart_far_mr_sigma_mean": float(mr_alarm.mean()),
        "mr_over_sd_sigma_median": float(np.median(mon.mr_sigma_ratio)),
        "mr_over_sd_sigma_min": float(np.min(mon.mr_sigma_ratio)),
        "lag1_autocorrelation_median": float(np.median(rho1)),
        "n_variables_lag1_above_0_5": int(np.sum(rho1 > 0.5)),
        "spearman_mr_ratio_vs_lag1": float(sps.spearmanr(mon.mr_sigma_ratio, rho1).statistic),
    }

    tep_cal = _calibration(model, data.train, (mon.oof_t2, mon.oof_spe))
    tep_cal["test_normal_d00_te"] = _rates(normal["textbook"])
    tep_cal["n_train"] = int(data.train.shape[0])
    tep_cal["median_excess_kurtosis"] = float(
        np.median(_excess_kurtosis((data.train - model.mean) / model.scale))
    )
    calibration: dict[str, Any] = {
        "nominal_per_chart": cfg.alpha,
        "tep": tep_cal,
        "tep_univariate": univariate,
    }
    if with_secom:
        log("SECOM: refitting the Phase I PCA model for the calibration comparison")
        calibration["secom"] = secom_calibration(root, cfg)

    sums = tep_dir / "SHA256SUMS"
    sha256: dict[str, str] = {}
    if sums.exists():
        for line in sums.read_text(encoding="utf-8").splitlines():
            if line.strip():
                digest, name = line.split("  ", 1)
                sha256[name] = digest
    results: dict[str, Any] = {
        "dataset": {
            "source": "Braatz TEP training/testing files (Chiang, Russell & Braatz, 2001)",
            "mirror": "https://github.com/camaramm/tennessee-eastman-profBraatz",
            "n_train": int(data.train.shape[0]),
            "n_test": tep.N_TEST,
            "n_variables": tep.N_VARS,
            "fault_onset_sample": tep.FAULT_ONSET,
            "sample_minutes": tep.SAMPLE_MINUTES,
            "sha256": sha256,
        },
        "settings": asdict(cfg),
        "model": {
            "n_components": model.n_components,
            "explained": model.explained,
            "kaiser_components": int(np.sum(model.eigenvalues > 1)),
            "variance_rule_components": a80,
            "t2_limit_F": model.t2_limit,
            "t2_limit_phase1_beta": model.t2_limit_phase1,
            "spe_limit": model.spe_limit,
            "spe_limit_method": model.spe_limit_method,
            "cv_t2_limit": mon.cv_t2_limit,
            "cv_spe_limit": mon.cv_spe_limit,
        },
        "false_alarms": {
            "d00_te": {lim: _rates(a) for lim, a in normal.items()},
            "pre_fault_pooled": pre,
        },
        "faults": faults,
        "benchmark_comparison": _benchmark_comparison(faults),
        "undetectable_by_pca": {
            str(k): {c: faults[k - 1]["textbook"][c]["detection_rate"] for c in ("t2", "spe")}
            for k in tep.UNDETECTABLE_BY_PCA
        },
        "calibration": calibration,
        "sensitivity": sensitivity,
    }
    figure_data = {"traces": traces, "model": model, "cv": (mon.cv_t2_limit, mon.cv_spe_limit)}
    return results, figure_data


def write_results(results: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Figures


def write_figures(
    results: dict[str, Any],
    figure_data: dict[str, Any],
    figures_dir: str | Path,
    *,
    fmt: str = "png",
    dpi: int = 130,
) -> list[Path]:
    """fig13 detection per fault, fig14 calibration TEP vs SECOM, fig15 example traces."""
    import matplotlib.pyplot as plt
    from matplotlib import ticker

    from .pipeline.plotting import ACCENT, BLUE, GOOD, INK, LIGHT, MUTED, SAND, configure

    configure(fmt, dpi)
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def save(fig: Any, name: str) -> None:
        path = out_dir / f"{name}.{fmt}"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        written.append(path)

    faults = results["faults"]
    ks = np.array([f["fault"] for f in faults])

    # fig13: detection rate per fault
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    w = 0.27
    series = (
        ("t2", "PCA T²", INK),
        ("spe", "PCA SPE", ACCENT),
        ("univariate", "univariate (Šidák)", SAND),
    )
    for i, (c, lab, col) in enumerate(series):
        vals = [f["textbook"][c]["detection_rate"] for f in faults]
        ax.bar(ks + (i - 1) * w, vals, width=w, color=col, label=lab)
    ref_t2 = [f["benchmark_pca"]["t2_detection_rate"] for f in faults]
    ref_spe = [f["benchmark_pca"]["spe_detection_rate"] for f in faults]
    ax.plot(ks - w, ref_t2, "_", color=BLUE, ms=11, mew=2, label="benchmark T² (Chiang et al.)")
    ax.plot(ks, ref_spe, "_", color=GOOD, ms=11, mew=2, label="benchmark SPE")
    nominal = results["settings"]["alpha"]
    ax.axhline(nominal, color=MUTED, ls="--", lw=1)
    ax.text(21.6, nominal + 0.015, "α", color=MUTED, fontsize=8)
    for k in tep.UNDETECTABLE_BY_PCA:
        ax.axvspan(k - 0.5, k + 0.5, color=LIGHT, alpha=0.5, zorder=0)
    ax.set_xticks(ks)
    ax.set_xlim(0.4, 22)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("TEP fault")
    ax.set_ylabel("detection rate after onset")
    ax.set_title("Tennessee Eastman: this repo's charts against the published PCA benchmark")
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(axis="y", lw=0.6)
    fig.tight_layout()
    save(fig, "fig13_tep_detection")

    # fig14: calibration of the textbook limits, TEP vs SECOM
    cal = results["calibration"]
    groups: list[tuple[str, dict[str, float]]] = [
        ("TEP\nin-sample", cal["tep"]["in_sample"]),
        ("TEP\nout-of-fold", cal["tep"]["out_of_fold"]),
        ("TEP\nnew normal run", cal["tep"]["test_normal_d00_te"]),
    ]
    if "secom" in cal:
        groups += [
            ("SECOM\nin-sample", cal["secom"]["in_sample"]),
            ("SECOM\nout-of-fold", cal["secom"]["out_of_fold"]),
        ]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    x = np.arange(len(groups))
    for i, (c, lab, col) in enumerate((("t2", "T²", INK), ("spe", "SPE", ACCENT))):
        vals = [g[1][c] for g in groups]
        pos = x + (i - 0.5) * 0.36
        ax.bar(pos, vals, width=0.36, color=col, label=lab)
        for xp, v in zip(pos, vals, strict=True):
            ax.text(xp, v + 0.01, f"{v:.1%}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_axisbelow(True)
    ax.axhline(nominal, color=GOOD, ls="--", lw=1.2, label=f"nominal α = {nominal:g}")
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=7.5)
    ax.set_ylabel("false-alarm rate, textbook limits")
    ax.set_title("Textbook limits: hold in-sample on TEP, fail even in-sample on SECOM")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(axis="y", lw=0.6)
    fig.tight_layout()
    save(fig, "fig14_tep_calibration")

    # fig15: a detectable and an undetectable fault
    model: PCAModel = figure_data["model"]
    traces = figure_data["traces"]
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 4.6), sharex=True)
    t_hours = np.arange(tep.N_TEST) * tep.SAMPLE_MINUTES / 60
    onset_h = tep.FAULT_ONSET * tep.SAMPLE_MINUTES / 60
    titles = {1: "fault 1, step: detected", 9: "fault 9, random variation: not detected"}
    for ci, k in enumerate((1, 9)):
        for ri, (stat, lim, lab) in enumerate(
            (("t2", model.t2_limit, "T²"), ("spe", model.spe_limit, "SPE"))
        ):
            ax = axes[ri, ci]
            ax.plot(t_hours, traces[k][stat], color=INK, lw=0.7)
            ax.axhline(lim, color=ACCENT, ls="--", lw=1.1)
            ax.axvline(onset_h, color=GOOD, lw=1.1)
            if k == 1:  # spans two decades; fault 9 stays near the limit
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
                ax.yaxis.set_minor_formatter(ticker.NullFormatter())
            ax.grid(axis="y", lw=0.6)
            if ri == 0:
                ax.set_title(titles[k], fontsize=9)
            if ci == 0:
                ax.set_ylabel(f"{lab} (log scale)")
            else:
                ax.set_ylabel(lab)
            if ri == 1:
                ax.set_xlabel("hours (fault introduced at the green line)")
    fig.tight_layout()
    save(fig, "fig15_tep_traces")
    return written
