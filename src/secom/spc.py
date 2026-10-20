"""SPC charts: individuals, EWMA, Nelson rules, p / Laney p', g-chart, Bernoulli CUSUM.

Written by hand mainly so that Phase I limits stay fixed during Phase II, sigma
comes from the moving range instead of the overall SD, and p-charts get checked
for overdispersion before picking the limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd
from scipy import stats as sps

#: d2 constants by subgroup size (E[range] / sigma). d2(2) = 2/sqrt(pi).
D2 = {
    2: 1.128379167,
    3: 1.692569319,
    4: 2.058750993,
    5: 2.325929219,
    6: 2.534413951,
    7: 2.704357983,
    8: 2.847200100,
    9: 2.970026451,
    10: 3.077505101,
}

#: D4 upper control-limit factor for the moving-range chart, subgroup size 2.
D4_MR = 3.266
D3_MR = 0.0


def c4(n: int) -> float:
    """c4 unbiasing constant, E[s] = c4(n) * sigma. Uses the gamma formula so any n works."""
    if n < 2:
        return np.nan
    from scipy.special import gammaln

    return float(np.sqrt(2.0 / (n - 1)) * np.exp(gammaln(n / 2.0) - gammaln((n - 1) / 2.0)))


# Variables charts


@dataclass(frozen=True)
class ControlLimits:
    """Fixed limits estimated in Phase I and applied unchanged in Phase II."""

    center: float
    sigma: float
    lcl: float
    ucl: float
    n_baseline: int
    n_excluded: int = 0
    source: str = ""

    def violations(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return (x < self.lcl) | (x > self.ucl)


def moving_range_sigma(x: np.ndarray) -> float:
    """Short-term sigma from the average moving range: MRbar / d2(2)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    return float(np.mean(np.abs(np.diff(x))) / D2[2])


def individuals_limits(
    x: np.ndarray,
    *,
    k: float = 3.0,
    trim_iterations: int = 3,
) -> ControlLimits:
    """Phase I individuals chart limits.

    Points outside the trial limits are dropped and the limits recalculated, up
    to trim_iterations times. n_excluded says how many got dropped.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 10:
        return ControlLimits(np.nan, np.nan, np.nan, np.nan, len(x), 0, "insufficient")

    keep = np.ones(len(x), dtype=bool)
    for _ in range(trim_iterations):
        sub = x[keep]
        if len(sub) < 10:
            break
        center = float(np.mean(sub))
        sigma = moving_range_sigma(sub)
        if not np.isfinite(sigma) or sigma <= 0:
            break
        out = np.abs(x - center) > k * sigma
        new_keep = ~out
        if new_keep.sum() < 10 or np.array_equal(new_keep, keep):
            keep = new_keep if new_keep.sum() >= 10 else keep
            break
        keep = new_keep

    sub = x[keep]
    center = float(np.mean(sub))
    sigma = moving_range_sigma(sub)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(sub, ddof=1))
    return ControlLimits(
        center=center,
        sigma=float(sigma),
        lcl=center - k * sigma,
        ucl=center + k * sigma,
        n_baseline=len(x),
        n_excluded=int((~keep).sum()),
        source="individuals/MRbar-d2",
    )


def moving_range_limits(x: np.ndarray) -> ControlLimits:
    """Limits for the companion moving-range chart."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    mr = np.abs(np.diff(x))
    mrbar = float(np.mean(mr)) if len(mr) else np.nan
    return ControlLimits(
        center=mrbar,
        sigma=mrbar / D2[2] if np.isfinite(mrbar) else np.nan,
        lcl=D3_MR * mrbar,
        ucl=D4_MR * mrbar,
        n_baseline=len(x),
        source="moving-range",
    )


def ewma(
    x: np.ndarray,
    limits: ControlLimits,
    *,
    lam: float = 0.2,
    L: float = 2.7,
) -> pd.DataFrame:
    """EWMA chart using fixed Phase I limits.

    lam=0.2, L=2.7 is the usual small-shift setup from Montgomery (ARL0 ~500).
    Limits use the exact time-varying variance, so the first points are tighter.
    """
    x = np.asarray(x, dtype=float)
    z = np.empty(len(x))
    prev = limits.center
    for i, xi in enumerate(x):
        if np.isnan(xi):
            z[i] = prev
        else:
            prev = lam * xi + (1 - lam) * prev
            z[i] = prev
    t = np.arange(1, len(x) + 1)
    width = L * limits.sigma * np.sqrt((lam / (2 - lam)) * (1 - (1 - lam) ** (2 * t)))
    return pd.DataFrame(
        {
            "value": x,
            "ewma": z,
            "center": limits.center,
            "lcl": limits.center - width,
            "ucl": limits.center + width,
            "violation": (z < limits.center - width) | (z > limits.center + width),
        }
    )


# Nelson rules

NELSON_DESCRIPTIONS = {
    1: "1 point beyond 3 sigma",
    2: "9 points in a row on the same side of centre",
    3: "6 points in a row steadily increasing or decreasing",
    4: "14 points in a row alternating up and down",
    5: "2 of 3 consecutive points beyond 2 sigma, same side",
    6: "4 of 5 consecutive points beyond 1 sigma, same side",
    7: "15 points in a row within 1 sigma (stratification)",
    8: "8 points in a row beyond 1 sigma, either side (mixture)",
}


def nelson_rules(x: np.ndarray, limits: ControlLimits) -> pd.DataFrame:
    """Check all 8 Nelson rules against fixed limits, one bool column per rule.

    On this data rules 7 and 8 are the ones that fire when a sensor's variance
    changes.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    c, s = limits.center, limits.sigma
    out = {r: np.zeros(n, dtype=bool) for r in range(1, 9)}
    if not np.isfinite(s) or s <= 0:
        return pd.DataFrame(out)

    z = (x - c) / s
    finite = np.isfinite(z)

    def _window_flag(rule: int, size: int, test) -> None:
        for i in range(size - 1, n):
            w = z[i - size + 1 : i + 1]
            if not finite[i - size + 1 : i + 1].all():
                continue
            if test(w):
                out[rule][i - size + 1 : i + 1] = True

    out[1] = finite & (np.abs(z) > 3)
    _window_flag(2, 9, lambda w: np.all(w > 0) or np.all(w < 0))
    _window_flag(3, 6, lambda w: np.all(np.diff(w) > 0) or np.all(np.diff(w) < 0))
    _window_flag(4, 14, lambda w: np.all(np.diff(np.sign(np.diff(w))) != 0))
    _window_flag(5, 3, lambda w: (w > 2).sum() >= 2 or (w < -2).sum() >= 2)
    _window_flag(6, 5, lambda w: (w > 1).sum() >= 4 or (w < -1).sum() >= 4)
    _window_flag(7, 15, lambda w: np.all(np.abs(w) < 1))
    _window_flag(8, 8, lambda w: np.all(np.abs(w) > 1))
    return pd.DataFrame(out)


# p-chart


@dataclass
class PChartResult:
    """Per-subgroup p-chart (or Laney p') with variable control limits."""

    table: pd.DataFrame
    p_bar: float
    sigma_z: float
    overdispersed: bool
    chart_type: str
    dispersion_p: float = np.nan
    notes: list[str] = field(default_factory=list)


def p_chart(
    defectives: np.ndarray,
    sizes: np.ndarray,
    *,
    labels: np.ndarray | None = None,
    k: float = 3.0,
    baseline: np.ndarray | None = None,
    dispersion_subset: np.ndarray | None = None,
    force_type: str | None = None,
) -> PChartResult:
    """p-chart with variable subgroup sizes, switches to Laney p' if overdispersed.

    sigma_z is Laney's statistic (average moving range of the z-scores / d2).
    It's about 1 for binomial data; above 1.2 the limits get widened by sigma_z.
    A chi-square dispersion test is reported as well.

    baseline: subgroups used for the centre line (default: all).
    dispersion_subset: subgroups used for sigma_z (default: all). Separate from
    baseline because Phase I here is only 5 weeks, which isn't enough moving
    ranges to estimate sigma_z from.
    """
    defectives = np.asarray(defectives, dtype=float)
    sizes = np.asarray(sizes, dtype=float)
    if defectives.shape != sizes.shape:
        raise ValueError("defectives and sizes must have the same shape")
    if np.any(sizes <= 0):
        raise ValueError("subgroup sizes must be positive")

    base = np.ones(len(sizes), dtype=bool) if baseline is None else np.asarray(baseline, bool)
    p_bar = float(defectives[base].sum() / sizes[base].sum())
    p = defectives / sizes

    sigma_i = np.sqrt(p_bar * (1 - p_bar) / sizes)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (p - p_bar) / sigma_i

    # Laney sigma_z
    disp = (
        np.ones(len(sizes), dtype=bool)
        if dispersion_subset is None
        else np.asarray(dispersion_subset, bool)
    )
    zb = z[disp]
    zb = zb[np.isfinite(zb)]
    sigma_z = float(np.mean(np.abs(np.diff(zb))) / D2[2]) if len(zb) > 1 else 1.0

    # chi-square dispersion test on the baseline subgroups
    obs = defectives[base]
    exp = sizes[base] * p_bar
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = float(np.nansum((obs - exp) ** 2 / (exp * (1 - p_bar))))
    dof = max(int(base.sum()) - 1, 1)
    dispersion_p = float(sps.chi2.sf(chi2, dof))

    overdispersed = sigma_z > 1.2
    chart_type = force_type or ("laney_p_prime" if overdispersed else "p")
    scale = sigma_z if chart_type == "laney_p_prime" else 1.0

    ucl = np.clip(p_bar + k * scale * sigma_i, 0.0, 1.0)
    lcl = np.clip(p_bar - k * scale * sigma_i, 0.0, 1.0)

    table = pd.DataFrame(
        {
            "subgroup": labels if labels is not None else np.arange(len(sizes)),
            "n": sizes.astype(int),
            "defectives": defectives.astype(int),
            "p": p,
            "center": p_bar,
            "lcl": lcl,
            "ucl": lcl * 0 + ucl,
            "z": z,
            "in_baseline": base,
        }
    )
    table["above_ucl"] = table["p"] > table["ucl"]
    table["below_lcl"] = table["p"] < table["lcl"]
    table["violation"] = table["above_ucl"] | table["below_lcl"]

    verdict = (
        "overdispersed, Laney p-prime limits applied"
        if overdispersed
        else "consistent with binomial variation"
    )
    notes = [
        f"sigma_z = {sigma_z:.3f} ({verdict})",
        f"chi-square dispersion test: chi2 = {chi2:.1f} on {dof} df, p = {dispersion_p:.3g}",
    ]
    return PChartResult(
        table=table,
        p_bar=p_bar,
        sigma_z=sigma_z,
        overdispersed=overdispersed,
        chart_type=chart_type,
        dispersion_p=dispersion_p,
        notes=notes,
    )


# g-chart


@dataclass
class GChartResult:
    """Runs-between-failures chart with exact geometric probability limits."""

    table: pd.DataFrame
    p_hat: float
    center: float
    lcl: float
    ucl: float
    alpha: float
    notes: list[str] = field(default_factory=list)


def g_chart(
    fail_positions: np.ndarray,
    *,
    n_total: int,
    alpha: float = 0.0027,
    baseline_p: float | None = None,
) -> GChartResult:
    """g-chart: number of passing runs between consecutive failures.

    At a 6.6% fail rate a p-chart needs 127+ runs per subgroup before the LCL is
    above zero, so it can't show an improvement. This one can.

    Limits are exact geometric quantiles since the distribution is very skewed.
    alpha=0.0027 matches a normal 3-sigma chart.

    Note the LCL is 0 whenever p > alpha/2 (~0.14%), which includes this dataset,
    so in practice this only detects improvement. Use bernoulli_cusum for
    degradation.

    Parameters
    ----------
    fail_positions:
        Zero-based run indices at which a failure occurred, ascending.
    n_total:
        Total number of runs, used to estimate the failure rate.
    baseline_p:
        Failure rate for the centre line. Supply the Phase I rate to keep this a
        fixed-limit Phase II chart; otherwise it is estimated from all data.
    """
    fail_positions = np.asarray(fail_positions, dtype=int)
    if fail_positions.ndim != 1 or np.any(np.diff(fail_positions) < 0):
        raise ValueError("fail_positions must be a 1-D ascending array")
    if len(fail_positions) < 2:
        raise ValueError("need at least two failures to form a gap")

    # passing runs between consecutive failures
    gaps = np.diff(fail_positions) - 1
    p = float(baseline_p if baseline_p is not None else len(fail_positions) / n_total)
    p = min(max(p, 1e-9), 1 - 1e-9)

    # loc=-1 shifts scipy's geom so the support starts at 0
    dist = sps.geom(p, loc=-1)
    lcl = float(dist.ppf(alpha / 2.0))
    ucl = float(dist.ppf(1.0 - alpha / 2.0))
    center = float((1.0 - p) / p)

    table = pd.DataFrame(
        {
            "failure_index": np.arange(2, len(fail_positions) + 1),
            "run_index": fail_positions[1:] + 1,
            "runs_since_previous_fail": gaps,
            "center": center,
            "lcl": lcl,
            "ucl": ucl,
        }
    )
    table["below_lcl"] = table["runs_since_previous_fail"] < lcl
    table["above_ucl"] = table["runs_since_previous_fail"] > ucl
    table["violation"] = table["below_lcl"] | table["above_ucl"]

    notes = [
        f"p = {p:.4f}; geometric centre = {center:.1f} passing runs between failures",
        f"exact probability limits at alpha = {alpha} (matched to 3-sigma false-alarm rate)",
        "above UCL = sustained improvement",
    ]
    if lcl <= 0:
        notes.append(
            f"LOWER LIMIT INACTIVE: at p = {p:.4f} the exact lower quantile is 0, so "
            "this chart cannot signal degradation. Pair it with bernoulli_cusum."
        )
    else:
        notes.append("below LCL = failures closer together than expected (degradation)")
    return GChartResult(
        table=table, p_hat=p, center=center, lcl=lcl, ucl=ucl, alpha=alpha, notes=notes
    )


# Bernoulli CUSUM


@dataclass
class BernoulliCusumResult:
    """Run-level CUSUM on a pass/fail stream."""

    table: pd.DataFrame
    p0: float
    p1: float
    h: float
    arl0: float
    first_signal_index: int | None
    notes: list[str] = field(default_factory=list)


def bernoulli_cusum(
    is_fail: np.ndarray,
    *,
    p0: float,
    p1: float,
    h: float,
    reset_on_signal: bool = True,
) -> BernoulliCusumResult:
    """One-sided upper Bernoulli CUSUM (Reynolds & Stoumbos).

    Per run it adds:

        log(p1/p0)            on a fail
        log((1-p1)/(1-p0))    on a pass (negative)

    floored at 0, and signals when the total goes above h. Get h from
    calibrate_cusum_h.

    reset_on_signal sets the total back to 0 after a signal. Without it the
    early excursion in this data pushes the statistic to ~28 and it takes ~900
    good runs to come back down, so the rest of the campaign looks like one
    long alarm.
    """
    x = np.asarray(is_fail, dtype=bool)
    if not (0 < p0 < 1 and 0 < p1 < 1):
        raise ValueError("p0 and p1 must lie strictly in (0, 1)")
    if p1 <= p0:
        raise ValueError("p1 must exceed p0 for an upper CUSUM")

    w_fail = np.log(p1 / p0)
    w_pass = np.log((1 - p1) / (1 - p0))
    increments = np.where(x, w_fail, w_pass)

    s = np.empty(len(x))
    signal = np.zeros(len(x), dtype=bool)
    acc = 0.0
    for i, inc in enumerate(increments):
        acc = max(0.0, acc + float(inc))
        s[i] = acc
        if acc > h:
            signal[i] = True
            if reset_on_signal:
                acc = 0.0
    first = int(np.argmax(signal)) if signal.any() else None

    table = pd.DataFrame(
        {
            "is_fail": x,
            "increment": increments,
            "cusum": s,
            "h": h,
            "signal": signal,
        }
    )
    notes = [
        f"design: p0 = {p0:.4f} -> p1 = {p1:.4f} "
        f"(odds ratio {(p1 / (1 - p1)) / (p0 / (1 - p0)):.2f}), h = {h:.3f}",
        f"weights: fail {w_fail:+.4f}, pass {w_pass:+.4f} per run",
        f"reset on signal: {reset_on_signal}",
    ]
    return BernoulliCusumResult(
        table=table, p0=p0, p1=p1, h=h, arl0=np.nan, first_signal_index=first, notes=notes
    )


def calibrate_cusum_h(
    p0: float,
    p1: float,
    *,
    target_arl0: float = 500.0,
    n_sim: int = 20000,
    max_run: int = 20000,
    seed: int = 20080719,
) -> tuple[float, float]:
    """Find the h that gives target_arl0 when the fail rate is p0.

    Returns (h, achieved_arl0). Simulation + bisection, since the usual ARL
    approximations aren't great for this at low p0.
    """
    rng = np.random.default_rng(seed)
    w_fail = np.log(p1 / p0)
    w_pass = np.log((1 - p1) / (1 - p0))
    stream = rng.random((n_sim, max_run)) < p0
    # numpy's stubs resolve this to the one-argument overload; it is an array
    inc = cast(np.ndarray, np.where(stream, w_fail, w_pass))

    # simulate the paths once and reuse them for every h
    acc = np.zeros(n_sim)
    paths = np.empty((n_sim, max_run), dtype=np.float32)
    for t in range(max_run):
        acc = np.maximum(0.0, acc + inc[:, t])
        paths[:, t] = acc

    def arl(h: float) -> float:
        hit = paths > h
        any_hit = hit.any(axis=1)
        first = np.where(any_hit, hit.argmax(axis=1) + 1, max_run)
        # paths that never signal count as max_run, so this is a bit low
        return float(first.mean())

    lo, hi = 0.5, 20.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if arl(mid) < target_arl0:
            lo = mid
        else:
            hi = mid
    h = (lo + hi) / 2
    return float(h), arl(h)
