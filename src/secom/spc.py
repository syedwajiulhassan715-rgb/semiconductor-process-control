"""Statistical process control charts.

Three things in here are the parts that usually get done wrong, and are the
reason this is hand-rolled rather than pulled from a library:

**Phase I / Phase II are separated.** Control limits are estimated once from a
window that has been argued to be stable, then held fixed while later data is
judged against them. Recomputing limits from all the data -- the default in most
tutorials -- lets an excursion inflate the very limits meant to catch it.

**Limits come from short-term variation, not the overall standard deviation.**
For individuals charts sigma is estimated from the average moving range over
d2(2), which is insensitive to a sustained shift sitting inside the window. The
sample standard deviation is not.

**Attribute charts are tested for overdispersion.** A textbook p-chart assumes
the only run-to-run variation in failure rate is binomial. Real fab yield data
is almost never that well behaved, and an uncorrected p-chart then alarms
constantly. This module measures the dispersion and applies the Laney p' chart
when it is warranted, rather than assuming either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sps

#: Hartley's d2 for subgroup size n -- E[range] / sigma for a normal sample.
#: d2(2) = 1.128379... is 2/sqrt(pi), used for moving-range sigma estimation.
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
    """Unbiasing constant for the sample standard deviation.

    ``E[s] = c4(n) * sigma``. Computed from the exact gamma-function definition
    rather than a lookup table so it is valid at any n.
    """
    if n < 2:
        return np.nan
    from scipy.special import gammaln

    return float(
        np.sqrt(2.0 / (n - 1)) * np.exp(gammaln(n / 2.0) - gammaln((n - 1) / 2.0))
    )


# --------------------------------------------------------------------------
# Variables charts
# --------------------------------------------------------------------------


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
    """Phase I individuals-chart limits with iterative exclusion of outliers.

    A Phase I baseline is only usable if it is itself in control. Points outside
    the trial limits are excluded and the limits recomputed, up to
    ``trim_iterations`` times or until stable -- the standard Phase I practice of
    removing assignable-cause points before fixing the limits. The number
    excluded is reported so that an over-trimmed baseline is visible rather than
    hidden.
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
    """EWMA chart against fixed Phase I ``limits``.

    Defaults of lambda = 0.2 with L = 2.7 are the standard small-shift design
    (Montgomery): roughly an in-control ARL of 500 with good sensitivity to a
    1-sigma sustained shift, which an individuals chart detects only slowly.
    Limits use the exact time-varying variance factor, so the first few points
    are correctly tighter rather than being drawn at the asymptote.
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
    width = L * limits.sigma * np.sqrt(
        (lam / (2 - lam)) * (1 - (1 - lam) ** (2 * t))
    )
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


# --------------------------------------------------------------------------
# Nelson run rules
# --------------------------------------------------------------------------

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
    """Evaluate all eight Nelson rules against fixed limits.

    Returns a boolean frame with one column per rule. Rules 7 and 8 detect
    limits that no longer match the process (stratification and mixture) rather
    than the process moving, which is why they are worth keeping: on this
    dataset they are the ones that fire when a sensor's variance changes.
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


# --------------------------------------------------------------------------
# Attribute chart for yield
# --------------------------------------------------------------------------


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
    """Proportion-defective chart with variable subgroup sizes.

    Control limits vary by subgroup because the sizes vary, which is the whole
    point of charting a proportion rather than a count.

    Overdispersion is tested explicitly. ``sigma_z`` is the standard deviation of
    the moving range of the standardised proportions divided by d2(2) -- Laney's
    statistic. Under pure binomial variation it is 1. Above roughly 1.2 the
    between-subgroup variation exceeds what binomial sampling can explain, the
    conventional p-chart is too tight, and limits are widened by ``sigma_z``
    (the Laney p' chart). A complementary chi-square dispersion test is reported
    alongside so the call does not rest on one threshold.

    ``baseline`` selects the subgroups used to estimate the centre line; when
    omitted all subgroups are used. Supplying a stable window makes this a
    Phase I / Phase II chart rather than a self-referential one.

    ``dispersion_subset`` separately selects the subgroups used to estimate
    ``sigma_z``, defaulting to all of them. The two are deliberately decoupled.
    The centre line should be pinned to a window argued to be stable, or an
    excursion redefines "normal" as it happens. The dispersion estimate should
    use every subgroup available: it is a variance component of the process, the
    moving-range estimator is by construction insensitive to sustained level
    shifts, and a short Phase I window would leave it resting on three or four
    moving ranges. On this dataset Phase I is five weekly subgroups, which is
    far too thin to estimate a dispersion factor from.
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

    # Laney's sigma_z from the moving range of standardised proportions.
    disp = (
        np.ones(len(sizes), dtype=bool)
        if dispersion_subset is None
        else np.asarray(dispersion_subset, bool)
    )
    zb = z[disp]
    zb = zb[np.isfinite(zb)]
    sigma_z = float(np.mean(np.abs(np.diff(zb))) / D2[2]) if len(zb) > 1 else 1.0

    # Independent chi-square test of binomial dispersion on the baseline.
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

    notes = [
        f"sigma_z = {sigma_z:.3f} "
        f"({'overdispersed, Laney p-prime limits applied' if overdispersed else 'consistent with binomial variation'})",
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


# --------------------------------------------------------------------------
# Low-rate attribute chart
# --------------------------------------------------------------------------


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
    """Chart the number of passing runs between consecutive failures.

    Motivation: at a failure rate of 6.6% a proportion chart needs a subgroup of
    at least 127 runs before its lower limit rises above zero, so at any
    realistic subgrouping it is structurally incapable of signalling that yield
    has *improved*. The g-chart covers that gap. It charts the interval between
    failures directly, needs no subgrouping, and its upper limit responds as
    soon as one unusually long clean run occurs.

    Limits are exact geometric quantiles, not mean +/- 3 sigma. The geometric
    distribution is strongly right-skewed, so symmetric limits would put the
    lower limit below zero and place the upper limit in the wrong tail
    probability. ``alpha`` defaults to 0.0027, matching the false-alarm rate of a
    conventional 3-sigma chart so the two are comparable.

    **Known limitation, and the reason this is not the whole answer.** The
    lower limit is useful only at very low failure rates. Because P(gap = 0)
    equals ``p`` exactly, as soon as ``p`` exceeds ``alpha / 2`` -- about 0.14%
    at the default alpha -- the lower quantile pins to zero and *no* single gap,
    however short, can breach it. At the 3.1% baseline rate of this dataset the
    lower limit is zero, so this chart detects improvement but not degradation.
    Degradation detection is the job of :func:`bernoulli_cusum`, which
    accumulates evidence across runs instead of judging each gap alone. Use the
    two together; neither is sufficient by itself.

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

    # Passing runs strictly between consecutive failures.
    gaps = np.diff(fail_positions) - 1
    p = float(baseline_p if baseline_p is not None else len(fail_positions) / n_total)
    p = min(max(p, 1e-9), 1 - 1e-9)

    # Number of failures (successes=passes) before the first success, support 0,1,2,...
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


# --------------------------------------------------------------------------
# Memory-based attribute chart
# --------------------------------------------------------------------------


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
    """One-sided upper CUSUM on Bernoulli outcomes (Reynolds & Stoumbos).

    A Shewhart p-chart judges each subgroup on its own and therefore reacts to
    spikes, not to levels. A sustained shift from ``p0`` to ``p1`` is exactly the
    case it handles worst. The CUSUM accumulates the log-likelihood ratio for
    each run:

        +log(p1/p0)            on a fail
        +log((1-p1)/(1-p0))    on a pass   (negative, so good runs pay down)

    and signals when the running total exceeds ``h``. It needs no subgrouping,
    so it is immune to the rational-subgrouping argument entirely, and it uses
    every run the moment it happens.

    ``h`` should be obtained from :func:`calibrate_cusum_h` for a stated
    in-control ARL rather than guessed.

    ``reset_on_signal`` restarts the statistic at zero once it crosses ``h``,
    which is how a CUSUM is actually operated: it signals, an engineer
    investigates, and the chart is restarted against the process as it now
    stands. Without the reset the statistic keeps integrating -- on this dataset
    the opening excursion drives it to ~28, and since each passing run pays down
    only ``log((1-p1)/(1-p0))`` it would take roughly 900 consecutive good runs
    to fall back below the limit. The chart would then read as alarming for the
    entire campaign, including the five weeks the process was demonstrably
    healthy, and would be incapable of flagging the later excursion as a *new*
    event. Set it to False only for a retrospective view of cumulative evidence.
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
        acc = max(0.0, acc + inc)
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
    """Find the CUSUM threshold ``h`` giving ``target_arl0`` under ``p0``.

    Returns ``(h, achieved_arl0)``. Solved by simulation rather than by a
    Markov-chain approximation because the Bernoulli CUSUM lives on a lattice of
    only two increment sizes, where the usual continuous approximations to the
    ARL are poor at low ``p0``.
    """
    rng = np.random.default_rng(seed)
    w_fail = np.log(p1 / p0)
    w_pass = np.log((1 - p1) / (1 - p0))
    stream = rng.random((n_sim, max_run)) < p0
    inc = np.where(stream, w_fail, w_pass)

    # Running maxima of the CUSUM path, computed once and reused per candidate h.
    acc = np.zeros(n_sim)
    paths = np.empty((n_sim, max_run), dtype=np.float32)
    for t in range(max_run):
        acc = np.maximum(0.0, acc + inc[:, t])
        paths[:, t] = acc

    def arl(h: float) -> float:
        hit = paths > h
        any_hit = hit.any(axis=1)
        first = np.where(any_hit, hit.argmax(axis=1) + 1, max_run)
        # Censored runs bias ARL downwards; report the censored mean and flag it.
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
