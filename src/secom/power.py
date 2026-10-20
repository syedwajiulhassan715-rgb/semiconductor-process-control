"""Detection power of the yield monitoring schemes, by simulation.

The report argues the weekly p-chart can't see the yield shift and the
Bernoulli CUSUM can. This module measures that instead of arguing it: simulate
pass/fail streams whose failure rate steps from the Phase I rate p0 to r * p0,
run every scheme over the same streams, and record when each first signals.

Conventions
-----------
- Zero-state: the shift is present from run 1, so a scheme's first-signal run
  is its detection delay. Shewhart charts get their most favourable case,
  since the first subgroup is wholly post-shift.
- Weekly subgroups cycle through the observed weekly sizes, starting at a
  random week for each simulated stream. Only complete subgroups are judged.
- The g-chart is run live: it signals as soon as the count of passes since the
  last failure (or since the start) exceeds the UCL, which is the same rule as
  a completed gap above the UCL, just without waiting for the next failure.
- Common random numbers: one uniform matrix is shared across shift ratios, so
  differences between ratios are not Monte Carlo noise.
- Matched designs: the deployed CUSUM (ARL0 ~500 runs) false-alarms far more
  often than a weekly p-chart, so comparing them as deployed partly compares
  false-alarm budgets. The ``*_matched`` CUSUMs have h chosen so their chance
  of a false alarm within the longest horizon equals the Shewhart chart's.

- Direction: a detection is a signal in the direction of the shift. After a
  fall in the fail rate, a p-chart point above the UCL is a false alarm, not
  a detection, so ``side`` restricts which signals count. In control
  (``side="both"``) every signal counts, as a false alarm.

A first-signal value of :data:`NO_SIGNAL` means no signal within the horizon.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

from .spc import g_chart

NO_SIGNAL = -1

#: Which signals count: rate rises, rate falls, or either (in control).
SIDES = ("both", "degradation", "improvement")

#: Scheme name -> (label, direction it can detect).
SCHEMES: dict[str, tuple[str, str]] = {
    "p_chart": ("weekly p-chart (naive)", "both"),
    "laney_p_prime": ("weekly Laney p′", "both"),
    "laney_p_prime_phase1_sigma": ("weekly Laney p′, Phase I-centred σ_z", "both"),
    "g_chart": ("g-chart (runs between failures)", "both"),
    "cusum_upper": ("Bernoulli CUSUM, upper", "degradation"),
    "cusum_upper_matched": ("Bernoulli CUSUM, upper, matched to p-chart", "degradation"),
    "cusum_lower": ("Bernoulli CUSUM, lower", "improvement"),
    "cusum_lower_matched": ("Bernoulli CUSUM, lower, matched to g-chart", "improvement"),
}


@dataclass(frozen=True)
class MonitorDesign:
    """Every chart parameter the simulation needs, fixed from Phase I."""

    p0: float
    subgroup_sizes: tuple[int, ...]
    sigma_z: float
    sigma_z_phase1: float
    cusum_upper_p1: float
    cusum_upper_h: float
    cusum_lower_p1: float
    cusum_lower_h: float
    g_lcl: float
    g_ucl: float
    cusum_upper_matched_h: float | None = None
    cusum_lower_matched_h: float | None = None
    k: float = 3.0


def g_limits(p: float, alpha: float = 0.0027) -> tuple[float, float]:
    """Exact geometric (LCL, UCL) at failure rate ``p``, as :func:`spc.g_chart` sets them."""
    res = g_chart(np.array([0, 1]), n_total=2, baseline_p=p, alpha=alpha)
    return res.lcl, res.ucl


def _subgroup_layout(sizes: np.ndarray, offset: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """(starts, ends) of the complete subgroups within ``horizon`` runs."""
    seq = np.roll(sizes, -offset)
    reps = horizon // int(seq.sum()) + 2
    tiled = np.tile(seq, reps)
    ends = np.cumsum(tiled)
    keep = ends <= horizon
    ends = ends[keep]
    return ends - tiled[keep], ends


def _check_side(side: str) -> None:
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")


def _shewhart_limits(
    n: np.ndarray, p0: float, k: float, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    sigma = np.sqrt(p0 * (1 - p0) / n)
    return (
        np.clip(p0 - k * scale * sigma, 0.0, 1.0),
        np.clip(p0 + k * scale * sigma, 0.0, 1.0),
    )


def shewhart_first_signal(
    fails: np.ndarray,
    sizes: np.ndarray,
    offsets: np.ndarray,
    *,
    p0: float,
    k: float = 3.0,
    scale: float = 1.0,
    side: str = "both",
) -> np.ndarray:
    """First-signal run (1-based) of a p-chart per stream, or NO_SIGNAL.

    ``scale`` widens the limits (1 for the naive chart, sigma_z for Laney p′).
    The signal is dated to the last run of the subgroup that breaches.
    ``side`` picks the limit that counts: UCL for degradation, LCL for
    improvement, either for both.
    """
    _check_side(side)
    fails = np.asarray(fails, dtype=bool)
    sizes = np.asarray(sizes, dtype=int)
    n_sim, horizon = fails.shape
    cum = np.zeros((n_sim, horizon + 1), dtype=np.int64)
    np.cumsum(fails, axis=1, out=cum[:, 1:])
    out = np.full(n_sim, NO_SIGNAL, dtype=np.int64)
    for off in np.unique(offsets):
        rows = np.flatnonzero(offsets == off)
        starts, ends = _subgroup_layout(sizes, int(off), horizon)
        if len(ends) == 0:
            continue
        n = ends - starts
        lcl, ucl = _shewhart_limits(n.astype(float), p0, k, scale)
        p = (cum[np.ix_(rows, ends)] - cum[np.ix_(rows, starts)]) / n
        up, down = p > ucl, p < lcl
        sig = {"both": up | down, "degradation": up, "improvement": down}[side]
        hit = sig.any(axis=1)
        out[rows[hit]] = ends[sig[hit].argmax(axis=1)]
    return out


def exact_shewhart_detection(
    p: float,
    sizes: np.ndarray,
    horizon: int,
    *,
    p0: float,
    k: float = 3.0,
    scale: float = 1.0,
    side: str = "both",
) -> float:
    """P(signal within ``horizon``) for the p-chart, exactly, averaged over start weeks.

    ``side`` as in :func:`shewhart_first_signal`.

    Subgroups are independent binomials, so no simulation is needed. This is
    the reference the simulated Shewhart results are checked against.
    """
    _check_side(side)
    sizes = np.asarray(sizes, dtype=int)
    probs = []
    for off in range(len(sizes)):
        _, ends = _subgroup_layout(sizes, off, horizon)
        n = np.diff(np.r_[0, ends]).astype(float)
        if len(n) == 0:
            probs.append(0.0)
            continue
        lcl, ucl = _shewhart_limits(n, p0, k, scale)
        above = sps.binom.sf(np.floor(n * ucl), n, p)
        below = np.where(lcl > 0, sps.binom.cdf(np.ceil(n * lcl) - 1, n, p), 0.0)
        if side == "degradation":
            below = np.zeros_like(below)
        elif side == "improvement":
            above = np.zeros_like(above)
        probs.append(1.0 - float(np.prod(1.0 - (above + below))))
    return float(np.mean(probs))


def cusum_first_signal(fails: np.ndarray, *, p0: float, p1: float, h: float) -> np.ndarray:
    """First-signal run (1-based) of a one-sided Bernoulli CUSUM per stream.

    ``p1 > p0`` gives the upper (degradation) CUSUM of :func:`spc.bernoulli_cusum`;
    ``p1 < p0`` gives the lower (improvement) CUSUM, which accumulates evidence
    from passes and is paid down by failures.
    """
    fails = np.asarray(fails, dtype=bool)
    n_sim, horizon = fails.shape
    w_fail = np.log(p1 / p0)
    w_pass = np.log((1 - p1) / (1 - p0))
    acc = np.zeros(n_sim)
    out = np.full(n_sim, NO_SIGNAL, dtype=np.int64)
    open_ = np.ones(n_sim, dtype=bool)
    for t in range(horizon):
        acc = np.maximum(0.0, acc + np.where(fails[:, t], w_fail, w_pass))
        hit = open_ & (acc > h)
        out[hit] = t + 1
        open_ &= ~hit
        if not open_.any():
            break
    return out


def g_chart_first_signal(
    fails: np.ndarray, *, lcl: float, ucl: float, side: str = "both"
) -> np.ndarray:
    """First-signal run (1-based) of a live g-chart per stream.

    Above UCL (improvement): the running count of passes since the last
    failure exceeds ``ucl``. Below LCL (degradation): a failure arrives after
    fewer than ``lcl`` passes, which never fires when the LCL is pinned at 0.
    """
    _check_side(side)
    use_lcl = side in ("both", "degradation")
    use_ucl = side in ("both", "improvement")
    fails = np.asarray(fails, dtype=bool)
    n_sim, horizon = fails.shape
    count = np.zeros(n_sim, dtype=np.int64)
    out = np.full(n_sim, NO_SIGNAL, dtype=np.int64)
    open_ = np.ones(n_sim, dtype=bool)
    for t in range(horizon):
        f = fails[:, t]
        hit = np.where(f, use_lcl & (count < lcl), use_ucl & (count + 1 > ucl)) & open_
        count = np.where(f, 0, count + 1)
        out[hit] = t + 1
        open_ &= ~hit
        if not open_.any():
            break
    return out


def cusum_path_max(fails: np.ndarray, *, p0: float, p1: float) -> np.ndarray:
    """Largest value the CUSUM statistic reaches on each stream (no reset)."""
    fails = np.asarray(fails, dtype=bool)
    w_fail = np.log(p1 / p0)
    w_pass = np.log((1 - p1) / (1 - p0))
    acc = np.zeros(fails.shape[0])
    peak = np.zeros(fails.shape[0])
    for t in range(fails.shape[1]):
        acc = np.maximum(0.0, acc + np.where(fails[:, t], w_fail, w_pass))
        np.maximum(peak, acc, out=peak)
    return peak


def calibrate_h_for_false_alarm(
    p0: float, p1: float, *, horizon: int, target: float, n_sim: int, seed: int
) -> float:
    """h giving P(false alarm within ``horizon`` runs) = ``target``.

    A CUSUM signals within the horizon exactly when its path maximum exceeds
    h, so h is the (1 - target) quantile of in-control path maxima. No search.
    """
    if not 0 < target < 1:
        raise ValueError("target must lie strictly in (0, 1)")
    rng = np.random.default_rng(seed)
    peak = cusum_path_max(rng.random((n_sim, horizon)) < p0, p0=p0, p1=p1)
    return float(np.quantile(peak, 1.0 - target))


def first_signals(
    fails: np.ndarray, offsets: np.ndarray, design: MonitorDesign, *, side: str = "both"
) -> dict[str, np.ndarray]:
    """First-signal runs of every scheme in :data:`SCHEMES` over the same streams.

    Only signals on ``side`` count, so a one-sided CUSUM facing a shift in the
    other direction never detects it. The matched CUSUMs are included only
    when the design sets their h.
    """
    _check_side(side)
    sizes = np.asarray(design.subgroup_sizes, dtype=int)
    p0, k = design.p0, design.k
    out = {
        "p_chart": shewhart_first_signal(fails, sizes, offsets, p0=p0, k=k, side=side),
        "laney_p_prime": shewhart_first_signal(
            fails, sizes, offsets, p0=p0, k=k, scale=design.sigma_z, side=side
        ),
        "laney_p_prime_phase1_sigma": shewhart_first_signal(
            fails, sizes, offsets, p0=p0, k=k, scale=design.sigma_z_phase1, side=side
        ),
        "g_chart": g_chart_first_signal(fails, lcl=design.g_lcl, ucl=design.g_ucl, side=side),
        "cusum_upper": cusum_first_signal(
            fails, p0=p0, p1=design.cusum_upper_p1, h=design.cusum_upper_h
        ),
        "cusum_lower": cusum_first_signal(
            fails, p0=p0, p1=design.cusum_lower_p1, h=design.cusum_lower_h
        ),
    }
    if design.cusum_upper_matched_h is not None:
        out["cusum_upper_matched"] = cusum_first_signal(
            fails, p0=p0, p1=design.cusum_upper_p1, h=design.cusum_upper_matched_h
        )
    if design.cusum_lower_matched_h is not None:
        out["cusum_lower_matched"] = cusum_first_signal(
            fails, p0=p0, p1=design.cusum_lower_p1, h=design.cusum_lower_matched_h
        )
    for name, signals in out.items():
        detects = SCHEMES[name][1]
        if side != "both" and detects not in ("both", side):
            signals[:] = NO_SIGNAL
    return {name: out[name] for name in SCHEMES if name in out}


def summarise_signals(first: np.ndarray, horizons: tuple[int, ...]) -> dict[str, float | None]:
    """Detection probability by horizon, plus delay statistics.

    ``median_run_length`` treats streams that never signal as infinitely late,
    so it is None when fewer than half signal within the longest horizon.
    ``mean_delay_given_detection`` is conditional and flatters a weak scheme;
    read it alongside the detection probability.
    """
    first = np.asarray(first)
    detected = first != NO_SIGNAL
    row: dict[str, float | None] = {
        f"p_detect_{h}": float(np.mean(detected & (first <= h))) for h in horizons
    }
    t = np.where(detected, first, np.inf).astype(float)
    med = float(np.median(t))
    row["median_run_length"] = med if np.isfinite(med) else None
    row["mean_delay_given_detection"] = float(first[detected].mean()) if detected.any() else None
    return row


def side_for_ratio(ratio: float) -> str:
    """The signal direction that counts as detecting a shift of ``ratio``."""
    if np.isclose(ratio, 1.0):
        return "both"
    return "degradation" if ratio > 1 else "improvement"


def simulate_detection(
    design: MonitorDesign,
    ratios: tuple[float, ...],
    *,
    horizons: tuple[int, ...],
    n_sim: int,
    seed: int,
) -> pd.DataFrame:
    """One row per (scheme, shift ratio): detection probability and delay.

    Ratio 1 is the in-control case, where ``p_detect_*`` is the false-alarm
    probability within that horizon. Otherwise only signals in the direction
    of the shift count as detections.
    """
    rng = np.random.default_rng(seed)
    horizon = int(max(horizons))
    uniforms = rng.random((n_sim, horizon))
    offsets = rng.integers(0, len(design.subgroup_sizes), n_sim)
    rows = []
    for r in ratios:
        p = min(r * design.p0, 1.0)
        side = side_for_ratio(r)
        signals = first_signals(uniforms < p, offsets, design, side=side)
        for scheme, first in signals.items():
            rows.append(
                {
                    "scheme": scheme,
                    "ratio": float(r),
                    "side": side,
                    "fail_rate": float(p),
                    "n_sim": int(n_sim),
                    **summarise_signals(first, horizons),
                }
            )
    return pd.DataFrame(rows)
