"""Change points in a pass/fail sequence.

The Phase I baseline starts at ISO week 35 because that is where the weekly
rates settle (D7), a judgement made at week grain. This module locates the
change at run grain instead:

- the exact maximum-likelihood single change point of a Bernoulli sequence
  (profile likelihood over every split, so no search heuristic);
- a likelihood-ratio test of "no change", calibrated by permutation. Plain
  permutation assumes runs are exchangeable. The weekly rates are
  overdispersed (sigma_z = 1.46), which makes that assumption optimistic, so a
  block permutation that shuffles whole days is reported as well;
- a parametric-bootstrap confidence interval for the change location;
- binary segmentation for further change points, splitting a segment only
  while the block-permutation p-value stays below alpha.

Split ``k`` means the first segment is runs ``[0, k)`` (zero-based), so the
first run of the new regime is run ``k + 1`` in 1-based run numbering.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike


def _segment_loglik(s: ArrayLike, n: ArrayLike) -> np.ndarray:
    """Maximised Bernoulli log-likelihood of segments with ``s`` fails in ``n`` runs."""
    s = np.asarray(s, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = s / n
        a = np.where(s > 0, s * np.log(np.where(s > 0, p, 1.0)), 0.0)
        b = np.where(n - s > 0, (n - s) * np.log(np.where(n - s > 0, 1 - p, 1.0)), 0.0)
    return a + b


def profile_loglik(x: np.ndarray, min_segment: int = 1) -> np.ndarray:
    """Log-likelihood of the best two-segment fit at every split.

    ``x`` is 1-D (one sequence) or 2-D (one sequence per row). Entry ``k`` is
    the fit with the first segment ``[0, k)``; splits leaving a segment shorter
    than ``min_segment`` are NaN.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    n = x.shape[1]
    if min_segment < 1:
        raise ValueError("min_segment must be at least 1")
    if n < 2 * min_segment:
        raise ValueError(f"sequence of {n} runs is too short for min_segment={min_segment}")
    cum = np.cumsum(x, axis=1)
    total = cum[:, -1:]
    k = np.arange(1, n)
    s_left = cum[:, :-1]
    ll = _segment_loglik(s_left, k) + _segment_loglik(total - s_left, n - k)
    out = np.full((x.shape[0], n), np.nan)
    out[:, 1:] = ll
    out[:, :min_segment] = np.nan
    out[:, n - min_segment + 1 :] = np.nan
    return out


def lr_statistic(x: np.ndarray, min_segment: int = 1) -> np.ndarray:
    """2 x (best split log-likelihood - no-change log-likelihood), per row."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    null = _segment_loglik(x.sum(axis=1), np.full(x.shape[0], x.shape[1]))
    return 2.0 * (np.nanmax(profile_loglik(x, min_segment), axis=1) - null)


@dataclass(frozen=True)
class ChangePointFit:
    """The maximum-likelihood single change point of one sequence."""

    k: int
    lr_stat: float
    rate_before: float
    rate_after: float
    n_before: int
    n_after: int
    profile: np.ndarray = field(repr=False)


def fit_single(x: np.ndarray, min_segment: int = 1) -> ChangePointFit:
    """Exact MLE of one change point; ties go to the earliest split."""
    x = np.asarray(x, dtype=float)
    prof = profile_loglik(x, min_segment)[0]
    k = int(np.nanargmax(prof))
    null = float(_segment_loglik(x.sum(), len(x)))
    return ChangePointFit(
        k=k,
        lr_stat=float(2.0 * (prof[k] - null)),
        rate_before=float(x[:k].mean()),
        rate_after=float(x[k:].mean()),
        n_before=k,
        n_after=len(x) - k,
        profile=prof,
    )


def _block_permutations(blocks: np.ndarray, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """Index arrays that reorder whole contiguous blocks, one row per permutation."""
    blocks = np.asarray(blocks)
    change = np.flatnonzero(blocks[1:] != blocks[:-1]) + 1
    bounds = np.r_[0, change, len(blocks)]
    pieces = [np.arange(a, b) for a, b in zip(bounds[:-1], bounds[1:], strict=True)]
    return np.stack(
        [np.concatenate([pieces[i] for i in rng.permutation(len(pieces))]) for _ in range(n_perm)]
    )


def permutation_null(
    x: np.ndarray,
    *,
    min_segment: int,
    n_perm: int,
    rng: np.random.Generator,
    blocks: np.ndarray | None = None,
) -> np.ndarray:
    """LR statistics of ``n_perm`` reorderings of ``x`` (the no-change null).

    With ``blocks`` (a label per run; runs with the same label must be
    contiguous), whole blocks are shuffled instead of single runs, which keeps
    any within-block clustering of failures intact under the null.
    """
    x = np.asarray(x, dtype=float)
    if blocks is None:
        perms = rng.permuted(np.tile(x, (n_perm, 1)), axis=1)
    else:
        perms = x[_block_permutations(blocks, n_perm, rng)]
    return lr_statistic(perms, min_segment)


def permutation_pvalue(
    x: np.ndarray,
    *,
    min_segment: int,
    n_perm: int,
    rng: np.random.Generator,
    blocks: np.ndarray | None = None,
) -> float:
    """Permutation p-value of the LR statistic, (1 + #{T* >= T}) / (1 + n_perm).

    See :func:`permutation_null` for ``blocks``. The smallest attainable value
    is ``1 / (1 + n_perm)``.
    """
    observed = float(lr_statistic(x, min_segment)[0])
    null = permutation_null(x, min_segment=min_segment, n_perm=n_perm, rng=rng, blocks=blocks)
    return pvalue_from_null(observed, null)


def pvalue_from_null(observed: float, null: np.ndarray) -> float:
    """(1 + #{null >= observed}) / (1 + len(null)), counting float-noise ties as ties."""
    null = np.asarray(null, dtype=float)
    return float((1 + np.sum(null >= observed - 1e-9)) / (1 + len(null)))


def bootstrap_location_ci(
    fit: ChangePointFit,
    *,
    n_boot: int,
    rng: np.random.Generator,
    level: float = 0.95,
    min_segment: int = 1,
) -> tuple[int, int, np.ndarray]:
    """Basic parametric-bootstrap interval for the split ``k``.

    Simulates sequences from the fitted two-rate model, refits the split, and
    inverts the distribution of ``k* - k``. Returns ``(lo, hi, k_star)``.
    """
    n = fit.n_before + fit.n_after
    p = np.r_[np.full(fit.n_before, fit.rate_before), np.full(fit.n_after, fit.rate_after)]
    sims = rng.random((n_boot, n)) < p
    prof = profile_loglik(sims, min_segment)
    k_star = np.nanargmax(prof, axis=1)
    err = k_star - fit.k
    q_lo, q_hi = np.quantile(err, [(1 - level) / 2, (1 + level) / 2])
    lo = int(max(min_segment, np.floor(fit.k - q_hi)))
    hi = int(min(n - min_segment, np.ceil(fit.k - q_lo)))
    return lo, hi, k_star


@dataclass(frozen=True)
class SplitTest:
    """One segment tested during binary segmentation."""

    start: int
    stop: int
    k: int
    lr_stat: float
    p_perm: float
    p_block: float
    accepted: bool


def binary_segmentation(
    x: np.ndarray,
    blocks: np.ndarray,
    *,
    min_segment: int,
    alpha: float,
    n_perm: int,
    rng: np.random.Generator,
    max_changepoints: int = 5,
) -> tuple[list[int], list[SplitTest]]:
    """Split recursively while the block-permutation p-value is below ``alpha``.

    Returns the accepted change points (absolute split indices, ascending) and
    every test performed, in the order performed. Segments shorter than
    ``2 * min_segment`` are not tested. The per-test alpha is not adjusted for
    the number of tests, and later tests are conditional on earlier splits,
    so treat secondary change points as descriptive.
    """
    x = np.asarray(x, dtype=float)
    blocks = np.asarray(blocks)
    queue = [(0, len(x))]
    cps: list[int] = []
    tests: list[SplitTest] = []
    while queue and len(cps) < max_changepoints:
        a, b = queue.pop(0)
        if b - a < 2 * min_segment:
            continue
        seg = x[a:b]
        if seg.min() == seg.max():
            continue
        fit = fit_single(seg, min_segment)
        p_perm = permutation_pvalue(seg, min_segment=min_segment, n_perm=n_perm, rng=rng)
        p_block = permutation_pvalue(
            seg, min_segment=min_segment, n_perm=n_perm, rng=rng, blocks=blocks[a:b]
        )
        accepted = p_block < alpha
        tests.append(SplitTest(a, b, a + fit.k, fit.lr_stat, p_perm, p_block, accepted))
        if accepted:
            cps.append(a + fit.k)
            queue += [(a, a + fit.k), (a + fit.k, b)]
    return sorted(cps), tests


def segments(x: np.ndarray, changepoints: list[int]) -> list[tuple[int, int, int, float]]:
    """``(start, stop, n_fail, rate)`` for each segment between change points."""
    x = np.asarray(x, dtype=float)
    bounds = [0, *sorted(changepoints), len(x)]
    return [
        (a, b, int(x[a:b].sum()), float(x[a:b].mean()))
        for a, b in zip(bounds[:-1], bounds[1:], strict=True)
    ]
