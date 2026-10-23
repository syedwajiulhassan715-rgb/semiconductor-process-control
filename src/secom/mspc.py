"""PCA-based multivariate SPC: Hotelling T², SPE (Q) and per-sensor contributions.

The model is fitted once on the Phase I baseline and its limits are held fixed
for every run, the same Phase I / Phase II split the univariate charts use.

Scaling is mean / SD of the baseline (autoscaling) rather than median / MAD:
the T² and SPE limits are derived under multivariate normality from the
sample mean and covariance, and several SECOM sensors have a baseline MAD of
zero, so a robust z is undefined for them. Sensors that are constant within
the baseline carry no variance to model and are left out.

Only the retained components are ever inverted. On this data the full
baseline covariance has eigenvalues down to ~1e-11, so a T² built on the full
inverse would be dominated by numerical noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats as sps


@dataclass(frozen=True)
class PCAModel:
    """PCA model of the baseline window and its fixed control limits."""

    sensors: list[str]
    mean: np.ndarray
    scale: np.ndarray
    loadings: np.ndarray  # p x A
    eigenvalues: np.ndarray  # all p eigenvalues of the baseline correlation matrix
    n_components: int
    n_baseline: int
    alpha: float
    t2_limit: float  # F-based, for new (Phase II) observations
    t2_limit_phase1: float  # Beta-based, for the baseline points themselves
    spe_limit: float
    spe_limit_method: str
    dropped_constant: list[str] = field(default_factory=list)

    @property
    def explained(self) -> float:
        """Fraction of baseline variance captured by the retained components."""
        return float(self.eigenvalues[: self.n_components].sum() / self.eigenvalues.sum())

    @property
    def condition_number(self) -> float:
        """lambda_1 / lambda_A of the retained components."""
        lam = self.eigenvalues
        return float(lam[0] / lam[self.n_components - 1])


def n_components_for_variance(eigenvalues: np.ndarray, target: float) -> int:
    """Smallest A whose leading eigenvalues explain at least ``target`` of the variance."""
    if not 0 < target <= 1:
        raise ValueError("target must lie in (0, 1]")
    lam = np.asarray(eigenvalues, dtype=float)
    cum = np.cumsum(lam) / lam.sum()
    return min(int(np.searchsorted(cum, target - 1e-12)) + 1, len(lam))


def t2_limit_phase2(a: int, n: int, alpha: float) -> float:
    """Upper limit for T² of a new observation: A(n²-1)/(n(n-A)) F(A, n-A)."""
    if not 0 < a < n:
        raise ValueError("need 0 < A < n")
    return float(a * (n**2 - 1) / (n * (n - a)) * sps.f.ppf(1 - alpha, a, n - a))


def t2_limit_phase1(a: int, n: int, alpha: float) -> float:
    """Upper limit for T² of a baseline point: (n-1)²/n Beta(A/2, (n-A-1)/2)."""
    if not 0 < a < n - 1:
        raise ValueError("need 0 < A < n - 1")
    return float((n - 1) ** 2 / n * sps.beta.ppf(1 - alpha, a / 2, (n - a - 1) / 2))


def spe_limit_jackson_mudholkar(residual_eigenvalues: np.ndarray, alpha: float) -> float:
    """Jackson-Mudholkar (1979) upper limit for SPE from the discarded eigenvalues.

    ``theta_i = sum(lam_j ** i)`` over the residual eigenvalues (those of the
    autoscaled baseline's covariance, i.e. its correlation matrix, after the
    A retained ones) and ``h0 = 1 - 2 theta1 theta3 / (3 theta2**2)``. The
    approximation treats ``(SPE / theta1) ** h0`` as normal, so it needs
    ``h0 > 0``: for ``h0 <= 0`` that power is decreasing (or constant) in SPE
    and the usual closed form, with its ``sqrt(2 theta2 h0**2)`` term, returns
    a *lower*-tail value that nearly every point exceeds. That happens when
    one residual eigenvalue dominates a long tail of small ones. NaN is
    returned then, and :func:`fit_pca_model` falls back to
    :func:`spe_limit_box`.
    """
    lam = np.asarray(residual_eigenvalues, dtype=float)
    lam = lam[lam > 0]
    t1, t2, t3 = (float(np.sum(lam**k)) for k in (1, 2, 3))
    h0 = 1 - 2 * t1 * t3 / (3 * t2**2)
    if not h0 > 0:
        return float("nan")
    c = sps.norm.ppf(1 - alpha)
    inner = c * np.sqrt(2 * t2 * h0**2) / t1 + 1 + t2 * h0 * (h0 - 1) / t1**2
    if not inner > 0:
        return float("nan")
    return float(t1 * inner ** (1 / h0))


def spe_limit_box(residual_eigenvalues: np.ndarray, alpha: float) -> float:
    """Box (1954) scaled chi-square limit g * chi2(h), g = t2/t1, h = t1²/t2.

    Matches the mean and variance of SPE; valid for any residual spectrum,
    so it is the fallback when Jackson-Mudholkar is undefined.
    """
    lam = np.asarray(residual_eigenvalues, dtype=float)
    t1, t2 = float(lam.sum()), float(np.sum(lam**2))
    g, h = t2 / t1, t1**2 / t2
    return float(g * sps.chi2.ppf(1 - alpha, h))


def fit_pca_model(
    baseline: np.ndarray,
    sensors: list[str],
    *,
    variance_target: float = 0.80,
    n_components: int | None = None,
    alpha: float = 0.0027,
) -> PCAModel:
    """Fit the PCA monitoring model on the ``baseline`` rows (n x p).

    ``n_components`` overrides ``variance_target`` when given.
    """
    x = np.asarray(baseline, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(sensors):
        raise ValueError("baseline must be n x p with one column per sensor")
    if np.isnan(x).any():
        raise ValueError("baseline contains NaN; impute before fitting")
    n = x.shape[0]
    sd = x.std(axis=0, ddof=1)
    keep = sd > 0
    dropped = [s for s, k in zip(sensors, keep, strict=True) if not k]
    x = x[:, keep]
    mean, scale = x.mean(axis=0), sd[keep]
    z = (x - mean) / scale

    _, s, vt = np.linalg.svd(z, full_matrices=False)
    lam = s**2 / (n - 1)
    a = n_components or n_components_for_variance(lam, variance_target)
    if not 0 < a < min(n - 1, z.shape[1]):
        raise ValueError(f"n_components={a} is out of range for a {n} x {z.shape[1]} baseline")

    residual = lam[a:]
    jm = spe_limit_jackson_mudholkar(residual, alpha)
    method = "jackson_mudholkar"
    if not np.isfinite(jm):
        jm, method = spe_limit_box(residual, alpha), "box"

    return PCAModel(
        sensors=[s for s, k in zip(sensors, keep, strict=True) if k],
        mean=mean,
        scale=scale,
        loadings=vt[:a].T,
        eigenvalues=lam,
        n_components=a,
        n_baseline=n,
        alpha=alpha,
        t2_limit=t2_limit_phase2(a, n, alpha),
        t2_limit_phase1=t2_limit_phase1(a, n, alpha),
        spe_limit=jm,
        spe_limit_method=method,
        dropped_constant=dropped,
    )


@dataclass(frozen=True)
class MSPCScores:
    """T² and SPE for a block of runs, with per-sensor contributions."""

    t2: np.ndarray
    spe: np.ndarray
    t2_contrib: np.ndarray  # n x p, rows sum to t2
    spe_contrib: np.ndarray  # n x p, rows sum to spe


def score(model: PCAModel, x: np.ndarray) -> MSPCScores:
    """Score runs ``x`` (n x p, columns in ``model.sensors`` order) against fixed limits.

    T² contribution of sensor j is z_j * (P Λ⁻¹ Pᵀ z)_j, the complete
    decomposition: it sums exactly to T² and may be negative for a sensor
    that offsets others. SPE contribution is the squared residual e_j².
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(model.sensors):
        raise ValueError("x must have one column per model sensor")
    z = (x - model.mean) / model.scale
    p = model.loadings
    lam = model.eigenvalues[: model.n_components]
    t = z @ p
    resid = z - t @ p.T
    t2_contrib = z * ((t / lam) @ p.T)
    spe_contrib = resid**2
    return MSPCScores(
        t2=t2_contrib.sum(axis=1),
        spe=spe_contrib.sum(axis=1),
        t2_contrib=t2_contrib,
        spe_contrib=spe_contrib,
    )


def cross_validated_scores(
    baseline: np.ndarray,
    sensors: list[str],
    *,
    n_components: int,
    n_folds: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold T² and SPE for every baseline row.

    Folds are contiguous blocks in time order, so each row is scored by a
    model that never saw it or its immediate neighbours. In-sample SPE is
    biased low (the components were fitted to those very rows), so limits
    meant for new runs should be calibrated on these scores instead.
    """
    x = np.asarray(baseline, dtype=float)
    n = x.shape[0]
    if not 2 <= n_folds <= n:
        raise ValueError("need 2 <= n_folds <= n")
    t2 = np.empty(n)
    spe = np.empty(n)
    for idx in np.array_split(np.arange(n), n_folds):
        train = np.ones(n, dtype=bool)
        train[idx] = False
        m = fit_pca_model(x[train], sensors, n_components=n_components)
        cols = np.array([sensors.index(s) for s in m.sensors], dtype=np.intp)
        sc = score(m, x[np.ix_(idx, cols)])
        t2[idx], spe[idx] = sc.t2, sc.spe
    return t2, spe


def empirical_limit(values: np.ndarray, alpha: float) -> float:
    """Upper (1 - alpha) sample quantile, linear interpolation."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        raise ValueError("no values")
    return float(np.quantile(v, 1 - alpha))


def contribution_share(contrib: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Mean share of each sensor in the selected rows' statistic.

    Negative T² contributions are clipped to zero before normalising, so each
    row's shares are non-negative and sum to one.
    """
    c = np.clip(np.asarray(contrib, dtype=float)[rows], 0.0, None)
    tot = c.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(tot > 0, c / tot, 0.0)
    return share.mean(axis=0) if len(share) else np.zeros(c.shape[1])
