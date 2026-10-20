"""Tests for secom.mspc and the multivariate stage.

Limits are checked by their defining property: under the model's own
assumptions (Gaussian data), the fraction of points above the limit should
match alpha.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import stats as sps

from secom.config import load_config
from secom.mspc import (
    contribution_share,
    cross_validated_scores,
    empirical_limit,
    fit_pca_model,
    n_components_for_variance,
    score,
    spe_limit_box,
    spe_limit_jackson_mudholkar,
    t2_limit_phase1,
    t2_limit_phase2,
)
from secom.pipeline import Paths, PipelineContext, run_pipeline

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
needs_data = pytest.mark.skipif(
    not (RAW / "secom.data").exists(),
    reason="run scripts/00_fetch_data.py to enable data-dependent tests",
)


def _gaussian(n: int, p: int, seed: int) -> np.ndarray:
    """Correlated Gaussian data with a decaying spectrum."""
    rng = np.random.default_rng(seed)
    mix = rng.normal(size=(p, p)) * np.linspace(3.0, 0.3, p)
    return rng.normal(size=(n, p)) @ mix.T


def _sensors(p: int) -> list[str]:
    return [f"S{i:03d}" for i in range(1, p + 1)]


class TestComponentCount:
    def test_variance_target(self) -> None:
        lam = np.array([4.0, 3.0, 2.0, 1.0])  # cumulative 0.4, 0.7, 0.9, 1.0
        assert n_components_for_variance(lam, 0.4) == 1
        assert n_components_for_variance(lam, 0.7) == 2
        assert n_components_for_variance(lam, 0.71) == 3
        assert n_components_for_variance(lam, 1.0) == 4

    def test_rejects_bad_target(self) -> None:
        with pytest.raises(ValueError):
            n_components_for_variance(np.ones(3), 0.0)


class TestLimits:
    def test_phase2_limit_formula(self) -> None:
        # A(n^2-1)/(n(n-A)) F(A, n-A) at A=2, n=20, alpha=0.05
        expected = 2 * 399 / (20 * 18) * sps.f.ppf(0.95, 2, 18)
        assert t2_limit_phase2(2, 20, 0.05) == pytest.approx(expected)
        assert t2_limit_phase2(2, 20, 0.05) == pytest.approx(7.879, abs=1e-3)

    def test_phase1_limit_is_tighter_than_phase2(self) -> None:
        # a baseline point helped fit the mean and covariance, so its T² is smaller
        assert t2_limit_phase1(5, 100, 0.01) < t2_limit_phase2(5, 100, 0.01)

    def test_phase1_limit_holds_in_sample(self) -> None:
        # T² of baseline points against their own fitted model follows
        # (n-1)^2/n Beta(A/2, (n-A-1)/2) for Gaussian data (exact for the full
        # Mahalanobis distance; the fit requires A < p, so A = p - 1 here)
        n, p, alpha = 60, 4, 0.05
        exceed = []
        for seed in range(400):
            x = _gaussian(n, p, seed)
            m = fit_pca_model(x, _sensors(p), n_components=p - 1, alpha=alpha)
            sc = score(m, x)
            exceed.append(np.mean(sc.t2 > m.t2_limit_phase1))
        assert np.mean(exceed) == pytest.approx(alpha, abs=0.006)

    def test_phase2_limit_holds_for_new_points(self) -> None:
        n, p, a, alpha = 80, 6, 3, 0.05
        exceed = []
        for seed in range(300):
            x = _gaussian(n + 200, p, seed)
            m = fit_pca_model(x[:n], _sensors(p), n_components=a, alpha=alpha)
            exceed.append(np.mean(score(m, x[n:]).t2 > m.t2_limit))
        assert np.mean(exceed) == pytest.approx(alpha, abs=0.008)

    def test_spe_limits_hold_and_agree(self) -> None:
        n, p, a, alpha = 400, 12, 4, 0.05
        exceed = []
        for seed in range(60):
            x = _gaussian(n + 400, p, seed)
            m = fit_pca_model(x[:n], _sensors(p), n_components=a, alpha=alpha)
            exceed.append(np.mean(score(m, x[n:]).spe > m.spe_limit))
        assert m.spe_limit_method == "jackson_mudholkar"
        assert np.mean(exceed) == pytest.approx(alpha, abs=0.012)
        resid = m.eigenvalues[a:]
        jm = spe_limit_jackson_mudholkar(resid, alpha)
        assert spe_limit_box(resid, alpha) == pytest.approx(jm, rel=0.05)

    def test_box_limit_for_equal_eigenvalues_is_chi_square(self) -> None:
        # equal residual eigenvalues lam: SPE = lam * chi2(k) exactly
        lam = np.full(5, 2.0)
        assert spe_limit_box(lam, 0.01) == pytest.approx(2.0 * sps.chi2.ppf(0.99, 5))


class TestFitAndScore:
    def test_contributions_sum_to_statistics(self) -> None:
        x = _gaussian(200, 8, 1)
        m = fit_pca_model(x[:150], _sensors(8), n_components=3)
        sc = score(m, x[150:])
        assert np.allclose(sc.t2_contrib.sum(axis=1), sc.t2)
        assert np.allclose(sc.spe_contrib.sum(axis=1), sc.spe)
        assert (sc.spe_contrib >= 0).all()

    def test_t2_matches_score_space_definition(self) -> None:
        x = _gaussian(120, 6, 2)
        m = fit_pca_model(x, _sensors(6), n_components=2)
        z = (x - m.mean) / m.scale
        t = z @ m.loadings
        expected = (t**2 / m.eigenvalues[:2]).sum(axis=1)
        assert np.allclose(score(m, x).t2, expected)

    def test_constant_sensor_is_dropped(self) -> None:
        x = _gaussian(50, 5, 3)
        x[:, 2] = 7.0
        m = fit_pca_model(x, _sensors(5), n_components=2)
        assert m.dropped_constant == ["S003"]
        assert "S003" not in m.sensors and len(m.sensors) == 4

    def test_variance_target_picks_components(self) -> None:
        x = _gaussian(300, 10, 4)
        m = fit_pca_model(x, _sensors(10), variance_target=0.8)
        assert m.explained >= 0.8
        assert m.n_components == n_components_for_variance(m.eigenvalues, 0.8)

    def test_rejects_nan_and_bad_shapes(self) -> None:
        x = _gaussian(30, 4, 5)
        with pytest.raises(ValueError, match="NaN"):
            bad = x.copy()
            bad[0, 0] = np.nan
            fit_pca_model(bad, _sensors(4), n_components=2)
        with pytest.raises(ValueError):
            fit_pca_model(x, _sensors(3), n_components=2)
        with pytest.raises(ValueError, match="out of range"):
            fit_pca_model(x, _sensors(4), n_components=4)
        m = fit_pca_model(x, _sensors(4), n_components=2)
        with pytest.raises(ValueError):
            score(m, x[:, :3])


class TestCrossValidation:
    def test_out_of_fold_spe_exceeds_in_sample(self) -> None:
        x = _gaussian(200, 15, 6)
        m = fit_pca_model(x, _sensors(15), n_components=4)
        _, oof_spe = cross_validated_scores(x, _sensors(15), n_components=4, n_folds=5)
        assert oof_spe.mean() > score(m, x).spe.mean()

    def test_a_row_is_scored_without_its_own_fold(self) -> None:
        # perturbing a fold-mate must not change a row's out-of-fold score
        x = _gaussian(100, 6, 7)
        t2, spe = cross_validated_scores(x, _sensors(6), n_components=2, n_folds=5)
        y = x.copy()
        y[5] += 50.0  # rows 0-19 are fold 0
        t2b, speb = cross_validated_scores(y, _sensors(6), n_components=2, n_folds=5)
        assert t2b[0] == pytest.approx(t2[0]) and speb[0] == pytest.approx(spe[0])
        assert not np.isclose(speb[30], spe[30])  # row 5 trained the other folds

    def test_bad_fold_count(self) -> None:
        with pytest.raises(ValueError):
            cross_validated_scores(np.ones((5, 2)), _sensors(2), n_components=1, n_folds=1)


def test_empirical_limit_is_the_upper_quantile() -> None:
    v = np.arange(101, dtype=float)
    assert empirical_limit(v, 0.01) == pytest.approx(99.0)
    with pytest.raises(ValueError):
        empirical_limit(np.array([]), 0.01)


def test_contribution_share_is_a_distribution() -> None:
    c = np.array([[1.0, 3.0, -2.0], [2.0, 2.0, 0.0], [5.0, 5.0, 5.0]])
    rows = np.array([True, True, False])
    share = contribution_share(c, rows)
    # row 0 -> [0.25, 0.75, 0], row 1 -> [0.5, 0.5, 0]
    assert np.allclose(share, [0.375, 0.625, 0.0])
    assert share.sum() == pytest.approx(1.0)
    assert np.allclose(contribution_share(c, np.zeros(3, bool)), 0.0)


@needs_data
def test_multivariate_stage_on_real_data(tmp_path: Path) -> None:
    root = Paths.from_root(tmp_path)
    paths = Paths(
        root=root.root,
        raw=RAW,
        processed=root.processed,
        powerbi=root.powerbi,
        figures=root.figures,
        reports=root.reports,
    )
    ctx = PipelineContext(config=load_config(None), paths=paths, log=lambda _: None)
    run_pipeline(ctx, only=["multivariate"], skip_figures=True)
    res = ctx.results["multivariate"]
    model = res["model"]
    assert model["n_sensors"] == 411 and model["dropped_constant_in_baseline"] == ["S075"]
    assert model["baseline_rank"] == 411
    assert model["n_components"] == 81
    nominal = res["nominal_alarm_rate"]["either_max"]
    theo = res["regimes"]["theoretical"]["phase_i_baseline"]["either_alarm_rate"]
    emp = res["regimes"]["empirical_cv"]["phase_i_baseline"]["either_alarm_rate"]
    # the textbook limits are badly miscalibrated on these heavy-tailed sensors;
    # the cross-validated ones are not
    assert theo > 5 * nominal
    assert emp <= nominal
    assert res["contributions"]["spe"]["S060_rank"] > 100
