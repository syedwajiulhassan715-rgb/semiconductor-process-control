"""Tests for secom.tep and secom.tep_validation.

Synthetic tests run everywhere. The real-data tests need the Tennessee Eastman
files (``secom fetch --dataset tep``) and assert the qualitative facts of the
published PCA benchmark rather than exact numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import stats as sps

from secom import tep
from secom.cli import main
from secom.mspc import fit_pca_model, score
from secom.tep_validation import (
    TEPValidationConfig,
    alarms,
    benchmark_limits,
    fit_monitors,
    lag1_autocorrelation,
    rank_threshold,
    run_tep_validation,
)

ROOT = Path(__file__).resolve().parents[1]
TEP_DIR = ROOT / "data" / "raw" / "tep"

needs_tep = pytest.mark.skipif(
    not (TEP_DIR / "d21_te.dat").exists(),
    reason="run `secom fetch --dataset tep` to enable Tennessee Eastman tests",
)


# metrics


class TestDetectionMetrics:
    def test_rates_split_at_the_onset(self) -> None:
        a = np.zeros(10, bool)
        a[[1, 6, 7]] = True
        assert tep.false_alarm_rate(a, onset=5) == pytest.approx(0.2)
        assert tep.detection_rate(a, onset=5) == pytest.approx(0.4)

    def test_first_alarm_delay(self) -> None:
        a = np.zeros(20, bool)
        a[[2, 13]] = True
        assert tep.detection_delay(a, onset=10) == 3

    def test_persistent_delay_skips_isolated_alarms(self) -> None:
        a = np.zeros(30, bool)
        a[11] = True  # isolated
        a[15:21] = True  # six in a row
        assert tep.detection_delay(a, onset=10, run_length=1) == 1
        assert tep.detection_delay(a, onset=10, run_length=6) == 5

    def test_no_detection_returns_none(self) -> None:
        a = np.zeros(20, bool)
        a[3] = True  # before the onset only
        assert tep.detection_delay(a, onset=10) is None
        assert tep.detection_delay(np.ones(12, bool), onset=10, run_length=6) is None

    def test_run_length_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            tep.detection_delay(np.ones(5, bool), onset=0, run_length=0)


class TestSidak:
    def test_single_chart_is_the_plain_normal_quantile(self) -> None:
        assert tep.sidak_k(0.01, 1) == pytest.approx(sps.norm.ppf(0.995))

    def test_family_rate_is_exact_for_independent_charts(self) -> None:
        k = tep.sidak_k(0.01, 52)
        per_chart = 2 * sps.norm.sf(k)
        assert 1 - (1 - per_chart) ** 52 == pytest.approx(0.01, rel=1e-9)

    def test_simulated_family_false_alarm_rate(self) -> None:
        rng = np.random.default_rng(3)
        z = rng.standard_normal((200_000, 10))
        far = (np.abs(z) > tep.sidak_k(0.05, 10)).any(axis=1).mean()
        assert far == pytest.approx(0.05, abs=0.003)

    @pytest.mark.parametrize("alpha,m", [(0.0, 5), (1.0, 5), (0.01, 0)])
    def test_rejects_bad_arguments(self, alpha: float, m: int) -> None:
        with pytest.raises(ValueError):
            tep.sidak_k(alpha, m)


def test_lag1_autocorrelation_of_an_ar1_process() -> None:
    rng = np.random.default_rng(5)
    phi, n = 0.8, 20_000
    x = np.empty(n)
    x[0] = 0.0
    e = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    white = rng.standard_normal(n)
    rho = lag1_autocorrelation(np.column_stack([x, white]))
    assert rho[0] == pytest.approx(phi, abs=0.02)
    assert abs(rho[1]) < 0.03


# loader


def _write(path: Path, shape: tuple[int, int], seed: int = 0) -> None:
    np.savetxt(path, np.random.default_rng(seed).standard_normal(shape))


class TestLoader:
    def _fake_dir(self, tmp_path: Path) -> Path:
        _write(tmp_path / "d00.dat", (tep.N_VARS, tep.N_TRAIN))
        for k in range(22):
            _write(tmp_path / f"d{k:02d}_te.dat", (tep.N_TEST, tep.N_VARS), seed=k + 1)
        return tmp_path

    def test_loads_and_transposes_the_training_file(self, tmp_path: Path) -> None:
        data = tep.load_tep(self._fake_dir(tmp_path))
        assert data.train.shape == (tep.N_TRAIN, tep.N_VARS)
        assert sorted(data.test) == list(range(22))
        assert all(x.shape == (tep.N_TEST, tep.N_VARS) for x in data.test.values())
        assert len(data.variables) == tep.N_VARS

    def test_untransposed_training_file_is_rejected(self, tmp_path: Path) -> None:
        d = self._fake_dir(tmp_path)
        _write(d / "d00.dat", (tep.N_TRAIN, tep.N_VARS))
        with pytest.raises(ValueError, match="d00.dat"):
            tep.load_tep(d)

    def test_wrong_test_shape_is_rejected(self, tmp_path: Path) -> None:
        d = self._fake_dir(tmp_path)
        _write(d / "d07_te.dat", (480, tep.N_VARS))
        with pytest.raises(ValueError, match="d07_te.dat"):
            tep.load_tep(d)

    def test_missing_files_point_to_the_fetch_command(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="--dataset tep"):
            tep.load_tep(tmp_path)


def test_benchmark_table_covers_every_fault() -> None:
    assert sorted(tep.BENCHMARK_PCA_MISSED) == list(tep.FAULTS)
    assert sorted(tep.FAULT_DESCRIPTIONS) == list(tep.FAULTS)
    for t2, q in tep.BENCHMARK_PCA_MISSED.values():
        assert 0 <= t2 <= 1 and 0 <= q <= 1
    assert len(tep.VARIABLES) == tep.N_VARS


@pytest.mark.parametrize(
    ("fault", "t2", "q"),
    [
        (1, 0.008, 0.003),
        (4, 0.956, 0.038),
        (9, 0.994, 0.981),
        (11, 0.794, 0.356),
        (21, 0.736, 0.570),
    ],
)
def test_benchmark_matches_the_published_table(fault: int, t2: float, q: float) -> None:
    # Russell, Chiang & Braatz (2000), Table 4. Fault 9 once carried its Q value
    # in the T² column, so it is pinned here.
    assert tep.BENCHMARK_PCA_MISSED[fault] == (t2, q)


def test_benchmark_false_alarms_match_the_published_table() -> None:
    # Russell, Chiang & Braatz (2000), Table 3, PCA rows.
    assert tep.BENCHMARK_PCA_FALSE_ALARM == {"training": (0.002, 0.004), "testing": (0.014, 0.016)}


# PCA limits on data that meets their assumptions


def test_textbook_limits_hold_on_gaussian_factor_data() -> None:
    """New observations from the training distribution alarm at about alpha.

    This is the property TEP is used to check on real data; here it is
    checked where the assumptions hold exactly.
    """
    rng = np.random.default_rng(11)
    p, a, n = 30, 4, 2000
    w = rng.standard_normal((p, a)) * 2

    def draw(m: int) -> np.ndarray:
        return rng.standard_normal((m, a)) @ w.T + rng.standard_normal((m, p))

    names = [f"v{i}" for i in range(p)]
    model = fit_pca_model(draw(n), names, n_components=a, alpha=0.01)
    sc = score(model, draw(40_000))
    assert (sc.t2 > model.t2_limit).mean() == pytest.approx(0.01, abs=0.004)
    assert (sc.spe > model.spe_limit).mean() == pytest.approx(0.01, abs=0.004)


def test_monitors_flag_a_mean_shift() -> None:
    rng = np.random.default_rng(2)
    train = rng.standard_normal((500, tep.N_VARS))
    cfg = TEPValidationConfig(n_components=5)
    mon = fit_monitors(train, cfg)
    shifted = rng.standard_normal((300, tep.N_VARS))
    shifted[:, 0] += 8.0
    bench = benchmark_limits(mon, rng.standard_normal((960, tep.N_VARS)))
    for lim in ("textbook", "cv", "benchmark"):
        a = alarms(mon, shifted, limits=lim, bench=bench)
        assert a["univariate"].mean() > 0.95
        assert (a["t2"] | a["spe"]).mean() > 0.95
    with pytest.raises(ValueError):
        alarms(mon, shifted, limits="nonsense")
    with pytest.raises(ValueError, match="bench"):
        alarms(mon, shifted, limits="benchmark")


def test_rank_threshold_is_the_published_rule() -> None:
    # "adjusted to the tenth highest value": 9 of 960 normal samples exceed it,
    # an empirical alpha just under 0.01
    x = np.random.default_rng(3).permutation(np.arange(960.0))
    thr = rank_threshold(x)
    assert thr == 950.0
    assert (x > thr).sum() == 9
    with pytest.raises(ValueError):
        rank_threshold(x, rank=0)


def test_cli_reports_missing_tep_data(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", "tep", "--root", str(tmp_path), "--no-secom", "--no-figures"]) == 2
    assert "--dataset tep" in capsys.readouterr().err


# real data


@pytest.fixture(scope="module")
def tep_results() -> dict:
    results, _ = run_tep_validation(ROOT, with_secom=False, log=lambda _m: None)
    return results


@needs_tep
class TestRealTEP:
    def _det(self, results: dict, fault: int, chart: str) -> float:
        return results["faults"][fault - 1]["textbook"][chart]["detection_rate"]

    @pytest.mark.parametrize("fault", [1, 2, 6, 7, 8, 12, 13, 14, 18])
    def test_easy_faults_are_detected(self, tep_results: dict, fault: int) -> None:
        assert self._det(tep_results, fault, "spe") > 0.85

    def test_fault_1_t2_detection_matches_the_benchmark(self, tep_results: dict) -> None:
        assert self._det(tep_results, 1, "t2") > 0.98

    @pytest.mark.parametrize("fault", tep.UNDETECTABLE_BY_PCA)
    def test_known_undetectable_faults_stay_near_the_false_alarm_rate(
        self, tep_results: dict, fault: int
    ) -> None:
        normal = tep_results["false_alarms"]["d00_te"]["textbook"]
        for chart in ("t2", "spe"):
            assert self._det(tep_results, fault, chart) < max(0.15, 2.5 * normal[chart])

    def test_fault_4_is_an_spe_fault(self, tep_results: dict) -> None:
        # reactor cooling water step: invisible to T², obvious in the residual
        assert self._det(tep_results, 4, "t2") < 0.2
        assert self._det(tep_results, 4, "spe") > 0.95

    def test_textbook_t2_limit_is_calibrated_on_new_normal_data(self, tep_results: dict) -> None:
        far = tep_results["false_alarms"]["d00_te"]["textbook"]["t2"]
        assert 0.003 < far < 0.03

    def test_textbook_limits_hold_in_sample(self, tep_results: dict) -> None:
        ins = tep_results["calibration"]["tep"]["in_sample"]
        assert ins["t2"] <= 0.02 and ins["spe"] <= 0.02

    def test_detection_reproduces_the_published_table(self, tep_results: dict) -> None:
        # Same threshold rule as Russell et al. (2000), so every fault should
        # land close to Table 4, not just correlate with it.
        cmp_ = tep_results["benchmark_comparison"]
        for chart in ("t2", "spe"):
            assert cmp_[chart]["mean_abs_diff"] < 0.02
            assert cmp_[chart]["max_abs_diff"] < 0.05
            assert cmp_[chart]["n_within_0_05"] == len(tep.FAULTS)

    def test_benchmark_rule_sets_an_empirical_one_percent(self, tep_results: dict) -> None:
        far = tep_results["false_alarms"]["d00_te"]["benchmark"]
        assert far["t2"] == far["spe"] == pytest.approx(9 / tep.N_TEST)

    def test_moving_range_sigma_breaks_on_autocorrelated_variables(self, tep_results: dict) -> None:
        uni = tep_results["calibration"]["tep_univariate"]
        assert uni["d00_te_family_far_mr_sigma"] > 0.5
        assert uni["d00_te_family_far_sd_sigma"] < 0.1
        assert uni["spearman_mr_ratio_vs_lag1"] < -0.8
