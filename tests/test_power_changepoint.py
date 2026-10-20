"""Tests for secom.power and secom.changepoint, and their pipeline stages.

Simulated quantities are checked against exact values or an independent
implementation wherever one exists, rather than against their own output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import stats as sps

from secom.changepoint import (
    _block_permutations,
    binary_segmentation,
    bootstrap_location_ci,
    fit_single,
    lr_statistic,
    permutation_pvalue,
    profile_loglik,
    pvalue_from_null,
    segments,
)
from secom.config import config_from_dict
from secom.pipeline import Paths, PipelineContext, run_pipeline
from secom.power import (
    NO_SIGNAL,
    MonitorDesign,
    calibrate_h_for_false_alarm,
    cusum_first_signal,
    exact_shewhart_detection,
    first_signals,
    g_chart_first_signal,
    g_limits,
    shewhart_first_signal,
    side_for_ratio,
    simulate_detection,
    summarise_signals,
)
from secom.spc import bernoulli_cusum, calibrate_cusum_h, g_chart

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def _detect_rate(first: np.ndarray, horizon: int) -> float:
    return float(np.mean((first != NO_SIGNAL) & (first <= horizon)))


# power: Shewhart charts


class TestShewhart:
    SIZES = np.array([50, 120, 200])

    @pytest.mark.parametrize(
        ("ratio", "side"), [(1.0, "both"), (2.0, "degradation"), (3.0, "both")]
    )
    def test_simulation_matches_exact_binomial(self, ratio: float, side: str) -> None:
        p0, horizon, n_sim = 0.03, 1000, 20000
        rng = np.random.default_rng(1)
        fails = rng.random((n_sim, horizon)) < ratio * p0
        offsets = rng.integers(0, len(self.SIZES), n_sim)
        sim = _detect_rate(
            shewhart_first_signal(fails, self.SIZES, offsets, p0=p0, side=side), horizon
        )
        exact = exact_shewhart_detection(ratio * p0, self.SIZES, horizon, p0=p0, side=side)
        assert abs(sim - exact) < 4 * np.sqrt(exact * (1 - exact) / n_sim) + 1e-3

    def test_single_subgroup_is_one_binomial_tail(self) -> None:
        p0, n, p = 0.05, 100, 0.12
        ucl = p0 + 3 * np.sqrt(p0 * (1 - p0) / n)
        expected = sps.binom.sf(np.floor(n * ucl), n, p)  # LCL is 0 at this n
        assert exact_shewhart_detection(p, np.array([n]), n, p0=p0) == pytest.approx(expected)

    def test_cannot_detect_improvement_when_every_lcl_is_zero(self) -> None:
        # 9(1-p)/p = 291 > every subgroup size, so LCL = 0 throughout
        p0 = 0.03
        rng = np.random.default_rng(2)
        fails = rng.random((2000, 1200)) < 0.3 * p0
        offsets = rng.integers(0, len(self.SIZES), 2000)
        first = shewhart_first_signal(fails, self.SIZES, offsets, p0=p0, side="improvement")
        assert np.all(first == NO_SIGNAL)
        assert exact_shewhart_detection(0.3 * p0, self.SIZES, 1200, p0=p0, side="improvement") == 0

    def test_wider_limits_signal_less(self) -> None:
        exact = [
            exact_shewhart_detection(0.06, self.SIZES, 1000, p0=0.03, scale=s)
            for s in (1.0, 1.5, 2.0)
        ]
        assert exact[0] > exact[1] > exact[2]

    def test_signal_is_dated_to_the_end_of_the_subgroup(self) -> None:
        fails = np.zeros((1, 30), dtype=bool)
        fails[0, 12:20] = True  # 8 fails inside the second subgroup of 10
        first = shewhart_first_signal(fails, np.array([10]), np.array([0]), p0=0.05)
        assert first[0] == 20

    def test_bad_side_raises(self) -> None:
        with pytest.raises(ValueError, match="side"):
            exact_shewhart_detection(0.1, self.SIZES, 100, p0=0.05, side="up")


# power: g-chart


class TestGChart:
    def test_limits_match_spc_g_chart(self) -> None:
        for p in (0.03, 1e-4):
            ref = g_chart(np.array([0, 5]), n_total=10, baseline_p=p)
            assert g_limits(p) == (ref.lcl, ref.ucl)

    def test_signals_once_the_run_of_passes_exceeds_ucl(self) -> None:
        fails = np.zeros((1, 20), dtype=bool)
        fails[0, 2] = True
        # after the fail at run 3, passes 4..9 bring the count to 6 > 5 at run 9
        assert g_chart_first_signal(fails, lcl=0, ucl=5)[0] == 9

    def test_short_gap_signals_only_with_an_active_lcl(self) -> None:
        fails = np.zeros((1, 20), dtype=bool)
        fails[0, [5, 7]] = True
        assert g_chart_first_signal(fails, lcl=0, ucl=100)[0] == NO_SIGNAL
        assert g_chart_first_signal(fails, lcl=3, ucl=100)[0] == 8
        assert g_chart_first_signal(fails, lcl=3, ucl=100, side="improvement")[0] == NO_SIGNAL


# power: CUSUM


class TestCusumSimulation:
    def test_upper_matches_spc_bernoulli_cusum(self) -> None:
        rng = np.random.default_rng(3)
        fails = rng.random((40, 600)) < 0.06
        first = cusum_first_signal(fails, p0=0.03, p1=0.06, h=1.8)
        for row, f in zip(fails, first, strict=True):
            ref = bernoulli_cusum(row, p0=0.03, p1=0.06, h=1.8).first_signal_index
            assert f == (NO_SIGNAL if ref is None else ref + 1)

    def test_lower_matches_a_direct_loop(self) -> None:
        rng = np.random.default_rng(4)
        fails = rng.random((30, 800)) < 0.015
        p0, p1, h = 0.03, 0.015, 1.5
        first = cusum_first_signal(fails, p0=p0, p1=p1, h=h)
        for row, f in zip(fails, first, strict=True):
            acc, ref = 0.0, NO_SIGNAL
            for t, x in enumerate(row):
                acc = max(0.0, acc + (np.log(p1 / p0) if x else np.log((1 - p1) / (1 - p0))))
                if acc > h:
                    ref = t + 1
                    break
            assert f == ref

    def test_in_control_run_length_matches_the_calibrated_arl0(self) -> None:
        p0, p1, horizon = 0.03, 0.06, 4000
        h, arl0 = calibrate_cusum_h(p0, p1, target_arl0=200, n_sim=2000, max_run=horizon, seed=5)
        rng = np.random.default_rng(6)
        first = cusum_first_signal(rng.random((4000, horizon)) < p0, p0=p0, p1=p1, h=h)
        run_length = np.where(first == NO_SIGNAL, horizon, first)
        assert run_length.mean() == pytest.approx(arl0, rel=0.08)

    def test_false_alarm_calibration_hits_its_target(self) -> None:
        p0, p1, horizon, target = 0.03, 0.06, 500, 0.10
        h = calibrate_h_for_false_alarm(p0, p1, horizon=horizon, target=target, n_sim=6000, seed=7)
        rng = np.random.default_rng(8)
        first = cusum_first_signal(rng.random((6000, horizon)) < p0, p0=p0, p1=p1, h=h)
        assert _detect_rate(first, horizon) == pytest.approx(target, abs=4 * np.sqrt(0.09 / 6000))

    def test_false_alarm_target_must_be_a_probability(self) -> None:
        with pytest.raises(ValueError):
            calibrate_h_for_false_alarm(0.03, 0.06, horizon=10, target=1.0, n_sim=10, seed=0)


# power: assembly


def _design(**kw: float) -> MonitorDesign:
    base = dict(
        p0=0.03,
        subgroup_sizes=(50, 120, 200),
        sigma_z=1.5,
        sigma_z_phase1=2.0,
        cusum_upper_p1=0.06,
        cusum_upper_h=1.8,
        cusum_lower_p1=0.015,
        cusum_lower_h=1.5,
        g_lcl=0.0,
        g_ucl=212.0,
        cusum_upper_matched_h=4.0,
        cusum_lower_matched_h=4.0,
    )
    base.update(kw)
    return MonitorDesign(**base)  # type: ignore[arg-type]


class TestAssembly:
    def test_side_for_ratio(self) -> None:
        assert [side_for_ratio(r) for r in (0.5, 1.0, 2.0)] == [
            "improvement",
            "both",
            "degradation",
        ]

    def test_one_sided_schemes_never_detect_the_wrong_direction(self) -> None:
        rng = np.random.default_rng(9)
        fails = rng.random((500, 800)) < 0.01
        sig = first_signals(fails, rng.integers(0, 3, 500), _design(), side="improvement")
        assert np.all(sig["cusum_upper"] == NO_SIGNAL)
        assert np.all(sig["cusum_upper_matched"] == NO_SIGNAL)
        assert np.any(sig["cusum_lower"] != NO_SIGNAL)

    def test_matched_schemes_are_optional(self) -> None:
        d = _design(cusum_upper_matched_h=None, cusum_lower_matched_h=None)
        sig = first_signals(np.zeros((2, 50), dtype=bool), np.zeros(2, int), d)
        assert "cusum_upper_matched" not in sig and "cusum_lower_matched" not in sig

    def test_summarise_signals(self) -> None:
        row = summarise_signals(np.array([10, 20, NO_SIGNAL, 30]), (15, 25))
        assert row["p_detect_15"] == 0.25 and row["p_detect_25"] == 0.5
        assert row["median_run_length"] == 25.0
        assert row["mean_delay_given_detection"] == 20.0
        none = summarise_signals(np.array([5, NO_SIGNAL, NO_SIGNAL]), (10,))
        assert none["median_run_length"] is None

    def test_power_is_monotone_in_the_shift(self) -> None:
        # Common random numbers: more fails per stream can only bring an upper
        # CUSUM's signal forward, so detection probability is monotone exactly.
        table = simulate_detection(
            _design(), (1.5, 2.0, 3.0), horizons=(100, 400), n_sim=800, seed=10
        )
        cusum = table[table["scheme"] == "cusum_upper"].sort_values("ratio")
        assert cusum["p_detect_400"].is_monotonic_increasing
        assert set(table["side"]) == {"degradation"}


# changepoint


class TestProfileLikelihood:
    def test_matches_a_brute_force_fit(self) -> None:
        rng = np.random.default_rng(11)
        x = (rng.random(60) < 0.3).astype(float)

        def ll(seg: np.ndarray) -> float:
            p = seg.mean()
            return sum(np.log(p if v else 1 - p) for v in seg if 0 < p < 1)

        prof = profile_loglik(x, min_segment=5)[0]
        for k in range(5, 56):
            assert prof[k] == pytest.approx(ll(x[:k]) + ll(x[k:]))
        assert np.isnan(prof[:5]).all() and np.isnan(prof[56:]).all()

    def test_constant_sequence_has_no_evidence_of_change(self) -> None:
        assert lr_statistic(np.zeros(100), 10)[0] == pytest.approx(0.0)

    def test_too_short_raises(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            profile_loglik(np.zeros(10), min_segment=6)


class TestChangePoint:
    @staticmethod
    def _two_regimes(seed: int = 12) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return np.r_[rng.random(800) < 0.15, rng.random(1200) < 0.04].astype(float)

    def test_recovers_a_known_split_and_covers_it(self) -> None:
        x = self._two_regimes()
        fit = fit_single(x, 30)
        assert abs(fit.k - 800) < 60
        assert fit.rate_before == pytest.approx(x[: fit.k].mean())
        lo, hi, k_star = bootstrap_location_ci(
            fit, n_boot=400, rng=np.random.default_rng(13), min_segment=30
        )
        assert lo <= 800 <= hi and lo <= fit.k <= hi
        assert len(k_star) == 400

    def test_strong_change_gets_the_smallest_attainable_pvalue(self) -> None:
        p = permutation_pvalue(
            self._two_regimes(), min_segment=30, n_perm=199, rng=np.random.default_rng(14)
        )
        assert p == pytest.approx(1 / 200)

    def test_null_pvalues_are_not_too_small(self) -> None:
        rng = np.random.default_rng(15)
        ps = [
            permutation_pvalue(
                (rng.random(300) < 0.05).astype(float), min_segment=20, n_perm=99, rng=rng
            )
            for _ in range(40)
        ]
        assert np.mean(np.array(ps) < 0.10) <= 0.25

    def test_pvalue_from_null(self) -> None:
        assert pvalue_from_null(3.0, np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx(3 / 5)

    def test_block_permutations_keep_blocks_whole(self) -> None:
        blocks = np.repeat([0, 1, 2, 3], [3, 1, 4, 2])
        perms = _block_permutations(blocks, 25, np.random.default_rng(16))
        for row in perms:
            assert sorted(row) == list(range(10))
            relabelled = blocks[row]
            changes = np.flatnonzero(relabelled[1:] != relabelled[:-1])
            assert len(changes) == 3  # each block still contiguous

    def test_block_permutation_is_more_conservative_under_clustering(self) -> None:
        # Fails clustered by day with no trend: run-level permutation should be
        # fooled more often than day-block permutation.
        rng = np.random.default_rng(17)
        blocks = np.repeat(np.arange(40), 25)
        plain, block = [], []
        for _ in range(15):
            day_rate = rng.choice([0.01, 0.2], size=40, p=[0.8, 0.2])
            x = (rng.random(1000) < day_rate[blocks]).astype(float)
            kw = dict(min_segment=50, n_perm=99, rng=rng)
            plain.append(permutation_pvalue(x, **kw))
            block.append(permutation_pvalue(x, blocks=blocks, **kw))
        assert np.mean(np.array(plain) < 0.05) > np.mean(np.array(block) < 0.05)

    def test_binary_segmentation_finds_two_changes_and_stops(self) -> None:
        rng = np.random.default_rng(18)
        x = np.r_[rng.random(400) < 0.25, rng.random(400) < 0.02, rng.random(400) < 0.25]
        blocks = np.repeat(np.arange(60), 20)
        cps, tests = binary_segmentation(
            x.astype(float), blocks, min_segment=30, alpha=0.05, n_perm=199, rng=rng
        )
        assert len(cps) == 2
        assert abs(cps[0] - 400) < 50 and abs(cps[1] - 800) < 50
        assert sum(t.accepted for t in tests) == 2 and len(tests) > 2

    def test_binary_segmentation_on_a_stable_sequence_finds_nothing(self) -> None:
        rng = np.random.default_rng(19)
        x = (rng.random(1000) < 0.05).astype(float)
        cps, tests = binary_segmentation(
            x, np.repeat(np.arange(50), 20), min_segment=30, alpha=0.01, n_perm=199, rng=rng
        )
        assert cps == [] and len(tests) == 1

    def test_segments(self) -> None:
        x = np.array([1, 0, 0, 1, 1, 1], dtype=float)
        assert segments(x, [3]) == [(0, 3, 1, 1 / 3), (3, 6, 3, 1.0)]


# pipeline stages on the real data


def _ctx(tmp_path: Path, data: dict | None = None) -> PipelineContext:
    p = Paths.from_root(tmp_path)
    paths = Paths(
        root=p.root,
        raw=RAW,
        processed=p.processed,
        powerbi=p.powerbi,
        figures=p.figures,
        reports=p.reports,
    )
    return PipelineContext(config=config_from_dict(data), paths=paths, log=lambda _: None)


needs_data = pytest.mark.skipif(not (RAW / "secom.data").exists(), reason="needs data/raw/")


@needs_data
class TestRealData:
    def test_changepoint_lands_in_week_34_and_agrees_with_the_baseline(
        self, tmp_path: Path
    ) -> None:
        ctx = _ctx(tmp_path, {"changepoint": {"n_perm": 199, "n_boot": 200}})
        run_pipeline(ctx, only=["changepoint"], skip_figures=True)
        cp = ctx.results["changepoint"]
        assert cp["estimate"]["first_run_after"]["run"] == 375
        assert cp["estimate"]["first_run_after"]["year_week"] == "2008-W34"
        assert cp["baseline_boundary"]["first_run"]["run"] == 450
        assert cp["agrees_with_baseline"] is True
        assert cp["segmentation"]["change_first_runs"] == [375]

    def test_simulation_reproduces_the_structural_blind_spot(self, tmp_path: Path) -> None:
        ctx = _ctx(tmp_path, {"shift_simulation": {"n_sim": 600}})
        run_pipeline(ctx, only=["shift_simulation"], skip_figures=True)
        ss = ctx.results["shift_simulation"]
        assert ss["design"]["weeks_with_lcl_zero_at_p0"] == 14
        assert ss["headline"]["shewhart_improvement_max_p_detect"] == 0.0
        two = ss["headline"]["degradation_2x"]
        assert two["cusum_upper"]["p_detect_400"] > two["p_chart"]["p_detect_400"]
        assert ss["exact_check"]["max_abs_diff"] < 4 * ss["exact_check"]["mc_se_max"]
