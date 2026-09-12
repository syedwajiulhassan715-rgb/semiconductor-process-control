"""Tests for the control-chart engine.

Where a published reference value exists, the test asserts against that value
rather than against whatever the code currently returns. A test that only
asserts self-consistency cannot catch a wrong constant.
"""

from __future__ import annotations

import numpy as np
import pytest

from secom.spc import (
    D2,
    ControlLimits,
    bernoulli_cusum,
    c4,
    calibrate_cusum_h,
    ewma,
    g_chart,
    individuals_limits,
    moving_range_sigma,
    nelson_rules,
    p_chart,
)

# c4(n) as published in Montgomery, *Introduction to Statistical Quality
# Control*, Appendix VI, and ASTM E2587.
C4_PUBLISHED = {
    2: 0.7979, 3: 0.8862, 4: 0.9213, 5: 0.9400, 6: 0.9515,
    7: 0.9594, 8: 0.9650, 9: 0.9693, 10: 0.9727, 25: 0.9896,
}


@pytest.mark.parametrize(("n", "expected"), sorted(C4_PUBLISHED.items()))
def test_c4_matches_published_table(n: int, expected: float) -> None:
    assert c4(n) == pytest.approx(expected, abs=5e-5)


def test_d2_for_subgroup_two_is_exact() -> None:
    # d2(2) has the closed form 2/sqrt(pi); the others are integrals.
    assert D2[2] == pytest.approx(2 / np.sqrt(np.pi), abs=1e-9)


def test_d2_table_is_increasing() -> None:
    values = [D2[n] for n in sorted(D2)]
    assert values == sorted(values)


class TestMovingRangeSigma:
    def test_is_unbiased_for_normal_data(self) -> None:
        est = [
            moving_range_sigma(np.random.default_rng(s).normal(100, 5, 400))
            for s in range(200)
        ]
        # Unbiased to within Monte Carlo error of the mean over 200 replicates.
        assert np.mean(est) == pytest.approx(5.0, rel=0.02)

    def test_is_robust_to_a_sustained_shift(self) -> None:
        """The property that makes MR-based sigma the right Phase I estimator.

        A step change inflates the overall standard deviation but barely touches
        the average moving range, because only one of the n-1 ranges spans the
        step.
        """
        rng = np.random.default_rng(0)
        x = np.concatenate([rng.normal(100, 5, 300), rng.normal(120, 5, 300)])
        assert moving_range_sigma(x) == pytest.approx(5.0, rel=0.10)
        assert x.std(ddof=1) > 10  # overall SD is more than double the truth

    def test_requires_two_points(self) -> None:
        assert np.isnan(moving_range_sigma(np.array([1.0])))


class TestIndividualsLimits:
    def test_recovers_centre_and_spread(self) -> None:
        x = np.random.default_rng(1).normal(50, 2, 500)
        lim = individuals_limits(x)
        assert lim.center == pytest.approx(50, abs=0.3)
        assert lim.sigma == pytest.approx(2, rel=0.10)
        assert lim.ucl - lim.lcl == pytest.approx(6 * lim.sigma)

    def test_excludes_assignable_cause_points(self) -> None:
        x = np.concatenate([np.random.default_rng(2).normal(0, 1, 200), [40.0, -35.0]])
        lim = individuals_limits(x)
        assert lim.n_excluded >= 2
        assert abs(lim.center) < 0.5  # outliers did not drag the centre

    def test_trimming_bias_is_small_and_conservative(self) -> None:
        """Phase I trimming biases sigma slightly low, tightening limits.

        Quantified rather than ignored: the bias must stay within a few percent,
        otherwise the chart would over-alarm by construction.
        """
        untrimmed, trimmed = [], []
        for s in range(150):
            x = np.random.default_rng(s).normal(0, 3, 400)
            untrimmed.append(moving_range_sigma(x))
            trimmed.append(individuals_limits(x).sigma)
        ratio = np.mean(trimmed) / np.mean(untrimmed)
        assert 0.95 < ratio <= 1.0

    def test_insufficient_data_is_reported_not_guessed(self) -> None:
        lim = individuals_limits(np.arange(5.0))
        assert lim.source == "insufficient"
        assert np.isnan(lim.sigma)


class TestNelsonRules:
    LIMITS = ControlLimits(center=0.0, sigma=1.0, lcl=-3.0, ucl=3.0, n_baseline=100)

    def _fired(self, seq: list[float]) -> set[int]:
        res = nelson_rules(np.array(seq, dtype=float), self.LIMITS)
        return {int(c) for c in res.columns if res[c].any()}

    def test_rule_1_beyond_three_sigma(self) -> None:
        assert 1 in self._fired([0, 0, 4.0])
        assert 1 not in self._fired([0, 0, 2.9])

    def test_rule_2_nine_on_one_side(self) -> None:
        assert 2 in self._fired([0.5] * 9)
        assert 2 not in self._fired([0.5] * 8)

    def test_rule_3_six_increasing(self) -> None:
        assert 3 in self._fired(list(np.linspace(-0.4, 0.4, 6)))
        assert 3 not in self._fired(list(np.linspace(-0.4, 0.4, 5)))

    def test_rule_4_fourteen_alternating(self) -> None:
        assert 4 in self._fired([0.1, -0.1] * 7)

    def test_rule_5_two_of_three_beyond_two_sigma(self) -> None:
        assert 5 in self._fired([2.5, 0.1, 2.6])
        assert 5 not in self._fired([2.5, 0.1, 0.2])

    def test_rule_6_four_of_five_beyond_one_sigma(self) -> None:
        assert 6 in self._fired([1.5, 1.6, 1.7, 1.8, 0.1])

    def test_rule_7_stratification(self) -> None:
        assert 7 in self._fired([0.1, -0.1] * 8)

    def test_rule_8_mixture(self) -> None:
        assert 8 in self._fired([1.5, -1.5] * 4)

    def test_rule_5_is_side_specific(self) -> None:
        """Two points beyond 2 sigma on *opposite* sides must not fire rule 5."""
        assert 5 not in self._fired([2.5, 0.1, -2.6])

    def test_degenerate_sigma_fires_nothing(self) -> None:
        bad = ControlLimits(0.0, 0.0, 0.0, 0.0, 10)
        res = nelson_rules(np.arange(50.0), bad)
        assert not res.to_numpy().any()


class TestEwma:
    def test_limits_widen_to_the_asymptote(self) -> None:
        lim = ControlLimits(center=0.0, sigma=1.0, lcl=-3, ucl=3, n_baseline=100)
        out = ewma(np.zeros(200), lim, lam=0.2, L=2.7)
        width = out["ucl"] - out["center"]
        assert width.iloc[0] < width.iloc[-1]          # time-varying, not flat
        asymptote = 2.7 * np.sqrt(0.2 / 1.8)
        assert width.iloc[-1] == pytest.approx(asymptote, rel=1e-6)

    def test_detects_a_small_sustained_shift(self) -> None:
        rng = np.random.default_rng(3)
        lim = ControlLimits(center=0.0, sigma=1.0, lcl=-3, ucl=3, n_baseline=100)
        x = np.concatenate([rng.normal(0, 1, 100), rng.normal(1.0, 1, 100)])
        out = ewma(x, lim)
        assert not out["violation"].iloc[:100].any()
        assert out["violation"].iloc[100:].any()


class TestPChart:
    def test_limit_formula(self) -> None:
        d = np.array([5.0, 6, 4, 5, 6, 5, 4, 6, 5, 5])
        n = np.full(10, 100.0)
        res = p_chart(d, n, force_type="p")
        expected = res.p_bar + 3 * np.sqrt(res.p_bar * (1 - res.p_bar) / 100)
        assert res.table["ucl"].iloc[0] == pytest.approx(expected)

    def test_limits_are_clipped_to_valid_proportions(self) -> None:
        res = p_chart(np.array([1.0, 0, 1]), np.array([5.0, 5, 5]))
        assert (res.table["lcl"] >= 0).all()
        assert (res.table["ucl"] <= 1).all()

    def test_limits_vary_with_subgroup_size(self) -> None:
        res = p_chart(np.array([5.0, 50.0]), np.array([100.0, 1000.0]), force_type="p")
        # The smaller subgroup must get the wider limits.
        assert res.table["ucl"].iloc[0] > res.table["ucl"].iloc[1]

    def test_sigma_z_is_about_one_for_binomial_data(self) -> None:
        rng = np.random.default_rng(11)
        d = rng.binomial(200, 0.05, 60).astype(float)
        res = p_chart(d, np.full(60, 200.0))
        assert 0.75 < res.sigma_z < 1.25
        assert not res.overdispersed
        assert res.chart_type == "p"

    def test_detects_overdispersion(self) -> None:
        rng = np.random.default_rng(12)
        p = rng.beta(2, 38, 60)
        d = np.array([rng.binomial(200, pi) for pi in p], dtype=float)
        res = p_chart(d, np.full(60, 200.0))
        assert res.overdispersed
        assert res.chart_type == "laney_p_prime"
        assert res.dispersion_p < 0.01

    def test_laney_limits_are_wider_than_naive(self) -> None:
        rng = np.random.default_rng(13)
        d = np.array([rng.binomial(200, pi) for pi in rng.beta(2, 38, 40)], dtype=float)
        n = np.full(40, 200.0)
        assert (p_chart(d, n).table["ucl"] >= p_chart(d, n, force_type="p").table["ucl"]).all()

    def test_baseline_and_dispersion_subsets_are_independent(self) -> None:
        """Centre line from Phase I, dispersion from everything -- see docstring."""
        rng = np.random.default_rng(14)
        d = np.concatenate([rng.binomial(200, 0.02, 20), rng.binomial(200, 0.20, 20)]).astype(float)
        n = np.full(40, 200.0)
        base = np.r_[np.ones(20, bool), np.zeros(20, bool)]
        res = p_chart(d, n, baseline=base)
        assert res.p_bar == pytest.approx(d[:20].sum() / n[:20].sum())
        # sigma_z used all 40 subgroups, so it differs from the baseline-only value
        assert res.sigma_z != pytest.approx(p_chart(d, n, baseline=base, dispersion_subset=base).sigma_z)

    def test_rejects_mismatched_inputs(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            p_chart(np.array([1.0]), np.array([10.0, 10.0]))
        with pytest.raises(ValueError, match="positive"):
            p_chart(np.array([1.0]), np.array([0.0]))


class TestGChart:
    def test_centre_matches_geometric_mean(self) -> None:
        rng = np.random.default_rng(15)
        x = rng.random(100000) < 0.05
        res = g_chart(np.flatnonzero(x), n_total=len(x))
        assert res.center == pytest.approx((1 - 0.05) / 0.05, rel=0.05)

    def test_false_alarm_rate_does_not_exceed_alpha(self) -> None:
        """Discreteness makes exact geometric limits conservative, never liberal."""
        rng = np.random.default_rng(16)
        x = rng.random(200000) < 0.05
        res = g_chart(np.flatnonzero(x), n_total=len(x), alpha=0.0027)
        assert res.table["violation"].mean() <= 0.0027

    def test_lower_limit_is_inactive_at_realistic_failure_rates(self) -> None:
        """Documented limitation, asserted so it cannot be quietly forgotten.

        P(gap = 0) equals p exactly, so once p exceeds alpha/2 the lower
        quantile pins to zero and no single gap can breach it. At SECOM's 3.1%
        baseline the g-chart is an improvement detector only; degradation is the
        CUSUM's job.
        """
        res = g_chart(np.arange(0, 200, 2), n_total=1000, baseline_p=0.031)
        assert res.lcl == 0
        assert not res.table["below_lcl"].any()
        assert any("LOWER LIMIT INACTIVE" in n for n in res.notes)

    def test_lower_limit_becomes_active_at_very_low_rates(self) -> None:
        res = g_chart(np.array([0, 1200, 2400]), n_total=3000, baseline_p=0.0005)
        assert res.lcl > 0
        assert not any("LOWER LIMIT INACTIVE" in n for n in res.notes)

    def test_upper_limit_detects_sustained_improvement(self) -> None:
        """The capability the p-chart structurally lacks at this failure rate."""
        res = g_chart(np.array([0, 900, 1800]), n_total=2000, baseline_p=0.0664)
        assert res.table["above_ucl"].all()

    def test_requires_two_failures(self) -> None:
        with pytest.raises(ValueError, match="two failures"):
            g_chart(np.array([5]), n_total=100)

    def test_rejects_unsorted_positions(self) -> None:
        with pytest.raises(ValueError, match="ascending"):
            g_chart(np.array([10, 3, 20]), n_total=100)


class TestBernoulliCusum:
    def test_weights_have_the_right_signs(self) -> None:
        res = bernoulli_cusum(np.array([True, False]), p0=0.03, p1=0.06, h=2.0)
        assert res.table["increment"].iloc[0] > 0   # a fail pushes up
        assert res.table["increment"].iloc[1] < 0   # a pass pays down

    def test_statistic_is_non_negative(self) -> None:
        rng = np.random.default_rng(17)
        res = bernoulli_cusum(rng.random(2000) < 0.01, p0=0.03, p1=0.06, h=2.0)
        assert (res.table["cusum"] >= 0).all()

    def test_reset_prevents_a_permanent_alarm(self) -> None:
        """Without reset, one big excursion leaves the chart alarming forever.

        This is the defect that made the un-reset chart useless on SECOM: the
        statistic integrates to ~28 and each good run pays down only ~0.03.
        """
        x = np.concatenate([np.ones(120, bool), np.zeros(900, bool)])
        no_reset = bernoulli_cusum(x, p0=0.03, p1=0.06, h=2.0, reset_on_signal=False)
        with_reset = bernoulli_cusum(x, p0=0.03, p1=0.06, h=2.0, reset_on_signal=True)
        quiet = slice(200, 900)
        assert no_reset.table["signal"].iloc[quiet].all()
        assert not with_reset.table["signal"].iloc[quiet].any()

    def test_detects_a_sustained_shift_a_pchart_would_miss(self) -> None:
        rng = np.random.default_rng(18)
        h, _ = calibrate_cusum_h(0.03, 0.06, target_arl0=500, n_sim=800, max_run=4000)
        shifted = bernoulli_cusum(rng.random(600) < 0.09, p0=0.03, p1=0.06, h=h)
        assert shifted.first_signal_index is not None
        assert shifted.first_signal_index < 200

    def test_rejects_a_downward_design(self) -> None:
        with pytest.raises(ValueError, match="p1 must exceed p0"):
            bernoulli_cusum(np.array([True]), p0=0.09, p1=0.03, h=2.0)

    def test_calibration_hits_the_target_arl(self) -> None:
        h, arl = calibrate_cusum_h(0.03, 0.06, target_arl0=400, n_sim=1500, max_run=6000)
        assert arl == pytest.approx(400, rel=0.15)
        assert h > 0

    def test_higher_threshold_gives_longer_arl(self) -> None:
        h_low, arl_low = calibrate_cusum_h(0.03, 0.06, target_arl0=200, n_sim=800, max_run=4000)
        h_high, arl_high = calibrate_cusum_h(0.03, 0.06, target_arl0=800, n_sim=800, max_run=4000)
        assert h_high > h_low
        assert arl_high > arl_low
