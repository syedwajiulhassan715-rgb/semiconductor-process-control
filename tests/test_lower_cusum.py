"""Tests for the lower (improvement-detecting) Bernoulli CUSUM in secom.spc."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from secom.config import config_from_dict
from secom.pipeline import Paths, PipelineContext, run_pipeline
from secom.power import NO_SIGNAL, cusum_first_signal
from secom.spc import bernoulli_cusum, bernoulli_cusum_lower, calibrate_cusum_h

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def _page_reference(
    x: np.ndarray, p0: float, p1: float, h: float, reset: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force CUSUM from Page's definition, not the recursion.

    Since the last reset, S_t = max(0, max over k <= t of the sum of the
    log-likelihood ratios of runs k..t). Every partial sum is recomputed at
    every step, so this shares no code path with the recursive form.
    """
    llr = np.where(x, np.log(p1 / p0), np.log((1 - p1) / (1 - p0)))
    s = np.zeros(len(x))
    sig = np.zeros(len(x), dtype=bool)
    start = 0
    for t in range(len(x)):
        seg = llr[start : t + 1]
        tails = np.cumsum(seg[::-1])  # sums of runs k..t for k = t, t-1, ..., start
        s[t] = max(0.0, float(tails.max()))
        if s[t] > h:
            sig[t] = True
            if reset:
                start = t + 1
    return s, sig


class TestLowerCusum:
    def test_weights_are_the_mirror_of_the_upper_chart(self) -> None:
        res = bernoulli_cusum_lower(np.array([True, False]), p0=0.03, p1=0.015, h=2.0)
        assert res.table["increment"].iloc[0] == pytest.approx(np.log(0.5))  # a fail pays down
        assert res.table["increment"].iloc[1] > 0  # a pass builds evidence
        assert res.side == "lower"
        assert bernoulli_cusum(np.array([True]), p0=0.03, p1=0.06, h=2.0).side == "upper"

    def test_rejects_an_upward_design(self) -> None:
        with pytest.raises(ValueError, match="p1 must be below p0"):
            bernoulli_cusum_lower(np.array([True]), p0=0.03, p1=0.06, h=2.0)
        with pytest.raises(ValueError, match="strictly"):
            bernoulli_cusum_lower(np.array([True]), p0=0.03, p1=0.0, h=2.0)

    @pytest.mark.parametrize("reset", [True, False])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_matches_the_brute_force_page_definition(self, seed: int, reset: bool) -> None:
        rng = np.random.default_rng(seed)
        x = rng.random(700) < 0.012
        p0, p1, h = 0.03, 0.015, 1.2
        res = bernoulli_cusum_lower(x, p0=p0, p1=p1, h=h, reset_on_signal=reset)
        s, sig = _page_reference(x, p0, p1, h, reset)
        np.testing.assert_allclose(res.table["cusum"].to_numpy(), s, atol=1e-9)
        np.testing.assert_array_equal(res.table["signal"].to_numpy(), sig)
        assert sig.any()  # the check is not vacuous

    def test_upper_chart_still_matches_the_brute_force_definition(self) -> None:
        rng = np.random.default_rng(9)
        x = rng.random(600) < 0.07
        res = bernoulli_cusum(x, p0=0.03, p1=0.06, h=1.8)
        s, sig = _page_reference(x, 0.03, 0.06, 1.8, True)
        np.testing.assert_allclose(res.table["cusum"].to_numpy(), s, atol=1e-9)
        np.testing.assert_array_equal(res.table["signal"].to_numpy(), sig)

    def test_matches_the_vectorised_simulation_engine(self) -> None:
        rng = np.random.default_rng(10)
        fails = rng.random((40, 900)) < 0.015
        first = cusum_first_signal(fails, p0=0.03, p1=0.015, h=1.5)
        for row, f in zip(fails, first, strict=True):
            ref = bernoulli_cusum_lower(row, p0=0.03, p1=0.015, h=1.5).first_signal_index
            assert f == (NO_SIGNAL if ref is None else ref + 1)

    def test_statistic_is_non_negative_and_silent_on_degradation(self) -> None:
        rng = np.random.default_rng(11)
        h, _ = calibrate_cusum_h(0.03, 0.015, target_arl0=300, n_sim=600, max_run=4000)
        res = bernoulli_cusum_lower(rng.random(3000) < 0.09, p0=0.03, p1=0.015, h=h)
        assert (res.table["cusum"] >= 0).all()
        assert res.first_signal_index is None

    def test_an_all_pass_stream_signals_after_h_over_the_pass_weight(self) -> None:
        h = 1.5
        res = bernoulli_cusum_lower(np.zeros(2000, dtype=bool), p0=0.03, p1=0.015, h=h)
        # with no failures, evidence grows by log((1-p1)/(1-p0)) per run
        step = np.log((1 - 0.015) / (1 - 0.03))
        assert res.first_signal_index == int(np.floor(h / step))

    def test_detects_a_halving_faster_than_its_in_control_arl(self) -> None:
        rng = np.random.default_rng(12)
        h, arl0 = calibrate_cusum_h(0.03, 0.015, target_arl0=500, n_sim=800, max_run=5000)
        first = cusum_first_signal(rng.random((500, 3000)) < 0.015, p0=0.03, p1=0.015, h=h)
        assert (first != NO_SIGNAL).all()
        assert np.median(first) < arl0 / 2

    def test_reset_restarts_against_the_process_as_it_now_stands(self) -> None:
        x = np.zeros(2000, dtype=bool)
        res = bernoulli_cusum_lower(x, p0=0.03, p1=0.015, h=1.5, reset_on_signal=True)
        stuck = bernoulli_cusum_lower(x, p0=0.03, p1=0.015, h=1.5, reset_on_signal=False)
        gap = np.diff(np.flatnonzero(res.table["signal"].to_numpy()))
        assert len(gap) > 5 and np.all(gap == gap[0])  # periodic re-signals
        assert stuck.table["signal"].iloc[200:].all()  # without reset: one endless alarm


class TestLowerCusumCalibration:
    def test_in_control_arl_matches_the_target_by_simulation(self) -> None:
        """h from calibrate_cusum_h gives the target ARL0 on fresh in-control streams."""
        p0, p1, target, horizon = 0.03, 0.015, 300.0, 5000
        h, arl0 = calibrate_cusum_h(p0, p1, target_arl0=target, n_sim=2000, max_run=horizon, seed=5)
        assert arl0 == pytest.approx(target, rel=0.05)
        rng = np.random.default_rng(6)
        first = cusum_first_signal(rng.random((4000, horizon)) < p0, p0=p0, p1=p1, h=h)
        assert (first != NO_SIGNAL).mean() > 0.999
        run_length = np.where(first == NO_SIGNAL, horizon, first)
        # SE of a mean of 4000 run lengths is about ARL / 63, so 8% is > 4 SE
        assert run_length.mean() == pytest.approx(target, rel=0.08)

    def test_in_control_arl_with_the_reference_implementation(self) -> None:
        """Same check through bernoulli_cusum_lower itself, on fewer streams."""
        p0, p1 = 0.03, 0.015
        h, arl0 = calibrate_cusum_h(p0, p1, target_arl0=200, n_sim=1500, max_run=3000, seed=7)
        rng = np.random.default_rng(8)
        lengths = []
        for _ in range(300):
            idx = bernoulli_cusum_lower(rng.random(3000) < p0, p0=p0, p1=p1, h=h).first_signal_index
            lengths.append(3000 if idx is None else idx + 1)
        assert np.mean(lengths) == pytest.approx(arl0, rel=0.2)

    def test_higher_threshold_gives_longer_arl(self) -> None:
        h_lo, a_lo = calibrate_cusum_h(0.03, 0.015, target_arl0=200, n_sim=600, max_run=4000)
        h_hi, a_hi = calibrate_cusum_h(0.03, 0.015, target_arl0=800, n_sim=600, max_run=4000)
        assert h_hi > h_lo and a_hi > a_lo


@pytest.mark.skipif(not (RAW / "secom.data").exists(), reason="needs data/raw/")
def test_cusum_stage_runs_the_lower_chart_on_the_real_data(tmp_path: Path) -> None:
    p = Paths.from_root(tmp_path)
    paths = Paths(
        root=p.root,
        raw=RAW,
        processed=p.processed,
        powerbi=p.powerbi,
        figures=p.figures,
        reports=p.reports,
    )
    ctx = PipelineContext(config=config_from_dict(None), paths=paths, log=lambda _: None)
    run_pipeline(ctx, only=["cusum"], skip_figures=True)
    c = ctx.results["cusum"]
    low = c["lower"]
    assert low["p0"] == c["p0"]
    assert low["p1"] == pytest.approx(0.5 * c["p0"])
    assert low["arl0_runs"] == pytest.approx(500, rel=0.1)
    # The lower chart is silent in the degraded period and in Phase II, whose
    # rates are above p0; its only alarms are inside the Phase I baseline.
    assert low["detection"]["pre_baseline"]["runs_to_signal"] is None
    assert low["detection"]["phase_ii"]["runs_to_signal"] is None
    assert low["all_runs"]["n_signals"] == len(low["all_runs"]["signal_runs"])
    assert all(
        w in {"2008-W35", "2008-W36", "2008-W37", "2008-W38", "2008-W39"}
        for w in low["all_runs"]["signal_weeks"]
    )
    lower_all = ctx["cusum_lower_all"]
    assert lower_all.side == "lower"
    assert int(lower_all.table["signal"].sum()) == low["all_runs"]["n_signals"]
    retro = c["lower_retrospective"]
    assert retro["p0"] > c["p0"]
    assert retro["first_signal_run"] is not None
