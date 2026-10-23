"""Tests for the beta-binomial overdispersion simulation in secom.power, and its stage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from secom.config import config_from_dict
from secom.pipeline import Paths, PipelineContext, run_pipeline
from secom.power import (
    Calendar,
    Dispersion,
    MonitorDesign,
    _subgroup_layout,
    beta_block_probs,
    block_index,
    calibrate_dispersion,
    n_blocks_needed,
    overdispersed_fails,
    shewhart_first_signal,
    simulate_detection,
    subgroup_sigma_z,
)
from secom.spc import p_chart

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# 3 weeks: days of 5, 7, 3 runs | 10, 2 | 4, 4, 4
CAL = Calendar(day_sizes=(5, 7, 3, 10, 2, 4, 4, 4), day_week=(0, 0, 0, 1, 1, 2, 2, 2))


def _realistic_calendar() -> Calendar:
    """14 weeks of 6 days with sizes roughly like the SECOM campaign."""
    rng = np.random.default_rng(0)
    sizes = tuple(int(v) for v in rng.integers(5, 30, 14 * 6))
    return Calendar(sizes, tuple(i // 6 for i in range(14 * 6)))


class TestCalendar:
    def test_week_sizes_sum_the_days(self) -> None:
        assert CAL.week_sizes == (15, 12, 12)
        assert CAL.n_weeks == 3

    @pytest.mark.parametrize(
        "sizes, weeks",
        [((1, 2), (0,)), ((1, 0), (0, 0)), ((1, 2), (1, 1)), ((1, 2, 3), (0, 2, 2))],
    )
    def test_rejects_bad_layouts(self, sizes: tuple, weeks: tuple) -> None:
        with pytest.raises(ValueError):
            Calendar(sizes, weeks)

    def test_dispersion_validates(self) -> None:
        with pytest.raises(ValueError):
            Dispersion(0.1, "month", CAL)
        with pytest.raises(ValueError):
            Dispersion(1.0, "day", CAL)


class TestBlocks:
    def test_day_blocks_follow_the_day_sizes(self) -> None:
        blk = block_index(CAL, 0, 39, "day")
        assert np.bincount(blk).tolist() == [5, 7, 3, 10, 2, 4, 4, 4]
        # the layout repeats with new blocks
        blk = block_index(CAL, 0, 45, "day")
        assert np.bincount(blk).tolist() == [5, 7, 3, 10, 2, 4, 4, 4, 5, 1]

    @pytest.mark.parametrize("offset", [0, 1, 2])
    def test_week_blocks_line_up_with_the_shewhart_subgroups(self, offset: int) -> None:
        horizon = 80
        blk = block_index(CAL, offset, horizon, "week")
        starts, ends = _subgroup_layout(np.array(CAL.week_sizes), offset, horizon)
        for b, (s, e) in enumerate(zip(starts, ends, strict=True)):
            assert (blk[s:e] == b).all()

    def test_start_week_rotates_the_layout(self) -> None:
        blk = block_index(CAL, 1, 12, "day")
        assert np.bincount(blk).tolist() == [10, 2]

    def test_blocks_needed_covers_every_start(self) -> None:
        for level in ("day", "week"):
            need = n_blocks_needed(CAL, 100, level)
            assert all(block_index(CAL, o, 100, level).max() < need for o in range(3))


class TestBetaBlocks:
    def test_mean_and_variance(self) -> None:
        u = np.random.default_rng(1).random(400_000)
        p, rho = 0.05, 0.04
        probs = beta_block_probs(p, rho, u)
        assert probs.mean() == pytest.approx(p, rel=0.01)
        assert probs.var() == pytest.approx(p * (1 - p) * rho, rel=0.03)

    def test_rho_zero_is_constant(self) -> None:
        assert (beta_block_probs(0.03, 0.0, np.random.default_rng(2).random(10)) == 0.03).all()

    def test_monotone_in_the_uniform(self) -> None:
        u = np.linspace(0.01, 0.99, 50)
        assert np.all(np.diff(beta_block_probs(0.03, 0.05, u)) > 0)

    def test_rho_zero_streams_are_independent_bernoulli(self) -> None:
        rng = np.random.default_rng(3)
        u = rng.random((20, 60))
        bu = rng.random((20, n_blocks_needed(CAL, 60, "day")))
        off = rng.integers(0, 3, 20)
        out = overdispersed_fails(u, bu, off, 0.1, Dispersion(0.0, "day", CAL))
        np.testing.assert_array_equal(out, u < 0.1)

    def test_runs_in_a_block_share_their_probability(self) -> None:
        """Block counts are beta-binomial: Var = n p (1-p) (1 + (n - 1) rho)."""
        cal = Calendar((40,) * 10, tuple(range(10)))
        rng = np.random.default_rng(4)
        n_sim, horizon, p, rho = 6000, 400, 0.06, 0.05
        u = rng.random((n_sim, horizon))
        bu = rng.random((n_sim, n_blocks_needed(cal, horizon, "week")))
        fails = overdispersed_fails(u, bu, np.zeros(n_sim, int), p, Dispersion(rho, "week", cal))
        counts = fails.reshape(n_sim, 10, 40).sum(axis=2)
        assert counts.mean() / 40 == pytest.approx(p, rel=0.02)
        expected = 40 * p * (1 - p) * (1 + 39 * rho)
        assert counts.var() == pytest.approx(expected, rel=0.05)


class TestSigmaZ:
    def test_matches_spc_p_chart(self) -> None:
        rng = np.random.default_rng(5)
        sizes = np.array([13, 21, 48, 108, 51, 208, 169, 133])
        fails = rng.random((6, int(sizes.sum()))) < 0.07
        got = subgroup_sigma_z(fails, sizes, np.zeros(6, int))
        for row, g in zip(fails, got, strict=True):
            d = np.add.reduceat(row.astype(int), np.r_[0, np.cumsum(sizes)[:-1]])
            assert g == pytest.approx(p_chart(d, sizes).sigma_z)

    def test_about_one_for_independent_streams(self) -> None:
        cal = _realistic_calendar()
        sizes = np.array(cal.week_sizes)
        rng = np.random.default_rng(6)
        fails = rng.random((3000, int(sizes.sum()))) < 0.05
        sz = subgroup_sigma_z(fails, sizes, np.zeros(3000, int))
        assert np.nanmean(sz) == pytest.approx(1.0, abs=0.04)


class TestCalibration:
    @pytest.mark.parametrize("level", ["week", "day"])
    def test_calibrated_rho_reproduces_the_target_on_fresh_streams(self, level: str) -> None:
        cal = _realistic_calendar()
        target = 1.45
        rho, achieved = calibrate_dispersion(
            target, p=0.04, calendar=cal, level=level, n_sim=1500, seed=7
        )
        assert rho > 0 and achieved == pytest.approx(target, abs=1e-3)
        sizes = np.array(cal.week_sizes)
        horizon = int(sizes.sum())
        rng = np.random.default_rng(8)
        n = 3000
        u = rng.random((n, horizon))
        bu = rng.random((n, n_blocks_needed(cal, horizon, level)))
        off = np.zeros(n, int)
        fails = overdispersed_fails(u, bu, off, 0.04, Dispersion(rho, level, cal))
        sz = subgroup_sigma_z(fails, sizes, off)
        se = np.nanstd(sz) / np.sqrt(n)
        # calibration and verification each carry Monte Carlo error
        assert np.nanmean(sz) == pytest.approx(target, abs=5 * se * np.sqrt(2))

    def test_day_effects_need_a_larger_rho_than_week_effects(self) -> None:
        cal = _realistic_calendar()
        kw = dict(p=0.04, calendar=cal, n_sim=800, seed=9)
        rho_w, _ = calibrate_dispersion(1.45, level="week", **kw)  # type: ignore[arg-type]
        rho_d, _ = calibrate_dispersion(1.45, level="day", **kw)  # type: ignore[arg-type]
        assert rho_d > rho_w

    def test_target_below_independent_gives_rho_zero(self) -> None:
        rho, achieved = calibrate_dispersion(
            0.5, p=0.04, calendar=_realistic_calendar(), level="week", n_sim=300, seed=10
        )
        assert rho == 0.0 and achieved > 0.5


def _design(cal: Calendar) -> MonitorDesign:
    return MonitorDesign(
        p0=0.03,
        subgroup_sizes=cal.week_sizes,
        sigma_z=1.5,
        sigma_z_phase1=2.0,
        cusum_upper_p1=0.06,
        cusum_upper_h=1.8,
        cusum_lower_p1=0.015,
        cusum_lower_h=1.5,
        g_lcl=0.0,
        g_ucl=212.0,
    )


class TestSimulateDetection:
    def test_rho_zero_reproduces_the_independent_table(self) -> None:
        cal = _realistic_calendar()
        kw = dict(horizons=(100, 400), n_sim=300, seed=11)
        base = simulate_detection(_design(cal), (0.5, 1.0, 2.0), **kw)  # type: ignore[arg-type]
        same = simulate_detection(
            _design(cal),
            (0.5, 1.0, 2.0),
            dispersion=Dispersion(0.0, "day", cal),
            **kw,  # type: ignore[arg-type]
        )
        pd.testing.assert_frame_equal(base, same)

    def test_calendar_must_match_the_subgroups(self) -> None:
        with pytest.raises(ValueError, match="calendar weeks"):
            simulate_detection(
                _design(_realistic_calendar()),
                (1.0,),
                horizons=(50,),
                n_sim=10,
                seed=0,
                dispersion=Dispersion(0.1, "day", CAL),
            )

    def test_overdispersion_inflates_the_p_chart_false_alarm_rate(self) -> None:
        cal = _realistic_calendar()
        sizes = np.array(cal.week_sizes)
        horizon = int(sizes.sum())
        rng = np.random.default_rng(12)
        n = 3000
        u = rng.random((n, horizon))
        off = rng.integers(0, len(sizes), n)
        bu = rng.random((n, n_blocks_needed(cal, horizon, "week")))
        rates = []
        for rho in (0.0, 0.01, 0.03):
            fails = overdispersed_fails(u, bu, off, 0.03, Dispersion(rho, "week", cal))
            first = shewhart_first_signal(fails, sizes, off, p0=0.03)
            rates.append(float(np.mean(first > 0)))
        assert rates[0] < rates[1] < rates[2]


# the stage on the real data


@pytest.mark.skipif(not (RAW / "secom.data").exists(), reason="needs data/raw/")
def test_robustness_stage_on_the_real_data(tmp_path: Path) -> None:
    p = Paths.from_root(tmp_path)
    paths = Paths(
        root=p.root,
        raw=RAW,
        processed=p.processed,
        powerbi=p.powerbi,
        figures=p.figures,
        reports=p.reports,
    )
    cfg = config_from_dict(
        {
            "changepoint": {"n_perm": 99, "n_boot": 50},
            "shift_simulation": {"n_sim": 400},
            "cusum_robustness": {
                "p0_sweep": [0.03, 0.05],
                "arl_n_sim": 400,
                "arl_max_run": 4000,
                "calibration_n_sim": 600,
                "verify_n_sim": 600,
                "ratios": [1.0, 2.0],
            },
        }
    )
    ctx = PipelineContext(config=cfg, paths=paths, log=lambda _: None)
    run_pipeline(ctx, only=["cusum_robustness"], skip_figures=True)
    r = ctx.results["cusum_robustness"]

    pr = r["p0_reestimate"]
    assert pr["rates"]["post_change"] == pytest.approx(51 / 1193)
    dep, red = pr["headline"]["deployed"], pr["headline"]["redesigned"]
    assert dep["h"] == ctx.results["cusum"]["h"]
    assert red["p0"] > dep["p0"] and red["h"] > dep["h"]
    # a chart designed on too low a rate false-alarms sooner at the higher rate
    assert dep["arl_at_post_change_rate"]["arl_runs"] < red["arl_at_post_change_rate"]["arl_runs"]
    assert red["alarms"]["w35_onward"] <= dep["alarms"]["w35_onward"]
    assert dep["alarms"]["baseline_self_check"] == 2

    od = r["overdispersion"]
    assert od["target_sigma_z"] == pytest.approx(1.463, abs=1e-3)
    for lv in ("week", "day"):
        v = od["levels"][lv]["verification"]
        assert v["mean_sigma_z"] == pytest.approx(od["target_sigma_z"], abs=6 * v["mc_se"])
    assert od["levels"]["day"]["rho"] > od["levels"]["week"]["rho"]
    ic = od["in_control_headline"]
    assert ic["day"]["arl_runs"] < ic["independent"]["arl_runs"]
    fa = od["detection_headline"]["false_alarm_1567"]["p_chart"]
    assert fa["day"] > fa["independent"]

    det = ctx["robustness_detection"]
    assert set(det["condition"]) == {"independent", "week", "day"}
    # independent rows are the shift_simulation table, row for row
    ss = ctx["power_table"]
    ind = det[det["condition"] == "independent"].drop(columns="condition")
    for ratio in (1.0, 2.0):
        a = ind[np.isclose(ind["ratio"], ratio)].reset_index(drop=True)
        b = ss[np.isclose(ss["ratio"], ratio)].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)

    assert r["lower_retrospective"]["first_signal_at_or_after_change"] >= 375
