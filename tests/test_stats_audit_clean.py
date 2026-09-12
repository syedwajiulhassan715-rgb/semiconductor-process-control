"""Tests for the statistics, audit and cleaning layers.

The most important test in this file is
``TestCausalImputation::test_imputation_cannot_see_the_future``. Every
performance number in the project rests on the claim that no future information
reaches a row, and that claim is worth testing directly rather than trusting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from secom.audit import audit_sensors, missingness_vs_outcome
from secom.clean import add_time_context, build_clean, causal_impute
from secom.io import N_RUNS, N_SENSORS, load_raw, sensor_ids
from secom.stats import (
    benjamini_hochberg,
    cliffs_delta,
    cohens_d,
    compare_groups,
    screen_sensors,
)


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


class TestBenjaminiHochberg:
    def test_matches_statsmodels(self) -> None:
        rng = np.random.default_rng(0)
        p = np.concatenate([rng.uniform(0, 1, 300), rng.uniform(0, 1e-3, 30)])
        assert np.allclose(benjamini_hochberg(p), multipletests(p, method="fdr_bh")[1])

    def test_is_monotone_in_p(self) -> None:
        p = np.array([0.001, 0.01, 0.02, 0.04, 0.2, 0.9])
        q = benjamini_hochberg(p)
        assert np.all(np.diff(q) >= -1e-12)

    def test_q_never_below_p(self) -> None:
        rng = np.random.default_rng(1)
        p = rng.uniform(0, 1, 500)
        assert np.all(benjamini_hochberg(p) >= p - 1e-12)

    def test_nan_excluded_from_the_correction_count(self) -> None:
        """A sensor that could not be tested must not inflate m for the rest."""
        p = np.array([0.01, 0.02, np.nan, 0.03])
        q = benjamini_hochberg(p)
        assert np.isnan(q[2])
        assert np.allclose(q[[0, 1, 3]], benjamini_hochberg(np.array([0.01, 0.02, 0.03])))

    def test_all_nan_returns_all_nan(self) -> None:
        assert np.isnan(benjamini_hochberg(np.array([np.nan, np.nan]))).all()

    def test_controls_fdr_under_the_global_null(self) -> None:
        rng = np.random.default_rng(2)
        rejections = [
            (benjamini_hochberg(rng.uniform(0, 1, 400)) < 0.05).sum() for _ in range(200)
        ]
        # Under a global null, BH rejects nothing in the large majority of runs.
        assert np.mean(np.array(rejections) > 0) < 0.10


class TestEffectSizes:
    def test_cliffs_delta_matches_brute_force(self) -> None:
        rng = np.random.default_rng(3)
        x, y = rng.normal(0, 1, 80), rng.normal(0.7, 1, 90)
        brute = np.mean(np.sign(x[:, None] - y[None, :]))
        assert cliffs_delta(x, y) == pytest.approx(brute, abs=1e-9)

    def test_cliffs_delta_bounds(self) -> None:
        assert cliffs_delta(np.array([5.0, 6, 7]), np.array([1.0, 2, 3])) == pytest.approx(1.0)
        assert cliffs_delta(np.array([1.0, 2, 3]), np.array([5.0, 6, 7])) == pytest.approx(-1.0)

    def test_cohens_d_recovers_a_known_shift(self) -> None:
        rng = np.random.default_rng(4)
        a, b = rng.normal(0.5, 1, 4000), rng.normal(0, 1, 4000)
        assert cohens_d(a, b) == pytest.approx(0.5, abs=0.06)

    def test_effect_sizes_ignore_nan(self) -> None:
        a = np.array([1.0, 2.0, np.nan, 3.0])
        assert cohens_d(a, np.array([1.0, 2.0, 3.0])) == pytest.approx(0.0, abs=1e-9)

    def test_cliffs_delta_is_robust_where_cohens_d_is_not(self) -> None:
        """The reason the screens rank on the rank-based measure."""
        rng = np.random.default_rng(5)
        a = np.r_[rng.normal(1, 0.3, 200), [500.0]]   # one wild outlier
        b = rng.normal(0, 0.3, 200)
        assert cliffs_delta(a, b) > 0.9      # still sees near-total separation
        assert cohens_d(a, b) < 0.5          # swamped by the outlier's variance


class TestCompareGroups:
    def test_degenerate_input_returns_nan_not_an_exception(self) -> None:
        res = compare_groups(np.ones(5), np.ones(5))
        assert np.isnan(res.p_mannwhitney)

    def test_too_few_observations(self) -> None:
        res = compare_groups(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        assert res.n_a == 2 and np.isnan(res.cohens_d)

    def test_screen_sorts_by_q_and_adds_both_corrections(self) -> None:
        rng = np.random.default_rng(6)
        frame = pd.DataFrame({f"S{i:03d}": rng.normal(size=200) for i in range(1, 21)})
        frame["S001"] += np.r_[np.ones(100) * 3, np.zeros(100)]
        mask = np.r_[np.ones(100, bool), np.zeros(100, bool)]
        out = screen_sensors(frame, list(frame.columns), mask, ~mask)
        assert {"q_mannwhitney", "q_welch"} <= set(out.columns)
        assert out["sensor"].iloc[0] == "S001"
        assert out["q_mannwhitney"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


@pytest.fixture
def audit_frame() -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(7)
    n = 200
    base = rng.normal(0, 1, n)
    frame = pd.DataFrame(
        {
            "S001": base,                                  # retained
            "S002": np.zeros(n),                           # constant
            "S003": np.zeros(n),                           # constant, duplicate of S002
            "S004": base.copy(),                           # exact duplicate of S001
            "S005": base * 2 + 1,                          # perfectly correlated
            "S006": rng.normal(5, 2, n),                   # retained
            "S007": np.where(np.arange(n) < 40, rng.normal(size=n), np.nan),  # sparse
        }
    )
    return frame, [f"S00{i}" for i in range(1, 8)]


class TestAudit:
    def test_dispositions_partition_the_sensors(self, audit_frame) -> None:
        frame, sensors = audit_frame
        audit = audit_sensors(frame, sensors)
        assert audit["disposition"].value_counts().sum() == len(sensors)

    def test_identifies_each_defect_with_its_own_reason(self, audit_frame) -> None:
        frame, sensors = audit_frame
        audit = audit_sensors(frame, sensors)
        assert audit.at["S002", "disposition"] == "drop_constant"
        assert audit.at["S003", "disposition"] == "drop_constant"
        assert audit.at["S004", "disposition"] == "drop_duplicate"
        assert audit.at["S004", "duplicate_of"] == "S001"
        assert audit.at["S005", "disposition"] == "drop_near_duplicate"
        assert audit.at["S007", "disposition"] == "drop_sparse"
        assert audit.at["S001", "retained"]
        assert audit.at["S006", "retained"]

    def test_constants_are_not_reported_as_duplicates(self, audit_frame) -> None:
        """A frozen column duplicates every other frozen column.

        Attributing it to duplication would hide the more specific defect, so
        the cascade must resolve constants first.
        """
        frame, sensors = audit_frame
        audit = audit_sensors(frame, sensors)
        assert (audit.loc[["S002", "S003"], "disposition"] == "drop_constant").all()

    def test_nan_pattern_participates_in_duplicate_identity(self) -> None:
        """Two columns with equal present values but different gaps differ."""
        a = np.array([1.0, 2.0, np.nan, 4.0])
        b = np.array([1.0, 2.0, 3.0, 4.0])
        frame = pd.DataFrame({"S001": a, "S002": b})
        audit = audit_sensors(frame, ["S001", "S002"], max_missing_frac=0.9)
        assert audit.at["S002", "disposition"] != "drop_duplicate"

    def test_sparse_threshold_is_respected(self, audit_frame) -> None:
        frame, sensors = audit_frame
        lenient = audit_sensors(frame, sensors, max_missing_frac=0.95)
        assert lenient.at["S007", "disposition"] != "drop_sparse"

    def test_robust_outlier_fraction_resists_masking(self) -> None:
        """Masking: several outliers together hide each other from an SD screen.

        Ten extreme points inflate the standard deviation enough that the
        +/- 5 SD envelope swallows all ten. The MAD is unmoved by 10% of the
        data, so the same points remain 5 MAD-sigma out. SECOM has sensors with
        excess kurtosis above 60, which is exactly this regime.
        """
        rng = np.random.default_rng(20)
        x = np.r_[rng.normal(0, 1, 90), np.full(10, 1000.0)]
        audit = audit_sensors(pd.DataFrame({"S001": x}), ["S001"])
        assert audit.at["S001", "robust_outlier_frac"] == pytest.approx(0.10)

        sd_flagged = np.abs(x - x.mean()) > 5 * x.std(ddof=1)
        assert sd_flagged.sum() == 0  # the SD screen finds none of them

    def test_outlier_fraction_when_the_mad_is_degenerate(self) -> None:
        """A sensor pinned at one value except for a few excursions.

        MAD is 0, so there is no scale to measure against; the differing points
        are still unambiguously outliers and must not be reported as zero.
        """
        x = np.r_[np.zeros(99), [1000.0]]
        audit = audit_sensors(pd.DataFrame({"S001": x}), ["S001"])
        assert audit.at["S001", "robust_outlier_frac"] == pytest.approx(0.01)


class TestMissingnessVsOutcome:
    def test_detects_informative_missingness(self) -> None:
        n = 400
        is_fail = np.r_[np.ones(100, bool), np.zeros(300, bool)]
        col = np.where(is_fail, np.nan, 1.0)   # missing only when failing
        out = missingness_vs_outcome(pd.DataFrame({"S001": col}), ["S001"], is_fail)
        assert out["p_fisher"].iloc[0] < 1e-10

    def test_ignores_fully_present_columns(self) -> None:
        out = missingness_vs_outcome(
            pd.DataFrame({"S001": np.ones(50)}), ["S001"], np.zeros(50, bool)
        )
        assert out.empty


# ---------------------------------------------------------------------------
# cleaning
# ---------------------------------------------------------------------------


class TestCausalImputation:
    def test_imputation_cannot_see_the_future(self) -> None:
        """The load-bearing guarantee of the whole project.

        Mutating values *after* a gap must not change what is imputed into that
        gap. If this fails, every forward-evaluated model number is optimistic
        and the comparison against the shuffled protocol is meaningless.
        """
        rng = np.random.default_rng(8)
        col = rng.normal(size=300)
        col[150] = np.nan
        original = pd.DataFrame({"S001": col})
        mutated = original.copy()
        mutated.loc[151:, "S001"] = 999.0

        a, _ = causal_impute(original, ["S001"])
        b, _ = causal_impute(mutated, ["S001"])
        assert a.loc[150, "S001"] == pytest.approx(b.loc[150, "S001"])

    def test_uses_trailing_window_only(self) -> None:
        col = np.array([1.0, 1.0, 1.0, np.nan, 50.0, 50.0])
        out, _ = causal_impute(pd.DataFrame({"S001": col}), ["S001"], window=3)
        assert out.loc[3, "S001"] == pytest.approx(1.0)  # not pulled towards 50

    def test_leading_gap_is_seeded_and_logged(self) -> None:
        col = np.array([np.nan, np.nan, 5.0, 6.0])
        out, log = causal_impute(pd.DataFrame({"S001": col}), ["S001"])
        assert out["S001"].notna().all()
        assert log["filled_seed_lookahead"].iloc[0] == 2

    def test_leaves_complete_columns_untouched(self) -> None:
        col = np.arange(10.0)
        out, log = causal_impute(pd.DataFrame({"S001": col}), ["S001"])
        assert np.array_equal(out["S001"].to_numpy(), col)
        assert log.empty


class TestTimeContext:
    def test_shift_boundaries(self) -> None:
        ts = pd.to_datetime(["2008-07-19 07:00", "2008-07-19 15:00", "2008-07-19 23:00"])
        out = add_time_context(pd.DataFrame({"timestamp": ts}))
        assert list(out["shift"]) == ["A", "B", "C"]

    def test_gap_column(self) -> None:
        ts = pd.to_datetime(["2008-07-19 10:00", "2008-07-19 10:37"])
        out = add_time_context(pd.DataFrame({"timestamp": ts}))
        assert np.isnan(out["minutes_since_prev"].iloc[0])
        assert out["minutes_since_prev"].iloc[1] == pytest.approx(37.0)


# ---------------------------------------------------------------------------
# io + end-to-end, skipped when the raw download is absent
# ---------------------------------------------------------------------------

RAW = pytest.importorskip("pathlib").Path(__file__).resolve().parents[1] / "data" / "raw"
needs_data = pytest.mark.skipif(
    not (RAW / "secom.data").exists(),
    reason="run scripts/00_fetch_data.py to enable data-dependent tests",
)


def test_sensor_ids_are_one_based_and_padded() -> None:
    ids = sensor_ids(590)
    assert ids[0] == "S001" and ids[-1] == "S590" and len(ids) == 590


@needs_data
class TestRealData:
    def test_documented_shape_and_labels(self) -> None:
        data = load_raw(RAW)
        assert len(data.runs) == N_RUNS
        assert len(data.sensors) == N_SENSORS
        assert int(data.is_fail.sum()) == 104
        assert data.runs["timestamp"].is_monotonic_increasing

    def test_pipeline_leaves_no_missing_values(self) -> None:
        data = load_raw(RAW)
        audit = audit_sensors(data.runs, data.sensors)
        clean, retained, _ = build_clean(data.runs, audit)
        assert not clean[retained].isna().to_numpy().any()
        assert len(retained) == int(audit["retained"].sum())

    def test_audit_accounts_for_every_raw_sensor(self) -> None:
        data = load_raw(RAW)
        audit = audit_sensors(data.runs, data.sensors)
        assert len(audit) == N_SENSORS
        assert audit["disposition"].value_counts().sum() == N_SENSORS


def test_missing_raw_files_raise_a_helpful_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="00_fetch_data"):
        load_raw(tmp_path)
