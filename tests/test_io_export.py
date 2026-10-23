"""Raw-file loading, the Power BI export builders, data download, models and cleaning helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import secom.clean
import secom.fetch
from _synthetic import N_RUNS, N_SENSORS, patch_shape, write_raw
from secom.clean import build_clean, standardise
from secom.export import (
    DATE_FORMAT,
    build_dim_date,
    build_dim_sensor,
    build_fact_measurement,
    build_fact_run,
    write_all,
)
from secom.io import load_raw
from secom.model import (
    EvalResult,
    _score,
    evaluate_forward,
    evaluate_shuffled,
    make_models,
    permutation_null,
)

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


# io


class TestLoadRaw:
    def test_loads_synthetic_files(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        patch_shape(monkeypatch)
        data = load_raw(write_raw(tmp_path))
        assert data.values.shape == (N_RUNS, N_SENSORS)
        assert list(data.runs.columns[:3]) == ["run_id", "timestamp", "is_fail"]
        assert data.is_fail.dtype == bool
        assert data.runs["run_id"].iloc[0] == 1

    def test_wrong_shape(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        patch_shape(monkeypatch, n_sensors=N_SENSORS + 1)
        with pytest.raises(ValueError, match="expected secom.data"):
            load_raw(write_raw(tmp_path))

    def test_row_count_mismatch(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        patch_shape(monkeypatch)
        write_raw(tmp_path)
        labels = tmp_path / "secom_labels.data"
        lines = labels.read_text(encoding="utf-8").splitlines()
        labels.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="row mismatch"):
            load_raw(tmp_path)

    def test_unexpected_labels(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        patch_shape(monkeypatch)
        write_raw(tmp_path)
        labels = tmp_path / "secom_labels.data"
        lines = labels.read_text(encoding="utf-8").splitlines()
        lines[0] = "0" + lines[0][lines[0].index(" ") :]
        labels.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="unexpected label values"):
            load_raw(tmp_path)

    def test_out_of_order_timestamps(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        patch_shape(monkeypatch)
        write_raw(tmp_path)
        labels = tmp_path / "secom_labels.data"
        lines = labels.read_text(encoding="utf-8").splitlines()
        lines[0], lines[-1] = lines[-1], lines[0]
        labels.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="not in non-decreasing"):
            load_raw(tmp_path)

    def test_missing_labels_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        write_raw(tmp_path)
        (tmp_path / "secom_labels.data").unlink()
        with pytest.raises(FileNotFoundError, match="secom_labels.data"):
            load_raw(tmp_path)


# export builders


@pytest.fixture
def runs() -> pd.DataFrame:
    ts = pd.to_datetime(
        ["2008-07-19 11:55", "2008-07-19 15:20", "2008-07-22 08:00", "2008-07-27 23:59"]
    )
    frame = pd.DataFrame(
        {
            "run_id": [1, 2, 3, 4],
            "timestamp": ts,
            "is_fail": [False, True, True, False],
            "S001": [1.0, 2.0, 3.0, 50.0],
            "S002": [5.0, 5.0, 5.0, 5.0],
        }
    )
    return secom.clean.add_time_context(frame)


class TestDimDate:
    def test_every_day_is_present(self, runs: pd.DataFrame) -> None:
        dim = build_dim_date(runs)
        assert len(dim) == 9  # 19 Jul .. 27 Jul inclusive
        assert dim["date_key"].iloc[0] == 20080719
        assert dim["has_production"].sum() == 3
        assert dim.loc[dim["date_key"] == 20080719, "is_weekend"].item()
        assert dim["year_week"].iloc[0] == "2008-W29"


class TestFactRun:
    def test_episode_starts_and_scores(self, runs: pd.DataFrame) -> None:
        cusum = pd.DataFrame({"cusum": [0.0, 1.0, 2.0, 0.0], "signal": [False, True, True, False]})
        phase = pd.Series(["a", "a", "b", "b"])
        fact = build_fact_run(runs, cusum=cusum, model_score=None, phase=phase)
        assert fact["cusum_episode_start"].tolist() == [False, True, False, False]
        assert fact["fail_count"].tolist() == [0, 1, 1, 0]
        assert fact["model_score"].isna().all()
        scored = build_fact_run(
            runs, cusum=cusum, model_score=np.array([0.1, 0.2, 0.3, 0.4]), phase=phase
        )
        assert scored["model_score"].tolist() == [0.1, 0.2, 0.3, 0.4]


def _limits() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sensor_id": ["S001", "S002"],
            "baseline_center": [2.0, 5.0],
            "baseline_sigma": [1.0, 0.0],
            "baseline_lcl": [-1.0, 5.0],
            "baseline_ucl": [5.0, 5.0],
            "baseline_median": [2.0, 5.0],
            "baseline_mad": [1.0, 0.0],
        }
    )


class TestFactMeasurement:
    def test_long_format_with_flags(self, runs: pd.DataFrame) -> None:
        long = build_fact_measurement(runs, ["S001", "S002"], _limits())
        assert len(long) == 8
        assert "baseline_mad" not in long.columns
        s1 = long[long["sensor_id"] == "S001"]
        assert s1["beyond_limits"].tolist() == [False, False, False, True]
        assert s1["robust_z"].tolist() == [-1.0, 0.0, 1.0, 48.0]
        # zero MAD: z is undefined rather than an error
        assert long.loc[long["sensor_id"] == "S002", "robust_z"].isna().all()


class TestDimSensor:
    def test_evidence_tiers(self) -> None:
        ids = ["S001", "S002", "S003", "S004", "S005"]
        audit = pd.DataFrame(
            {"retained": [True, True, True, True, False], "disposition": ["keep"] * 4 + ["drop"]},
            index=pd.Index(ids, name="sensor"),
        )
        drift = pd.DataFrame({"sensor_id": ids[:4], "drift_rho": 0.1, "drift_q": 0.5})

        def screen(q: list[float]) -> pd.DataFrame:
            return pd.DataFrame({"sensor": ids[:4], "cliffs_delta": 0.2, "q_mannwhitney": q})

        dim = build_dim_sensor(
            audit,
            drift=drift,
            regime=screen([0.9, 0.9, 0.01, 0.9]),
            fail_all=screen([0.9, 0.01, 0.9, 0.9]),
            fail_stable=screen([0.01, 0.9, 0.9, 0.9]),
            limits=_limits(),
            watchlist=["S001"],
        )
        tiers = dim.set_index("sensor_id")["evidence_tier"].str[0].to_dict()
        assert tiers == {"S001": "1", "S002": "2", "S003": "3", "S004": "4", "S005": "0"}
        assert dim["is_watchlist"].tolist() == [True, False, False, False, False]
        assert {"q_regime", "cliffs_delta_fail_all", "baseline_ucl"} <= set(dim.columns)


class TestWriteAll:
    def test_writes_tables_and_manifest(self, tmp_path: Path, runs: pd.DataFrame) -> None:
        manifest = write_all(tmp_path / "out", {"a": runs[["run_id", "timestamp"]], "b": runs})
        assert manifest["table"].tolist() == ["a", "b"]
        assert manifest["rows"].tolist() == [4, 4]
        text = (tmp_path / "out" / "a.csv").read_text(encoding="utf-8")
        assert pd.Timestamp("2008-07-19 11:55").strftime(DATE_FORMAT) in text
        assert (tmp_path / "out" / "_manifest.csv").exists()
        assert manifest["bytes"].iloc[0] == (tmp_path / "out" / "a.csv").stat().st_size


# fetch


def _fake_urlretrieve(calls: list[str]):
    def retrieve(url: str, dest: Path) -> None:
        calls.append(url)
        Path(dest).write_bytes(url.encode())

    return retrieve


class TestFetch:
    def test_downloads_missing_files_and_skips_present(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls: list[str] = []
        monkeypatch.setattr(secom.fetch.urllib.request, "urlretrieve", _fake_urlretrieve(calls))
        (tmp_path / "raw").mkdir()
        (tmp_path / "raw" / "secom.names").write_text("present", encoding="utf-8")
        log: list[str] = []
        paths = secom.fetch.fetch(tmp_path / "raw", log=log.append)
        assert [p.name for p in paths] == list(secom.fetch.FILES)
        assert calls == [
            f"{secom.fetch.BASE_URL}/secom.data",
            f"{secom.fetch.BASE_URL}/secom_labels.data",
        ]
        assert any(m.startswith("skip  secom.names") for m in log)
        assert sum(m.startswith("fetch ") for m in log) == 2

    def test_tep_writes_checksums(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        calls: list[str] = []
        monkeypatch.setattr(secom.fetch.urllib.request, "urlretrieve", _fake_urlretrieve(calls))
        paths = secom.fetch.fetch_tep(tmp_path, log=lambda _: None)
        assert len(paths) == len(calls) == len(secom.fetch.TEP_FILES) == 24
        sums = (tmp_path / "tep" / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        first = paths[0]
        assert sums[0] == f"{hashlib.sha256(first.read_bytes()).hexdigest()}  {first.name}"
        assert len(sums) == len(paths)


# model


@pytest.fixture(scope="module")
def xy() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(3)
    n = 160
    y = rng.random(n) < 0.2
    y[[100, 130, 150]] = True
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["a", "b", "c", "d"])
    X["a"] += 2.0 * y
    return X, y


class TestModel:
    def test_forward_split_learns_a_planted_signal(self, xy) -> None:
        X, y = xy
        res = evaluate_forward(X, y, make_models(seed=1), train_frac=0.6)
        assert [r.name for r in res] == ["logistic_l2", "hist_gradient_boosting"]
        lr = res[0]
        assert (lr.n_train, lr.n_test) == (96, 64)
        assert lr.roc_auc > 0.8
        assert "train window ends at run 96" in lr.notes[0]
        row = lr.as_row()
        assert row["lift_over_prevalence"] == pytest.approx(lr.pr_auc / lr.prevalence)

    def test_shuffled_cv_scores_every_run(self, xy) -> None:
        X, y = xy
        res = evaluate_shuffled(X, y, {"lr": make_models()["logistic_l2"]}, n_splits=3, seed=1)
        assert res[0].protocol == "stratified_3fold_shuffled"
        assert res[0].n_test == len(y)

    def test_permutation_null(self, xy) -> None:
        X, y = xy
        obs, null = permutation_null(X, y, make_models()["logistic_l2"], n_perm=4, seed=2)
        assert null.shape == (4,)
        assert obs > null.mean()

    def test_score_handles_a_class_missing_from_predictions(self) -> None:
        y = np.array([False, False, True, True])
        res = _score("m", "p", y, np.array([0.1, 0.2, 0.3, 0.9]), 10, threshold=0.5)
        assert (res.sensitivity, res.specificity) == (0.5, 1.0)
        assert res.ber == pytest.approx(0.25)

    def test_zero_prevalence_lift_is_nan(self) -> None:
        r = EvalResult("m", "p", 1, 1, 0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5)
        assert np.isnan(r.as_row()["lift_over_prevalence"])


# clean helpers


class TestCleanHelpers:
    def test_standardise_against_baseline(self) -> None:
        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 10.0], "b": [4.0, 4.0, 4.0, 5.0]})
        base = np.array([True, True, True, False])
        z = standardise(frame, ["a", "b"], base)
        assert z["a"].iloc[1] == 0.0
        assert z["a"].iloc[3] == pytest.approx(8 / 1.4826)
        assert z["b"].isna().all()  # zero MAD

    def test_build_clean_refuses_to_leave_gaps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runs = pd.DataFrame(
            {
                "run_id": [1, 2],
                "timestamp": pd.to_datetime(["2008-07-19", "2008-07-20"]),
                "is_fail": [False, True],
                "S001": [1.0, np.nan],
            }
        )
        audit = pd.DataFrame({"retained": [True]}, index=["S001"])
        monkeypatch.setattr(secom.clean, "causal_impute", lambda f, s, window: (f, pd.DataFrame()))
        with pytest.raises(ValueError, match="imputation left missing values"):
            build_clean(runs, audit)
