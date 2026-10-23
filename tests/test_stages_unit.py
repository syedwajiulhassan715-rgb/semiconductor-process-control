"""Unit tests for the core pipeline stages on a small synthetic dataset.

The end-to-end run on the real data is slow and skips without ``data/raw/``.
These tests drive each stage's ``run`` directly on a few hundred synthetic runs
so every stage is exercised in a fresh clone too. The CUSUM stage is not run
here: export only needs its result table, so a stand-in is injected.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from _synthetic import FAST_CONFIG, N_RUNS, N_SENSORS, patch_shape, write_raw
from secom.config import config_from_dict
from secom.pipeline import PipelineContext, plotting
from secom.pipeline.context import MissingArtifactError, Paths
from secom.pipeline.stages import (
    audit,
    changepoint,
    clean,
    export,
    load,
    model,
    sensor_screen,
    yield_charts,
)

# sklearn deprecation noise from the model stage; run_pipeline silences it too
pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def _fake_cusum(n: int) -> SimpleNamespace:
    signal = np.zeros(n, dtype=bool)
    signal[[10, 11, 12, 50]] = True
    return SimpleNamespace(
        table=pd.DataFrame({"cusum": np.linspace(0.0, 1.0, n), "signal": signal})
    )


@pytest.fixture(scope="module")
def ctx(tmp_path_factory: pytest.TempPathFactory) -> PipelineContext:
    root = tmp_path_factory.mktemp("synthetic")
    write_raw(root / "data" / "raw")
    cfg = config_from_dict(FAST_CONFIG)
    logs: list[str] = []
    context = PipelineContext(config=cfg, paths=Paths.from_root(root), log=logs.append)
    context.paths.ensure()
    plotting.configure(cfg.figures.format, cfg.figures.dpi)
    with pytest.MonkeyPatch.context() as mp:
        patch_shape(mp)
        load.run(context)
    for stage in (audit, clean, yield_charts, sensor_screen, model, changepoint):
        stage.run(context)
    context.artifacts["cusum_all"] = _fake_cusum(len(context["clean"]))
    export.run(context)
    context.artifacts["_logs"] = logs
    return context


class TestLoad:
    def test_dataset_summary(self, ctx: PipelineContext) -> None:
        ds = ctx.results["dataset"]
        assert ds["n_runs"] == N_RUNS
        assert ds["n_sensors_raw"] == N_SENSORS
        assert 0 < ds["fail_rate"] < 0.5
        assert 0 < ds["cell_missing_rate"] < 0.1
        assert ds["span_days"] > 80


class TestAudit:
    def test_every_defect_is_dropped(self, ctx: PipelineContext) -> None:
        aud = ctx["audit"]
        assert len(aud) == N_SENSORS
        dropped = set(aud.index[~aud["retained"]])
        assert {"S016", "S017", "S018"} <= dropped
        assert ctx.results["audit"]["n_retained"] == int(aud["retained"].sum())

    def test_missingness_tests_run(self, ctx: PipelineContext) -> None:
        a = ctx.results["audit"]
        assert a["missingness_tests"] == len(ctx["missingness"]) > 0
        assert a["missingness_expected_by_chance"] == round(0.05 * a["missingness_tests"], 1)


class TestClean:
    def test_no_missing_values_and_logged_imputation(self, ctx: PipelineContext) -> None:
        clean_df, retained = ctx["clean"], ctx["retained"]
        assert not clean_df[retained].isna().any().any()
        c = ctx.results["cleaning"]
        assert c["cells_imputed"] > 0
        assert c["lookahead_cells"] >= 5  # at least the leading gap in S020
        log = ctx["impute_log"].set_index("sensor")
        assert log.loc["S020", "filled_seed_lookahead"] == 5

    def test_masks_partition_runs(self, ctx: PipelineContext) -> None:
        m = ctx["masks"]
        total = m["pre"].astype(int) + m["base"].astype(int) + m["post"].astype(int)
        assert (total == 1).all()
        assert set(ctx["phase"]) == set(clean.PHASE_LABELS)


class TestYieldCharts:
    def test_weekly_subgroups_and_regime(self, ctx: PipelineContext) -> None:
        y = ctx.results["yield"]
        assert y["weekly_subgroups"] == len(ctx["weekly"]) == 13
        assert y["regime_pre_rate"] > y["regime_post_rate"]
        assert y["regime_or_ci"][0] < y["regime_odds_ratio"] < y["regime_or_ci"][1]
        assert y["laney_violations"] <= y["naive_violations"]


class TestSensorScreen:
    def test_planted_effects_are_found(self, ctx: PipelineContext) -> None:
        drift = ctx["drift"].set_index("sensor_id")
        assert drift.loc["S002", "drift_q"] < 0.05
        assert ctx["regime"]["sensor"].iloc[0] == "S003"
        assert (
            "S004" in ctx["fail_all"].loc[ctx["fail_all"]["q_mannwhitney"] < 0.05, "sensor"].values
        )

    def test_watchlist_and_limits(self, ctx: PipelineContext) -> None:
        watch = ctx["watchlist"]
        assert 0 < len(watch) <= FAST_CONFIG["sensors"]["watchlist_max"]
        assert len(set(watch)) == len(watch)
        limits = ctx["limits"]
        assert list(limits["sensor_id"]) == ctx["retained"]
        assert (limits["baseline_lcl"] <= limits["baseline_ucl"]).all()


class TestModel:
    def test_results_and_artifacts(self, ctx: PipelineContext) -> None:
        m = ctx.results["model"]
        assert len(m["results"]) == 4  # two models x two protocols
        assert len(m["rolling_origin"]) == 2
        assert m["permutation"]["n_perm"] == 3
        assert 0 <= m["permutation"]["p_value"] <= 1
        assert [t["top_k_pct"] for t in m["triage"]] == [10, 50]
        cut = ctx["forward_cut"]
        assert cut == int(N_RUNS * 0.7)
        assert len(ctx["forward_scores"]) == N_RUNS - cut


class TestChangepoint:
    def test_estimate_and_interval(self, ctx: PipelineContext) -> None:
        cp = ctx.results["changepoint"]
        est = cp["estimate"]
        assert est["n_before"] + est["n_after"] == N_RUNS
        assert est["rate_before"] > est["rate_after"]
        lo = cp["ci"]["first_run_lo"]["run"]
        hi = cp["ci"]["first_run_hi"]["run"]
        assert lo <= est["first_run_after"]["run"] <= hi
        assert est["p_min_attainable"] == pytest.approx(1 / 31)
        assert cp["segmentation"]["segments"]

    def test_figure_written(self, ctx: PipelineContext) -> None:
        assert (ctx.paths.figures / "fig12_changepoint.svg").exists()


class TestExport:
    def test_star_schema_written(self, ctx: PipelineContext) -> None:
        manifest = ctx["manifest"]
        assert set(manifest["table"]) == {
            "dim_date",
            "dim_sensor",
            "fact_run",
            "fact_measurement",
            "fact_yield_week",
            "fact_gchart",
        }
        for f in manifest["file"]:
            assert (ctx.paths.powerbi / f).exists()
        assert (ctx.paths.powerbi / "_manifest.csv").exists()

    def test_processed_tables_written(self, ctx: PipelineContext) -> None:
        for name in (
            "secom_clean_wide.csv",
            "sensor_audit.csv",
            "model_results.csv",
            "confound_check.csv",
            "imputation_log.csv",
        ):
            assert (ctx.paths.processed / name).exists()

    def test_results_json_holds_every_stage(self, ctx: PipelineContext) -> None:
        results = json.loads((ctx.paths.reports / "results.json").read_text(encoding="utf-8"))
        for key in ("dataset", "audit", "cleaning", "yield", "sensors", "model", "changepoint"):
            assert key in results
        assert results["watchlist"] == ctx["watchlist"]

    def test_fact_run_scores_only_the_test_window(self, ctx: PipelineContext) -> None:
        fact = ctx["tables"]["fact_run"]
        cut = ctx["forward_cut"]
        assert fact["model_score"].iloc[:cut].isna().all()
        assert fact["model_score"].iloc[cut:].notna().all()
        # signals at rows 10-12 and 50 are two episodes
        assert fact["cusum_episode_start"].sum() == 2

    def test_files_use_lf_line_endings(self, ctx: PipelineContext) -> None:
        data = (ctx.paths.powerbi / "fact_run.csv").read_bytes()
        assert b"\r\n" not in data


def test_stage_logs_are_emitted(ctx: PipelineContext) -> None:
    logs = ctx["_logs"]
    assert any("runs x" in m for m in logs)
    assert any("results.json written" in m for m in logs)


# context


class TestContext:
    def _ctx(self, tmp_path: Path, enabled: bool) -> PipelineContext:
        cfg = config_from_dict({"figures": {"format": "svg", "dpi": 40}})
        return PipelineContext(config=cfg, paths=Paths.from_root(tmp_path), figures_enabled=enabled)

    def test_missing_artifact_names_the_key(self, tmp_path: Path) -> None:
        c = self._ctx(tmp_path, True)
        with pytest.raises(MissingArtifactError, match="'nope'"):
            c["nope"]
        with pytest.raises(KeyError):
            c["nope"]

    def test_save_figure_writes_and_closes(self, tmp_path: Path) -> None:
        import matplotlib.pyplot as plt

        plotting.configure("svg", 40)
        c = self._ctx(tmp_path, True)
        fig = plt.figure()
        path = c.save_figure(fig, "figXX_test")
        assert path == c.paths.figures / "figXX_test.svg"
        assert path.exists()
        assert not plt.fignum_exists(fig.number)

    def test_save_figure_disabled_discards(self, tmp_path: Path) -> None:
        import matplotlib.pyplot as plt

        plotting.configure("svg", 40)
        c = self._ctx(tmp_path, False)
        fig = plt.figure()
        assert c.save_figure(fig, "figXX_test") is None
        assert not c.paths.figures.exists()
        assert not plt.fignum_exists(fig.number)

    def test_paths_layout_and_figures_override(self, tmp_path: Path) -> None:
        p = Paths.from_root(tmp_path, figures=tmp_path / "figs")
        assert p.raw == tmp_path.resolve() / "data" / "raw"
        assert p.figures == (tmp_path / "figs").resolve()
        p.ensure()
        for d in (p.processed, p.powerbi, p.figures, p.reports):
            assert d.is_dir()


def test_changepoint_without_figures(tmp_path: Path, ctx: PipelineContext) -> None:
    """With figures off, the stage still records results and draws nothing."""
    c = PipelineContext(
        config=ctx.config,
        paths=Paths.from_root(tmp_path),
        artifacts={k: ctx[k] for k in changepoint.REQUIRES},
        figures_enabled=False,
        log=lambda _: None,
    )
    changepoint.run(c)
    assert "changepoint" in c.results
    assert not (c.paths.figures / "fig12_changepoint.svg").exists()


# runner


class TestRunner:
    def test_no_selection_runs_every_stage(self) -> None:
        from secom.pipeline import STAGES, resolve_stages

        assert resolve_stages(None) == STAGES

    def test_unproduced_requirement_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from secom.pipeline import PipelineError, resolve_stages

        monkeypatch.setattr(audit, "REQUIRES", ("raw", "never_made"))
        with pytest.raises(PipelineError, match="'never_made', which no stage produces"):
            resolve_stages(["audit"])

    def test_stage_that_produces_nothing_stops_the_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from secom.pipeline import PipelineError, run_pipeline

        monkeypatch.setattr(load, "run", lambda ctx: None)
        c = PipelineContext(
            config=config_from_dict(FAST_CONFIG), paths=Paths.from_root(tmp_path), log=print
        )
        with pytest.raises(PipelineError, match="'audit' is missing raw"):
            run_pipeline(c, only=["audit"], skip_figures=True)
        assert c.log is print  # the per-stage prefix is undone on failure
