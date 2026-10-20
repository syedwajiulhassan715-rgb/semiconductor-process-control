"""Tests for secom.config, the stage registry and the CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from secom.cli import main
from secom.config import (
    Config,
    ConfigError,
    config_from_dict,
    load_config,
    register_section,
    unregister_section,
)
from secom.pipeline import (
    STAGES,
    Paths,
    PipelineContext,
    PipelineError,
    resolve_stages,
    run_pipeline,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YAML = ROOT / "config" / "default.yaml"
RAW = ROOT / "data" / "raw"


# config


class TestConfig:
    def test_default_yaml_matches_builtin_defaults(self) -> None:
        assert load_config(DEFAULT_YAML) == load_config(None)

    def test_defaults_match_the_published_analysis(self) -> None:
        cfg = load_config(DEFAULT_YAML)
        assert cfg.seed == 20080719
        assert cfg.alpha == 0.05
        assert cfg.regimes.baseline_weeks == (35, 39)
        assert cfg.regimes.pre_baseline_max_week == 34
        assert cfg.audit.max_missing_frac == 0.20
        assert cfg.audit.near_duplicate_r == 0.999
        assert cfg.clean.impute_window == 25
        assert cfg.cusum.target_arl0 == 500
        assert (cfg.cusum.n_sim, cfg.cusum.max_run) == (3000, 8000)
        assert cfg.cusum.shift_ratio == 2.0
        assert cfg.sensors.watchlist_max == 40
        assert cfg.sensors.watchlist_regime_top == 25
        assert cfg.model.train_frac == 0.70
        assert cfg.model.n_perm == 60
        assert cfg.model.cv_folds == 10
        assert cfg.model.rolling_origin_fracs == (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85)
        assert cfg.model.triage_top_k_pct == (5, 10, 15, 20, 25, 30, 40, 50)
        assert (cfg.figures.format, cfg.figures.dpi) == ("png", 130)

    def test_partial_file_keeps_other_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        path.write_text("model:\n  n_perm: 5\n", encoding="utf-8")
        cfg = load_config(path)
        assert cfg.model.n_perm == 5
        assert cfg.model.train_frac == 0.70
        assert cfg.cusum == Config().cusum

    @pytest.mark.parametrize(
        "data, match",
        [
            ({"sede": 1}, "unknown top-level"),
            ({"model": {"n_perms": 5}}, r"unknown key\(s\) in model"),
            ({"model": {"n_perm": 5.5}}, "expected an integer"),
            ({"model": {"train_frac": "0.7"}}, "expected a number"),
            ({"regimes": {"baseline_weeks": [35]}}, "expected 2 items"),
            ({"regimes": {"baseline_weeks": [39, 35]}}, "ascending"),
            ({"figures": {"format": "jpg"}}, "figures.format"),
        ],
    )
    def test_invalid_config_raises(self, data: dict, match: str) -> None:
        with pytest.raises(ConfigError, match=match):
            config_from_dict(data)

    def test_integers_are_accepted_for_floats(self) -> None:
        assert config_from_dict({"cusum": {"target_arl0": 400}}).cusum.target_arl0 == 400.0

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nope.yaml")


@register_section("_test_section")
@dataclass(frozen=True)
class _TestSection:
    k: int = 3


class TestSections:
    def teardown_class(self) -> None:
        unregister_section("_test_section")

    def test_missing_section_falls_back_to_defaults(self) -> None:
        assert config_from_dict({}).section(_TestSection) == _TestSection()
        assert Config().section(_TestSection) == _TestSection()

    def test_section_is_parsed_and_validated(self) -> None:
        assert config_from_dict({"_test_section": {"k": 7}}).section(_TestSection).k == 7
        with pytest.raises(ConfigError, match="unknown key"):
            config_from_dict({"_test_section": {"j": 1}})

    def test_section_cannot_shadow_a_core_key(self) -> None:
        with pytest.raises(ConfigError, match="clashes"):
            register_section("model")(_TestSection)

    def test_unregistered_class_is_an_error(self) -> None:
        @dataclass(frozen=True)
        class Unregistered:
            pass

        with pytest.raises(ConfigError, match="not a registered"):
            Config().section(Unregistered)


# stage registry


class TestStageRegistry:
    def test_every_stage_exposes_the_contract(self) -> None:
        for stage in STAGES:
            assert isinstance(stage.NAME, str) and stage.NAME
            assert isinstance(stage.REQUIRES, tuple)
            assert callable(stage.run)

    def test_names_are_unique(self) -> None:
        names = [s.NAME for s in STAGES]
        assert len(names) == len(set(names))

    def test_every_requirement_is_produced_upstream(self) -> None:
        produced: set[str] = set()
        for stage in STAGES:
            missing = set(stage.REQUIRES) - produced
            assert not missing, f"{stage.NAME} needs {missing} before it runs"
            produced |= set(getattr(stage, "PRODUCES", ()))

    def test_no_artifact_is_produced_twice(self) -> None:
        seen: dict[str, str] = {}
        for stage in STAGES:
            for key in getattr(stage, "PRODUCES", ()):
                assert key not in seen, f"{key} produced by {seen.get(key)} and {stage.NAME}"
                seen[key] = stage.NAME

    def test_figures_and_export_stay_last(self) -> None:
        assert [s.NAME for s in STAGES[-2:]] == ["figures", "export"]

    def test_only_pulls_in_upstream_stages(self) -> None:
        names = [s.NAME for s in resolve_stages(["cusum"])]
        assert names == ["load", "audit", "clean", "cusum"]

    def test_unknown_stage_is_rejected(self) -> None:
        with pytest.raises(PipelineError, match="unknown stage"):
            resolve_stages(["nope"])


@pytest.mark.skipif(not (RAW / "secom.data").exists(), reason="needs data/raw/")
def test_partial_run_writes_nothing_but_produces_artifacts(tmp_path: Path) -> None:
    paths = Paths.from_root(tmp_path)
    paths = Paths(
        root=paths.root,
        raw=RAW,
        processed=paths.processed,
        powerbi=paths.powerbi,
        figures=paths.figures,
        reports=paths.reports,
    )
    ctx = PipelineContext(config=load_config(None), paths=paths, log=lambda _: None)
    ran = run_pipeline(ctx, only=["yield_charts"], skip_figures=True)
    assert ran == ["load", "audit", "clean", "yield_charts"]
    assert round(ctx.results["yield"]["sigma_z"], 3) == 1.463
    assert ctx["masks"]["base"].sum() == 717
    assert not (paths.reports / "results.json").exists()


# CLI


class TestCli:
    def test_stages_lists_every_stage_in_order(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["stages"]) == 0
        out = capsys.readouterr().out
        positions = [out.index(f" {s.NAME} ") for s in STAGES]
        assert positions == sorted(positions)

    def test_stages_verbose_shows_artifacts(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["stages", "-v"]) == 0
        assert "produces: raw" in capsys.readouterr().out

    def test_bad_config_exits_with_usage_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(yaml.safe_dump({"sede": 1}), encoding="utf-8")
        assert main(["run", "--root", str(tmp_path), "--config", str(bad)]) == 2
        assert "unknown top-level" in capsys.readouterr().err

    def test_unknown_stage_is_an_argparse_error(self) -> None:
        with pytest.raises(SystemExit):
            main(["run", "--only", "nope"])
