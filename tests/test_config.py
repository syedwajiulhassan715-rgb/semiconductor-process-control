"""Edge cases of the config loader not covered by test_pipeline.py."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from secom.config import (
    ConfigError,
    RegimeConfig,
    config_from_dict,
    load_config,
    register_section,
    unregister_section,
)


@dataclass(frozen=True)
class _Typed:
    flag: bool = False
    label: str = "x"
    pair: tuple[int, int] = (1, 2)
    extra: Any = None


@pytest.fixture
def typed_section() -> Iterator[type[_Typed]]:
    register_section("_typed")(_Typed)
    yield _Typed
    unregister_section("_typed")


class TestCoercion:
    def test_valid_values_round_trip(self, typed_section: type[_Typed]) -> None:
        cfg = config_from_dict(
            {"_typed": {"flag": True, "label": "y", "pair": [3, 4], "extra": {"a": 1}}}
        )
        assert cfg.section(typed_section) == _Typed(True, "y", (3, 4), {"a": 1})

    @pytest.mark.parametrize(
        ("data", "match"),
        [
            ({"flag": "yes"}, "expected true/false"),
            ({"label": 3}, "expected a string"),
            ({"pair": 5}, "expected a list"),
            ({"pair": [1, True]}, r"_typed\.pair\[1\]: expected an integer"),
        ],
    )
    def test_wrong_types_are_rejected(
        self, typed_section: type[_Typed], data: dict, match: str
    ) -> None:
        with pytest.raises(ConfigError, match=match):
            config_from_dict({"_typed": data})

    def test_bool_is_not_a_number(self) -> None:
        with pytest.raises(ConfigError, match="expected a number"):
            config_from_dict({"alpha": True})

    def test_section_must_be_a_mapping(self) -> None:
        with pytest.raises(ConfigError, match="expected a mapping"):
            config_from_dict({"model": [1, 2]})

    def test_variadic_tuple_elements_are_checked(self) -> None:
        with pytest.raises(ConfigError, match=r"model\.triage_top_k_pct\[1\]"):
            config_from_dict({"model": {"triage_top_k_pct": [5, "ten"]}})

    def test_empty_section_keeps_defaults(self) -> None:
        assert config_from_dict({"model": None}).model == config_from_dict(None).model


class TestValidation:
    def test_pre_baseline_must_precede_baseline(self) -> None:
        with pytest.raises(ConfigError, match="must precede"):
            RegimeConfig(pre_baseline_max_week=35, baseline_weeks=(35, 39))


class TestRegistration:
    def test_section_must_be_a_dataclass(self) -> None:
        class NotADataclass:
            pass

        with pytest.raises(TypeError, match="must be a dataclass"):
            register_section("_nope")(NotADataclass)

    def test_name_cannot_be_taken_twice(self, typed_section: type[_Typed]) -> None:
        @dataclass(frozen=True)
        class Other:
            k: int = 0

        with pytest.raises(ConfigError, match="already registered"):
            register_section("_typed")(Other)

    def test_reregistering_the_same_class_is_allowed(self, typed_section: type[_Typed]) -> None:
        assert register_section("_typed")(_Typed) is _Typed

    def test_unregistering_an_unknown_name_is_a_no_op(self) -> None:
        unregister_section("_never_registered")


class TestLoadConfig:
    def test_top_level_must_be_a_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        path.write_text("- 1\n- 2\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="top level must be a mapping"):
            load_config(path)

    def test_empty_file_gives_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        path.write_text("", encoding="utf-8")
        assert load_config(path) == load_config(None)


class TestCusumValidation:
    @pytest.mark.parametrize(
        ("data", "match"),
        [
            ({"shift_ratio": 1.0}, "shift_ratio must exceed 1"),
            ({"lower_ratio": 1.5}, r"lower_ratio must lie in \(0, 1\)"),
        ],
    )
    def test_ratios_must_point_the_right_way(self, data: dict, match: str) -> None:
        with pytest.raises(ConfigError, match=match):
            config_from_dict({"cusum": data})
