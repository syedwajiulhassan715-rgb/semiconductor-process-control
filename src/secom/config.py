"""Analysis configuration: every tunable constant in one typed, frozen object.

``config/default.yaml`` mirrors the defaults below exactly (a test asserts it),
so the YAML file documents the analysis and the dataclasses are the schema.
Unknown keys raise :class:`ConfigError` rather than being ignored, so a typo in
a threshold can't silently fall back to the default.

Adding a section for a new stage
--------------------------------
Define a frozen dataclass and register it under a top-level YAML key::

    from dataclasses import dataclass
    from secom.config import register_section

    @register_section("mspc")
    @dataclass(frozen=True)
    class MSPCConfig:
        n_components: int = 10

Read it in the stage with ``ctx.config.section(MSPCConfig)``. If the YAML file
has no ``mspc:`` block, the section's defaults are used, so existing config
files keep working. Registration happens at import time, and ``load_config``
imports :mod:`secom.pipeline` (and with it every stage module) first. Define
the section in the stage module itself, or in a module that the stage imports.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeVar, cast, get_args, get_origin, get_type_hints

import yaml

T = TypeVar("T")

FIGURE_FORMATS = ("png", "svg")


class ConfigError(ValueError):
    """Raised for unknown keys, wrong types or invalid values in a config file."""


@dataclass(frozen=True)
class AuditConfig:
    #: Sensors missing more than this are dropped; any value in [0.175, 0.456]
    #: drops the same 32 sensors (D2).
    max_missing_frac: float = 0.20
    #: |r| at or above which a surviving sensor is a near-duplicate (D4).
    near_duplicate_r: float = 0.999


@dataclass(frozen=True)
class CleanConfig:
    #: Trailing rolling-median window for causal imputation, in runs (D5).
    impute_window: int = 25


@dataclass(frozen=True)
class RegimeConfig:
    #: Last ISO week of the degraded pre-baseline period.
    pre_baseline_max_week: int = 34
    #: Inclusive ISO week range of the Phase I baseline (D7).
    baseline_weeks: tuple[int, int] = (35, 39)

    def __post_init__(self) -> None:
        lo, hi = self.baseline_weeks
        if lo > hi:
            raise ConfigError(f"regimes.baseline_weeks must be ascending, got {lo}..{hi}")
        if self.pre_baseline_max_week >= lo:
            raise ConfigError("regimes.pre_baseline_max_week must precede the baseline weeks")


@dataclass(frozen=True)
class CusumConfig:
    #: Out-of-control rate as a multiple of the Phase I rate (p1 = ratio * p0).
    shift_ratio: float = 2.0
    #: In-control average run length the threshold h is calibrated to (D12).
    target_arl0: float = 500.0
    #: Simulated in-control paths used for the calibration.
    n_sim: int = 3000
    #: Length of each simulated path, in runs.
    max_run: int = 8000
    #: The lower (improvement) CUSUM is designed for p1 = lower_ratio * p0.
    lower_ratio: float = 0.5

    def __post_init__(self) -> None:
        if self.shift_ratio <= 1:
            raise ConfigError("cusum.shift_ratio must exceed 1 (the upper CUSUM detects a rise)")
        if not 0 < self.lower_ratio < 1:
            raise ConfigError(
                "cusum.lower_ratio must lie in (0, 1) (the lower CUSUM detects a fall)"
            )


@dataclass(frozen=True)
class SensorConfig:
    #: Regime movers added to the watchlist, ranked by q-value (D19).
    watchlist_regime_top: int = 25
    #: Hard cap on watchlist size.
    watchlist_max: int = 40
    #: |Cliff's delta| above which a regime shift counts as a large effect (D14).
    large_effect_delta: float = 0.5


@dataclass(frozen=True)
class ModelConfig:
    #: Fraction of runs, in time order, used to train the forward model (D16).
    train_frac: float = 0.70
    #: Folds for the shuffled-CV contrast.
    cv_folds: int = 10
    #: Label permutations for the forward-split null distribution.
    n_perm: int = 60
    #: Training fractions for the rolling-origin robustness check.
    rolling_origin_fracs: tuple[float, ...] = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85)
    #: Review budgets (% of runs) for the triage capture curve.
    triage_top_k_pct: tuple[int, ...] = (5, 10, 15, 20, 25, 30, 40, 50)


@dataclass(frozen=True)
class FigureConfig:
    #: ``png`` (committed figures) or ``svg`` (pure-Python backend).
    format: str = "png"
    dpi: int = 130

    def __post_init__(self) -> None:
        if self.format not in FIGURE_FORMATS:
            raise ConfigError(
                f"figures.format must be one of {FIGURE_FORMATS}, got {self.format!r}"
            )


@dataclass(frozen=True)
class Config:
    """The full analysis configuration. Build it with :func:`load_config`."""

    #: One seed for model fits, CV splits, CUSUM simulation and permutations.
    seed: int = 20080719
    #: Significance level for every FDR-controlled screen.
    alpha: float = 0.05
    audit: AuditConfig = field(default_factory=AuditConfig)
    clean: CleanConfig = field(default_factory=CleanConfig)
    regimes: RegimeConfig = field(default_factory=RegimeConfig)
    cusum: CusumConfig = field(default_factory=CusumConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    figures: FigureConfig = field(default_factory=FigureConfig)
    #: Registered extension sections by YAML key; see the module docstring.
    sections: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def section(self, cls: type[T]) -> T:
        """The registered section of type ``cls``, or its defaults if absent."""
        name = _SECTION_NAMES.get(cls)
        if name is None:
            raise ConfigError(f"{cls.__name__} is not a registered config section")
        value = self.sections.get(name)
        return cast(T, value) if value is not None else cls()


_SECTIONS: dict[str, type[Any]] = {}
_SECTION_NAMES: dict[type[Any], str] = {}
_CORE_KEYS = frozenset(f.name for f in fields(Config) if f.name != "sections")


def register_section(name: str) -> Callable[[type[T]], type[T]]:
    """Class decorator registering a dataclass as top-level config section ``name``."""

    def deco(cls: type[T]) -> type[T]:
        if not is_dataclass(cls):
            raise TypeError(f"{cls.__name__} must be a dataclass to be a config section")
        if name in _CORE_KEYS:
            raise ConfigError(f"section name {name!r} clashes with a core config key")
        existing = _SECTIONS.get(name)
        if existing is not None and existing.__qualname__ != cls.__qualname__:
            raise ConfigError(f"section {name!r} is already registered to {existing.__name__}")
        _SECTIONS[name] = cls
        _SECTION_NAMES[cls] = name
        return cls

    return deco


def unregister_section(name: str) -> None:
    """Remove a registered section (for tests)."""
    cls = _SECTIONS.pop(name, None)
    if cls is not None:
        _SECTION_NAMES.pop(cls, None)


def _coerce(tp: Any, value: Any, where: str) -> Any:
    if isinstance(tp, type) and is_dataclass(tp):
        return _build(tp, value, where)
    origin = get_origin(tp)
    if origin is tuple:
        if not isinstance(value, list | tuple):
            raise ConfigError(f"{where}: expected a list, got {type(value).__name__}")
        args = get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(args[0], v, f"{where}[{i}]") for i, v in enumerate(value))
        if len(value) != len(args):
            raise ConfigError(f"{where}: expected {len(args)} items, got {len(value)}")
        return tuple(
            _coerce(a, v, f"{where}[{i}]") for i, (a, v) in enumerate(zip(args, value, strict=True))
        )
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true/false, got {value!r}")
        return value
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a string, got {value!r}")
        return value
    return value


def _build(cls: type[T], data: Any, where: str) -> T:
    if data is None:
        return cls()
    if not isinstance(data, Mapping):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    hints = get_type_hints(cls)
    allowed = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(f"unknown key(s) in {where or 'config'}: {', '.join(map(str, unknown))}")
    kwargs = {k: _coerce(hints[k], v, f"{where}.{k}" if where else k) for k, v in data.items()}
    return cls(**kwargs)


def config_from_dict(data: Mapping[str, Any] | None) -> Config:
    """Build a :class:`Config` from an already-parsed mapping."""
    # Stage modules register their sections on import.
    importlib.import_module("secom.pipeline")
    data = dict(data or {})
    unknown = sorted(set(data) - _CORE_KEYS - set(_SECTIONS))
    if unknown:
        raise ConfigError(f"unknown top-level config key(s): {', '.join(map(str, unknown))}")
    core = {k: v for k, v in data.items() if k in _CORE_KEYS}
    sections = {name: _build(cls, data.get(name), name) for name, cls in sorted(_SECTIONS.items())}
    cfg = _build(Config, core, "")
    return Config(
        **{f.name: getattr(cfg, f.name) for f in fields(Config) if f.name != "sections"},
        sections=MappingProxyType(sections),
    )


def load_config(path: str | Path | None = None) -> Config:
    """Load a YAML config file, or the built-in defaults when ``path`` is None."""
    if path is None:
        return config_from_dict(None)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is not None and not isinstance(data, Mapping):
        raise ConfigError(f"{path}: top level must be a mapping")
    return config_from_dict(data)
