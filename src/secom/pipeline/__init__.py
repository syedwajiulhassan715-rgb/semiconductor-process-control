"""The analysis as an ordered list of stages.

Adding a stage
--------------
1. Create ``src/secom/pipeline/stages/<name>.py`` exposing::

       NAME = "<name>"
       REQUIRES = ("clean", "is_fail")   # artifact keys read from ctx
       PRODUCES = ("my_result",)        # artifact keys written to ctx
       def run(ctx: PipelineContext) -> None: ...

2. Add the module to :data:`STAGES` below. Nothing else needs to change.

Ordering rule: new analysis stages go after ``model`` and before ``figures``
and ``export``, which always stay last (a test enforces this). Each stage
writes one top-level key into ``ctx.results`` and owns its figures, saving
them itself with ``ctx.save_figure(fig, "figNN_name")``. The core ``figures``
stage draws fig01-fig06 only. ``export`` writes ``results.json`` last, so new
keys land after ``model`` while existing keys keep their order.

Artifacts produced by the core stages
-------------------------------------
load            raw: SecomData (all 590 sensors, file order)
audit           audit: per-sensor DataFrame indexed by sensor id, with
                ``disposition`` and ``retained``; missingness: Fisher tests
clean           clean: analysis frame (412 retained sensors, imputed, time
                context, RangeIndex); retained: list of sensor ids;
                impute_log; is_fail: bool ndarray; masks: dict of bool
                ndarrays ``pre`` / ``base`` / ``post`` (weeks <=34, 35-39,
                >=40); phase: Series of phase labels as exported
yield_charts    weekly: DataFrame year_week / n / d; naive, laney, laney_p1:
                PChartResult; gchart: GChartResult
cusum           cusum_all: BernoulliCusumResult over all runs;
                cusum_design: dict p0 / p1 / h / arl0
sensor_screen   drift (sensor_id / drift_rho / drift_q); regime, fail_all,
                fail_stable: screen_sensors frames; confound; watchlist:
                list of sensor ids; limits: Phase I limits per sensor
model           models_df; rolling_df; perm_null: ndarray; obs_auc: float;
                triage_df; forward_cut: int (first test row);
                forward_scores: ndarray of test-window scores
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Sequence
from types import ModuleType

from . import plotting
from .context import MissingArtifactError, Paths, PipelineContext
from .stages import (
    audit,
    changepoint,
    clean,
    cusum,
    export,
    figures,
    load,
    model,
    multivariate,
    sensor_screen,
    shift_simulation,
    yield_charts,
)

STAGES: list[ModuleType] = [
    load,
    audit,
    clean,
    yield_charts,
    cusum,
    sensor_screen,
    model,
    multivariate,
    shift_simulation,
    changepoint,
    figures,
    export,
]

__all__ = [
    "STAGES",
    "MissingArtifactError",
    "Paths",
    "PipelineContext",
    "PipelineError",
    "resolve_stages",
    "run_pipeline",
    "stage_names",
]


class PipelineError(RuntimeError):
    """Invalid stage selection or an unsatisfied stage dependency."""


def stage_names() -> list[str]:
    return [s.NAME for s in STAGES]


def _produces(stage: ModuleType) -> tuple[str, ...]:
    return tuple(getattr(stage, "PRODUCES", ()))


def resolve_stages(only: Iterable[str] | None = None) -> list[ModuleType]:
    """The stages to run, in pipeline order.

    With ``only``, the named stages plus every upstream stage they need.
    """
    if only is None:
        return list(STAGES)
    by_name = {s.NAME: s for s in STAGES}
    wanted = list(dict.fromkeys(only))
    unknown = [n for n in wanted if n not in by_name]
    if unknown:
        raise PipelineError(f"unknown stage(s): {', '.join(unknown)}")
    producer = {key: s.NAME for s in STAGES for key in _produces(s)}
    selected: set[str] = set()
    stack = wanted[:]
    while stack:
        name = stack.pop()
        if name in selected:
            continue
        selected.add(name)
        for key in by_name[name].REQUIRES:
            if key not in producer:
                raise PipelineError(f"stage {name!r} requires {key!r}, which no stage produces")
            stack.append(producer[key])
    return [s for s in STAGES if s.NAME in selected]


def _prefixed(log: Callable[[str], None], prefix: str) -> Callable[[str], None]:
    def emit(msg: str) -> None:
        log(f"{prefix}: {msg}")

    return emit


def run_pipeline(
    ctx: PipelineContext,
    only: Sequence[str] | None = None,
    skip_figures: bool = False,
) -> list[str]:
    """Run the selected stages against ``ctx``. Returns the stage names run.

    ``results.json`` is written by ``export``, so a run that excludes it
    leaves the committed report untouched.
    """
    stages = resolve_stages(only)
    if skip_figures:
        ctx.figures_enabled = False
        stages = [s for s in stages if s.NAME != "figures"]
    fmt = ctx.config.figures.format if ctx.figures_enabled else "svg"
    plotting.configure(fmt, ctx.config.figures.dpi)
    ctx.paths.ensure()

    log = ctx.log
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, stage in enumerate(stages, start=1):
            missing = [k for k in stage.REQUIRES if k not in ctx.artifacts]
            if missing:
                raise PipelineError(f"stage {stage.NAME!r} is missing {', '.join(missing)}")
            ctx.log = _prefixed(log, f"[{i}/{len(stages)}] {stage.NAME}")
            try:
                stage.run(ctx)
            finally:
                ctx.log = log
    return [s.NAME for s in stages]
