# Pipeline architecture

`secom run` executes an ordered list of stages against one shared
[`PipelineContext`][secom.pipeline.context.PipelineContext]. Each stage reads
named artifacts that earlier stages produced, writes its own artifacts and one
top-level key of `reports/results.json`, and saves its own figures. The registry
is the `STAGES` list in [`secom.pipeline`][secom.pipeline]; decision **D25** in
the [decision log](decisions.md) explains why the analysis is staged.

## Stages

The table and diagram below are read from the source when the site is built, so
they always match the code.

| # | Stage | Does | Reads | Produces |
|---|---|---|---|---|
{{STAGE_TABLE}}

Arrows run from the stage that produces an artifact to every stage that reads
it, labelled with the artifact keys (truncated after three).

{{STAGE_DIAGRAM}}

## The stage contract

A stage is one module in `src/secom/pipeline/stages/`:

```python
"""One-line summary: shown by `secom stages` and in the table above."""

from __future__ import annotations

from ..context import PipelineContext

NAME = "my_stage"
REQUIRES = ("clean", "is_fail")  # artifact keys read from ctx
PRODUCES = ("my_result",)  # artifact keys written to ctx


def run(ctx: PipelineContext) -> None:
    clean = ctx["clean"]  # raises a clear error if it is missing
    ...
    ctx.artifacts["my_result"] = result
    ctx.results["my_stage"] = {"headline": 0.5}  # lands in results.json
    ctx.save_figure(fig, "fig13_my_stage")  # no-op with --no-figures
```

Registering it is one line in `STAGES`. The rules, enforced by tests:

- **Names are unique**, and every key in `REQUIRES` is in the `PRODUCES` of an
  earlier stage.
- **`figures` and `export` stay last.** New analysis stages go after `model`.
  `export` writes `results.json`, so a new stage's key lands after the existing
  ones and the committed keys keep their order.
- **Outputs stay reproducible.** Adding a stage must leave every existing CSV and
  every existing `results.json` key byte-for-byte unchanged.

`--only` uses `REQUIRES` and `PRODUCES` to pull in upstream stages:
`secom run --only cusum` runs `load`, `audit`, `clean`, `yield_charts` and
`cusum`. An `--only` run that stops before `export` leaves the committed outputs
untouched.

## Figures

[`ctx.save_figure`][secom.pipeline.context.PipelineContext.save_figure] writes
`<figures>/<name>.<format>` and closes the figure. The format comes from
`figures.format` (or `--figure-format`), and the matplotlib backend is chosen
from it when the run starts: `png` uses Agg, `svg` uses the pure-Python SVG
backend, which needs no compiled extension. Use the shared palette in
[`secom.pipeline.plotting`][secom.pipeline.plotting] so new figures match the
report.

## Determinism

One seed (`seed` in the [configuration](configuration.md)) drives the model
fits, the CV splits, the CUSUM ARL simulation and the permutation test. On a
given platform and set of library versions, a clean run reproduces
`data/powerbi/`, `data/processed/` and `reports/results.json` byte-for-byte, run
after run.

Across platforms the outputs agree to floating-point rounding, not to the byte.
The [Docker image](getting-started.md#docker) pins the exact library versions,
and its Linux run differs from the committed Windows run in two ways only:

- **Last-digit rounding**, at most 6×10⁻¹⁶ absolute, in a few hundred CSV cells
  (model scores, q-values) and one confidence bound. The platforms' maths
  libraries round differently; every reported figure is unchanged.
- **File sizes.** The manifest and `results.json` record each CSV's size in
  bytes, and CSV line endings follow the platform (CRLF on Windows, LF on
  Linux).

The figures are pixel-identical.
