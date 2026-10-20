"""Read the raw SECOM files."""

from __future__ import annotations

from ...io import load_raw
from ..context import PipelineContext

NAME = "load"
REQUIRES: tuple[str, ...] = ()
PRODUCES = ("raw",)


def run(ctx: PipelineContext) -> None:
    data = load_raw(ctx.paths.raw)
    runs = data.runs
    ctx.results["dataset"] = {
        "n_runs": int(len(runs)),
        "n_sensors_raw": len(data.sensors),
        "n_fail": int(runs["is_fail"].sum()),
        "fail_rate": float(runs["is_fail"].mean()),
        "start": str(runs["timestamp"].min()),
        "end": str(runs["timestamp"].max()),
        "span_days": int((runs["timestamp"].max() - runs["timestamp"].min()).days),
        "cell_missing_rate": float(runs[data.sensors].isna().to_numpy().mean()),
    }
    ctx.artifacts["raw"] = data
    ctx.log(f"{len(runs)} runs x {len(data.sensors)} sensors")
