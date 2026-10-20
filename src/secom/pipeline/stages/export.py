"""Write the Power BI star schema, the processed tables and results.json.

Always the last stage: results.json collects every key the earlier stages set.
"""

from __future__ import annotations

import json

from ...export import (
    build_dim_date,
    build_dim_sensor,
    build_fact_measurement,
    build_fact_run,
    write_all,
)
from ..context import PipelineContext

NAME = "export"
REQUIRES = (
    "clean",
    "audit",
    "impute_log",
    "phase",
    "naive",
    "laney",
    "laney_p1",
    "gchart",
    "cusum_all",
    "drift",
    "regime",
    "fail_all",
    "fail_stable",
    "confound",
    "watchlist",
    "limits",
    "models_df",
    "forward_cut",
    "forward_scores",
)
PRODUCES = ("tables", "manifest")

DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def run(ctx: PipelineContext) -> None:
    clean, watch, paths = ctx["clean"], ctx["watchlist"], ctx.paths
    naive, laney, laney_p1 = ctx["naive"], ctx["laney"], ctx["laney_p1"]

    dim_sensor = build_dim_sensor(
        ctx["audit"],
        drift=ctx["drift"],
        regime=ctx["regime"],
        fail_all=ctx["fail_all"],
        fail_stable=ctx["fail_stable"],
        limits=ctx["limits"],
        watchlist=watch,
    )
    fact_run = build_fact_run(
        clean, cusum=ctx["cusum_all"].table, model_score=None, phase=ctx["phase"]
    )
    fact_run.loc[fact_run.index[ctx["forward_cut"] :], "model_score"] = ctx["forward_scores"]
    yield_week = laney.table.rename(columns={"subgroup": "year_week"}).copy()
    yield_week["naive_ucl"] = naive.table["ucl"].to_numpy()
    yield_week["naive_violation"] = naive.table["violation"].to_numpy()
    # Phase I centred limits (what the DAX should match)
    yield_week["center_phase1"] = laney_p1.table["center"].to_numpy()
    yield_week["lcl_phase1"] = laney_p1.table["lcl"].to_numpy()
    yield_week["ucl_phase1"] = laney_p1.table["ucl"].to_numpy()
    yield_week["violation_phase1"] = laney_p1.table["violation"].to_numpy()

    tables = {
        "dim_date": build_dim_date(clean),
        "dim_sensor": dim_sensor,
        "fact_run": fact_run,
        "fact_measurement": build_fact_measurement(clean, watch, ctx["limits"]),
        "fact_yield_week": yield_week,
        "fact_gchart": ctx["gchart"].table,
    }
    manifest = write_all(paths.powerbi, tables)
    clean.to_csv(paths.processed / "secom_clean_wide.csv", index=False, date_format=DATE_FORMAT)
    ctx["audit"].to_csv(paths.processed / "sensor_audit.csv")
    ctx["models_df"].to_csv(paths.processed / "model_results.csv", index=False)
    ctx["confound"].to_csv(paths.processed / "confound_check.csv", index=False)
    ctx["impute_log"].to_csv(paths.processed / "imputation_log.csv", index=False)

    ctx.results["watchlist"] = watch
    ctx.results["exports"] = manifest.to_dict("records")
    (paths.reports / "results.json").write_text(
        json.dumps(ctx.results, indent=2, default=str), encoding="utf-8"
    )
    ctx.artifacts["tables"] = tables
    ctx.artifacts["manifest"] = manifest
    ctx.log(f"star schema -> {paths.powerbi}, processed -> {paths.processed}")
    ctx.log("results.json written\n" + manifest.to_string(index=False))
