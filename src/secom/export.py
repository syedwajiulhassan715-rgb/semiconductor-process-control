"""Star-schema export for Power BI.

Power BI's engine (VertiPaq) compresses and scans a narrow fact table joined to
small dimensions far better than one wide table, and DAX time intelligence
requires a contiguous marked date table. The layout here is therefore:

    dim_date ---<  fact_run  >--- dim_shift
                      |
                      v
              fact_measurement >--- dim_sensor
              fact_yield_week

All relationships are single-direction one-to-many from the dimension to the
fact, which is what keeps measures predictable; nothing here needs bidirectional
filtering.

``fact_measurement`` is restricted to the watchlist sensors rather than all 412.
At 1567 runs a full unpivot is 645,604 rows, which Power BI handles, but a
watchlist of ~40 sensors gives 60k rows, refreshes instantly, and matches how a
sensor chart is actually used -- nobody scrolls 412 sensors. The full cleaned
table is exported separately for anyone who wants to widen the watchlist.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_dim_date(runs: pd.DataFrame) -> pd.DataFrame:
    """Contiguous daily date table spanning the campaign.

    Contiguous, including days the line did not run: a date table with holes
    breaks DAX time intelligence. ``has_production`` flags the real gaps so
    measures can exclude them rather than silently averaging zeros in.
    """
    start, end = runs["timestamp"].min().normalize(), runs["timestamp"].max().normalize()
    dates = pd.date_range(start, end, freq="D")
    produced = set(runs["timestamp"].dt.normalize())
    iso = dates.isocalendar()
    return pd.DataFrame(
        {
            "date_key": dates.strftime("%Y%m%d").astype(int),
            "date": dates,
            "year": dates.year,
            "month": dates.month,
            "month_name": dates.strftime("%b"),
            "day_of_month": dates.day,
            "day_name": dates.strftime("%a"),
            "iso_year": iso["year"].to_numpy(),
            "iso_week": iso["week"].to_numpy(),
            "year_week": [
                f"{y}-W{w:02d}" for y, w in zip(iso["year"], iso["week"], strict=True)
            ],
            "is_weekend": dates.dayofweek >= 5,
            "has_production": [d in produced for d in dates],
        }
    )


def build_dim_sensor(
    audit: pd.DataFrame,
    *,
    drift: pd.DataFrame,
    regime: pd.DataFrame,
    fail_all: pd.DataFrame,
    fail_stable: pd.DataFrame,
    limits: pd.DataFrame,
    watchlist: list[str],
) -> pd.DataFrame:
    """One row per raw sensor, carrying every disposition and evidence figure.

    Dropped sensors are kept in the dimension rather than filtered out. The
    reason a sensor is absent from the analysis is itself a reportable fact, and
    a reviewer should be able to see the whole 590 accounted for in one table.
    """
    dim = audit.reset_index().rename(columns={"sensor": "sensor_id"})
    dim = dim.merge(drift, on="sensor_id", how="left")
    for src, suffix in (
        (regime, "regime"),
        (fail_all, "fail_all"),
        (fail_stable, "fail_stable"),
    ):
        cols = src[["sensor", "cliffs_delta", "q_mannwhitney"]].rename(
            columns={
                "sensor": "sensor_id",
                "cliffs_delta": f"cliffs_delta_{suffix}",
                "q_mannwhitney": f"q_{suffix}",
            }
        )
        dim = dim.merge(cols, on="sensor_id", how="left")
    dim = dim.merge(limits, on="sensor_id", how="left")
    dim["is_watchlist"] = dim["sensor_id"].isin(watchlist)
    dim["evidence_tier"] = np.select(
        [
            dim["q_fail_stable"] < 0.05,
            dim["q_fail_all"] < 0.05,
            dim["q_regime"] < 0.05,
        ],
        [
            "1 - discriminates failures within a stable regime",
            "2 - associated with failure, time-confounded",
            "3 - shifted with the yield regime only",
        ],
        default="4 - no evidence",
    )
    dim.loc[~dim["retained"], "evidence_tier"] = "0 - excluded by data audit"
    return dim


def build_fact_run(
    clean: pd.DataFrame,
    *,
    cusum: pd.DataFrame,
    model_score: np.ndarray | None,
    phase: pd.Series,
) -> pd.DataFrame:
    """One row per production run: the grain of the whole model."""
    fact = pd.DataFrame(
        {
            "run_id": clean["run_id"],
            "timestamp": clean["timestamp"],
            "date_key": clean["timestamp"].dt.strftime("%Y%m%d").astype(int),
            "year_week": clean["year_week"],
            "shift": clean["shift"],
            "hour": clean["hour"],
            "run_index": clean["run_index"],
            "is_fail": clean["is_fail"],
            "fail_count": clean["is_fail"].astype(int),
            "minutes_since_prev": clean["minutes_since_prev"],
            "phase": phase.to_numpy(),
            "cusum": cusum["cusum"].to_numpy(),
            "cusum_signal": cusum["signal"].to_numpy(),
        }
    )
    # An "episode" is a contiguous block of signalling runs -- one investigation,
    # not one alarm per run. Detecting the transition needs a row-ordered lag,
    # which DAX can only do awkwardly and slowly over a 1567-row fact table, so
    # it is computed once here and exposed as a plain boolean for the measure
    # layer to SUM.
    sig = cusum["signal"].to_numpy()
    fact["cusum_episode_start"] = sig & ~np.concatenate(([False], sig[:-1]))
    fact["model_score"] = model_score if model_score is not None else np.nan
    return fact


def build_fact_measurement(
    clean: pd.DataFrame,
    watchlist: list[str],
    limits: pd.DataFrame,
) -> pd.DataFrame:
    """Unpivoted watchlist readings with Phase I limits and violation flags.

    ``robust_z`` is included as a pre-computed column rather than left to DAX:
    it depends on baseline median and MAD, which are Phase I constants, and
    recomputing them in DAX under arbitrary filter context would silently change
    the baseline every time a user sliced the report.
    """
    lim = limits.set_index("sensor_id")
    long = clean.melt(
        id_vars=["run_id"],
        value_vars=watchlist,
        var_name="sensor_id",
        value_name="value",
    )
    long = long.merge(
        lim[["baseline_center", "baseline_sigma", "baseline_lcl", "baseline_ucl",
             "baseline_median", "baseline_mad"]],
        left_on="sensor_id",
        right_index=True,
        how="left",
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        long["robust_z"] = (long["value"] - long["baseline_median"]) / long["baseline_mad"]
    long["beyond_limits"] = (long["value"] < long["baseline_lcl"]) | (
        long["value"] > long["baseline_ucl"]
    )
    return long.drop(columns=["baseline_median", "baseline_mad"])


def write_all(out_dir: str | Path, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Write every table as UTF-8 CSV and return a manifest.

    ISO-8601 timestamps and no index column, so Power Query's type inference
    lands on the right types without locale-dependent date parsing -- the raw
    SECOM timestamps are day-first, which Power BI will silently misread as
    month-first if handed the original format.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, table in tables.items():
        path = out_dir / f"{name}.csv"
        table.to_csv(path, index=False, date_format="%Y-%m-%dT%H:%M:%S")
        rows.append(
            {
                "table": name,
                "rows": len(table),
                "columns": table.shape[1],
                "bytes": path.stat().st_size,
                "file": path.name,
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "_manifest.csv", index=False)
    return manifest
