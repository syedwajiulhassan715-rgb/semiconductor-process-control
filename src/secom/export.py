"""Star schema CSV export for Power BI.

    dim_date ---< fact_run ---< fact_measurement >--- dim_sensor

    fact_yield_week, fact_gchart   (standalone reference tables)

Relationships are one-to-many, single direction (dimension -> fact). Shift is a
column on fact_run rather than its own dimension. powerbi/README.md has the
full model setup.

fact_measurement only has the watchlist sensors (at most 40; 35 on this data,
54,845 rows) instead of all 412. The full cleaned table is written out
separately.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_dim_date(runs: pd.DataFrame) -> pd.DataFrame:
    """Daily date table for the whole campaign with no missing days.

    DAX time intelligence needs every day present. has_production marks the
    days that actually had runs.
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
            "year_week": [f"{y}-W{w:02d}" for y, w in zip(iso["year"], iso["week"], strict=True)],
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
    """One row per raw sensor with its audit and screening results.

    Dropped sensors are kept too so you can see why they were dropped.
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
    """One row per production run."""
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
    # first run of each block of consecutive signals. Much easier to do here
    # than in DAX.
    sig = cusum["signal"].to_numpy()
    fact["cusum_episode_start"] = sig & ~np.concatenate(([False], sig[:-1]))
    fact["model_score"] = model_score if model_score is not None else np.nan
    return fact


def build_fact_measurement(
    clean: pd.DataFrame,
    watchlist: list[str],
    limits: pd.DataFrame,
) -> pd.DataFrame:
    """Watchlist readings in long format with Phase I limits and flags.

    robust_z is calculated here and not in DAX so the baseline doesn't change
    when someone uses a slicer.
    """
    lim = limits.set_index("sensor_id")
    long = clean.melt(
        id_vars=["run_id"],
        value_vars=watchlist,
        var_name="sensor_id",
        value_name="value",
    )
    long = long.merge(
        lim[
            [
                "baseline_center",
                "baseline_sigma",
                "baseline_lcl",
                "baseline_ucl",
                "baseline_median",
                "baseline_mad",
            ]
        ],
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
    """Write each table to CSV and return a manifest.

    Timestamps go out as ISO-8601, otherwise Power BI reads the day-first
    source dates as month-first.
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
