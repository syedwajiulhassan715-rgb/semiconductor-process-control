# Power BI model

Setup for the star schema in [`data/powerbi/`](../data/powerbi/). There are two
source files here:

| File | What it holds |
|---|---|
| [`queries.pq`](queries.pq) | One Power Query (M) query per table, with every column type declared explicitly |
| [`measures.dax`](measures.dax) | All DAX measures: base counts, Phase I baseline, Laney p′ limits, CUSUM, g-chart, sensor charts, triage |

The CSVs are produced by `python scripts/run_analysis.py`. They are committed, so
a fresh clone can build the model without running Python.

## 1. Load the tables

1. **Home → Transform data → Advanced Editor.**
2. Create a blank query named **`SourceFolder`** and paste the first block of
   `queries.pq`. Replace the placeholder with the absolute path to
   `data/powerbi` in your clone (no trailing backslash).
3. Create one blank query per remaining block, named exactly as its
   `// --- Query: <name>` header says: `fnLoadCsv`, `dim_date`, `dim_sensor`,
   `fact_run`, `fact_measurement`, `fact_yield_week`, `fact_gchart`. The DAX
   refers to these names.
4. **Close & Apply.**

Don't let Power BI auto-detect types. The queries declare them for two reasons:
day-first source dates would otherwise be read month-first, and `model_score`
(null outside the forward test window) would otherwise be typed as text.

## 2. Relationships

Create these in **Model view**. All are one-to-many, single cross-filter
direction (dimension filters fact):

| From (one) | To (many) |
|---|---|
| `dim_date[date_key]` | `fact_run[date_key]` |
| `fact_run[run_id]` | `fact_measurement[run_id]` |
| `dim_sensor[sensor_id]` | `fact_measurement[sensor_id]` |

```
dim_date ──< fact_run ──< fact_measurement >── dim_sensor

fact_yield_week   fact_gchart        (no relationships: reference tables)
```

`fact_yield_week` is the Python reference for the weekly p-chart and
`fact_gchart` is one row per failure-to-failure gap; neither shares a grain with
the dimensions. Shift is a column on `fact_run`, not a separate dimension.

Then:

- **Mark `dim_date` as the date table** (Table tools → Mark as date table →
  `date`). `DATESINPERIOD` and `DATEADD` in section 10 of the measures return
  wrong windows without it.
- **Sort `dim_date[year_week]` by `dim_date[year_week_sort]`** so weeks order
  chronologically.

## 3. Columns added in Power Query

These are derived on load, so they do not appear in the CSVs:

| Table | Column | Definition | Used by |
|---|---|---|---|
| `dim_date` | `year_week_sort` | `iso_year * 100 + iso_week` | `[Laney Sigma Z]` moving range order; week sort |
| `dim_sensor` | `missing_pct_label` | `missing_frac * 100`, 1 dp | report labels |
| `fact_run` | `pass_count` | `1 - fail_count` | `[Passes]`, `[Yield %]` |
| `fact_measurement` | `sigma_zone` | band of \|`robust_z`\| | zone formatting |

## 4. Measures

Create a table named `_Measures` (Enter data → one empty column → load, then
hide the column) and add each measure from `measures.dax` to it with
**Modeling → New measure**.

Section 9 (triage) needs a what-if parameter: **Modeling → New parameter →
Numeric range**, name `Review Budget`, min 5, max 100, increment 5, default 20.

## 5. Validating

At the unfiltered weekly grain, the DAX limits must reproduce the Python
reference:

1. Put `dim_date[year_week]` on a table visual with `[p-chart UCL]` and
   `[Baseline Fail Rate]`.
2. Next to it, show `fact_yield_week[year_week]`, `ucl_phase1` and
   `center_phase1`.
3. The values should agree to floating-point precision. The last check gave a
   maximum absolute difference of 8×10⁻¹⁷.

Compare against the `*_phase1` columns, not `ucl` / `center`. The plain
columns are centred on all 14 weeks (p̄ = 0.0664) and answer the retrospective
question. The DAX centres on the Phase I baseline (p̄ = 0.0307) and answers the
monitoring question. See decision **D10** in [`DECISIONS.md`](../DECISIONS.md).

The other sanity checks against `reports/results.json`:

| Measure | Expected |
|---|---:|
| `[Runs]` | 1,567 |
| `[Fails]` | 104 |
| `[Baseline Fail Rate]` | 0.0307 |
| `[Laney Sigma Z]` | 2.250 (`yield.sigma_z_phase1_centred`; z-scores are taken about the Phase I rate, so this is not the 1.463 quoted for the all-weeks chart) |
| `[Sensors Raw]` / `[Sensors Retained]` | 590 / 412 |
| `[Sensors On Watchlist]` | 35 |
