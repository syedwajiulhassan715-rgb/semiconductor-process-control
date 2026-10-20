<div align="center">

# Semiconductor Process Control

**Statistical process control and root-cause screening on 1,567 production runs of real fab data**

<p>
<a href="https://github.com/syedwajiulhassan715-rgb/semiconductor-process-control/actions/workflows/ci.yml"><img src="https://github.com/syedwajiulhassan715-rgb/semiconductor-process-control/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<img src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white" alt="Python 3.11 | 3.12 | 3.13">
<img src="https://img.shields.io/badge/tests-87-2E7D4F" alt="87 tests">
<a href="https://docs.astral.sh/ruff/"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
<img src="https://img.shields.io/badge/pipeline-deterministic%20%C2%B7%2023s-2E7D4F" alt="Deterministic, 23 seconds">
<img src="https://img.shields.io/badge/dataset-UCI%20SECOM-6E7B8B" alt="UCI SECOM">
<img src="https://img.shields.io/badge/BI-Power%20BI%20star%20schema-C43E1C" alt="Power BI star schema">
</p>

<p>
<a href="reports/ENGINEERING_REPORT.md"><b>Engineering report</b></a> ·
<a href="DECISIONS.md"><b>Decision log (D1–D24)</b></a> ·
<a href="reports/results.json"><b>results.json</b></a> ·
<a href="powerbi/README.md"><b>Power BI setup</b></a>
</p>

</div>

---

590 unnamed sensor signals, pass/fail from in-house line test, 89 days of 2008,
104 failures (6.64%). The conclusion is that this line needs a **monitoring
scheme, not a prediction model** — and the project is built to demonstrate that
rather than assert it.

Three things make it non-trivial, and each one is a place the obvious analysis
goes wrong:

- The standard chart **misses a threefold yield improvement** that is
  unambiguously present in the data.
- 16 of 17 apparent root causes are **a time confound**, not a cause.
- The failure-prediction result the published baselines report **does not
  survive an honest train/test split**.

---

## 1 · The yield change is real, and the standard chart cannot see it

The failure rate falls from **12.92% to 4.11%** at ISO week 35 — odds ratio
**3.46** (95% CI 2.26–5.30), Fisher **p = 2.0×10⁻⁹**.

<p align="center">
  <img src="reports/figures/fig01_yield_pchart.png" width="100%" alt="Weekly failure rate with naive p-chart limits and Laney p-prime limits">
</p>
<p align="center">
  <sub><b>The correction that erases the only alarm.</b> Week-to-week variation is 46% wider than
  binomial sampling explains (σ_z = 1.463, χ² p = 5.2×10⁻⁸), so the Laney p′ correction is
  warranted — and widening the limits by 1.46× absorbs the single flagged week (W31, circled).
  Naive chart: 1 alarm. Corrected chart: 0.</sub>
</p>

Worse, the chart is *structurally* blind to what actually happened. At this
failure rate a subgroup needs **n ≥ 127** before its lower limit clears zero,
and **7 of 14 weeks** fall short — so the chart cannot signal an improvement at
all. The data model surfaces that as `[Improvement Detectable]` rather than
leaving a user to discover it.

The right reading is not "the process is fine." It is that a proportion chart at
this grain cannot separate signal from process noise, and the variation has to
be attributed another way.

## 2 · A CUSUM sees it in half a day

<p align="center">
  <img src="reports/figures/fig02_cusum.png" width="100%" alt="Bernoulli CUSUM across the campaign, with alarms marked and the three regimes shaded">
</p>
<p align="center">
  <sub><b>The scheme that works.</b> Shewhart charts judge each subgroup alone, so they are worst at
  exactly what happened here — a sustained level shift. The Bernoulli CUSUM accumulates per-run
  log-likelihood and needs no subgrouping. Threshold h = 1.822 set <i>by simulation</i> for an
  in-control ARL of ~509 runs, not guessed.</sub>
</p>

| Window | Runs | Failures | Rate | Runs to signal | Days | Alarms |
|---|---:|---:|---:|---:|---:|---:|
| Pre-baseline (degraded) | 449 | 58 | 12.92% | **12** | **0.46** | 15 |
| Phase I baseline — self-check | 717 | 22 | 3.07% | 157 | 6.06 | 2 |
| Phase II (forward) | 401 | 24 | 5.99% | 73 | 3.29 | 4 |

The baseline self-check is the guard on the design: 2 alarms against ~1.4
expected is consistent with a genuinely stable window, so the baseline does not
secretly contain the excursion it is meant to detect.

## 3 · Of 590 sensors, exactly one survives

**71% of retained sensors drift with run order**, and the failure rate changes
over time — so drift alone manufactures association with failure. The gate
applied: a sensor must discriminate fail from pass *within* a stable regime,
where time is held constant.

<table>
<tr>
<td width="52%" valign="top">
  <img src="reports/figures/fig03_evidence_funnel.png" width="100%" alt="Evidence funnel from 590 sensors down to 1">
</td>
<td width="48%" valign="top">
  <img src="reports/figures/fig06_triage.png" width="100%" alt="Review effort versus failures captured, against a random-review diagonal">
</td>
</tr>
<tr>
<td valign="top"><sub><b>17 → 1.</b> 17 sensors associate with failure across the campaign. 16 have
substantial drift and a non-significant within-regime p-value: they are confounded, and reporting
them as root causes would be wrong.</sub></td>
<td valign="top"><sub><b>Triage is not fundable.</b> Best achievable lift is 1.54× random, at a 40%
review budget. Re-inspecting 40% of production to catch 62% of failures is not an operating
point anyone would approve.</sub></td>
</tr>
</table>

The single survivor is **S060** — drift ρ = −0.442, within-stable-regime
**p = 5.6×10⁻⁶**.

<p align="center">
  <img src="reports/figures/fig04_surviving_sensor.png" width="100%" alt="S060 across the campaign and its pass/fail separation within each regime">
</p>
<p align="center">
  <sub><b>The caveat travels with the finding.</b> S060 separates pass from fail in W29–34
  (δ = 0.21) and strongly in W35–39 (δ = 0.57) — but the separation is <i>absent</i> in W40–42
  (δ = −0.04, p = 0.76). Either the mechanism changed or the association was never causal. This
  is stated as a caveat on the recommendation, not omitted.</sub>
</p>

## 4 · Run-level prediction does not deploy

The published SECOM baselines use 10-fold CV — which trains on October to
predict August. With 71% drift and a failure rate that halves partway through, a
shuffled split lets a model learn *when* a run happened. A monitoring model can
only see the past, so the forward chronological split is the primary result.

<p align="center">
  <img src="reports/figures/fig05_protocol.png" width="100%" alt="Shuffled versus forward-chronological ROC AUC, and a permutation null distribution">
</p>
<p align="center">
  <sub><b>Left:</b> the 0.698 → 0.606 gap is leakage, not skill; balanced error goes 0.373 → 0.504,
  indistinguishable from a coin flip. <b>Right:</b> against a 60-run label permutation
  (null 0.487 ± 0.054) the observed 0.606 is still real signal — just nowhere near enough to
  act on.</sub>
</p>

| Protocol | Model | BER | Sensitivity | ROC AUC | PR AUC |
|---|---|---:|---:|---:|---:|
| Shuffled 10-fold | logistic L2 | 0.373 | 0.433 | 0.698 | 0.152 |
| Shuffled 10-fold | grad. boosting | 0.470 | 0.106 | 0.715 | 0.148 |
| **Forward chronological** | **logistic L2** | **0.504** | **0.038** | **0.606** | **0.073** |
| **Forward chronological** | grad. boosting | 0.502 | 0.000 | 0.565 | 0.072 |

Gradient boosting flags *nothing* (sensitivity 0.000). Two deliberately
different model families hit the same ceiling, so the limit is informational,
not a modelling failure. Rolling-origin evaluation across 8 cut points confirms
the forward number is not one unlucky cut: mean AUC 0.641 (sd 0.106).

---

## Why the numbers can be trusted

**Every figure is regenerated, never remembered.** `scripts/run_analysis.py`
writes every quoted statistic to `reports/results.json`, so each claim in the
write-up traces to a computed value.

**The pipeline is deterministic.** One seed (`20080719`) covers model fits, CV
splits, the CUSUM ARL simulation, and the permutation test. A clean re-run
reproduces every committed output byte-for-byte in **~23 s**.

**The choices that could quietly invalidate the result are pinned by tests** —
87 of them:

| Risk | Guard |
|---|---|
| Imputation leaking the future | `test_imputation_cannot_see_the_future` mutates every value after a gap and asserts the imputed value is unchanged |
| Overdispersion correction applied when unwarranted | `test_sigma_z_is_about_one_for_binomial_data` checks dispersion in both directions |
| The g-chart's inactive lower limit being forgotten | `test_lower_limit_is_inactive_at_realistic_failure_rates` |
| A CUSUM that alarms forever after one excursion | `test_reset_prevents_a_permanent_alarm` |
| FDR implementation drift | validated against `statsmodels` to 2×10⁻¹⁶ |
| A silent change in dataset shape | `io.load_raw` hard-errors unless the matrix is 1567×590 |

Where a published reference value exists, tests assert against *that* value
rather than against whatever the code currently returns.

**Thresholds are shown to be insensitive, not tuned.** Any missingness cutoff in
[0.175, 0.456] drops exactly the same 32 sensors. Re-running the near-duplicate
cut at 0.99 / 0.9999 moves retention 412 → 405 / 418 and changes no conclusion.

---

## Quick start

Python ≥ 3.11.

```bash
pip install -r requirements.txt
python scripts/00_fetch_data.py     # downloads SECOM into data/raw/
python scripts/run_analysis.py      # ~23 s -> results.json, figures, Power BI CSVs
pytest -q                           # 87 tests
```

`data/raw/` is gitignored; the fetch script is its only source, and it skips
files already present. Without it, the three real-data tests skip and the
remaining 84 still run.

### Development

```bash
pip install -e ".[dev]"     # editable install + pytest + ruff
ruff check . && ruff format --check .
pytest -q
```

The editable install also makes `secom` importable outside the test suite, from
a notebook for example. CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml))
runs lint plus the full suite, including the real-data tests, on Python 3.11,
3.12 and 3.13.

## Layout

```
src/secom/
  io.py       raw load; hard-errors if the matrix is not 1567x590
  audit.py    sensor disposition, missingness-vs-outcome testing
  clean.py    causal imputation (trailing rolling-median cascade)
  spc.py      p-chart / Laney p', g-chart, Bernoulli CUSUM + ARL calibration;
              also individuals/MR, EWMA and Nelson rules (tested library API)
  stats.py    rank-based screening, Cliff's delta, Benjamini-Hochberg
  model.py    shuffled vs forward-chronological evaluation
  export.py   Power BI star schema
scripts/
  00_fetch_data.py   dataset download
  run_analysis.py    end-to-end: raw in, star schema and figures out
powerbi/
  README.md          model setup: load, relationships, measures, validation
  measures.dax       control limits, baselines, interactive triage
  queries.pq         explicit Power Query types (locale + null-inference traps)
tests/               87 tests across spc / stats / audit / clean
reports/
  ENGINEERING_REPORT.md   the write-up
  results.json            every quoted number
  figures/                fig01..fig06
data/
  raw/         gitignored - fetched
  processed/   cleaned wide table, audit, imputation log, model results,
               confound check
  powerbi/     dim_date, dim_sensor, fact_run, fact_measurement,
               fact_yield_week, fact_gchart
```

### Delivered star schema

| Table | Rows | Cols | Grain |
|---|---:|---:|---|
| `dim_date` | 91 | 12 | one calendar day |
| `dim_sensor` | 590 | 32 | one raw signal — **all 590**, including the 178 excluded |
| `fact_run` | 1,567 | 15 | one production run |
| `fact_measurement` | 54,845 | 9 | run × watchlist sensor |
| `fact_yield_week` | 14 | 18 | one ISO week |
| `fact_gchart` | 103 | 9 | one failure-to-failure gap |

`dim_sensor` deliberately keeps every excluded sensor so *"why is this one not
on the chart?"* is answerable inside the report. Calculations are split between
Python and DAX on a stated rule (**D20**): control limits are DAX so they
respond to slicers; centre lines use `REMOVEFILTERS` so an excursion cannot
redefine normal as you drill into it; anything depending on the Phase I baseline
is precomputed in Python so filter context cannot silently move it. The DAX
measures were verified against the Python chart to 8×10⁻¹⁷; the setup and
the validation procedure are in [`powerbi/README.md`](powerbi/README.md).

---

## Scope boundaries

No capability analysis — SECOM publishes no specification limits and the sensors
are unnamed and unitless, so Cp/Cpk would be authoritative-looking and
meaningless (**D22**). No physical interpretation of any sensor; the signals are
anonymised, and naming one would be invention (**D23**).

What the analysis *cannot* establish — causation, whether week 35 was a process
change or a measurement change, S060's stability, generalisation beyond one
product and 89 days — is stated in
[§7 of the report](reports/ENGINEERING_REPORT.md#7-what-this-analysis-cannot-establish)
rather than left implicit.

## Recommendation

Run the **Bernoulli CUSUM against the week 35–39 baseline** as the live yield
monitor, with the g-chart alongside as an improvement detector. Keep the weekly
p′-chart for reporting, labelled with its blind spot. Put **S060** on a
watchlist with a scheduled re-test of its within-regime association — not on a
root-cause report. Do **not** deploy run-level failure prediction on this data.

## Data

McCann, M. & Johnston, A. (2008). *SECOM* [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C54305
