<div align="center">

# Semiconductor Process Control

**Statistical process control and root-cause screening on 1,567 production runs of real fab data**

<p>
<a href="https://github.com/syedwajiulhassan715-rgb/semiconductor-process-control/actions/workflows/ci.yml"><img src="https://github.com/syedwajiulhassan715-rgb/semiconductor-process-control/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<a href="https://github.com/syedwajiulhassan715-rgb/semiconductor-process-control/actions/workflows/docs.yml"><img src="https://github.com/syedwajiulhassan715-rgb/semiconductor-process-control/actions/workflows/docs.yml/badge.svg" alt="Docs"></a>
<img src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white" alt="Python 3.11 | 3.12 | 3.13">
<img src="https://img.shields.io/badge/tests-254-2E7D4F" alt="254 tests">
<a href="https://docs.astral.sh/ruff/"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
<img src="https://img.shields.io/badge/types-mypy-2E7D4F" alt="mypy">
<img src="https://img.shields.io/badge/docker-ready-2496ED?logo=docker&logoColor=white" alt="Docker">
<img src="https://img.shields.io/badge/dataset-UCI%20SECOM-6E7B8B" alt="UCI SECOM">
<img src="https://img.shields.io/badge/BI-Power%20BI%20star%20schema-C43E1C" alt="Power BI star schema">
</p>

<p>
<a href="reports/ENGINEERING_REPORT.md"><b>Engineering report</b></a> ·
<a href="DECISIONS.md"><b>Decision log (D1–D40)</b></a> ·
<a href="https://syedwajiulhassan715-rgb.github.io/semiconductor-process-control/"><b>Documentation</b></a> ·
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

The project then tests its own conclusions: it measures detection power by
simulation, locates the yield change statistically, tries the multivariate
chart a fab would reach for next, and validates the implementation on a
benchmark where the faults are known.

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
and **7 of 14 weeks** fall short. Centred on the Phase I rate, the threshold is
284 and **every** week falls short, so the chart cannot signal an improvement at
all. The data model surfaces that as `[Improvement Detectable]` rather than
leaving a user to discover it.

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

### Measured, not argued

Simulating step shifts of known size through the real chart designs puts
numbers on the comparison. Held to the p-chart's own false-alarm risk, the
CUSUM signals a doubling of the failure rate within 400 runs in **70.0%** of
streams; the p-chart does in **49.6%**, and the Laney p′ chart in 12.8%. For an
improvement, the p-chart and Laney p′ **never** signal.

<p align="center">
  <img src="reports/figures/fig10_detection_delay.png" width="100%" alt="Median runs to first signal against the shift ratio, for each charting scheme">
</p>

### Where the change actually happened

The exact change-point estimate puts the shift at **run 375 (21 Aug, W34)**,
14.17% before and 4.27% after (permutation p = 0.0005). Its 95% interval,
runs 280–463, contains the week-35 boundary used for the baseline, so the
baseline choice holds. There is no second change. That carries a caveat: the
sustained post-change rate is 4.27%, so the 3.07% baseline may be optimistic.

<p align="center">
  <img src="reports/figures/fig12_changepoint.png" width="100%" alt="Profile likelihood ratio over every split point, with the estimate, its interval and the week-35 boundary">
</p>

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

### The multivariate chart does not rescue it

A PCA model on the Phase I baseline (411 sensors, 81 components) with Hotelling
T² and SPE is how fab fault-detection systems usually watch hundreds of
sensors. Here the textbook F and Jackson–Mudholkar limits alarm on **15.8%** of
baseline runs against a nominal ≤ 2%, and on 86% of the degraded period. With
limits calibrated out of fold, the chart flags 6.9% of degraded-period runs,
mostly outliers on a handful of heavy-tailed sensors; S060 ranks 259th of 411.
It is not adopted as the yield monitor.

<p align="center">
  <img src="reports/figures/fig07_mspc_t2_spe.png" width="100%" alt="Hotelling T-squared and SPE across the campaign with cross-validated and theoretical limits">
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

## 5 · Validated where the answer is known

SECOM has no fault labels, so nothing above can show the code detects a fault
it should. The Tennessee Eastman benchmark has 21 faults injected at known
times. Run unchanged on it, this repo's PCA code reproduces the published
T² detection rates to a mean gap of **0.027** (correlation 0.999), catches
fault 1, sees fault 4 only in SPE, and leaves the known-undetectable faults 3,
9 and 15 at the false-alarm rate.

<p align="center">
  <img src="reports/figures/fig14_tep_calibration.png" width="85%" alt="False-alarm rate of the textbook limits on Tennessee Eastman and SECOM against the nominal alpha">
</p>
<p align="center">
  <sub><b>The same code, two datasets.</b> In-sample, the textbook T² limit alarms on 0.6% of
  Tennessee Eastman runs against a nominal 1%, and on 12.0% of SECOM runs. The calibration
  failure in §3 is a property of SECOM's heavy-tailed data, not a bug. Benchmark values are
  transcribed from secondary sources (D38).</sub>
</p>

## 6 · The scheme, operating

`secom monitor` replays the campaign run by run the way a line engineer would
see it: the CUSUM against its threshold, alarms that stay open until
acknowledged, the g-chart alongside, and S060 on watch. The online CUSUM
reproduces the batch computation exactly. The server uses only the standard
library.

<p align="center">
  <img src="docs/img/monitor.png" width="100%" alt="Replay monitor at run 1,310: CUSUM trace with alarms, g-chart, S060 watchlist panel and the alarm log">
</p>

---

## Why the numbers can be trusted

**Every figure is regenerated, never remembered.** `secom run` writes every
quoted statistic to `reports/results.json`, and `secom validate tep` writes the
benchmark numbers to `reports/validation/tep_results.json`. Every threshold
behind those numbers lives in one file,
[`config/default.yaml`](config/default.yaml).

**The pipeline is deterministic.** One seed (`20080719`) covers model fits, CV
splits, every simulation and every permutation test. On a given platform a
re-run reproduces every committed output byte for byte; between Windows and
Linux the numbers agree to floating-point rounding (at most 6×10⁻¹⁶). The
committed outputs come from the Docker image.

**The choices that could quietly invalidate the result are pinned by tests** —
254 of them:

| Risk | Guard |
|---|---|
| Imputation leaking the future | `test_imputation_cannot_see_the_future` mutates every value after a gap and asserts the imputed value is unchanged |
| Overdispersion correction applied when unwarranted | `test_sigma_z_is_about_one_for_binomial_data` checks dispersion in both directions |
| The g-chart's inactive lower limit being forgotten | `test_lower_limit_is_inactive_at_realistic_failure_rates` |
| A CUSUM that alarms forever after one excursion | `test_reset_prevents_a_permanent_alarm` |
| The live monitor drifting from the analysis | `test_reproduces_batch_cusum_exactly` |
| Simulated detection power being wrong | `test_simulation_matches_exact_binomial` checks every Shewhart case against exact values |
| A change-point method that cannot find a change | `test_recovers_a_known_split_and_covers_it` |
| A PCA implementation error passing unnoticed | `test_fault_1_t2_detection_matches_the_benchmark`, `test_known_undetectable_faults_stay_near_the_false_alarm_rate` |
| FDR implementation drift | validated against `statsmodels` to 2×10⁻¹⁶ |
| A silent change in dataset shape | `io.load_raw` hard-errors unless the matrix is 1567×590 |

Where a published reference value exists, tests assert against *that* value
rather than against whatever the code currently returns.

**Thresholds are shown to be insensitive, not tuned.** Any missingness cutoff in
[0.175, 0.456] drops exactly the same 32 sensors. Re-running the near-duplicate
cut at 0.99 / 0.9999 moves retention 412 → 405 / 418 and changes no conclusion.
The PCA verdict holds across 30–150 components.

---

## Quick start

Python ≥ 3.11.

```bash
pip install -e .
secom fetch                 # downloads SECOM into data/raw/
secom run                   # ~40 s -> results.json, figures, Power BI CSVs
secom monitor --open        # replay monitor at http://127.0.0.1:8050
```

Or with Docker, which needs nothing installed locally:

```bash
docker build -t secom-process-control .
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/reports:/app/reports" secom-process-control
docker run --rm -p 8050:8050 secom-process-control monitor --host 0.0.0.0
```

Run from the repository root (or pass `--root`). `data/raw/` is gitignored;
`secom fetch` is its only source, and it skips files already present.

### Commands

```bash
secom stages                     # list the pipeline stages in order
secom run --only cusum           # cusum plus the stages it depends on
secom run --no-figures           # everything except figure rendering
secom run --figure-format svg    # pure-Python backend, no compiled Agg needed
secom run --config my.yaml       # rerun under different thresholds
secom fetch --dataset tep        # Tennessee Eastman benchmark files
secom validate tep               # the §5 validation
secom monitor --speed 50         # replay at 50 runs per second
```

A custom config only needs the keys it changes, and an unknown key or a wrong
type is an error rather than a silent fallback. Only the `export` stage writes
`results.json` and the CSVs, so an `--only` run that stops short of it leaves
the committed outputs untouched.

### Development

```bash
pip install -e ".[dev]"     # pytest, pytest-cov, ruff, mypy, pre-commit
pre-commit install          # ruff + hygiene hooks on every commit
ruff check . && ruff format --check .
mypy                        # type-checks src/secom
pytest -q --cov=secom       # 254 tests

pip install -e ".[docs]"    # MkDocs Material + mkdocstrings
mkdocs serve                # docs site at http://127.0.0.1:8000
```

Without the raw data, 25 real-data tests skip and the remaining 229 still run.
CI runs lint, mypy, the full suite with coverage on Python 3.11–3.13, a
`secom run` and `secom validate tep` smoke test, and a Docker build. The docs
workflow builds the site with `--strict` and deploys it to GitHub Pages from
`main`.

**Adding an analysis stage** is one module in `src/secom/pipeline/stages/` plus
one line in `STAGES`. The contract (artifacts each stage reads and writes,
config sections, figures) is in the docstring of
[`src/secom/pipeline/__init__.py`](src/secom/pipeline/__init__.py); see
decision **D25**.

## Layout

```
config/
  default.yaml   every analysis constant, with its decision number
src/secom/
  cli.py         `secom` command: fetch / run / stages / validate / monitor
  config.py      typed, validated config loader (frozen dataclasses)
  pipeline/      staged pipeline: context, registry, plotting style
    stages/      load, audit, clean, yield_charts, cusum, sensor_screen, model,
                 multivariate, shift_simulation, changepoint, figures, export
  fetch.py       dataset download (SECOM, Tennessee Eastman)
  io.py          raw load; hard-errors if the matrix is not 1567x590
  audit.py       sensor disposition, missingness-vs-outcome testing
  clean.py       causal imputation (trailing rolling-median cascade)
  spc.py         p-chart / Laney p', g-chart, Bernoulli CUSUM + ARL calibration;
                 also individuals/MR, EWMA and Nelson rules
  stats.py       rank-based screening, Cliff's delta, Benjamini-Hochberg
  model.py       shuffled vs forward-chronological evaluation
  mspc.py        PCA monitoring: Hotelling T², SPE, contributions, CV limits
  power.py       detection-power simulation against exact Shewhart values
  changepoint.py Bernoulli change-point MLE, permutation tests, segmentation
  tep.py, tep_validation.py   Tennessee Eastman loader and validation
  monitor.py, monitor_server.py, monitor_static/   replay monitor
  export.py      Power BI star schema
scripts/         wrappers for `secom fetch` and `secom run`
powerbi/
  README.md      model setup: load, relationships, measures, validation
  measures.dax   control limits, baselines, interactive triage
  queries.pq     explicit Power Query types (locale + null-inference traps)
docs/            documentation site pages; docs/_gen builds the config,
                 architecture and API pages and includes the report
mkdocs.yml       documentation site configuration
Dockerfile, docker/   image for the pipeline and monitor; pinned versions
tests/           254 tests
reports/
  ENGINEERING_REPORT.md   the write-up
  results.json            every quoted number
  validation/             Tennessee Eastman results
  figures/                fig01..fig15
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
change or a measurement change, S060's stability, the true in-control rate,
whether the monitoring would catch real fab faults, generalisation beyond one
product and 89 days — is stated in
[§12 of the report](reports/ENGINEERING_REPORT.md#12-what-this-analysis-cannot-establish)
rather than left implicit.

## Recommendation

Run the **Bernoulli CUSUM against the week 35–39 baseline** as the live yield
monitor, and re-estimate its baseline rate as post-change data accumulates. For
improvement, use a lower Bernoulli CUSUM with the g-chart as its readable
companion. Keep the weekly p′-chart for reporting, labelled with its blind
spot. Put **S060** on a watchlist with a scheduled re-test of its within-regime
association — not on a root-cause report. Do **not** adopt the PCA chart as the
yield monitor, and do **not** deploy run-level failure prediction on this data.

## Data

McCann, M. & Johnston, A. (2008). *SECOM* [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C54305

Tennessee Eastman Process data: Braatz group files, as mirrored at
[camaramm/tennessee-eastman-profBraatz](https://github.com/camaramm/tennessee-eastman-profBraatz);
fetched on demand and never committed.
