# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-10-23

### Added

- SPE limit investigation (D41): the Jackson–Mudholkar limit is exact on
  Gaussian data; the 7.1% false-alarm rate on TEP `d00_te` comes from a 24%
  rise in new-run SPE. `tep_validation.spe_limit_diagnostics` records the
  evidence in `tep_results.json`.
- Lower (improvement) Bernoulli CUSUM, `spc.bernoulli_cusum_lower` (D42), run
  on the real data in the `cusum` stage and shown in fig02.
- `cusum_robustness` pipeline stage and fig16:
  - p₀ re-estimation (D43): CUSUM redesigned over a p₀ sweep; at 4.27% the
    deployed chart's ARL is 210 runs, not 509.
  - Overdispersion simulation (D44): beta-binomial day effect calibrated to
    σ_z = 1.463; the deployed CUSUM's ARL₀ falls from 494 to 293 runs.
- Browser-based live demo of the replay monitor at `/demo/` on the docs site
  (D45): `engine.js` ports the replay engine to JavaScript, and
  `secom monitor --export DIR` writes the static page.
- The replay monitor alarms on the lower CUSUM; the g-chart is kept as
  context.
- Unit tests for the core pipeline stages, CLI, config loader, raw-data
  loading, Power BI export and data download, on small synthetic data, so
  they run in a fresh clone without `data/raw/`.
- Coverage configuration in `pyproject.toml` with a `fail_under` gate, and a
  coverage badge published by CI.
- `CITATION.cff` so the repository can be cited from GitHub.
- This changelog.

### Changed

- CI enforces the coverage threshold and writes a coverage badge on pushes to
  `main`.
- Recommendation: re-design the CUSUM on p₀ = 4.27% and quote its false-alarm
  cadence under overdispersion.

### Fixed

- `mspc.spe_limit_jackson_mudholkar` returns NaN for h₀ ≤ 0, where the closed
  form gave a lower-tail value, so the fit falls back to the Box limit. No
  model in this project reaches h₀ ≤ 0, so no reported number changes.

## [1.0.0] - 2026-10-23

First tagged release, reconstructed from the git history.

### Added

- Loading of the UCI SECOM files into a tidy frame with sensor ids S001-S590,
  `secom fetch` to download them, and a sensor audit (constant, sparse and
  near-duplicate sensors, informative missingness).
- Causal (look-back only) imputation and shift / ISO-week time context.
- Weekly yield charts: naive p-chart, Laney p'-chart, Phase I-centred limits,
  the regime test and a g-chart of runs between failures.
- Bernoulli CUSUM designed on the Phase I failure rate, with the threshold
  calibrated by simulation to a target in-control ARL.
- Sensor screens for drift, regime shift and failure association with
  Benjamini-Hochberg control, the time-confound gate, a sensor watchlist and
  Phase I individuals limits.
- Failure models evaluated with a forward chronological split against
  shuffled CV, rolling-origin checks, a permutation null and a triage curve.
- PCA multivariate SPC (T², SPE, contributions) with cross-validated limits.
- Detection-power simulation under known step shifts and a Bernoulli
  change-point analysis with permutation tests and a bootstrap interval.
- Online monitors and a replay session served by `secom monitor` as a local
  web page with a JSON API.
- Tennessee Eastman ground-truth validation (`secom fetch --dataset tep`,
  `secom validate tep`), checked against the PCA columns of Russell, Chiang &
  Braatz (2000) using the paper's threshold protocol.
- Power BI star schema export (`dim_date`, `dim_sensor`, `fact_run`,
  `fact_measurement`, `fact_yield_week`, `fact_gchart`) with ISO-8601 dates.
- Staged pipeline with a typed, frozen YAML configuration and the `secom`
  CLI (`fetch`, `run`, `stages`, `monitor`, `validate`).
- Engineering report, decision log (D1-D40), MkDocs Material documentation
  site, Docker image and GitHub Actions CI (ruff, mypy, pytest on Python
  3.11-3.13, coverage, CLI smoke test, Docker build).
- MIT license.

### Fixed

- Tests run from a clean clone (`pythonpath` set for pytest).
- Power BI documentation and stale exported artifacts.
- Fault 9 T² entry of the TEP benchmark table (0.994, not 0.981), and the
  validation now matches the original paper's threshold protocol.

[Unreleased]: https://github.com/syedwajiulhassan715-rgb/semiconductor-process-control/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/syedwajiulhassan715-rgb/semiconductor-process-control/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/syedwajiulhassan715-rgb/semiconductor-process-control/releases/tag/v1.0.0
