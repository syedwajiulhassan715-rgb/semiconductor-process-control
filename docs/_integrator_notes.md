# Integrator notes (feat/docs-docker) — apply, then delete this file

## README changes wanted

1. **Badges:** add a Docs badge linking to the site,
   `https://syedwajiulhassan715-rgb.github.io/semiconductor-process-control/`,
   and add "Documentation" to the top link row.
2. **Quick start:** add a Docker block after the pip block:

   ```bash
   docker build -t secom-process-control .
   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/reports:/app/reports" secom-process-control
   docker run --rm -p 8050:8050 secom-process-control monitor --host 0.0.0.0
   ```

3. **Development:** add `pip install -e ".[docs]"`, then `mkdocs serve` and
   `mkdocs build --strict`.
4. **Layout:** add `mkdocs.yml`; `docs/` (site pages, plus `docs/_gen/`, which
   generates the config, architecture and API pages and pulls in the report,
   decision log and Power BI guide); `Dockerfile`; and `docker/` (the
   entrypoint, plus `constraints.txt`, which pins the library versions behind
   the committed outputs).
5. **Determinism claim:** "A clean re-run reproduces every committed output
   byte-for-byte" holds on one platform only. Measured with the Docker image
   (Linux, versions pinned to this machine's):
   - 22/22 output files are identical between two container runs.
   - Against the committed (Windows) blobs, 8 of 13 files are identical. The
     rest differ only by last-digit float rounding (at most 6×10⁻¹⁶ absolute:
     `dim_sensor` q-values, `fact_run.model_score`, 2 confound p-values, and
     `results.json` `yield.regime_or_ci[0]`) and by the manifest /
     `results.json` `exports[].bytes` values. Those values record CSV file
     sizes, which follow platform line endings (CRLF on Windows, LF on Linux).
   - The figures fig01–fig06 are pixel-identical; only PNG metadata differs.

   Suggest rewording to "deterministic: same bytes on every run on a given
   platform; across platforms the numbers agree to float rounding". The docs
   say this in `architecture.md#determinism`.

## Other things for the integrator

- **Line endings (optional code fix, outside my scope):**
  `export.write_all` / `to_csv` write `os.linesep`. Passing
  `lineterminator="\n"` would make the CSVs and the manifest byte counts
  platform-independent. That changes the recorded `bytes` values once.
- **First PNGs of fig07–fig09:** the container rendered them, since Agg works on
  Linux. They are in
  `C:\Users\syedw\AppData\Local\Temp\claude\C--Users-syedw-Downloads-September-2026-semiconductor-process-control\2c1b8c81-57b6-4397-a27f-6b49b3d651c8\scratchpad\docker-run\reports\figures\`
  (not committed). To regenerate every figure for the merged branch, use the
  image: `docker build -t secom-process-control . && docker run --rm -v ...`.
- **GitHub Pages:** the docs workflow deploys only after Settings → Pages →
  Source is set to "GitHub Actions".
- **Docs home page:** `docs/index.md` has finding cards for the four core
  results only. Add cards for the multivariate, detection-power, change-point
  and Tennessee Eastman sections once they are in the report.
- **Report anchors:** the home cards link to `report.md#2-yield-behaviour`,
  `#3-the-monitoring-scheme-that-works`, `#4-which-sensor-if-any-explains-it`
  and `#5-can-failures-be-predicted-per-run`. Keep those headings when merging
  sections (strict mode warns on broken anchors).
- **Missing link targets:** after the sections are merged into
  `ENGINEERING_REPORT.md`, any link to a figure that has not been generated
  (e.g. fig07–fig12) makes `mkdocs build --strict` fail with
  "links to ..., which does not exist". That is deliberate: commit the figures
  first.
- **Ruff and Markdown:** `ruff format` (0.17.1) also formats Python code blocks
  inside `.md` files. `ruff format --check .` in CI covers them.
