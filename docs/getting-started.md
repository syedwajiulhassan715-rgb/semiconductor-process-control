# Getting started

## Install

Python 3.11 or later.

```bash
git clone https://github.com/syedwajiulhassan715-rgb/semiconductor-process-control
cd semiconductor-process-control
pip install -e .
```

## Fetch and run

```bash
secom fetch     # download SECOM into data/raw/
secom run       # about 20 s: results.json, figures, Power BI CSVs
secom stages    # list the pipeline stages in order
```

Run from the repository root, or pass `--root`. `data/raw/` is gitignored, and
`secom fetch` is its only source; it skips files already present.

A run writes:

| Output | Contents |
|---|---|
| `reports/results.json` | Every statistic quoted in the report |
| `reports/figures/` | The report figures |
| `data/powerbi/` | The Power BI star schema (see [Power BI](powerbi.md)) |
| `data/processed/` | Cleaned wide table, sensor audit, imputation log, model results, confound check |

The pipeline is deterministic: on a given platform, every clean run produces
the same bytes. Across platforms the numbers agree to floating-point rounding;
see [Determinism](architecture.md#determinism).

## The `secom` command

| Command | Does |
|---|---|
| `secom fetch [--root PATH]` | Download the raw SECOM files |
| `secom run [options]` | Run the analysis pipeline |
| `secom stages [-v]` | List the stages in order |
| `secom monitor [options]` | Start the [replay monitor](monitor.md) |

`secom run` options:

| Option | Effect |
|---|---|
| `--config PATH` | Use this YAML config (default `config/default.yaml`); see [Configuration](configuration.md) |
| `--only STAGE ...` | Run these stages plus the upstream stages they depend on |
| `--no-figures` | Skip figure rendering |
| `--figure-format png\|svg` | Override `figures.format`; `svg` needs no compiled matplotlib backend |
| `--figures-dir PATH` | Write figures somewhere other than `reports/figures/` |
| `--root PATH` | Repository root holding `data/` and `reports/` |

```bash
secom run --only cusum           # cusum plus the stages it depends on
secom run --figure-format svg    # pure-Python figure backend
secom run --config my.yaml       # rerun under different thresholds
```

An `--only` run that stops before the `export` stage leaves the committed
outputs untouched.

## Docker

The image runs the whole analysis on Linux without a local Python setup. Its
default command fetches the data if it is missing and runs the full pipeline
with PNG figures:

```bash
docker build -t secom-process-control .
docker run --rm \
  -v "$PWD/data:/app/data" \
  -v "$PWD/reports:/app/reports" \
  secom-process-control
```

The two mounts put the outputs on the host: `data/raw/` (the download, reused on
later runs), `data/powerbi/`, `data/processed/`, `reports/results.json` and
`reports/figures/`. Without them the run happens inside the container and its
outputs are discarded with it. On Windows PowerShell, use `${PWD}` instead of
`$PWD`.

Any `secom` command can follow the image name:

```bash
docker run --rm secom-process-control stages
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/reports:/app/reports" \
  secom-process-control run --only cusum
docker run --rm -p 8050:8050 secom-process-control monitor --host 0.0.0.0
```

The image already contains the committed outputs, so the monitor works without
mounts. The image pins the exact library versions that produced the committed
outputs (`docker/constraints.txt`).

On Linux, files written to a bind mount are owned by the container user
(uid 1000). If that is not you, add `--user "$(id -u):$(id -g)"` to the
`docker run` command.

## Development

```bash
pip install -e ".[dev]"     # pytest, pytest-cov, ruff, mypy, pre-commit
pre-commit install
ruff check . && ruff format --check .
mypy
pytest -q --cov=secom
```

Without `data/raw/`, the real-data tests skip and the rest still run. CI runs
lint, mypy, the full suite with coverage, a `secom run` smoke test and a Docker
build on every push and pull request.

To build this site locally:

```bash
pip install -e ".[docs]"
mkdocs serve                # http://127.0.0.1:8000/
mkdocs build --strict       # what CI runs
```
