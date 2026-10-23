"""Export the replay monitor as a static site that runs entirely in the browser.

``secom monitor --export DIR`` writes three files:

- ``index.html``, the same page ``secom monitor`` serves, switched to static
  mode by one injected line that names the data file;
- ``engine.js``, the JavaScript port of :class:`secom.monitor.ReplaySession`
  that the page uses in place of the JSON API;
- ``monitor-data.json``, the design and the run stream, in compact columns.

There is no server: any static host serves it (the docs site publishes it at
``/demo/``). The design and run data come from the committed pipeline outputs,
exactly as for ``secom monitor``, and the CUSUM weights are exported as the
doubles Python computed, so the browser reproduces the Python engine run for
run. ``tests/test_monitor_static.py`` checks that with node.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .monitor import (
    MonitorDesign,
    ReplayRuns,
    design_payload,
    json_safe,
    load_from_outputs,
)
from .monitor_server import static_file

#: Bumped when the layout of ``monitor-data.json`` changes.
FORMAT_VERSION = 1

DATA_FILE = "monitor-data.json"

#: Marker in index.html replaced by the static-mode switch.
STATIC_MARKER = "<!-- secom:static-data -->"


def _codes(values: Any) -> tuple[list[str], list[int]]:
    """Dictionary-encode a column: (distinct values in order seen, index per row)."""
    distinct: dict[str, int] = {}
    idx = [distinct.setdefault(str(v), len(distinct)) for v in values]
    return list(distinct), idx


def build_payload(design: MonitorDesign, runs: ReplayRuns, speed: float = 20.0) -> dict[str, Any]:
    """Everything the page needs, as strict-JSON-ready Python objects."""
    weeks, week_idx = _codes(runs.year_week)
    phases, phase_idx = _codes(runs.phase)
    return json_safe(
        {
            "format": FORMAT_VERSION,
            "design": design_payload(design, runs, speed),
            "runs": {
                "run_id": [int(v) for v in runs.run_id],
                "timestamp": [str(v) for v in runs.timestamp],
                "year_week_values": weeks,
                "year_week_idx": week_idx,
                "phase_values": phases,
                "phase_idx": phase_idx,
                # One character per run keeps 1,567 booleans to 1.5 kB.
                "is_fail": "".join("1" if v else "0" for v in runs.is_fail),
                "watch_value": [float(v) for v in runs.watch_value],
            },
        }
    )


def static_index_html(data_file: str = DATA_FILE) -> str:
    """The monitor page, switched to read ``data_file`` instead of the API."""
    html = static_file("index.html").decode("utf-8")
    if STATIC_MARKER not in html:
        raise RuntimeError(f"index.html has no {STATIC_MARKER} marker")
    switch = f"<script>window.SECOM_MONITOR_DATA = {json.dumps(data_file)};</script>"
    return html.replace(STATIC_MARKER, switch, 1)


def static_files(
    design: MonitorDesign, runs: ReplayRuns, *, speed: float = 20.0
) -> dict[str, bytes]:
    """The static monitor as {file name: contents}; the docs build writes these."""
    payload = build_payload(design, runs, speed)
    return {
        "index.html": static_index_html().encode("utf-8"),
        "engine.js": static_file("engine.js"),
        DATA_FILE: json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8"),
    }


def export_static(
    out_dir: str | Path,
    design: MonitorDesign,
    runs: ReplayRuns,
    *,
    speed: float = 20.0,
) -> list[Path]:
    """Write the static monitor into ``out_dir``; return the files written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, body in static_files(design, runs, speed=speed).items():
        path = out / name
        path.write_bytes(body)
        written.append(path)
    return written


def export_from_outputs(
    root: str | Path, out_dir: str | Path, *, speed: float = 20.0
) -> list[Path]:
    """:func:`export_static` with the design and runs from committed outputs."""
    design, runs = load_from_outputs(root)
    return export_static(out_dir, design, runs, speed=speed)
