"""Tests for the static, server-free monitor (secom.monitor_export + engine.js).

The Python checks always run. The equivalence checks run the JavaScript engine
under node, when it is installed, and require it to reproduce the Python
session's state exactly: every CUSUM value, alarm and log entry.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from secom.monitor import (
    MonitorDesign,
    ReplayRuns,
    ReplaySession,
    json_safe,
    load_from_outputs,
)
from secom.monitor_export import (
    DATA_FILE,
    STATIC_MARKER,
    build_payload,
    export_from_outputs,
    export_static,
    static_index_html,
)
from secom.monitor_server import static_file

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


def _fixture(seed: int = 3) -> tuple[MonitorDesign, ReplayRuns]:
    """600 runs: a burst of failures, then a long clean stretch, NaN readings."""
    rng = np.random.default_rng(seed)
    n = 600
    p = np.full(n, 0.03)
    p[150:260] = 0.25
    p[260:] = 0.005
    watch = rng.normal(size=n)
    watch[::37] = np.nan
    watch[::11] = 5.0
    design = MonitorDesign(
        p0=0.03,
        p1=0.06,
        h=1.8,
        arl0=500.0,
        baseline_weeks=(35, 39),
        watch_sensor="S060",
        watch_center=0.0,
        watch_lcl=-3.0,
        watch_ucl=3.0,
        watch_median=0.0,
        watch_mad=1.0,
    )
    runs = ReplayRuns(
        run_id=np.arange(101, 101 + n),
        timestamp=np.array([f"2008-07-{19 + i // 100:02d}T{i % 24:02d}:00:00" for i in range(n)]),
        year_week=np.array([f"2008-W{29 + i // 100}" for i in range(n)]),
        phase=np.array(["0 - pre"] * 200 + ["1 - base"] * 300 + ["2 - post"] * 100),
        is_fail=rng.random(n) < p,
        watch_value=watch,
    )
    return design, runs


# Python side


def test_export_writes_the_three_files(tmp_path: Path) -> None:
    design, runs = _fixture()
    written = export_static(tmp_path, design, runs, speed=50)
    assert sorted(p.name for p in written) == sorted(["index.html", "engine.js", DATA_FILE])
    assert (tmp_path / "engine.js").read_bytes() == static_file("engine.js")
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert f'window.SECOM_MONITOR_DATA = "{DATA_FILE}"' in html
    assert STATIC_MARKER not in html
    # Chart.js stays pinned to the exact cdnjs build the served page uses
    assert "cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js" in html


def test_payload_is_strict_json_and_round_trips_the_runs(tmp_path: Path) -> None:
    design, runs = _fixture()
    export_static(tmp_path, design, runs, speed=50)
    data = json.loads((tmp_path / DATA_FILE).read_text(encoding="utf-8"))
    assert data == build_payload(design, runs, 50)
    r = data["runs"]
    assert r["run_id"] == runs.run_id.tolist()
    assert [r["year_week_values"][i] for i in r["year_week_idx"]] == runs.year_week.tolist()
    assert [r["phase_values"][i] for i in r["phase_idx"]] == runs.phase.tolist()
    assert [c == "1" for c in r["is_fail"]] == runs.is_fail.tolist()
    nan = np.isnan(runs.watch_value)
    assert [v is None for v in r["watch_value"]] == nan.tolist()
    assert np.array_equal(np.array(r["watch_value"], dtype=float)[~nan], runs.watch_value[~nan])
    d = data["design"]
    assert d["default_speed"] == 50 and d["n_runs"] == 600
    assert (d["lower_p1"], d["lower_h"]) == design.lower


def test_exports_from_committed_outputs(tmp_path: Path) -> None:
    export_from_outputs(ROOT, tmp_path)
    data = json.loads((tmp_path / DATA_FILE).read_text(encoding="utf-8"))
    design, runs = load_from_outputs(ROOT)
    assert data["design"]["n_runs"] == len(runs) == 1567
    assert data["design"]["h"] == design.h and data["design"]["lower_h"] == design.lower_h
    # the full run stream is small enough to ship with the docs site
    assert (tmp_path / DATA_FILE).stat().st_size < 200_000


def test_marker_is_required() -> None:
    assert STATIC_MARKER in static_file("index.html").decode()
    assert static_index_html("x.json").count("SECOM_MONITOR_DATA") >= 2


def test_cli_export(tmp_path: Path) -> None:
    from secom.cli import main

    assert main(["monitor", "--root", str(ROOT), "--export", str(tmp_path / "demo")]) == 0
    assert (tmp_path / "demo" / DATA_FILE).exists()


# JavaScript engine, under node

# Drives engine.js through a script of API calls and prints every response.
_DRIVER = r"""
const fs = require("fs");
const [engine, data, script] = process.argv.slice(2);
const E = require(engine);
const api = E.localApi(JSON.parse(fs.readFileSync(data, "utf8")));
const out = [];
for (const [path, body] of JSON.parse(fs.readFileSync(script, "utf8"))) {
  try { out.push(api.request(path, body === null ? undefined : body)); }
  catch (e) { out.push({ error: e.message }); }
}
process.stdout.write(JSON.stringify(out));
"""


def _python_replies(session: ReplaySession, script: list[tuple[str, Any]]) -> list[Any]:
    """What monitor_server would answer for each call in ``script``."""
    out: list[Any] = []
    for path, body in script:
        if body is None:
            since = int(path.split("since=")[1]) if "since=" in path else 0
            out.append(session.snapshot(since))
            continue
        try:
            if path == "/api/step":
                session.step(body["n"])
            elif path == "/api/seek":
                session.seek(body["run"])
            elif path == "/api/reset":
                session.reset()
            elif path == "/api/ack":
                session.acknowledge(body["alarm_id"], body.get("note", "").strip())
        except Exception as exc:  # the server returns these as {"error": ...}
            out.append({"error": str(exc)})
            continue
        out.append(session.snapshot(body.get("since", 0)))
    return json_safe(out)


def _run_node(tmp_path: Path, design: MonitorDesign, runs: ReplayRuns, script: list[Any]) -> Any:
    assert NODE is not None
    export_static(tmp_path, design, runs)
    (tmp_path / "script.json").write_text(json.dumps(script), encoding="utf-8")
    (tmp_path / "driver.js").write_text(_DRIVER, encoding="utf-8")
    proc = subprocess.run(
        [
            NODE,
            str(tmp_path / "driver.js"),
            str(tmp_path / "engine.js"),
            str(tmp_path / DATA_FILE),
            str(tmp_path / "script.json"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def _script(first_alarm: int, n: int) -> list[tuple[str, Any]]:
    return [
        ("/api/step", {"n": 37}),
        ("/api/step", {"n": 400, "since": 37}),
        ("/api/seek", {"run": first_alarm + 3}),
        ("/api/ack", {"alarm_id": 1, "note": "  checked tool 4  "}),
        ("/api/ack", {"alarm_id": 1}),
        ("/api/ack", {"alarm_id": 99}),
        ("/api/seek", {"run": n}),
        ("/api/state?since=" + str(n - 5), None),
        ("/api/seek", {"run": first_alarm + 1}),
        ("/api/step", {"n": 2000, "since": first_alarm + 1}),
        ("/api/reset", {}),
        ("/api/step", {"n": 1}),
    ]


needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


@needs_node
def test_js_engine_matches_python_on_a_synthetic_stream(tmp_path: Path) -> None:
    design, runs = _fixture()
    session = ReplaySession(design, runs)
    session.step(len(runs))
    sources = {a.source for a in session.alarms.alarms}
    assert {"cusum", "cusum_lower"} <= sources, "fixture should exercise both CUSUMs"
    first = session.alarms.alarms[0].raised_run
    script = _script(first, len(runs))
    expected = _python_replies(ReplaySession(design, runs), script)
    assert _run_node(tmp_path, design, runs, script) == expected


@needs_node
def test_js_engine_matches_python_on_the_secom_runs(tmp_path: Path) -> None:
    design, runs = load_from_outputs(ROOT)
    script: list[tuple[str, Any]] = [("/api/step", {"n": 2000}), ("/api/seek", {"run": len(runs)})]
    expected = _python_replies(ReplaySession(design, runs), script)
    got = _run_node(tmp_path, design, runs, script)
    assert got == expected
    history = got[-1]["history"]
    assert len(history) == 1567
    # the CUSUM values are the exact doubles, not merely close
    assert [p["cusum"] for p in history] == [p["cusum"] for p in expected[-1]["history"]]
    assert any(p["cusum_lower_signal"] for p in history)
