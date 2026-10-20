"""Tests for the replay monitor: online monitors, alarms, session and HTTP API."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from secom.cli import main
from secom.monitor import (
    AlarmBook,
    AlarmError,
    MonitorDesign,
    OnlineBernoulliCusum,
    OnlineGChart,
    ReplayRuns,
    ReplaySession,
    WatchlistTracker,
    geometric_limits,
    load_from_outputs,
    phase_spans,
)
from secom.monitor_server import MonitorServer, serve
from secom.spc import bernoulli_cusum, g_chart

ROOT = Path(__file__).resolve().parents[1]


def _design(**kw: Any) -> MonitorDesign:
    base: dict[str, Any] = {
        "p0": 0.03,
        "p1": 0.06,
        "h": 1.8,
        "arl0": 500.0,
        "baseline_weeks": (35, 39),
        "watch_sensor": "S060",
        "watch_center": 0.0,
        "watch_lcl": -3.0,
        "watch_ucl": 3.0,
        "watch_median": 0.0,
        "watch_mad": 1.0,
    }
    base.update(kw)
    return MonitorDesign(**base)


def _runs(is_fail: np.ndarray, watch: np.ndarray | None = None) -> ReplayRuns:
    n = len(is_fail)
    phase = np.array(["0 - pre"] * (n // 2) + ["1 - base"] * (n - n // 2))
    return ReplayRuns(
        run_id=np.arange(1, n + 1),
        timestamp=np.array([f"2008-07-19T{i % 24:02d}:00:00" for i in range(n)]),
        year_week=np.array(["2008-W29"] * n),
        phase=phase,
        is_fail=np.asarray(is_fail, dtype=bool),
        watch_value=np.zeros(n) if watch is None else np.asarray(watch, dtype=float),
    )


def _bursty(n: int = 600, seed: int = 3) -> np.ndarray:
    """Low fail rate with a high-rate burst in the middle, so the CUSUM signals."""
    rng = np.random.default_rng(seed)
    p = np.full(n, 0.03)
    p[200:320] = 0.25
    return rng.random(n) < p


# Online CUSUM


class TestOnlineCusum:
    @pytest.mark.parametrize("reset", [True, False])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_reproduces_batch_cusum_exactly(self, reset: bool, seed: int) -> None:
        x = _bursty(seed=seed)
        batch = bernoulli_cusum(x, p0=0.03, p1=0.06, h=1.8, reset_on_signal=reset).table
        online = OnlineBernoulliCusum(0.03, 0.06, 1.8, reset_on_signal=reset)
        states = [online.update(bool(v)) for v in x]
        assert np.array_equal([s.value for s in states], batch["cusum"].to_numpy())
        assert np.array_equal([s.signal for s in states], batch["signal"].to_numpy())
        assert np.array_equal([s.increment for s in states], batch["increment"].to_numpy())
        assert batch["signal"].any()

    def test_resets_after_signal(self) -> None:
        c = OnlineBernoulliCusum(0.03, 0.06, 1.0)
        s1 = c.update(True)
        s2 = c.update(True)
        assert s1.value == pytest.approx(np.log(2))
        assert s2.signal and s2.value > 1.0
        assert c.value == 0.0

    @pytest.mark.parametrize(("p0", "p1"), [(0.0, 0.1), (0.1, 1.0), (0.1, 0.05), (0.1, 0.1)])
    def test_rejects_invalid_design(self, p0: float, p1: float) -> None:
        with pytest.raises(ValueError):
            OnlineBernoulliCusum(p0, p1, 1.0)


# g-chart


class TestOnlineGChart:
    def test_limits_match_batch_g_chart(self) -> None:
        fails = np.flatnonzero(_bursty())
        batch = g_chart(fails, n_total=600, baseline_p=0.0307)
        center, lcl, ucl = geometric_limits(0.0307)
        assert (center, lcl, ucl) == (batch.center, batch.lcl, batch.ucl)

    def test_closed_gaps_match_batch_table(self) -> None:
        x = _bursty()
        g = OnlineGChart(0.03)
        closed = [s.closed_gap for s in (g.update(bool(v)) for v in x) if s.closed_gap is not None]
        batch = g_chart(np.flatnonzero(x), n_total=len(x), baseline_p=0.03)
        assert closed == batch.table["runs_since_previous_fail"].tolist()

    def test_signals_once_when_open_gap_crosses_ucl(self) -> None:
        g = OnlineGChart(0.2)
        assert not g.update(True).signal
        ucl = int(g.ucl)
        states = [g.update(False) for _ in range(ucl + 20)]
        signals = [i for i, s in enumerate(states, start=1) if s.signal]
        assert signals == [ucl + 1]

    def test_no_signal_before_first_failure(self) -> None:
        g = OnlineGChart(0.2)
        assert not any(g.update(False).signal for _ in range(500))

    def test_lower_limit_inactive_at_realistic_rates(self) -> None:
        assert not OnlineGChart(0.0307).lower_limit_active


# Watchlist


def test_watchlist_counts_recent_exceedances() -> None:
    w = WatchlistTracker("S1", center=0, lcl=-1, ucl=1, median=0, mad=2, window=3)
    states = [w.update(v) for v in (0.0, 2.0, -2.0, 0.5, 0.5)]
    assert [s.beyond for s in states] == [False, True, True, False, False]
    assert [s.recent_beyond for s in states] == [0, 1, 2, 2, 1]
    assert states[1].robust_z == 1.0


# Alarms


class TestAlarmBook:
    def test_repeat_signals_fold_into_the_open_alarm(self) -> None:
        book = AlarmBook()
        a = book.signal("cusum", "degradation", "m", 10, "t")
        b = book.signal("cusum", "degradation", "m", 15, "t")
        assert a is b and a.signals == 2 and a.last_signal_run == 15
        assert [e["event"] for e in book.log] == ["raised", "repeat"]

    def test_acknowledged_alarm_closes_and_next_signal_opens_a_new_one(self) -> None:
        book = AlarmBook()
        a = book.signal("cusum", "degradation", "m", 10, "t")
        book.acknowledge(a.id, 12, "t", "checked tool 4")
        assert not a.is_open and a.note == "checked tool 4"
        b = book.signal("cusum", "degradation", "m", 20, "t")
        assert b.id == 2 and b.is_open
        assert book.log[1]["event"] == "acknowledged"

    def test_sources_are_independent(self) -> None:
        book = AlarmBook()
        book.signal("cusum", "degradation", "m", 1, "t")
        book.signal("gchart", "improvement", "m", 2, "t")
        assert len(book.open_alarms) == 2

    def test_bad_acknowledgements_raise(self) -> None:
        book = AlarmBook()
        a = book.signal("cusum", "degradation", "m", 1, "t")
        with pytest.raises(AlarmError):
            book.acknowledge(99, 2, "t")
        book.acknowledge(a.id, 2, "t")
        with pytest.raises(AlarmError, match="already"):
            book.acknowledge(a.id, 3, "t")


# Session


class TestReplaySession:
    def _session(self) -> ReplaySession:
        return ReplaySession(_design(h=1.8), _runs(_bursty()))

    def test_steps_through_every_run(self) -> None:
        s = self._session()
        assert s.step(10_000)[-1]["run"] == 600
        assert s.finished and s.step(5) == []
        assert s.history[-1]["fail_rate"] == pytest.approx(_bursty().mean())

    def test_seek_is_the_same_as_stepping_from_scratch(self) -> None:
        a, b = self._session(), self._session()
        a.step(450)
        a.seek(250)
        b.step(250)
        assert a.history == b.history
        assert a.snapshot() == b.snapshot()

    def test_acknowledgements_survive_a_forward_seek_and_drop_on_rewind(self) -> None:
        s = self._session()
        s.step(600)
        first = s.alarms.alarms[0]
        s.seek(first.raised_run + 5)
        s.acknowledge(first.id, "investigated")
        ack_run = s.cursor
        s.seek(600)
        assert s.alarms.alarms[0].acknowledged_run == ack_run
        assert s.alarms.alarms[0].note == "investigated"
        s.seek(ack_run - 1)
        assert s.alarms.alarms[0].is_open

    def test_reset_clears_everything(self) -> None:
        s = self._session()
        s.step(600)
        s.acknowledge(s.alarms.alarms[0].id)
        s.reset()
        assert s.cursor == 0 and not s.alarms.alarms
        s.step(600)
        assert s.alarms.alarms[0].is_open

    def test_cannot_acknowledge_before_any_run(self) -> None:
        with pytest.raises(AlarmError):
            self._session().acknowledge(1)

    def test_negative_step_raises(self) -> None:
        with pytest.raises(ValueError):
            self._session().step(-1)

    def test_snapshot_since_returns_only_new_points(self) -> None:
        s = self._session()
        s.step(20)
        snap = s.snapshot(since=15)
        assert snap["since"] == 15 and [p["run"] for p in snap["history"]] == [16, 17, 18, 19, 20]
        assert s.snapshot(since=999)["history"] == []

    def test_phase_spans_cover_the_stream(self) -> None:
        spans = phase_spans(_runs(np.zeros(10, bool)))
        assert spans == [
            {"label": "0 - pre", "start": 1, "end": 5},
            {"label": "1 - base", "start": 6, "end": 10},
        ]


# Committed outputs (always present in the repo)


@pytest.fixture(scope="module")
def real_session() -> ReplaySession:
    design, runs = load_from_outputs(ROOT)
    session = ReplaySession(design, runs)
    session.step(len(runs))
    return session


class TestAgainstCommittedOutputs:
    def test_design_comes_from_the_pipeline(self, real_session: ReplaySession) -> None:
        results = json.loads((ROOT / "reports" / "results.json").read_text(encoding="utf-8"))
        d = real_session.design
        assert (d.p0, d.p1, d.h) == (
            results["cusum"]["p0"],
            results["cusum"]["p1"],
            results["cusum"]["h"],
        )
        assert d.watch_sensor == results["sensors"]["hits_surviving_confound_check"][0]

    def test_cusum_matches_exported_fact_run(self, real_session: ReplaySession) -> None:
        fact = pd.read_csv(ROOT / "data" / "powerbi" / "fact_run.csv")
        values = np.array([p["cusum"] for p in real_session.history])
        signals = np.array([p["cusum_signal"] for p in real_session.history])
        # The CSV round-trips floats to ~1e-16; the batch comparison above is exact.
        assert np.allclose(values, fact["cusum"].to_numpy(), rtol=0, atol=1e-12)
        assert np.array_equal(signals, fact["cusum_signal"].to_numpy())

    def test_watchlist_flags_match_exported_measurements(self, real_session: ReplaySession) -> None:
        meas = pd.read_csv(ROOT / "data" / "powerbi" / "fact_measurement.csv")
        sensor = real_session.design.watch_sensor
        expected = meas.loc[meas["sensor_id"] == sensor, "beyond_limits"].to_numpy()
        got = np.array([p["watch_beyond"] for p in real_session.history])
        assert np.array_equal(got, expected)

    def test_first_alarm_is_the_opening_excursion(self, real_session: ReplaySession) -> None:
        results = json.loads((ROOT / "reports" / "results.json").read_text(encoding="utf-8"))
        first = real_session.alarms.alarms[0]
        assert first.source == "cusum"
        assert first.raised_run == results["cusum"]["detection"]["pre_baseline"]["runs_to_signal"]

    def test_missing_outputs_raise_a_helpful_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="secom run"):
            load_from_outputs(tmp_path)


# HTTP API


@pytest.fixture()
def server() -> Iterator[MonitorServer]:
    session = ReplaySession(_design(h=1.8), _runs(_bursty(), watch=np.full(600, np.nan)))
    srv = serve(session, port=0, speed=10)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def _get(srv: MonitorServer, path: str) -> tuple[int, Any]:
    with urllib.request.urlopen(srv.url.rstrip("/") + path, timeout=5) as res:
        body = res.read()
        ctype = res.headers["Content-Type"]
        return res.status, json.loads(body) if "json" in ctype else body.decode()


def _post(srv: MonitorServer, path: str, body: dict[str, Any]) -> tuple[int, Any]:
    req = urllib.request.Request(
        srv.url.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as res:
            return res.status, json.loads(res.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


class TestHttpApi:
    def test_serves_the_page(self, server: MonitorServer) -> None:
        status, html = _get(server, "/")
        assert status == 200 and "<canvas" in html and "Chart.js/4.4.1" in html

    def test_design_endpoint(self, server: MonitorServer) -> None:
        status, d = _get(server, "/api/design")
        assert status == 200
        assert d["n_runs"] == 600 and d["default_speed"] == 10
        assert d["gchart"]["lower_limit_active"] is False
        assert d["phases"][0]["start"] == 1

    def test_step_returns_only_new_points_and_strict_json(self, server: MonitorServer) -> None:
        _, s1 = _post(server, "/api/step", {"n": 5})
        assert s1["cursor"] == 5 and len(s1["history"]) == 5
        _, s2 = _post(server, "/api/step", {"n": 3, "since": 5})
        assert [p["run"] for p in s2["history"]] == [6, 7, 8]
        # watch values are NaN in this fixture; they must arrive as null
        assert s2["history"][0]["watch_value"] is None

    def test_seek_ack_and_reset(self, server: MonitorServer) -> None:
        _, s = _post(server, "/api/seek", {"run": 600})
        alarm = s["open_alarms"][0]
        status, s = _post(server, "/api/ack", {"alarm_id": alarm["id"], "note": "ok"})
        assert status == 200
        assert s["alarms"][0]["status"] == "acknowledged" and s["alarms"][0]["note"] == "ok"
        status, err = _post(server, "/api/ack", {"alarm_id": alarm["id"]})
        assert status == 409 and "already" in err["error"]
        _, s = _post(server, "/api/reset", {})
        assert s["cursor"] == 0 and s["alarms"] == []

    @pytest.mark.parametrize(
        ("path", "body", "status"),
        [
            ("/api/step", {"n": -1}, 400),
            ("/api/step", {"n": "many"}, 400),
            ("/api/seek", {}, 400),
            ("/api/ack", {"alarm_id": 5}, 409),
            ("/api/ack", {"alarm_id": 1, "note": 3}, 400),
            ("/api/nope", {}, 404),
        ],
    )
    def test_rejects_bad_requests(
        self, server: MonitorServer, path: str, body: dict[str, Any], status: int
    ) -> None:
        if path == "/api/ack":
            _post(server, "/api/step", {"n": 1})
        code, err = _post(server, path, body)
        assert code == status and "error" in err

    def test_state_get(self, server: MonitorServer) -> None:
        _post(server, "/api/step", {"n": 4})
        status, s = _get(server, "/api/state?since=2")
        assert status == 200 and s["since"] == 2 and len(s["history"]) == 2


# CLI


def test_cli_monitor_reports_missing_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["monitor", "--root", str(tmp_path), "--port", "0"]) == 2
    assert "secom run" in capsys.readouterr().err


def test_cli_monitor_rejects_non_positive_speed(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["monitor", "--root", str(ROOT), "--speed", "0"]) == 2
    assert "--speed" in capsys.readouterr().err
