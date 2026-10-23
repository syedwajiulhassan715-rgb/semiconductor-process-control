"""Run-by-run replay of the recommended monitoring scheme.

The report recommends a Bernoulli CUSUM against the Phase I baseline as the
live yield monitor, a lower Bernoulli CUSUM as the improvement detector (at the
same false-alarm risk it detects a halving far sooner than the g-chart), and
S060 on a watchlist. This module runs them online, one run at a time, the way
a line engineer would see them:

- :class:`OnlineBernoulliCusum` reproduces :func:`secom.spc.bernoulli_cusum`
  exactly, including reset-on-signal (D13). With ``side="lower"`` and
  ``p1 < p0`` it is the improvement CUSUM: the same log-likelihood-ratio
  recursion, which now builds up from passes and is paid down by failures.
- :class:`OnlineGChart` counts passing runs since the last failure against
  fixed geometric limits at the Phase I rate. It stays on the page as context,
  since it is the chart engineers know, but it no longer raises alarms: the
  lower CUSUM is the improvement alarm, and two alarms for one improvement
  would open the same investigation twice. UCL crossings are still marked.
- :class:`WatchlistTracker` reports S060 against its Phase I individuals
  limits. It is informational and never raises an alarm, because S060 is a
  watchlist item, not a root cause.
- :class:`AlarmBook` turns statistical signals into alarms an engineer works.
  The CUSUM resets the moment it signals; the alarm stays raised until someone
  acknowledges it, and repeat signals while it is open fold into it.

:class:`ReplaySession` drives all of them through the SECOM runs in time order.
The design is read from committed pipeline outputs (``reports/results.json`` and
``data/powerbi/``), so a session starts without rerunning the analysis.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

#: Same false-alarm rate as a 3-sigma chart; matches spc.g_chart's default.
GCHART_ALPHA = 0.0027

#: Trailing window, in runs, for the recent fail rate and watchlist exceedances.
RECENT_WINDOW = 50

#: Per-point exceedance probability of a 3-sigma individuals chart.
CHANCE_EXCEEDANCE = 0.0027


# Online monitors


@dataclass(frozen=True)
class CusumState:
    """CUSUM after one run. ``value`` is recorded before any reset."""

    value: float
    increment: float
    signal: bool


class OnlineBernoulliCusum:
    """Incremental one-sided Bernoulli CUSUM.

    Per run it adds the log-likelihood ratio of p1 against p0::

        S_t = max(0, S_{t-1} + x_t log(p1/p0) + (1 - x_t) log((1-p1)/(1-p0)))

    and signals when ``S_t > h``. ``side="upper"`` (p1 > p0) is the degradation
    chart of :func:`secom.spc.bernoulli_cusum`, with the same weights and the
    same floor-then-compare order, so feeding it a sequence gives bit-identical
    values to the batch function. ``side="lower"`` (p1 < p0) is the improvement
    chart: the pass weight is now positive and the fail weight negative, so a
    long run of passes builds evidence that the fail rate has fallen. It is the
    recursion of :func:`secom.power.cusum_first_signal` with ``p1 < p0``.
    """

    def __init__(
        self,
        p0: float,
        p1: float,
        h: float,
        *,
        reset_on_signal: bool = True,
        side: str = "upper",
    ) -> None:
        if side not in ("upper", "lower"):
            raise ValueError("side must be 'upper' or 'lower'")
        if not (0 < p0 < 1 and 0 < p1 < 1):
            raise ValueError("p0 and p1 must lie strictly in (0, 1)")
        if side == "upper" and p1 <= p0:
            raise ValueError("p1 must exceed p0 for an upper CUSUM")
        if side == "lower" and p1 >= p0:
            raise ValueError("p1 must be below p0 for a lower CUSUM")
        self.side = side
        self.p0, self.p1, self.h = p0, p1, h
        self.reset_on_signal = reset_on_signal
        # np.log rather than math.log so the weights match the batch version to
        # the last bit.
        self.w_fail = float(np.log(p1 / p0))
        self.w_pass = float(np.log((1 - p1) / (1 - p0)))
        self.value = 0.0

    def update(self, is_fail: bool) -> CusumState:
        inc = self.w_fail if is_fail else self.w_pass
        acc = max(0.0, self.value + inc)
        signal = acc > self.h
        self.value = 0.0 if (signal and self.reset_on_signal) else acc
        return CusumState(value=acc, increment=inc, signal=signal)


def geometric_limits(p: float, alpha: float = GCHART_ALPHA) -> tuple[float, float, float]:
    """(center, lcl, ucl) for passing runs between failures at fail rate ``p``.

    Same exact quantiles as :func:`secom.spc.g_chart`.
    """
    p = min(max(p, 1e-9), 1 - 1e-9)
    dist = sps.geom(p, loc=-1)
    return (
        float((1.0 - p) / p),
        float(dist.ppf(alpha / 2.0)),
        float(dist.ppf(1.0 - alpha / 2.0)),
    )


@dataclass(frozen=True)
class GChartState:
    """g-chart after one run."""

    gap: int
    closed_gap: int | None
    signal: bool


class OnlineGChart:
    """Passing runs since the last failure, against fixed geometric limits.

    ``closed_gap`` is set on the run that ends a gap (a failure), which is
    the value the batch chart plots. ``signal`` fires once per gap, on the run
    where the open gap first exceeds the UCL. That is the earliest moment the
    batch chart's eventual above-UCL point becomes certain. Before the first
    failure there is no gap to measure, so nothing signals.
    """

    def __init__(self, p: float, alpha: float = GCHART_ALPHA) -> None:
        self.p = p
        self.alpha = alpha
        self.center, self.lcl, self.ucl = geometric_limits(p, alpha)
        self.gap = 0
        self.seen_fail = False
        self._signalled = False

    @property
    def lower_limit_active(self) -> bool:
        return self.lcl > 0

    def update(self, is_fail: bool) -> GChartState:
        if is_fail:
            closed = self.gap if self.seen_fail else None
            self.seen_fail = True
            self.gap = 0
            self._signalled = False
            return GChartState(gap=0, closed_gap=closed, signal=False)
        self.gap += 1
        signal = self.seen_fail and not self._signalled and self.gap > self.ucl
        if signal:
            self._signalled = True
        return GChartState(gap=self.gap, closed_gap=None, signal=signal)


@dataclass(frozen=True)
class WatchState:
    """Watchlist sensor after one run."""

    value: float
    robust_z: float
    beyond: bool
    recent_beyond: int
    recent_n: int


class WatchlistTracker:
    """One sensor against its Phase I individuals limits, plus recent exceedances."""

    def __init__(
        self,
        sensor: str,
        *,
        center: float,
        lcl: float,
        ucl: float,
        median: float,
        mad: float,
        window: int = RECENT_WINDOW,
    ) -> None:
        self.sensor = sensor
        self.center, self.lcl, self.ucl = center, lcl, ucl
        self.median, self.mad = median, mad
        self._recent: deque[bool] = deque(maxlen=window)

    def update(self, value: float) -> WatchState:
        beyond = bool(value < self.lcl or value > self.ucl)
        self._recent.append(beyond)
        z = (value - self.median) / self.mad if self.mad else float("nan")
        return WatchState(
            value=float(value),
            robust_z=float(z),
            beyond=beyond,
            recent_beyond=int(sum(self._recent)),
            recent_n=len(self._recent),
        )


# Alarms


@dataclass
class Alarm:
    """An alarm as an engineer sees it: raised once, worked, acknowledged."""

    id: int
    source: str
    kind: str
    message: str
    raised_run: int
    raised_at: str
    signals: int = 1
    last_signal_run: int = 0
    acknowledged_run: int | None = None
    acknowledged_at: str | None = None
    note: str = ""

    @property
    def is_open(self) -> bool:
        return self.acknowledged_run is None

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = "open" if self.is_open else "acknowledged"
        return d


class AlarmError(ValueError):
    """Acknowledging an alarm that does not exist or is already closed."""


class AlarmBook:
    """Open alarms per source, plus a log of every raise and acknowledgement.

    A source has at most one open alarm. Signals while it is open are counted
    on that alarm instead of raising new ones, so one excursion means one
    investigation however many runs it takes to acknowledge.
    """

    def __init__(self) -> None:
        self.alarms: list[Alarm] = []
        self.log: list[dict[str, Any]] = []

    def open_for(self, source: str) -> Alarm | None:
        return next((a for a in reversed(self.alarms) if a.source == source and a.is_open), None)

    @property
    def open_alarms(self) -> list[Alarm]:
        return [a for a in self.alarms if a.is_open]

    def signal(self, source: str, kind: str, message: str, run: int, at: str) -> Alarm:
        current = self.open_for(source)
        if current is not None:
            current.signals += 1
            current.last_signal_run = run
            self.log.append(
                {
                    "run": run,
                    "at": at,
                    "event": "repeat",
                    "alarm_id": current.id,
                    "source": source,
                    "message": f"repeat signal ({current.signals} so far)",
                }
            )
            return current
        alarm = Alarm(
            id=len(self.alarms) + 1,
            source=source,
            kind=kind,
            message=message,
            raised_run=run,
            raised_at=at,
            last_signal_run=run,
        )
        self.alarms.append(alarm)
        self.log.append(
            {
                "run": run,
                "at": at,
                "event": "raised",
                "alarm_id": alarm.id,
                "source": source,
                "message": message,
            }
        )
        return alarm

    def acknowledge(self, alarm_id: int, run: int, at: str, note: str = "") -> Alarm:
        alarm = next((a for a in self.alarms if a.id == alarm_id), None)
        if alarm is None:
            raise AlarmError(f"no alarm with id {alarm_id}")
        if not alarm.is_open:
            raise AlarmError(
                f"alarm {alarm_id} was already acknowledged at run {alarm.acknowledged_run}"
            )
        alarm.acknowledged_run = run
        alarm.acknowledged_at = at
        alarm.note = note
        self.log.append(
            {
                "run": run,
                "at": at,
                "event": "acknowledged",
                "alarm_id": alarm_id,
                "source": alarm.source,
                "message": note or "acknowledged",
            }
        )
        return alarm


# Design and data


#: ``lower_source`` when ``results.json`` carries no lower CUSUM design.
LOWER_FALLBACK = "fallback: p1 = p0/2, h of the upper CUSUM"


@dataclass(frozen=True)
class MonitorDesign:
    """Fixed Phase I design for every monitor in a session.

    ``lower_p1`` and ``lower_h`` design the improvement CUSUM. The pipeline's
    calibrated values are used whenever ``results.json`` has them (see
    :func:`lower_cusum_design`). Left as ``None`` they fall back to
    p1 = p0/2, a halving that mirrors the upper chart's doubling, and the upper
    chart's h. That fallback is close: the pipeline's own calibration of the
    lower chart to the same ARL0 of about 500 runs gives h = 1.51 against the
    upper chart's 1.82, so reusing h errs towards fewer false alarms.
    """

    p0: float
    p1: float
    h: float
    arl0: float
    baseline_weeks: tuple[int, int]
    watch_sensor: str
    watch_center: float
    watch_lcl: float
    watch_ucl: float
    watch_median: float
    watch_mad: float
    gchart_alpha: float = GCHART_ALPHA
    lower_p1: float | None = None
    lower_h: float | None = None
    lower_arl0: float | None = None
    lower_source: str = LOWER_FALLBACK

    def __post_init__(self) -> None:
        if self.lower_p1 is None:
            object.__setattr__(self, "lower_p1", self.p0 / 2.0)
        if self.lower_h is None:
            object.__setattr__(self, "lower_h", self.h)

    @property
    def lower(self) -> tuple[float, float]:
        """(p1, h) of the lower CUSUM, after any fallback."""
        assert self.lower_p1 is not None and self.lower_h is not None
        return self.lower_p1, self.lower_h

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        center, lcl, ucl = geometric_limits(self.p0, self.gchart_alpha)
        d["gchart"] = {"center": center, "lcl": lcl, "ucl": ucl, "lower_limit_active": lcl > 0}
        d["baseline_weeks"] = list(self.baseline_weeks)
        # The exact weights, so a client re-implementing the recursion (the
        # static demo's JavaScript) adds the same doubles rather than
        # recomputing logs that may differ in the last bit.
        up = OnlineBernoulliCusum(self.p0, self.p1, self.h)
        lo = OnlineBernoulliCusum(self.p0, *self.lower, side="lower")
        d["weights"] = {
            "upper": {"fail": up.w_fail, "pass": up.w_pass},
            "lower": {"fail": lo.w_fail, "pass": lo.w_pass},
        }
        return d


@dataclass(frozen=True)
class ReplayRuns:
    """The run stream in time order: one entry per production run."""

    run_id: np.ndarray
    timestamp: np.ndarray
    year_week: np.ndarray
    phase: np.ndarray
    is_fail: np.ndarray
    watch_value: np.ndarray

    def __len__(self) -> int:
        return len(self.run_id)


#: Places ``results.json`` may hold the lower CUSUM design, most specific
#: first. Each is a key path to a mapping with ``p1`` and ``h``.
LOWER_DESIGN_KEYS: tuple[tuple[str, ...], ...] = (
    ("cusum", "lower"),
    ("cusum", "lower_cusum"),
    ("cusum", "cusum_lower"),
    ("cusum_lower",),
    ("lower_cusum",),
    ("shift_simulation", "design", "cusum_lower"),
)


def lower_cusum_design(
    results: dict[str, Any],
) -> tuple[float, float, float | None, str] | None:
    """(p1, h, arl0, source) of the lower CUSUM in ``results``, or None.

    The pipeline calibrates h by simulation to an in-control ARL of about 500
    runs, as it does for the upper chart. The first of
    :data:`LOWER_DESIGN_KEYS` present is used, and ``source`` names it.
    """
    for path in LOWER_DESIGN_KEYS:
        node: Any = results
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, dict) and "p1" in node and "h" in node:
            arl0 = node.get("arl0_runs", node.get("arl0"))
            return (
                float(node["p1"]),
                float(node["h"]),
                None if arl0 is None else float(arl0),
                "results.json: " + ".".join(path),
            )
    return None


def load_from_outputs(root: str | Path) -> tuple[MonitorDesign, ReplayRuns]:
    """Read the design and run stream from committed pipeline outputs.

    Uses ``reports/results.json`` for the CUSUM design and the surviving sensor,
    ``dim_sensor.csv`` for its Phase I limits, and ``fact_run.csv`` plus
    ``fact_measurement.csv`` for the stream. Run ``secom run`` first if these
    are missing.
    """
    root = Path(root)
    results_path = root / "reports" / "results.json"
    powerbi = root / "data" / "powerbi"
    for path in (results_path, powerbi / "fact_run.csv", powerbi / "dim_sensor.csv"):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `secom run` to produce it.")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    cusum = results["cusum"]
    survivors = results["sensors"]["hits_surviving_confound_check"]
    sensor = survivors[0] if survivors else results["watchlist"][0]

    dim = pd.read_csv(powerbi / "dim_sensor.csv").set_index("sensor_id")
    lim = dim.loc[sensor]
    lower = lower_cusum_design(results)
    lower_kw: dict[str, Any] = {}
    if lower is not None:
        lower_kw = {
            "lower_p1": lower[0],
            "lower_h": lower[1],
            "lower_arl0": lower[2],
            "lower_source": lower[3],
        }
    design = MonitorDesign(
        p0=float(cusum["p0"]),
        p1=float(cusum["p1"]),
        h=float(cusum["h"]),
        arl0=float(cusum["arl0_runs"]),
        baseline_weeks=(int(cusum["baseline_weeks"][0]), int(cusum["baseline_weeks"][1])),
        watch_sensor=sensor,
        watch_center=float(lim["baseline_center"]),
        watch_lcl=float(lim["baseline_lcl"]),
        watch_ucl=float(lim["baseline_ucl"]),
        watch_median=float(lim["baseline_median"]),
        watch_mad=float(lim["baseline_mad"]),
        **lower_kw,
    )

    runs = pd.read_csv(powerbi / "fact_run.csv").sort_values("run_index", kind="stable")
    meas = pd.read_csv(powerbi / "fact_measurement.csv", usecols=["run_id", "sensor_id", "value"])
    watch = meas.loc[meas["sensor_id"] == sensor].set_index("run_id")["value"]
    stream = ReplayRuns(
        run_id=runs["run_id"].to_numpy(),
        timestamp=runs["timestamp"].astype(str).to_numpy(),
        year_week=runs["year_week"].astype(str).to_numpy(),
        phase=runs["phase"].astype(str).to_numpy(),
        is_fail=runs["is_fail"].astype(bool).to_numpy(),
        watch_value=watch.reindex(runs["run_id"]).to_numpy(dtype=float),
    )
    return design, stream


# Session


@dataclass
class _Ack:
    alarm_id: int
    after_run: int
    note: str


@dataclass
class ReplaySession:
    """Steps the run stream through every monitor and keeps the history.

    ``cursor`` is the number of runs processed. :meth:`seek` replays from the
    start, so a jump is deterministic, and acknowledgements made at or before
    the target are re-applied on the way.
    """

    design: MonitorDesign
    runs: ReplayRuns
    history: list[dict[str, Any]] = field(default_factory=list, init=False)
    _acks: list[_Ack] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._reset_monitors()

    def _reset_monitors(self) -> None:
        d = self.design
        self.cusum = OnlineBernoulliCusum(d.p0, d.p1, d.h)
        self.cusum_lower = OnlineBernoulliCusum(d.p0, *d.lower, side="lower")
        self.gchart = OnlineGChart(d.p0, d.gchart_alpha)
        self.watch = WatchlistTracker(
            d.watch_sensor,
            center=d.watch_center,
            lcl=d.watch_lcl,
            ucl=d.watch_ucl,
            median=d.watch_median,
            mad=d.watch_mad,
        )
        self.alarms = AlarmBook()
        self.history = []
        self._fails = 0
        self._recent: deque[bool] = deque(maxlen=RECENT_WINDOW)

    @property
    def cursor(self) -> int:
        return len(self.history)

    @property
    def finished(self) -> bool:
        return self.cursor >= len(self.runs)

    def _process_next(self) -> dict[str, Any]:
        i = self.cursor
        r = self.runs
        fail = bool(r.is_fail[i])
        run, at = i + 1, str(r.timestamp[i])
        c = self.cusum.update(fail)
        lo = self.cusum_lower.update(fail)
        g = self.gchart.update(fail)
        w = self.watch.update(float(r.watch_value[i]))
        self._fails += fail
        self._recent.append(fail)

        if c.signal:
            self.alarms.signal(
                "cusum",
                "degradation",
                f"CUSUM {c.value:.2f} > h = {self.design.h:.2f}: failure rate above baseline",
                run,
                at,
            )
        if lo.signal:
            self.alarms.signal(
                "cusum_lower",
                "improvement",
                f"lower CUSUM {lo.value:.2f} > h = {self.cusum_lower.h:.2f}: "
                "failure rate below baseline",
                run,
                at,
            )

        point = {
            "run": run,
            "run_id": int(r.run_id[i]),
            "timestamp": at,
            "year_week": str(r.year_week[i]),
            "phase": str(r.phase[i]),
            "is_fail": fail,
            "cusum": c.value,
            "cusum_signal": c.signal,
            "cusum_lower": lo.value,
            "cusum_lower_signal": lo.signal,
            "gap": g.gap,
            "closed_gap": g.closed_gap,
            "gchart_signal": g.signal,
            "watch_value": w.value,
            "watch_z": w.robust_z,
            "watch_beyond": w.beyond,
            "fail_rate": self._fails / run,
            "recent_fail_rate": sum(self._recent) / len(self._recent),
        }
        self.history.append(point)
        for ack in self._acks:
            if ack.after_run == run:
                self.alarms.acknowledge(ack.alarm_id, run, at, ack.note)
        return point

    def step(self, n: int = 1) -> list[dict[str, Any]]:
        """Process up to ``n`` more runs and return their history points."""
        if n < 0:
            raise ValueError("n must be non-negative")
        out = []
        for _ in range(n):
            if self.finished:
                break
            out.append(self._process_next())
        return out

    def seek(self, run: int) -> None:
        """Replay from the start so that exactly ``run`` runs have been processed."""
        target = min(max(int(run), 0), len(self.runs))
        self._acks = [a for a in self._acks if a.after_run <= target]
        self._reset_monitors()
        self.step(target)

    def reset(self) -> None:
        """Back to run 0, discarding acknowledgements."""
        self._acks = []
        self._reset_monitors()

    def acknowledge(self, alarm_id: int, note: str = "") -> Alarm:
        """Acknowledge an open alarm at the current run."""
        if self.cursor == 0:
            raise AlarmError("nothing has run yet")
        at = str(self.runs.timestamp[self.cursor - 1])
        alarm = self.alarms.acknowledge(alarm_id, self.cursor, at, note)
        self._acks.append(_Ack(alarm_id, self.cursor, note))
        return alarm

    def snapshot(self, since: int = 0) -> dict[str, Any]:
        """JSON-ready state, with history points from index ``since`` onward."""
        since = min(max(int(since), 0), self.cursor)
        last = self.history[-1] if self.history else None
        recent_n = self.watch._recent.maxlen or RECENT_WINDOW
        return {
            "cursor": self.cursor,
            "n_runs": len(self.runs),
            "finished": self.finished,
            "current": last,
            "watch": {
                "sensor": self.design.watch_sensor,
                "recent_beyond": int(sum(self.watch._recent)),
                "recent_n": len(self.watch._recent),
                "window": recent_n,
                "expected_by_chance": CHANCE_EXCEEDANCE * len(self.watch._recent),
            },
            "open_alarms": [a.as_dict() for a in self.alarms.open_alarms],
            "alarms": [a.as_dict() for a in self.alarms.alarms],
            "log": list(self.alarms.log),
            "since": since,
            "history": self.history[since:],
        }


def phase_spans(runs: ReplayRuns) -> list[dict[str, Any]]:
    """Contiguous blocks of the phase label, as 1-based inclusive run ranges."""
    spans: list[dict[str, Any]] = []
    for i, label in enumerate(runs.phase, start=1):
        if spans and spans[-1]["label"] == label:
            spans[-1]["end"] = i
        else:
            spans.append({"label": str(label), "start": i, "end": i})
    return spans


def design_payload(design: MonitorDesign, runs: ReplayRuns, speed: float) -> dict[str, Any]:
    """The design as the page reads it: ``/api/design`` and the static export."""
    d = design.as_dict()
    d["n_runs"] = len(runs)
    d["phases"] = phase_spans(runs)
    d["default_speed"] = speed
    return d


def json_safe(obj: Any) -> Any:
    """Replace NaN/inf with None, recursively, so the output is strict JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [json_safe(v) for v in obj]
    return obj


def session_from_outputs(root: str | Path) -> ReplaySession:
    design, runs = load_from_outputs(root)
    return ReplaySession(design, runs)
