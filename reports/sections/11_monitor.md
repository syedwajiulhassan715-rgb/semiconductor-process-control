## 11. The monitoring scheme, operating

The recommendation is a Bernoulli CUSUM against the week 35–39 baseline as the
live yield monitor, a g-chart alongside it as an improvement detector, and S060
on a watchlist. `secom monitor` runs that scheme through the 1,567 SECOM runs in
time order, one run at a time, so the recommendation can be seen working rather
than read about.

<p align="center">
  <img src="../../docs/img/monitor.png" width="100%" alt="Replay monitor at run 1,310: CUSUM trace with alarms, g-chart, S060 watchlist panel and the alarm log">
</p>
<p align="center">
  <sub><b>Run 1,310, week 40.</b> The opening excursion has been worked as two acknowledged
  alarms. A third, raised at run 606 in the baseline window, is open. The g-chart has never
  approached its 212-run UCL. S060 is shown against its Phase I limits, for information only.
  A dark-theme capture is in <code>docs/img/monitor-dark.png</code>.</sub>
</p>

### Running it

```bash
secom monitor                 # http://127.0.0.1:8050/
secom monitor --speed 50 --open
```

| Flag | Default | Purpose |
|---|---|---|
| `--port` | 8050 | Port to listen on |
| `--host` | 127.0.0.1 | Interface to bind |
| `--speed` | 20 | Initial playback speed, runs per second |
| `--open` | off | Open the page in the default browser |
| `--root` | cwd | Repository root holding `reports/` and `data/powerbi/` |

The page has play/pause, single step (→), speed, jump-to-run and reset controls.
Space toggles playback. URL options `?theme=light|dark`, `?speed=N` and
`?autoplay=1` are there for demos and screenshots.

It starts in about a second because it reads the committed pipeline outputs
(`reports/results.json` and `data/powerbi/`) instead of rerunning the analysis.
After changing the config, run `secom run` first so the monitor picks up the new
design. Nothing beyond the core dependencies is required. The server is the
standard library, and the page loads Chart.js 4.4.1 from cdnjs.

### What each panel is

**CUSUM.** It shows the statistic, the decision interval h = 1.822, a marker on
every signalling run, and bands for the three regimes. The values are the batch
CUSUM's, not a lookalike. The online implementation reproduces
`spc.bernoulli_cusum` bit for bit, and the exported `fact_run.cusum` column to
the CSV's 4×10⁻¹⁶ rounding. Both are asserted in `tests/test_monitor.py`.

**Alarms.** This panel is the operational layer the batch chart does not have,
and the distinction matters (D35):

- **The statistic resets the moment it signals (D13).** The chart starts again
  against the process as it now stands.
- **The alarm does not reset.** It stays open until someone acknowledges it,
  with an optional note for the shift log. Signals while it is open count
  against it instead of raising new alarms, so one excursion is one
  investigation.

Left unacknowledged, all 21 signalling runs of the campaign fold into a single
alarm raised at run 12, 0.46 days in. That is the same detection time as §3,
seen from the line.

**g-chart.** It shows passing runs since the last failure against exact
geometric limits at the Phase I rate (UCL 212, centre 31.6). The open gap is
tracked live, so an improvement would be signalled on the run where the gap
crosses the UCL, without waiting for the next failure to close it. Over this
campaign it never signals: the longest gap is 99 runs. The page states that the
lower limit is inactive at this rate (D11), so the chart cannot be mistaken for
a degradation detector.

**S060.** Each reading is plotted against S060's Phase I individuals limits,
with failed runs marked and the exceedance count over the last 50 runs shown
against the 0.14 expected by chance. It never raises an alarm. 278 of 1,567
readings (17.7%) fall outside limits that expect 0.27%, because the sensor
drifts. An alarm on it would fire continually and teach the line to ignore
alarms. That is the practical meaning of "watchlist, not root cause" (D36).

### API

The page is a thin client over a JSON API that can be scripted directly:

| Method | Path | Body | Effect |
|---|---|---|---|
| GET | `/api/design` | – | Fixed design: p₀, p₁, h, ARL₀, g-chart and S060 limits, phase spans |
| GET | `/api/state?since=K` | – | Current state plus history points from index K |
| POST | `/api/step` | `{"n": 5}` | Advance n runs (at most 2,000 per call) |
| POST | `/api/seek` | `{"run": 400}` | Replay from the start to run 400 |
| POST | `/api/reset` | `{}` | Back to run 0; acknowledgements cleared |
| POST | `/api/ack` | `{"alarm_id": 1, "note": "…"}` | Acknowledge at the current run (409 if unknown or closed) |

Every response is a deterministic function of the requests before it. A seek
replays from run 0 and re-applies acknowledgements made at or before the
target. A seek back past an acknowledgement removes it, as if the
acknowledgement had not happened yet.

### Where it lives

| File | Contents |
|---|---|
| `src/secom/monitor.py` | Online CUSUM, g-chart and watchlist trackers, alarm book, replay session, loader |
| `src/secom/monitor_server.py` | Standard-library HTTP server and JSON API |
| `src/secom/monitor_static/index.html` | The page (light and dark themes) |
| `tests/test_monitor.py` | 47 tests: batch equivalence, alarm semantics, replay determinism, the HTTP API |
