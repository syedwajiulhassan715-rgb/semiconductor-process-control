# Replay monitor

`secom monitor` replays the 1,567 SECOM runs in time order through the
recommended monitoring scheme: the Bernoulli CUSUM against the week 35–39
baseline as the yield monitor, the g-chart beside it as an improvement detector,
and S060 on a watchlist. The page shows that scheme as a line engineer would
see it, one run at a time.

```bash
secom monitor                    # http://127.0.0.1:8050/
secom monitor --speed 50 --open  # faster playback, open a browser
```

=== "Light"

    ![Replay monitor at run 1,310: CUSUM trace with alarms, g-chart, S060 watchlist panel and the alarm log](img/monitor.png)

=== "Dark"

    ![The same view in the dark theme](img/monitor-dark.png)

## What it shows

- **CUSUM** with the decision interval h, a marker on every signalling run, and
  the three regimes shaded. The online implementation reproduces
  [`bernoulli_cusum`][secom.spc.bernoulli_cusum] exactly; a test asserts it.
- **Alarms.** The statistic resets the moment it signals (D13), but the alarm
  stays open until someone acknowledges it, with an optional note. Further
  signals count against the open alarm, so one excursion is one investigation.
- **g-chart** of passing runs since the last failure, tracked live against exact
  geometric limits. At this failure rate its lower limit is inactive (D11), and
  the page says so.
- **S060** against its Phase I individuals limits, for information only. It
  never raises an alarm: the sensor drifts, so an alarm on it would fire
  continually.

The page has play/pause, single step, speed, jump-to-run and reset controls.
URL options `?theme=light|dark`, `?speed=N` and `?autoplay=1` are there for
demos.

## How it is built

It reads the committed pipeline outputs (`reports/results.json` and
`data/powerbi/`), so it starts in about a second. After changing the
configuration, run `secom run` first so the monitor picks up the new design.

The server is the Python standard library, and the page loads Chart.js from
cdnjs, so nothing beyond the core dependencies is needed. The page is a thin
client over a JSON API (`/api/design`, `/api/state`, `/api/step`, `/api/seek`,
`/api/reset`, `/api/ack`) that can be scripted directly. See
[`secom.monitor`][secom.monitor] and [`secom.monitor_server`][secom.monitor_server],
and the operating section of the [engineering report](report.md) for the design
decisions behind it (D34–D36).

To serve it from the Docker image, bind to all interfaces:

```bash
docker run --rm -p 8050:8050 secom-process-control monitor --host 0.0.0.0
```
