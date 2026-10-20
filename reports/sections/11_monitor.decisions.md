## Replay monitor

### D34. The monitor reads committed outputs and runs on the standard library

`secom monitor` loads its design from `reports/results.json` and
`data/powerbi/` instead of rerunning the pipeline. It starts in about a second,
and it shows exactly the design the report quotes: p₀, p₁ and h come from the
same `results.json` the write-up cites.

The server is `http.server` and the page loads one pinned chart library
(Chart.js 4.4.1) from a CDN. A Streamlit or Dash app would mean a heavy
dependency tree for a single page, and on locked-down machines (this project
already hits a blocked matplotlib DLL) every compiled extension is another
thing that may refuse to load.

Correctness is pinned by test, not by eye. The online CUSUM reproduces
`spc.bernoulli_cusum` bit for bit on random streams, with and without reset,
and matches the exported `fact_run.cusum` to the CSV's 4×10⁻¹⁶ rounding.

**If wrong:** the monitor goes stale if someone changes the config and does not
rerun `secom run`. The loader names the files it read, and the page displays the
design it is running.

### D35. The statistic resets on signal; the alarm does not

D13 resets the CUSUM the moment it signals, which is right for the statistic.
An engineer is still investigating, though, so the monitor separates the two:

- The **statistic** resets and keeps watching.
- The **alarm** stays open until acknowledged, with an optional note.
- **Repeat signals** while an alarm is open count against that alarm instead of
  raising new ones.

On this campaign the difference is large. The CUSUM signals on 21 runs. Left
unacknowledged, they are one alarm raised at run 12. Raising 21 alarms for one
degraded period is how a line learns to ignore its monitor.

**If wrong:** an acknowledged-too-early alarm could hide a second, unrelated
excursion behind the first. Acknowledgement closes the alarm, so the next signal
opens a fresh one; nothing is suppressed after an acknowledgement.

### D36. The monitor's g-chart uses the Phase I rate; S060 never alarms

The exported `fact_gchart` uses the all-data rate (p = 6.64%, UCL 96). That
chart answers the retrospective question. A monitor must be designed on the
baseline it protects, so the monitor's g-chart uses p₀ = 3.07% (UCL 212), the
same Phase I design as the CUSUM. It signals improvement as soon as the open gap
crosses the UCL, without waiting for the next failure. Over this campaign it
never signals: the longest gap is 99 runs. The page says so and states the
inactive lower limit (D11).

S060 is shown against its Phase I individuals limits, but it is informational
only. 278 of 1,567 readings (17.7%) are outside limits that expect 0.27%,
because the sensor drifts. An alarm would fire continually. The panel reports
recent exceedances against chance instead, which is what a watchlist is for.

**If wrong:** if S060 is causal after all, the designed experiment D24 calls for
is the way to promote it to an alarm, with limits set on a stable process. That
is not something to decide from this panel.
