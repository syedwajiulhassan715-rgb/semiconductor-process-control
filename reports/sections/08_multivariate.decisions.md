## Multivariate monitoring

### D26. PCA model: Phase I autoscaling, and only retained components are inverted

The PCA model is fitted on the Phase I baseline only (717 runs) and held fixed.
Sensors are scaled by their baseline mean and SD, not by median / MAD. The T²
and SPE limit theory is built on the sample mean and covariance, and 8 sensors
have a baseline MAD of zero, so a robust z is undefined for them. S075 is
constant within the baseline and is left out, leaving 411 sensors.

The baseline matrix has full rank (411 < 717), but its smallest eigenvalue is
3.5×10⁻¹¹. T² is therefore computed from the retained components only (condition
number 18.9), never from the inverse of the full covariance.

**If wrong:** a T² built on the full inverse is dominated by directions with
near-zero variance, so floating-point noise would decide which runs alarm.

### D27. Limits from out-of-fold baseline quantiles, at α = 0.01

The F / Beta limits for T² and the Jackson–Mudholkar limit for SPE assume
multivariate normality. On this data they put **15.8% of baseline runs** over
the limit, against a nominal 2% or less. Under those limits the chart flags 86%
of the degraded period and 83% of Phase II, and the flags have no relationship
with failure (p = 0.84).

Each baseline run is instead scored by a model fitted on the other four
contiguous time blocks. Each limit is set at the (1 − α) quantile of those
out-of-fold scores. In-sample scores would understate SPE, because the
components were fitted to those very rows.

α is 0.01 per chart, not the 3-sigma 0.0027 used for the univariate charts. The
out-of-fold statistics have excess kurtosis of about 712. Their 99.73% quantile
rests on about two of 717 runs, so it can't be estimated (it lands at T² = 8,328,
which is set by a few gross outliers).

**If wrong:** with the theoretical limits, the chart's apparent sensitivity is
pure miscalibration. A smaller α is not usable until there is a longer baseline
to estimate the tail from. Both limit schemes are kept in `results.json` so the
contrast stays visible.

### D28. Number of components: 80% of baseline variance, with a sensitivity sweep

A = 81 components, the smallest number explaining at least 80% of baseline
variance. The spectrum is flat (the first component explains 6.7%, and 105
eigenvalues exceed 1), so no elbow rule applies. The analysis is re-run at
50–95% (A = 30 to 150):

- The out-of-fold baseline alarm rate stays at 1.1–1.3%.
- The theoretical baseline rate stays at 13.4–16.0%.
- The degraded-period failure association stays at p = 0.001–0.012.
- Phase II never shows an association (p = 0.62 throughout).

**If wrong:** nothing in the verdict depends on A within that range. A outside
it was not tested.

### D29. The PCA chart is not adopted as the yield monitor

With calibrated limits it alarms on 6.9% of runs in the period when yield was
worst. The alarms are mostly single-sensor outliers: the top 15 contributors
have a median excess kurtosis of 276, and in 38% of alarmed runs one sensor is
more than half the SPE. A sensor's contribution is unrelated to its drift
(ρ = −0.02). The alarms mark failures only in the degraded period (p = 0.011),
which does not survive correcting for the three windows tested and shifts with
α (p = 0.092 at α = 0.05, 0.50 at α = 0.005).

The run-to-first-signal is not compared with the CUSUM as a race. The PCA
chart's 1.1% per-run false-alarm rate implies an in-control ARL of about 90
runs, against the CUSUM's 509, so its earlier first signal (run 4 vs 12)
reflects a chart that alarms more often.

S060 ranks 259th of 411 in SPE contribution, so contribution plots do not
rediscover the one sensor that survived the confound gate.

**If wrong:** if the degraded-period association is real, the chart is a useful
excursion flag in a degraded process. That is a reason to keep it as a
secondary flag on the heavy-tailed sensors, with out-of-fold limits, and
re-test it against new data. It is not a reason to replace the CUSUM.
