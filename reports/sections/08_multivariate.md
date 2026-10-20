## 8. Multivariate monitoring: PCA T² and SPE

Fab FDC systems don't watch 412 sensors one at a time. They fit a PCA model to
a known-good window and run two charts:

- **Hotelling's T²** flags runs that sit unusually far out *inside* the model's
  component space.
- **SPE (Q)** flags runs whose sensor pattern the model can't reproduce.

This section asks whether that chart adds anything to the Bernoulli CUSUM.
Specifically, does it detect the yield degradation, or does it mostly detect
the sensor drift that §4 showed affects 71% of retained sensors? Every number
here is under `multivariate` in `results.json`.

### The model

Fitted once on the Phase I baseline (weeks 35–39, 717 runs) and held fixed
for every run (**D26**):

- 411 of the 412 retained sensors. S075 is constant within the baseline and is
  dropped.
- Scaling uses the baseline mean and SD. Robust scaling isn't possible here,
  because 8 sensors have a baseline MAD of zero.
- **A = 81 components**, chosen by an 80% cumulative-variance rule (**D28**).
  They explain 80.1% of baseline variance. The spectrum is flat: the first
  component explains only 6.7%, and 105 eigenvalues exceed 1.
- The baseline has full rank (411), but its smallest eigenvalue is 3.5×10⁻¹¹.
  T² is therefore computed only from the retained components, whose condition
  number is 18.9; the full covariance is never inverted.

### The textbook limits do not hold on this data

The standard limits at α = 0.01 per chart are:

- **T²:** exact F limit 131.4 for new runs; Beta limit 110.9 for the baseline
  points themselves.
- **SPE:** Jackson–Mudholkar limit 111.5.

Applied to the baseline that defined them, **15.8% of baseline runs alarm**
against a nominal 2% or less: 12.0% on T² and 7.5% on SPE. Under these limits
the chart flags **86% of the degraded-period runs and 83% of Phase II runs**.

That looks like overwhelming detection, but it is an artefact. Alarmed runs in
the degraded period fail at 13.1%, quiet ones at 11.5% (Fisher p = 0.84). The
chart is flagging nearly everything because the limits are wrong. The cause is
the heavy tails: the limits assume multivariate normality, but the top
contributors to the alarms have excess kurtosis in the hundreds.

The limits are therefore recalibrated (**D27**):

1. Score every baseline run **out of fold**, using a model fitted on the other
   four contiguous time blocks of the baseline.
2. Set each limit at the 99% quantile of those out-of-fold scores.

This gives T² = 902.7 and SPE = 2,204.5, with 8 of 717 baseline runs above
each, so 1.1% of baseline runs alarm. α stays at 0.01 rather than the 3-sigma
0.27% used elsewhere. The out-of-fold statistics have excess kurtosis of 712
(the largest T² is 1.4 million), so their 99.73% quantile rests on about two
runs and can't be estimated: it comes out at 8,328.

### What the calibrated chart sees

<p align="center"><img src="figures/fig07_mspc_t2_spe.png" width="100%" alt="T² and SPE across the campaign with cross-validated and theoretical limits"></p>

| Window | Runs | Fail rate | Runs alarming (T² or SPE) | Alarmed vs quiet fail rate | Fisher p |
|---|---:|---:|---:|---|---:|
| Pre-baseline (degraded) | 449 | 12.9% | 6.9% (31) | 29.0% vs 11.7% | 0.011 |
| Phase I baseline (out of fold) | 717 | 3.1% | 1.1% (8) | 12.5% vs 3.0% | 0.22 |
| Phase II (forward) | 401 | 6.0% | 4.7% (19) | 0.0% vs 6.3% | 0.62 |

**The monitoring statistic moves with time, not with yield.** Compared with
the out-of-fold baseline, median SPE is higher in the degraded period
(157 vs 133, Cliff's δ = 0.29, p = 2×10⁻¹⁷), and still higher in Phase II
(139, δ = 0.13, p = 3×10⁻⁴). The Phase II shift appears even though Phase II
yield is better. T² behaves the same way (δ = 0.18 and 0.12). The chart is
measuring how far a run sits from the baseline weeks, and the sensor drift
moves runs away from those weeks in both directions in time.

**Alarms mark failures in one window only.** In the degraded period, the 31
alarmed runs fail at 29.0% against 11.7% for the rest (odds ratio 3.08,
p = 0.011). It doesn't replicate:

- In the baseline, 8 alarms give p = 0.22.
- In Phase II, none of the 19 alarmed runs failed.

How robust is it? The degraded-period association holds across the number of
components: p ranges from 0.001 to 0.012 for A = 30 to 150. It does not hold
across the alarm threshold:

| α per chart | 0.05 | 0.02 | 0.01 | 0.005 |
|---|---:|---:|---:|---:|
| Fisher p, degraded period | 0.092 | 0.045 | 0.011 | 0.50 |

Three regimes were tested, so p = 0.011 alone does not survive a Bonferroni
correction at 0.05/3 by a wide margin. It is a lead, not a finding. The
whole-campaign association (17.2% vs 6.2%, p = 0.004) is the same time
confound §4 removed from the sensor screen: alarms concentrate in the weeks
that also had the most failures.

**Detection speed is not comparable with the CUSUM.** The chart's first
degraded-period alarm comes at run 4 (0.12 days), against the CUSUM's 12 runs.
But a 1.1% per-run false-alarm rate implies an in-control ARL of about 90
runs, against the CUSUM's 509, and the baseline's own first false alarm comes
at run 9. The earlier first signal reflects a chart that alarms more often,
not one that detects better.

### Who drives the alarms

<p align="center"><img src="figures/fig08_mspc_contributions.png" width="85%" alt="Top SPE contributors in alarmed runs outside the baseline"></p>

The 29 SPE alarms outside the baseline are not spread across the drifting
sensors:

- They are concentrated in a few sensors with extreme tails. S008 carries
  15.5% of the SPE in an average alarmed run and S005 11.2%, against an equal
  share of 0.24%.
- The top 15 sensors carry 57%. Their median excess kurtosis is 276, against
  14 across all sensors.
- In 38% of alarmed runs, one sensor alone is more than half the statistic.
- A sensor's contribution share is unrelated to its drift (Spearman ρ = −0.02,
  p = 0.64). For T², the relationship is slightly negative (ρ = −0.11).

So the calibrated alarms are mostly **outlier events on individual
heavy-tailed sensors**, not a broad multivariate departure. The broad drift
shows up in the median statistic (above) but rarely crosses a limit.

**S060 does not appear.** It ranks 259th of 411 in SPE contribution and 192nd
in T². This is consistent with §4: S060's association with failure is a shift
small enough to be invisible in a 411-sensor statistic. Contribution analysis
cannot substitute for the targeted screen.

<p align="center"><img src="figures/fig09_mspc_sensitivity.png" width="100%" alt="Alarm rates and within-regime Fisher p-values against the number of components"></p>

The component count doesn't change any of this:

- Across A = 30 to 150, the out-of-fold baseline alarm rate stays at 1.1–1.3%.
- The theoretical limits stay miscalibrated at 13.4–16.0%.
- Phase II never shows a within-regime association (p = 0.62 at every A).

### Verdict

Don't adopt the PCA chart as the yield monitor (**D29**):

- With honest limits it flags 6.9% of runs in the period when yield was worst.
- The alarms that do fire are mostly single-sensor outliers.
- They mark failures in one of three windows only, and that result is
  sensitive to the threshold.

As textbook limits would deploy it, the chart would alarm on 86% of production,
which is worse than having no chart. Its practical use here is as an
**excursion flag for the handful of heavy-tailed sensors**, with limits taken
from out-of-fold baseline quantiles, never from the F or Jackson–Mudholkar
formulas. The Bernoulli CUSUM remains the yield monitor.
