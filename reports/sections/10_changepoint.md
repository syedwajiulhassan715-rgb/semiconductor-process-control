## 10. Where the yield changed, located at run grain

![Change-point profile likelihood](figures/fig12_changepoint.png)

The Phase I baseline starts at ISO week 35 because that is where the weekly rates
settle (**D7**). That is a judgement made at week grain. This section locates the
change at run grain instead, without a week boundary (**D33**):
- **The estimate:** the exact maximum-likelihood single change point of the
  1,567-run pass/fail sequence. The profile likelihood is evaluated at every
  split, so no search heuristic is involved.
- **The test:** a likelihood-ratio test of "no change".
- **The interval:** a parametric-bootstrap confidence interval for the change
  location.
- **Further changes:** binary segmentation to look for additional change points.

### Result

| | Before | After |
|---|---:|---:|
| Runs | 374 | 1,193 |
| Failures | 53 | 51 |
| Rate | **14.17%** | **4.27%** |

**The change falls at run 375, 21 Aug 2008 18:39, in ISO week 34.** The rate
drops by a factor of **3.31**, with LR = 38.59. The 95% bootstrap interval runs
from run 280 (18 Aug 23:03, W34) to run 463 (28 Aug 03:45, W35).

The evidence is unambiguous under either null:
- Run-level permutation and day-block permutation (86 calendar-day blocks) both
  give p = 0.0005. That is the smallest value 2,000 permutations can produce,
  so p < 0.001.
- The block null matters. Under day-block permutation, the 95th percentile of
  the maximum LR is **18.07**; under run-level permutation it is 10.42.
  Failures cluster within days, which roughly doubles the evidence needed to
  call a change, and the observed 38.59 clears both.

### Agreement with the Phase I boundary

**The change point agrees with D7.** The first baseline run (run 450, 25 Aug
09:29, W35) lies inside the interval, 75 runs (86.8 hours) after the point
estimate. Starting the baseline at the week boundary is slightly conservative:
the transition stays out of Phase I under the point estimate. Even at the far
end of the interval, only 13 of the 717 baseline runs could precede the change.

The run-level estimate sharpens one statement in the report. The "pre-baseline"
regime in Sections 2 and 3 is weeks ≤ 34 (12.92%). It includes the 75 runs of
week 34 that already belong to the improved process. Split at the estimated run
instead, the contrast is 14.17% against 4.27%.

### No second change point, and what that implies

Binary segmentation splits a segment while its day-block permutation p stays
below 0.05. It keeps only the one change:

| Segment tested | Best split | LR | p (run-level) | p (day-block) |
|---|---:|---:|---:|---:|
| Runs 1–374 | run 66 | 4.60 | 0.366 | 0.615 |
| Runs 375–1,567 | run 1,445 | 5.75 | 0.329 | 0.722 |

The second line matters for the monitoring design. After the change, the
process runs at 4.27% overall, and the data cannot separate the Phase I weeks
(3.07%) from W40–42 (5.99%) as distinct regimes. The CUSUM's four Phase II alarm
episodes (Section 3) are measured against the Phase I rate. This result says
they may reflect ordinary variation of the post-change process, not a new
degradation.

**The boundary of the baseline is right; its level may be optimistic.** If
W35–39 was a favourable stretch of a process whose sustained rate is nearer 4.3%,
then a CUSUM designed on p₀ = 3.07% will alarm more often than its designed
ARL₀. p₀ should be re-estimated as post-change data accumulates. This is a
caveat on the recommendation, not a reversal of it.

### Limits

- **The bootstrap interval assumes independent Bernoulli runs.** Day-level
  clustering makes the real interval wider than the one shown. The permutation
  tests account for the clustering; the interval does not.
- **Binary segmentation does not adjust alpha for the number of tests.** Each
  later test is also conditional on the earlier split, so treat secondary splits
  as descriptive. Here none were accepted, so this only matters in the
  conservative direction.
- **No segment shorter than 30 runs is considered.** A brief excursion of a day
  or less would not be found by this method; that job belongs to the CUSUM.
- **Locating the change is not explaining it.** A process change and a change
  in the test limits or gauge would look identical (**D24**).
