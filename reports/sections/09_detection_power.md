## 9. Detection power, measured rather than argued

![Median runs to first signal](figures/fig10_detection_delay.png)

Sections 2 and 3 argue that the weekly p-chart cannot see the yield shift and
the Bernoulli CUSUM can. That argument rests on control-limit algebra and on one
realised campaign. This section measures it. Pass/fail streams are simulated with
the fail rate stepping from the Phase I rate p₀ = 3.07% to r·p₀, and every scheme
is run over the same streams with the limits and designs the pipeline actually
uses (**D30**). There are 4,000 streams per ratio, and the ratios include the
two observed ones: **4.21×** for W29–34 and **1.95×** for W40–42.

The Shewhart results don't depend on the simulation alone. Subgroups are
independent binomials, so every p-chart detection probability also has an exact
value. The 99 simulated Shewhart probabilities agree with the exact ones to
within **0.0146**, against a worst-case Monte Carlo standard error of 0.0079.

### The comparison has to be matched

As deployed, the two schemes are not on equal terms. The CUSUM was designed for
an in-control ARL of ~500 runs (**D12**), which gives it a **96.6%** chance of at
least one false alarm over a 1,567-run campaign. The naive weekly p-chart's
chance is **7.3%**. Comparing them as deployed partly compares false-alarm
budgets. So each one-sided CUSUM is also run with h chosen to match the
false-alarm probability of the chart it is compared with (**D31**):

- **Upper CUSUM, matched to the p-chart:** h = 4.996, false-alarm probability
  7.6% simulated against a target of 7.2%.
- **Lower CUSUM, matched to the g-chart:** h = 4.626, 4.8% against a target of
  5.0%.

A detection only counts if the signal points the same way as the shift
(**D32**). For example, when the fail rate falls, a p-chart point above the UCL
is a false alarm, not a detection.

### Degradation

| Scheme | False alarm in 1,567 runs | 2×: P(signal ≤ 400 runs) | 2×: median runs | 4.21×: P(signal ≤ 120 runs) | 4.21×: median runs |
|---|---:|---:|---:|---:|---:|
| Weekly p-chart (naive) | 0.073 | 0.496 | 401 | 0.402 | 138 |
| Weekly Laney p′, σ_z = 1.463 | 0.004 | 0.128 | — | 0.291 | 156 |
| Weekly Laney p′, σ_z = 2.250 | 0.000 | 0.005 | — | 0.084 | 249 |
| **CUSUM as deployed** | 0.966 | **0.992** | **76** | **0.998** | **26** |
| **CUSUM matched to p-chart** | 0.076 | **0.700** | **292** | **0.871** | **76** |

"—" means fewer than half the streams signal within 1,567 runs. The 2× row
stands in for W40–42, which was observed at 1.95×; the simulation gives nearly
identical figures there (0.990 and 0.659 for the two CUSUMs within 400 runs).

**The CUSUM's advantage survives matching, but it is smaller than the deployed
design suggests.** At equal false-alarm risk:
- At a doubling, the CUSUM catches the shift within 400 runs 70% of the time,
  against 50% for the p-chart, with a median delay of 292 runs against 401.
- At the degradation actually observed in W29–34, it signals within about a
  week (120 runs) 87% of the time against 40%, with a median delay of 76 runs
  against 138.
- The deployed design's headline figure (99% within 400 runs at 2×) is mostly
  bought with a much larger false-alarm budget: 97% per campaign against 7%.

The real structural gap is in the p-chart's timing, not its sensitivity. A
weekly chart can only signal at the end of a week, so its median delay never
falls below about one subgroup. Even at 4.21×, the p-chart's median is 138
runs, while the deployed CUSUM's is 26.

**The Laney correction costs real power.** At a doubling, Laney p′ signals
within 400 runs in 13% of streams against 50% for the naive chart. With the
Phase I-centred σ_z of 2.250 that falls to 0.5%. That is the price of being
honest about overdispersion (**D9**) at this grain. The correction is right,
and the chart it produces is close to inert.

![Probability of signalling within 400 runs](figures/fig11_detection_probability.png)

### Improvement

| Scheme | False alarm in 1,567 runs | 0.5×: P(signal ≤ 1,567) | 0.5×: median runs | 0.33×: P(signal ≤ 400) | 0.33×: median runs |
|---|---:|---:|---:|---:|---:|
| Weekly p-chart, either correction | — | **0.000** | — | **0.000** | — |
| g-chart | 0.052 | 0.605 | 1,204 | 0.326 | 582 |
| Lower CUSUM as deployed | 0.973 | 1.000 | 148 | 0.994 | 130 |
| Lower CUSUM matched to g-chart | 0.048 | 0.949 | 705 | 0.294 | 476 |

**The blind spot is total at the monitoring centre line.** Against p₀ = 3.07%, a
weekly subgroup needs n ≥ 9(1−p₀)/p₀ = **284** runs before the lower limit clears
zero. The largest week has 208, so the LCL is zero in **all 14** weeks. The
improvement-detection probability is therefore exactly zero for every ratio and
horizon, both in simulation and in the exact calculation. This is stronger than
the 7 of 14 weeks in Section 2, which uses the all-weeks centre p̄ = 6.64%. At
the centre a monitoring chart actually uses, no week can show an improvement.

The g-chart covers the gap, but slowly. With UCL = 212 it cannot signal before
213 consecutive passes. At equal false-alarm risk the lower CUSUM is better over
a full campaign: it catches a halving 95% of the time against 61%, with a median
of 705 runs against 1,204. Over shorter horizons the two are close, and the
g-chart is slightly ahead within 400 runs (0.326 against 0.294 at 0.33×),
because the matched CUSUM's large h needs a long run of passes before it can
signal at all.

### What this changes

- **The Section 3 recommendation stands, on firmer ground.** The CUSUM beats
  the p-chart at every degradation ratio, including at matched false-alarm
  risk.
- **The deployed CUSUM's operating cost is now explicit.** In control, half of
  all streams false-alarm within 342 runs. At the observed cadence of 1,567
  runs in 89 days, that is under three weeks. The baseline self-check (2 alarm
  episodes in 717 runs) is consistent with that. It is a design choice, not a
  defect, and whoever runs the chart should be told it.
- **If detecting improvement matters,** a lower CUSUM should replace the
  g-chart as the improvement detector. The g-chart is defensible only as a
  readable companion chart.

### Limits of the simulation

- **Streams are independent Bernoulli.** The real weekly rates are
  overdispersed (σ_z = 1.46). Under overdispersion every scheme false-alarms
  more than simulated, and run-level accumulators like the CUSUM are affected
  most. The in-control figures above are lower bounds on real false-alarm
  rates.
- **Shewhart timing is favourable.** Shifts start at run 1, so the first weekly
  subgroup is wholly post-shift. A mid-week shift would cost the weekly charts
  up to one more week, so their delays here are best cases.
- **Only step shifts are modelled.** Gradual drifts are not, and an EWMA was not
  simulated.
