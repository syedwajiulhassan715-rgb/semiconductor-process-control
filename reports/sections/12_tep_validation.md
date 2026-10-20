## 12. Validation on a benchmark with known faults

SECOM has no ground truth. Nobody recorded which runs were faulty or when a
fault began, so nothing in sections 1–8 can show that the monitoring code
detects a fault it should. The Tennessee Eastman Process (TEP) benchmark has
21 faults injected at known times. This section runs the repo's own library
code on it, unchanged, and compares the results against the published PCA
benchmark on the same files.

Run it with `secom fetch --dataset tep` then `secom validate tep`. It is a
separate command, not a pipeline stage, so the SECOM analysis never needs this
second dataset. Every number below is written to
`reports/validation/tep_results.json`.

### Setup

The data are the Braatz training and testing files used by Chiang, Russell &
Braatz (2001): 52 variables (41 measurements, 11 manipulated variables)
sampled every 3 minutes. The models are trained on `d00.dat`, 500 samples of
normal operation. Each of the 22 testing files has 960 samples, with the fault
introduced after sample 160, so samples 0–159 are normal and 160–959 are
faulty. `d00_te.dat` is a second, independent normal run.

Three charts are applied, all at α = 0.01 per chart, the benchmark's setting
and the SECOM multivariate stage's:

- **PCA T²** and **PCA SPE** from `secom.mspc`, with A = 11 components
  (54.2% of the training variance). The textbook F and Jackson–Mudholkar
  limits are primary, because they are what the benchmark uses. The
  cross-validated limits from the SECOM multivariate section are reported
  alongside.
- **Univariate**: one Shewhart chart per variable, with Šidák-widened limits
  (k = 3.728, per-chart α = 1.9×10⁻⁴). The family of 52 charts therefore
  false-alarms at α per sample.

The benchmark detection rates come from the PCA missed-detection table in
Chiang et al. (2001), **transcribed from secondary reproductions and not
checked against the book in this work**. They are used as a qualitative
reference, and no test asserts them exactly.

### The implementation reproduces the benchmark

<p align="center"><img src="figures/fig13_tep_detection.png" width="100%" alt="Detection rate per TEP fault for T², SPE and the univariate scheme, with benchmark markers"></p>

| | T² | SPE |
|---|---:|---:|
| Mean \|ours − benchmark\| over 21 faults | 0.027 | 0.074 |
| Faults within ±0.05 of the benchmark | 17 / 21 | 10 / 21 |
| Correlation with the benchmark across faults | 0.999 | 0.985 |
| Largest gap | fault 11 (0.077) | fault 16 (0.255) |

The qualitative facts all hold, and the test suite asserts each of them:

- **Easy faults are caught.** Faults 1, 2, 6, 7, 8, 12, 13, 14 and 18 are
  detected on more than 85% of faulty samples by SPE. Fault 1 reaches 0.993
  on T² and 0.998 on SPE, against a benchmark of 0.992 and 0.997.
- **Fault 4 is a residual fault.** The reactor cooling-water step is invisible
  to T² (0.087) and obvious to SPE (0.996). The benchmark gives 0.044 and
  0.962.
- **Faults 3, 9 and 15 are undetectable, as published.** Detection stays at
  0.026–0.050 on T² and 0.081–0.120 on SPE. That is the false-alarm rate, not
  detection. Faults 3 and 9 never produce six consecutive T² or SPE alarms
  in the 800 faulty samples. Fault 15 first does so after 674 samples (34
  hours), which no one would call a detection.

The SPE gap runs one way. Ours is higher than the benchmark on average
(+0.074), and the excess matches the SPE false-alarm rate below. Our SPE
limit alarms on 7.1% of new normal samples, so part of the higher SPE
"detection" on the weak faults (10, 16, 19, 20, 21) is false alarms. The
benchmark's SPE false-alarm rate isn't reproduced here, so whether it used the
same limit can't be checked.

Detection delays, measured as the first of six consecutive alarms after the
onset (3 minutes per sample):

| Fault | T² | SPE | Univariate |
|---|---:|---:|---:|
| 1 | 18 min | 6 min | 6 min |
| 2 | 48 min | 30 min | 30 min |
| 6 | 18 min | 0 | 0 |
| 13 | 144 min | 108 min | 126 min |
| 17 | 84 min | 69 min | 63 min |

### Calibration: textbook limits work on TEP and fail on SECOM

The SECOM multivariate section found the textbook limits far too tight: 15.8%
of Phase I runs alarmed against a nominal ≤ 2%. That could be a bug in
`secom.mspc` or a property of SECOM. TEP separates the two, because the same
code at the same α is applied to data close to its assumptions (median excess
kurtosis −0.11, against 6.76 for SECOM's baseline sensors).

<p align="center"><img src="figures/fig14_tep_calibration.png" width="85%" alt="False-alarm rate of the textbook limits on TEP and SECOM against nominal alpha"></p>

| False-alarm rate, textbook limits | T² | SPE |
|---|---:|---:|
| TEP, training data (in-sample) | 0.6% | 0.4% |
| TEP, training data scored out of fold | 0.4% | 4.2% |
| TEP, independent normal run (`d00_te`) | 1.7% | 7.1% |
| TEP, pre-fault samples of all 21 test runs | 1.0% | 5.4% |
| SECOM, Phase I baseline (in-sample) | 12.0% | 7.5% |
| SECOM, Phase I baseline scored out of fold | 8.6% | 74.6% |
| Nominal | 1.0% | 1.0% |

The like-for-like comparison is in-sample: the same code, the same α, each
model scored on the data it was fitted to. TEP stays under nominal (0.6% and
0.4%), while SECOM runs at 12× and 7.5× nominal. **The SECOM failure is a
property of the data, not of the implementation.** The T² limit also holds
on new TEP data (1.7% on `d00_te`, 1.0% pooled over the pre-fault windows).

The SPE limit is less robust, and that applies to TEP too. On new normal data
it alarms at 5–7× nominal, and the rate rises with the number of retained
components: 3.5% at A = 5, 7.1% at A = 11, 16.6% at A = 24 (the 80% variance
rule used on SECOM). A model with more components fits the residual subspace
more closely to the one training run, and a new run doesn't share it. SECOM
shows the extreme case of the same effect: 81 components from about 574 runs
per fold, and an out-of-fold SPE false-alarm rate of 74.6%. On TEP, the
cross-validated limits bring SPE to 2.2% on `d00_te` but move T² from 1.7% to
4.7%. They are not uniformly better, so the textbook limits remain primary
here (D38).

### The moving-range sigma breaks on autocorrelated data

The SECOM sensor charts take sigma from the average moving range, which is
the textbook estimator for individuals charts and is unbiased only when
successive points are independent. TEP violates that. The median lag-1
autocorrelation is 0.55, 30 of 52 variables exceed 0.5, and the analyser
compositions XMEAS(37)–(41), which hold each value between samples, exceed
0.75.

The moving-range sigma falls to a median 0.40 of the overall SD (minimum
0.009), and the shortfall follows the autocorrelation closely (Spearman
ρ = −0.94). The same 52-chart Šidák family with moving-range sigma alarms on
**100%** of `d00_te`. With the overall SD it alarms on 4.0%. The univariate
results above therefore use the SD. That chart detects about as well as SPE
(fault 11: 0.769 vs 0.770), at a family false-alarm rate of 4.0% on `d00_te`
and 2.3% pooled over the pre-fault windows.

The finding carries back to SECOM only partly. SECOM runs are close to
independent (median lag-1 autocorrelation 0.04 over the baseline sensors), so
this failure mode doesn't apply there. Even so, SECOM's moving-range sigma is
a median 0.78 of the SD, because of heavy tails and level shifts. The
per-sensor Phase I limits are correspondingly narrower than SD-based limits
would be, which is one reason a raw beyond-limits count overstates excursions
(D40).

<p align="center"><img src="figures/fig15_tep_traces.png" width="100%" alt="T² and SPE traces for TEP fault 1 (detected) and fault 9 (not detected)"></p>
<p align="center"><sub><b>Detectable and undetectable, by construction.</b> Fault 1 (left) lifts both
statistics by two orders of magnitude within an hour. Fault 9 (right), a random variation in the D
feed temperature, leaves T² and SPE indistinguishable from normal operation, and from 0.981 missed
detection in the published benchmark.</sub></p>

### What this validation does not establish

- **That the methods work on SECOM.** TEP is a simulated chemical process with
  near-Gaussian, autocorrelated variables. SECOM is real, heavy-tailed and
  close to independent. The validation shows the implementation is correct
  and isolates the cause of the SECOM calibration failure. It does not show
  that any SECOM chart detects a real fab fault.
- **Exact agreement with the published table.** The benchmark values are a
  secondary transcription, and the benchmark's own false-alarm rates and
  limit details weren't available to compare against.
