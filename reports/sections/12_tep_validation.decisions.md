## Validation

### D37. Validate on Tennessee Eastman, as a separate command

SECOM has no fault labels, so no analysis of SECOM alone can show that a chart
detects a real fault. The Tennessee Eastman benchmark has 21 faults injected
at known samples and a published PCA result on the same files. Running the
repo's own `secom.mspc` and chart code on it, unchanged, tests the
implementation where the answer is known.

The validation is `secom validate tep`, not a pipeline stage, so the SECOM
analysis never depends on a second dataset. The TEP files are fetched into
`data/raw/tep/` with their SHA-256 digests recorded, and they are never
committed. Only the derived `reports/validation/tep_results.json` is.

**If wrong:** if TEP were a poor proxy, a pass would prove little. That is why
the claim is limited to implementation correctness and the cause of the
calibration failure (section 12, "What this validation does not establish"),
not SECOM detection power.

### D38. Benchmark settings: A = 11, α = 0.01, textbook limits primary

A = 11 retained components and α = 0.01 per chart are the settings commonly
attributed to the Chiang, Russell & Braatz (2001) PCA benchmark on these
files. The textbook F and Jackson–Mudholkar limits are primary because they
are what that benchmark uses. The benchmark detection rates are a
transcription from secondary reproductions, not checked against the book, so
the tests assert only qualitative facts (easy faults caught, faults 3/9/15
not, fault 4 an SPE fault). They do not assert the table.

The cross-validated limits from the SECOM multivariate stage are reported
alongside but are not primary here. On TEP they improve SPE (7.1% → 2.2%
false alarms on `d00_te`) but worsen T² (1.7% → 4.7%), so neither scheme
dominates.

**If wrong:** if the benchmark used another A, the detection comparison
shifts. The sensitivity table (A = 5 to 24) shows mean detection over the
detectable faults moving only 0.61–0.65 (T²) and 0.76–0.84 (SPE), and the
qualitative facts hold throughout.

### D39. TEP univariate charts use the overall SD, not the moving range

The family of 52 per-variable charts with moving-range sigma alarms on 100%
of the independent normal run. The median lag-1 autocorrelation is 0.55, the
moving range then underestimates sigma (median 0.40 of the SD), and the
shortfall tracks the autocorrelation (Spearman ρ = −0.94). With the overall
training SD, the same Šidák family alarms on 4.0%.

Šidák rather than Bonferroni because it is exact for independent charts and
still conservative under the positive dependence these variables have. An
FDR rule doesn't fit, because an alarm is a per-sample decision over the
whole family, and the family-wise rate is the quantity an operator lives
with.

**If wrong:** the SD estimate is inflated by any slow drift within the
training run, which widens the limits. On this training file that cost is
small: univariate detection matches SPE on most faults.

### D40. The moving-range finding is recorded for SECOM, not acted on here

SECOM baseline sensors are close to independent (median lag-1
autocorrelation 0.04), so the TEP failure mode does not apply. Their
moving-range sigma is still a median 0.78 of the SD, because of heavy tails
and level shifts, and the per-sensor Phase I limits (D20) are correspondingly
narrower than SD-based limits would be. That is recorded, not changed: it
alters no conclusion in the report, and changing the sensor limits would
change committed outputs outside this validation's scope.

**If wrong:** a beyond-limits count on SECOM sensor charts (e.g. S060 on the
watchlist) overstates excursions relative to an SD-based chart.
`[Violation Ratio vs Chance]` in the DAX already warns against reading raw
counts.
