# Semiconductor process control

**Statistical process control and root-cause screening on 1,567 production runs
of real fab data.**

UCI SECOM is 590 unnamed sensor signals with pass/fail from in-house line
test, covering 89 days of 2008 and 104 failures (6.64%). The conclusion is that
this line needs a **monitoring scheme, not a prediction model**. The project is
built to demonstrate that rather than assert it, and every number it quotes is
regenerated into `reports/results.json` by a deterministic pipeline.

## The findings

<div class="grid cards" markdown>

-   **The standard chart misses a threefold yield improvement**

    ---

    The failure rate falls from 12.92% to 4.11% at ISO week 35, but the
    weekly p-chart is overdispersed and too coarse to show it.

    [Yield behaviour →](report.md#2-yield-behaviour)

-   **A CUSUM sees it in half a day**

    ---

    A Bernoulli CUSUM with a simulation-calibrated threshold signals the
    degraded period after 12 runs, without needing subgroups.

    [The monitoring scheme →](report.md#3-the-monitoring-scheme-that-works)

-   **16 of 17 apparent root causes are a time confound**

    ---

    71% of sensors drift with run order. Only S060 still separates pass
    from fail within a stable regime.

    [Which sensor, if any →](report.md#4-which-sensor-if-any-explains-it)

-   **Run-level prediction does not deploy**

    ---

    Shuffled CV reports AUC 0.698; an honest forward split gives 0.606, and
    a balanced error rate no better than a coin flip.

    [Can failures be predicted →](report.md#5-can-failures-be-predicted-per-run)

-   **A multivariate PCA chart does not do better**

    ---

    Textbook T² and SPE limits alarm on 15.8% of baseline runs against a
    nominal 2%. Calibrated limits flag mostly heavy-tailed outliers.

    [Multivariate monitoring →](report.md#6-multivariate-monitoring-pca-t²-and-spe)

-   **The p-chart's blindness is measured, not argued**

    ---

    At a doubling, a CUSUM at matched false-alarm risk signals within 400
    runs 70% of the time against 50% for the p-chart; for improvement the
    p-chart never signals.

    [Detection power →](report.md#7-detection-power-measured-rather-than-argued)

-   **The change is at run 375, inside the baseline boundary**

    ---

    The exact change-point estimate (21 Aug, W34) supports the week-35
    baseline, but the post-change rate of 4.27% suggests p₀ is optimistic.

    [Change point →](report.md#8-where-the-yield-changed-located-at-run-grain)

-   **The implementation is validated on known faults**

    ---

    On Tennessee Eastman the PCA code reproduces published detection rates
    and its textbook limits hold, so SECOM's failure is in the data.

    [Validation →](report.md#9-validation-on-a-benchmark-with-known-faults)

</div>

## Where to go next

| If you want to | Read |
|---|---|
| Run the analysis yourself | [Getting started](getting-started.md) |
| See the full argument, with every caveat | [Engineering report](report.md) |
| Know why each choice was made, and what breaks if it is wrong | [Decision log](decisions.md) |
| Change a threshold and rerun | [Configuration](configuration.md) |
| Add an analysis stage | [Pipeline architecture](architecture.md) |
| Watch the recommended scheme operate run by run | [Replay monitor](monitor.md) |
| Build the dashboard | [Power BI](powerbi.md) |
| Use the library | [API reference](reference/secom/index.md) |

Data: McCann, M. & Johnston, A. (2008). *SECOM* [Dataset]. UCI Machine Learning
Repository. <https://doi.org/10.24432/C54305>
