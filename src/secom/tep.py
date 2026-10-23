"""Tennessee Eastman Process (TEP) benchmark data and detection metrics.

SECOM has no ground truth: nobody knows which runs were faulty or when a fault
began. The TEP benchmark has 21 faults injected at known times, so it is used
here to check that this repo's monitoring code detects what it should and
false-alarms at the rate it claims.

The files are the Braatz training and testing sets (Chiang, Russell & Braatz,
2001). Each testing file holds 960 samples taken every 3 minutes; the fault is
introduced after sample 160 (8 hours in), so rows 0-159 are normal operation
and rows 160-959 are faulty. Columns are XMEAS(1)-XMEAS(41) then
XMV(1)-XMV(11). ``d00.dat``, the normal-operation training file, is stored
transposed (52 x 500) in the distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats as sps

N_VARS = 52
N_TRAIN = 500
N_TEST = 960
#: index of the first faulty sample in every testing file
FAULT_ONSET = 160
SAMPLE_MINUTES = 3.0
FAULTS = tuple(range(1, 22))

VARIABLES = [f"XMEAS{i}" for i in range(1, 42)] + [f"XMV{i}" for i in range(1, 12)]

#: Fault descriptions from Downs & Vogel (1993), Table 8.
FAULT_DESCRIPTIONS = {
    1: "A/C feed ratio, B composition constant (stream 4): step",
    2: "B composition, A/C ratio constant (stream 4): step",
    3: "D feed temperature (stream 2): step",
    4: "Reactor cooling water inlet temperature: step",
    5: "Condenser cooling water inlet temperature: step",
    6: "A feed loss (stream 1): step",
    7: "C header pressure loss, reduced availability (stream 4): step",
    8: "A, B, C feed composition (stream 4): random variation",
    9: "D feed temperature (stream 2): random variation",
    10: "C feed temperature (stream 4): random variation",
    11: "Reactor cooling water inlet temperature: random variation",
    12: "Condenser cooling water inlet temperature: random variation",
    13: "Reaction kinetics: slow drift",
    14: "Reactor cooling water valve: sticking",
    15: "Condenser cooling water valve: sticking",
    16: "Unknown",
    17: "Unknown",
    18: "Unknown",
    19: "Unknown",
    20: "Unknown",
    21: "Valve for stream 4 fixed at its steady-state position",
}

#: Source of the published benchmark, checked against the original paper.
BENCHMARK_SOURCE = (
    "Russell, Chiang & Braatz (2000), Fault detection in industrial processes using "
    "canonical variate analysis and dynamic principal component analysis, Chemometrics "
    "and Intelligent Laboratory Systems 51, 81-93: Table 4 (missed detection) and "
    "Table 3 (false alarms); PCA with a = 11, alpha = 0.01"
)

#: PCA missed-detection rates (T², Q) per fault on the testing sets, alpha = 0.01,
#: from Table 4 of :data:`BENCHMARK_SOURCE` (the same values appear in Chiang,
#: Russell & Braatz 2001, *Fault Detection and Diagnosis in Industrial Systems*).
BENCHMARK_PCA_MISSED = {
    1: (0.008, 0.003),
    2: (0.020, 0.014),
    3: (0.998, 0.991),
    4: (0.956, 0.038),
    5: (0.775, 0.746),
    6: (0.011, 0.000),
    7: (0.085, 0.000),
    8: (0.034, 0.024),
    9: (0.994, 0.981),
    10: (0.666, 0.659),
    11: (0.794, 0.356),
    12: (0.029, 0.025),
    13: (0.060, 0.045),
    14: (0.158, 0.000),
    15: (0.988, 0.973),
    16: (0.834, 0.755),
    17: (0.259, 0.108),
    18: (0.113, 0.101),
    19: (0.996, 0.873),
    20: (0.701, 0.550),
    21: (0.736, 0.570),
}

#: Table 4 was computed with each threshold "adjusted to the tenth highest value
#: for the normal operating condition of the testing set" (p. 89), an empirical
#: alpha of about 0.01, not with the textbook limits.
BENCHMARK_THRESHOLD_RANK = 10

#: PCA false-alarm rates (T², Q) at alpha = 0.01, Table 3 of :data:`BENCHMARK_SOURCE`:
#: on the normal training data and on the normal testing set (d00_te).
BENCHMARK_PCA_FALSE_ALARM = {
    "training": (0.002, 0.004),
    "testing": (0.014, 0.016),
}

#: Faults that PCA is known not to detect (Chiang et al. 2001; Russell et al. 2000).
UNDETECTABLE_BY_PCA = (3, 9, 15)


@dataclass(frozen=True)
class TEPData:
    """Normal-operation training data and the 22 testing sets."""

    train: np.ndarray  # N_TRAIN x N_VARS, normal operation
    test: dict[int, np.ndarray]  # fault number (0 = normal) -> N_TEST x N_VARS

    @property
    def variables(self) -> list[str]:
        return list(VARIABLES)


def _read(path: Path, expected: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `secom fetch --dataset tep` to download it."
        )
    values = np.loadtxt(path)
    if values.shape != expected:
        raise ValueError(f"expected {path.name} to be {expected}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name} contains non-finite values")
    return values


def load_tep(tep_dir: str | Path) -> TEPData:
    """Read ``d00.dat`` and ``d00_te.dat`` ... ``d21_te.dat`` from ``tep_dir``.

    Raises
    ------
    FileNotFoundError
        If a file is absent.
    ValueError
        If any file has an unexpected shape. ``d00.dat`` must be the
        transposed 52 x 500 layout of the distribution; anything else would
        mean a different mirror or file, and every result would be suspect.
    """
    tep_dir = Path(tep_dir)
    train = _read(tep_dir / "d00.dat", (N_VARS, N_TRAIN)).T
    test = {k: _read(tep_dir / f"d{k:02d}_te.dat", (N_TEST, N_VARS)) for k in (0, *FAULTS)}
    return TEPData(train=train, test=test)


def sidak_k(alpha: float, m: int) -> float:
    """Two-sided normal multiplier giving family-wise false-alarm ``alpha`` over ``m`` charts.

    Each chart runs at 1 - (1 - alpha)^(1/m), which is exact for independent
    charts and conservative when they are positively correlated.
    """
    if not 0 < alpha < 1 or m < 1:
        raise ValueError("need 0 < alpha < 1 and m >= 1")
    per_chart = 1 - (1 - alpha) ** (1 / m)
    return float(sps.norm.ppf(1 - per_chart / 2))


def detection_rate(alarm: np.ndarray, onset: int = FAULT_ONSET) -> float:
    """Fraction of faulty samples (index >= onset) that alarm."""
    a = np.asarray(alarm, dtype=bool)
    return float(a[onset:].mean())


def false_alarm_rate(alarm: np.ndarray, onset: int = FAULT_ONSET) -> float:
    """Fraction of pre-fault samples (index < onset) that alarm."""
    a = np.asarray(alarm, dtype=bool)
    return float(a[:onset].mean())


def detection_delay(
    alarm: np.ndarray, onset: int = FAULT_ONSET, *, run_length: int = 1
) -> int | None:
    """Samples from the onset to the start of the first run of ``run_length`` alarms.

    ``run_length=1`` is the first alarm after the onset. A longer run
    discounts isolated false alarms that happen to fall just after the
    onset. Returns None if no qualifying run occurs.
    """
    if run_length < 1:
        raise ValueError("run_length must be >= 1")
    a = np.asarray(alarm, dtype=bool)[onset:].astype(int)
    if len(a) < run_length:
        return None
    window = np.convolve(a, np.ones(run_length, dtype=int), mode="valid")
    hits = np.flatnonzero(window == run_length)
    return int(hits[0]) if hits.size else None
