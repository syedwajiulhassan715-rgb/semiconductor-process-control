"""Loading of the raw SECOM files into a single tidy frame.

The raw distribution is two space-delimited text files with no header:

``secom.data``
    1567 rows x 590 unnamed real-valued sensor readings, ``NaN`` for missing.
``secom_labels.data``
    1567 rows of ``<label> "<dd/mm/yyyy HH:MM:SS>"`` where the label is
    ``-1`` for an in-house line test pass and ``1`` for a fail.

Sensors have no names in the source, so they're called S001..S590 by column
position in secom.data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

N_RUNS = 1567
N_SENSORS = 590
TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M:%S"


def sensor_ids(n: int = N_SENSORS) -> list[str]:
    """Stable sensor identifiers, 1-based on raw column position."""
    return [f"S{i:03d}" for i in range(1, n + 1)]


@dataclass(frozen=True)
class SecomData:
    """Raw SECOM data in tidy form.

    Attributes
    ----------
    runs:
        One row per production run, indexed 0..1566 in file order, with columns
        ``run_id``, ``timestamp``, ``is_fail`` (bool) and the sensor columns.
    sensors:
        The sensor column names present in ``runs``.
    """

    runs: pd.DataFrame
    sensors: list[str]

    @property
    def values(self) -> pd.DataFrame:
        return self.runs[self.sensors]

    @property
    def is_fail(self) -> pd.Series:
        return self.runs["is_fail"]


def load_raw(raw_dir: str | Path) -> SecomData:
    """Read ``secom.data`` and ``secom_labels.data`` into a :class:`SecomData`.

    Raises
    ------
    FileNotFoundError
        If the raw files are absent; run ``scripts/00_fetch_data.py`` first.
    ValueError
        If the two files have different row counts, or the shape isn't
        1567 x 590 (the sensor ids depend on it).
    """
    raw_dir = Path(raw_dir)
    data_path = raw_dir / "secom.data"
    label_path = raw_dir / "secom_labels.data"
    for path in (data_path, label_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run scripts/00_fetch_data.py to download it."
            )

    values = np.genfromtxt(data_path, delimiter=" ", filling_values=np.nan)
    if values.shape != (N_RUNS, N_SENSORS):
        raise ValueError(f"expected secom.data to be {(N_RUNS, N_SENSORS)}, got {values.shape}")

    labels = pd.read_csv(
        label_path,
        sep=" ",
        header=None,
        names=["label", "timestamp"],
        quotechar='"',
    )
    if len(labels) != len(values):
        raise ValueError(f"row mismatch: secom.data has {len(values)}, labels has {len(labels)}")
    if not set(labels["label"].unique()) <= {-1, 1}:
        raise ValueError(f"unexpected label values: {sorted(labels['label'].unique())}")

    cols = sensor_ids(values.shape[1])
    runs = pd.DataFrame(values, columns=cols)
    runs.insert(0, "run_id", np.arange(1, len(runs) + 1))
    runs.insert(
        1,
        "timestamp",
        pd.to_datetime(labels["timestamp"], format=TIMESTAMP_FORMAT),
    )
    # Source encodes fail as +1 and pass as -1.
    runs.insert(2, "is_fail", labels["label"].eq(1).to_numpy())

    if not runs["timestamp"].is_monotonic_increasing:
        raise ValueError("timestamps are not in non-decreasing file order")

    return SecomData(runs=runs, sensors=cols)
