"""State shared between pipeline stages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import Config

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class Paths:
    """Where the pipeline reads raw data and writes every output."""

    root: Path
    raw: Path
    processed: Path
    powerbi: Path
    figures: Path
    reports: Path

    @classmethod
    def from_root(cls, root: str | Path, *, figures: str | Path | None = None) -> Paths:
        """The standard repository layout under ``root``."""
        root = Path(root).resolve()
        return cls(
            root=root,
            raw=root / "data" / "raw",
            processed=root / "data" / "processed",
            powerbi=root / "data" / "powerbi",
            figures=Path(figures).resolve() if figures else root / "reports" / "figures",
            reports=root / "reports",
        )

    def ensure(self) -> None:
        """Create the output directories."""
        for d in (self.processed, self.powerbi, self.figures, self.reports):
            d.mkdir(parents=True, exist_ok=True)


class MissingArtifactError(KeyError):
    """A stage asked for an artifact no earlier stage produced."""


@dataclass
class PipelineContext:
    """Everything a stage can read or write.

    results
        Becomes ``reports/results.json``. Each stage writes one top-level key.
    artifacts
        In-memory objects passed between stages (frames, chart results,
        masks). See ``secom.pipeline`` for the keys each stage produces.
    figures_enabled
        False under ``--no-figures``. :meth:`save_figure` then discards
        figures, and a stage may skip building them altogether.
    """

    config: Config
    paths: Paths
    results: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    figures_enabled: bool = True
    log: Callable[[str], None] = print

    def __getitem__(self, key: str) -> Any:
        """Shorthand for ``artifacts[key]`` with an error naming the missing key."""
        try:
            return self.artifacts[key]
        except KeyError:
            raise MissingArtifactError(
                f"artifact {key!r} has not been produced; run its stage first"
            ) from None

    def save_figure(self, fig: Figure, name: str) -> Path | None:
        """Write ``<figures>/<name>.<format>`` and close ``fig``.

        Returns the written path, or None when figures are disabled.
        """
        import matplotlib.pyplot as plt

        try:
            if not self.figures_enabled:
                return None
            self.paths.figures.mkdir(parents=True, exist_ok=True)
            path = self.paths.figures / f"{name}.{self.config.figures.format}"
            fig.savefig(path, dpi=self.config.figures.dpi)
            return path
        finally:
            plt.close(fig)
