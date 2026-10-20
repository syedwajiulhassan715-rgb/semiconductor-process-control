"""Shared figure style and backend selection.

Import the palette from here so that figures added by new stages match
fig01-fig06::

    from secom.pipeline.plotting import ACCENT, GOOD, INK, MUTED
"""

from __future__ import annotations

INK = "#1b1b1b"
ACCENT = "#b3412c"
MUTED = "#8a8a8a"
GOOD = "#2f6f4e"
BLUE = "#3a5f9e"
SAND = "#c8a57d"
LIGHT = "#dddddd"

#: Backend per output format. The svg backend is pure Python, which matters
#: where policy blocks matplotlib's compiled Agg extension.
BACKENDS = {"png": "Agg", "svg": "svg"}


def configure(fmt: str, dpi: int) -> None:
    """Select the backend for ``fmt`` and apply the house style."""
    import matplotlib.pyplot as plt

    plt.switch_backend(BACKENDS[fmt])
    plt.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "font.size": 9,
            "axes.edgecolor": "#cccccc",
            "axes.labelcolor": INK,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": "#ebebeb",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
