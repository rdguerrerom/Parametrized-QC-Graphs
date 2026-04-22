"""Publication-quality matplotlib style for this project.

Honors PRL / PRX Quantum conventions:
  - single-column width: 3.4 in (8.6 cm)
  - double-column width: 7.0 in (17.8 cm)
  - body text: Times-like serif to match the typeset manuscript
  - axis labels in same serif, 9 pt
  - legends slightly smaller, 8 pt
  - no grid by default; minor ticks on
  - 300 dpi for both PDF and PNG export (PDF is vector, PNG is the
    backup for web / slides)

The module is a one-shot context installer: calling `use_style()` mutates
global rcParams. We prefer this over a style-file import so every figure
script that opts into it is self-documenting.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Column widths in inches (PRL / PRX Quantum convention).
SINGLE_COL_IN = 3.4
DOUBLE_COL_IN = 7.0
# Height-to-width ratio used by default (golden-ratio-ish, reads well).
DEFAULT_AR = 0.75

# Font sizes (pt). PRL body text is about 10 pt; axis labels slightly smaller
# to give headroom for superscripts.
FS_TITLE = 10
FS_LABEL = 9
FS_TICK = 8
FS_LEGEND = 8
FS_ANNOT = 8


def use_style() -> None:
    """Install the project-wide style. Idempotent — safe to call multiple times."""
    mpl.rcParams.update({
        # Fonts: prefer Times/STIX (LaTeX-like) but fall back cleanly.
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "legend.frameon": False,
        "figure.titlesize": FS_TITLE,

        # Axes / ticks: slim axes, ticks inside, minor ticks shown.
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,

        # Lines + markers.
        "lines.linewidth": 1.3,
        "lines.markersize": 4.5,
        "errorbar.capsize": 2.5,

        # Figure / export.
        "figure.dpi": 150,             # on-screen preview
        "savefig.dpi": 300,            # print-ready
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,            # Type-42 so text remains selectable
        "ps.fonttype": 42,
    })


def single_col(ar: float = DEFAULT_AR) -> tuple:
    """Figure size for a one-panel single-column PRL figure."""
    return (SINGLE_COL_IN, SINGLE_COL_IN * ar)


def double_col(ar: float = 0.45) -> tuple:
    """Figure size for a double-column PRL figure.

    Default aspect ratio narrower than single-column because these are
    wider and typically hold panels side-by-side.
    """
    return (DOUBLE_COL_IN, DOUBLE_COL_IN * ar)


def save_figure(fig, stem: str, fig_dir=None) -> None:
    """Save a figure as both PDF (vector, for the manuscript) and PNG (for web).

    `stem` is the base filename without suffix, e.g. "fig_d1_scaling".
    `fig_dir` defaults to `<repo>/figures/`.
    """
    from pathlib import Path
    if fig_dir is None:
        fig_dir = Path(__file__).resolve().parent.parent.parent / "figures"
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = fig_dir / f"{stem}.{ext}"
        fig.savefig(out)
    plt.close(fig)


def panel_label(ax, letter: str, x: float = -0.18, y: float = 1.05) -> None:
    """Add a consistent (a)/(b)/(c) label to a panel.

    Centralised so every figure uses the same weight / size / placement.
    Axes-fraction coords with a default offset that sits clear of the tick
    labels on the left and top of any axis.
    """
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")


__all__ = [
    "use_style",
    "single_col",
    "double_col",
    "save_figure",
    "SINGLE_COL_IN",
    "DOUBLE_COL_IN",
]
