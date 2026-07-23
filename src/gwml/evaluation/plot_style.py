"""Shared matplotlib style for publication-ready figures, gwml side.

Deliberate duplicate of experiments/plot_style.py -- kept as two small, independent
copies rather than one cross-package import so that gwml has no dependency on the
experiments/ tree (and vice versa). Keep the two in sync by hand if the shared
convention changes.

Import explicitly where needed (``from gwml.evaluation.plot_style import update_style``).
"""

from matplotlib import pyplot as plt

LABEL_FONT_SIZE = 12
LEGEND_FONT_SIZE = 10
LEGEND_TITLE_FONT_SIZE = LEGEND_FONT_SIZE
TICK_FONT_SIZE = 12
TITLE_FONT_SIZE = 13
ANNOTATION_FONT_SIZE = 10  # no matplotlib rcParam for this; pass explicitly to ax.annotate(...)

# Figure / line / marker geometry
FIGURE_DPI = 150
SAVE_DPI = 600
DEFAULT_FIGURE_SIZE = (8, 6)
LINE_WIDTH = 1.0
MARKER_SIZE = 6
CAP_SIZE = 5
AXES_LINE_WIDTH = 0.8

# Tick geometry
TICK_MAJOR_SIZE = 5
TICK_MAJOR_WIDTH = 0.8
TICK_MINOR_SIZE = 3
TICK_MINOR_WIDTH = 0.6
TICK_DIRECTION = "in"  # astronomy convention

# Grid
GRID_ALPHA = 0.25
GRID_LINESTYLE = "--"


def update_style():
    """Update rcParams for publication-ready figures.

    All values are driven by the constants defined above in this module — a single
    source of truth. Changing a constant here automatically updates every figure
    that calls this function.
    """
    plt.rcParams.update(
        {
            # Font
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": LABEL_FONT_SIZE,
            "axes.labelsize": LABEL_FONT_SIZE,
            "axes.titlesize": TITLE_FONT_SIZE,
            "xtick.labelsize": TICK_FONT_SIZE,
            "ytick.labelsize": TICK_FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "legend.title_fontsize": LEGEND_TITLE_FONT_SIZE,
            # Figure
            "figure.dpi": FIGURE_DPI,
            "figure.figsize": DEFAULT_FIGURE_SIZE,
            "savefig.dpi": SAVE_DPI,
            "savefig.bbox": "tight",
            # Lines / markers
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": MARKER_SIZE,
            "axes.linewidth": AXES_LINE_WIDTH,
            "errorbar.capsize": CAP_SIZE,
            # Ticks
            "xtick.major.size": TICK_MAJOR_SIZE,
            "xtick.major.width": TICK_MAJOR_WIDTH,
            "xtick.minor.size": TICK_MINOR_SIZE,
            "xtick.minor.width": TICK_MINOR_WIDTH,
            "xtick.minor.visible": True,
            "xtick.direction": TICK_DIRECTION,
            "ytick.major.size": TICK_MAJOR_SIZE,
            "ytick.major.width": TICK_MAJOR_WIDTH,
            "ytick.minor.size": TICK_MINOR_SIZE,
            "ytick.minor.width": TICK_MINOR_WIDTH,
            "ytick.minor.visible": True,
            "ytick.direction": TICK_DIRECTION,
            # Grid
            "axes.grid": True,
            "grid.alpha": GRID_ALPHA,
            "grid.linestyle": GRID_LINESTYLE,
            "axes.axisbelow": True,
        }
    )
