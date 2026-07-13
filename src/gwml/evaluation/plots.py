"""Plotting helpers shared by callbacks and the evaluate script.

Head labels come from the head registry; every function takes the active head
list so plots adapt to whatever heads a run trains. Periodic heads are shown
as angles in [0, period) — points near the wrap edge can legitimately sit far
off the diagonal, which is why the annotated MAE uses the wrap-aware error.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gwml.data.transforms import abs_error
from gwml.heads_spec import HEAD_SPECS


def _grid(n: int):
    cols = 2 if n <= 4 else 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.0 * rows),
                             squeeze=False)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def scatter_grid(
    true: dict[str, np.ndarray],
    pred: dict[str, np.ndarray],
    heads: list[str],
    path: str | Path,
    title: str | None = None,
) -> None:
    """Pred-vs-true scatter per head (physical units), y=x diagonal, MAE."""
    fig, axes = _grid(len(heads))
    for ax, head in zip(axes, heads):
        t, p = np.ravel(true[head]), np.ravel(pred[head])
        lo, hi = float(min(t.min(), p.min())), float(max(t.max(), p.max()))
        ax.plot([lo, hi], [lo, hi], color="gray", lw=1, zorder=1)
        ax.scatter(t, p, s=4, alpha=0.35, zorder=2)
        mae = float(np.mean(abs_error(head, t, p)))
        ax.set_title(f"{HEAD_SPECS[head].label}   MAE = {mae:.4g}")
        ax.set_xlabel("true")
        ax.set_ylabel("predicted")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def residuals_vs_snr(
    true: dict[str, np.ndarray],
    pred: dict[str, np.ndarray],
    snr: np.ndarray,
    heads: list[str],
    path: str | Path,
    n_bins: int = 8,
) -> None:
    """|residual| binned by true SNR per head — errors should shrink with SNR."""
    fig, axes = _grid(len(heads))
    edges = np.linspace(snr.min(), snr.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for ax, head in zip(axes, heads):
        abs_res = abs_error(head, true[head], pred[head])
        binned = [
            abs_res[(snr >= edges[i]) & (snr < edges[i + 1])].mean()
            if np.any((snr >= edges[i]) & (snr < edges[i + 1]))
            else np.nan
            for i in range(n_bins)
        ]
        ax.plot(centers, binned, marker="o")
        ax.set_title(HEAD_SPECS[head].label)
        ax.set_xlabel("true SNR")
        ax.set_ylabel("mean |residual|")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
