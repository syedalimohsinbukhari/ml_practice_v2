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
from gwml.heads_spec import HEAD_SPECS, TransformKind


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


def residuals_vs_param(
    true: dict[str, np.ndarray],
    pred: dict[str, np.ndarray],
    param_values: np.ndarray,
    param_label: str,
    heads: list[str],
    path: str | Path,
    n_bins: int = 8,
) -> None:
    """|residual| binned by an arbitrary true param per head.

    Used for SNR (errors should shrink with SNR) and, e.g., mchirp — a
    systematic bias toward the population mean in a low-signal regime reads
    differently from uncorrelated noise (memorization); see
    q_head_action_plan.md Phase 1 step 6.
    """
    fig, axes = _grid(len(heads))
    edges = np.linspace(param_values.min(), param_values.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for ax, head in zip(axes, heads):
        t, p = true[head], pred[head]
        # Multi-column heads (e.g. sky_position) need per-sample error
        # reduction before binning, otherwise np.ravel blows up the
        # array length to k*N while param_values is only N.
        if t.ndim == 2 and t.shape[1] > 1:
            abs_res = np.linalg.norm(t - p, axis=1)  # (N,)
        else:
            abs_res = abs_error(head, t, p)
        binned = [
            abs_res[(param_values >= edges[i]) & (param_values < edges[i + 1])].mean()
            if np.any((param_values >= edges[i]) & (param_values < edges[i + 1]))
            else np.nan
            for i in range(n_bins)
        ]
        ax.plot(centers, binned, marker="o")
        ax.set_title(HEAD_SPECS[head].label)
        ax.set_xlabel(f"true {param_label}")
        ax.set_ylabel("mean |residual|")
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
    residuals_vs_param(true, pred, snr, "SNR", heads, path, n_bins=n_bins)


def sigmoid_logit_hist(
    train_raw_pred: dict[str, np.ndarray],
    train_true_physical: dict[str, np.ndarray],
    val_raw_pred: dict[str, np.ndarray],
    val_true_physical: dict[str, np.ndarray],
    heads: list[str],
    path: str | Path,
    n_bins: int = 40,
) -> None:
    """Pre-sigmoid logit histograms, train vs val, split by true-value tercile.

    Recovers the logit algebraically from the sigmoid output —
    logit = ln(p/(1-p)) on the model's raw (pre-inverse-transform) [0,1]
    prediction — no architecture change needed. Train logits going extreme
    while val logits for the same true-value tercile stay compressed is the
    saturation-driven-overfitting signature from q_head_action_plan.md
    Phase 1 step 3. Only applies to UNIT_AFFINE heads with a sigmoid
    activation (e.g. q, merger_time); heads is filtered down to those.
    """
    sig_heads = [
        h for h in heads
        if HEAD_SPECS[h].transform is TransformKind.UNIT_AFFINE
        and HEAD_SPECS[h].activation == "sigmoid"
    ]
    if not sig_heads:
        return

    def _logit(raw: np.ndarray) -> np.ndarray:
        p = np.clip(np.ravel(raw), 1e-6, 1.0 - 1e-6)
        return np.log(p / (1.0 - p))

    fig, axes = plt.subplots(len(sig_heads), 2,
                             figsize=(9.0, 4.0 * len(sig_heads)), squeeze=False)
    for row, head in enumerate(sig_heads):
        terciles = np.quantile(np.ravel(train_true_physical[head]), [1 / 3, 2 / 3])
        splits = [
            ("train", train_raw_pred, train_true_physical),
            ("val", val_raw_pred, val_true_physical),
        ]
        for col, (name, raw_pred, true_physical) in enumerate(splits):
            ax = axes[row][col]
            logit = _logit(raw_pred[head])
            tercile_idx = np.digitize(np.ravel(true_physical[head]), terciles)
            for i, label in enumerate(["low", "mid", "high"]):
                vals = logit[tercile_idx == i]
                if len(vals):
                    ax.hist(vals, bins=n_bins, alpha=0.5, label=f"true {label}")
            ax.set_title(f"{HEAD_SPECS[head].label} — {name}")
            ax.set_xlabel("logit = ln(p/(1-p))")
            if col == 0:
                ax.legend(fontsize=7)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
