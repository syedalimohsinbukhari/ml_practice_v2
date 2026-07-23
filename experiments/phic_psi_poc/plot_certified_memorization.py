#!/usr/bin/env python
"""Train-vs-validation circular loss, for the two *certified* models specifically.

Adversarial review follow-up: the chapter's Sec 8.2 "capacity wasn't the binding
constraint" argument was illustrated with the aggregate memorization gap across all
seven architectures (Fig. 5.2), but never broken out for poc_b and cnn_attention
individually -- the two models the null is actually *certified* on. This script reads
each model's own history.csv (no model loading, no GPU) and plots train vs val circular
loss for exactly those two, so the reader can see the two certified models' own curves
rather than trust the aggregate pattern by extension.

Read-only over already-committed run artifacts: does not train or evaluate anything.

Usage:
    python experiments/phic_psi_poc/plot_certified_memorization.py

Output:
    diagnostic_output/certified_models_train_val_loss.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# Run-7-era (lambda=0.01) checkpoints for the two certified models.
DEFAULT_RUNS = {
    "poc_b": ROOT / "runs/phic_psi_poc_b/20260720_213202/history.csv",
    "cnn_attention": ROOT / "runs/phic_psi_cnn_attention/20260720_221625/history.csv",
}

# (train column, val column, display label) per model -- poc_b uses combo A/B,
# baseline-mode cnn_attention uses coa_phase/polarization_angle directly.
HEAD_COLUMNS = {
    "poc_b": [
        ("circular_loss_combo_A", "val_circular_loss_combo_A", "combo A (φ_c + 2ψ)"),
        ("circular_loss_combo_B", "val_circular_loss_combo_B", "combo B (φ_c − 2ψ)"),
    ],
    "cnn_attention": [
        ("circular_loss_coa_phase", "val_circular_loss_coa_phase", "φ_c"),
        ("circular_loss_polarization_angle", "val_circular_loss_polarization_angle", "ψ"),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poc-b-history", default=str(DEFAULT_RUNS["poc_b"]),
                         help="path to poc_b's history.csv")
    parser.add_argument("--cnn-attention-history", default=str(DEFAULT_RUNS["cnn_attention"]),
                         help="path to cnn_attention's history.csv")
    parser.add_argument("--out", default=str(ROOT / "experiments/phic_psi_poc/diagnostic_output/certified_models_train_val_loss.png"),
                         help="output PNG path")
    parser.add_argument("--dpi", type=int, default=150, help="figure DPI")
    parser.add_argument("--figsize", type=float, nargs=2, default=(11.0, 8.0),
                         help="figure size in inches, e.g. --figsize 11 8")
    parser.add_argument("--train-color", default="#d95f02", help="training-curve color")
    parser.add_argument("--val-color", default="#1b9e77", help="validation-curve color")
    parser.add_argument("--null-color", default="#666666", help="null-baseline line color")
    args = parser.parse_args()

    runs = {"poc_b": Path(args.poc_b_history), "cnn_attention": Path(args.cnn_attention_history)}

    fig, axes = plt.subplots(2, 2, figsize=tuple(args.figsize), sharey=True)

    for col, model in enumerate(("poc_b", "cnn_attention")):
        df = pd.read_csv(runs[model])
        for row, (train_col, val_col, label) in enumerate(HEAD_COLUMNS[model]):
            ax = axes[row, col]
            ax.plot(df["epoch"], df[train_col], color=args.train_color, label="train", linewidth=1.6)
            ax.plot(df["epoch"], df[val_col], color=args.val_color, label="val", linewidth=1.6)
            ax.axhline(1.0, color=args.null_color, linestyle="--", linewidth=1.0, label="null (≈ 1.0)")
            ax.set_title(f"{model} — {label}")
            ax.set_xlabel("epoch")
            if col == 0:
                ax.set_ylabel("circular loss")
            ax.set_ylim(0.4, 1.1)
            if row == 0 and col == 0:
                ax.legend(loc="lower left", fontsize=8)

    fig.suptitle("Train vs. validation circular loss, certified models only "
                 "(poc_b, cnn_attention)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
