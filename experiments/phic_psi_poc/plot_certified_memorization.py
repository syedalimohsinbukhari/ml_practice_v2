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
import sys
sys.path.insert(0, str(ROOT))
from experiments.plot_style import SERIES_COLORS, update_style, FIGURE_DPI, LEGEND_FONT_SIZE, LINE_WIDTH, \
    TITLE_FONT_SIZE, SAVE_DPI

# Run-7-era (lambda=0.01) checkpoints for the two certified models.
DEFAULT_RUNS = {
    "poc_b": ROOT / "runs/phic_psi_poc_b/20260720_213202/history.csv",
    "cnn_attention": ROOT / "runs/phic_psi_cnn_attention/20260720_221625/history.csv",
}

model_label_dict = {
    "poc_a": r"TCN [POC$_\text{A}$]",
    "poc_b": r"TCN [POC$_\text{B}$]",
    "tcn": "TCN",
    "cnn_baseline": "CNN [baseline]",
    "cnn_attention": "CNN [with attention]",
    "inception_time": "Time-Inception model",
    "resnet1d": "ResNet 1D"
}

log_var_targets = {
    "coa_phase": r"$\phi_\text{c}$",
    "combo_A": r"$\phi_c+2\psi$",
    "combo_B": r"$\phi_c-2\psi$",
    "inclination": "Inclination",
    "mchirp": r"M$_\text{chirp}$",
    "merger_time": r"t$_\text{merge}$",
    "polarization_angle": r"$\psi$",
    "sky_position": r"SKY$_\text{RA, Dec}$",
    "snr": "SNR"
}

# (train column, val column, display label) per model -- poc_b uses combo A/B,
# baseline-mode cnn_attention uses coa_phase/polarization_angle directly.
HEAD_COLUMNS = {
    "poc_b": [
        ("circular_loss_combo_A", "val_circular_loss_combo_A", "combo_A"),
        ("circular_loss_combo_B", "val_circular_loss_combo_B", "combo_B"),
    ],
    "cnn_attention": [
        ("circular_loss_coa_phase", "val_circular_loss_coa_phase", "coa_phase"),
        ("circular_loss_polarization_angle", "val_circular_loss_polarization_angle", "polarization_angle"),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poc-b-history", default=str(DEFAULT_RUNS["poc_b"]),
                         help="path to poc_b's history.csv")
    parser.add_argument("--cnn-attention-history", default=str(DEFAULT_RUNS["cnn_attention"]),
                         help="path to cnn_attention's history.csv")
    parser.add_argument("--out",
                        default=str(ROOT / "experiments/phic_psi_poc/diagnostic_output/certified_models_train_val_loss.png"),
                         help="output PNG path")
    parser.add_argument("--dpi", type=int, default=SAVE_DPI,
                         help="figure DPI (default: update_style()'s savefig.dpi)")
    parser.add_argument("--figsize", type=float, nargs=2, default=(11.0, 8.0),
                         help="figure size in inches, e.g. --figsize 11 8")
    parser.add_argument("--train-color", default=SERIES_COLORS["train"], help="training-curve color")
    parser.add_argument("--val-color", default=SERIES_COLORS["val"], help="validation-curve color")
    parser.add_argument("--null-color", default=SERIES_COLORS["null"], help="null-baseline line color")
    args = parser.parse_args()

    update_style()

    runs = {"poc_b": Path(args.poc_b_history), "cnn_attention": Path(args.cnn_attention_history)}

    fig, axes = plt.subplots(2, 2, figsize=tuple(args.figsize), sharey=True, sharex=True)

    for col, model in enumerate(("poc_b", "cnn_attention")):
        df = pd.read_csv(runs[model])
        for row, (train_col, val_col, label) in enumerate(HEAD_COLUMNS[model]):
            ax = axes[row, col]
            ax.plot(df["epoch"], df[train_col], color=args.train_color, label="Train", linewidth=LINE_WIDTH)
            ax.plot(df["epoch"], df[val_col], color=args.val_color, label="Validation", linewidth=LINE_WIDTH)
            ax.axhline(1.0, color=args.null_color, linestyle="--", linewidth=LINE_WIDTH, label=r"NULL ($\approx$ 1.0)")
            if row == 0:
                ax.set_title(f"{model_label_dict[model]}")
            if row == 1:
                ax.set_xlabel("Epoch")
            if col == 0:
                ax.set_ylabel("Circular Loss")
            ax.legend(loc="lower left", fontsize=LEGEND_FONT_SIZE, title=log_var_targets[label])

    # fig.suptitle("Circular Loss", fontsize=TITLE_FONT_SIZE)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path2 = Path(args.out.split('.png')[0] + '.pdf')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.dpi is not None:
        fig.savefig(out_path, dpi=args.dpi)
        fig.savefig(out_path2, dpi=args.dpi)
    else:
        fig.savefig(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
