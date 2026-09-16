#!/usr/bin/env python
"""Train-vs-validation circular loss, for the redo's down-select-confirmed model set.

Redo counterpart of ``experiments/phic_psi_poc/plot_certified_memorization.py``,
adapted rather than copied verbatim — see the definitional gap this required, below.

The original singled out exactly two models, poc_b and cnn_attention, because those
were the two the original's null result was *certified* on (Sec 8.2's "capacity
wasn't the binding constraint" argument). This redo has not certified a null on
anything: the Step 0 std_ratio gate never cleared at any λ tried (0.01/0.05/0.10),
so the central result is UNINTERPRETABLE, not NULL, for every model — there is no
principled subset of "the certified ones" to single out the way the original could.

Adaptation: apply the same train-vs-val memorization check uniformly to all four
models in this redo's down-select-confirmed set (NOTES.md, 2026-09-15) — poc_a,
poc_b, tcn, cnn_attention — rather than picking two arbitrarily. This changes the
scope (4 models, not 2) and the grid layout (2x4, not 2x2) but not the check's
logic: same circular-loss train/val curves, same null-baseline reference line at
1.0, same read (a small train/val gap alongside flat loss at ~1.0 means the flat
result isn't explained by "the model couldn't hold the data in the first place").

Read-only over already-committed run artifacts: does not train or evaluate
anything, no GPU/model loading — CPU-safe, unlike the redo's other newly-ported
scripts (inclination/SNR stratification, perturbation trace, analyse_predictions,
diagnostic_checks), which all need real weight loading.

Usage:
    python experiments/phic_psi_poc-redo/plot_certified_memorization.py

Output:
    diagnostic_output/redo_models_train_val_loss.{png,pdf}
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
from experiments.plot_style import SERIES_COLORS, update_style, LEGEND_FONT_SIZE, LINE_WIDTH, SAVE_DPI

# Round-1 (λ=0.01) checkpoints for the redo's down-select-confirmed 4-model set.
DEFAULT_RUNS = {
    "poc_a": ROOT / "runs/phic_psi_poc_redo_a/20260915_051948/history.csv",
    "poc_b": ROOT / "runs/phic_psi_poc_redo_b/20260915_061008/history.csv",
    "tcn": ROOT / "runs/phic_psi_redo_tcn/20260915_064106/history.csv",
    "cnn_attention": ROOT / "runs/phic_psi_redo_cnn_attention/20260915_054256/history.csv",
}

MODEL_ORDER = ["poc_a", "poc_b", "tcn", "cnn_attention"]

model_label_dict = {
    "poc_a": r"TCN [POC$_\text{A}$]",
    "poc_b": r"TCN [POC$_\text{B}$]",
    "tcn": "TCN",
    "cnn_baseline": "CNN [baseline]",
    "cnn_attention": "CNN [with attention]",
    "inception_time": "Time-Inception model",
    "resnet1d": "ResNet 1D",
}

log_var_targets = {
    "coa_phase": r"$\phi_\text{c}$",
    "combo_A": r"$2\phi_c+2\psi$",
    "combo_B": r"$2\phi_c-2\psi$",
    "inclination": "Inclination",
    "mchirp": r"M$_\text{chirp}$",
    "merger_time": r"t$_\text{merge}$",
    "polarization_angle": r"$\psi$",
    "sky_position": r"SKY$_\text{RA, Dec}$",
    "snr": "SNR",
}

# (train column, val column, display label) per model — poc_b (poc mode) uses
# combo A/B; poc_a/tcn/cnn_attention (baseline mode) use coa_phase/polarization_angle
# directly.
HEAD_COLUMNS = {
    "poc_a": [
        ("circular_loss_coa_phase", "val_circular_loss_coa_phase", "coa_phase"),
        ("circular_loss_polarization_angle", "val_circular_loss_polarization_angle", "polarization_angle"),
    ],
    "poc_b": [
        ("circular_loss_combo_A", "val_circular_loss_combo_A", "combo_A"),
        ("circular_loss_combo_B", "val_circular_loss_combo_B", "combo_B"),
    ],
    "tcn": [
        ("circular_loss_coa_phase", "val_circular_loss_coa_phase", "coa_phase"),
        ("circular_loss_polarization_angle", "val_circular_loss_polarization_angle", "polarization_angle"),
    ],
    "cnn_attention": [
        ("circular_loss_coa_phase", "val_circular_loss_coa_phase", "coa_phase"),
        ("circular_loss_polarization_angle", "val_circular_loss_polarization_angle", "polarization_angle"),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for model in MODEL_ORDER:
        parser.add_argument(f"--{model.replace('_', '-')}-history",
                             default=str(DEFAULT_RUNS[model]),
                             help=f"path to {model}'s history.csv")
    parser.add_argument("--out",
                         default=str(ROOT / "experiments/phic_psi_poc-redo/diagnostic_output/redo_models_train_val_loss.png"),
                         help="output PNG path (a matching .pdf is written alongside)")
    parser.add_argument("--dpi", type=int, default=SAVE_DPI,
                         help="figure DPI (default: update_style()'s savefig.dpi)")
    parser.add_argument("--figsize", type=float, nargs=2, default=(20.0, 8.0),
                         help="figure size in inches, e.g. --figsize 20 8")
    parser.add_argument("--train-color", default=SERIES_COLORS["train"], help="training-curve color")
    parser.add_argument("--val-color", default=SERIES_COLORS["val"], help="validation-curve color")
    parser.add_argument("--null-color", default=SERIES_COLORS["null"], help="null-baseline line color")
    args = parser.parse_args()

    update_style()

    runs = {model: Path(getattr(args, f"{model}_history")) for model in MODEL_ORDER}

    missing = {m: p for m, p in runs.items() if not p.exists()}
    if missing:
        for m, p in missing.items():
            print(f"  ✗ no history.csv for {m} at {p}")
        raise SystemExit(1)

    fig, axes = plt.subplots(2, len(MODEL_ORDER), figsize=tuple(args.figsize),
                              sharey=True, sharex=True, squeeze=False)

    for col, model in enumerate(MODEL_ORDER):
        df = pd.read_csv(runs[model])
        for row, (train_col, val_col, label) in enumerate(HEAD_COLUMNS[model]):
            ax = axes[row][col]
            ax.plot(df["epoch"], df[train_col], color=args.train_color, label="Train", linewidth=LINE_WIDTH)
            ax.plot(df["epoch"], df[val_col], color=args.val_color, label="Validation", linewidth=LINE_WIDTH)
            ax.axhline(1.0, color=args.null_color, linestyle="--", linewidth=LINE_WIDTH, label=r"NULL ($\approx$ 1.0)")
            if row == 0:
                ax.set_title(f"{model_label_dict[model]}")
            if row == len(HEAD_COLUMNS[model]) - 1:
                ax.set_xlabel("Epoch")
            if col == 0:
                ax.set_ylabel("Circular Loss")
            ax.legend(loc="lower left", fontsize=LEGEND_FONT_SIZE, title=log_var_targets[label])

    fig.suptitle("Train vs. Validation Circular Loss — Down-Select-Confirmed 4-Model Set (Redo)\n"
                  "No model is null-certified here (Step 0 gate never cleared) — shown uniformly, not restricted",
                  fontsize=10)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path_pdf = out_path.with_suffix(".pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.dpi is not None:
        fig.savefig(out_path, dpi=args.dpi)
        fig.savefig(out_path_pdf, dpi=args.dpi)
    else:
        fig.savefig(out_path)
        fig.savefig(out_path_pdf)
    print(f"Wrote {out_path} / {out_path_pdf}")


if __name__ == "__main__":
    main()
