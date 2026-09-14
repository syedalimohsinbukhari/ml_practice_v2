#!/usr/bin/env python
"""Evaluate a trained checkpoint: metrics CSV + scatter and residual plots.

    python scripts/evaluate.py configs/resnet1d.yaml --split validation
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT.parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="path to the experiment's YAML config")
    parser.add_argument("--split", default="validation",
                        choices=["training", "validation"])
    parser.add_argument("--weights", default=None,
                        help=("weights file "
                              "(default: <latest_run_dir>/best.weights.h5)"))
    args = parser.parse_args()

    from experiments.plot_style import update_style
    update_style()

    from gwml.data.loader import load_arrays
    from gwml.data.transforms import PARAM_COLUMNS, TargetTransforms
    from gwml.evaluation.metrics import evaluate_model
    from gwml.evaluation.plots import (
        residuals_vs_param,
        scatter_grid,
        sigmoid_logit_hist,
    )
    from gwml.training.train import build_trainer, latest_run_dir, load_config

    cfg = load_config(args.config)
    run_dir = latest_run_dir(cfg)
    weights = args.weights or run_dir / "best.weights.h5"

    strain, params = load_arrays(cfg["data"]["path"], args.split,
                                 cfg["data"].get("max_samples"))
    transforms = TargetTransforms.from_json(run_dir / "transforms.json")

    trainer = build_trainer(cfg)
    trainer(strain[:1])  # build variables before loading weights
    trainer.load_weights(weights)

    df = evaluate_model(trainer, strain, params, transforms)
    out_csv = run_dir / f"metrics_{args.split}.csv"
    df.to_csv(out_csv)
    print(df.to_string())
    print(f"\nwritten: {out_csv}")

    raw_pred = trainer.predict(strain, batch_size=256, verbose=0)
    pred = transforms.inverse(raw_pred)
    true = transforms.physical_targets(params)
    scatter_grid(true, pred, transforms.heads,
                 run_dir / f"scatter_{args.split}.png")
    residuals_vs_param(true, pred, params[:, PARAM_COLUMNS["snr"]], "SNR",
                       transforms.heads,
                       run_dir / f"residuals_snr_{args.split}.png")
    residuals_vs_param(true, pred, params[:, PARAM_COLUMNS["mchirp"]], "mchirp",
                       transforms.heads,
                       run_dir / f"residuals_mchirp_{args.split}.png")

    # Pre-sigmoid logit diagnostic (train vs val) for sigmoid/UNIT_AFFINE
    # heads (e.g. q) — needs the other split too; see
    # q_head_action_plan.md Phase 1 step 3.
    other_split = "validation" if args.split == "training" else "training"
    other_strain, other_params = load_arrays(
        cfg["data"]["path"], other_split, cfg["data"].get("max_samples")
    )
    other_raw_pred = trainer.predict(other_strain, batch_size=256, verbose=0)
    other_true = transforms.physical_targets(other_params)
    train_raw, train_true = (
        (raw_pred, true) if args.split == "training" else (other_raw_pred, other_true)
    )
    val_raw, val_true = (
        (other_raw_pred, other_true) if args.split == "training" else (raw_pred, true)
    )
    sigmoid_logit_hist(train_raw, train_true, val_raw, val_true,
                       transforms.heads, run_dir / "logits_train_vs_val.png")

    print(f"plots written to {run_dir}")


if __name__ == "__main__":
    main()
