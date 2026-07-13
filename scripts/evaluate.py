#!/usr/bin/env python
"""Evaluate a trained checkpoint: metrics CSV + scatter and residual plots.

    python scripts/evaluate.py configs/resnet1d.yaml --split validation
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="path to the experiment's YAML config")
    parser.add_argument("--split", default="validation",
                        choices=["training", "validation"])
    parser.add_argument("--weights", default=None,
                        help=("weights file "
                              "(default: <latest_run_dir>/best.weights.h5)"))
    args = parser.parse_args()

    from gwml.data.loader import load_arrays
    from gwml.data.transforms import PARAM_COLUMNS, TargetTransforms
    from gwml.evaluation.metrics import evaluate_model
    from gwml.evaluation.plots import residuals_vs_snr, scatter_grid
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

    pred = transforms.inverse(trainer.predict(strain, batch_size=256, verbose=0))
    true = transforms.physical_targets(params)
    scatter_grid(true, pred, transforms.heads,
                 run_dir / f"scatter_{args.split}.png")
    residuals_vs_snr(true, pred, params[:, PARAM_COLUMNS["snr"]],
                     transforms.heads,
                     run_dir / f"residuals_snr_{args.split}.png")
    print(f"plots written to {run_dir}")


if __name__ == "__main__":
    main()
