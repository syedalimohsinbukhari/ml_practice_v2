#!/usr/bin/env python
"""Bare-minimum rose plot (polar histogram) of circular residuals, certified 4-model set.

pred - true, wrap-aware, per periodic head (coa_phase period=2pi, polarization_angle
period=pi, inclination period=2pi -- matches heads_spec.py). A collapsed constant
predictor shows as a flat ring (const - true inherits true's own ~uniform spread);
real recovery shows as a spike near zero. No styling beyond what matplotlib defaults
give -- deliberately minimal, not a polished figure.

Usage (GPU machine -- loads best.weights.h5, calls trainer.predict):
    python experiments/phic_psi_poc-redo/rose_plot_residuals.py

Output: rose_plot_output/rose_residuals.{png,pdf}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "phic_psi_poc-redo"))

from gwml.data.loader import load_arrays
from gwml.data.transforms import TargetTransforms
from gwml.training.train import latest_run_dir, load_config

CONFIGS = {
    "poc_a": ROOT / "experiments/phic_psi_poc-redo/config_baseline.yaml",
    "poc_b": ROOT / "experiments/phic_psi_poc-redo/config_poc.yaml",
    "tcn": ROOT / "experiments/phic_psi_poc-redo/config_tcn.yaml",
    "cnn_attention": ROOT / "experiments/phic_psi_poc-redo/config_cnn_attention.yaml",
}
HEADS = [("coa_phase", 2 * np.pi), ("polarization_angle", np.pi), ("inclination", 2 * np.pi)]
N_SAMPLES = 2000


def wrap_resid(pred, true, period):
    r = pred - true
    return (r + period / 2) % period - period / 2


def main():
    out_dir = Path("experiments/phic_psi_poc-redo/rose_plot_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(CONFIGS), len(HEADS),
                              subplot_kw={"projection": "polar"},
                              figsize=(3 * len(HEADS), 3 * len(CONFIGS)))

    for row, (label, config_path) in enumerate(CONFIGS.items()):
        from train_poc import build_sumdiff_trainer

        cfg = load_config(str(config_path))
        run_dir = latest_run_dir(cfg)
        weights = run_dir / "best.weights.h5"
        strain, params = load_arrays(cfg["data"]["path"], "validation", max_samples=N_SAMPLES)
        transforms = TargetTransforms.from_json(run_dir / "transforms.json")

        trainer = build_sumdiff_trainer(cfg)
        trainer(strain[:1])
        trainer.load_weights(str(weights))

        raw_pred = trainer.predict(strain, batch_size=256, verbose=0)
        pred = transforms.inverse(raw_pred)
        true = transforms.physical_targets(params)

        for col, (head, period) in enumerate(HEADS):
            ax = axes[row][col]
            resid = wrap_resid(np.ravel(pred[head]), np.ravel(true[head]), period)
            ax.hist(resid, bins=36)
            ax.set_title(f"{label}/{head}", fontsize=8)
            ax.set_yticklabels([])

    fig.tight_layout()
    fig.savefig(out_dir / "rose_residuals.png")
    fig.savefig(out_dir / "rose_residuals.pdf")
    print(f"Wrote {out_dir / 'rose_residuals.png'}")


if __name__ == "__main__":
    main()
