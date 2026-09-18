#!/usr/bin/env python
"""Bare-minimum Mollweide plot, true->predicted sky position, certified 4-model set.

One thin line per sample from true (ra,dec) to predicted (ra,dec). A tight cluster of
short lines means good localization; long lines scattered everywhere means the ~85-90
degree mean separation already reported is genuinely uniform-bad, not bimodal
(some-good/some-antipodal) hiding behind the mean. No styling beyond matplotlib
defaults -- deliberately minimal, not a polished figure.

Usage (GPU machine -- loads best.weights.h5, calls trainer.predict):
    python experiments/phic_psi_poc-redo/sky_position_mollweide.py

Output: sky_mollweide_output/sky_mollweide.{png,pdf}
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
N_SAMPLES = 500  # keep the plot readable -- 500 line segments, not 5000


def to_lon(ra):
    """RA in [0, 2pi) -> mollweide longitude in (-pi, pi]."""
    return ((ra + np.pi) % (2 * np.pi)) - np.pi


def main():
    out_dir = Path("experiments/phic_psi_poc-redo/sky_mollweide_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, subplot_kw={"projection": "mollweide"}, figsize=(10, 8))

    for ax, (label, config_path) in zip(axes.ravel(), CONFIGS.items()):
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

        true_dec, true_ra = true["sky_position"][:, 0], true["sky_position"][:, 1]
        pred_dec, pred_ra = pred["sky_position"][:, 0], pred["sky_position"][:, 1]

        for i in range(len(true_dec)):
            ax.plot([to_lon(true_ra[i]), to_lon(pred_ra[i])],
                    [true_dec[i], pred_dec[i]],
                    linewidth=0.3, alpha=0.4)
        ax.set_title(label, fontsize=9)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    fig.tight_layout()
    fig.savefig(out_dir / "sky_mollweide.png")
    fig.savefig(out_dir / "sky_mollweide.pdf")
    print(f"Wrote {out_dir / 'sky_mollweide.png'}")


if __name__ == "__main__":
    main()
