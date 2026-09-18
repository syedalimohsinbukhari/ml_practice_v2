#!/usr/bin/env python
"""Bare-minimum histogram + CDF of per-sample sky-position angular separation.

Replaces the Mollweide true->predicted line plot (sky_position_mollweide.py),
which was too dense/overlapping to distinguish "uniformly ~85 degrees everywhere"
from "bimodal -- some very accurate, some near-antipodal" behind the same mean.
This answers that question directly: a tight low-separation peak with a long tail
is bimodal-ish; a single hump centered near 90 degrees is genuinely uniform-bad.

Usage (GPU machine -- loads best.weights.h5, calls trainer.predict):
    python experiments/phic_psi_poc-redo/sky_angular_separation_hist.py

Output: sky_mollweide_output/sky_angular_separation.{png,pdf}
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
from gwml.data.sky_transform import radec_to_unit_vector, angular_separation
from gwml.training.train import latest_run_dir, load_config

CONFIGS = {
    "poc_a": ROOT / "experiments/phic_psi_poc-redo/config_baseline.yaml",
    "poc_b": ROOT / "experiments/phic_psi_poc-redo/config_poc.yaml",
    "tcn": ROOT / "experiments/phic_psi_poc-redo/config_tcn.yaml",
    "cnn_attention": ROOT / "experiments/phic_psi_poc-redo/config_cnn_attention.yaml",
}
N_SAMPLES = 2000


def main():
    out_dir = Path("experiments/phic_psi_poc-redo/sky_mollweide_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(CONFIGS), 2, figsize=(8, 3 * len(CONFIGS)))

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

        # sky_position is (dec, ra) per HeadSpec.columns; radec_to_unit_vector wants (ra, dec).
        true_vec = radec_to_unit_vector(true["sky_position"][:, 1], true["sky_position"][:, 0])
        pred_vec = radec_to_unit_vector(pred["sky_position"][:, 1], pred["sky_position"][:, 0])
        sep_deg = np.degrees(angular_separation(true_vec, pred_vec))

        ax_hist, ax_cdf = axes[row]
        ax_hist.hist(sep_deg, bins=36)
        ax_hist.axvline(90, linestyle="--")
        ax_hist.set_title(f"{label} — separation histogram", fontsize=9)

        sorted_sep = np.sort(sep_deg)
        cdf = np.arange(1, len(sorted_sep) + 1) / len(sorted_sep)
        ax_cdf.plot(sorted_sep, cdf)
        ax_cdf.axvline(90, linestyle="--")
        ax_cdf.set_title(f"{label} — CDF", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / "sky_angular_separation.png")
    fig.savefig(out_dir / "sky_angular_separation.pdf")
    print(f"Wrote {out_dir / 'sky_angular_separation.png'}")


if __name__ == "__main__":
    main()
