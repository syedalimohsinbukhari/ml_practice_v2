#!/usr/bin/env python
"""Run train_poc -> plot_run -> evaluate back to back for all phic_psi configs.

    python experiments/phic_psi_poc/run_full.py
    python experiments/phic_psi_poc/run_full.py --split training

Searches ``experiments/phic_psi_poc/`` for ``config_*.yaml`` files, trains each
with ``train_poc.py``, plots with ``plot_run.py``, and evaluates.  Skips smoke
configs.

Works for both ``mode: baseline`` (circular loss on individual φc/ψ) and
``mode: poc`` (combo heads + curriculum weighting).  The evaluate step builds
the correct trainer variant for each mode.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = EXPERIMENTS_DIR.parents[1] / "scripts"
ROOT = EXPERIMENTS_DIR.parents[1]


def _run(script: str, *args: str) -> None:
    """Run a python script relative to the experiments dir."""
    script_path = EXPERIMENTS_DIR / script if "/" not in script else Path(script)
    if not script_path.exists():
        script_path = SCRIPTS_DIR / script
    cmd = [sys.executable, str(script_path), *args]
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _evaluate(config_path: str, split: str) -> None:
    """Evaluate a trained config — handles both baseline and poc modes.

    Uses the appropriate trainer class so weight loading succeeds for both
    ``mode: baseline`` (MultiHeadTrainer-compatible) and ``mode: poc``
    (SumDiffTrainer with combo log_vars).
    """
    import yaml

    sys.path.insert(0, str(ROOT / "src"))

    from gwml.data.loader import load_arrays
    from gwml.data.transforms import PARAM_COLUMNS, TargetTransforms
    from gwml.evaluation.metrics import evaluate_model
    from gwml.evaluation.plots import (
        residuals_vs_param,
        scatter_grid,
        sigmoid_logit_hist,
    )
    from gwml.training.train import latest_run_dir, load_config
    from train_poc import build_sumdiff_trainer

    cfg = load_config(config_path)
    run_dir = latest_run_dir(cfg)
    weights = run_dir / "best.weights.h5"
    mode = cfg.get("loss", {}).get("mode", "baseline")

    print(f"\nEvaluating {cfg['name']} (mode={mode}) — run_dir={run_dir}")

    strain, params = load_arrays(cfg["data"]["path"], split,
                                 cfg["data"].get("max_samples"))
    transforms = TargetTransforms.from_json(run_dir / "transforms.json")

    # Build the right trainer variant so weight names match the checkpoint
    if mode == "poc":
        trainer = build_sumdiff_trainer(cfg)
    else:
        # baseline mode: SumDiffTrainer in baseline mode is equivalent to
        # MultiHeadTrainer with the same model structure, so build_sumdiff_trainer
        # works for both loading and inference.
        trainer = build_sumdiff_trainer(cfg)

    trainer(strain[:1])  # build variables
    trainer.load_weights(weights)

    df = evaluate_model(trainer, strain, params, transforms)
    out_csv = run_dir / f"metrics_{split}.csv"
    df.to_csv(out_csv)
    print(df.to_string())
    print(f"\nwritten: {out_csv}")

    # NOTE: for mode=poc, the individual coa_phase / polarization_angle heads
    # are NOT trained directly (their losses are removed), so their
    # validation metrics above will show dead/frozen values.  That is
    # expected — the PoC design routes gradient through combo_A/combo_B.
    # The combo-level circular-loss metrics (tracked by SumDiffTrainer)
    # are only visible in the training progress bar, not in any CSV file.
    # For a proper A-vs-B comparison, compare the trunk-dependent heads
    # (mchirp, snr, merger_time, sky_position, inclination) between modes.

    raw_pred = trainer.predict(strain, batch_size=256, verbose=0)
    pred = transforms.inverse(raw_pred)
    true = transforms.physical_targets(params)
    scatter_grid(true, pred, transforms.heads,
                 run_dir / f"scatter_{split}.png")
    residuals_vs_param(true, pred, params[:, PARAM_COLUMNS["snr"]], "SNR",
                       transforms.heads,
                       run_dir / f"residuals_snr_{split}.png")
    residuals_vs_param(true, pred, params[:, PARAM_COLUMNS["mchirp"]], "mchirp",
                       transforms.heads,
                       run_dir / f"residuals_mchirp_{split}.png")

    # Pre-sigmoid logit diagnostic
    other_split = "validation" if split == "training" else "training"
    other_strain, other_params = load_arrays(
        cfg["data"]["path"], other_split, cfg["data"].get("max_samples")
    )
    other_raw_pred = trainer.predict(other_strain, batch_size=256, verbose=0)
    other_true = transforms.physical_targets(other_params)
    train_raw, train_true = (
        (raw_pred, true) if split == "training" else (other_raw_pred, other_true)
    )
    val_raw, val_true = (
        (other_raw_pred, other_true) if split == "training" else (raw_pred, true)
    )
    sigmoid_logit_hist(train_raw, train_true, val_raw, val_true,
                       transforms.heads, run_dir / "logits_train_vs_val.png")

    print(f"plots written to {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="validation",
                        choices=["training", "validation"],
                        help="split to evaluate against (default: validation)")
    parser.add_argument("--configs", default=None,
                        help="glob pattern for configs "
                             "(default: experiments/phic_psi_poc/config_*.yaml)")
    args = parser.parse_args()

    # Discover configs
    if args.configs:
        import glob as glob_mod
        config_files = sorted(glob_mod.glob(args.configs))
    else:
        config_files = sorted(
            str(p) for p in EXPERIMENTS_DIR.glob("config_*.yaml")
            if "smoke" not in p.name
        )

    if not config_files:
        print("No config files found.")
        return

    print(f"Found {len(config_files)} config(s):")
    for cf in config_files:
        print(f"  {cf}")

    for config_file in config_files:
        # 1. Train
        _run("train_poc.py", config_file)
        # 2. Plot
        _run(str(SCRIPTS_DIR / "plot_run.py"), config_file)
        # 3. Evaluate
        _evaluate(config_file, args.split)


if __name__ == "__main__":
    main()
