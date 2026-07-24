#!/usr/bin/env python
"""Focused runner for testing the normalize_unit magnitude-penalty fix.

Runs only the configs needed to evaluate whether fixing the |v| drift
unblocks PERIODIC head training::

    python experiments/phic_psi_poc/run_magnitude_fix.py
    python experiments/phic_psi_poc/run_magnitude_fix.py --split training

Target configs (4 total):
  - ``config_baseline.yaml``  → poc_a (baseline: circular loss on individual φc/ψ)
  - ``config_poc.yaml``       → poc_b (PoC: combo heads + curriculum weighting)
  - ``config_tcn.yaml``       → plain TCN (no SumDiffTrainer machinery)
  - ``config_cnn_attention.yaml`` → CNN Attention (second-best architecture)

These four are the minimal set to answer:
  1. Does the magnitude penalty stabilise |v|? (track std_ratio)
  2. If yes, do PERIODIC heads start learning?
  3. If PERIODIC heads learn, does poc_b's combo approach outperform poc_a's
     individual-head approach? (degeneracy hypothesis test)
  4. Is the result architecture-specific? (TCN vs CNN Attention)

Chains ``train_poc.py`` → ``plot_poc.py`` → ``evaluate_poc.py`` for each
config. Every step runs as an independent subprocess so GPU memory is freed
between steps.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent

# Only the configs needed to test the |v| fix
TARGET_CONFIGS = [
    "config_baseline.yaml",       # poc_a
    "config_poc.yaml",            # poc_b
    "config_tcn.yaml",            # plain TCN
    "config_cnn_attention.yaml",  # CNN Attention
]


def _run(script: str, *args: str) -> None:
    """Run a python script in the experiments directory as a subprocess."""
    cmd = [sys.executable, str(EXPERIMENTS_DIR / script), *args]
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", default="validation",
        choices=["training", "validation"],
        help="split to evaluate against (default: validation)",
    )
    args = parser.parse_args()

    config_files = [str(EXPERIMENTS_DIR / c) for c in TARGET_CONFIGS]

    # Verify all configs exist
    missing = [cf for cf in config_files if not Path(cf).exists()]
    if missing:
        print(f"Missing config(s): {missing}")
        return

    print(f"Running {len(config_files)} config(s) for magnitude-fix test:")
    for cf in config_files:
        print(f"  {cf}")

    for config_file in config_files:
        _run("train_poc.py", config_file)
        _run("plot_poc.py", config_file)
        _run("evaluate_poc.py", config_file, "--split", args.split)


if __name__ == "__main__":
    main()
