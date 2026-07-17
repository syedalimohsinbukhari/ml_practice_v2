#!/usr/bin/env python
"""Run train_poc -> plot_poc -> evaluate_poc back to back for all phic_psi configs.

    python experiments/phic_psi_poc/run_full.py
    python experiments/phic_psi_poc/run_full.py --split training

Searches ``experiments/phic_psi_poc/`` for ``config_*.yaml`` files, then chains
train, plot, and evaluate for each.  Every step runs as an independent subprocess
so GPU memory is freed between steps.

Works for both ``mode: baseline`` (circular loss on individual φc/ψ) and
``mode: poc`` (combo heads + curriculum weighting).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent


def _run(script: str, *args: str) -> None:
    """Run a python script in the experiments directory as a subprocess."""
    cmd = [sys.executable, str(EXPERIMENTS_DIR / script), *args]
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


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
        _run("train_poc.py", config_file)
        _run("plot_poc.py", config_file)
        _run("evaluate_poc.py", config_file, "--split", args.split)


if __name__ == "__main__":
    main()
