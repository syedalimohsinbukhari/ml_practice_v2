#!/usr/bin/env python
"""Run train -> plot_run -> evaluate back to back for one config.

    python scripts/run_all.py configs/resnet1d.yaml
    python scripts/run_all.py configs/resnet1d.yaml --split training

Each step is the same standalone script you'd run by hand; this just chains
them so a run always gets plotted and evaluated, not just trained. Stops
immediately if a step fails, so a checkpoint never gets plotted or evaluated
against a run that didn't actually finish.
"""

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent


def _run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(SCRIPTS_DIR / script), *args]
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="validation",
                        choices=["training", "validation"],
                        help="split to evaluate against (default: validation)")
    args = parser.parse_args()

    # glob all the files in the configs directory
    q = f'{Path(__file__).parent.parent / "configs"}'
    lo_yaml = [f'{q}/{f}' for f in os.listdir(q) if f.endswith(".yaml") and 'smoke' not in f]
    lo_yaml.sort()

    for yaml_file in lo_yaml:
        _run("train.py", yaml_file)
        _run("plot_run.py", yaml_file)
        _run("evaluate.py", yaml_file, "--split", args.split)


if __name__ == "__main__":
    main()
