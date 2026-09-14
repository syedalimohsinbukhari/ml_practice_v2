#!/usr/bin/env python
"""Train an experiment from a YAML config.

    python scripts/train.py configs/resnet1d.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="path to a YAML experiment config")
    args = parser.parse_args()

    from gwml.training.train import run_experiment

    run_experiment(args.config)


if __name__ == "__main__":
    main()
