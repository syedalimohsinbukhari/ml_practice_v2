#!/usr/bin/env python
"""Plot history.csv and diagnostics.csv from a PoC training run.

Thin proxy that imports ``plot_history`` and ``plot_diagnostics`` from
``scripts.plot_run`` — no logic duplication.  Reads only CSV files; does
not load any model or allocate GPU memory.

Examples::

    python experiments/phic_psi_poc/plot_poc.py \\
        experiments/phic_psi_poc/config_poc.yaml
    python experiments/phic_psi_poc/plot_poc.py \\
        experiments/phic_psi_poc/config_poc.yaml 20260713_141208
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is on the path so scripts.plot_run can import gwml
# (same pattern as train_poc.py:27)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        help="config name such as 'poc', or path such as experiments/phic_psi_poc/config_poc.yaml",
    )
    parser.add_argument(
        "run_folder",
        nargs="?",
        help=(
            "timestamp folder inside the config run_dir, e.g. 20260713_141208; "
            "default: latest run"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="directory for PNG outputs; default: the selected run directory",
    )
    args = parser.parse_args()

    from scripts.plot_run import (
        _load_config,
        _resolve_run_dir,
        plot_diagnostics,
        plot_history,
    )

    _, cfg = _load_config(args.config)
    run_dir = _resolve_run_dir(cfg, args.run_folder)
    if not run_dir.exists():
        raise FileNotFoundError(f"run directory not found: {run_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    history_csv = run_dir / "history.csv"
    diagnostics_csv = run_dir / "diagnostics.csv"
    outputs = []

    if history_csv.exists():
        history_png = out_dir / "history_summary.png"
        plot_history(history_csv, history_png)
        outputs.append(history_png)
    else:
        print(f"skipped missing file: {history_csv}")

    if diagnostics_csv.exists():
        diagnostics_png = out_dir / "diagnostics_summary.png"
        plot_diagnostics(diagnostics_csv, diagnostics_png)
        outputs.append(diagnostics_png)
    else:
        print(f"skipped missing file: {diagnostics_csv}")

    if not outputs:
        raise FileNotFoundError(
            f"neither history.csv nor diagnostics.csv found in {run_dir}"
        )

    print(f"run directory: {run_dir}")
    for output in outputs:
        print(f"written: {output}")


if __name__ == "__main__":
    main()
