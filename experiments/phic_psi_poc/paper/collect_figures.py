#!/usr/bin/env python3
"""HISTORICAL: flattened the figures listed in figures_registry.yaml into paper/paper1/ for the original paper.

The original paper moved out of this repository into the Overleaf-linked sibling repo ../gw_research_paper
(formerly paper/paper1/), and that repo now holds the redo paper, so this script no longer has a destination.
Use experiments/phic_psi_poc-redo/paper/collect_figures.py for the current paper.
The original paper survives only in ../gw_research_paper's git history (up to commit d83db4b).

Usage:
    python3 collect_figures.py # copy PNG only
    python3 collect_figures.py --formats png,pdf # copy PNGs and PDFs
    python3 collect_figures.py --dry-run # print what would be copied
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

PAPER_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PAPER_DIR.parent
DEST_DIR = PAPER_DIR / "paper1"
REGISTRY_PATH = PAPER_DIR / "figures_registry.yaml"


def load_registry(path: Path) -> list[dict]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["figures"]


def copy_figures(entries: list[dict], formats: set[str], dry_run: bool) -> int:
    missing = 0
    for entry in entries:
        source = EXPERIMENT_ROOT / entry["source"]
        dest_name = entry["dest_name"]
        entry_formats = set(entry.get("formats", ["png"])) & formats
        if not entry_formats:
            continue
        for fmt in sorted(entry_formats):
            src_path = source.with_suffix(f".{fmt}")
            dest_path = DEST_DIR / Path(dest_name).with_suffix(f".{fmt}")
            if not src_path.exists():
                print(f"MISSING: {src_path.relative_to(EXPERIMENT_ROOT)} (id={entry['id']})", file=sys.stderr)
                missing += 1
                continue
            if dry_run:
                print(f"{src_path.relative_to(EXPERIMENT_ROOT)} -> paper1/{dest_path.name}")
                continue
            shutil.copy2(src_path, dest_path)
            print(f"copied paper1/{dest_path.name}")
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formats",
        default="png",
        help="comma-separated formats to copy (default: png)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print actions without copying")
    args = parser.parse_args()

    if not DEST_DIR.is_dir():
        sys.exit(
            "paper/paper1/ no longer exists: the original paper moved to ../gw_research_paper, "
            "which now holds the redo paper. Use experiments/phic_psi_poc-redo/paper/collect_figures.py instead."
        )

    formats = {fmt.strip().lower() for fmt in args.formats.split(",") if fmt.strip()}
    entries = load_registry(REGISTRY_PATH)
    missing = copy_figures(entries, formats, args.dry_run)

    if missing:
        print(f"\n{missing} source file(s) missing -- see MISSING lines above.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
