#!/usr/bin/env python3
"""Flatten the figures listed in figures_registry.yaml into paper/ for an arXiv-style submission tree.

This redo's paper/ stays flat inside this experiment directory.
The Overleaf-linked sibling repo ../gw_research_paper (next to this repository) holds the same paper, replacing the original there as of 2026-09-29.
Figures are therefore copied into both paper/ and ../gw_research_paper; the .tex sources are still synced by hand.
arXiv compiles from a flat, self-contained source directory, so every file \\includegraphics'd from main.tex
(and its \\input sections) must sit alongside the .tex sources, not one level up in the experiment's *_output/
directories where they are actually produced.

This script and figures_registry.yaml both live in paper/ itself; copies land in this same directory, next to main.tex,
not in a subdirectory.

Usage:
    python3 collect_figures.py                    # copy PNG only
    python3 collect_figures.py --formats png,pdf  # copy PNGs and PDFs
    python3 collect_figures.py --dry-run          # print what would be copied
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

PAPER_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PAPER_DIR.parent
OVERLEAF_DIR = PAPER_DIR.parents[3] / "gw_research_paper"
DEST_DIRS = [PAPER_DIR, OVERLEAF_DIR]
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
            if not src_path.exists():
                print(f"MISSING: {src_path.relative_to(EXPERIMENT_ROOT)} (id={entry['id']})", file=sys.stderr)
                missing += 1
                continue
            for dest_dir in DEST_DIRS:
                dest_path = dest_dir / Path(dest_name).with_suffix(f".{fmt}")
                label = f"{dest_dir.name}/{dest_path.name}"
                if src_path.resolve() == dest_path.resolve():
                    print(f"skip (already in place): {label}")
                    continue
                if dry_run:
                    print(f"{src_path.relative_to(EXPERIMENT_ROOT)} -> {label}")
                    continue
                shutil.copy2(src_path, dest_path)
                print(f"copied {label}")
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

    formats = {fmt.strip().lower() for fmt in args.formats.split(",") if fmt.strip()}
    if not OVERLEAF_DIR.is_dir():
        sys.exit(f"Overleaf repo not found at {OVERLEAF_DIR}")
    entries = load_registry(REGISTRY_PATH)
    missing = copy_figures(entries, formats, args.dry_run)

    if missing:
        print(f"\n{missing} source file(s) missing -- see MISSING lines above.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
