#!/usr/bin/env python3
"""Flatten the figures listed in figures_registry.yaml into paper/ for an arXiv-style submission tree.

arXiv compiles from a flat, self-contained source directory:
every file \\includegraphics'd from paper_phic_psi_degeneracy.tex (and its \\input sections) must sit alongside the
.tex sources, not one level up in the experiment's *_output/ directories where they are actually produced.

Usage:
    python3 collect_figures.py # copy PNG (+ PDF where present)
    python3 collect_figures.py --formats png # copy only the PNGs
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
            dest_path = PAPER_DIR / Path(dest_name).with_suffix(f".{fmt}")
            if not src_path.exists():
                print(f"MISSING: {src_path.relative_to(EXPERIMENT_ROOT)} (id={entry['id']})", file=sys.stderr)
                missing += 1
                continue
            if dry_run:
                print(f"{src_path.relative_to(EXPERIMENT_ROOT)} -> paper/{dest_path.name}")
                continue
            shutil.copy2(src_path, dest_path)
            print(f"copied paper/{dest_path.name}")
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="comma-separated formats to copy (default: png,pdf)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print actions without copying")
    args = parser.parse_args()

    formats = {fmt.strip().lower() for fmt in args.formats.split(",") if fmt.strip()}
    entries = load_registry(REGISTRY_PATH)
    missing = copy_figures(entries, formats, args.dry_run)

    if missing:
        print(f"\n{missing} source file(s) missing -- see MISSING lines above.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
