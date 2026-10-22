"""Check a generated dataset file against the output contract (src/gwml/contract.py).

usage: python scripts/check_contract.py FILE [FILE ...] [--full] [--max-rows N]

Exit status 0 only if every check on every file passes.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gwml.contract import check_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+")
    parser.add_argument("--full", action="store_true",
                        help="Also require the D-03 counts (and 10 + 2 + 1 shards for a merged file).")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Rows read per dataset for the content checks (default: all).")
    args = parser.parse_args()

    failed_files = 0
    for path in args.files:
        results = check_file(path, full=args.full, max_rows=args.max_rows)
        failures = [r for r in results if not r.ok]
        print(f"{path}: {len(results) - len(failures)} of {len(results)} checks passed")
        for r in failures:
            print(f"  FAIL  {r.name}" + (f"  [{r.detail}]" if r.detail else ""))
        failed_files += bool(failures)
    return 1 if failed_files else 0


if __name__ == "__main__":
    sys.exit(main())
