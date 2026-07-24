#!/usr/bin/env python
"""Plot history.csv and diagnostics.csv from a training run.

Examples:
    python scripts/plot_run.py smoke 20260713_141208
    python scripts/plot_run.py configs/smoke.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ml_practice_v2_mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

sys.path.insert(0, str(ROOT / "src"))

try:
    from gwml.heads_spec import HEAD_SPECS
except ImportError:  # pragma: no cover - script remains useful outside package envs
    HEAD_SPECS = {}

sys.path.insert(0, str(ROOT))
from experiments.plot_style import update_style


def _load_config(config: str) -> tuple[Path, dict]:
    path = Path(config)
    if not path.exists() and path.suffix == "":
        path = ROOT / "configs" / f"{config}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"config not found: {config} "
            f"(tried {path if path.is_absolute() else path.resolve()})"
        )

    with open(path) as f:
        return path, yaml.safe_load(f)


def _run_base_dir(cfg: dict) -> Path:
    return ROOT / cfg.get("run_dir", f"runs/{cfg['name']}")


def _latest_run_dir(base_dir: Path) -> Path:
    legacy_weights = base_dir / "best.weights.h5"
    if legacy_weights.exists():
        return base_dir

    children = [p for p in base_dir.iterdir() if p.is_dir()] if base_dir.exists() else []
    if not children:
        raise FileNotFoundError(f"no run directories found under {base_dir}")

    return max(children, key=lambda p: (p.stat().st_mtime, p.name))


def _resolve_run_dir(cfg: dict, run_folder: str | None) -> Path:
    base_dir = _run_base_dir(cfg)
    if not run_folder:
        return _latest_run_dir(base_dir)

    path = Path(run_folder)
    if path.is_absolute():
        return path
    if path.exists() and (path / "history.csv").exists():
        return path
    return base_dir / run_folder


def _head_label(head: str) -> str:
    spec = HEAD_SPECS.get(head)
    return spec.label if spec else head


def _line_label(column: str, metric: str) -> str:
    validation = column.startswith("val_")
    name = column[4:] if validation else column
    head = name.removeprefix(f"{metric}_")
    label = _head_label(head)
    return f"val {label}" if validation else label


def _plot_columns(ax, df: pd.DataFrame, columns: list[str], metric: str) -> None:
    if "epoch" in df:
        x = df["epoch"]
        if not x.empty and x.iloc[0] == 0:
            x = x + 1
    else:
        x = df.index + 1
    for column in columns:
        linestyle = "--" if column.startswith("val_") else "-"
        ax.plot(x, df[column], label=_line_label(column, metric), linestyle=linestyle)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("epoch")


def plot_history(history_csv: Path, output_path: Path) -> None:
    df = pd.read_csv(history_csv)
    metric_groups = [
        ("loss", "Loss", ["loss", "val_loss"]),
        ("mae", "MAE", []),
        ("r2", "R2", []),
        ("std_ratio", "Std ratio", []),
        ("weight", "Loss weight", []),
    ]

    rows = []
    for metric, title, explicit in metric_groups:
        if explicit:
            columns = [column for column in explicit if column in df]
        else:
            prefixes = (f"{metric}_", f"val_{metric}_")
            columns = [column for column in df.columns if column.startswith(prefixes)]
        if columns:
            rows.append((metric, title, columns))

    if not rows:
        raise ValueError(f"no plottable numeric history columns found in {history_csv}")

    fig, axes = plt.subplots(3, 2, figsize=(10, 12), squeeze=False)
    flat_axes = axes.ravel()
    for ax, (metric, title, columns) in zip(flat_axes, rows):
        _plot_columns(ax, df, columns, metric)
        ax.set_title(title)
        ax.set_ylabel(title)
        # Per-panel legend, not shared: each panel plots a different set of
        # per-head lines (MAE/R2/std_ratio/weight each have their own
        # available heads), so the color-to-head mapping genuinely differs
        # panel to panel -- unlike plot_diagnostics() below, where every
        # panel repeats the exact same "subset" categories.
        ax.legend(fontsize="small", ncols=2)
    for ax in flat_axes[len(rows):]:
        ax.set_visible(False)

    fig.suptitle(f"Training history: {history_csv.parent.name}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _diagnostic_heads(df: pd.DataFrame) -> list[str]:
    heads = []
    for column in df.columns:
        if column.startswith("mae_"):
            heads.append(column.removeprefix("mae_"))
    return heads


def plot_diagnostics(diagnostics_csv: Path, output_path: Path) -> None:
    df = pd.read_csv(diagnostics_csv)
    if "epoch" not in df or "subset" not in df:
        raise ValueError("diagnostics.csv must contain epoch and subset columns")

    metrics = [("mae", "MAE"), ("r2", "R2"), ("std_ratio", "Std ratio")]
    heads = _diagnostic_heads(df)
    if not heads:
        raise ValueError(f"no mae_* diagnostic columns found in {diagnostics_csv}")

    rows = [
        (metric, title)
        for metric, title in metrics
        if any(f"{metric}_{head}" in df.columns for head in heads)
    ]
    n_rows, n_cols = len(rows), len(heads)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 3.0 * n_rows),
        squeeze=False,
        sharex=True,
    )

    subsets = list(df["subset"].drop_duplicates())
    for row_idx, (metric, title) in enumerate(rows):
        for col_idx, head in enumerate(heads):
            ax = axes[row_idx][col_idx]
            column = f"{metric}_{head}"
            if column not in df:
                ax.set_visible(False)
                continue

            for subset in subsets:
                subset_df = df[df["subset"] == subset]
                ax.plot(subset_df["epoch"], subset_df[column], marker="o", label=subset)

            if row_idx == 0:
                ax.set_title(_head_label(head))
            if col_idx == 0:
                ax.set_ylabel(title)
            ax.set_xlabel("epoch")
            ax.grid(True, alpha=0.25)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="outside lower center",
            ncols=min(len(labels), 4),
            fontsize="small",
        )
    fig.suptitle(f"Diagnostics by subset: {diagnostics_csv.parent.name}")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        help="config name such as 'smoke', or path such as configs/smoke.yaml",
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

    update_style()

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
