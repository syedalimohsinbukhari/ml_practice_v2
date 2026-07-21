#!/usr/bin/env python
"""λ=0 ablation — test whether the magnitude penalty drives the circular-loss drift.

Trains poc_a and tcn with magnitude_penalty_lambda=0.0, then overlays
circular-loss + std_ratio trajectories against the Run 7 λ=0.01 baselines.

Usage:
    python experiments/phic_psi_poc/run_lam0_ablation.py
    python experiments/phic_psi_poc/run_lam0_ablation.py --compare-only

After training, the existing diagnostic tools can be run on the λ=0 checkpoints
by editing their hardcoded config lists:
    - diagnostic_checks.py  (Check 3 = loss / std_ratio trajectories)
    - analyse_predictions.py (full per-head prediction analysis)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENTS_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENTS_DIR.parents[1]

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXPERIMENTS_DIR))

LAM0_CONFIGS = [
    "config_lam0_ablation.yaml",       # poc_a (baseline mode)
    "config_lam0_ablation_tcn.yaml",   # tcn (baseline mode)
]

# Run 7 λ=0.01 baselines for overlay
RUN7_PAIRS = {
    "config_lam0_ablation.yaml": {
        "label": "poc_a (baseline)",
        "run7_history": ROOT / "runs/phic_psi_poc_a/20260720_210936/history.csv",
    },
    "config_lam0_ablation_tcn.yaml": {
        "label": "tcn",
        "run7_history": ROOT / "runs/phic_psi_tcn/20260720_215403/history.csv",
    },
}

# (val_col, train_col, std_col) per head
METRICS = {
    "coa_phase": (
        "val_circular_loss_coa_phase",
        "circular_loss_coa_phase",
        "val_std_ratio_coa_phase",
    ),
    "polarization_angle": (
        "val_circular_loss_polarization_angle",
        "circular_loss_polarization_angle",
        "val_std_ratio_polarization_angle",
    ),
}

# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def _run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(EXPERIMENTS_DIR / script), *args]
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _latest_run_dir(config_file: str) -> Path | None:
    from gwml.training.train import load_config

    cfg = load_config(str(EXPERIMENTS_DIR / config_file))
    run_dir = Path(cfg["run_dir"])
    if not run_dir.exists():
        return None
    subdirs = sorted(
        [d for d in run_dir.iterdir() if d.is_dir() and d.name[:8].isdigit()],
        reverse=True,
    )
    return subdirs[0] if subdirs else None


# ---------------------------------------------------------------------------
# Training + plotting + evaluation
# ---------------------------------------------------------------------------


def train_all() -> None:
    config_files = [str(EXPERIMENTS_DIR / c) for c in LAM0_CONFIGS]
    missing = [cf for cf in config_files if not Path(cf).exists()]
    if missing:
        print(f"Missing config(s): {missing}")
        return

    print(f"Training {len(config_files)} λ=0 ablation config(s):")
    for cf in config_files:
        print(f"  {cf}")

    for config_file in config_files:
        _run("train_poc.py", config_file)
        _run("plot_poc.py", config_file)
        _run("evaluate_poc.py", config_file, "--split", "validation")


# ---------------------------------------------------------------------------
# Trajectory overlay — the one genuinely new analysis
# ---------------------------------------------------------------------------


def compare_trajectories() -> None:
    """Overlay λ=0 circular-loss + std_ratio trajectories on Run 7 λ=0.01."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = EXPERIMENTS_DIR / "lam0_ablation_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    COLOR_LAM0 = "#d62728"
    COLOR_RUN7 = "#1f77b4"

    head_labels = {"coa_phase": "φc (coa_phase)", "polarization_angle": "ψ (pol_angle)"}
    col_axes = [
        ("val circular loss", "1 − cosΔθ", 1.0),
        ("train circular loss", "1 − cosΔθ", 1.0),
        ("val std_ratio", "std_ratio", None),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    print("\n" + "=" * 90)
    print("λ=0 ABLATION — trajectory comparison vs Run 7 λ=0.01")
    print("=" * 90)

    for row_idx, (head_name, head_label) in enumerate(head_labels.items()):
        val_col, train_col, std_col = METRICS[head_name]
        loss_cols = [val_col, train_col, std_col]

        for col_idx, (col_title, ylabel, hline) in enumerate(col_axes):
            ax = axes[row_idx][col_idx]
            col = loss_cols[col_idx]

            for config_file in LAM0_CONFIGS:
                pair = RUN7_PAIRS[config_file]
                label_short = pair["label"]

                # λ=0
                lam0_dir = _latest_run_dir(config_file)
                if lam0_dir is None:
                    print(f"  ⚠ No run dir for {config_file}")
                    continue
                lam0_csv = lam0_dir / "history.csv"
                if not lam0_csv.exists():
                    print(f"  ⚠ No history.csv at {lam0_csv}")
                    continue
                df_lam0 = pd.read_csv(lam0_csv)
                n_ep = len(df_lam0)

                ax.plot(
                    df_lam0.index, df_lam0[col].values,
                    color=COLOR_LAM0, linewidth=1.2, alpha=0.8,
                    label=f"{label_short} λ=0",
                )

                # λ=0.01 (Run 7)
                r7_path = pair["run7_history"]
                if r7_path.exists():
                    df_r7 = pd.read_csv(r7_path)
                    ax.plot(
                        df_r7.index[:n_ep], df_r7[col].values[:n_ep],
                        color=COLOR_RUN7, linewidth=1.2, alpha=0.8,
                        linestyle="--",
                        label=f"{label_short} λ=0.01",
                    )

                # Numeric summary
                v0, v1 = df_lam0[col].iloc[0], df_lam0[col].iloc[-1]
                d0 = v1 - v0
                direction = "↑" if d0 > 0.002 else "↓" if d0 < -0.002 else "→"

                if r7_path.exists():
                    df_r7 = pd.read_csv(r7_path)
                    r7_d = df_r7[col].iloc[n_ep - 1] - df_r7[col].iloc[0]
                    r7_dir = "↑" if r7_d > 0.002 else "↓" if r7_d < -0.002 else "→"
                else:
                    r7_d, r7_dir = float("nan"), "—"

                print(
                    f"  {label_short:>20s}  {head_name:>20s}  {col_title:>22s}:  "
                    f"λ=0: {v0:.4f}→{v1:.4f} ({d0:+.4f} {direction})  "
                    f"λ=0.01: {r7_d:+.4f} {r7_dir}"
                )

            ax.set_title(f"{head_label} — {col_title}", fontsize=11)
            ax.set_xlabel("epoch")
            ax.set_ylabel(ylabel)
            if hline is not None:
                ax.axhline(y=hline, color="gray", linewidth=0.5, linestyle=":", alpha=0.6)
            elif col_title == "val std_ratio":
                ax.axhline(y=0.5, color="gray", linewidth=0.5, linestyle=":", alpha=0.6)
                ax.axhline(y=2.0, color="gray", linewidth=0.5, linestyle=":", alpha=0.6)
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        "λ=0 Ablation — Circular Loss & std_ratio vs Run 7 (λ=0.01)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    png_path = out_dir / "lam0_ablation_trajectories.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot: {png_path}")

    # --- Compact markdown summary ---
    md_path = out_dir / "lam0_ablation_report.md"
    with open(md_path, "w") as f:
        f.write("# λ=0 Ablation — Circular Loss Drift\n\n")
        f.write("**Question**: Does the upward drift in circular loss disappear at λ=0?\n\n")
        f.write("| Model | Head | Metric | λ=0 start | λ=0 end | λ=0 Δ | λ=0.01 Δ | Verdict |\n")
        f.write("|-------|------|--------|-----------|----------|-------|-----------|--------|\n")

        for config_file in LAM0_CONFIGS:
            pair = RUN7_PAIRS[config_file]
            label_short = pair["label"]
            lam0_dir = _latest_run_dir(config_file)
            if lam0_dir is None:
                continue
            df_lam0 = pd.read_csv(lam0_dir / "history.csv")
            n_ep = len(df_lam0)

            r7_path = pair["run7_history"]
            df_r7 = pd.read_csv(r7_path) if r7_path.exists() else None

            for head_name, (val_col, train_col, std_col) in METRICS.items():
                for metric_name, col in [
                    ("val circ loss", val_col),
                    ("train circ loss", train_col),
                    ("val std_ratio", std_col),
                ]:
                    if col not in df_lam0.columns:
                        continue
                    v0, v1 = df_lam0[col].iloc[0], df_lam0[col].iloc[-1]
                    d0 = v1 - v0
                    d7 = (
                        df_r7[col].iloc[n_ep - 1] - df_r7[col].iloc[0]
                        if df_r7 is not None and col in df_r7.columns
                        else float("nan")
                    )

                    if "std_ratio" in metric_name:
                        verdict = "expected (|v| diverges without penalty)"
                    elif abs(d0) < 0.003:
                        verdict = "drift ABSENT at λ=0"
                    elif d0 > 0.003:
                        verdict = (
                            "PERSISTS — penalty NOT the cause"
                            if np.isnan(d7) or d7 > 0.003
                            else "STOPPED — penalty IS the cause"
                        )
                    else:
                        verdict = "decreases at λ=0 (opposite direction)"

                    f.write(
                        f"| {label_short} | {head_name} | {metric_name} | "
                        f"{v0:.4f} | {v1:.4f} | {d0:+.4f} | {d7:+.4f} | {verdict} |\n"
                    )

        f.write(f"\n![trajectories](lam0_ablation_trajectories.png)\n\n")
        f.write("### Interpretation\n\n")
        f.write("- **PERSISTS**: drift occurs even without the penalty → penalty is NOT the cause.\n")
        f.write("- **STOPPED**: drift disappears at λ=0 → penalty (or its interaction with "
                "log_var uncertainty weighting) IS the cause. Tunable, not degeneracy evidence.\n")
        f.write("- **std_ratio drift expected**: without penalty, |v| diverges. Early epochs "
                "(before |v| drifts far) are the clean comparison window.\n")

    print(f"Report: {md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare-only", action="store_true",
        help="Skip training; only overlay trajectories (requires existing λ=0 runs)",
    )
    args = parser.parse_args()

    if not args.compare_only:
        train_all()

    compare_trajectories()


if __name__ == "__main__":
    main()