#!/usr/bin/env python
"""λ=0.05 retune — test whether a stronger magnitude penalty stabilises std_ratio.

Trains poc_a and tcn with magnitude_penalty_lambda=0.05, then overlays
circular-loss + std_ratio trajectories against both the Run 7 λ=0.01 baseline
and the λ=0 ablation, giving a 3-point curve (0, 0.01, 0.05) for each metric.

Targets identified in diagnostic_log.md Run 7 verification:
    - tcn coa_phase std_ratio still declining at λ=0.01 (0.34, -0.008/ep)
    - poc_a polarization_angle std_ratio stable but below 0.5 (0.44)

After training + plotting + evaluation, runs diagnostic_lam005_retune.py
(multi-step prediction-perturbation trace + std_ratio/circular-loss health
check) on the fresh checkpoints.

Usage:
    python experiments/phic_psi_poc/run_lam005_retune.py
    python experiments/phic_psi_poc/run_lam005_retune.py --compare-only
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
sys.path.insert(0, str(ROOT))

LAMBDA_LABEL = "0.05"

RETUNE_CONFIGS = [
    "config_lam005_retune.yaml",       # poc_a (baseline mode)
    "config_lam005_retune_tcn.yaml",   # tcn (baseline mode)
]

# Prior points on the λ sweep, for 3-way overlay.
SWEEP_PAIRS = {
    "config_lam005_retune.yaml": {
        "label": "poc_a (baseline)",
        "lam0_history": ROOT / "runs/phic_psi_lam0_ablation",
        "run7_history": ROOT / "runs/phic_psi_poc_a/20260720_210936/history.csv",
    },
    "config_lam005_retune_tcn.yaml": {
        "label": "tcn",
        "lam0_history": ROOT / "runs/phic_psi_lam0_ablation_tcn",
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


def _latest_run_dir_from_config(config_file: str) -> Path | None:
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


def _latest_run_dir(run_dir: Path) -> Path | None:
    if not run_dir.exists():
        return None
    subdirs = sorted(
        [d for d in run_dir.iterdir() if d.is_dir() and d.name[:8].isdigit()],
        reverse=True,
    )
    return subdirs[0] if subdirs else None


# ---------------------------------------------------------------------------
# Training + plotting + evaluation + diagnostics
# ---------------------------------------------------------------------------


def train_all() -> None:
    config_files = [str(EXPERIMENTS_DIR / c) for c in RETUNE_CONFIGS]
    missing = [cf for cf in config_files if not Path(cf).exists()]
    if missing:
        print(f"Missing config(s): {missing}")
        return

    print(f"Training {len(config_files)} λ={LAMBDA_LABEL} retune config(s):")
    for cf in config_files:
        print(f"  {cf}")

    for config_file in config_files:
        _run("train_poc.py", config_file)
        _run("plot_poc.py", config_file)
        _run("evaluate_poc.py", config_file, "--split", "validation")

    _run("diagnostic_lam005_retune.py")


# ---------------------------------------------------------------------------
# Trajectory overlay — 3-point λ sweep (0, 0.01, 0.05)
# ---------------------------------------------------------------------------


def compare_trajectories() -> None:
    """Overlay λ=0.05 circular-loss + std_ratio trajectories on λ=0 and λ=0.01."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.plot_style import SERIES_COLORS, update_style
    update_style()

    out_dir = EXPERIMENTS_DIR / "lam005_retune_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    COLOR_LAM0 = SERIES_COLORS["lam0_reference"]
    COLOR_RUN7 = SERIES_COLORS["run7_reference"]
    COLOR_RETUNE = SERIES_COLORS["retune"]

    head_labels = {"coa_phase": "φc (coa_phase)", "polarization_angle": "ψ (pol_angle)"}
    col_axes = [
        ("val circular loss", "1 − cosΔθ", 1.0),
        ("train circular loss", "1 − cosΔθ", 1.0),
        ("val std_ratio", "std_ratio", None),
    ]

    # Nx2 grid: metrics as rows, heads as columns (was 2x3 heads-as-rows).
    fig, axes = plt.subplots(len(col_axes), len(head_labels), figsize=(10, 15))

    print("\n" + "=" * 90)
    print(f"λ={LAMBDA_LABEL} RETUNE — trajectory comparison vs λ=0 and λ=0.01 (Run 7)")
    print("=" * 90)

    report_rows = []

    for col_idx, (head_name, head_label) in enumerate(head_labels.items()):
        val_col, train_col, std_col = METRICS[head_name]
        loss_cols = [val_col, train_col, std_col]

        for row_idx, (col_title, ylabel, hline) in enumerate(col_axes):
            ax = axes[row_idx][col_idx]
            col = loss_cols[row_idx]

            for config_file in RETUNE_CONFIGS:
                pair = SWEEP_PAIRS[config_file]
                label_short = pair["label"]

                retune_dir = _latest_run_dir_from_config(config_file)
                if retune_dir is None:
                    print(f"  ⚠ No run dir for {config_file}")
                    continue
                retune_csv = retune_dir / "history.csv"
                if not retune_csv.exists():
                    print(f"  ⚠ No history.csv at {retune_csv}")
                    continue
                df_retune = pd.read_csv(retune_csv)
                n_ep = len(df_retune)

                ax.plot(
                    df_retune.index, df_retune[col].values,
                    color=COLOR_RETUNE, linewidth=1.4, alpha=0.9,
                    label=f"{label_short} λ={LAMBDA_LABEL}",
                )

                lam0_dir = _latest_run_dir(pair["lam0_history"])
                df_lam0 = None
                if lam0_dir is not None and (lam0_dir / "history.csv").exists():
                    df_lam0 = pd.read_csv(lam0_dir / "history.csv")
                    ax.plot(
                        df_lam0.index[:n_ep], df_lam0[col].values[:n_ep],
                        color=COLOR_LAM0, linewidth=1.0, alpha=0.7,
                        linestyle=":", label=f"{label_short} λ=0",
                    )

                r7_path = pair["run7_history"]
                df_r7 = None
                if r7_path.exists():
                    df_r7 = pd.read_csv(r7_path)
                    ax.plot(
                        df_r7.index[:n_ep], df_r7[col].values[:n_ep],
                        color=COLOR_RUN7, linewidth=1.0, alpha=0.8,
                        linestyle="--", label=f"{label_short} λ=0.01",
                    )

                v0, v1 = df_retune[col].iloc[0], df_retune[col].iloc[-1]
                d_retune = v1 - v0
                d_lam0 = (
                    df_lam0[col].iloc[min(n_ep, len(df_lam0)) - 1] - df_lam0[col].iloc[0]
                    if df_lam0 is not None and col in df_lam0.columns else float("nan")
                )
                d_r7 = (
                    df_r7[col].iloc[min(n_ep, len(df_r7)) - 1] - df_r7[col].iloc[0]
                    if df_r7 is not None and col in df_r7.columns else float("nan")
                )

                print(
                    f"  {label_short:>20s}  {head_name:>20s}  {col_title:>22s}:  "
                    f"λ=0: {d_lam0:+.4f}  λ=0.01: {d_r7:+.4f}  "
                    f"λ={LAMBDA_LABEL}: {v0:.4f}→{v1:.4f} ({d_retune:+.4f})"
                )

                if col_title == "val std_ratio":
                    late = df_retune[col].iloc[-40:] if n_ep >= 40 else df_retune[col]
                    frac_unhealthy = float(((late < 0.5) | (late > 2.0)).mean())
                    late_trend = float(np.polyfit(np.arange(len(late)), late.values, 1)[0])
                    verdict = (
                        "HEALTHY" if frac_unhealthy < 0.1 and abs(late_trend) < 0.005
                        else "IMPROVED, not fully stable" if frac_unhealthy < 0.5
                        else "STILL UNHEALTHY"
                    )
                    report_rows.append(
                        (label_short, head_name, v0, v1, d_retune, d_r7,
                         frac_unhealthy, late_trend, verdict)
                    )

            ax.set_title(f"{head_label} — {col_title}", fontsize=11)
            ax.set_xlabel("epoch")
            ax.set_ylabel(ylabel)
            if hline is not None:
                ax.axhline(y=hline, color="gray", linewidth=0.5, linestyle=":", alpha=0.6)
            elif col_title == "val std_ratio":
                ax.axhline(y=0.5, color="gray", linewidth=0.5, linestyle=":", alpha=0.6)
                ax.axhline(y=2.0, color="gray", linewidth=0.5, linestyle=":", alpha=0.6)
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"λ={LAMBDA_LABEL} Retune — Circular Loss & std_ratio vs λ=0 and λ=0.01",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    png_path = out_dir / "lam005_retune_trajectories.png"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot: {png_path}")

    # --- Compact markdown summary ---
    md_path = out_dir / "lam005_retune_report.md"
    with open(md_path, "w") as f:
        f.write(f"# λ={LAMBDA_LABEL} Retune — std_ratio Stabilisation\n\n")
        f.write(
            "**Question**: Does raising the magnitude penalty from 0.01 to "
            f"{LAMBDA_LABEL} stabilise std_ratio for tcn coa_phase and poc_a "
            "polarization_angle into the healthy 0.5-2.0 band?\n\n"
        )
        f.write(
            "| Model | Head | start | end | Δ (λ={0}) | Δ (λ=0.01) | frac late "
            "epochs unhealthy | late trend/ep | Verdict |\n".format(LAMBDA_LABEL)
        )
        f.write("|-------|------|-------|-----|-----------|------------|-----"
                 "-----------------|----------------|--------|\n")
        for row in report_rows:
            label_short, head_name, v0, v1, d_retune, d_r7, frac, trend, verdict = row
            f.write(
                f"| {label_short} | {head_name} | {v0:.4f} | {v1:.4f} | "
                f"{d_retune:+.4f} | {d_r7:+.4f} | {frac:.2f} | {trend:+.5f} | "
                f"{verdict} |\n"
            )
        f.write(f"\n![trajectories](lam005_retune_trajectories.png)\n\n")
        f.write("### Interpretation\n\n")
        f.write(
            "- **HEALTHY**: <10% of the last 40 epochs outside [0.5, 2.0] and "
            "late-epoch trend within ±0.005/ep — treat this head/model as clean "
            "for the degeneracy verdict.\n"
        )
        f.write(
            "- **IMPROVED, not fully stable**: fewer unhealthy epochs than λ=0.01 "
            "but not yet clean — consider λ=0.10 (see run_lam010_retune.py).\n"
        )
        f.write(
            "- **STILL UNHEALTHY**: raising λ to {0} did not fix it — check "
            "diagnostic_lam005_retune.py's prediction-perturbation trace before "
            "concluding this architecture/head combination can't be evidence "
            "either way.\n".format(LAMBDA_LABEL)
        )

    print(f"Report: {md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare-only", action="store_true",
        help="Skip training/diagnostics; only overlay trajectories "
             "(requires existing λ=0.05 runs)",
    )
    args = parser.parse_args()

    if not args.compare_only:
        train_all()

    compare_trajectories()


if __name__ == "__main__":
    main()
