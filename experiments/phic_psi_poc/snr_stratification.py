#!/usr/bin/env python
"""SNR-stratified ang_MAE — final epoch, all models, coa_phase + polarization_angle.

Section E of the Run 7 verification plan: if even the highest-SNR tercile shows
no improvement over baseline, that's physics evidence, not engineering.

Usage (on GPU machine):
    python experiments/phic_psi_poc/snr_stratification.py

Output:
    snr_output/snr_stratification_<timestamp>.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "phic_psi_poc"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
from datetime import datetime as _dt

class _Tee:
    def __init__(self, file_path):
        self.stdout = sys.stdout
        try:
            self.file = open(file_path, "w", buffering=1)
        except OSError as e:
            print(f"WARNING: could not open log file {file_path}: {e}", file=self.stdout)
            self.file = None
    def write(self, data):
        self.stdout.write(data)
        if self.file:
            self.file.write(data)
    def flush(self):
        self.stdout.flush()
        if self.file:
            self.file.flush()
    def close(self):
        if self.file:
            self.file.close()

_TEE = None

def _setup_logging(out_dir):
    global _TEE
    ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"snr_stratification_{ts}.log"
    print(f"Logging to: {log_path}")
    _TEE = _Tee(str(log_path))
    sys.stdout = _TEE
    return ts

def _teardown_logging():
    global _TEE
    if _TEE:
        sys.stdout = _TEE.stdout
        _TEE.close()
        _TEE = None

# ---------------------------------------------------------------------------

from gwml.data.loader import load_arrays
from gwml.data.transforms import TargetTransforms
from gwml.training.train import latest_run_dir, load_config

CONFIGS = {
    "poc_a (baseline)":  ROOT / "experiments/phic_psi_poc/config_baseline.yaml",
    "poc_b (PoC)":       ROOT / "experiments/phic_psi_poc/config_poc.yaml",
    "tcn":               ROOT / "experiments/phic_psi_poc/config_tcn.yaml",
    "cnn_attention":     ROOT / "experiments/phic_psi_poc/config_cnn_attention.yaml",
}

HEADS = [
    ("coa_phase",           2 * np.pi, np.pi / 2),
    ("polarization_angle",  np.pi,     np.pi / 4),
    ("inclination",         2 * np.pi, np.pi / 2),
]

N_TERCILES = 3
N_SAMPLES = 5000


def angular_mae(true_rad: np.ndarray, pred_rad: np.ndarray, period: float) -> float:
    res = pred_rad - true_rad
    res_wrapped = (res + period / 2) % period - period / 2
    return float(np.mean(np.abs(res_wrapped)))


def circular_r(angles_rad: np.ndarray, period: float = 2 * np.pi) -> float:
    theta = angles_rad * (2 * np.pi / period)
    s = np.sin(theta).mean()
    c = np.cos(theta).mean()
    return float(np.sqrt(s**2 + c**2))


def main():
    out_dir = Path("experiments/phic_psi_poc/snr_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = _setup_logging(out_dir)

    print("=" * 100)
    print("SNR-STRATIFIED ang_MAE — final epoch, all models, all periodic heads")
    print(f"N terciles = {N_TERCILES}, validation samples = {N_SAMPLES}")
    print("=" * 100)

    all_data = {}

    for label, config_path in CONFIGS.items():
        print(f"\n{'─' * 100}")
        print(f"  MODEL: {label}")
        print(f"{'─' * 100}")

        try:
            from train_poc import build_sumdiff_trainer

            cfg = load_config(str(config_path))
            run_dir = latest_run_dir(cfg)
            weights = run_dir / "best.weights.h5"

            if not weights.exists():
                print(f"  ✗ no best.weights.h5 at {run_dir}")
                continue

            strain, params = load_arrays(
                cfg["data"]["path"], "validation", max_samples=N_SAMPLES
            )
            transforms = TargetTransforms.from_json(run_dir / "transforms.json")

            print(f"  Building model from {run_dir} ...")
            trainer = build_sumdiff_trainer(cfg)
            trainer(strain[:1])
            trainer.load_weights(str(weights))

            print(f"  Predicting on {len(strain)} samples ...")
            raw_pred = trainer.predict(strain, batch_size=256, verbose=0)
            pred = transforms.inverse(raw_pred)
            true = transforms.physical_targets(params)

            # --- Extract SNR ---
            if "snr" in true:
                snr_vals = np.ravel(true["snr"])
            else:
                print("  ✗ SNR not in true labels — cannot stratify")
                continue

            # --- Build tercile masks ---
            snr_sorted_idx = np.argsort(snr_vals)
            tercile_size = len(snr_vals) // N_TERCILES
            tercile_masks = {}
            tercile_snr_range = {}
            for t in range(N_TERCILES):
                if t < N_TERCILES - 1:
                    idx = snr_sorted_idx[t * tercile_size : (t + 1) * tercile_size]
                else:
                    idx = snr_sorted_idx[t * tercile_size :]
                mask = np.zeros(len(snr_vals), dtype=bool)
                mask[idx] = True
                tercile_masks[t] = mask
                tercile_snr_range[t] = (float(snr_vals[idx].min()),
                                        float(snr_vals[idx].max()))

            model_data = {"snr": snr_vals, "tercile_masks": tercile_masks,
                          "tercile_snr_range": tercile_snr_range}

            for head_name, period, null_expectation in HEADS:
                if head_name not in pred or head_name not in true:
                    print(f"  {head_name}: ✗ not in predictions")
                    continue

                pred_vals = np.ravel(pred[head_name])
                true_vals = np.ravel(true[head_name])

                tercile_results = {}
                print(f"\n  {head_name} (period={period:.4f}, null={null_expectation:.4f}):")
                print(f"  {'Tercile':<10} {'SNR range':<22} {'N':>6} {'ang_MAE':>10} "
                      f"{'circ_r':>8} {'vs null':>10}")
                print(f"  {'─'*10} {'─'*22} {'─'*6} {'─'*10} {'─'*8} {'─'*10}")

                # Full-population baseline
                full_mae = angular_mae(true_vals, pred_vals, period)
                full_circ_r = circular_r(pred_vals, period)

                for t in range(N_TERCILES):
                    mask = tercile_masks[t]
                    n_t = mask.sum()
                    mae_t = angular_mae(true_vals[mask], pred_vals[mask], period)
                    circ_r_t = circular_r(pred_vals[mask], period)
                    vs_null = null_expectation - mae_t  # positive = better than null
                    snr_lo, snr_hi = tercile_snr_range[t]
                    tercile_results[t] = {
                        "n": int(n_t), "ang_mae": mae_t, "circ_r": circ_r_t,
                        "snr_lo": snr_lo, "snr_hi": snr_hi,
                        "vs_null": vs_null,
                    }
                    direction = "▼" if vs_null > 0 else "▲"
                    print(f"  {'Low' if t==0 else 'Mid' if t==1 else 'High':<10} "
                          f"[{snr_lo:.1f}, {snr_hi:.1f}]      "
                          f"{n_t:>6} {mae_t:>10.4f} {circ_r_t:>8.4f} "
                          f"{vs_null:>+9.4f} {direction}")

                print(f"  {'ALL':<10} {'—':<22} {len(pred_vals):>6} "
                      f"{full_mae:>10.4f} {full_circ_r:>8.4f} "
                      f"{null_expectation - full_mae:>+9.4f}")

                # Trend test: Spearman rank correlation between SNR tercile and ang_MAE
                tercile_maes = [tercile_results[t]["ang_mae"] for t in range(N_TERCILES)]
                tercile_ranks = np.arange(N_TERCILES)

                # Check whether ang_MAE monotonically improves with SNR
                improves = tercile_maes[-1] < tercile_maes[0]  # high SNR < low SNR
                mono_improves = all(
                    tercile_maes[i] >= tercile_maes[i + 1]
                    for i in range(N_TERCILES - 1)
                )

                print(f"  Improve with SNR? {'YES (monotonic)' if mono_improves else 'partial' if improves else 'NO'}")

                model_data[head_name] = {
                    "full_mae": full_mae,
                    "full_circ_r": full_circ_r,
                    "terciles": tercile_results,
                    "improves_with_snr": improves,
                    "monotonic_improves": mono_improves,
                    "null_expectation": null_expectation,
                }

            all_data[label] = model_data

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n\n" + "=" * 100)
    print("SUMMARY: SNR-stratified ang_MAE — all models, all periodic heads")
    print("=" * 100)

    for head_name, period, null_expectation in HEADS:
        print(f"\n{'─' * 100}")
        print(f"  {head_name} (null = {null_expectation:.4f} rad)")
        print(f"{'─' * 100}")

        # Header
        header = f"{'Model':<22s} {'ALL':>8s}"
        for t in range(N_TERCILES):
            label_t = ['Low SNR', 'Mid SNR', 'High SNR'][t]
            header += f"  {label_t:>10s}"
        header += f"  {'improves?':>12s}"
        print(header)
        print("-" * len(header))

        for label in CONFIGS:
            if label not in all_data or head_name not in all_data[label]:
                print(f"{label:<22s} {'—':>8s}")
                continue

            d = all_data[label][head_name]
            row = f"{label:<22s} {d['full_mae']:>8.4f}"
            for t in range(N_TERCILES):
                row += f"  {d['terciles'][t]['ang_mae']:>10.4f}"
            improves = "YES ↓" if d['monotonic_improves'] else (
                "partial" if d['improves_with_snr'] else "NO")
            row += f"  {improves:>12s}"
            print(row)

    # ==================================================================
    # Key test: highest-SNR tercile vs null
    # ==================================================================
    print(f"\n\n{'=' * 100}")
    print("KEY TEST: Highest-SNR tercile ang_MAE vs null expectation")
    print("  (If even the loudest events show no improvement, the signal isn't there)")
    print(f"{'=' * 100}")

    for head_name, period, null_expectation in HEADS:
        print(f"\n  {head_name} (null = {null_expectation:.4f} rad):")
        print(f"  {'Model':<22s} {'High-SNR MAE':>14s} {'Δ from null':>12s} {'verdict':>20s}")
        print(f"  {'─'*22} {'─'*14} {'─'*12} {'─'*20}")
        for label in CONFIGS:
            if label not in all_data or head_name not in all_data[label]:
                continue
            d = all_data[label][head_name]
            high_mae = d['terciles'][N_TERCILES - 1]['ang_mae']
            delta = null_expectation - high_mae
            if delta > 0.02:
                verdict = "mild improvement"
            elif delta > 0.005:
                verdict = "negligible"
            elif delta > -0.005:
                verdict = "at baseline"
            else:
                verdict = "WORSE than baseline"
            print(f"  {label:<22s} {high_mae:>14.4f} {delta:>+12.4f} {verdict:>20s}")

    # ==================================================================
    # Write markdown
    # ==================================================================
    md_path = out_dir / f"snr_stratification_{ts}.md"
    with open(md_path, "w") as f:
        f.write(f"# SNR-Stratified ang_MAE — All Periodic Heads\n\n")
        f.write(f"**Generated**: {ts}\n")
        f.write(f"**N terciles**: {N_TERCILES}\n")
        f.write(f"**Validation samples**: {N_SAMPLES}\n\n")

        f.write("## Key question\n\n")
        f.write("If the degeneracy is breakable, the highest-SNR events (loudest, "
                "best-measured) should show lower ang_MAE than the lowest-SNR events. "
                "If even the high-SNR tercile sits at the random baseline, the signal "
                "genuinely isn't there — this is physics, not engineering.\n\n")

        for head_name, period, null_expectation in HEADS:
            f.write(f"### {head_name} (null = {null_expectation:.4f} rad)\n\n")
            f.write("| Model | ALL | Low SNR | Mid SNR | High SNR | Improves? | "
                    "High-SNR vs null |\n")
            f.write("|-------|-----|---------|---------|----------|-----------|"
                    "------------------|\n")
            for label in CONFIGS:
                if label not in all_data or head_name not in all_data[label]:
                    continue
                d = all_data[label][head_name]
                improves = "YES ↓" if d['monotonic_improves'] else (
                    "partial" if d['improves_with_snr'] else "NO")
                high_mae = d['terciles'][N_TERCILES - 1]['ang_mae']
                delta_null = null_expectation - high_mae
                f.write(f"| {label} | {d['full_mae']:.4f} | "
                        f"{d['terciles'][0]['ang_mae']:.4f} | "
                        f"{d['terciles'][1]['ang_mae']:.4f} | "
                        f"{d['terciles'][2]['ang_mae']:.4f} | "
                        f"{improves} | {delta_null:+.4f} |\n")
            f.write("\n")

        f.write("## Verdict\n\n")
        f.write("A positive result would be: high-SNR ang_MAE noticeably below null "
                "and below low-SNR ang_MAE, monotonically improving with SNR.\n\n")
        f.write("Anything else — flat across terciles, high-SNR at or above null, "
                "no monotonic trend — is consistent with the degeneracy hypothesis.\n")

    print(f"\n\nMarkdown report: {md_path}")
    print("Done.")
    _teardown_logging()


if __name__ == "__main__":
    main()