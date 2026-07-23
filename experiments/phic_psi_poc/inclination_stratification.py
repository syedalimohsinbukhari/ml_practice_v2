#!/usr/bin/env python
"""Inclination-stratified ang_MAE — final epoch, all models, coa_phase + polarization_angle.

Companion to snr_stratification.py (Section E of the Run 7 verification plan), on the one
axis that was never separately reported: does ang_MAE for phi_c/psi improve toward edge-on
systems, where the analytic degeneracy of the thesis chapter's Section 3 is weakest?
Bands mirror that section's face-on/edge-on split (|cos iota| > 0.9 and < 0.5).

Usage (on GPU machine):
    python experiments/phic_psi_poc/inclination_stratification.py

Output:
    inclination_output/inclination_stratification_<timestamp>.{log,md}
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
    log_path = out_dir / f"inclination_stratification_{ts}.log"
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

# Bands mirror the thesis chapter's Section 3 population-balance split.
FACE_ON_COS_IOTA = 0.9
EDGE_ON_COS_IOTA = 0.5
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


def band_masks(inclination_rad: np.ndarray) -> dict:
    cos_iota = np.cos(inclination_rad)
    return {
        "face-on":  np.abs(cos_iota) > FACE_ON_COS_IOTA,
        "mixed":    (np.abs(cos_iota) <= FACE_ON_COS_IOTA) & (np.abs(cos_iota) >= EDGE_ON_COS_IOTA),
        "edge-on":  np.abs(cos_iota) < EDGE_ON_COS_IOTA,
    }


def main():
    out_dir = Path("experiments/phic_psi_poc/inclination_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = _setup_logging(out_dir)

    print("=" * 100)
    print("INCLINATION-STRATIFIED ang_MAE — final epoch, all models, all periodic heads")
    print(f"Bands: face-on |cos(iota)| > {FACE_ON_COS_IOTA}, edge-on |cos(iota)| < {EDGE_ON_COS_IOTA}, "
          f"mixed in between. Validation samples = {N_SAMPLES}")
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

            # --- Extract inclination and build face-on/mixed/edge-on masks ---
            if "inclination" not in true:
                print("  ✗ inclination not in true labels — cannot stratify")
                continue
            iota_vals = np.ravel(true["inclination"])
            masks = band_masks(iota_vals)

            model_data = {"iota": iota_vals, "masks": masks}

            for head_name, period, null_expectation in HEADS:
                if head_name not in pred or head_name not in true:
                    print(f"  {head_name}: ✗ not in predictions")
                    continue

                pred_vals = np.ravel(pred[head_name])
                true_vals = np.ravel(true[head_name])

                band_results = {}
                print(f"\n  {head_name} (period={period:.4f}, null={null_expectation:.4f}):")
                print(f"  {'Band':<10} {'|cos iota| range':<22} {'N':>6} {'ang_MAE':>10} "
                      f"{'circ_r':>8} {'vs null':>10}")
                print(f"  {'─'*10} {'─'*22} {'─'*6} {'─'*10} {'─'*8} {'─'*10}")

                full_mae = angular_mae(true_vals, pred_vals, period)
                full_circ_r = circular_r(pred_vals, period)

                for band_name in ("face-on", "mixed", "edge-on"):
                    mask = masks[band_name]
                    n_b = int(mask.sum())
                    if n_b == 0:
                        continue
                    mae_b = angular_mae(true_vals[mask], pred_vals[mask], period)
                    circ_r_b = circular_r(pred_vals[mask], period)
                    vs_null = null_expectation - mae_b  # positive = better than null
                    band_results[band_name] = {
                        "n": n_b, "ang_mae": mae_b, "circ_r": circ_r_b, "vs_null": vs_null,
                    }
                    direction = "▼" if vs_null > 0 else "▲"
                    range_str = {
                        "face-on": f"[{FACE_ON_COS_IOTA:.1f}, 1.0]",
                        "mixed":   f"[{EDGE_ON_COS_IOTA:.1f}, {FACE_ON_COS_IOTA:.1f}]",
                        "edge-on": f"[0.0, {EDGE_ON_COS_IOTA:.1f})",
                    }[band_name]
                    print(f"  {band_name:<10} {range_str:<22} {n_b:>6} {mae_b:>10.4f} "
                          f"{circ_r_b:>8.4f} {vs_null:>+9.4f} {direction}")

                print(f"  {'ALL':<10} {'—':<22} {len(pred_vals):>6} "
                      f"{full_mae:>10.4f} {full_circ_r:>8.4f} "
                      f"{null_expectation - full_mae:>+9.4f}")

                # Does ang_MAE improve from face-on to edge-on, as the analytic
                # degeneracy structure predicts if any recoverable signal is present?
                if "face-on" in band_results and "edge-on" in band_results:
                    improves = band_results["edge-on"]["ang_mae"] < band_results["face-on"]["ang_mae"]
                else:
                    improves = None
                print(f"  Improves face-on -> edge-on? {'YES' if improves else 'NO' if improves is False else 'N/A'}")

                model_data[head_name] = {
                    "full_mae": full_mae,
                    "full_circ_r": full_circ_r,
                    "bands": band_results,
                    "improves_edge_on": improves,
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
    print("SUMMARY: Inclination-stratified ang_MAE — all models, all periodic heads")
    print("=" * 100)

    for head_name, period, null_expectation in HEADS:
        print(f"\n{'─' * 100}")
        print(f"  {head_name} (null = {null_expectation:.4f} rad)")
        print(f"{'─' * 100}")

        header = f"{'Model':<22s} {'ALL':>8s}"
        for band_name in ("face-on", "mixed", "edge-on"):
            header += f"  {band_name:>10s}"
        header += f"  {'improves?':>12s}"
        print(header)
        print("-" * len(header))

        for label in CONFIGS:
            if label not in all_data or head_name not in all_data[label]:
                print(f"{label:<22s} {'—':>8s}")
                continue

            d = all_data[label][head_name]
            row = f"{label:<22s} {d['full_mae']:>8.4f}"
            for band_name in ("face-on", "mixed", "edge-on"):
                b = d["bands"].get(band_name)
                row += f"  {b['ang_mae']:>10.4f}" if b else f"  {'—':>10s}"
            improves = d["improves_edge_on"]
            row += f"  {'YES' if improves else 'NO' if improves is False else 'N/A':>12s}"
            print(row)

    # ==================================================================
    # Key test: edge-on tercile vs null
    # ==================================================================
    print(f"\n\n{'=' * 100}")
    print("KEY TEST: edge-on-band ang_MAE vs null expectation")
    print("  (If even the edge-on band shows no improvement, the degeneracy is exact")
    print("  across the population as tested, not just at face-on)")
    print(f"{'=' * 100}")

    for head_name, period, null_expectation in HEADS:
        print(f"\n  {head_name} (null = {null_expectation:.4f} rad):")
        print(f"  {'Model':<22s} {'Edge-on MAE':>14s} {'Δ from null':>12s} {'verdict':>20s}")
        print(f"  {'─'*22} {'─'*14} {'─'*12} {'─'*20}")
        for label in CONFIGS:
            if label not in all_data or head_name not in all_data[label]:
                continue
            d = all_data[label][head_name]
            b = d["bands"].get("edge-on")
            if not b:
                continue
            edge_mae = b["ang_mae"]
            delta = null_expectation - edge_mae
            if delta > 0.02:
                verdict = "mild improvement"
            elif delta > 0.005:
                verdict = "negligible"
            elif delta > -0.005:
                verdict = "at baseline"
            else:
                verdict = "WORSE than baseline"
            print(f"  {label:<22s} {edge_mae:>14.4f} {delta:>+12.4f} {verdict:>20s}")

    # ==================================================================
    # Write markdown
    # ==================================================================
    md_path = out_dir / f"inclination_stratification_{ts}.md"
    with open(md_path, "w") as f:
        f.write(f"# Inclination-Stratified ang_MAE — All Periodic Heads\n\n")
        f.write(f"**Generated**: {ts}\n")
        f.write(f"**Bands**: face-on |cos(iota)| > {FACE_ON_COS_IOTA}, "
                f"edge-on |cos(iota)| < {EDGE_ON_COS_IOTA}, mixed in between "
                f"(matches thesis chapter Section 3)\n")
        f.write(f"**Validation samples**: {N_SAMPLES}\n\n")

        f.write("## Key question\n\n")
        f.write("The chapter's analytic prerequisite study finds the phi_c-psi degeneracy "
                "is exact face-on and only partially breakable edge-on. If any recoverable "
                "signal survives in this population, ang_MAE should improve — face-on to "
                "edge-on — below the null. If it stays flat at the null in every band, the "
                "degeneracy is exact across the tested population, not just at face-on.\n\n")

        for head_name, period, null_expectation in HEADS:
            f.write(f"### {head_name} (null = {null_expectation:.4f} rad)\n\n")
            f.write("| Model | ALL | face-on | mixed | edge-on | Improves face-on->edge-on? | "
                    "Edge-on vs null |\n")
            f.write("|-------|-----|---------|-------|---------|------------------------------|"
                    "------------------|\n")
            for label in CONFIGS:
                if label not in all_data or head_name not in all_data[label]:
                    continue
                d = all_data[label][head_name]
                b_face = d["bands"].get("face-on")
                b_mix = d["bands"].get("mixed")
                b_edge = d["bands"].get("edge-on")
                face_mae = b_face["ang_mae"] if b_face else float("nan")
                mix_mae = b_mix["ang_mae"] if b_mix else float("nan")
                edge_mae = b_edge["ang_mae"] if b_edge else float("nan")
                improves = d["improves_edge_on"]
                improves_str = "YES" if improves else "NO" if improves is False else "N/A"
                delta_null = (null_expectation - edge_mae) if b_edge else float("nan")
                f.write(f"| {label} | {d['full_mae']:.4f} | "
                        f"{face_mae:.4f} | {mix_mae:.4f} | {edge_mae:.4f} | "
                        f"{improves_str} | {delta_null:+.4f} |\n")
            f.write("\n")

        f.write("## Verdict\n\n")
        f.write("A positive result would be: edge-on ang_MAE noticeably below null and below "
                "face-on ang_MAE, consistent with the analytic degeneracy weakening away from "
                "face-on (thesis chapter Section 3).\n\n")
        f.write("Anything else — flat across bands, edge-on at or above null, no improvement "
                "toward edge-on — is consistent with the degeneracy hypothesis holding across "
                "the entire tested population, not just face-on.\n")

    print(f"\n\nMarkdown report: {md_path}")
    print("Done.")
    _teardown_logging()


if __name__ == "__main__":
    main()
