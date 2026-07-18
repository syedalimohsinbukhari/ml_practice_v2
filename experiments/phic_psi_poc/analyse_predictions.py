#!/usr/bin/env python
"""Comprehensive prediction-distribution analysis across all phic_psi runs.

For every head that each model predicts, reports:
  - Scalar heads (mchirp, merger_time, snr): MAE, R², bias, std_ratio
  - Periodic heads (coa_phase, polarization_angle, inclination):
    circular mean, circular r (concentration), histogram peaks, bimodality
  - Sky position: angular MAE, dec/ra errors

Usage (on GPU machine):
    python experiments/phic_psi_poc/analyse_predictions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "phic_psi_poc"))

from gwml.data.loader import load_arrays
from gwml.data.transforms import TargetTransforms
from gwml.training.train import latest_run_dir, load_config

CONFIGS = {
    "poc_a (baseline)":    ROOT / "experiments/phic_psi_poc/config_baseline.yaml",
    "poc_b (PoC)":         ROOT / "experiments/phic_psi_poc/config_poc.yaml",
    "tcn":                 ROOT / "experiments/phic_psi_poc/config_tcn.yaml",
    "cnn_baseline":        ROOT / "experiments/phic_psi_poc/config_cnn_baseline.yaml",
    "cnn_attention":       ROOT / "experiments/phic_psi_poc/config_cnn_attention.yaml",
    "inception_time":      ROOT / "experiments/phic_psi_poc/config_inception_time.yaml",
    "resnet1d":            ROOT / "experiments/phic_psi_poc/config_resnet1d.yaml",
}


# ======================================================================
# Helpers for periodic heads
# ======================================================================

def circular_stats(angles_rad: np.ndarray, period: float = 2 * np.pi):
    """Circular mean, concentration r, and histogram peaks for angles."""
    theta = angles_rad * (2 * np.pi / period)  # map to [0, 2π)
    s = np.sin(theta).mean()
    c = np.cos(theta).mean()
    r = np.sqrt(s**2 + c**2)
    mean_angle = (np.arctan2(s, c) % (2 * np.pi)) * (period / (2 * np.pi))
    return mean_angle, r


def period_peak_report(angles_rad: np.ndarray, period: float, label: str, n_bins=36):
    """Find histogram peaks for a periodic parameter."""
    rng = (0, period)
    counts, edges = np.histogram(angles_rad, bins=n_bins, range=rng)
    centres = (edges[:-1] + edges[1:]) / 2.0
    total = counts.sum()
    uniform = 1.0 / n_bins

    # Find local maxima significantly above uniform
    peaks = []
    for i in range(1, n_bins - 1):
        if counts[i] > counts[i - 1] and counts[i] > counts[i + 1]:
            frac = counts[i] / total
            if frac > uniform * 1.3:
                peaks.append((centres[i], frac))

    # Also find the absolute top bin
    top_idx = np.argmax(counts)
    top_angle = centres[top_idx]
    top_frac = counts[top_idx] / total

    # Chi² deviation from uniform
    expected = total * uniform
    chi2 = float(np.sum((counts - expected)**2 / (expected + 1e-12)))

    peaks.sort(key=lambda x: -x[1])

    return {
        "top_bin_angle": top_angle,
        "top_bin_frac": top_frac,
        "uniform_frac": uniform,
        "n_peaks": len(peaks),
        "peaks": [(float(p[0]), float(p[1])) for p in peaks[:4]],
        "chi2": chi2,
        "is_structured": chi2 > n_bins * 2,
    }


# ======================================================================
# Helpers for scalar heads
# ======================================================================

def scalar_stats(true_vals: np.ndarray, pred_vals: np.ndarray):
    """MAE, R², bias, std_ratio for a scalar parameter."""
    t = np.ravel(true_vals)
    p = np.ravel(pred_vals)
    res = p - t
    mae = float(np.mean(np.abs(res)))
    bias = float(np.mean(res))
    ss_tot = float(np.sum((t - t.mean())**2))
    ss_res = float(np.sum(res**2))
    r2 = float(1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan)
    std_ratio = float(np.std(p) / (np.std(t) + 1e-12))
    return {"mae": mae, "bias": bias, "r2": r2, "std_ratio": std_ratio}


# ======================================================================
# Main analysis
# ======================================================================

def analyse_run(label: str, config_path: Path) -> dict:
    from train_poc import build_sumdiff_trainer

    cfg = load_config(str(config_path))
    run_dir = latest_run_dir(cfg)
    weights = run_dir / "best.weights.h5"

    if not weights.exists():
        print(f"  {label}: ✗ no best.weights.h5 at {run_dir}")
        return None

    strain, params = load_arrays(cfg["data"]["path"], "validation", max_samples=2000)
    transforms = TargetTransforms.from_json(run_dir / "transforms.json")

    trainer = build_sumdiff_trainer(cfg)
    trainer(strain[:1])
    trainer.load_weights(str(weights))

    raw_pred = trainer.predict(strain, batch_size=256, verbose=0)
    pred = transforms.inverse(raw_pred)
    true = transforms.physical_targets(params)

    heads = transforms.heads
    result = {"label": label, "run_dir": str(run_dir), "heads": heads}

    # --- Scalar heads ---
    for h in ["mchirp", "merger_time", "snr"]:
        if h in pred and h in true:
            result[h] = scalar_stats(true[h], pred[h])

    # --- Sky position ---
    if "sky_position" in pred and "sky_position" in true:
        tp = true["sky_position"]   # (N, 3) unit vectors
        pp = pred["sky_position"]
        dot = np.sum(tp * pp, axis=-1)
        dot = np.clip(dot, -1.0, 1.0)
        ang_error_rad = np.arccos(dot)
        result["sky_position"] = {
            "angular_mae_deg": float(np.degrees(np.mean(np.abs(ang_error_rad)))),
            "angular_med_deg": float(np.degrees(np.median(np.abs(ang_error_rad)))),
        }

    # --- Periodic heads ---
    for h, period in [("coa_phase", 2*np.pi), ("polarization_angle", np.pi),
                       ("inclination", 2*np.pi)]:
        if h in pred and h in true:
            pred_vals = np.ravel(pred[h])
            true_vals = np.ravel(true[h])
            mean_angle, r = circular_stats(pred_vals, period)
            peak_info = period_peak_report(pred_vals, period, h)
            # also correlation with true
            mae = float(np.mean(np.abs(pred_vals - true_vals)))
            # For circular params, use angular MAE
            res = pred_vals - true_vals
            # wrap-aware residual for period
            res_wrapped = (res + period/2) % period - period/2
            angular_mae = float(np.mean(np.abs(res_wrapped)))
            result[h] = {
                "circular_mean": float(mean_angle),
                "circular_r": float(r),
                "angular_mae": angular_mae,
                "raw_mae": mae,
                **peak_info,
            }

    return result


def main():
    print("=" * 100)
    print("COMPREHENSIVE PREDICTION ANALYSIS — all heads, all models")
    print("=" * 100)

    results = {}
    for label, config_path in CONFIGS.items():
        print(f"\nProcessing {label} ...")
        try:
            r = analyse_run(label, config_path)
            if r:
                results[label] = r
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        print("\nNo results collected.")
        return

    # ==================================================================
    # Table 1: Scalar heads (mchirp, merger_time, snr)
    # ==================================================================
    print("\n\n" + "=" * 100)
    print("TABLE 1: SCALAR HEADS — MAE / R² / bias / std_ratio")
    print("=" * 100)

    for h in ["mchirp", "merger_time", "snr"]:
        print(f"\n--- {h} ---")
        print(f"{'model':<22s} {'MAE':>10s} {'R²':>8s} {'bias':>10s} {'std_ratio':>10s}")
        print("-" * 65)
        for label, r in results.items():
            if h in r:
                s = r[h]
                print(f"{label:<22s} {s['mae']:>10.4f} {s['r2']:>8.4f} "
                      f"{s['bias']:>10.4f} {s['std_ratio']:>10.4f}")

    # ==================================================================
    # Table 2: Sky position
    # ==================================================================
    print("\n\n" + "=" * 100)
    print("TABLE 2: SKY POSITION — angular error")
    print("=" * 100)
    print(f"{'model':<22s} {'angular MAE':>14s} {'angular median':>16s}")
    print("-" * 55)
    for label, r in results.items():
        if "sky_position" in r:
            s = r["sky_position"]
            print(f"{label:<22s} {s['angular_mae_deg']:>13.1f}° {s['angular_med_deg']:>15.1f}°")

    # ==================================================================
    # Table 3: Periodic heads — circular concentration
    # ==================================================================
    print("\n\n" + "=" * 100)
    print("TABLE 3: PERIODIC HEADS — circular concentration (r) and angular MAE")
    print("        r → 0 = uniform/random, r → 1 = perfectly concentrated")
    print("=" * 100)

    for h, period_name in [("coa_phase", "φc [0,2π)"), ("polarization_angle", "ψ [0,π)"),
                            ("inclination", "ι [0,π]")]:
        print(f"\n--- {h} ({period_name}) ---")
        print(f"{'model':<22s} {'circ_r':>8s} {'circ_mean':>10s} {'ang_MAE':>10s} "
              f"{'structured?':>12s} {'peaks':>30s}")
        print("-" * 100)
        for label, r in results.items():
            if h in r:
                s = r[h]
                peak_str = ", ".join(f"{np.degrees(p[0]):.0f}°({p[1]:.2f})"
                                     for p in s["peaks"][:3])
                if not peak_str:
                    peak_str = "—"
                print(f"{label:<22s} {s['circular_r']:>8.4f} "
                      f"{np.degrees(s['circular_mean']):>9.1f}° "
                      f"{s['angular_mae']:>10.4f} "
                      f"{'YES' if s['is_structured'] else 'no':>12s} "
                      f"{peak_str:>30s}")

    # ==================================================================
    # Table 4: Quick health check — which params are well-predicted?
    # ==================================================================
    print("\n\n" + "=" * 100)
    print("TABLE 4: QUICK HEALTH CHECK")
    print("        ✓ = well predicted   ~ = marginal   ✗ = dead/random")
    print("=" * 100)

    criteria = {
        "mchirp": lambda s: "✓" if s["r2"] > 0.8 else ("~" if s["r2"] > 0.5 else "✗"),
        "merger_time": lambda s: "✓" if s["r2"] > 0.8 else ("~" if s["r2"] > 0.5 else "✗"),
        "snr": lambda s: "✓" if s["r2"] > 0.6 else ("~" if s["r2"] > 0.3 else "✗"),
        "sky_position": lambda s: "✓" if s["angular_mae_deg"] < 45 else ("~" if s["angular_mae_deg"] < 80 else "✗"),
        "coa_phase": lambda s: "✓" if s["circular_r"] > 0.5 else ("~" if s["circular_r"] > 0.15 else "✗"),
        "polarization_angle": lambda s: "✓" if s["circular_r"] > 0.5 else ("~" if s["circular_r"] > 0.15 else "✗"),
        "inclination": lambda s: "✓" if s["circular_r"] > 0.5 else ("~" if s["circular_r"] > 0.15 else "✗"),
    }

    heads_order = ["mchirp", "merger_time", "snr", "sky_position",
                   "coa_phase", "polarization_angle", "inclination"]
    print(f"{'model':<22s} " + "".join(f"{h:>12s}" for h in heads_order))
    print("-" * (22 + 12 * len(heads_order)))
    for label, r in results.items():
        symbols = []
        for h in heads_order:
            if h in r:
                symbols.append(criteria[h](r[h]))
            else:
                symbols.append("—")
        print(f"{label:<22s} " + "".join(f"{s:>12s}" for s in symbols))

    print("\n\nDone.")
    print("Run the analyse_phic_distributions.py script for detailed φc histogram data.")


if __name__ == "__main__":
    main()
