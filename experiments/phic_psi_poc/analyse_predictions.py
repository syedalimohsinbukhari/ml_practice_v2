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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "phic_psi_poc"))

# ---------------------------------------------------------------------------
# Logging: tee stdout to both console and a timestamped log file
# ---------------------------------------------------------------------------
from datetime import datetime as _dt

class _Tee:
    """Write to both a file and the original stdout."""
    def __init__(self, file_path):
        self.file = open(file_path, "w", buffering=1)
        self.stdout = sys.stdout
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        self.file.close()

_TEE = None
_LOG_FILE = None

def _setup_logging(script_name, out_dir):
    global _TEE, _LOG_FILE
    ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    _LOG_FILE = out_dir / f"{script_name}_{ts}.log"
    _TEE = _Tee(str(_LOG_FILE))
    sys.stdout = _TEE

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

    # Store raw predictions for plotting
    result["_pred"] = pred
    result["_true"] = true
    result["_heads"] = heads

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
    # Table 4: Quick health check
    # ==================================================================
    print("\n\n" + "=" * 100)
    print("TABLE 4: QUICK HEALTH CHECK")
    print("  ok = well-predicted   marginal   dead   COLLAPSE = mode collapse")
    print("  (COLLAPSE: circ_r ~ 1 + high angular MAE = all preds at single value)")
    print("=" * 100)

    def _periodic_grade(s):
        """Grade a periodic head: distinguish mode collapse from good concentration."""
        r = s["circular_r"]
        mae = s["angular_mae"]
        # Mode collapse: perfectly concentrated (r > 0.9) but wrong (high MAE)
        if r > 0.9 and mae > 0.5:
            return "COLLAPSE"
        if mae < 0.3:
            return "ok"
        if mae < 0.8:
            return "~"
        return "XX"

    criteria = {
        "mchirp": lambda s: "ok" if s["r2"] > 0.8 else ("~" if s["r2"] > 0.5 else "XX"),
        "merger_time": lambda s: "ok" if s["r2"] > 0.8 else ("~" if s["r2"] > 0.5 else "XX"),
        "snr": lambda s: "ok" if s["r2"] > 0.6 else ("~" if s["r2"] > 0.3 else "XX"),
        "sky_position": lambda s: "ok" if s["angular_mae_deg"] < 45 else ("~" if s["angular_mae_deg"] < 80 else "XX"),
        "coa_phase": _periodic_grade,
        "polarization_angle": _periodic_grade,
        "inclination": _periodic_grade,
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

    # ==================================================================
    # Write all tables to disk
    # ==================================================================
    import csv
    from datetime import datetime

    out_dir = Path("experiments/phic_psi_poc/analysis_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    _setup_logging("analyse_predictions", out_dir)
    if _LOG_FILE:
        print(f"Log file: {_LOG_FILE}")

    # --- CSV: scalar heads ---
    scalar_path = out_dir / f"scalar_heads_{ts}.csv"
    with open(scalar_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "head", "mae", "r2", "bias", "std_ratio"])
        for label, r in results.items():
            for h in ["mchirp", "merger_time", "snr"]:
                if h in r:
                    s = r[h]
                    writer.writerow([label, h, s["mae"], s["r2"], s["bias"], s["std_ratio"]])
    print(f"\nScalar heads CSV: {scalar_path}")

    # --- CSV: sky position ---
    sky_path = out_dir / f"sky_position_{ts}.csv"
    with open(sky_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "angular_mae_deg", "angular_median_deg"])
        for label, r in results.items():
            if "sky_position" in r:
                writer.writerow([label, r["sky_position"]["angular_mae_deg"],
                                 r["sky_position"]["angular_med_deg"]])
    print(f"Sky position CSV: {sky_path}")

    # --- CSV: periodic heads ---
    period_path = out_dir / f"periodic_heads_{ts}.csv"
    with open(period_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "head", "circular_r", "circular_mean_deg",
                         "angular_mae", "n_peaks", "peaks", "is_structured", "chi2"])
        for label, r in results.items():
            for h in ["coa_phase", "polarization_angle", "inclination"]:
                if h in r:
                    s = r[h]
                    peak_str = "; ".join(f"{np.degrees(p[0]):.0f}°:{p[1]:.3f}"
                                         for p in s["peaks"][:4])
                    writer.writerow([label, h, s["circular_r"],
                                     np.degrees(s["circular_mean"]),
                                     s["angular_mae"], s["n_peaks"],
                                     peak_str, s["is_structured"], s["chi2"]])
    print(f"Periodic heads CSV: {period_path}")

    # --- CSV: health check ---
    health_path = out_dir / f"health_check_{ts}.csv"
    with open(health_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + heads_order)
        for label, r in results.items():
            row = [label]
            for h in heads_order:
                if h in r:
                    row.append(criteria[h](r[h]))
                else:
                    row.append("—")
            writer.writerow(row)
    print(f"Health check CSV: {health_path}")

    # --- Consolidated markdown report ---
    md_path = out_dir / f"analysis_report_{ts}.md"
    with open(md_path, "w") as f:
        f.write(f"# Prediction Analysis Report — {ts}\n\n")

        f.write("## Scalar Heads\n\n")
        for h in ["mchirp", "merger_time", "snr"]:
            f.write(f"### {h}\n\n")
            f.write("| model | MAE | R² | bias | std_ratio |\n")
            f.write("|-------|-----|----|------|----------|\n")
            for label, r in results.items():
                if h in r:
                    s = r[h]
                    f.write(f"| {label} | {s['mae']:.4f} | {s['r2']:.4f} | {s['bias']:.4f} | {s['std_ratio']:.4f} |\n")
            f.write("\n")

        f.write("## Sky Position\n\n")
        f.write("| model | angular MAE | angular median |\n")
        f.write("|-------|------------|---------------|\n")
        for label, r in results.items():
            if "sky_position" in r:
                s = r["sky_position"]
                f.write(f"| {label} | {s['angular_mae_deg']:.1f}° | {s['angular_med_deg']:.1f}° |\n")
        f.write("\n")

        f.write("## Periodic Heads\n\n")
        for h, period_name in [("coa_phase", "φc [0,2π)"), ("polarization_angle", "ψ [0,π)"),
                                ("inclination", "ι [0,π]")]:
            f.write(f"### {h} ({period_name})\n\n")
            f.write("| model | circ_r | circ_mean | ang_MAE | peaks |\n")
            f.write("|-------|--------|-----------|---------|-------|\n")
            for label, r in results.items():
                if h in r:
                    s = r[h]
                    peak_str = ", ".join(f"{np.degrees(p[0]):.0f}°({p[1]:.2f})"
                                         for p in s["peaks"][:3]) or "—"
                    f.write(f"| {label} | {s['circular_r']:.4f} | "
                            f"{np.degrees(s['circular_mean']):.1f}° | "
                            f"{s['angular_mae']:.4f} | {peak_str} |\n")
            f.write("\n")

        f.write("## Health Check\n\n")
        f.write("| model | " + " | ".join(heads_order) + " |\n")
        f.write("|-------|" + "|".join(":---:" for _ in heads_order) + "|\n")
        for label, r in results.items():
            symbols = []
            for h in heads_order:
                if h in r:
                    symbols.append(criteria[h](r[h]))
                else:
                    symbols.append("—")
            f.write(f"| {label} | " + " | ".join(symbols) + " |\n")

    print(f"Markdown report: {md_path}")

    # ==================================================================
    # Generate plots
    # ==================================================================
    _generate_plots(results, out_dir, ts)

    print("\n\nDone.")
    _teardown_logging()


# ======================================================================
# Plot generation
# ======================================================================

def _generate_plots(results, out_dir, ts):
    """Generate PNG visualizations for all models × heads."""
    labels = list(results.keys())
    n_models = len(labels)
    if n_models == 0:
        return

    # Color palette (colorblind-friendly)
    MODEL_COLORS = plt.cm.tab10(np.linspace(0, 1, max(n_models, 10)))

    # ---- Plot 1: Periodic head prediction histograms (4x2 grid) ----
    n_cols = 2
    n_rows = (n_models + n_cols - 1) // n_cols
    for h, period, title in [("coa_phase", 2*np.pi, "φc (coa_phase)"),
                              ("polarization_angle", np.pi, "ψ (polarization_angle)"),
                              ("inclination", 2*np.pi, "ι (inclination)")]:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows),
                                 sharex=True, squeeze=False)
        idx = 0
        for label, r in results.items():
            if h not in r.get("_pred", {}):
                continue
            ax = axes[idx // n_cols][idx % n_cols]
            pred_vals = np.ravel(r["_pred"][h])
            true_vals = np.ravel(r["_true"][h])

            ax.hist(pred_vals, bins=40, range=(0, period), alpha=0.7,
                    color=MODEL_COLORS[idx], label="predicted", edgecolor="white",
                    linewidth=0.3)
            ax.hist(true_vals, bins=40, range=(0, period), alpha=0.3,
                    color="grey", label="true", edgecolor="white",
                    linewidth=0.3)

            stats = r.get(h, {})
            circ_r = stats.get("circular_r", 0)
            ang_mae = stats.get("angular_mae", 0)
            peak_str = ", ".join(f"{np.degrees(p[0]):.0f}°" for p in stats.get("peaks", [])[:2])
            ax.set_title(label, fontsize=10)
            ax.yaxis.set_major_locator(MaxNLocator(3))
            if circ_r > 0.9 and ang_mae > 0.5:
                status = "COLLAPSE"
            elif ang_mae < 0.3:
                status = "good"
            elif ang_mae < 0.8:
                status = "marginal"
            else:
                status = "dead"
            ax.text(0.99, 0.95, f"r={circ_r:.3f}  MAE={ang_mae:.2f}  {status}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
            if peak_str:
                ax.text(0.99, 0.78, f"peaks: {peak_str}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=7, fontfamily="monospace",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
            idx += 1

        # Hide unused subplots
        for j in range(idx, n_rows * n_cols):
            axes[j // n_cols][j % n_cols].set_visible(False)
        axes[0][0].legend(fontsize=8, loc="upper left")
        fig.supxlabel(f"{title} [rad]", fontsize=10)
        fig.suptitle(f"Prediction Distributions — {title}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        png_path = out_dir / f"histogram_{h}_{ts}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot: {png_path}")

    # ---- Plot 2: Scalar heads — true vs predicted scatter ----
    for h, xlabel in [("mchirp", r"$\mathcal{M}_c$ true"),
                       ("merger_time", "$t_{merger}$ true [s]"),
                       ("snr", "SNR true")]:
        cols = min(n_models, 4)
        rows = (n_models + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows),
                                 squeeze=False)
        idx = 0
        for label, r in results.items():
            if h not in r.get("_pred", {}):
                continue
            ax = axes[idx // cols][idx % cols]
            t = np.ravel(r["_true"][h])
            p = np.ravel(r["_pred"][h])
            ax.scatter(t, p, s=0.5, alpha=0.3, color=MODEL_COLORS[idx])
            # Diagonal
            lims = [min(t.min(), p.min()), max(t.max(), p.max())]
            ax.plot(lims, lims, "k--", linewidth=0.5, alpha=0.5)
            stats = r.get(h, {})
            ax.set_title(f"{label}\nR²={stats.get('r2',0):.3f}  MAE={stats.get('mae',0):.3f}",
                        fontsize=9)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(f"{h} predicted", fontsize=8)
            idx += 1
        # Hide unused subplots
        for j in range(idx, rows * cols):
            axes[j // cols][j % cols].set_visible(False)
        fig.suptitle(f"True vs Predicted — {h}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        png_path = out_dir / f"scatter_{h}_{ts}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot: {png_path}")

    # ---- Plot 3: Health check heatmap ----
    _plot_health_heatmap(results, out_dir, ts)


def _plot_health_heatmap(results, out_dir, ts):
    """Color-coded grid: models × heads with ✓/~ /✗/💀."""
    heads_order = ["mchirp", "merger_time", "snr", "sky_position",
                   "coa_phase", "polarization_angle", "inclination"]
    labels = list(results.keys())

    data = np.zeros((len(labels), len(heads_order)))
    annot = []
    for label, r in results.items():
        row = []
        row_vals = []
        for h in heads_order:
            if h not in r:
                row.append("—")
                row_vals.append(-1)
            elif h in ["mchirp", "merger_time", "snr"]:
                s = r[h]
                row.append(f"R²={s['r2']:.2f}")
                row_vals.append(max(0, s["r2"]))
            elif h == "sky_position":
                s = r[h]
                row.append(f"{s['angular_mae_deg']:.0f}°")
                row_vals.append(max(0, 1 - s["angular_mae_deg"]/180))
            else:
                s = r[h]
                circ_r = s["circular_r"]
                ang_mae = s["angular_mae"]
                if circ_r > 0.9 and ang_mae > 0.5:
                    row.append(f"COLLAPSE r={circ_r:.2f}")
                    row_vals.append(-0.2)
                elif ang_mae < 0.3:
                    row.append(f"ok MAE={ang_mae:.2f}")
                    row_vals.append(1.0)
                elif ang_mae < 0.8:
                    row.append(f"~ MAE={ang_mae:.2f}")
                    row_vals.append(0.5)
                else:
                    row.append(f"XX MAE={ang_mae:.2f}")
                    row_vals.append(0.1)
        annot.append(row)
        data[len(annot)-1] = row_vals

    fig, ax = plt.subplots(figsize=(2 + 1.8*len(heads_order), 1.2 + 0.45*len(labels)))

    # Custom colormap: red (collapse) → white (dead) → yellow (marginal) → green (good)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("health",
        [(0.8, 0.1, 0.1),    # red for collapse
         (0.95, 0.95, 0.95),  # white for dead
         (1.0, 0.9, 0.5),     # yellow for marginal
         (0.3, 0.8, 0.3)],    # green for good
        N=256)

    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=-0.3, vmax=1.2)

    for i in range(len(labels)):
        for j in range(len(heads_order)):
            val = data[i, j]
            color = "white" if val < 0 else "black"
            ax.text(j, i, annot[i][j], ha="center", va="center",
                    fontsize=8, fontfamily="monospace", color=color,
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white" if val >= 0 else "none",
                              alpha=0.6 if val >= 0 else 0))

    ax.set_xticks(range(len(heads_order)))
    ax.set_xticklabels(heads_order, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("Health Check — ok=good  ~=marginal  XX=dead  COLLAPSE=mode collapse",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    png_path = out_dir / f"health_check_{ts}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot: {png_path}")


if __name__ == "__main__":
    main()
