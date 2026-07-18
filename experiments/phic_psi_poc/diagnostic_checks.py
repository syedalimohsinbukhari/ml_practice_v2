#!/usr/bin/env python
"""Deep diagnostic checks for the phic_psi mode-collapse investigation.

Four checks, cheapest first:

  1. TRUE LABEL DISTRIBUTION — are true coa_phase / pol_angle values actually
     collapsed post-HDF5-load / post-TargetTransforms?  If the pipeline itself
     is feeding near-constant labels, the model predicting a constant is
     *correct* behaviour.

  2. LOSS FUNCTION — is the isotropic 1−cosΔθ circular loss actually running,
     or is a fallback Huber still active for coa_phase / pol_angle?

  3. LOG_VAR TRAJECTORY — pull the log_var (uncertainty weight) values over
     training for coa_phase / pol_angle / combo_A / combo_B.  Look for
     runaway/collapse dynamics early in training (the DeepSeek-flagged
     exp(−log_var) amplification interaction).

  4. GRADIENT ROUTING — for poc_b specifically, do a single gradient step on
     a small batch and verify that coa_phase / pol_angle weights actually
     change.  Confirms the combo-transform gradient path is wired, not just
     in the config.

Usage (on GPU machine):
    python experiments/phic_psi_poc/diagnostic_checks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "phic_psi_poc"))

# ---------------------------------------------------------------------------
# Logging: tee stdout to both console and a timestamped log file
# ---------------------------------------------------------------------------
from datetime import datetime as _dt
import contextlib as _cl

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

_LOG_FILE = None
_TEE = None

def _setup_logging(script_name: str):
    global _LOG_FILE, _TEE
    ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    _LOG_FILE = OUT_DIR / f"{script_name}_{ts}.log"
    _TEE = _Tee(str(_LOG_FILE))
    sys.stdout = _TEE

def _teardown_logging():
    if _TEE:
        sys.stdout = _TEE.stdout
        _TEE.close()

# ---------------------------------------------------------------------------

from gwml.data.loader import load_arrays
from gwml.data.transforms import TargetTransforms, PARAM_COLUMNS
from gwml.training.train import latest_run_dir, load_config

OUT_DIR = Path(__file__).resolve().parent / "diagnostic_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================================
# CHECK 1: True label distribution
# ======================================================================

def check_true_labels():
    """Histogram true coa_phase and pol_angle as they arrive at the network."""
    print("=" * 80)
    print("CHECK 1: True label distribution (post-HDF5, post-TargetTransforms)")
    print("=" * 80)

    data_path = "combined_repackaged.hdf"
    all_stats = []

    for split in ["training", "validation"]:
        print(f"\n--- {split} split ---")
        strain, params = load_arrays(data_path, split)
        iota_col = PARAM_COLUMNS["inclination"]
        phic_col = PARAM_COLUMNS["coa_phase"]
        psi_col = PARAM_COLUMNS["polarization_angle"]

        # Build transforms as train_poc does (7 heads including all periodic)
        heads = ["mchirp", "merger_time", "snr", "sky_position",
                 "coa_phase", "polarization_angle", "inclination"]
        transforms = TargetTransforms(heads=heads).fit(params)

        # Get raw (pre-transform) values
        iota_raw = params[:, iota_col]
        phic_raw = params[:, phic_col]
        psi_raw = params[:, psi_col]

        # Get transformed values (what the model actually sees as y_true)
        targets = transforms.transform(params)
        phic_transformed = targets["coa_phase"]  # (N, 2): [sin(φc), cos(φc)]
        psi_transformed = targets["polarization_angle"]  # (N, 2): [sin(2ψ), cos(2ψ)]
        incl_transformed = targets["inclination"]  # (N, 2): [sin(ι), cos(ι)]

        # Convert transformed vectors back to angles
        phic_from_transform = np.arctan2(phic_transformed[:, 0], phic_transformed[:, 1]) % (2 * np.pi)
        psi_from_transform = np.arctan2(psi_transformed[:, 0], psi_transformed[:, 1]) % (2 * np.pi)
        # Note: for ψ, the transformed angle is 2ψ (period=π), so ψ = angle/2
        psi_from_transform = psi_from_transform / 2.0

        for name, raw_vals, transformed_vals in [
            ("coa_phase (φc)", phic_raw, phic_from_transform),
            ("polarization_angle (ψ)", psi_raw, psi_from_transform),
            ("inclination (ι)", iota_raw, np.arctan2(incl_transformed[:, 0],
                                                      incl_transformed[:, 1]) % (2 * np.pi)),
        ]:
            # Basic stats
            n = len(raw_vals)
            unique_raw = len(np.unique(raw_vals.round(decimals=4)))
            unique_transformed = len(np.unique(transformed_vals.round(decimals=4)))

            # Concentration: for a uniform distribution on [0, period],
            # circular R → 0.  R → 1 means all values identical.
            period = np.pi if name.startswith("pol") else 2 * np.pi
            theta = transformed_vals * (2 * np.pi / period)
            s = np.sin(theta).mean()
            c = np.cos(theta).mean()
            circ_r = np.sqrt(s**2 + c**2)

            # Decile boundaries
            deciles_raw = np.percentile(raw_vals, [0, 10, 50, 90, 100])
            deciles_tf = np.percentile(transformed_vals, [0, 10, 50, 90, 100])

            collapsed = circ_r > 0.5

            print(f"\n  {name}:")
            print(f"    n={n}, unique (raw)={unique_raw}, unique (tf)={unique_transformed}")
            print(f"    circular R = {circ_r:.6f}  {'⚠ COLLAPSED' if collapsed else '✓ well-spread'}")
            print(f"    raw deciles:    {deciles_raw[0]:.3f}  {deciles_raw[1]:.3f}  {deciles_raw[2]:.3f}  {deciles_raw[3]:.3f}  {deciles_raw[4]:.3f}")
            print(f"    tf deciles:     {deciles_tf[0]:.3f}  {deciles_tf[1]:.3f}  {deciles_tf[2]:.3f}  {deciles_tf[3]:.3f}  {deciles_tf[4]:.3f}")

            # Bin histogram
            n_bins = 20
            counts, edges = np.histogram(transformed_vals, bins=n_bins,
                                         range=(0, period))
            max_bin_frac = counts.max() / n
            print(f"    max bin fraction = {max_bin_frac:.4f}  "
                  f"(uniform would be {1/n_bins:.4f}, collapsed would be 1.0)")

            all_stats.append({
                "split": split, "parameter": name, "n": n,
                "unique_raw": unique_raw, "unique_transformed": unique_transformed,
                "circular_r": float(circ_r), "collapsed": collapsed,
                "max_bin_frac": float(max_bin_frac),
                "uniform_frac": 1.0 / n_bins,
                "raw_min": float(deciles_raw[0]), "raw_median": float(deciles_raw[2]),
                "raw_max": float(deciles_raw[4]),
            })

    # Write CSV
    import csv
    csv_path = OUT_DIR / "true_label_stats.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "parameter", "n", "unique_raw", "unique_transformed",
                         "circular_r", "collapsed", "max_bin_frac", "uniform_frac",
                         "raw_min", "raw_median", "raw_max"])
        for s in all_stats:
            writer.writerow([s["split"], s["parameter"], s["n"],
                             s["unique_raw"], s["unique_transformed"],
                             s["circular_r"], s["collapsed"], s["max_bin_frac"],
                             s["uniform_frac"], s["raw_min"], s["raw_median"],
                             s["raw_max"]])
    print(f"\nTrue label stats CSV: {csv_path}")

    print("\n→ If any true label shows circ_r > 0.5 or max_bin > 3× uniform: "
          "DATA PIPELINE BUG — stop here, fix before anything else.")
    print("→ If all true labels are well-spread: the collapse is a training "
          "phenomenon, proceed to Check 2.")

    # ---- Plot: true label distributions ----
    _plot_true_labels(data_path, OUT_DIR, all_stats)

    return True


def _plot_true_labels(data_path, out_dir, all_stats=None):
    """Generate histograms of true φc / ψ / ι for training and validation."""
    from gwml.data.loader import load_arrays
    from gwml.data.transforms import TargetTransforms, PARAM_COLUMNS

    heads = ["mchirp", "merger_time", "snr", "sky_position",
             "coa_phase", "polarization_angle", "inclination"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    params_map = [
        (0, 0, "coa_phase", r"$\phi_c$ true [rad]", 2*np.pi, PARAM_COLUMNS["coa_phase"]),
        (0, 1, "polarization_angle", r"$\psi$ true [rad]", np.pi, PARAM_COLUMNS["polarization_angle"]),
        (0, 2, "inclination", r"$\iota$ true [rad]", np.pi, PARAM_COLUMNS["inclination"]),
    ]

    for row, split in enumerate(["training", "validation"]):
        strain, params = load_arrays(data_path, split)
        transforms = TargetTransforms(heads=heads).fit(params)
        targets = transforms.transform(params)

        for col_idx, (_, _, label, xlabel, period, param_col) in enumerate(params_map):
            ax = axes[row][col_idx]
            raw_vals = params[:, param_col]
            ax.hist(raw_vals, bins=40, range=(0, period), alpha=0.7,
                    color="steelblue", edgecolor="white", linewidth=0.3)
            # Circular stats
            theta = raw_vals * (2 * np.pi / period)
            circ_r = np.sqrt(np.sin(theta).mean()**2 + np.cos(theta).mean()**2)
            ax.set_title(f"{label} — {split}\nn={len(raw_vals)}  circ_r={circ_r:.4f}",
                        fontsize=10)
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel("count", fontsize=9)

    fig.suptitle("True Label Distributions (post-HDF5 load, pre-transform)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    png_path = out_dir / "true_label_distributions.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot: {png_path}")


# ======================================================================
# CHECK 2: Loss function verification
# ======================================================================

def check_loss_function():
    """Verify the isotropic circular loss is active, not Huber."""
    print("\n\n" + "=" * 80)
    print("CHECK 2: Loss function verification")
    print("=" * 80)

    from train_poc import build_sumdiff_trainer
    from trainer import SumDiffTrainer

    for config_name, config_path in [
        ("poc_a (baseline)", ROOT / "experiments/phic_psi_poc/config_baseline.yaml"),
        ("poc_b (PoC)", ROOT / "experiments/phic_psi_poc/config_poc.yaml"),
    ]:
        print(f"\n--- {config_name} ---")
        cfg = load_config(str(config_path))
        trainer = build_sumdiff_trainer(cfg)

        mode = trainer._poc_mode
        print(f"  mode: {mode}")
        print(f"  weighting: {trainer.weighting}")

        # Check head_loss dict — what loss function is registered per head?
        print(f"  head_loss keys: {list(trainer.head_loss.keys())}")
        for h, loss_fn in trainer.head_loss.items():
            print(f"    {h}: {loss_fn.name if hasattr(loss_fn, 'name') else type(loss_fn).__name__}")

        # In poc mode: coa_phase / pol_angle should NOT be in head_loss
        # (their individual losses are removed).  combo_A / combo_B should
        # use circular loss inside _total_loss, which doesn't show in head_loss.
        if mode == "poc":
            missing = [h for h in SumDiffTrainer._SUMDIFF_SOURCE_HEADS
                       if h in trainer.head_loss]
            if missing:
                print(f"  ⚠  INDIVIDUAL LOSSES STILL REGISTERED for: {missing}")
                print(f"     → these heads may be receiving both combo + individual gradients")
            else:
                print(f"  ✓ individual coa_phase / pol_angle losses correctly removed")

        # Check log_vars
        print(f"  log_vars keys: {[str(k) for k in trainer.log_vars.keys()]}")
        for k, v in trainer.log_vars.items():
            print(f"    {k}: {v.numpy():.4f}")

    print("\n→ If any individual loss is still registered for coa_phase / pol_angle "
          "in poc mode: loss wiring bug.")
    print("→ If head_loss uses 'huber' anywhere for these heads: isotropic loss "
          "not applied — fix.")

    return True


# ======================================================================
# CHECK 3: Log-var trajectory over training
# ======================================================================

def check_logvar_trajectory():
    """Pull log_var from history.csv for φc/ψ/combo heads."""
    print("\n\n" + "=" * 80)
    print("CHECK 3: Log-var trajectory over training")
    print("=" * 80)

    runs = {
        "poc_a (baseline)": ROOT / "experiments/phic_psi_poc/config_baseline.yaml",
        "poc_b (PoC)": ROOT / "experiments/phic_psi_poc/config_poc.yaml",
        "tcn": ROOT / "experiments/phic_psi_poc/config_tcn.yaml",
    }

    for label, config_path in runs.items():
        cfg = load_config(str(config_path))
        run_dir = latest_run_dir(cfg)
        history_csv = run_dir / "history.csv"

        if not history_csv.exists():
            print(f"\n{label}: no history.csv at {run_dir}")
            continue

        print(f"\n--- {label} ---")
        df = pd.read_csv(history_csv)

        # Find log-var columns (they're named like 'weight_coa_phase' or
        # 'log_var_coa_phase' — check what columns exist)
        relevant_cols = [c for c in df.columns
                         if any(k in c.lower() for k in
                                ["coa_phase", "polarization", "combo_a", "combo_b",
                                 "log_var", "weight_"])]

        if not relevant_cols:
            # Try looking at weight columns (exp(-log_var))
            weight_cols = [c for c in df.columns if "weight" in c.lower()]
            print(f"  weight-related columns: {weight_cols}")
            if weight_cols:
                print(f"  first 3 epochs:")
                print(df[weight_cols].head(3).to_string())
                print(f"  last 3 epochs:")
                print(df[weight_cols].tail(3).to_string())

                # Check for collapse: weights going to 0 or exploding
                for col in weight_cols:
                    first = df[col].iloc[5] if len(df) > 5 else df[col].iloc[0]
                    last = df[col].iloc[-1]
                    if first > 0:
                        ratio = last / first
                        if ratio < 0.1:
                            print(f"  ⚠  {col}: collapsed — {first:.4f} → {last:.4f} ({ratio:.3f}x)")
                        elif ratio > 10:
                            print(f"  ⚠  {col}: exploded — {first:.4f} → {last:.4f} ({ratio:.1f}x)")
        else:
            print(f"  relevant columns: {relevant_cols[:10]}")
            print(f"  first 3 epochs:")
            print(df[relevant_cols[:8]].head(3).to_string())
            print(f"  last 3 epochs:")
            print(df[relevant_cols[:8]].tail(3).to_string())

        # Also show loss trajectory for combo heads in poc mode
        loss_cols = [c for c in df.columns if "loss" in c.lower()]
        combo_loss_cols = [c for c in loss_cols if "combo" in c.lower() or
                           "circular" in c.lower()]
        if combo_loss_cols:
            print(f"\n  combo/circular loss columns: {combo_loss_cols}")
            print(f"  first 5 epochs:")
            subset_cols = combo_loss_cols[:6]
            print(df[subset_cols].head(5).to_string())
            print(f"  last 5 epochs:")
            print(df[subset_cols].tail(5).to_string())

            # Check for early collapse
            for col in combo_loss_cols[:4]:
                early = df[col].iloc[:10].mean()
                late = df[col].iloc[-10:].mean()
                if early > 0 and late / max(early, 0.001) < 0.1:
                    print(f"  ⚠  {col}: early collapse — {early:.4f} → {late:.4f}")
                elif early > 0 and late / early > 10:
                    print(f"  ⚠  {col}: loss explosion — {early:.4f} → {late:.4f}")

    print("\n→ If log_var/weight for coa_phase collapsed to zero early: "
          "exp(−log_var) amplified gradients → total mode collapse is the "
          "expected outcome.")
    print("→ If combo loss plateaued to near-zero immediately: the loss "
          "surface is flat in φc/ψ directions regardless of curriculum.")

    # ---- Plot: log-var / weight trajectories ----
    _plot_logvar_trajectories(OUT_DIR)

    return True


def _plot_logvar_trajectories(out_dir):
    """Plot weight and loss trajectories for all phic_psi runs."""
    runs = {
        "poc_a": ROOT / "runs/phic_psi_poc_a",
        "poc_b": ROOT / "runs/phic_psi_poc_b",
        "tcn": ROOT / "runs/phic_psi_tcn",
        "cnn_baseline": ROOT / "runs/phic_psi_cnn_baseline",
        "cnn_attention": ROOT / "runs/phic_psi_cnn_attention",
        "inception_time": ROOT / "runs/phic_psi_inception_time",
        "resnet1d": ROOT / "runs/phic_psi_resnet1d",
    }

    # Collect history data
    history_data = {}
    for label, run_parent in runs.items():
        latest = sorted(run_parent.glob("*/history.csv"))
        if not latest:
            continue
        df = pd.read_csv(latest[-1])
        history_data[label] = df

    if not history_data:
        return

    # Find common columns across all runs
    weight_cols = set()
    loss_cols = set()
    for df in history_data.values():
        for c in df.columns:
            if "weight" in c.lower() and "sample" not in c.lower():
                weight_cols.add(c)
            if "loss" in c.lower() and "combo" in c.lower() or "circular" in c.lower():
                loss_cols.add(c)
    weight_cols = sorted(weight_cols)
    loss_cols = sorted(loss_cols)

    n_models = len(history_data)

    # --- Plot: Weight trajectories (4x2 grid) ---
    if weight_cols:
        n_weight = len(weight_cols)
        n_cols = 2
        n_rows = (n_models + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows),
                                 sharex=True, squeeze=False)
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_weight, 10)))
        for idx, (label, df) in enumerate(history_data.items()):
            ax = axes[idx // n_cols][idx % n_cols]
            for col, color in zip(weight_cols, colors):
                if col in df.columns:
                    ax.plot(df.index, df[col], color=color, linewidth=0.8,
                            alpha=0.8, label=col)
            ax.set_title(label, fontsize=10)
            ax.legend(fontsize=7, loc="upper right", ncol=2)
            ax.grid(True, alpha=0.3)
            for col in weight_cols:
                if col in df.columns and len(df) > 10:
                    first = df[col].iloc[5]
                    last = df[col].iloc[-1]
                    if first > 0.01 and last < first * 0.1:
                        ax.annotate(f"{col} collapsed", xy=(len(df)*0.7, last),
                                    fontsize=7, color="red",
                                    bbox=dict(facecolor="white", alpha=0.8))
        # Hide unused subplots
        for j in range(n_models, n_rows * n_cols):
            axes[j // n_cols][j % n_cols].set_visible(False)
        fig.suptitle("Uncertainty Weight Trajectories (exp(-log_var))",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        png_path = out_dir / "logvar_trajectories.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot: {png_path}")

    # --- Plot: Combo loss trajectories (4x2 grid) ---
    if loss_cols:
        n_cols = 2
        n_rows = (n_models + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows),
                                 sharex=True, squeeze=False)
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(loss_cols), 10)))
        for idx, (label, df) in enumerate(history_data.items()):
            ax = axes[idx // n_cols][idx % n_cols]
            for col, color in zip(loss_cols, colors):
                if col in df.columns:
                    ax.plot(df.index, df[col], color=color, linewidth=0.8,
                            alpha=0.8, label=col)
            ax.set_title(label, fontsize=10)
            ax.legend(fontsize=7, loc="upper right", ncol=2)
            ax.grid(True, alpha=0.3)
        for j in range(n_models, n_rows * n_cols):
            axes[j // n_cols][j % n_cols].set_visible(False)
        fig.suptitle("Combo / Circular Loss Trajectories",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        png_path = out_dir / "combo_loss_trajectories.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot: {png_path}")


# ======================================================================
# CHECK 4: Gradient routing (poc_b only)
# ======================================================================

def check_gradient_routing():
    """Single gradient step, verify φc/ψ weights change."""
    print("\n\n" + "=" * 80)
    print("CHECK 4: Gradient routing — do φc/ψ weights change in poc_b?")
    print("=" * 80)

    import tensorflow as tf
    from train_poc import build_sumdiff_trainer
    from gwml.data.loader import make_dataset

    config_path = ROOT / "experiments/phic_psi_poc/config_poc.yaml"
    cfg = load_config(str(config_path))
    run_dir = latest_run_dir(cfg)
    weights_path = run_dir / "best.weights.h5"

    if not weights_path.exists():
        print(f"  ✗ no weights at {weights_path}")
        return False

    # Load a small batch
    strain, params = load_arrays(cfg["data"]["path"], "validation", max_samples=128)
    iota_col = PARAM_COLUMNS["inclination"]
    transforms = TargetTransforms(heads=["mchirp", "merger_time", "snr",
                                          "sky_position", "coa_phase",
                                          "polarization_angle", "inclination"])
    transforms.fit(params)
    ds = make_dataset(strain, params, transforms, 128, shuffle=False)
    batch = next(iter(ds))
    strain_batch, targets = batch

    # Build trainer with poc mode
    trainer = build_sumdiff_trainer(cfg)
    trainer(strain_batch[:1])
    trainer.load_weights(str(weights_path))

    # Snapshot φc / ψ head weights before gradient step
    weight_snapshots = {}
    for w in trainer.trainable_weights:
        if any(k in w.name for k in ["coa_phase", "polarization_angle"]):
            weight_snapshots[w.name] = w.numpy().copy()

    print(f"\n  Snapshot weights ({len(weight_snapshots)} tensors):")
    for name, arr in weight_snapshots.items():
        print(f"    {name}: shape={arr.shape}, norm={np.linalg.norm(arr):.6f}")

    # Manual gradient step
    with tf.GradientTape() as tape:
        y_pred = trainer(strain_batch, training=True)
        loss = trainer._total_loss(targets, y_pred)
    grads = tape.gradient(loss, trainer.trainable_weights)

    # Check gradients for φc / ψ heads
    grad_info = {}
    for w, g in zip(trainer.trainable_weights, grads):
        if any(k in w.name for k in ["coa_phase", "polarization_angle"]):
            grad_norm = float(tf.linalg.global_norm([g]) if g is not None else -1)
            grad_info[w.name] = grad_norm

    print(f"\n  Gradient norms:")
    for name, gn in grad_info.items():
        if gn < 0:
            print(f"    {name}: None (NO GRADIENT!)")
        elif gn < 1e-8:
            print(f"    {name}: {gn:.2e} ⚠ ZERO GRADIENT")
        else:
            print(f"    {name}: {gn:.6f} ✓")

    # Also check combo head weights (combo_A / combo_B are log_vars, not
    # model weights — but they drive the loss weighting)
    for w, g in zip(trainer.trainable_weights, grads):
        if any(k in w.name for k in ["combo_A", "combo_B", "log_var"]):
            gn = float(tf.linalg.global_norm([g]) if g is not None else -1)
            if "combo" in w.name:
                print(f"    {w.name}: grad_norm={gn:.6f}")

    # Apply gradients and check weight change
    trainer.optimizer.apply_gradients(zip(grads, trainer.trainable_weights))

    print(f"\n  After one gradient step:")
    any_changed = False
    for name, before in weight_snapshots.items():
        after = None
        for w in trainer.trainable_weights:
            if w.name == name:
                after = w.numpy()
                break
        if after is not None:
            delta = np.linalg.norm(after - before)
            rel_delta = delta / max(np.linalg.norm(before), 1e-12)
            flag = "✓" if delta > 1e-10 else "⚠ ZERO CHANGE"
            if delta > 1e-10:
                any_changed = True
            print(f"    {name}: Δ={delta:.2e} (rel={rel_delta:.2e}) {flag}")

    if not any_changed:
        print(f"\n  ⚠  NO φc/ψ WEIGHT CHANGE — gradient signal is not reaching these heads!")
        print(f"  → The combo-transform gradient path may be broken.")
    else:
        print(f"\n  ✓ Gradient signal reaches φc/ψ weights via combo path.")

    return True


# ======================================================================
# Main
# ======================================================================

def main():
    _setup_logging("diagnostic_checks")

    print("PHIC_PSI DEEP DIAGNOSTICS")
    print("=" * 80)
    print(f"Output directory: {OUT_DIR}")
    if _LOG_FILE:
        print(f"Log file: {_LOG_FILE}")
    print()

    check_true_labels()
    check_loss_function()
    check_logvar_trajectory()
    check_gradient_routing()

    print("\n\nAll checks complete.")
    _teardown_logging()


if __name__ == "__main__":
    main()
