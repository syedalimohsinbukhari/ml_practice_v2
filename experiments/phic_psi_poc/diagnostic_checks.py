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

class _Tee:
    """Write to both a file and the original stdout."""
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

_LOG_FILE = None
_TEE = None

def _setup_logging(script_name: str):
    global _LOG_FILE, _TEE
    ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    _LOG_FILE = OUT_DIR / f"{script_name}_{ts}.log"
    # Print to original stdout BEFORE redirecting
    print(f"Logging to: {_LOG_FILE}", file=sys.stdout)
    _TEE = _Tee(str(_LOG_FILE))
    sys.stdout = _TEE

def _teardown_logging():
    global _TEE
    if _TEE:
        sys.stdout = _TEE.stdout
        _TEE.close()
        _TEE = None

# ---------------------------------------------------------------------------

from gwml.data.loader import load_arrays, make_dataset
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

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    params_map = [
        (0, "coa_phase", r"$\phi_c$ true [rad]", 2*np.pi, PARAM_COLUMNS["coa_phase"]),
        (1, "polarization_angle", r"$\psi$ true [rad]", np.pi, PARAM_COLUMNS["polarization_angle"]),
        (2, "inclination", r"$\iota$ true [rad]", np.pi, PARAM_COLUMNS["inclination"]),
        (3, "declination", "Dec true [rad]", np.pi, PARAM_COLUMNS["declination"]),
        (4, "ra", "RA true [rad]", 2*np.pi, PARAM_COLUMNS["ra"]),
    ]

    for row, split in enumerate(["training", "validation"]):
        strain, params = load_arrays(data_path, split)
        transforms = TargetTransforms(heads=heads).fit(params)
        targets = transforms.transform(params)

        for col_idx, label, xlabel, period, param_col in params_map:
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
        ("tcn (TNC)", ROOT / "experiments/phic_psi_poc/config_tcn.yaml"),
        ("cnn_attention (ATX)", ROOT / "experiments/phic_psi_poc/config_cnn_attention.yaml")
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
        n_cols = 4
        n_rows = (n_models + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows),
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

    # --- Plot: Combo loss trajectories (2x4 grid) ---
    if loss_cols:
        n_cols = 4
        n_rows = (n_models + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows),
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

    # ------------------------------------------------------------------
    # Weight discovery: use layer-name lookup (same approach as Check 5).
    #
    # trainer.base.trainable_weights returns flat names like "kernel", "bias"
    # — Keras strips the layer prefix, so substring matching on name fails
    # (the root cause of the original bug).
    #
    # Instead, access layers by name via get_layer() and pull their weights
    # directly.  Check 5 already demonstrates this pattern works.
    # ------------------------------------------------------------------
    base_model = trainer.base
    phi_psi_heads = ["coa_phase", "polarization_angle"]

    # Snapshot φc/ψ head weights and collect their variable tensors
    phi_psi_snapshots = {}    # w.name -> numpy snapshot (for delta check)
    phi_psi_vars = []          # tf.Variable list (for gradient computation)

    for head_name in phi_psi_heads:
        try:
            layer = base_model.get_layer(head_name)
        except ValueError:
            print(f"  Layer '{head_name}' not found in model — skipping")
            continue

        print(f"  Layer '{head_name}' has "
              f"{len(layer.trainable_weights)} trainable weight(s):")
        for w in layer.trainable_weights:
            phi_psi_snapshots[w.name] = w.numpy().copy()
            phi_psi_vars.append(w)
            print(f"    {w.name}: shape={w.shape}  "
                  f"norm={np.linalg.norm(w.numpy()):.6f}")

    if not phi_psi_snapshots:
        print(f"\n  ⚠ No coa_phase/polarization_angle weights found "
              f"via layer-name lookup!")
        return False

    # Also collect combo log-var weights (these ARE discoverable by name
    # since they're added via self.add_weight on the trainer itself)
    log_var_vars = []
    for w in trainer.trainable_weights:
        if any(k in w.name for k in ["log_var_combo", "combo_A", "combo_B"]):
            log_var_vars.append(w)

    # Combine all variables we care about for a single gradient computation
    all_grad_vars = phi_psi_vars + log_var_vars

    # ------------------------------------------------------------------
    # Manual gradient step
    # ------------------------------------------------------------------
    # Capture predictions BEFORE the step for later perturbation check
    y_pred_before = trainer(strain_batch[:8], training=False)

    with tf.GradientTape() as tape:
        y_pred = trainer(strain_batch, training=True)
        loss = trainer._total_loss(targets, y_pred)
    grads = tape.gradient(loss, all_grad_vars)

    if grads is None:
        print(f"\n  ⚠ tape.gradient returned None!")
        return False

    # Gradient norms for φc/ψ head weights
    n_phi = len(phi_psi_vars)
    print(f"\n  Gradient norms for φc/ψ-related weights:")
    for w, g in zip(phi_psi_vars, grads[:n_phi]):
        gn = float(tf.linalg.global_norm([g]) if g is not None else -1)
        if gn < 0:
            print(f"    {w.name}: None (NO GRADIENT)")
        elif gn < 1e-8:
            print(f"    {w.name}: {gn:.2e} (ZERO)")
        else:
            print(f"    {w.name}: {gn:.6f} ok")

    # Gradient norms for combo log-var weights
    print(f"\n  Gradient norms for combo log-var weights:")
    for w, g in zip(log_var_vars, grads[n_phi:]):
        gn = float(tf.linalg.global_norm([g]) if g is not None else -1)
        print(f"    {w.name}: grad_norm={gn:.6f}")

    # Apply gradients
    trainer.optimizer.apply_gradients(zip(grads, all_grad_vars))

    # Get post-step predictions
    y_pred_after = trainer(strain_batch[:8], training=False)

    # ------------------------------------------------------------------
    # Weight deltas for φc/ψ weights
    # ------------------------------------------------------------------
    print(f"\n  Weight deltas after one gradient step:")
    any_changed = False
    for name, before in phi_psi_snapshots.items():
        # Find the current value from the updated variable
        after = None
        for w in phi_psi_vars:
            if w.name == name:
                after = w.numpy()
                break
        if after is not None:
            delta = np.linalg.norm(after - before)
            rel_delta = delta / max(np.linalg.norm(before), 1e-12)
            if delta > 1e-10:
                any_changed = True
                print(f"    {name}: delta={delta:.2e} (rel={rel_delta:.2e}) ok")
            else:
                print(f"    {name}: delta={delta:.2e} ZERO CHANGE")

    if phi_psi_snapshots and not any_changed:
        print(f"\n  ⚠ NO φc/ψ WEIGHT CHANGE — gradient signal is not "
              f"reaching these heads!")
    elif phi_psi_snapshots and any_changed:
        print(f"\n  ✓ gradient signal reaches φc/ψ weights via combo path.")

    # ------------------------------------------------------------------
    # Prediction perturbation check
    # ------------------------------------------------------------------
    print(f"\n  Prediction perturbation per head (mean|Δ| after gradient step):")
    perturbed_any = False
    for head_name in y_pred_before.keys():
        # Drop auxiliary outputs (e.g. sky_position_mu_raw, sky_position_kappa_raw)
        # to avoid double counting.
        if head_name in targets:
            before_val = y_pred_before[head_name].numpy()
            after_val = y_pred_after[head_name].numpy()
            diff_norm = np.mean(np.linalg.norm(
                after_val - before_val, axis=-1
            )) if before_val.ndim >= 2 else np.mean(np.abs(after_val - before_val))
            before_norm = np.mean(np.linalg.norm(
                before_val, axis=-1
            )) if before_val.ndim >= 2 else np.mean(np.abs(before_val))
            rel_change = diff_norm / max(before_norm, 1e-12)
            flagged = " *changed*" if diff_norm > 1e-8 else ""
            if diff_norm > 1e-8:
                perturbed_any = True
            print(f"    {head_name}: mean|Δ|={diff_norm:.2e}  "
                  f"rel_change={rel_change:.2e}{flagged}")
        else:
            print(f"    {head_name}: (auxiliary output, skipped)")

    if perturbed_any:
        print(f"\n  ✓ Gradient step perturbed model predictions — "
              f"backprop is wired through the loss to at least some heads.")
    else:
        print(f"\n  ⚠ Predictions unchanged — gradient step had no effect "
              f"on any head's output.")

    return True


# ======================================================================
# CHECK 5: Pre-tanh logit saturation check
# ======================================================================

def check_tanh_saturation():
    """Dump pre-activation logits for coa_phase/pol_angle vs mchirp.

    PERIODIC heads use tanh activation.  If the pre-tanh logits are large
    magnitude (|x| > 5), tanh'(x) = 1−tanh²(x) ≈ 0 — vanishing gradient.
    This would explain frozen weights even with correct loss wiring.
    """
    print("\n\n" + "=" * 80)
    print("CHECK 5: Pre-tanh logit saturation")
    print("=" * 80)

    for config_name, config_path in [
        ("poc_a (baseline)", ROOT / "experiments/phic_psi_poc/config_baseline.yaml"),
        ("poc_b (PoC)", ROOT / "experiments/phic_psi_poc/config_poc.yaml"),
        ("tcn", ROOT / "experiments/phic_psi_poc/config_tcn.yaml"),
    ]:
        print(f"\n--- {config_name} ---")
        cfg = load_config(str(config_path))
        run_dir = latest_run_dir(cfg)
        weights_path = run_dir / "best.weights.h5"

        from train_poc import build_sumdiff_trainer

        # Build fresh (random init) and trained versions
        trainer_init = build_sumdiff_trainer(cfg)
        strain, params = load_arrays(cfg["data"]["path"], "validation", max_samples=128)
        batch = next(iter(make_dataset(strain, params,
                       TargetTransforms(heads=trainer_init.head_names).fit(params),
                       128, shuffle=False)))
        strain_batch, targets = batch

        # Get intermediate layer outputs for tanh-activated heads
        # The Dense output layer named e.g. "coa_phase" produces pre-activation
        # logits, and the activation (tanh) is applied by the layer's activation
        # attribute.  We need the output BEFORE activation.

        # Build an intermediate model that outputs pre-activation for tanh heads
        tanh_heads = ["coa_phase", "polarization_angle"]
        base_model = trainer_init.base

        # Check which heads exist in the model
        model_output_names = [o.name.split("/")[0] for o in base_model.outputs]
        # Actually outputs are named e.g. "coa_phase", "coa_phase_hidden" etc.
        # We want the pre-activation of the final Dense layer.

        # Simpler approach: get the layer by name and inspect its weights
        for head_name in tanh_heads + ["mchirp"]:
            # Find the output layer for this head
            try:
                layer = base_model.get_layer(head_name)
            except ValueError:
                print(f"  {head_name}: layer not found in model")
                continue

            # Get layer weights
            kernel, bias = layer.get_weights() if layer.weights else (None, None)
            if kernel is not None:
                kernel_norm = np.linalg.norm(kernel)
                bias_norm = np.linalg.norm(bias) if bias is not None else 0
                # For a hidden_dim=64 input, output_dim=2
                # Expected activation magnitude ≈ bias ± kernel·features
                # Features are post-GAP/GMP, typically small (~0.1-0.5)
                # So pre-activation ≈ bias ± O(kernel_norm * feature_norm)
                max_input_norm = 2.0  # rough estimate of GAP feature magnitude
                est_logit_mag = abs(bias).max() + kernel_norm * max_input_norm / np.sqrt(kernel.shape[0])
                print(f"  {head_name}: kernel_norm={kernel_norm:.4f} bias_range=[{bias.min():.4f}, {bias.max():.4f}] "
                      f"est_logit_mag≈{est_logit_mag:.2f} "
                      f"({'SATURATED' if est_logit_mag > 3 else 'ok' if est_logit_mag < 1.5 else 'marginal'})")
            else:
                print(f"  {head_name}: no weights")

        # Also load trained weights and compare
        trainer_init(strain_batch[:1])
        trainer_init.load_weights(str(weights_path))
        print(f"  --- after loading trained weights ---")
        for head_name in tanh_heads + ["mchirp"]:
            try:
                layer = trainer_init.base.get_layer(head_name)
            except ValueError:
                continue
            kernel, bias = layer.get_weights() if layer.weights else (None, None)
            if kernel is not None:
                kernel_norm = np.linalg.norm(kernel)
                est_logit_mag = abs(bias).max() + kernel_norm * 2.0 / np.sqrt(kernel.shape[0])
                print(f"  {head_name}: kernel_norm={kernel_norm:.4f} bias_range=[{bias.min():.4f}, {bias.max():.4f}] "
                      f"est_logit_mag≈{est_logit_mag:.2f}")

    print("\n→ If est_logit_mag > 3 for coa_phase/pol_angle: tanh saturation confirmed.")
    print("  Fix: switch activation to 'linear' for PERIODIC heads, or reduce init variance.")

    return True


# ======================================================================
# CHECK 6: Gradient chain through circular loss pipeline
# ======================================================================

def check_gradient_chain():
    """Trace gradient norms at each stage of the circular loss pipeline.

    Replicates the ``_poc_total_loss`` computation inside a
    ``tf.GradientTape`` while watching intermediate tensors, then reports
    the gradient norm at each stage to identify where signal attenuates.

    Key intermediates (in order):
        z_phic_raw / z_psi_raw  (tanh model outputs)
        z_phic_norm / z_psi_norm  (after ``tf_normalize_unit``)
        combo_A_pred / combo_B_pred  (after ``tf_complex_mul`` / conj)
        per-sample circular losses, w-iota weighted, log-var scaled

    Also computes the gradient for ``inclination`` as a healthy baseline
    (same PERIODIC encoding, trains fine per Check 4).
    """
    print("\n\n" + "=" * 80)
    print("CHECK 6: Gradient chain through circular loss pipeline")
    print("=" * 80)

    import tensorflow as tf
    from train_poc import build_sumdiff_trainer
    from curriculum import tf_w_iota
    from transform_utils import (
        tf_normalize_unit,
        tf_complex_mul,
        tf_complex_mul_conj,
    )

    config_path = ROOT / "experiments/phic_psi_poc/config_poc.yaml"
    cfg = load_config(str(config_path))
    run_dir = latest_run_dir(cfg)
    weights_path = run_dir / "best.weights.h5"

    if not weights_path.exists():
        print(f"  no weights at {weights_path}")
        return False

    # ---- Load a single small batch (8 samples) ----
    strain, params = load_arrays(cfg["data"]["path"], "validation", max_samples=128)
    transforms = TargetTransforms(heads=["mchirp", "merger_time", "snr",
                                          "sky_position", "coa_phase",
                                          "polarization_angle", "inclination"])
    transforms.fit(params)
    ds = make_dataset(strain, params, transforms, 8, shuffle=False)
    batch = next(iter(ds))
    strain_batch, targets = batch
    n = int(strain_batch.shape[0])

    print(f"\n  Batch size: {n}")

    # ---- Build trainer and load trained weights ----
    trainer = build_sumdiff_trainer(cfg)
    trainer(strain_batch[:1])  # build variable shapes
    trainer.load_weights(str(weights_path))

    # ---- Replicate the full loss pipeline inside a GradientTape ----
    with tf.GradientTape(persistent=True) as tape:
        # Forward pass
        y_pred = trainer(strain_batch, training=True)

        # ── Raw model outputs (post-tanh) ──
        z_phic_raw = y_pred["coa_phase"]
        z_psi_raw = y_pred["polarization_angle"]
        incl_raw = y_pred["inclination"]

        # Watch every intermediate tensor for gradient interrogation
        tape.watch(z_phic_raw)
        tape.watch(z_psi_raw)
        tape.watch(incl_raw)

        # ── Step 1: Normalise to unit vectors ──
        z_phic_norm = tf_normalize_unit(z_phic_raw)
        z_psi_norm = tf_normalize_unit(z_psi_raw)
        tape.watch(z_phic_norm)
        tape.watch(z_psi_norm)

        # ── Step 2: Complex multiply to build combo vectors ──
        combo_A_pred = tf_complex_mul(z_phic_norm, z_psi_norm)
        combo_B_pred = tf_complex_mul_conj(z_phic_norm, z_psi_norm)
        tape.watch(combo_A_pred)
        tape.watch(combo_B_pred)

        # True combo vectors (unit by construction; normalise for safety)
        combo_A_true = tf_complex_mul(
            tf_normalize_unit(targets["coa_phase"]),
            tf_normalize_unit(targets["polarization_angle"]),
        )
        combo_B_true = tf_complex_mul_conj(
            tf_normalize_unit(targets["coa_phase"]),
            tf_normalize_unit(targets["polarization_angle"]),
        )

        # ── Step 3: Isotropic circular loss per combo ──
        loss_A_per_sample = 1.0 - tf.reduce_sum(
            combo_A_true * combo_A_pred, axis=-1
        )
        loss_B_per_sample = 1.0 - tf.reduce_sum(
            combo_B_true * combo_B_pred, axis=-1
        )

        # ── Step 4: Curriculum weighting w(ι) ──
        cos_iota = targets["inclination"][:, 1]
        w_iota = tf_w_iota(cos_iota)

        if trainer._sign_dependent:
            pos_mask = tf.cast(cos_iota >= 0.0, tf.float32)
            neg_mask = 1.0 - pos_mask
            if trainer._well_constrained == "combo_A":
                w_A = pos_mask + neg_mask * w_iota
                w_B = neg_mask + pos_mask * w_iota
            else:  # combo_B well-constrained
                w_B = pos_mask + neg_mask * w_iota
                w_A = neg_mask + pos_mask * w_iota
        else:
            if trainer._well_constrained == "combo_A":
                w_A = tf.ones_like(w_iota)
                w_B = w_iota
            else:  # combo_B well-constrained
                w_B = tf.ones_like(w_iota)
                w_A = w_iota

        # ── Step 5: Weighted mean losses ──
        loss_A_mean = tf.reduce_mean(w_A * loss_A_per_sample)
        loss_B_mean = tf.reduce_mean(w_B * loss_B_per_sample)

        # ── Step 6: Log-var uncertainty weighting ──
        s_A = trainer.log_vars["combo_A"]
        s_B = trainer.log_vars["combo_B"]

        combo_total = (
            tf.exp(-s_A) * loss_A_mean + s_A
            + tf.exp(-s_B) * loss_B_mean + s_B
        )

        # ── All other heads (inclination, mchirp, snr, ...) ──
        other_total = trainer._other_heads_loss(targets, y_pred, None)

        total_loss = combo_total + other_total

    # ============================
    # Report gradient norms
    # ============================
    print(f"\n  {'=' * 70}")
    print(f"  {'Gradient chain — d(total_loss)/dtensor norms':^70}")
    print(f"  {'=' * 70}")
    print(f"  {'Tensor':<44s} {'Grad norm':<18s} {'Status':<10s}")
    print(f"  {'-' * 44} {'-' * 18} {'-' * 10}")

    def _report_gn(label, tensor):
        g = tape.gradient(total_loss, tensor)
        gn = float(tf.linalg.global_norm([g]) if g is not None else -1.0)
        if gn < 0:
            status = "NONE"
        elif gn < 1e-10:
            status = "ZERO"
        elif gn < 1e-6:
            status = "tiny"
        elif gn < 1e-3:
            status = "small"
        else:
            status = "OK"
        print(f"  {label:<44s} {gn:<18.8f} {status:<10s}")
        return gn

    _report_gn("dL/d(combo_A_pred)", combo_A_pred)
    _report_gn("dL/d(combo_B_pred)", combo_B_pred)
    _report_gn("dL/d(z_phic_norm) [after normalize_unit]", z_phic_norm)
    _report_gn("dL/d(z_psi_norm) [after normalize_unit]", z_psi_norm)
    _report_gn("dL/d(z_phic_raw) [model output]", z_phic_raw)
    _report_gn("dL/d(z_psi_raw) [model output]", z_psi_raw)
    _report_gn("dL/d(inclination_raw) [healthy baseline]", incl_raw)

    # Log-var gradients
    print(f"\n  {'Log-var gradients:':44s}")
    for lv_name in ("combo_A", "combo_B",):
        g = tape.gradient(total_loss, trainer.log_vars[lv_name])
        gn = float(tf.linalg.global_norm([g]) if g is not None else -1.0)
        print(f"  {'dL/d(log_var_' + lv_name + ')':<44s} {gn:<18.8f}")

    del tape  # persistent tape no longer needed

    # ============================
    # Per-sample forward-pass values
    # ============================
    print(f"\n  {'─' * 70}")
    print(f"  Per-sample forward-pass values (first 3 of {n} samples)")
    print(f"  {'─' * 70}")

    for i in range(min(3, n)):
        print(f"\n  Sample {i}:")

        la = float(loss_A_per_sample[i])
        lb = float(loss_B_per_sample[i])
        wa = float(w_A[i])
        wb = float(w_B[i])
        print(f"    1-cosDelta combo_A  loss={la:.6f}  weight={wa:.4f}  "
              f"weighted={la * wa:.6f}")
        print(f"    1-cosDelta combo_B  loss={lb:.6f}  weight={wb:.4f}  "
              f"weighted={lb * wb:.6f}")

        # Combo vectors
        for tag, vp, vt in [
            ("combo_A", combo_A_pred, combo_A_true),
            ("combo_B", combo_B_pred, combo_B_true),
        ]:
            p = vp[i].numpy()
            t = vt[i].numpy()
            print(f"    {tag}_pred  = [{p[0]:+.6f}, {p[1]:+.6f}]  "
                  f"angle={np.arctan2(p[0], p[1]):.4f}")
            print(f"    {tag}_true  = [{t[0]:+.6f}, {t[1]:+.6f}]  "
                  f"angle={np.arctan2(t[0], t[1]):.4f}")

        # Raw model outputs
        for tag, vec in [("z_phic", z_phic_raw), ("z_psi", z_psi_raw)]:
            v = vec[i].numpy()
            norm = np.linalg.norm(v)
            print(f"    {tag}_raw  = [{v[0]:+.6f}, {v[1]:+.6f}]  "
                  f"norm={norm:.4f}")

    # ============================
    # Interpretation guide
    # ============================
    print(f"\n  {'─' * 70}")
    print(f"  Interpretation guide")
    print(f"  {'─' * 70}")
    print(f"  1. combo_A_pred / combo_B_pred have healthy grad  → loss reaches combos")
    print(f"  2. Grad drops from combo_A_pred to z_phic_norm    → complex_mul path")
    print(f"  3. Grad drops from z_phic_norm to z_phic_raw       → normalize_unit")
    print(f"  4. Grad at z_phic_raw is tiny but incl_raw is OK   → φc/ψ-specific issue")
    print(f"  5. If even combo pred gradient is tiny              → loss fn / weighting")
    print(f"  6. If z_phic_raw grad is OK but weights don't move → layer-level (Check 5)")

    return True


# ======================================================================
# CHECK 7: Early training tanh saturation timing
# ======================================================================

def check_early_training_saturation():
    """Track tanh outputs at steps 0-10 to determine WHEN saturation happens.

    For poc_b config only (where Check 6 confirmed full saturation at epoch 80).
    Builds a FRESH trainer with random init (no loaded weights), runs 10 training
    steps, and inspects coa_phase/polarization_angle tanh outputs at each step.

    Key diagnostic:
      - |value| > 0.99 at step 0  → INIT VARIANCE problem (random weights already
        push pre-activation logits past tanh saturation).
      - Healthy at step 0, drifts to +/-1 over steps 1-5  → GRADIENT SPIKE from
        circular loss driving weights into saturation.
      - Never saturates in first 10 steps  → slow drift across epochs, different
        mechanism.
    """
    print("\n\n" + "=" * 80)
    print("CHECK 7: Early training tanh saturation timing")
    print("=" * 80)

    import tensorflow as tf
    from train_poc import build_sumdiff_trainer

    config_path = ROOT / "experiments/phic_psi_poc/config_poc.yaml"
    cfg = load_config(str(config_path))

    # ------------------------------------------------------------------
    # 1. Build fresh trainer with RANDOM init (don't load trained weights)
    # ------------------------------------------------------------------
    trainer = build_sumdiff_trainer(cfg)
    print(f"\n  Trainer built with mode={trainer._poc_mode}, random init (no weights loaded)")

    # ------------------------------------------------------------------
    # 2. Load a small fixed batch (8 samples) of validation data
    # ------------------------------------------------------------------
    strain, params = load_arrays(cfg["data"]["path"], "validation", max_samples=128)
    transforms = TargetTransforms(heads=trainer.head_names).fit(params)
    ds = make_dataset(strain, params, transforms, 8, shuffle=False)
    batch = next(iter(ds))
    strain_batch, targets = batch
    n = int(strain_batch.shape[0])
    print(f"  Batch size: {n}")

    # Build model variable shapes by running one sample through
    trainer(strain_batch[:1], training=False)

    # Reference to tanh-activated Dense layers for weight/gradient inspection
    coa_phase_layer = trainer.base.get_layer("coa_phase")
    psi_layer = trainer.base.get_layer("polarization_angle")

    # Track when saturation first appears
    first_saturation_step = None

    # ------------------------------------------------------------------
    # 3-5. Run 10 training steps, diagnose BEFORE each step
    # ------------------------------------------------------------------
    for step in range(11):  # steps 0..10 = 11 evaluations, 10 train_steps
        print(f"\n  {'─' * 70}")
        print(f"  Step {step} {'(before any training)' if step == 0 else ''}")
        print(f"  {'─' * 70}")

        # 4a. Forward pass BEFORE the gradient step (training=False)
        y_pred = trainer(strain_batch, training=False)

        z_phic_raw = y_pred["coa_phase"]             # (N, 2), post-tanh
        z_psi_raw = y_pred["polarization_angle"]     # (N, 2), post-tanh

        # 4b. Print first 3 samples in detail
        for i in range(min(3, n)):
            print(f"\n  Sample {i}:")
            for tag, vec in [("z_phic_raw  (coa_phase)", z_phic_raw),
                             ("z_psi_raw   (pol_angle)", z_psi_raw)]:
                v = vec[i].numpy()
                norm = float(np.linalg.norm(v))
                s_val, c_val = float(v[0]), float(v[1])
                flags = ""
                for comp_name, comp_val in [("sin", s_val), ("cos", c_val)]:
                    if abs(comp_val) > 0.99:
                        flags += f"  {comp_name}=+/-1"
                print(f"    {tag:<30s} = [{s_val:+.6f}, {c_val:+.6f}]  "
                      f"norm={norm:.4f}{flags}")

        # 4c. Check whether ANY component is saturated (|value| > 0.99)
        phic_saturated = bool(
            tf.reduce_any(tf.abs(z_phic_raw) > 0.99).numpy()
        )
        psi_saturated = bool(
            tf.reduce_any(tf.abs(z_psi_raw) > 0.99).numpy()
        )
        any_sat = phic_saturated or psi_saturated
        if any_sat and first_saturation_step is None:
            first_saturation_step = step

        print(f"\n  Any |value| > 0.99 across all {n} samples:")
        print(f"    coa_phase:          {'SATURATED' if phic_saturated else 'healthy'}")
        print(f"    polarization_angle: {'SATURATED' if psi_saturated else 'healthy'}")

        # For step 10 there is no train_step after the diagnostic pass
        if step >= 10:
            continue

        # ------------------------------------------------------------------
        # Gather gradient info for coa_phase kernel before applying updates
        # ------------------------------------------------------------------
        with tf.GradientTape() as tape:
            y_pred_train = trainer(strain_batch, training=True)
            loss = trainer._total_loss(targets, y_pred_train, None)

        kernel_var = coa_phase_layer.trainable_weights[0]   # weight matrix
        grad_kernel = tape.gradient(loss, kernel_var)
        grad_norm = float(
            tf.linalg.global_norm([grad_kernel])
            if grad_kernel is not None
            else -1.0
        )

        # Also grab bias gradient (2-element vector per periodic head)
        bias_var = (
            coa_phase_layer.trainable_weights[1]
            if len(coa_phase_layer.trainable_weights) > 1
            else None
        )
        grad_bias_norm = -1.0
        if bias_var is not None:
            gb = tape.gradient(loss, bias_var)
            if gb is not None:
                grad_bias_norm = float(tf.linalg.global_norm([gb]))
        del tape

        # ------------------------------------------------------------------
        # 5. Full training step (updates weights, updates metrics)
        # ------------------------------------------------------------------
        metrics = trainer.train_step(batch)

        # ------------------------------------------------------------------
        # After-step diagnostics
        # ------------------------------------------------------------------
        gn_str = f"{grad_norm:.8f}"
        if grad_norm < 0:
            gn_str += " (NO GRADIENT)"
        elif grad_norm < 1e-10:
            gn_str += " (ZERO)"
        elif grad_norm < 1e-6:
            gn_str += " (tiny)"

        print(f"\n  After gradient step:")
        print(f"    coa_phase kernel   |nabla| = {gn_str}")
        print(f"    coa_phase bias     |nabla| = {grad_bias_norm:.8f}")

        for mkey in ["circular_loss_combo_A", "circular_loss_combo_B"]:
            val = metrics.get(mkey)
            if val is not None:
                print(f"    {mkey:<30s} = {val:.6f}")

        for mkey in ["weight_combo_A", "weight_combo_B"]:
            val = metrics.get(mkey)
            if val is not None:
                print(f"    {mkey:<30s} = {val:.6f}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n  {'=' * 70}")
    print(f"  SUMMARY")
    print(f"  {'=' * 70}")
    if first_saturation_step is None:
        print(f"  No tanh saturation detected in any of the first 10 training steps.")
        print(f"  -> Saturation at epoch 80 is a slow drift, not init/early-training.")
        print(f"  -> Consider checking later epochs (epoch 1-5 range) for drift onset.")
    elif first_saturation_step == 0:
        print(f"  SATURATION PRESENT AT STEP 0 — before any training.")
        print(f"  -> Root cause: INIT VARIANCE — random weights already push")
        print(f"     pre-activation logits past tanh(|logit| ~3).")
        print(f"  -> Fix: switch periodic head activation to 'linear'.")
    else:
        n_steps = first_saturation_step
        print(f"  Tanh saturation first appears at STEP {first_saturation_step}.")
        print(f"  -> Root cause: EARLY-TRAINING GRADIENT SPIKE — the circular")
        print(f"     loss drives weight updates into saturation within the first "
              f"{n_steps} step(s).")
        print(f"  -> Fix: activation='linear' + gradient clipping.")

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
    check_tanh_saturation()
    check_gradient_chain()
    check_early_training_saturation()

    print("\n\nAll checks complete.")
    _teardown_logging()


if __name__ == "__main__":
    main()
