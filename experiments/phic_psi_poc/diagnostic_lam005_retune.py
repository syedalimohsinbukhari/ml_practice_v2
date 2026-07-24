#!/usr/bin/env python
"""λ=0.05 retune diagnostics — mechanical verdict per preregistration_lam_retune.md.

Three checks against the freshly trained λ=0.05 checkpoints:

  Step 0 (gate)       — std_ratio healthy in [0.5, 2.0], from history.csv.
  Step 1 (significance) — bootstrap shuffle-null test on val ang_MAE
                           (same procedure as bootstrap_ang_mae.py),
                           Bonferroni-corrected for 2 primary tests (p<0.025).
  Step 2 (effect size)  — Δang_MAE (null theory − observed) >= 0.10 rad.
  Step 3 (SNR check)    — improvement must be monotonic with SNR tercile and
                           the high-SNR tercile must independently clear the
                           0.10 rad floor (same tercile logic as
                           snr_stratification.py).

Plus a multi-step prediction-perturbation trace (5 consecutive gradient
steps on the same batch) to check whether coa_phase/pol_angle weight
movement is directionally coherent or oscillatory noise — the
"early-epochs Check 4" item from diagnostic_log.md's Run 7 retrain
interpretation guide.

The two primary pre-registered tests are:
  - tcn            / coa_phase           (still-declining std_ratio at λ=0.01)
  - poc_a baseline / polarization_angle  (stable-but-low std_ratio at λ=0.01)

Any other head/model combination produced here is exploratory only and does
not receive a pre-registered verdict.

Usage (on GPU machine, after training):
    python experiments/phic_psi_poc/diagnostic_lam005_retune.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
    log_path = out_dir / f"diagnostic_lam005_retune_{ts}.log"
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

import tensorflow as tf

from gwml.data.loader import load_arrays, make_dataset
from gwml.data.transforms import TargetTransforms
from gwml.training.train import latest_run_dir, load_config

LAMBDA_LABEL = "0.05"
EXPERIMENTS_DIR = Path(__file__).resolve().parent
OUT_DIR = EXPERIMENTS_DIR / "lam005_retune_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = {
    "poc_a (baseline)": EXPERIMENTS_DIR / "config_lam005_retune.yaml",
    "tcn": EXPERIMENTS_DIR / "config_lam005_retune_tcn.yaml",
}

# label -> (head, period, null_theory) for the pre-registered primary test
PRIMARY_TEST = {
    "tcn": ("coa_phase", 2 * np.pi, np.pi / 2),
    "poc_a (baseline)": ("polarization_angle", np.pi, np.pi / 4),
}

STD_RATIO_LOW, STD_RATIO_HIGH = 0.5, 2.0
STD_RATIO_FRAC_THRESHOLD = 0.10
STD_RATIO_TREND_THRESHOLD = 0.005
N_BOOTSTRAP = 10_000
N_SAMPLES = 5000
ALPHA_BONFERRONI = 0.025  # 0.05 / 2 primary tests
EFFECT_FLOOR_RAD = 0.10
N_TERCILES = 3
N_PERTURBATION_STEPS = 5


# ======================================================================
# Step 0 — std_ratio gate
# ======================================================================


def std_ratio_gate(run_dir: Path, head: str) -> dict:
    history_csv = run_dir / "history.csv"
    col = f"val_std_ratio_{head}"
    if not history_csv.exists():
        return {"passed": False, "reason": f"no history.csv at {run_dir}"}
    df = pd.read_csv(history_csv)
    if col not in df.columns:
        return {"passed": False, "reason": f"{col} not in history.csv"}
    late = df[col].iloc[-40:] if len(df) >= 40 else df[col]
    frac_unhealthy = float(((late < STD_RATIO_LOW) | (late > STD_RATIO_HIGH)).mean())
    trend = float(np.polyfit(np.arange(len(late)), late.values, 1)[0])
    passed = frac_unhealthy < STD_RATIO_FRAC_THRESHOLD and abs(trend) < STD_RATIO_TREND_THRESHOLD
    return {
        "passed": passed,
        "frac_unhealthy": frac_unhealthy,
        "trend": trend,
        "final_value": float(df[col].iloc[-1]),
    }


# ======================================================================
# Step 1 — bootstrap significance (same procedure as bootstrap_ang_mae.py)
# ======================================================================


def angular_mae(true_rad: np.ndarray, pred_rad: np.ndarray, period: float) -> float:
    res = pred_rad - true_rad
    res_wrapped = (res + period / 2) % period - period / 2
    return float(np.mean(np.abs(res_wrapped)))


def bootstrap_null(true_vals, pred_vals, period, n_bootstrap=N_BOOTSTRAP, seed=42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(true_vals)
    null_maes = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        shuffled_true = true_vals[rng.permutation(n)]
        null_maes[i] = angular_mae(shuffled_true, pred_vals, period)

    observed = angular_mae(true_vals, pred_vals, period)
    p_value = float(np.mean(null_maes <= observed))
    null_mean = float(np.mean(null_maes))
    null_std = float(np.std(null_maes))
    z_score = float((null_mean - observed) / null_std) if null_std > 0 else 0.0

    return {
        "observed": observed,
        "null_mean": null_mean,
        "null_std": null_std,
        "p_value": p_value,
        "z_score": z_score,
        "significant_bonferroni": p_value < ALPHA_BONFERRONI,
    }


# ======================================================================
# Step 3 — SNR stratification (same tercile logic as snr_stratification.py)
# ======================================================================


def snr_stratified_check(true_vals, pred_vals, snr_vals, period, null_theory) -> dict:
    snr_sorted_idx = np.argsort(snr_vals)
    tercile_size = len(snr_vals) // N_TERCILES
    tercile_maes = []
    for t in range(N_TERCILES):
        if t < N_TERCILES - 1:
            idx = snr_sorted_idx[t * tercile_size: (t + 1) * tercile_size]
        else:
            idx = snr_sorted_idx[t * tercile_size:]
        tercile_maes.append(angular_mae(true_vals[idx], pred_vals[idx], period))

    monotonic_improves = all(
        tercile_maes[i] >= tercile_maes[i + 1] for i in range(N_TERCILES - 1)
    )
    high_snr_delta = null_theory - tercile_maes[-1]

    return {
        "tercile_maes": tercile_maes,
        "monotonic_improves": monotonic_improves,
        "high_snr_delta": high_snr_delta,
    }


# ======================================================================
# Multi-step prediction perturbation trace
# ======================================================================


def multi_step_perturbation(trainer, strain_batch, targets, heads_to_track, n_steps=N_PERTURBATION_STEPS) -> dict:
    """N consecutive real gradient steps on the same batch; track per-head
    prediction deltas and whether consecutive deltas point the same
    direction (coherent drift) or cancel out (noise/oscillation)."""
    probe = strain_batch[:8]
    deltas_per_head = {h: [] for h in heads_to_track}
    step_vectors = {h: [] for h in heads_to_track}

    y_pred_prev = trainer(probe, training=False)

    for step in range(n_steps):
        with tf.GradientTape() as tape:
            y_pred = trainer(strain_batch, training=True)
            loss = trainer._total_loss(targets, y_pred)
        grads = tape.gradient(loss, trainer.trainable_weights)
        grads_and_vars = [(g, v) for g, v in zip(grads, trainer.trainable_weights) if g is not None]
        trainer.optimizer.apply_gradients(grads_and_vars)

        y_pred_now = trainer(probe, training=False)
        for h in heads_to_track:
            before = y_pred_prev[h].numpy()
            after = y_pred_now[h].numpy()
            delta_vec = (after - before).ravel()
            diff_norm = float(np.linalg.norm(delta_vec))
            before_norm = float(np.linalg.norm(before))
            rel_change = diff_norm / max(before_norm, 1e-12)
            deltas_per_head[h].append(rel_change)
            step_vectors[h].append(delta_vec)
        y_pred_prev = y_pred_now

    coherence = {}
    for h in heads_to_track:
        vecs = step_vectors[h]
        cos_sims = []
        for i in range(len(vecs) - 1):
            a, b = vecs[i], vecs[i + 1]
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            cos_sims.append(float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0)
        coherence[h] = {
            "rel_changes": deltas_per_head[h],
            "mean_cos_sim": float(np.mean(cos_sims)) if cos_sims else float("nan"),
        }
    return coherence


# ======================================================================
# Main
# ======================================================================


def main():
    ts = _setup_logging(OUT_DIR)
    print("=" * 100)
    print(f"λ={LAMBDA_LABEL} RETUNE DIAGNOSTICS — mechanical verdict per preregistration_lam_retune.md")
    print("=" * 100)

    verdict_rows = []

    for label, config_path in CONFIGS.items():
        print(f"\n{'─' * 100}")
        print(f"  MODEL: {label}")
        print(f"{'─' * 100}")

        cfg = load_config(str(config_path))
        run_dir = latest_run_dir(cfg)
        weights_path = run_dir / "best.weights.h5"
        if not weights_path.exists():
            print(f"  ✗ no best.weights.h5 at {run_dir}")
            continue

        primary_head, primary_period, primary_null = PRIMARY_TEST[label]

        # ---- Step 0: gate ----
        gate = std_ratio_gate(run_dir, primary_head)
        print(f"\n  Step 0 — std_ratio gate ({primary_head}):")
        print(f"    frac unhealthy (last 40 ep) = {gate.get('frac_unhealthy', float('nan')):.3f}")
        print(f"    late-epoch trend/ep         = {gate.get('trend', float('nan')):+.5f}")
        print(f"    GATE {'PASSED' if gate['passed'] else 'FAILED'}")

        if not gate["passed"]:
            print(f"\n  → UNINTERPRETABLE at λ={LAMBDA_LABEL}. Try λ=0.10 "
                  f"(run_lam010_retune.py) before drawing any conclusion.")
            verdict_rows.append((label, primary_head, "UNINTERPRETABLE", gate, None, None, None))
            continue

        # ---- Load data + model for Steps 1 & 3 ----
        from train_poc import build_sumdiff_trainer

        strain, params = load_arrays(cfg["data"]["path"], "validation", max_samples=N_SAMPLES)
        transforms = TargetTransforms.from_json(run_dir / "transforms.json")
        trainer = build_sumdiff_trainer(cfg)
        trainer(strain[:1])
        trainer.load_weights(str(weights_path))

        raw_pred = trainer.predict(strain, batch_size=256, verbose=0)
        pred = transforms.inverse(raw_pred)
        true = transforms.physical_targets(params)
        snr_vals = np.ravel(true["snr"])

        pred_vals = np.ravel(pred[primary_head])
        true_vals = np.ravel(true[primary_head])

        # ---- Step 1: bootstrap significance ----
        print(f"\n  Step 1 — bootstrap significance ({N_BOOTSTRAP} shuffles, "
              f"Bonferroni p<{ALPHA_BONFERRONI}):")
        boot = bootstrap_null(true_vals, pred_vals, primary_period)
        print(f"    observed = {boot['observed']:.4f} rad, "
              f"null_mean = {boot['null_mean']:.4f} rad, "
              f"z = {boot['z_score']:+.2f}σ, p = {boot['p_value']:.4f}")
        print(f"    significant (Bonferroni)? {boot['significant_bonferroni']}")

        # ---- Step 2: effect size floor ----
        effect = primary_null - boot["observed"]
        effect_passes = effect >= EFFECT_FLOOR_RAD
        print(f"\n  Step 2 — effect size (floor = {EFFECT_FLOOR_RAD} rad):")
        print(f"    Δang_MAE (null - observed) = {effect:+.4f} rad  "
              f"{'PASSES' if effect_passes else 'below floor'}")

        # ---- Step 3: SNR stratification ----
        print(f"\n  Step 3 — SNR stratification:")
        snr_check = snr_stratified_check(true_vals, pred_vals, snr_vals, primary_period, primary_null)
        print(f"    tercile ang_MAE (low→high) = {[f'{m:.4f}' for m in snr_check['tercile_maes']]}")
        print(f"    monotonic improvement with SNR? {snr_check['monotonic_improves']}")
        print(f"    high-SNR Δ from null = {snr_check['high_snr_delta']:+.4f} rad "
              f"({'clears' if snr_check['high_snr_delta'] >= EFFECT_FLOOR_RAD else 'below'} floor)")

        # ---- Final mechanical verdict (decision table in preregistration doc) ----
        if not boot["significant_bonferroni"]:
            verdict = "NULL"
        elif not effect_passes:
            verdict = "NULL (flag for replication — significant but below effect floor)"
        elif not (snr_check["monotonic_improves"] and snr_check["high_snr_delta"] >= EFFECT_FLOOR_RAD):
            verdict = "NULL (flag — population-bias signature, SNR-flat)"
        else:
            verdict = "COUNTER-EVIDENCE — escalate, do not fold into null tally"

        print(f"\n  >>> VERDICT: {verdict} <<<")
        verdict_rows.append((label, primary_head, verdict, gate, boot, effect, snr_check))

        # ---- Multi-step perturbation trace (both periodic heads + control) ----
        heads = ["mchirp", "merger_time", "snr", "sky_position",
                 "coa_phase", "polarization_angle", "inclination"]
        full_transforms = TargetTransforms(heads=heads)
        full_transforms.fit(params)
        ds = make_dataset(strain, params, full_transforms, 128, shuffle=False)
        batch = next(iter(ds))
        strain_batch, targets = batch

        print(f"\n  Multi-step perturbation trace ({N_PERTURBATION_STEPS} steps, same batch):")
        coherence = multi_step_perturbation(
            trainer, strain_batch, targets,
            heads_to_track=["coa_phase", "polarization_angle", "mchirp"],
        )
        for h, data in coherence.items():
            rel_str = ", ".join(f"{v:.2e}" for v in data["rel_changes"])
            direction = (
                "coherent (directional)" if data["mean_cos_sim"] > 0.2
                else "oscillatory (noise-like)" if data["mean_cos_sim"] < -0.2
                else "ambiguous"
            )
            print(f"    {h:>20s}: rel_change per step = [{rel_str}]  "
                  f"mean_cos_sim={data['mean_cos_sim']:+.3f} ({direction})")

    # ==================================================================
    # Markdown report
    # ==================================================================
    md_path = OUT_DIR / f"diagnostic_lam005_retune_{ts}.md"
    with open(md_path, "w") as f:
        f.write(f"# λ={LAMBDA_LABEL} Retune Diagnostics — Mechanical Verdict\n\n")
        f.write("Decision procedure fixed in advance — see "
                "[`preregistration_lam_retune.md`](preregistration_lam_retune.md).\n\n")
        f.write("| Model | Head | Gate | z | p | Δ (rad) | SNR-monotonic | Verdict |\n")
        f.write("|-------|------|------|---|---|---------|----------------|--------|\n")
        for label, head, verdict, gate, boot, effect, snr_check in verdict_rows:
            gate_str = "pass" if gate["passed"] else "FAIL"
            z_str = f"{boot['z_score']:+.2f}" if boot else "—"
            p_str = f"{boot['p_value']:.4f}" if boot else "—"
            eff_str = f"{effect:+.4f}" if effect is not None else "—"
            snr_str = str(snr_check["monotonic_improves"]) if snr_check else "—"
            f.write(f"| {label} | {head} | {gate_str} | {z_str} | {p_str} | "
                    f"{eff_str} | {snr_str} | {verdict} |\n")
        f.write("\nSee log for the multi-step perturbation trace and full "
                "per-tercile SNR breakdown.\n")

    print(f"\n\nMarkdown report: {md_path}")
    print("Done.")
    _teardown_logging()


if __name__ == "__main__":
    main()
