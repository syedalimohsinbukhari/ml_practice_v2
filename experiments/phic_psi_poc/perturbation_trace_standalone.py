#!/usr/bin/env python
"""Standalone multi-step prediction-perturbation trace (open item A.3).

Un-gated version of the trace embedded in diagnostic_lam005_retune.py /
diagnostic_lam010_retune.py. Those copies are gated behind the pre-registered
Step 0 std_ratio gate, which never passed for either primary target, so the
trace never ran. This script decouples it: it runs against the existing
Run 7 (λ=0.01) checkpoints directly, with NO training and NO gate.

Question it answers (diagnostic_log.md, Run 7 verification A.3): the
single-step perturbation probe found coa_phase rel_change ~89x larger than
mchirp's (1.61e-02 vs 1.80e-04). A one-step snapshot cannot distinguish
  (a) directional-but-slow learning   — steps accumulate,   net ≈ sum
  (b) oscillation around a constant   — steps cancel,       net ≈ sum/sqrt(N)
This trace takes N consecutive real gradient steps on one fixed batch and
reports, per head:
  - rel_change per step (magnitude of prediction movement)
  - mean cosine similarity between consecutive step vectors
  - net-vs-sum displacement ratio, against the 1/sqrt(N) random-walk
    reference
  - circular loss on a fixed validation probe before vs after the N steps

Usage (lab GPU machine only — do not run on the T530):
    python experiments/phic_psi_poc/perturbation_trace_standalone.py

Reads the latest run dir for each Run 7 config (which is the Run 7
checkpoint as of 2026-07-23 — the printed run-dir line should be checked
against diagnostic_log.md's Run 7 run IDs before trusting the output).
Writes log + markdown report to perturbation_trace_output/.
"""

from __future__ import annotations

import gc
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "phic_psi_poc"))

import tensorflow as tf  # noqa: E402

from gwml.data.loader import load_arrays, make_dataset  # noqa: E402
from gwml.data.transforms import TargetTransforms  # noqa: E402
from gwml.training.train import latest_run_dir, load_config  # noqa: E402

EXPERIMENTS_DIR = Path(__file__).resolve().parent
OUT_DIR = EXPERIMENTS_DIR / "perturbation_trace_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Run 7 (λ=0.01) configs — all four verified models.
CONFIGS = {
    "poc_a (baseline)": EXPERIMENTS_DIR / "config_baseline.yaml",
    "poc_b (poc)": EXPERIMENTS_DIR / "config_poc.yaml",
    "tcn": EXPERIMENTS_DIR / "config_tcn.yaml",
    "cnn_attention": EXPERIMENTS_DIR / "config_cnn_attention.yaml",
}

HEADS = ["mchirp", "merger_time", "snr", "sky_position",
         "coa_phase", "polarization_angle", "inclination"]
TRACK = ["coa_phase", "polarization_angle", "mchirp"]
N_STEPS = 25
PROBE_SIZE = 512     # fixed probe set, predictions tracked across steps
BATCH_SIZE = 128     # gradient-step batch, disjoint from the probe set
PRED_BATCH = 64      # probe forward-pass chunk size (memory bound)
SEED = 42


class _Tee:
    def __init__(self, file_path):
        self.stdout = sys.stdout
        self.file = open(file_path, "w", buffering=1)

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def predict_heads(trainer, strain_arr, heads):
    """Chunked forward pass — a single 512-sample call OOMs on the TCN."""
    outs = {h: [] for h in heads}
    for i in range(0, len(strain_arr), PRED_BATCH):
        y = trainer(strain_arr[i:i + PRED_BATCH], training=False)
        for h in heads:
            outs[h].append(y[h].numpy())
    return {h: np.concatenate(v, axis=0) for h, v in outs.items()}


def circular_loss(pred_vec, true_vec):
    """Mean 1 - cos(dtheta) from (sin, cos) 2-vectors, unit-normalizing both."""
    def unit(v):
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)
    return float(np.mean(1.0 - np.sum(unit(pred_vec) * unit(true_vec), axis=-1)))


def trace_model(trainer, strain_batch, targets, probe_strain, probe_targets):
    """N consecutive gradient steps on one batch; per-head movement stats."""
    per_step_vecs = {h: [] for h in TRACK}
    rel_changes = {h: [] for h in TRACK}

    pred_prev = predict_heads(trainer, probe_strain, TRACK)
    pred_start = dict(pred_prev)

    loss_before = {
        h: circular_loss(pred_start[h], probe_targets[h])
        for h in ("coa_phase", "polarization_angle")
    }

    for _ in range(N_STEPS):
        with tf.GradientTape() as tape:
            y_pred = trainer(strain_batch, training=True)
            loss = trainer._total_loss(targets, y_pred)
        grads = tape.gradient(loss, trainer.trainable_weights)
        gv = [(g, v) for g, v in zip(grads, trainer.trainable_weights) if g is not None]
        trainer.optimizer.apply_gradients(gv)

        pred_now = predict_heads(trainer, probe_strain, TRACK)
        for h in TRACK:
            delta = (pred_now[h] - pred_prev[h]).ravel()
            per_step_vecs[h].append(delta)
            rel_changes[h].append(
                float(np.linalg.norm(delta))
                / max(float(np.linalg.norm(pred_prev[h])), 1e-12)
            )
        pred_prev = pred_now

    loss_after = {
        h: circular_loss(pred_prev[h], probe_targets[h])
        for h in ("coa_phase", "polarization_angle")
    }

    results = {}
    for h in TRACK:
        vecs = per_step_vecs[h]
        cos_sims = []
        for a, b in zip(vecs[:-1], vecs[1:]):
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            cos_sims.append(float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0)
        net = float(np.linalg.norm(np.sum(vecs, axis=0)))
        total = float(sum(np.linalg.norm(v) for v in vecs))
        net_vs_sum = net / max(total, 1e-12)
        results[h] = {
            "rel_changes": rel_changes[h],
            "mean_cos_sim": float(np.mean(cos_sims)),
            "net_vs_sum": net_vs_sum,
        }
    return results, loss_before, loss_after


def classify(stats):
    """Directional vs oscillatory, on cos-sim and the random-walk reference."""
    rw = 1.0 / np.sqrt(N_STEPS)
    if stats["mean_cos_sim"] > 0.2 and stats["net_vs_sum"] > 2 * rw:
        return "DIRECTIONAL (coherent drift)"
    if stats["mean_cos_sim"] < 0.2 and stats["net_vs_sum"] < 2 * rw:
        return "OSCILLATORY (noise-like, random-walk scale)"
    return "AMBIGUOUS"


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tee = _Tee(OUT_DIR / f"perturbation_trace_{ts}.log")
    sys.stdout = tee

    print("=" * 90)
    print(f"STANDALONE MULTI-STEP PERTURBATION TRACE (A.3) — {N_STEPS} steps, "
          f"random-walk ref net/sum = {1.0 / np.sqrt(N_STEPS):.3f}")
    print("=" * 90)

    rows = []
    for label, config_path in CONFIGS.items():
        print(f"\n{'─' * 90}\n  MODEL: {label}\n{'─' * 90}")
        cfg = load_config(str(config_path))
        run_dir = latest_run_dir(cfg)
        weights = run_dir / "best.weights.h5"
        print(f"  run_dir: {run_dir}")
        if not weights.exists():
            print("  ✗ no best.weights.h5 — skipped")
            continue

        from train_poc import build_sumdiff_trainer

        keras_seed = getattr(tf.keras.utils, "set_random_seed", None)
        if keras_seed:
            keras_seed(SEED)

        # Only PROBE_SIZE + BATCH_SIZE samples are ever used — loading the
        # full 5000-sample validation split just wastes host/GPU memory.
        strain, params = load_arrays(cfg["data"]["path"], "validation",
                                     max_samples=PROBE_SIZE + BATCH_SIZE)
        transforms = TargetTransforms(heads=HEADS)
        transforms.fit(params)
        trainer = build_sumdiff_trainer(cfg)
        trainer(strain[:1])
        trainer.load_weights(str(weights))

        # Gradient-step batch is disjoint from the probe set, so the steps
        # never optimize on the samples whose predictions are being tracked.
        ds = make_dataset(strain[PROBE_SIZE:], params[PROBE_SIZE:],
                          transforms, BATCH_SIZE, shuffle=False)
        strain_batch, targets = next(iter(ds))

        probe_strain = strain[:PROBE_SIZE]
        probe_ds = make_dataset(strain[:PROBE_SIZE], params[:PROBE_SIZE],
                                transforms, PROBE_SIZE, shuffle=False)
        _, probe_targets_tf = next(iter(probe_ds))
        probe_targets = {k: np.asarray(v) for k, v in probe_targets_tf.items()}

        stats, loss_before, loss_after = trace_model(
            trainer, strain_batch, targets, probe_strain, probe_targets)

        for h in TRACK:
            s = stats[h]
            verdict = classify(s)
            rel = ", ".join(f"{v:.2e}" for v in s["rel_changes"][:5])
            print(f"    {h:>20s}: rel/step (first 5) = [{rel}] ...")
            print(f"    {'':>20s}  mean_cos_sim = {s['mean_cos_sim']:+.3f}   "
                  f"net/sum = {s['net_vs_sum']:.3f}   → {verdict}")
            rows.append((label, h, s["mean_cos_sim"], s["net_vs_sum"], verdict))
        for h in ("coa_phase", "polarization_angle"):
            print(f"    probe circular loss {h}: "
                  f"{loss_before[h]:.4f} → {loss_after[h]:.4f} "
                  f"(Δ = {loss_after[h] - loss_before[h]:+.4f})")

        # Free this model's graph, weights, and optimizer slots before the
        # next config — without this the four models accumulate on the GPU
        # and the third/fourth OOM.
        del trainer, ds, probe_ds, strain, params, strain_batch, targets
        tf.keras.backend.clear_session()
        gc.collect()

    md = OUT_DIR / f"perturbation_trace_{ts}.md"
    with open(md, "w") as f:
        f.write("# Standalone Multi-Step Perturbation Trace (A.3)\n\n")
        f.write(f"{N_STEPS} consecutive gradient steps on one fixed batch, "
                f"Run 7 (λ=0.01) checkpoints. Random-walk reference "
                f"net/sum = {1.0 / np.sqrt(N_STEPS):.3f}; directional drift "
                f"approaches 1.0.\n\n")
        f.write("| Model | Head | mean cos-sim | net/sum | Verdict |\n")
        f.write("|---|---|---|---|---|\n")
        for label, h, cos_sim, nvs, verdict in rows:
            f.write(f"| {label} | {h} | {cos_sim:+.3f} | {nvs:.3f} | {verdict} |\n")
        f.write("\nInterpretation: if coa_phase/pol_angle are OSCILLATORY while "
                "mchirp is DIRECTIONAL, the 89x single-step rel_change asymmetry "
                "(Run 7, A.3) was movement without learning — large steps that "
                "cancel — and the item closes as consistent with the null. A "
                "DIRECTIONAL verdict on a periodic head with decreasing probe "
                "circular loss would instead be the first mechanistic hint of "
                "slow learning and should be escalated, not filed.\n")
    print(f"\nReport: {md}")


if __name__ == "__main__":
    main()
