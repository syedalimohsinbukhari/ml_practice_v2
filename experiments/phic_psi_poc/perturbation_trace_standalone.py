#!/usr/bin/env python
"""Standalone multi-step prediction-perturbation trace (open item A.3).

Un-gated version of the trace embedded in diagnostic_lam005_retune.py /
diagnostic_lam010_retune.py. Runs against existing checkpoints (or a fresh
initialization) directly, with NO full training and NO Step 0 gate.

Question (diagnostic_log.md, Run 7 verification A.3): the single-step
perturbation probe found coa_phase rel_change ~89x larger than mchirp's.
A one-step snapshot cannot distinguish
  (a) directional-but-slow learning   — steps accumulate,   net ≈ sum
  (b) oscillation around a constant   — steps cancel,       net ≈ sum/sqrt(N)

Two stages (2026-07-23 review addendum — the first run's mchirp positive
control read AMBIGUOUS at the converged epoch-79 checkpoints, so the
instrument needs calibration against a head KNOWN to be learning):

  final  — trace the trained Run 7 checkpoints (original mode).
  early  — fresh initialization + WARMUP_STEPS (~1 epoch) of training-split
           gradient steps, then the same trace. A working instrument must
           read mchirp as DIRECTIONAL here; per-epoch checkpoints were not
           saved during the runs, so this substitutes for an epoch-5..10
           checkpoint.

Per-sample paired statistics (also added in the review addendum): probe
circular-loss change is reported as mean ± paired SE with a t statistic —
the marginal probe SE is NOT the right comparator for a before/after
change on the same fixed probe samples.

Usage (lab GPU machine only — do not run on the T530):
    python experiments/phic_psi_poc/perturbation_trace_standalone.py          # final
    python experiments/phic_psi_poc/perturbation_trace_standalone.py early    # calibration

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
WARMUP_STEPS = 200   # 'early' stage: ~1 epoch of training-split steps
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


def circular_loss_per_sample(pred_vec, true_vec):
    """Per-sample 1 - cos(dtheta) from (sin, cos) 2-vectors."""
    def unit(v):
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)
    return 1.0 - np.sum(unit(pred_vec) * unit(true_vec), axis=-1)


def mse_per_sample(pred, true):
    return np.mean((pred - np.asarray(true)) ** 2, axis=-1)


def paired_stats(before, after):
    """Paired before/after statistics on the same fixed probe samples."""
    d = np.asarray(after, dtype=np.float64) - np.asarray(before, dtype=np.float64)
    n = len(d)
    mean = float(np.mean(d))
    se = float(np.std(d, ddof=1) / np.sqrt(n))
    return {
        "mean": mean,
        "se": se,
        "t": mean / se if se > 0 else float("nan"),
        "frac_improved": float(np.mean(d < 0)),
    }


def one_train_step(trainer, strain_batch, targets):
    with tf.GradientTape() as tape:
        y_pred = trainer(strain_batch, training=True)
        loss = trainer._total_loss(targets, y_pred)
    grads = tape.gradient(loss, trainer.trainable_weights)
    gv = [(g, v) for g, v in zip(grads, trainer.trainable_weights) if g is not None]
    trainer.optimizer.apply_gradients(gv)
    return float(loss)


def trace_model(trainer, strain_batch, targets, probe_strain, probe_targets):
    """N consecutive gradient steps on one batch; per-head movement stats."""
    per_step_vecs = {h: [] for h in TRACK}
    rel_changes = {h: [] for h in TRACK}

    pred_prev = predict_heads(trainer, probe_strain, TRACK)
    pred_start = dict(pred_prev)

    # Batch total loss per step: if it falls while the probe loss rises, that
    # is the direct signature of single-batch overfitting (the mechanism
    # behind the significantly positive final-stage mchirp probe deltas).
    batch_losses = []
    for _ in range(N_STEPS):
        batch_losses.append(one_train_step(trainer, strain_batch, targets))
        pred_now = predict_heads(trainer, probe_strain, TRACK)
        for h in TRACK:
            delta = (pred_now[h] - pred_prev[h]).ravel()
            per_step_vecs[h].append(delta)
            rel_changes[h].append(
                float(np.linalg.norm(delta))
                / max(float(np.linalg.norm(pred_prev[h])), 1e-12)
            )
        pred_prev = pred_now

    # Per-sample paired probe statistics, per head (review addendum).
    probe = {}
    for h in ("coa_phase", "polarization_angle"):
        before = circular_loss_per_sample(pred_start[h], probe_targets[h])
        after = circular_loss_per_sample(pred_prev[h], probe_targets[h])
        probe[h] = paired_stats(before, after)
        probe[h]["metric"] = "circ"
    before = mse_per_sample(pred_start["mchirp"], probe_targets["mchirp"])
    after = mse_per_sample(pred_prev["mchirp"], probe_targets["mchirp"])
    probe["mchirp"] = paired_stats(before, after)
    probe["mchirp"]["metric"] = "mse"

    print(f"    batch total loss: {batch_losses[0]:.4f} → {batch_losses[-1]:.4f} "
          f"(Δ = {batch_losses[-1] - batch_losses[0]:+.4f} over {N_STEPS} steps)")

    results = {}
    for h in TRACK:
        vecs = per_step_vecs[h]
        cos_sims = []
        for a, b in zip(vecs[:-1], vecs[1:]):
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            cos_sims.append(float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0)
        net = float(np.linalg.norm(np.sum(vecs, axis=0)))
        total = float(sum(np.linalg.norm(v) for v in vecs))
        results[h] = {
            "rel_changes": rel_changes[h],
            "mean_cos_sim": float(np.mean(cos_sims)),
            "net_vs_sum": net / max(total, 1e-12),
        }
    return results, probe


def classify(stats):
    """Directional vs oscillatory, on cos-sim and the random-walk reference."""
    rw = 1.0 / np.sqrt(N_STEPS)
    if stats["mean_cos_sim"] > 0.2 and stats["net_vs_sum"] > 2 * rw:
        return "DIRECTIONAL (coherent drift)"
    if stats["mean_cos_sim"] < 0.2 and stats["net_vs_sum"] < 2 * rw:
        return "OSCILLATORY (noise-like, random-walk scale)"
    return "AMBIGUOUS"


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "final"
    assert stage in ("final", "early"), f"stage must be final|early, got {stage}"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tee = _Tee(OUT_DIR / f"perturbation_trace_{stage}_{ts}.log")
    sys.stdout = tee

    print("=" * 90)
    print(f"STANDALONE MULTI-STEP PERTURBATION TRACE (A.3) — stage={stage}, "
          f"{N_STEPS} steps, random-walk ref net/sum = {1.0 / np.sqrt(N_STEPS):.3f}")
    if stage == "early":
        print(f"CALIBRATION MODE: fresh init + {WARMUP_STEPS} warmup steps on the "
              f"training split (per-epoch checkpoints were not saved during the "
              f"runs). A working instrument must read mchirp as DIRECTIONAL here.")
    print("=" * 90)

    rows = []
    for label, config_path in CONFIGS.items():
        print(f"\n{'─' * 90}\n  MODEL: {label} [{stage}]\n{'─' * 90}")
        cfg = load_config(str(config_path))

        from train_poc import build_sumdiff_trainer

        keras_seed = getattr(tf.keras.utils, "set_random_seed", None)
        if keras_seed:
            keras_seed(SEED)

        # Only PROBE_SIZE + BATCH_SIZE validation samples are ever used for
        # the trace itself.
        strain, params = load_arrays(cfg["data"]["path"], "validation",
                                     max_samples=PROBE_SIZE + BATCH_SIZE)
        transforms = TargetTransforms(heads=HEADS)
        transforms.fit(params)
        trainer = build_sumdiff_trainer(cfg)
        trainer(strain[:1])

        if stage == "final":
            run_dir = latest_run_dir(cfg)
            weights = run_dir / "best.weights.h5"
            print(f"  run_dir: {run_dir}")
            if not weights.exists():
                print("  ✗ no best.weights.h5 — skipped")
                continue
            trainer.load_weights(str(weights))
        else:
            print(f"  fresh init (seed {SEED}); warming up "
                  f"{WARMUP_STEPS} steps on the training split ...")
            tr_strain, tr_params = load_arrays(cfg["data"]["path"], "training")
            tr_ds = make_dataset(tr_strain, tr_params, transforms,
                                 BATCH_SIZE, shuffle=True)
            step = 0
            while step < WARMUP_STEPS:
                for tb_strain, tb_targets in tr_ds:
                    one_train_step(trainer, tb_strain, tb_targets)
                    step += 1
                    if step >= WARMUP_STEPS:
                        break
            del tr_strain, tr_params, tr_ds
            gc.collect()
            print(f"  warmup done ({step} steps ≈ {step * BATCH_SIZE / 25000:.2f} "
                  f"epochs).")

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

        stats, probe = trace_model(
            trainer, strain_batch, targets, probe_strain, probe_targets)

        for h in TRACK:
            s = stats[h]
            verdict = classify(s)
            p = probe[h]
            rel = ", ".join(f"{v:.2e}" for v in s["rel_changes"][:5])
            print(f"    {h:>20s}: rel/step (first 5) = [{rel}] ...")
            print(f"    {'':>20s}  mean_cos_sim = {s['mean_cos_sim']:+.3f}   "
                  f"net/sum = {s['net_vs_sum']:.3f}   → {verdict}")
            print(f"    {'':>20s}  probe Δ{p['metric']} = {p['mean']:+.4f} "
                  f"± {p['se']:.4f} (paired SE)   t = {p['t']:+.2f}   "
                  f"frac improved = {p['frac_improved']:.2f}")
            rows.append((label, h, s["mean_cos_sim"], s["net_vs_sum"],
                         p["mean"], p["se"], p["t"], verdict))

        # Free this model's graph, weights, and optimizer slots before the
        # next config — without this the four models accumulate on the GPU
        # and the third/fourth OOM.
        del trainer, ds, probe_ds, strain, params, strain_batch, targets
        tf.keras.backend.clear_session()
        gc.collect()

    md = OUT_DIR / f"perturbation_trace_{stage}_{ts}.md"
    with open(md, "w") as f:
        f.write(f"# Standalone Multi-Step Perturbation Trace (A.3) — stage: {stage}\n\n")
        if stage == "early":
            f.write(f"Calibration run: fresh init + {WARMUP_STEPS} warmup steps "
                    f"(~1 epoch) on the training split, then the standard trace. "
                    f"A working instrument must read mchirp as DIRECTIONAL here; "
                    f"if it does, the AMBIGUOUS mchirp verdicts at the converged "
                    f"checkpoints are a convergence effect and the final-stage "
                    f"table is interpretable. If mchirp stays AMBIGUOUS even "
                    f"here, the trace methodology itself is unsound and no "
                    f"verdict from it should be used.\n\n")
        f.write(f"{N_STEPS} consecutive gradient steps on one fixed batch. "
                f"Random-walk reference net/sum = {1.0 / np.sqrt(N_STEPS):.3f}; "
                f"directional drift approaches 1.0. Probe Δ is a per-sample "
                f"paired statistic on a fixed disjoint {PROBE_SIZE}-sample "
                f"probe (circular loss for periodic heads, transformed-target "
                f"MSE for mchirp).\n\n")
        f.write("| Model | Head | mean cos-sim | net/sum | probe Δ ± SE | t | Verdict |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for label, h, cos_sim, nvs, dm, dse, t, verdict in rows:
            f.write(f"| {label} | {h} | {cos_sim:+.3f} | {nvs:.3f} | "
                    f"{dm:+.4f} ± {dse:.4f} | {t:+.2f} | {verdict} |\n")
        f.write("\nEscalation rule (unchanged): a DIRECTIONAL verdict on a "
                "periodic head with a decreasing probe circular loss that is "
                "significant under the paired statistic is a mechanistic hint "
                "of slow learning and should be escalated, not filed.\n")
    print(f"\nReport: {md}")


if __name__ == "__main__":
    main()
