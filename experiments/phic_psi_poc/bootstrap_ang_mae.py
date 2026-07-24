#!/usr/bin/env python
"""Bootstrap CI on ang_MAE for periodic heads across all Run 7 models.

Null hypothesis: strain carries no angular information → shuffling true labels
should not change ang_MAE.  If a model is genuinely learning, its observed
ang_MAE should be significantly *below* the null distribution.

Usage (on GPU machine):
    python experiments/phic_psi_poc/bootstrap_ang_mae.py

Output:
    bootstrap_output/bootstrap_ang_mae_<timestamp>.md
    bootstrap_output/bootstrap_ang_mae_<timestamp>.log
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
    log_path = out_dir / f"bootstrap_ang_mae_{ts}.log"
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
    "poc_a (baseline)": ROOT / "experiments/phic_psi_poc/config_baseline.yaml",
    "poc_b (PoC)": ROOT / "experiments/phic_psi_poc/config_poc.yaml",
    "tcn": ROOT / "experiments/phic_psi_poc/config_tcn.yaml",
    "cnn_attention": ROOT / "experiments/phic_psi_poc/config_cnn_attention.yaml",
}

PERIODIC_HEADS = [
    ("coa_phase", 2 * np.pi, np.pi / 2),  # null = π/2 ≈ 1.571
    ("polarization_angle", np.pi, np.pi / 4),  # null = π/4 ≈ 0.785
    ("inclination", 2 * np.pi, np.pi / 2),  # null = π/2 ≈ 1.571
]

N_BOOTSTRAP = 10_000
N_SAMPLES = 5000  # full validation set
ALPHA = 0.05  # 95% CI


def angular_mae(true_rad: np.ndarray, pred_rad: np.ndarray, period: float) -> float:
    """Angular MAE with wrap-aware residual."""
    res = pred_rad - true_rad
    res_wrapped = (res + period / 2) % period - period / 2
    return float(np.mean(np.abs(res_wrapped)))


def bootstrap_null(
    true_vals: np.ndarray,
    pred_vals: np.ndarray,
    period: float,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = 42,
) -> dict:
    """Bootstrap null distribution by shuffling true labels.

    The null hypothesis is "no association between input and true angle."
    Shuffling true labels across samples destroys any such association while
    preserving the marginal distributions of both predictions and true values.
    """
    rng = np.random.default_rng(seed)
    n = len(true_vals)
    null_maes = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        shuffled_true = true_vals[rng.permutation(n)]
        null_maes[i] = angular_mae(shuffled_true, pred_vals, period)

    observed = angular_mae(true_vals, pred_vals, period)

    # One-sided test: is observed significantly BETTER (lower) than null?
    p_value = float(np.mean(null_maes <= observed))

    # Two-sided CI on the null mean
    null_mean = float(np.mean(null_maes))
    null_ci_lo = float(np.percentile(null_maes, 100 * ALPHA / 2))
    null_ci_hi = float(np.percentile(null_maes, 100 * (1 - ALPHA / 2)))

    # How many sigmas below null mean?
    null_std = float(np.std(null_maes))
    z_score = float((null_mean - observed) / null_std) if null_std > 0 else 0.0

    return {
        "observed": observed,
        "null_mean": null_mean,
        "null_std": null_std,
        "null_ci_lo": null_ci_lo,
        "null_ci_hi": null_ci_hi,
        "p_value": p_value,
        "z_score": z_score,
        "significant": p_value < ALPHA,
        "n_bootstrap": n_bootstrap,
    }


def main():
    out_dir = Path("experiments/phic_psi_poc/bootstrap_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = _setup_logging(out_dir)

    print("=" * 100)
    print("BOOTSTRAP CI ON ang_MAE — periodic heads, all Run 7 models")
    print(f"N bootstrap = {N_BOOTSTRAP}, validation samples = {N_SAMPLES}")
    print("=" * 100)

    all_results = {}

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

            # Load data
            strain, params = load_arrays(
                cfg["data"]["path"], "validation", max_samples=N_SAMPLES
            )
            transforms = TargetTransforms.from_json(run_dir / "transforms.json")

            # Build and load model
            print(f"  Building model from {run_dir} ...")
            trainer = build_sumdiff_trainer(cfg)
            trainer(strain[:1])
            trainer.load_weights(str(weights))

            # Predict
            print(f"  Predicting on {len(strain)} samples ...")
            raw_pred = trainer.predict(strain, batch_size=256, verbose=0)
            pred = transforms.inverse(raw_pred)
            true = transforms.physical_targets(params)

            model_results = {}

            for head_name, period, null_expectation in PERIODIC_HEADS:
                if head_name not in pred or head_name not in true:
                    print(f"  {head_name}: ✗ not in predictions")
                    continue

                pred_vals = np.ravel(pred[head_name])
                true_vals = np.ravel(true[head_name])

                print(f"  {head_name}: running {N_BOOTSTRAP} bootstrap iterations ...")
                result = bootstrap_null(true_vals, pred_vals, period)

                model_results[head_name] = result

                # Print inline
                sig_marker = " ★ SIG" if result["significant"] else ""
                direction = "BETTER" if result["z_score"] > 0 else "WORSE"
                print(f"    observed  = {result['observed']:.4f} rad")
                print(f"    null mean = {result['null_mean']:.4f} rad  "
                      f"(theory: {null_expectation:.4f})")
                print(f"    null 95% CI = [{result['null_ci_lo']:.4f}, "
                      f"{result['null_ci_hi']:.4f}]")
                print(f"    z = {result['z_score']:+.2f}σ ({direction})  "
                      f"p = {result['p_value']:.4f}{sig_marker}")

            all_results[label] = model_results

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()

    # ==================================================================
    # Summary table
    # ==================================================================
    print("\n\n" + "=" * 100)
    print("SUMMARY: Bootstrap significance test on ang_MAE")
    print("=" * 100)

    for head_name, period, null_theory in PERIODIC_HEADS:
        print(f"\n{'─' * 100}")
        print(f"  {head_name}  (period={period:.4f}, null_theory={null_theory:.4f} rad)")
        print(f"{'─' * 100}")
        header = (f"{'Model':<22s} {'obs':>8s} {'null_mean':>10s} {'null_CI':>20s} "
                  f"{'z':>8s} {'p':>8s} {'verdict':>12s}")
        print(header)
        print("-" * len(header))

        for label in CONFIGS:
            if label not in all_results or head_name not in all_results[label]:
                print(f"{label:<22s} {'—':>8s}")
                continue
            r = all_results[label][head_name]
            null_ci_str = f"[{r['null_ci_lo']:.4f}, {r['null_ci_hi']:.4f}]"
            if r["significant"]:
                verdict = "★ BETTER" if r["z_score"] > 0 else "★ WORSE"
            else:
                verdict = "not sig" if r["z_score"] > 0 else "not sig (worse)"
            print(f"{label:<22s} {r['observed']:>8.4f} {r['null_mean']:>10.4f} "
                  f"{null_ci_str:>20s} {r['z_score']:>+7.2f} {r['p_value']:>8.4f} "
                  f"{verdict:>12s}")

    # ==================================================================
    # Write markdown report
    # ==================================================================
    md_path = out_dir / f"bootstrap_ang_mae_{ts}.md"
    with open(md_path, "w") as f:
        f.write(f"# Bootstrap CI on ang_MAE — Periodic Heads\n\n")
        f.write(f"**Generated**: {ts}\n")
        f.write(f"**N bootstrap**: {N_BOOTSTRAP}\n")
        f.write(f"**Validation samples**: {N_SAMPLES}\n")
        f.write(
            f"**Null hypothesis**: Shuffling true labels (destroying strain→angle association) does not change ang_MAE.\n")
        f.write(f"**Test**: One-sided — is observed ang_MAE significantly *below* the null distribution?\n\n")

        f.write("## Summary\n\n")
        f.write("| Model | Head | Observed | Null mean | Null 95% CI | z (σ) | p | Significant? |\n")
        f.write("|-------|------|----------|-----------|-------------|-------|---|-------------|\n")
        for label in CONFIGS:
            for head_name, period, null_theory in PERIODIC_HEADS:
                if label not in all_results or head_name not in all_results[label]:
                    continue
                r = all_results[label][head_name]
                null_ci = f"[{r['null_ci_lo']:.4f}, {r['null_ci_hi']:.4f}]"
                sig = "★ YES" if r["significant"] else "no"
                dirn = "better" if r["z_score"] > 0 else "worse"
                f.write(f"| {label} | {head_name} | {r['observed']:.4f} | "
                        f"{r['null_mean']:.4f} | {null_ci} | "
                        f"{r['z_score']:+.2f} | {r['p_value']:.4f} | "
                        f"{sig} ({dirn}) |\n")

        f.write("\n## Interpretation\n\n")
        f.write("- **p < 0.05, z > 0**: ang_MAE is significantly better than random → evidence of learning.\n")
        f.write("- **p ≥ 0.05**: ang_MAE is not distinguishable from random → no evidence of learning.\n")
        f.write(
            "- **z < 0**: ang_MAE is *worse* than random — model does worse than guessing (usually a collapse or bias artifact).\n\n")

        f.write("## Per-model details\n\n")
        for label in CONFIGS:
            if label not in all_results:
                continue
            f.write(f"### {label}\n\n")
            for head_name, period, null_theory in PERIODIC_HEADS:
                if head_name not in all_results[label]:
                    continue
                r = all_results[label][head_name]
                f.write(f"**{head_name}**: observed={r['observed']:.4f}, "
                        f"null_mean={r['null_mean']:.4f}±{r['null_std']:.4f}, "
                        f"z={r['z_score']:+.2f}σ, p={r['p_value']:.4f}\n\n")

    print(f"\n\nMarkdown report: {md_path}")
    print("Done.")
    _teardown_logging()


if __name__ == "__main__":
    main()
