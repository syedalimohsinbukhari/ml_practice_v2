#!/usr/bin/env python
"""Bootstrap CI on sky-position angular separation, certified 4-model set.

Closes the one caveat left open by sky_angular_separation_hist.py (2026-09-18):
that histogram settled bimodal-vs-uniform (uniform, confirmed) but was descriptive
only. This is the formal significance test `coa_phase`/`polarization_angle`/
`inclination` already got from `bootstrap_ang_mae.py`, applied to `sky_position`
via the same shuffle-null procedure -- same null hypothesis (shuffling true labels
destroys any true<->predicted correspondence), same one-sided test (is observed
mean angular separation significantly *below* the null distribution?), same
N_bootstrap/N_samples, just using great-circle separation on S^2
(`gwml.data.sky_transform.angular_separation`) instead of wrap-aware 1D ang_MAE,
since sky_position is a joint 2D/spherical quantity, not a single periodic angle.

Usage (on GPU machine -- loads best.weights.h5, calls trainer.predict; not run
locally per this repo's CLAUDE.md CPU-only rule):
    python experiments/phic_psi_poc-redo/bootstrap_sky_separation.py

Output:
    bootstrap_output/bootstrap_sky_separation_<timestamp>.{log,md}
"""
from __future__ import annotations

import sys
from datetime import datetime as _dt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "phic_psi_poc-redo"))


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

    def close(self):
        self.file.close()


_TEE = None


def _setup_logging(out_dir):
    global _TEE
    ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"bootstrap_sky_separation_{ts}.log"
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


from gwml.data.loader import load_arrays
from gwml.data.transforms import TargetTransforms
from gwml.data.sky_transform import radec_to_unit_vector, angular_separation
from gwml.training.train import latest_run_dir, load_config

CONFIGS = {
    "poc_a (baseline)": ROOT / "experiments/phic_psi_poc-redo/config_baseline.yaml",
    "poc_b (PoC)": ROOT / "experiments/phic_psi_poc-redo/config_poc.yaml",
    "tcn": ROOT / "experiments/phic_psi_poc-redo/config_tcn.yaml",
    "cnn_attention": ROOT / "experiments/phic_psi_poc-redo/config_cnn_attention.yaml",
}

N_BOOTSTRAP = 10_000
N_SAMPLES = 5000
ALPHA = 0.05


def bootstrap_null(true_vec: np.ndarray, pred_vec: np.ndarray,
                    n_bootstrap: int = N_BOOTSTRAP, seed: int = 42) -> dict:
    """Bootstrap null distribution by shuffling true sky-position vectors."""
    rng = np.random.default_rng(seed)
    n = len(true_vec)
    null_means = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        shuffled_true = true_vec[rng.permutation(n)]
        null_means[i] = float(np.mean(np.degrees(angular_separation(shuffled_true, pred_vec))))

    observed = float(np.mean(np.degrees(angular_separation(true_vec, pred_vec))))

    # One-sided: is observed significantly BETTER (lower separation) than null?
    p_value = float(np.mean(null_means <= observed))

    null_mean = float(np.mean(null_means))
    null_ci_lo = float(np.percentile(null_means, 100 * ALPHA / 2))
    null_ci_hi = float(np.percentile(null_means, 100 * (1 - ALPHA / 2)))
    null_std = float(np.std(null_means))
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
    out_dir = Path("experiments/phic_psi_poc-redo/bootstrap_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = _setup_logging(out_dir)

    print("=" * 100)
    print("BOOTSTRAP CI ON SKY-POSITION ANGULAR SEPARATION — redo certified 4-model set")
    print(f"N bootstrap = {N_BOOTSTRAP}, validation samples = {N_SAMPLES}")
    print("Null hypothesis: shuffling true sky positions does not change mean angular separation.")
    print("Test: one-sided — is observed separation significantly BELOW the null distribution?")
    print("=" * 100)

    all_results = {}

    for label, config_path in CONFIGS.items():
        print(f"\n{'-' * 100}\n  MODEL: {label}\n{'-' * 100}")
        try:
            from train_poc import build_sumdiff_trainer

            cfg = load_config(str(config_path))
            run_dir = latest_run_dir(cfg)
            weights = run_dir / "best.weights.h5"
            if not weights.exists():
                print(f"  ✗ no best.weights.h5 at {run_dir}")
                continue

            strain, params = load_arrays(cfg["data"]["path"], "validation", max_samples=N_SAMPLES)
            transforms = TargetTransforms.from_json(run_dir / "transforms.json")

            trainer = build_sumdiff_trainer(cfg)
            trainer(strain[:1])
            trainer.load_weights(str(weights))

            raw_pred = trainer.predict(strain, batch_size=256, verbose=0)
            pred = transforms.inverse(raw_pred)
            true = transforms.physical_targets(params)

            # sky_position is (dec, ra); radec_to_unit_vector wants (ra, dec).
            true_vec = radec_to_unit_vector(true["sky_position"][:, 1], true["sky_position"][:, 0])
            pred_vec = radec_to_unit_vector(pred["sky_position"][:, 1], pred["sky_position"][:, 0])

            print(f"  running {N_BOOTSTRAP} bootstrap iterations ...")
            result = bootstrap_null(true_vec, pred_vec)
            all_results[label] = result

            sig_marker = " ★ SIG" if result["significant"] else ""
            direction = "BETTER" if result["z_score"] > 0 else "WORSE"
            print(f"    observed mean separation = {result['observed']:.4f} deg")
            print(f"    null mean = {result['null_mean']:.4f} deg  "
                  f"null 95% CI = [{result['null_ci_lo']:.4f}, {result['null_ci_hi']:.4f}]")
            print(f"    z = {result['z_score']:+.2f}σ ({direction})  "
                  f"p = {result['p_value']:.4f}{sig_marker}")

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    header = f"{'Model':<22s} {'obs (deg)':>10s} {'null_mean':>10s} {'null_CI':>22s} {'z':>8s} {'p':>8s} {'verdict':>12s}"
    print(header)
    print("-" * len(header))
    for label in CONFIGS:
        if label not in all_results:
            print(f"{label:<22s} {'—':>10s}")
            continue
        r = all_results[label]
        ci_str = f"[{r['null_ci_lo']:.2f}, {r['null_ci_hi']:.2f}]"
        verdict = ("★ BETTER" if r["z_score"] > 0 else "★ WORSE") if r["significant"] else "not sig"
        print(f"{label:<22s} {r['observed']:>10.4f} {r['null_mean']:>10.4f} {ci_str:>22s} "
              f"{r['z_score']:>+7.2f} {r['p_value']:>8.4f} {verdict:>12s}")

    md_path = out_dir / f"bootstrap_sky_separation_{ts}.md"
    with open(md_path, "w") as f:
        f.write("# Bootstrap CI on Sky-Position Angular Separation (Redo)\n\n")
        f.write(f"**Generated**: {ts}\n")
        f.write(f"**N bootstrap**: {N_BOOTSTRAP}\n")
        f.write(f"**Validation samples**: {N_SAMPLES}\n")
        f.write("**Null hypothesis**: shuffling true sky positions does not change mean angular separation.\n")
        f.write("**Test**: one-sided — is observed separation significantly *below* the null distribution?\n\n")
        f.write("Companion to `sky_angular_separation_hist.py` (2026-09-18), which found the separation "
                "distribution matches a uniform-on-sphere null shape (no bimodal structure) but was "
                "descriptive only. This is the formal significance test for that same quantity.\n\n")
        f.write("## Summary\n\n")
        f.write("| Model | Observed (deg) | Null mean (deg) | Null 95% CI | z (σ) | p | Significant? |\n")
        f.write("|-------|-----------------|-------------------|-------------|-------|---|-------------|\n")
        for label in CONFIGS:
            if label not in all_results:
                continue
            r = all_results[label]
            ci = f"[{r['null_ci_lo']:.4f}, {r['null_ci_hi']:.4f}]"
            sig = "★ YES" if r["significant"] else "no"
            dirn = "better" if r["z_score"] > 0 else "worse"
            f.write(f"| {label} | {r['observed']:.4f} | {r['null_mean']:.4f} | {ci} | "
                    f"{r['z_score']:+.2f} | {r['p_value']:.4f} | {sig} ({dirn}) |\n")
        f.write("\n## Interpretation\n\n")
        f.write("- **p < 0.05, z > 0**: mean angular separation is significantly lower than random → evidence of localization.\n")
        f.write("- **p ≥ 0.05**: not distinguishable from random placement.\n")
        f.write("- **z < 0**: separation is *worse* than random (collapse/bias artifact).\n")

    print(f"\nMarkdown report: {md_path}")
    print("Done.")
    _teardown_logging()


if __name__ == "__main__":
    main()
