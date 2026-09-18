#!/usr/bin/env python
"""SNR-stratified angular separation for sky_position, certified 4-model set.

The decisive follow-up to bootstrap_sky_separation.py's 2026-09-18 finding: all
4 models show statistically significant better-than-random sky localization,
and cnn_attention's effect (Delta=12.58 deg, z=+23.23 sigma) clears this
investigation's own 0.10 rad materiality floor -- the first head/model
combination in the whole investigation (original or redo) to do so. A direct
check already ruled out the simplest population-bias explanation (always
guessing the mean direction only scores ~89.4 deg, not 77.5 deg -- see NOTES.md,
2026-09-18), leaving two live candidates: (a) genuine, if partial, localization
via cross-detector (H1/L1) amplitude-ratio information -- sky position drives
each detector's antenna-pattern response (F_plus/F_cross in curriculum.py /
formulae_reference.md A.5), independent of the timing-triangulation this
2-detector setup can't really do; or (b) a subtler shortcut than pure
population-mean collapse, consistent with cnn_attention's already-documented
tendency to memorize coa_phase/polarization_angle without generalizing.

This is the discriminator: real physical recovery (path a) should improve
monotonically with SNR -- louder, cleaner signals carry more amplitude-ratio
information. A memorization-style shortcut (path b) has no reason to track
signal quality at all. Same tercile logic as snr_stratification.py (Section E
of the original's verification plan), same monotonic-improvement and
floor-clearing-at-high-SNR requirements as preregistration_lam_retune.md's
Step 3 -- adapted here for great-circle separation on S^2 instead of 1D
wrap-aware ang_MAE, since sky_position is a joint spherical quantity.

Usage (on GPU machine -- loads best.weights.h5, calls trainer.predict; not run
locally per this repo's CLAUDE.md CPU-only rule):
    python experiments/phic_psi_poc-redo/snr_stratification_sky_position.py

Output:
    snr_output/snr_stratification_sky_position_<timestamp>.{log,md}
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
    log_path = out_dir / f"snr_stratification_sky_position_{ts}.log"
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

N_TERCILES = 3
N_SAMPLES = 5000
NULL_EXPECTATION_DEG = 90.0  # theoretical mean separation, two independent uniform points on S^2
FLOOR_DEG = np.degrees(0.10)  # this investigation's reused 0.10 rad materiality floor


def main():
    out_dir = Path("experiments/phic_psi_poc-redo/snr_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = _setup_logging(out_dir)

    print("=" * 100)
    print("SNR-STRATIFIED ANGULAR SEPARATION — sky_position, redo certified 4-model set")
    print(f"N terciles = {N_TERCILES}, validation samples = {N_SAMPLES}")
    print(f"Null expectation (theory) = {NULL_EXPECTATION_DEG:.2f} deg, "
          f"materiality floor = {FLOOR_DEG:.2f} deg (0.10 rad)")
    print("=" * 100)

    all_data = {}

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

            if "snr" not in true:
                print("  ✗ SNR not in true labels — cannot stratify")
                continue

            snr_vals = np.ravel(true["snr"])
            true_vec = radec_to_unit_vector(true["sky_position"][:, 1], true["sky_position"][:, 0])
            pred_vec = radec_to_unit_vector(pred["sky_position"][:, 1], pred["sky_position"][:, 0])
            sep_deg = np.degrees(angular_separation(true_vec, pred_vec))

            snr_sorted_idx = np.argsort(snr_vals)
            tercile_size = len(snr_vals) // N_TERCILES
            tercile_results = {}
            for t in range(N_TERCILES):
                idx = (snr_sorted_idx[t * tercile_size:(t + 1) * tercile_size] if t < N_TERCILES - 1
                       else snr_sorted_idx[t * tercile_size:])
                sep_t = sep_deg[idx]
                snr_lo, snr_hi = snr_vals[idx].min(), snr_vals[idx].max()
                mean_sep = float(sep_t.mean())
                vs_null = NULL_EXPECTATION_DEG - mean_sep
                tercile_results[t] = {
                    "n": len(idx), "mean_sep": mean_sep, "snr_lo": snr_lo, "snr_hi": snr_hi,
                    "vs_null": vs_null,
                }

            full_mean = float(sep_deg.mean())
            tercile_means = [tercile_results[t]["mean_sep"] for t in range(N_TERCILES)]
            improves = tercile_means[-1] < tercile_means[0]
            mono_improves = all(tercile_means[i] >= tercile_means[i + 1] for i in range(N_TERCILES - 1))
            high_delta = NULL_EXPECTATION_DEG - tercile_means[-1]
            high_clears_floor = high_delta >= FLOOR_DEG

            print(f"\n  {'Tercile':<10} {'SNR range':<20} {'N':>6} {'mean sep (deg)':>15} {'vs null':>10}")
            print(f"  {'-'*10} {'-'*20} {'-'*6} {'-'*15} {'-'*10}")
            for t in range(N_TERCILES):
                r = tercile_results[t]
                name = "Low" if t == 0 else "Mid" if t == 1 else "High"
                print(f"  {name:<10} [{r['snr_lo']:.1f}, {r['snr_hi']:.1f}]{'':<7} "
                      f"{r['n']:>6} {r['mean_sep']:>15.4f} {r['vs_null']:>+9.4f}")
            print(f"  {'ALL':<10} {'—':<20} {len(sep_deg):>6} {full_mean:>15.4f} "
                  f"{NULL_EXPECTATION_DEG - full_mean:>+9.4f}")
            print(f"\n  Monotonic improvement with SNR? {'YES' if mono_improves else 'partial' if improves else 'NO'}")
            print(f"  High-SNR Δ from null = {high_delta:+.4f} deg  "
                  f"({'CLEARS' if high_clears_floor else 'does not clear'} the {FLOOR_DEG:.2f} deg floor)")

            verdict = ("REAL-RECOVERY-CONSISTENT" if (mono_improves and high_clears_floor)
                       else "POPULATION-BIAS-SIGNATURE" if high_clears_floor
                       else "NULL")
            print(f"  Verdict (mirrors preregistration_lam_retune.md's Step 3 logic): {verdict}")

            all_data[label] = {
                "full_mean": full_mean, "terciles": tercile_results,
                "mono_improves": mono_improves, "improves": improves,
                "high_delta": high_delta, "high_clears_floor": high_clears_floor,
                "verdict": verdict,
            }

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n\n" + "=" * 100)
    print("SUMMARY: SNR-stratified sky_position angular separation, all models")
    print("=" * 100)
    header = (f"{'Model':<20s} {'ALL':>8s} {'Low':>8s} {'Mid':>8s} {'High':>8s} "
              f"{'Monotonic?':>12s} {'High Δ':>10s} {'Verdict':>26s}")
    print(header)
    print("-" * len(header))
    for label in CONFIGS:
        if label not in all_data:
            print(f"{label:<20s} {'—':>8s}")
            continue
        d = all_data[label]
        row = (f"{label:<20s} {d['full_mean']:>8.2f} "
               f"{d['terciles'][0]['mean_sep']:>8.2f} {d['terciles'][1]['mean_sep']:>8.2f} "
               f"{d['terciles'][2]['mean_sep']:>8.2f} "
               f"{'YES' if d['mono_improves'] else 'partial' if d['improves'] else 'NO':>12s} "
               f"{d['high_delta']:>+9.2f} {d['verdict']:>26s}")
        print(row)

    md_path = out_dir / f"snr_stratification_sky_position_{ts}.md"
    with open(md_path, "w") as f:
        f.write("# SNR-Stratified Angular Separation — sky_position (Redo)\n\n")
        f.write(f"**Generated**: {ts}\n")
        f.write(f"**N terciles**: {N_TERCILES}\n")
        f.write(f"**Validation samples**: {N_SAMPLES}\n")
        f.write(f"**Null expectation (theory)**: {NULL_EXPECTATION_DEG:.2f} deg "
                f"(mean separation, two independent uniform points on S²)\n")
        f.write(f"**Materiality floor**: {FLOOR_DEG:.2f} deg (0.10 rad, reused from "
                "`preregistration_lam_retune.md`'s Step 2)\n\n")
        f.write("## Key question\n\n")
        f.write("Follow-up to `bootstrap_sky_separation.py` (2026-09-18): `cnn_attention` showed a "
                "floor-clearing, highly significant sky-localization effect (Δ=12.58°, z=+23.23σ), "
                "and a direct check ruled out the simplest population-bias explanation (always guessing "
                "the mean direction only scores ~89.4°, not 77.5°). This test is the discriminator: real "
                "per-sample recovery (e.g. via cross-detector H1/L1 amplitude-ratio information, which "
                "this 2-detector setup can partially provide independent of timing triangulation) should "
                "improve monotonically with SNR; a memorization-style shortcut has no reason to track "
                "signal quality.\n\n")
        f.write("| Model | ALL (deg) | Low SNR | Mid SNR | High SNR | Monotonic? | High-SNR Δ from null | "
                "Clears 0.10 rad floor? | Verdict |\n")
        f.write("|-------|-----------|---------|---------|----------|-------------|----------------------|"
                "-------------------------|--------|\n")
        for label in CONFIGS:
            if label not in all_data:
                continue
            d = all_data[label]
            f.write(f"| {label} | {d['full_mean']:.2f} | {d['terciles'][0]['mean_sep']:.2f} | "
                    f"{d['terciles'][1]['mean_sep']:.2f} | {d['terciles'][2]['mean_sep']:.2f} | "
                    f"{'YES' if d['mono_improves'] else 'partial' if d['improves'] else 'NO'} | "
                    f"{d['high_delta']:+.2f} | {'YES' if d['high_clears_floor'] else 'no'} | "
                    f"{d['verdict']} |\n")
        f.write("\n## Verdict logic (mirrors `preregistration_lam_retune.md`'s Step 3)\n\n")
        f.write("- **REAL-RECOVERY-CONSISTENT**: monotonic improvement with SNR *and* the high-SNR "
                "tercile's own Δ independently clears the 0.10 rad floor — the signature of a genuine "
                "per-sample strain→direction mapping, easiest to recover in the loudest events.\n")
        f.write("- **POPULATION-BIAS-SIGNATURE**: high-SNR Δ clears the floor but improvement is flat "
                "or non-monotonic across terciles — a population-level effect that doesn't require "
                "using the signal's own information content, consistent with a memorization-driven "
                "shortcut rather than real recovery.\n")
        f.write("- **NULL**: high-SNR tercile doesn't clear the floor either — same category as "
                "`poc_a`/`poc_b`/`tcn`'s pooled bootstrap result.\n")

    print(f"\n\nMarkdown report: {md_path}")
    print("Done.")
    _teardown_logging()


if __name__ == "__main__":
    main()
