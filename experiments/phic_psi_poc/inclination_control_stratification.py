#!/usr/bin/env python
"""Inclination-stratified chirp-mass MAE/R^2 -- a scalar-control check on inclination_stratification.py.

Adversarial-review follow-up (v2): the phi_c/psi inclination stratification (§6.7 of the
thesis chapter) uses the known-uninformative inclination (iota) head as a same-model noise
floor for finite-band sampling. One objection is that this is circular if there is a
common-cause pathology upstream of any individual loss (shared trunk / batchnorm / head
design) that makes ALL angular heads fluctuate together, in which case iota's fluctuation
isn't independent noise.

This script does not resolve that directly, but it tests the *other* half of the concern:
does the face-on/mixed/edge-on banding scheme itself inject spurious metric-scale variance,
even into a head we know is well-learned and non-angular (chirp mass, R^2 ~ 0.93-0.96)?
If chirp-mass MAE/R^2 stays essentially flat across bands, that rules out "the banding
itself is noisy at these N" as an explanation for the iota head's band-to-band swings,
strengthening (not proving) the noise-floor argument in §6.7.

Read-only over existing checkpoints and validation data -- no retraining, no new injections.
Mirrors the structure of inclination_stratification.py / snr_stratification.py.

Usage (on GPU machine):
    python experiments/phic_psi_poc/inclination_control_stratification.py

Output:
    inclination_output/inclination_control_stratification_<timestamp>.{log,md}
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
    log_path = out_dir / f"inclination_control_stratification_{ts}.log"
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
    "poc_a (baseline)":  ROOT / "experiments/phic_psi_poc/config_baseline.yaml",
    "poc_b (PoC)":       ROOT / "experiments/phic_psi_poc/config_poc.yaml",
    "tcn":               ROOT / "experiments/phic_psi_poc/config_tcn.yaml",
    "cnn_attention":     ROOT / "experiments/phic_psi_poc/config_cnn_attention.yaml",
}

CONTROL_HEAD = "mchirp"

# Bands mirror inclination_stratification.py / thesis chapter Section 3.
FACE_ON_COS_IOTA = 0.9
EDGE_ON_COS_IOTA = 0.5
N_SAMPLES = 5000


def r_squared(true_vals: np.ndarray, pred_vals: np.ndarray) -> float:
    ss_res = np.sum((true_vals - pred_vals) ** 2)
    ss_tot = np.sum((true_vals - true_vals.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def band_masks(inclination_rad: np.ndarray) -> dict:
    cos_iota = np.cos(inclination_rad)
    return {
        "face-on":  np.abs(cos_iota) > FACE_ON_COS_IOTA,
        "mixed":    (np.abs(cos_iota) <= FACE_ON_COS_IOTA) & (np.abs(cos_iota) >= EDGE_ON_COS_IOTA),
        "edge-on":  np.abs(cos_iota) < EDGE_ON_COS_IOTA,
    }


def main() -> None:
    out_dir = Path("experiments/phic_psi_poc/inclination_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = _setup_logging(out_dir)

    print("=" * 100)
    print("INCLINATION-STRATIFIED CHIRP-MASS MAE/R^2 -- scalar-control check")
    print(f"Bands: face-on |cos(iota)| > {FACE_ON_COS_IOTA}, edge-on |cos(iota)| < {EDGE_ON_COS_IOTA}, "
          f"mixed in between. Validation samples = {N_SAMPLES}")
    print("Question: does the banding itself inject spurious MAE/R^2 variance into a head")
    print("we already know is well-learned and non-angular?")
    print("=" * 100)

    all_data = {}

    for label, config_path in CONFIGS.items():
        print(f"\n{'-' * 100}")
        print(f"  MODEL: {label}")
        print(f"{'-' * 100}")

        try:
            from train_poc import build_sumdiff_trainer

            cfg = load_config(str(config_path))
            run_dir = latest_run_dir(cfg)
            weights = run_dir / "best.weights.h5"

            if not weights.exists():
                print(f"  x no best.weights.h5 at {run_dir}")
                continue

            strain, params = load_arrays(
                cfg["data"]["path"], "validation", max_samples=N_SAMPLES
            )
            transforms = TargetTransforms.from_json(run_dir / "transforms.json")

            print(f"  Building model from {run_dir} ...")
            trainer = build_sumdiff_trainer(cfg)
            trainer(strain[:1])
            trainer.load_weights(str(weights))

            print(f"  Predicting on {len(strain)} samples ...")
            raw_pred = trainer.predict(strain, batch_size=256, verbose=0)
            pred = transforms.inverse(raw_pred)
            true = transforms.physical_targets(params)

            if "inclination" not in true or CONTROL_HEAD not in pred or CONTROL_HEAD not in true:
                print(f"  x inclination or {CONTROL_HEAD} not in predictions/labels")
                continue

            iota_vals = np.ravel(true["inclination"])
            masks = band_masks(iota_vals)
            pred_vals = np.ravel(pred[CONTROL_HEAD])
            true_vals = np.ravel(true[CONTROL_HEAD])

            full_mae = float(np.mean(np.abs(pred_vals - true_vals)))
            full_r2 = r_squared(true_vals, pred_vals)

            band_results = {}
            print(f"\n  {CONTROL_HEAD}:")
            print(f"  {'Band':<10} {'|cos iota| range':<22} {'N':>6} {'MAE':>10} {'R^2':>8}")
            print(f"  {'-'*10} {'-'*22} {'-'*6} {'-'*10} {'-'*8}")

            for band_name in ("face-on", "mixed", "edge-on"):
                mask = masks[band_name]
                n_b = int(mask.sum())
                if n_b == 0:
                    continue
                mae_b = float(np.mean(np.abs(pred_vals[mask] - true_vals[mask])))
                r2_b = r_squared(true_vals[mask], pred_vals[mask])
                band_results[band_name] = {"n": n_b, "mae": mae_b, "r2": r2_b}
                range_str = {
                    "face-on": f"[{FACE_ON_COS_IOTA:.1f}, 1.0]",
                    "mixed":   f"[{EDGE_ON_COS_IOTA:.1f}, {FACE_ON_COS_IOTA:.1f}]",
                    "edge-on": f"[0.0, {EDGE_ON_COS_IOTA:.1f})",
                }[band_name]
                print(f"  {band_name:<10} {range_str:<22} {n_b:>6} {mae_b:>10.4f} {r2_b:>8.4f}")

            print(f"  {'ALL':<10} {'-':<22} {len(pred_vals):>6} {full_mae:>10.4f} {full_r2:>8.4f}")

            mae_spread = max(b["mae"] for b in band_results.values()) - min(b["mae"] for b in band_results.values())
            r2_spread = max(b["r2"] for b in band_results.values()) - min(b["r2"] for b in band_results.values())
            print(f"  Band-to-band MAE spread: {mae_spread:.4f}   R^2 spread: {r2_spread:.4f}")

            all_data[label] = {
                "full_mae": full_mae, "full_r2": full_r2,
                "bands": band_results, "mae_spread": mae_spread, "r2_spread": r2_spread,
            }

        except Exception as e:
            print(f"  x ERROR: {e}")
            import traceback
            traceback.print_exc()

    # ==================================================================
    # Write markdown
    # ==================================================================
    md_path = out_dir / f"inclination_control_stratification_{ts}.md"
    with open(md_path, "w") as f:
        f.write("# Inclination-Stratified Chirp-Mass MAE/R^2 -- Scalar-Control Check\n\n")
        f.write(f"**Generated**: {ts}\n")
        f.write(f"**Bands**: face-on |cos(iota)| > {FACE_ON_COS_IOTA}, "
                f"edge-on |cos(iota)| < {EDGE_ON_COS_IOTA}, mixed in between "
                f"(matches thesis chapter Section 3 / inclination_stratification.py)\n")
        f.write(f"**Validation samples**: {N_SAMPLES}\n\n")

        f.write("## Key question\n\n")
        f.write("Section 6.7 of the thesis chapter uses the known-uninformative inclination "
                "head as a same-model noise floor for the phi_c/psi inclination stratification. "
                "This script checks the companion question: does the face-on/mixed/edge-on "
                "banding scheme itself inject spurious MAE/R^2 variance, even into a head "
                "(chirp mass) we already know is well-learned and non-angular? If the spread "
                "here is small relative to the phi_c/psi deviations in Table 6.6, banding-"
                "induced noise is not a plausible alternative explanation for those deviations.\n\n")

        f.write("| Model | ALL MAE | face-on MAE | mixed MAE | edge-on MAE | "
                "MAE spread | R^2 spread |\n")
        f.write("|-------|---------|-------------|-----------|-------------|"
                "------------|------------|\n")
        for label in CONFIGS:
            if label not in all_data:
                continue
            d = all_data[label]
            b_face = d["bands"].get("face-on")
            b_mix = d["bands"].get("mixed")
            b_edge = d["bands"].get("edge-on")
            face_mae = b_face["mae"] if b_face else float("nan")
            mix_mae = b_mix["mae"] if b_mix else float("nan")
            edge_mae = b_edge["mae"] if b_edge else float("nan")
            f.write(f"| {label} | {d['full_mae']:.4f} | {face_mae:.4f} | {mix_mae:.4f} | "
                    f"{edge_mae:.4f} | {d['mae_spread']:.4f} | {d['r2_spread']:.4f} |\n")
        f.write("\n")

        f.write("## Verdict\n\n")
        f.write("Small band-to-band MAE/R^2 spread here, relative to the phi_c/psi edge-on "
                "deviations in Table 6.6, supports treating those deviations as head-specific "
                "(iota-noise-floor-scale) rather than an artifact of slicing the validation "
                "set into unequal, differently-composed bands.\n")

    print(f"\n\nMarkdown report: {md_path}")
    print("Done.")
    _teardown_logging()


if __name__ == "__main__":
    main()
