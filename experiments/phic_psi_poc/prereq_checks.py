#!/usr/bin/env python
"""Prerequisite checks for the φc/ψ degeneracy PoC (Step 1 of the plan).

Implements Steps 1.1, 1.2, and 1.6 from
``phic_psi_implementation_plan_v4.md``.  Run this BEFORE writing any loss
code — the results determine:

  - Which combo (φc+2ψ or φc−2ψ) is well-constrained → ``well_constrained_combo``
  - The curriculum weight function w(ι)
  - Whether the training population has enough edge-on samples for a
    statistically meaningful test

Usage::

    python experiments/phic_psi_poc/prereq_checks.py

Flags::

    --data-path PATH    Path to HDF5 file (default: combined_repackaged.hdf)
    --skip-histogram    Skip the cos ι histogram (Step 1.6)
    --skip-sign-check   Skip the sign/combination check (Step 1.1)
    --skip-w-derivation Skip the w(ι) derivation (Step 1.2)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure repo root and src/ are on the path
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, str(Path(_REPO_ROOT) / "src"))


# ---------------------------------------------------------------------------
# Step 1.1 — Sign / combination check
# ---------------------------------------------------------------------------


def _check_sign_combination(
    n_sky_samples: int = 50, seed: int = 42
) -> dict:
    """Using the toy antenna-response model (Appendix A.5):

    1. Generate random (a, b) sky-position coefficients.
    2. At each sky position, sweep ψ across [0, π) and φc across [0, 2π).
    3. Compute (R, δ) from the detector_signal model.
    4. Confirm that at ι=0, (R, δ) is constant along φc+2ψ = const lines
       (harness correctness check).
    5. At non-zero ι, compute which linear combination (φc+2ψ or φc−2ψ)
       correlates more strongly with R, δ.

    Returns dict with keys: harness_passes, well_constrained_combo,
    correlation_ratio, sign_flip.
    """
    from experiments.phic_psi_poc.curriculum import (
        _random_sky_coefficients,
        detector_signal,
        project_to_R_delta,
    )

    rng = np.random.default_rng(seed)
    sky_coeffs = _random_sky_coefficients(n_sky_samples, rng)

    n_Phi = 100
    Phi = np.linspace(0.0, 2.0 * np.pi, n_Phi)

    # --- Harness correctness check: ι=0 must be fully degenerate ---
    # For each (a,b) pair, fix one const_line = φc + 2ψ.  Sweep ψ
    # (adjusting φc to stay on the line).  At ι=0, (R,δ) must be constant
    # within each group — the defining property of the degeneracy.
    iota_zero = 0.0
    max_R_std = 0.0
    max_delta_std = 0.0

    for a, b in sky_coeffs[:10]:  # 10 sky positions is enough
        phic_base = rng.uniform(0.0, 2.0 * np.pi)
        psi_base = rng.uniform(0.0, np.pi)
        const_line = phic_base + 2.0 * psi_base

        R_group = []
        delta_group = []
        for psi in np.linspace(0.0, np.pi, 20, endpoint=False):
            phic = (const_line - 2.0 * psi) % (2.0 * np.pi)
            sig = detector_signal(psi, phic, iota_zero, a, b, Phi)
            R, delta = project_to_R_delta(sig, Phi)
            R_group.append(R)
            delta_group.append(delta)

        R_group = np.array(R_group)
        delta_group = np.array(delta_group)
        max_R_std = max(max_R_std, float(np.std(R_group)))
        max_delta_std = max(max_delta_std, float(np.std(delta_group)))

    harness_passes = bool(max_R_std < 1e-10 and max_delta_std < 1e-10)

    if not harness_passes:
        print("  *** HARNESS CHECK FAILED: (R,δ) not constant at ι=0 along φc+2ψ=const")
        print(f"      max within-group std(R) = {max_R_std:.2e}, max within-group std(δ) = {max_delta_std:.2e}")
        print("      DO NOT PROCEED — the analytical harness has a bug.")
        return {
            "harness_passes": False,
            "well_constrained_combo": None,
            "correlation_ratio": None,
            "sign_flip": None,
        }
    print(f"  ✓ Harness check passed: (R,δ) constant at ι=0 along φc+2ψ=const "
          f"(max std(R)={max_R_std:.2e}, max std(δ)={max_delta_std:.2e})")

    # --- At non-zero ι, check which combo is better constrained ---
    # Strategy: for each sky position and moderate ι, sweep φc+2ψ (combo_A)
    # and φc−2ψ (combo_B), measuring how much (R, δ) varies with each.
    # The combo with *less* variation is the well-constrained one.

    iota_moderate = np.pi / 4.0  # 45°
    n_sweep = 30

    combo_A_corr_R = []
    combo_A_corr_delta = []
    combo_B_corr_R = []
    combo_B_corr_delta = []

    for a, b in sky_coeffs:
        # Sweep combo_A (φc+2ψ) while holding combo_B (φc−2ψ) fixed
        combo_B_fixed = rng.uniform(0.0, 4.0 * np.pi)
        combo_A_vals = np.linspace(0.0, 4.0 * np.pi, n_sweep)
        R_vals_A = []
        delta_vals_A = []
        for cA in combo_A_vals:
            # Solve: φc = (cA + cB_fixed)/2, 2ψ = (cA − cB_fixed)/2
            phic = (cA + combo_B_fixed) / 2.0
            two_psi = (cA - combo_B_fixed) / 2.0
            psi = two_psi / 2.0
            sig = detector_signal(psi, phic, iota_moderate, a, b, Phi)
            R, delta = project_to_R_delta(sig, Phi)
            R_vals_A.append(R)
            delta_vals_A.append(delta)

        R_vals_A = np.array(R_vals_A)
        delta_vals_A = np.array(delta_vals_A)
        # Correlation: does combo_A predict R/delta?
        combo_A_corr_R.append(np.abs(np.corrcoef(combo_A_vals, R_vals_A)[0, 1]))
        combo_A_corr_delta.append(np.abs(np.corrcoef(combo_A_vals, delta_vals_A)[0, 1]))

        # Sweep combo_B (φc−2ψ) while holding combo_A fixed
        combo_A_fixed = rng.uniform(0.0, 4.0 * np.pi)
        combo_B_vals = np.linspace(0.0, 4.0 * np.pi, n_sweep)
        R_vals_B = []
        delta_vals_B = []
        for cB in combo_B_vals:
            phic = (combo_A_fixed + cB) / 2.0
            two_psi = (combo_A_fixed - cB) / 2.0
            psi = two_psi / 2.0
            sig = detector_signal(psi, phic, iota_moderate, a, b, Phi)
            R, delta = project_to_R_delta(sig, Phi)
            R_vals_B.append(R)
            delta_vals_B.append(delta)

        R_vals_B = np.array(R_vals_B)
        delta_vals_B = np.array(delta_vals_B)
        combo_B_corr_R.append(np.abs(np.corrcoef(combo_B_vals, R_vals_B)[0, 1]))
        combo_B_corr_delta.append(np.abs(np.corrcoef(combo_B_vals, delta_vals_B)[0, 1]))

    mean_corr_A = np.mean(combo_A_corr_R + combo_A_corr_delta)
    mean_corr_B = np.mean(combo_B_corr_R + combo_B_corr_delta)

    # The well-constrained combo should have *higher* correlation with (R,δ)
    # (it drives the observables; the poorly-constrained one doesn't)
    if mean_corr_A > mean_corr_B:
        well_constrained = "combo_A"
        poorly_constrained = "combo_B"
        ratio = mean_corr_A / max(mean_corr_B, 1e-12)
    else:
        well_constrained = "combo_B"
        poorly_constrained = "combo_A"
        ratio = mean_corr_B / max(mean_corr_A, 1e-12)

    # Check sign flip at cos ι = 0
    iota_neg = np.pi * 3.0 / 4.0  # cos ι < 0
    # Quick check: at negative cos ι, does the well-constrained combo change?
    # For GW signals, the sign of cos ι flips h_cross → this can flip which
    # combo is well-constrained. Check with a single sky position.
    a_test, b_test = sky_coeffs[0]
    combo_A_corr_neg = []
    combo_B_corr_neg = []
    for _ in range(10):
        cB_fixed = rng.uniform(0.0, 4.0 * np.pi)
        cA_vals = np.linspace(0.0, 4.0 * np.pi, n_sweep)
        R_vals, delta_vals = [], []
        for cA in cA_vals:
            phic = (cA + cB_fixed) / 2.0
            two_psi = (cA - cB_fixed) / 2.0
            psi = two_psi / 2.0
            sig = detector_signal(psi, phic, iota_neg, a_test, b_test, Phi)
            R, delta = project_to_R_delta(sig, Phi)
            R_vals.append(R)
            delta_vals.append(delta)
        combo_A_corr_neg.append(np.abs(np.corrcoef(cA_vals, np.array(R_vals))[0, 1]))
        combo_B_corr_neg.append(np.abs(np.corrcoef(cA_vals, np.array(delta_vals))[0, 1]))

    sign_flip = bool(np.mean(combo_A_corr_neg) < np.mean(combo_B_corr_neg)) != (well_constrained == "combo_A")

    print(f"  ✓ Well-constrained combo: {well_constrained}")
    print(f"     (correlation ratio well/poorly = {ratio:.1f}x)")
    print(f"  ✓ Sign flip at cos ι < 0: {'YES' if sign_flip else 'NO'}")

    return {
        "harness_passes": harness_passes,
        "well_constrained_combo": well_constrained,
        "poorly_constrained_combo": poorly_constrained,
        "correlation_ratio": float(ratio),
        "sign_flip": sign_flip,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path", default="combined_repackaged.hdf",
        help="Path to HDF5 file (default: combined_repackaged.hdf)",
    )
    parser.add_argument(
        "--skip-histogram", action="store_true",
        help="Skip cos ι histogram (Step 1.6)",
    )
    parser.add_argument(
        "--skip-sign-check", action="store_true",
        help="Skip sign/combination check (Step 1.1)",
    )
    parser.add_argument(
        "--skip-w-derivation", action="store_true",
        help="Skip w(ι) derivation (Step 1.2)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("φc/ψ Degeneracy PoC — Prerequisite Checks")
    print("=" * 70)

    # ---- Step 1.1: Sign / combination check ----
    if not args.skip_sign_check:
        print("\n── Step 1.1: Sign / combination check ──")
        result_1_1 = _check_sign_combination()
        if not result_1_1["harness_passes"]:
            print("\n  *** ABORT: harness check failed. Fix the analytical model.")
            sys.exit(1)
    else:
        print("\n── Step 1.1: SKIPPED ──")

    # ---- Step 1.2: w(ι) derivation ----
    if not args.skip_w_derivation:
        print("\n── Step 1.2: Curriculum weight derivation ──")
        from experiments.phic_psi_poc.curriculum import derive_w_iota

        result_1_2 = derive_w_iota(n_sky_samples=50, n_iota_points=100)
        print(f"  {result_1_2['fit_info']}")
        print(f"  w(0) ≈ {result_1_2['w_values'][0]:.3f} (face-on)")
        print(f"  w(π/2) ≈ {result_1_2['w_values'][-1]:.3f} (edge-on)")
        print(f"  Fit coeffs: {result_1_2['fit_coeffs']}")
    else:
        print("\n── Step 1.2: SKIPPED ──")

    # ---- Step 1.6: cos ι histogram ----
    if not args.skip_histogram:
        print("\n── Step 1.6: cos ι histogram ──")
        from experiments.phic_psi_poc.curriculum import histogram_cos_iota

        result_1_6 = histogram_cos_iota(args.data_path)
        print(f"  Face-on fraction (|cos ι| > 0.9): {result_1_6['face_on_frac']:.1%}")
        print(f"  Edge-on fraction (|cos ι| < 0.5): {result_1_6['edge_on_frac']:.1%}")
        if result_1_6["warning"]:
            print(f"\n  {result_1_6['warning']}")
    else:
        print("\n── Step 1.6: SKIPPED ──")

    print("\n" + "=" * 70)
    print("Prerequisite checks complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
