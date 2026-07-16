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
    n_sky_samples: int = 200, seed: int = 42
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
    # CRITICAL (h/t rev2/rev3 reviews):
    #   1. Compute correlations SEPARATELY for cos ι > 0 and cos ι < 0.
    #   2. SWEEP ι from near-face-on (ι≈0.1) to near-edge-on (ι≈1.4) to
    #      confirm the ratio TRENDS correctly — should grow as ι→0 and
    #      shrink as ι→π/2 per the underlying physics.
    #   3. Bootstrap the ratio at the two reference ι values to check
    #      whether 1.2× is statistically distinguishable from 1.0×.
    n_sweep = 50  # deeper grid: 50 combo values per sky position

    def _correlations_at_iota(iota_val, rng_local):
        """Compute mean combo_A and combo_B correlations at a single ι."""
        cA_corr_R, cA_corr_delta = [], []
        cB_corr_R, cB_corr_delta = [], []
        for a, b in sky_coeffs:
            # Sweep combo_A while holding combo_B fixed
            cB_fixed = rng_local.uniform(0.0, 4.0 * np.pi)
            cA_vals = np.linspace(0.0, 4.0 * np.pi, n_sweep)
            R_vals, delta_vals = [], []
            for cA in cA_vals:
                phic = (cA + cB_fixed) / 2.0
                psi = (cA - cB_fixed) / 4.0
                sig = detector_signal(psi, phic, iota_val, a, b, Phi)
                R, delta = project_to_R_delta(sig, Phi)
                R_vals.append(R); delta_vals.append(delta)
            R_vals = np.array(R_vals); delta_vals = np.array(delta_vals)
            cA_corr_R.append(np.abs(np.corrcoef(cA_vals, R_vals)[0, 1]))
            cA_corr_delta.append(np.abs(np.corrcoef(cA_vals, delta_vals)[0, 1]))

            # Sweep combo_B while holding combo_A fixed
            cA_fixed = rng_local.uniform(0.0, 4.0 * np.pi)
            cB_vals = np.linspace(0.0, 4.0 * np.pi, n_sweep)
            R_vals, delta_vals = [], []
            for cB in cB_vals:
                phic = (cA_fixed + cB) / 2.0
                psi = (cA_fixed - cB) / 4.0
                sig = detector_signal(psi, phic, iota_val, a, b, Phi)
                R, delta = project_to_R_delta(sig, Phi)
                R_vals.append(R); delta_vals.append(delta)
            R_vals = np.array(R_vals); delta_vals = np.array(delta_vals)
            cB_corr_R.append(np.abs(np.corrcoef(cB_vals, R_vals)[0, 1]))
            cB_corr_delta.append(np.abs(np.corrcoef(cB_vals, delta_vals)[0, 1]))

        mean_A = np.mean(cA_corr_R) + np.mean(cA_corr_delta)
        mean_B = np.mean(cB_corr_R) + np.mean(cB_corr_delta)
        raw_A = np.array(cA_corr_R) + np.array(cA_corr_delta)
        raw_B = np.array(cB_corr_R) + np.array(cB_corr_delta)
        return float(mean_A), float(mean_B), raw_A, raw_B

    # --- ι SWEEP: confirm the ratio trends correctly with inclination ---
    # Physics prediction: the gap between well- and poorly-constrained combos
    # should WIDEN as ι→0 (more degenerate) and CLOSE as ι→π/2 (less degenerate).
    # A ratio that stays flat at ~1.2× across all ι would be suspicious.
    # Deep grid sweep: 25 points per sign regime (50 total) from near-face-on
    # to near-edge-on.  Avoid the exact poles where finite differences break.
    sweep_iotas_pos = np.linspace(0.05, np.pi / 2 - 0.02, 25)       # cos ι > 0
    sweep_iotas_neg = np.linspace(np.pi / 2 + 0.02, np.pi - 0.05, 25)  # cos ι < 0

    sweep_results = []
    for iota_s in sweep_iotas_pos:
        mA, mB, _, _ = _correlations_at_iota(iota_s, rng)
        winner_s = "combo_A" if mA > mB else "combo_B"
        ratio_s = max(mA, mB) / max(min(mA, mB), 1e-12)
        sweep_results.append({
            "iota": float(iota_s), "cos_iota_sign": "+",
            "winner": winner_s, "ratio": float(ratio_s),
            "corr_A": float(mA), "corr_B": float(mB),
        })
    for iota_s in sweep_iotas_neg:
        mA, mB, _, _ = _correlations_at_iota(iota_s, rng)
        winner_s = "combo_A" if mA > mB else "combo_B"
        ratio_s = max(mA, mB) / max(min(mA, mB), 1e-12)
        sweep_results.append({
            "iota": float(iota_s), "cos_iota_sign": "−",
            "winner": winner_s, "ratio": float(ratio_s),
            "corr_A": float(mA), "corr_B": float(mB),
        })

    # --- Sign-split reference points (π/4 and 3π/4) with bootstrap ---
    results_by_sign = {}
    for sign_label, iota_val in [("cos_ι_>_0", np.pi / 4.0),
                                  ("cos_ι_<_0", 3.0 * np.pi / 4.0)]:
        mA, mB, raw_A, raw_B = _correlations_at_iota(iota_val, rng)

        winner = "combo_A" if mA > mB else "combo_B"
        loser = "combo_B" if winner == "combo_A" else "combo_A"
        ratio = max(mA, mB) / max(min(mA, mB), 1e-12)

        # Bootstrap 95% CI on the ratio
        n_sky = len(raw_A)
        n_boot = 5000
        boot_ratios = np.zeros(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n_sky, size=n_sky)
            bA = np.mean(raw_A[idx])
            bB = np.mean(raw_B[idx])
            if winner == "combo_A":
                boot_ratios[b] = bA / max(bB, 1e-12)
            else:
                boot_ratios[b] = bB / max(bA, 1e-12)
        ci_lo = float(np.percentile(boot_ratios, 2.5))
        ci_hi = float(np.percentile(boot_ratios, 97.5))

        results_by_sign[sign_label] = {
            "well_constrained": winner,
            "poorly_constrained": loser,
            "ratio": float(ratio),
            "ratio_95ci": (ci_lo, ci_hi),
            "mean_corr_A": float(mA),
            "mean_corr_B": float(mB),
        }

    # Determine whether the label flips across cos ι = 0
    sign_flip = (
        results_by_sign["cos_ι_>_0"]["well_constrained"]
        != results_by_sign["cos_ι_<_0"]["well_constrained"]
    )
    well_constrained = results_by_sign["cos_ι_>_0"]["well_constrained"]
    poorly_constrained = results_by_sign["cos_ι_>_0"]["poorly_constrained"]

    r_pos = results_by_sign["cos_ι_>_0"]
    r_neg = results_by_sign["cos_ι_<_0"]

    print(f"  cos ι > 0 (ι=π/4):  well-constrained = {r_pos['well_constrained']} "
          f"(ratio = {r_pos['ratio']:.2f}x, 95% CI = [{r_pos['ratio_95ci'][0]:.2f}, {r_pos['ratio_95ci'][1]:.2f}])")
    print(f"  cos ι < 0 (ι=3π/4): well-constrained = {r_neg['well_constrained']} "
          f"(ratio = {r_neg['ratio']:.2f}x, 95% CI = [{r_neg['ratio_95ci'][0]:.2f}, {r_neg['ratio_95ci'][1]:.2f}])")
    print(f"  Sign flip at cos ι = 0: {'YES — label depends on sign(cos ι)' if sign_flip else 'NO'}")

    # --- ι SWEEP summary ---
    print(f"\n  ι sweep ({len(sweep_results)} points, cos ι > 0 and cos ι < 0):")
    print(f"  {'ι [rad]':>8s}  {'sign':>5s}  {'winner':>8s}  {'ratio':>7s}")
    print(f"  {'-'*8}  {'-'*5}  {'-'*8}  {'-'*7}")
    for s in sweep_results:
        print(f"  {s['iota']:8.3f}  {s['cos_iota_sign']:>5s}  {s['winner']:>8s}  {s['ratio']:6.2f}x")

    # Check: does ratio increase as ι→0 (toward face-on)?
    ratios_pos = [s["ratio"] for s in sweep_results if s["cos_iota_sign"] == "+"]
    ratios_neg = [s["ratio"] for s in sweep_results if s["cos_iota_sign"] == "−"]
    # For cos ι > 0: ι small (face-on) → ratio should be HIGHER
    trend_ok_pos = ratios_pos[0] > ratios_pos[-1] if len(ratios_pos) > 1 else True
    # For cos ι < 0: ι large (near π, face-on) → ratio should be HIGHER
    trend_ok_neg = ratios_neg[-1] > ratios_neg[0] if len(ratios_neg) > 1 else True
    print(f"\n  Trend check (ratio ↑ as ι→0 or ι→π, i.e. toward face-on):")
    print(f"    cos ι > 0: ratio at ι≈0.1 = {ratios_pos[0]:.2f}x, "
          f"at ι≈π/2 = {ratios_pos[-1]:.2f}x → "
          f"{'✓ widens toward face-on' if trend_ok_pos else '✗ FLAT or wrong direction'}")
    print(f"    cos ι < 0: ratio at ι≈π/2 = {ratios_neg[0]:.2f}x, "
          f"at ι≈π = {ratios_neg[-1]:.2f}x → "
          f"{'✓ widens toward face-on' if trend_ok_neg else '✗ FLAT or wrong direction'}")

    # Significance check: does the 95% CI exclude 1.0?
    pos_significant = r_pos["ratio_95ci"][0] > 1.0
    neg_significant = r_neg["ratio_95ci"][0] > 1.0
    print(f"\n  Significance (95% CI excludes 1.0 = statistically distinguishable):")
    print(f"    cos ι > 0: {'YES' if pos_significant else 'NO — ratio not significantly different from 1.0'}")
    print(f"    cos ι < 0: {'YES' if neg_significant else 'NO — ratio not significantly different from 1.0'}")

    if sign_flip:
        print(f"  → Recommendation: condition well_constrained_combo on sign(cos ι)")
        print(f"    at training time, not a single fixed label.")
    print(f"  → Primary (cos ι > 0): well_constrained = {well_constrained}")

    # --- Export sweep results to CSV ---
    sweep_csv_path = "experiments/phic_psi_poc/sweep_1_1_ratio_vs_iota.csv"
    _export_sweep_csv(sweep_results, results_by_sign, sweep_csv_path)
    print(f"\n  Sweep results exported to: {sweep_csv_path}")

    # --- Plot ratio vs ι ---
    try:
        _plot_ratio_vs_iota(sweep_results, results_by_sign)
    except Exception as e:
        print(f"  (plot skipped: {e})")

    return {
        "harness_passes": harness_passes,
        "well_constrained_combo": well_constrained,
        "poorly_constrained_combo": poorly_constrained,
        "sign_flip": sign_flip,
        "results_by_sign": results_by_sign,
        "sweep_results": sweep_results,
    }


def _export_sweep_csv(sweep_results, results_by_sign, path):
    """Export ι sweep results as CSV."""
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iota_rad", "cos_iota_sign", "winner", "ratio",
                     "corr_A", "corr_B"])
        for s in sweep_results:
            w.writerow([s["iota"], s["cos_iota_sign"], s["winner"],
                        s["ratio"], s["corr_A"], s["corr_B"]])
        # Append bootstrap summary rows
        for sign_label, r in results_by_sign.items():
            ci = r.get("ratio_95ci", (None, None))
            w.writerow([f"# bootstrap {sign_label}", "", r["well_constrained"],
                        f"{r['ratio']:.3f}", f"CI=[{ci[0]:.3f},{ci[1]:.3f}]", ""])


def _plot_ratio_vs_iota(sweep_results, results_by_sign):
    """Plot combo correlation ratio vs inclination angle ι."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iotas_pos = [s["iota"] for s in sweep_results if s["cos_iota_sign"] == "+"]
    ratios_pos = [s["ratio"] for s in sweep_results if s["cos_iota_sign"] == "+"]
    winners_pos = [s["winner"] for s in sweep_results if s["cos_iota_sign"] == "+"]

    iotas_neg = [s["iota"] for s in sweep_results if s["cos_iota_sign"] == "−"]
    ratios_neg = [s["ratio"] for s in sweep_results if s["cos_iota_sign"] == "−"]
    winners_neg = [s["winner"] for s in sweep_results if s["cos_iota_sign"] == "−"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: cos ι > 0
    colors_pos = ["C0" if w == "combo_A" else "C1" for w in winners_pos]
    ax1.scatter(iotas_pos, ratios_pos, c=colors_pos, s=60, zorder=5)
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="no preference (1.0)")
    ax1.set_xlabel("ι [rad]")
    ax1.set_ylabel("correlation ratio (well / poorly)")
    ax1.set_title("cos ι > 0  (face-on ← ι=0,  ι=π/2 → edge-on)")
    ax1.legend(loc="upper right")
    ax1.set_ylim(0.9, None)
    # Annotate the reference point
    r45 = results_by_sign["cos_ι_>_0"]
    ax1.axvline(x=np.pi/4, color="gray", linestyle=":", alpha=0.3)
    ax1.annotate(f"ι=π/4: {r45['ratio']:.2f}x\n({r45['well_constrained']})",
                 xy=(np.pi/4, r45["ratio"]), xytext=(np.pi/4+0.2, r45["ratio"]+0.02),
                 arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9)

    # Right: cos ι < 0
    colors_neg = ["C0" if w == "combo_A" else "C1" for w in winners_neg]
    ax2.scatter(iotas_neg, ratios_neg, c=colors_neg, s=60, zorder=5)
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="no preference (1.0)")
    ax2.set_xlabel("ι [rad]")
    ax2.set_ylabel("correlation ratio (well / poorly)")
    ax2.set_title("cos ι < 0  (edge-on ← ι=π/2,  ι=π → face-on)")
    ax2.legend(loc="upper left")
    ax2.set_ylim(0.9, None)
    r135 = results_by_sign["cos_ι_<_0"]
    ax2.axvline(x=3*np.pi/4, color="gray", linestyle=":", alpha=0.3)
    ax2.annotate(f"ι=3π/4: {r135['ratio']:.2f}x\n({r135['well_constrained']})",
                 xy=(3*np.pi/4, r135["ratio"]),
                 xytext=(3*np.pi/4-0.5, r135["ratio"]+0.02),
                 arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9)

    fig.suptitle("Step 1.1 — Combo correlation ratio vs inclination",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig("experiments/phic_psi_poc/sweep_1_1_ratio_vs_iota.png", dpi=120)
    plt.close(fig)
    print("  Plot saved to: experiments/phic_psi_poc/sweep_1_1_ratio_vs_iota.png")


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

        n_sky = 200
        n_iota = 200
        result_1_2 = derive_w_iota(n_sky_samples=n_sky, n_iota_points=n_iota)
        # rev3: confirm sky-averaging and ι sampling density
        print(f"  Sky-averaging: ✓ averaged over {n_sky} random sky-location (a,b) pairs")
        print(f"  ι grid: {n_iota} points from ι≈0.001 to ι≈π/2")
        print(f"  {result_1_2['fit_info']}")
        # Endpoint values from the interpolated curve
        w_interp = result_1_2["w_interpolated"]
        print(f"  w(cos²ι=1) ≈ {w_interp[0]:.4f} (face-on, ι≈0)")
        print(f"  w(cos²ι=0) ≈ {w_interp[-1]:.4f} (edge-on, ι≈π/2)")
        # Intermediate shape check: report 5 equally-spaced ι values
        n = len(result_1_2["iota_grid"])
        indices = [0, n//4, n//2, 3*n//4, n-1]
        print(f"  Intermediate shape (5 of {n} points):")
        for idx in indices:
            iota_v = result_1_2["iota_grid"][idx]
            cos2_v = result_1_2["cos2_iota_grid"][idx]
            w_v = w_interp[idx]
            print(f"    ι={iota_v:.3f}  cos²ι={cos2_v:.3f}  w={w_v:.4f}")
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
