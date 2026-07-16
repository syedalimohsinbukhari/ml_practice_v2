#!/usr/bin/env python
"""Standalone validation script for φc/ψ PoC transform utilities.

Tests the complex-multiplication utilities and reconstruction logic using
synthetic known-answer data.  Does NOT require a trained model or GPU.

Run::

    python experiments/phic_psi_poc/validation_script.py

Exits with code 0 only if all tests pass.

Tests
-----
1.  ``normalize_unit`` — known unit vector unchanged; near-zero doesn't NaN.
2.  ``complex_mul`` — angle sum matches atan2(s_out, c_out).
3.  ``complex_mul_conj`` — angle difference matches atan2(s_out, c_out).
4.  ``circular_loss`` — L = 0 for identical, L = 1 for opposite vectors.
5.  Reconstruction round-trip (A.4) — known (φc, ψ) recovered among
    jointly-consistent candidates.
6.  Known alias — (φc+π, ψ+π/2) also recovered.
7.  Joint-consistency filter — exactly 4 of 8 raw candidates pass.
8.  ``w_iota`` — w=1 at edge-on (cos ι=0), w=0 at face-on (|cos ι|=1).
9.  Analytical model correctness (A.5) — at ι=0, (R,δ) constant along
    φc+2ψ = const lines.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure repo root is importable
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, _REPO_ROOT)

from experiments.phic_psi_poc.transform_utils import (
    circular_loss,
    complex_mul,
    complex_mul_conj,
    normalize_unit,
    pick_nearest_candidate,
    reconstruct_phic_psi,
)
from experiments.phic_psi_poc.curriculum import (
    _random_sky_coefficients,
    detector_signal,
    project_to_R_delta,
    w_iota_default,
)


_PASS = 0
_FAIL = 0


def check(description: str, condition: bool, detail: str = ""):
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"  ✓ {description}")
    else:
        _FAIL += 1
        print(f"  ✗ FAIL: {description}")
        if detail:
            print(f"    {detail}")


# ---------------------------------------------------------------------------
# Test 1 — normalize_unit
# ---------------------------------------------------------------------------


def test_normalize_unit():
    print("\n── Test 1: normalize_unit ──")

    # Known unit vector should be unchanged
    z = np.array([[0.6, 0.8]])  # norm = 1.0
    zn = normalize_unit(z)
    check("unit vector unchanged", np.allclose(zn, z, atol=1e-7),
          f"got {zn}")

    # Non-unit vector should be projected
    z2 = np.array([[3.0, 4.0]])  # norm = 5.0
    zn2 = normalize_unit(z2)
    check("non-unit projected to unit", np.allclose(zn2, [[0.6, 0.8]], atol=1e-7),
          f"got {zn2}")

    # Near-zero vector should not NaN
    z3 = np.array([[0.0, 0.0]])
    zn3 = normalize_unit(z3)
    check("zero vector doesn't NaN", not np.any(np.isnan(zn3)),
          f"got {zn3}")

    # Batch of vectors
    z4 = np.random.randn(10, 2)
    zn4 = normalize_unit(z4)
    norms = np.sqrt(zn4[..., 0] ** 2 + zn4[..., 1] ** 2)
    check("batch all unit norm", np.allclose(norms, 1.0, atol=1e-6),
          f"max deviation = {np.max(np.abs(norms - 1.0)):.2e}")


# ---------------------------------------------------------------------------
# Test 2 — complex_mul (angle sum)
# ---------------------------------------------------------------------------


def test_complex_mul():
    print("\n── Test 2: complex_mul (angle sum) ──")

    for theta1, theta2 in [(0.5, 0.3), (1.2, -0.8), (3.0, -2.5), (0.0, np.pi)]:
        z1 = np.array([[np.sin(theta1), np.cos(theta1)]])
        z2 = np.array([[np.sin(theta2), np.cos(theta2)]])
        z_out = complex_mul(z1, z2)

        expected = theta1 + theta2
        actual = np.arctan2(z_out[0, 0], z_out[0, 1])

        # Wrap-aware comparison
        diff = (actual - expected + np.pi) % (2 * np.pi) - np.pi
        ok = abs(diff) < 1e-10
        check(f"θ₁={theta1:.2f} + θ₂={theta2:.2f} → θ={actual:.6f}", ok,
              f"expected {expected:.6f}, diff={diff:.2e}")


# ---------------------------------------------------------------------------
# Test 3 — complex_mul_conj (angle difference)
# ---------------------------------------------------------------------------


def test_complex_mul_conj():
    print("\n── Test 3: complex_mul_conj (angle difference) ──")

    for theta1, theta2 in [(0.5, 0.3), (1.2, -0.8), (3.0, 2.5), (0.0, np.pi)]:
        z1 = np.array([[np.sin(theta1), np.cos(theta1)]])
        z2 = np.array([[np.sin(theta2), np.cos(theta2)]])
        z_out = complex_mul_conj(z1, z2)

        expected = theta1 - theta2
        actual = np.arctan2(z_out[0, 0], z_out[0, 1])

        diff = (actual - expected + np.pi) % (2 * np.pi) - np.pi
        ok = abs(diff) < 1e-10
        check(f"θ₁={theta1:.2f} − θ₂={theta2:.2f} → θ={actual:.6f}", ok,
              f"expected {expected:.6f}, diff={diff:.2e}")


# ---------------------------------------------------------------------------
# Test 4 — circular_loss
# ---------------------------------------------------------------------------


def test_circular_loss():
    print("\n── Test 4: circular_loss ──")

    # L ≈ 0 for identical vectors (tiny deviation from normalize_unit's eps)
    z = normalize_unit(np.array([[1.0, 2.0]]))
    loss = circular_loss(z, z)
    check("L≈0 for identical vectors", np.allclose(loss, 0.0, atol=1e-7),
          f"loss = {loss}")

    # L = 2 for opposite vectors (π apart): 1 − cos(π) = 1 − (−1) = 2
    z1 = np.array([[np.sin(0.0), np.cos(0.0)]])      # angle = 0
    z2 = np.array([[np.sin(np.pi), np.cos(np.pi)]])   # angle = π
    loss = circular_loss(z1, z2)
    check("L=2 for opposite vectors (Δθ=π)", np.allclose(loss, 2.0, atol=1e-10),
          f"loss = {loss}")

    # L ≈ 1 − cos(Δθ) for general angles
    for dtheta in [0.1, 0.5, 1.0, 2.0]:
        z1 = np.array([[np.sin(0.3), np.cos(0.3)]])
        z2 = np.array([[np.sin(0.3 + dtheta), np.cos(0.3 + dtheta)]])
        loss = circular_loss(z1, z2)
        expected = 1.0 - np.cos(dtheta)
        check(f"L ≈ 1−cos({dtheta:.1f})", np.allclose(loss, expected, atol=1e-10),
              f"loss={loss[0]:.6f}, expected={expected:.6f}")

    # Batch of vectors
    n = 50
    angles1 = np.random.uniform(0, 2 * np.pi, n)
    angles2 = np.random.uniform(0, 2 * np.pi, n)
    z1 = np.stack([np.sin(angles1), np.cos(angles1)], axis=-1)
    z2 = np.stack([np.sin(angles2), np.cos(angles2)], axis=-1)
    loss = circular_loss(z1, z2)
    expected = 1.0 - np.cos(angles1 - angles2)
    check("batch matches 1−cos(Δθ)", np.allclose(loss, expected, atol=1e-10),
          f"max dev = {np.max(np.abs(loss - expected)):.2e}")


# ---------------------------------------------------------------------------
# Test 5 — Reconstruction round-trip (A.4)
# ---------------------------------------------------------------------------


def test_reconstruction_roundtrip():
    print("\n── Test 5: Reconstruction round-trip ──")

    rng = np.random.default_rng(42)
    n = 20

    for i in range(n):
        phic_true = rng.uniform(0.0, 2.0 * np.pi)
        psi_true = rng.uniform(0.0, np.pi)  # period = π

        # Build combo_A and combo_B from known truth
        z_phic = np.array([np.sin(phic_true), np.cos(phic_true)])
        # ψ PERIODIC encoding: z_ψ = [sin(2ψ), cos(2ψ)] because period=π
        z_2psi = np.array([np.sin(2.0 * psi_true), np.cos(2.0 * psi_true)])

        combo_A = complex_mul(z_phic, z_2psi)       # φc + 2ψ
        combo_B = complex_mul_conj(z_phic, z_2psi)   # φc − 2ψ

        # Reconstruct candidates
        phic_cands, psi_cands, all_pairs = reconstruct_phic_psi(
            combo_A[np.newaxis, :], combo_B[np.newaxis, :]
        )

        # Check that the true pair is among jointly-consistent candidates
        found_true = False
        for k in range(phic_cands.shape[-1]):
            pc = phic_cands[0, k]
            ps = psi_cands[0, k]
            # Wrap-aware check
            d_phic = (pc - phic_true + np.pi) % (2.0 * np.pi) - np.pi
            d_psi = (ps - psi_true + np.pi / 2) % np.pi - np.pi / 2
            if abs(d_phic) < 1e-10 and abs(d_psi) < 1e-10:
                found_true = True
                break

        if not found_true:
            check(
                f"true (φc={phic_true:.3f}, ψ={psi_true:.3f}) recovered",
                False,
                f"candidates: φc={phic_cands[0]}, ψ={psi_cands[0]}"
            )
            return
    check("all 20 random true pairs recovered among candidates", True)


# ---------------------------------------------------------------------------
# Test 6 — Known alias (φc+π, ψ+π/2)
# ---------------------------------------------------------------------------


def test_alias():
    print("\n── Test 6: Known alias (φc+π, ψ+π/2) ──")

    rng = np.random.default_rng(99)
    n = 20

    for i in range(n):
        phic_true = rng.uniform(0.0, 2.0 * np.pi)
        psi_true = rng.uniform(0.0, np.pi)

        z_phic = np.array([np.sin(phic_true), np.cos(phic_true)])
        z_2psi = np.array([np.sin(2.0 * psi_true), np.cos(2.0 * psi_true)])

        combo_A = complex_mul(z_phic, z_2psi)
        combo_B = complex_mul_conj(z_phic, z_2psi)

        phic_cands, psi_cands, _ = reconstruct_phic_psi(
            combo_A[np.newaxis, :], combo_B[np.newaxis, :]
        )

        # Known alias
        alias_phic = (phic_true + np.pi) % (2.0 * np.pi)
        alias_psi = (psi_true + np.pi / 2.0) % np.pi

        found_alias = False
        for k in range(phic_cands.shape[-1]):
            pc = phic_cands[0, k]
            ps = psi_cands[0, k]
            d_phic = (pc - alias_phic + np.pi) % (2.0 * np.pi) - np.pi
            d_psi = (ps - alias_psi + np.pi / 2) % np.pi - np.pi / 2
            if abs(d_phic) < 1e-10 and abs(d_psi) < 1e-10:
                found_alias = True
                break

        if not found_alias:
            check("alias (φc+π, ψ+π/2) recovered", False,
                  f"candidates: φc={phic_cands[0]}, ψ={psi_cands[0]}")
            return
    check("all 20 alias pairs recovered among candidates", True)


# ---------------------------------------------------------------------------
# Test 7 — Joint-consistency filter cardinality
# ---------------------------------------------------------------------------


def test_joint_consistency_filter():
    print("\n── Test 7: Joint-consistency filter ──")

    rng = np.random.default_rng(123)
    n = 50

    for i in range(n):
        phic = rng.uniform(0.0, 2.0 * np.pi)
        psi = rng.uniform(0.0, np.pi)

        z_phic = np.array([np.sin(phic), np.cos(phic)])
        z_2psi = np.array([np.sin(2.0 * psi), np.cos(2.0 * psi)])

        combo_A = complex_mul(z_phic, z_2psi)
        combo_B = complex_mul_conj(z_phic, z_2psi)

        phic_cands, psi_cands, all_pairs = reconstruct_phic_psi(
            combo_A[np.newaxis, :], combo_B[np.newaxis, :]
        )

        n_consistent = sum(1 for p in all_pairs if p["consistent"])
        n_total = len(all_pairs)

        if n_total != 8:
            check(f"total candidates = 8", False, f"got {n_total}")
            return
        if n_consistent != 4:
            check(f"consistent candidates = 4", False, f"got {n_consistent}")
            return

    check(f"all {n} cases: exactly 4 of 8 candidates jointly consistent", True)


# ---------------------------------------------------------------------------
# Test 8 — w_iota extremes
# ---------------------------------------------------------------------------


def test_w_iota():
    print("\n── Test 8: w_iota extremes ──")

    w = w_iota_default

    # Edge-on: cos ι = 0 → w = 1
    check("w(cos ι=0) = 1 (edge-on, full weight)",
          np.allclose(w(np.array([0.0])), 1.0, atol=1e-10))

    # Face-on: |cos ι| = 1 → w = 0
    check("w(cos ι=1) = 0 (face-on, suppressed)",
          np.allclose(w(np.array([1.0])), 0.0, atol=1e-10))
    check("w(cos ι=−1) = 0 (face-on, suppressed)",
          np.allclose(w(np.array([-1.0])), 0.0, atol=1e-10))

    # Monotonic between 0 and 1
    cos_vals = np.linspace(0.0, 1.0, 20)
    w_vals = w(cos_vals)
    check("monotonically decreasing as |cos ι| increases",
          np.all(np.diff(w_vals) <= 0))

    # Values in [0, 1]
    check("all weights ∈ [0, 1]", np.all((w_vals >= 0.0) & (w_vals <= 1.0)))


# ---------------------------------------------------------------------------
# Test 9 — Analytical model correctness (A.5 ι=0 check)
# ---------------------------------------------------------------------------


def test_analytical_model_iota_zero():
    print("\n── Test 9: Analytical model ι=0 degeneracy check ──")

    rng = np.random.default_rng(7)
    sky_coeffs = _random_sky_coefficients(20, rng)
    Phi = np.linspace(0.0, 2.0 * np.pi, 100)
    iota_zero = 0.0

    # For each (a,b) pair, fix one const_line = φc + 2ψ, then sweep ψ
    # (adjusting φc to stay on the line).  At ι=0, (R,δ) must be constant
    # along this line — the defining property of the degeneracy.
    max_R_std = 0.0
    max_delta_std = 0.0
    n_checks = 0

    for a, b in sky_coeffs:
        phic_base = rng.uniform(0.0, 2.0 * np.pi)
        psi_base = rng.uniform(0.0, np.pi)
        const_line = phic_base + 2.0 * psi_base

        # Sweep ψ, keeping φc + 2ψ = const_line
        R_group = []
        delta_group = []
        n_sweep = 15
        for psi in np.linspace(0.0, np.pi, n_sweep, endpoint=False):
            phic = (const_line - 2.0 * psi) % (2.0 * np.pi)
            sig = detector_signal(psi, phic, iota_zero, a, b, Phi)
            R, delta = project_to_R_delta(sig, Phi)
            R_group.append(R)
            delta_group.append(delta)

        R_group = np.array(R_group)
        delta_group = np.array(delta_group)

        max_R_std = max(max_R_std, float(np.std(R_group)))
        max_delta_std = max(max_delta_std, float(np.std(delta_group)))
        n_checks += 1

    # Check: within each (a,b,const_line) group, std should be near zero
    check(
        f"max within-group std(R) < 1e-10 at ι=0 (got {max_R_std:.2e})",
        max_R_std < 1e-10,
    )
    check(
        f"max within-group std(δ) < 1e-10 at ι=0 (got {max_delta_std:.2e})",
        max_delta_std < 1e-10,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    global _PASS, _FAIL
    print("=" * 70)
    print("φc/ψ PoC — Validation Script")
    print("=" * 70)

    test_normalize_unit()
    test_complex_mul()
    test_complex_mul_conj()
    test_circular_loss()
    test_reconstruction_roundtrip()
    test_alias()
    test_joint_consistency_filter()
    test_w_iota()
    test_analytical_model_iota_zero()

    print("\n" + "=" * 70)
    print(f"Results: {_PASS} passed, {_FAIL} failed")
    print("=" * 70)

    if _FAIL > 0:
        print("\n*** SOME TESTS FAILED — fix before proceeding to training ***")
        sys.exit(1)
    else:
        print("\nAll tests passed. ✓")
        sys.exit(0)


if __name__ == "__main__":
    main()
