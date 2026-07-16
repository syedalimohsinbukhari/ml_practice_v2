"""Complex-multiplication vector utilities for φc/ψ degeneracy PoC.

Provides both NumPy and TensorFlow implementations of the complex-multiply
operations defined in Appendix A.1 of ``phic_psi_implementation_plan_v4.md``.

Convention: a PERIODIC head's raw vector is z = (s, c) where s = sin(θ),
c = cos(θ).  This matches ``atan2(s, c) = θ`` and is consistent with how
``TargetTransforms.transform_head`` encodes PERIODIC parameters (sin in
column 0, cos in column 1).

NumPy versions are used by the validation script and prerequisite checks;
TF versions are used inside the trainer's ``_total_loss`` graph.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# NumPy implementations
# ---------------------------------------------------------------------------


def normalize_unit(z: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Project (..., 2) vectors to unit modulus.

    Reuses the same epsilon convention as the vMF head's ``mu_raw``
    normalisation (``losses.py:52-55``).

    Args:
        z: (..., 2) array where ``[..., 0]`` = sin, ``[..., 1]`` = cos.
        eps: Floor on the denominator magnitude to prevent division by zero.

    Returns:
        (..., 2) unit-norm vectors.
    """
    norm = np.sqrt(z[..., 0] ** 2 + z[..., 1] ** 2) + eps
    s = z[..., 0] / norm
    c = z[..., 1] / norm
    return np.stack([s, c], axis=-1)


def complex_mul(z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
    """Complex multiplication: angle(z1) + angle(z2).

    Given z1 at angle θ₁ and z2 at angle θ₂, returns a unit vector at
    angle (θ₁ + θ₂).  Works for arbitrary batch dimensions.

    Args:
        z1, z2: (..., 2) arrays, each [sin, cos].

    Returns:
        (..., 2) array at angle (θ₁ + θ₂).
    """
    s1, c1 = z1[..., 0], z1[..., 1]
    s2, c2 = z2[..., 0], z2[..., 1]
    s_out = s1 * c2 + c1 * s2
    c_out = c1 * c2 - s1 * s2
    return np.stack([s_out, c_out], axis=-1)


def complex_mul_conj(z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
    """Complex multiply z1 by conjugate of z2: angle(z1) − angle(z2).

    Args:
        z1, z2: (..., 2) arrays, each [sin, cos].

    Returns:
        (..., 2) array at angle (θ₁ − θ₂).
    """
    s1, c1 = z1[..., 0], z1[..., 1]
    s2, c2 = z2[..., 0], z2[..., 1]
    s_out = s1 * c2 - c1 * s2
    c_out = c1 * c2 + s1 * s2
    return np.stack([s_out, c_out], axis=-1)


def circular_loss(
    y_true_vec: np.ndarray,
    y_pred_vec: np.ndarray,
) -> np.ndarray:
    """Isotropic circular loss: L = 1 − dot(pred, true) = 1 − cos(Δθ).

    Assumes both inputs are already normalised to unit vectors.  For unit
    vectors this is provably a function of angular difference only (Appendix
    A.3) — never of absolute direction.

    Args:
        y_true_vec: (..., 2) unit vector.
        y_pred_vec: (..., 2) unit vector.

    Returns:
        (...) scalar per-sample loss.
    """
    dot = y_pred_vec[..., 0] * y_true_vec[..., 0] + \
          y_pred_vec[..., 1] * y_true_vec[..., 1]
    return 1.0 - dot


# ---------------------------------------------------------------------------
# TensorFlow implementations (for use inside tf.function / train_step)
# ---------------------------------------------------------------------------


def _ensure_tf():
    """Lazy import so this module is importable without TensorFlow."""
    import tensorflow as tf
    return tf


def tf_normalize_unit(z, eps: float = 1e-8):
    """TF-graph version of :func:`normalize_unit`."""
    tf = _ensure_tf()
    sq = tf.reduce_sum(tf.square(z), axis=-1, keepdims=True)
    norm = tf.sqrt(tf.maximum(sq, eps))
    return z / norm


def tf_complex_mul(z1, z2):
    """TF-graph version of :func:`complex_mul`."""
    tf = _ensure_tf()
    s1, c1 = z1[..., 0], z1[..., 1]
    s2, c2 = z2[..., 0], z2[..., 1]
    s_out = s1 * c2 + c1 * s2
    c_out = c1 * c2 - s1 * s2
    return tf.stack([s_out, c_out], axis=-1)


def tf_complex_mul_conj(z1, z2):
    """TF-graph version of :func:`complex_mul_conj`."""
    tf = _ensure_tf()
    s1, c1 = z1[..., 0], z1[..., 1]
    s2, c2 = z2[..., 0], z2[..., 1]
    s_out = s1 * c2 - c1 * s2
    c_out = c1 * c2 + s1 * s2
    return tf.stack([s_out, c_out], axis=-1)


# ---------------------------------------------------------------------------
# Reconstruction utilities (Appendix A.4)
# ---------------------------------------------------------------------------


def reconstruct_phic_psi(
    combo_A: np.ndarray,
    combo_B: np.ndarray,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Reconstruct (φc, ψ) candidates from combo vectors with joint-consistency
    filtering (Appendix A.4).

    The two combo vectors are:
      combo_A → angle = φc + 2ψ
      combo_B → angle = φc − 2ψ

    Squaring them to solve for φc and ψ introduces branch ambiguities (2-fold
    for φc, 4-fold for ψ), but the two angles' branches are *coupled* — picking
    independently produces spurious combinations.  This function enumerates all
    2×4=8 candidates, re-encodes each, and keeps only jointly-consistent pairs.

    Args:
        combo_A: (..., 2) vectors at angle φc + 2ψ.
        combo_B: (..., 2) vectors at angle φc − 2ψ.
        tol: Tolerance for the joint-consistency check (via ``1 − dot(...)``).

    Returns:
        (phic_candidates, psi_candidates, all_pairs):
          - phic_candidates: (..., K) jointly-consistent φc values [rad].
          - psi_candidates: (..., K) jointly-consistent ψ values [rad].
          - all_pairs: list of dicts with keys (k_phic, k_psi, phic, psi,
            consistent) for debugging.
    """
    # Step 1: complex multiply / multiply-by-conjugate to isolate angles
    z_prod = complex_mul(combo_A, combo_B)          # angle = 2·φc
    z_ratio = complex_mul_conj(combo_A, combo_B)    # angle = 4·ψ

    # Step 2: extract half-angles (atan2 gives values in (−π, π])
    phic_half = np.arctan2(z_prod[..., 0], z_prod[..., 1])      # 2·φc
    four_psi = np.arctan2(z_ratio[..., 0], z_ratio[..., 1])     # 4·ψ

    # Step 3: branch candidates
    phic_branches = [phic_half / 2.0 + k * np.pi for k in (0, 1)]
    psi_branches = [four_psi / 4.0 + k * (np.pi / 2.0) for k in (0, 1, 2, 3)]

    # Step 4: joint-consistency filter — enumerate all 8 pairs
    all_pairs = []
    for k_p, phic_cand in enumerate(phic_branches):
        for k_psi, psi_cand in enumerate(psi_branches):
            # Re-encode candidate
            z_phic_cand = np.stack(
                [np.sin(phic_cand), np.cos(phic_cand)], axis=-1
            )
            # Note: ψ's PERIODIC head encodes 2ψ internally (period=π)
            two_psi = 2.0 * psi_cand
            z_2psi_cand = np.stack(
                [np.sin(two_psi), np.cos(two_psi)], axis=-1
            )

            recombo_A = complex_mul(z_phic_cand, z_2psi_cand)
            recombo_B = complex_mul_conj(z_phic_cand, z_2psi_cand)

            err_A = circular_loss(combo_A, recombo_A)
            err_B = circular_loss(combo_B, recombo_B)
            consistent = bool(np.all(err_A < tol) and np.all(err_B < tol))

            all_pairs.append({
                "k_phic": k_p,
                "k_psi": k_psi,
                "phic": phic_cand,
                "psi": psi_cand,
                "err_A": err_A,
                "err_B": err_B,
                "consistent": consistent,
            })

    # Collect jointly-consistent candidates
    consistent_pairs = [p for p in all_pairs if p["consistent"]]
    phic_candidates = np.stack([p["phic"] for p in consistent_pairs], axis=-1)
    psi_candidates = np.stack([p["psi"] for p in consistent_pairs], axis=-1)

    return phic_candidates, psi_candidates, all_pairs


def pick_nearest_candidate(
    phic_candidates: np.ndarray,
    psi_candidates: np.ndarray,
    phic_true: np.ndarray,
    psi_true: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick the jointly-consistent candidate closest to known ground truth.

    This is a convenience for the *validation script only* — real inference
    has no ground truth to compare against.  Resolving branch ambiguity at
    inference time is an open problem explicitly out of scope for this PoC
    (see plan Sec. 6).

    Args:
        phic_candidates: (..., K) jointly-consistent φc values [rad].
        psi_candidates: (..., K) jointly-consistent ψ values [rad].
        phic_true: (...,) true φc [rad].
        psi_true: (...,) true ψ [rad].

    Returns:
        (phic_best, psi_best): (...,) arrays of the nearest candidate.
    """
    # Wrap-aware angular distance for periodic parameters
    def _wrap_dist(cand, truth, period):
        d = cand - truth[..., None]
        d = (d + period / 2) % period - period / 2
        return d ** 2

    dist_phic = _wrap_dist(phic_candidates, phic_true, 2.0 * np.pi)
    dist_psi = _wrap_dist(psi_candidates, psi_true, np.pi)

    total_dist = dist_phic + dist_psi  # equal weighting
    best_idx = np.argmin(total_dist, axis=-1)  # (...,)

    # Index into candidates
    phic_best = np.take_along_axis(
        phic_candidates, best_idx[..., None], axis=-1
    )[..., 0]
    psi_best = np.take_along_axis(
        psi_candidates, best_idx[..., None], axis=-1
    )[..., 0]

    return phic_best, psi_best
