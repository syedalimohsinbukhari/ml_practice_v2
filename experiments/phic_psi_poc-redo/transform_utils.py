"""Complex-multiplication vector utilities for the φc/ψ degeneracy PoC redo.

Provides both NumPy and TensorFlow implementations of the complex-multiply
operations defined in ``formulae_reference.md`` A.1/A.1.5. Copied from
``experiments/phic_psi_poc/transform_utils.py`` with two changes: an added
``double_angle``/``tf_double_angle`` helper (A.1.5), and a corrected
``reconstruct_phic_psi`` (A.8) reflecting the ``2φc±2ψ`` combo formula
instead of the original's ``φc±2ψ``.

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

    Uses an additive epsilon on the denominator (``norm = ||z|| + eps``),
    not a floor/clamp on the norm itself. This differs from both the vMF
    head's ``mu_raw`` normalisation (``max(||v||, eps)``, losses.py:52-55)
    and this codebase's own TF training-path periodic-head normalisation
    (``tf_normalize_unit`` below, which floors ``||v||^2`` rather than
    ``||v||`` -- an effective floor on ``||v||`` of ``sqrt(eps)``, not
    ``eps``). This NumPy function is used by validation/self-consistency
    scripts only, not by training. See UNDERSTANDING.md Topic 2 for the
    full three-way comparison and why the discrepancy doesn't affect any
    result in this study.

    Args:
        z: (..., 2) array where ``[..., 0]`` = sin, ``[..., 1]`` = cos.
        eps: Additive floor on the denominator to prevent division by zero.

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


def double_angle(z: np.ndarray) -> np.ndarray:
    """Double a unit vector's angle: complex_mul(z, z) -> angle 2*theta (A.1.5).

    Used to turn ``z_phic`` (angle phi_c, from the unchanged period-2π
    ``coa_phase`` head) into a ``2*phi_c``-angle vector before combining
    with ``z_psi`` in the corrected combo construction (A.2).

    Args:
        z: (..., 2) array, [sin, cos].

    Returns:
        (..., 2) array at angle 2*theta.
    """
    return complex_mul(z, z)


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


def tf_double_angle(z):
    """TF-graph version of :func:`double_angle` (A.1.5)."""
    return tf_complex_mul(z, z)


# ---------------------------------------------------------------------------
# Reconstruction utilities (formulae_reference.md A.8, corrected)
# ---------------------------------------------------------------------------


def reconstruct_phic_psi(
    combo_A: np.ndarray,
    combo_B: np.ndarray,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Reconstruct (φc, ψ) candidates from combo vectors with joint-consistency
    filtering (formulae_reference.md A.8, corrected for the ``2φc±2ψ`` combo).

    The two combo vectors are:
      combo_A → angle = 2φc + 2ψ
      combo_B → angle = 2φc − 2ψ

    ``complex_mul(combo_A, combo_B)`` isolates ``4φc``; ``complex_mul_conj``
    isolates ``4ψ``. Recovering φc (own period 2π) from ``4φc`` mod 2π gives
    **4 raw branches** (spaced π/2 apart); recovering ψ (own period π) from
    ``4ψ`` mod 2π gives **2 raw branches** in ψ's own physical range (the
    naive 4 branches spaced π/2 apart collapse to 2 distinct physical
    values, since ψ+π ≡ ψ). Empirically, exactly 4 of the naive 4×2=8
    (φc-branch, ψ-branch) combinations are jointly consistent, always — but
    **there is no fixed closed-form rule** (e.g. "branch indices share
    parity") that predicts *which* 4 without checking: which of the two
    possible parity patterns occurs is data-dependent (confirmed empirically
    ~50/50 across 20,000 random trials, 2026-09-14) because it depends on an
    integer wrap-count that ``arctan2``'s mod-2π reduction destroys before
    it ever reaches this function. An earlier version of this function
    (and of `formulae_reference.md` A.8) claimed a fixed parity rule based
    on an algebraic derivation that implicitly assumed the branch indexing
    was anchored at the true φc/ψ themselves, rather than at ``arctan2``'s
    arbitrary representative — that assumption doesn't hold, and the claim
    was retracted the same day after this direct numerical check disagreed
    with it. This function therefore does the same brute-force
    re-encode-and-filter the original (``φc±2ψ``) version always did —
    there is no shortcut to skip here, just an updated branch count.

    Args:
        combo_A: (..., 2) vectors at angle 2φc + 2ψ.
        combo_B: (..., 2) vectors at angle 2φc − 2ψ.
        tol: Tolerance for the joint-consistency check (via ``1 − dot(...)``).

    Returns:
        (phic_candidates, psi_candidates, all_pairs):
          - phic_candidates: (..., 4) jointly-consistent φc values [rad].
          - psi_candidates: (..., 4) jointly-consistent ψ values [rad].
          - all_pairs: list of dicts with keys (k_phic, k_psi, phic, psi,
            consistent) for debugging — covers all 8 naive (k,j)
            combinations.
    """
    # Step 1: complex multiply / multiply-by-conjugate to isolate angles
    z_prod = complex_mul(combo_A, combo_B)          # angle = 4·φc
    z_ratio = complex_mul_conj(combo_A, combo_B)    # angle = 4·ψ

    # Step 2: extract quarter-angles (atan2 gives values in (−π, π])
    four_phic = np.arctan2(z_prod[..., 0], z_prod[..., 1])      # 4·φc
    four_psi = np.arctan2(z_ratio[..., 0], z_ratio[..., 1])     # 4·ψ

    # Step 3: branch candidates.
    # φc: own period 2π, 4 distinct branches spaced π/2 apart (k=0..3).
    # ψ: own period π, only 2 distinct physical branches (j=0,1) — j=2,3
    #    would just repeat j=0,1 shifted by ψ's own period π.
    phic_branches = [four_phic / 4.0 + k * (np.pi / 2.0) for k in (0, 1, 2, 3)]
    psi_branches = [four_psi / 4.0 + j * (np.pi / 2.0) for j in (0, 1)]

    # Step 4: joint-consistency filter — enumerate all 4×2=8 pairs and
    # re-encode each. No closed-form shortcut (see docstring) — brute force
    # is the only correct approach here, same as the original.
    all_pairs = []
    for k_p, phic_cand in enumerate(phic_branches):
        for k_psi, psi_cand in enumerate(psi_branches):
            # Re-encode candidate. Note: coa_phase's head vector encodes
            # phi_c directly (period=2π, unchanged) — double it (A.1.5)
            # before combining, same as the forward combo construction (A.2).
            z_phic_cand = np.stack(
                [np.sin(phic_cand), np.cos(phic_cand)], axis=-1
            )
            z_2phic_cand = double_angle(z_phic_cand)
            # ψ's PERIODIC head encodes 2ψ internally (period=π), unchanged.
            two_psi = 2.0 * psi_cand
            z_2psi_cand = np.stack(
                [np.sin(two_psi), np.cos(two_psi)], axis=-1
            )

            recombo_A = complex_mul(z_2phic_cand, z_2psi_cand)
            recombo_B = complex_mul_conj(z_2phic_cand, z_2psi_cand)

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

    # Collect jointly-consistent candidates (empirically always exactly 4)
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
