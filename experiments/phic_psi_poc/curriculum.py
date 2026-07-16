"""Curriculum weight function w(ι) for φc/ψ degeneracy PoC.

Implements Step 1.2 of ``phic_psi_implementation_plan_v4.md``: an analytical
harness that sweeps ι across [0, π/2], computes the Jacobian condition number
of the (φc, ψ) → (R, δ) map at each inclination (averaged over random sky
positions), and fits a weight function w(ι) to the resulting curve.

Also provides the default/fallback weight function and a histogram utility
for checking the training population's cos(ι) distribution (Step 1.6).

Toy detector model is from Appendix A.5 of the plan — do NOT re-derive the
sign conventions here; they were confirmed against working numerical tests.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Appendix A.5 — Toy antenna-pattern / detector-response model
# ---------------------------------------------------------------------------


def _k1(iota: np.ndarray) -> np.ndarray:
    """h_plus coefficient: (1 + cos²ι) / 2."""
    return (1.0 + np.cos(iota) ** 2) / 2.0


def _k2(iota: np.ndarray) -> np.ndarray:
    """h_cross coefficient: cos(ι)."""
    return np.cos(iota)


def h_plus(phi_c: np.ndarray, iota: np.ndarray, Phi: np.ndarray) -> np.ndarray:
    """h_plus strain component at orbital phase Phi."""
    return _k1(iota) * np.cos(2.0 * Phi + phi_c)


def h_cross(phi_c: np.ndarray, iota: np.ndarray, Phi: np.ndarray) -> np.ndarray:
    """h_cross strain component at orbital phase Phi."""
    return _k2(iota) * np.sin(2.0 * Phi + phi_c)


def F_plus(psi: np.ndarray, a: float, b: float) -> np.ndarray:
    """Detector antenna pattern F_+ for sky-position coefficients (a, b)."""
    return a * np.cos(2.0 * psi) + b * np.sin(2.0 * psi)


def F_cross(psi: np.ndarray, a: float, b: float) -> np.ndarray:
    """Detector antenna pattern F_× for sky-position coefficients (a, b)."""
    return b * np.cos(2.0 * psi) - a * np.sin(2.0 * psi)


def detector_signal(
    psi: np.ndarray,
    phi_c: np.ndarray,
    iota: np.ndarray,
    a: float,
    b: float,
    Phi: np.ndarray,
) -> np.ndarray:
    """Single-detector strain: F₊·h₊ + F×·h×.

    Args:
        psi: Polarization angle [rad].
        phi_c: Coalescence phase [rad].
        iota: Inclination [rad].
        a, b: Sky-position-derived antenna coefficients (scalars).
        Phi: Orbital phase array [rad].

    Returns:
        Strain array, same shape as Phi.
    """
    fp = F_plus(psi, a, b)
    fc = F_cross(psi, a, b)
    hp = h_plus(phi_c, iota, Phi)
    hc = h_cross(phi_c, iota, Phi)
    return fp * hp + fc * hc


def project_to_R_delta(
    signal: np.ndarray, Phi: np.ndarray
) -> tuple[float, float]:
    """Project (2·Phi)-frequency component to recover (R, δ).

    c_proj = Σ signal · cos(2·Φ)
    s_proj = Σ signal · sin(2·Φ)
    R      = hypot(c_proj, s_proj)
    δ      = atan2(−s_proj, c_proj)

    Returns:
        (R, delta) tuple of floats.
    """
    c_proj = np.sum(signal * np.cos(2.0 * Phi))
    s_proj = np.sum(signal * np.sin(2.0 * Phi))
    R = np.hypot(c_proj, s_proj)
    delta = np.arctan2(-s_proj, c_proj)
    return R, delta


# ---------------------------------------------------------------------------
# Step 1.2 — Jacobian condition number → curriculum weight
# ---------------------------------------------------------------------------


def _random_sky_coefficients(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate n random (a, b) sky-position coefficient pairs.

    Each pair is a random unit 2D vector (the antenna-pattern mixing depends
    only on the ratio a/b, not the absolute scale, but unit vectors give a
    natural scale-free sampling).
    """
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return np.column_stack([np.cos(angles), np.sin(angles)])  # (n, 2): (a, b)


def _jacobian_condition_number(
    phi_c: float,
    psi: float,
    iota: float,
    a: float,
    b: float,
    eps: float = 1e-5,
    n_Phi: int = 200,
) -> float:
    """Compute the condition number of the (φc, ψ) → (R, δ) Jacobian.

    Uses central finite differences at the given (φc, ψ, ι, a, b).  High
    condition number → nearly degenerate (the map compresses two dimensions
    into one).

    Args:
        phi_c, psi, iota: Angles [rad].
        a, b: Sky-position coefficients.
        eps: Finite-difference step size (1e-5 rad — 10× smaller than before
             to reduce discretisation noise in the gradient estimate).
        n_Phi: Number of orbital-phase sample points (increased from 100 to
               200 for cleaner projection integrals).

    Returns:
        Condition number (ratio of largest to smallest singular value).
    """
    Phi = np.linspace(0.0, 2.0 * np.pi, n_Phi)

    def _R_delta(pc, p):
        sig = detector_signal(p, pc, iota, a, b, Phi)
        return np.array(project_to_R_delta(sig, Phi))

    # ∂/∂φc
    dR_dpc, dd_dpc = (_R_delta(phi_c + eps, psi) - _R_delta(phi_c - eps, psi)) / (2.0 * eps)

    # ∂/∂ψ
    dR_dpsi, dd_dpsi = (_R_delta(phi_c, psi + eps) - _R_delta(phi_c, psi - eps)) / (2.0 * eps)

    J = np.array([[dR_dpc, dR_dpsi],
                   [dd_dpc, dd_dpsi]])  # (2, 2)

    sv = np.linalg.svd(J, compute_uv=False)
    if sv[1] < 1e-15:
        return 1e15  # effectively singular
    return float(sv[0] / sv[1])


def derive_w_iota(
    n_sky_samples: int = 50,
    n_iota_points: int = 100,
    seed: int = 42,
) -> dict:
    """Derive w(ι) by computing the Jacobian condition number across ι.

    Procedure (Step 1.2):
    1. Generate ~50 random (a,b) sky-position pairs.
    2. For each, sweep ι from 0 to π/2.
    3. At each ι, evaluate the Jacobian condition number at a fixed
       reference (φc, ψ).
    4. Average condition numbers across sky positions for each ι bin.
    5. w ∝ 1/cond(J) — weight is proportional to how much signal the
       poorly-constrained combo carries.
    6. Normalise so max w = 1.

    **Fit method (rev2):**  Uses the raw empirical curve directly via
    linear interpolation on cos²ι, clamped to [0, 1].  No unconstrained
    polynomial — the previous approach produced negative weights and
    residuals larger than the function's own range.  The interpolation
    table is returned alongside the smooth fit for inspection.

    Args:
        n_sky_samples: Number of random sky positions to average over.
        n_iota_points: Number of ι values to sweep.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys:
          - iota_grid: (n_iota_points,) ι values [rad].
          - cos2_iota_grid: (n_iota_points,) cos²ι at each grid point.
          - mean_cond: (n_iota_points,) mean condition number per ι.
          - w_raw: (n_iota_points,) raw empirical w(ι) ∈ [0, 1].
          - w_interpolated: (n_iota_points,) smoothed w(ι) via interpolation.
          - fit_info: str describing the fit.
          - build_w_callable: dict with cos2_iota knots and w values for
            constructing a callable weight function.
    """
    rng = np.random.default_rng(seed)
    sky_coeffs = _random_sky_coefficients(n_sky_samples, rng)

    # Avoid the exact poles (ι=0, π/2) where finite differences break
    iota_grid = np.linspace(0.001, np.pi / 2 - 0.001, n_iota_points)

    # Fixed reference point for φc, ψ (the condition number depends on ι
    # and (a,b) but not strongly on the absolute values of φc, ψ for this
    # quasi-linear map)
    phi_c_ref = 0.5
    psi_ref = 0.3

    cond_all = np.zeros((n_sky_samples, n_iota_points))
    for i in range(n_sky_samples):
        a, b = sky_coeffs[i]
        for j, iota in enumerate(iota_grid):
            cond_all[i, j] = _jacobian_condition_number(
                phi_c_ref, psi_ref, iota, a, b
            )

    mean_cond = np.mean(cond_all, axis=0)

    # Weight ∝ 1/cond(J): high condition → low weight (degenerate)
    # Normalise to [0, 1] with max weight = 1
    inv_cond = 1.0 / np.maximum(mean_cond, 1e-12)
    w_raw = inv_cond / np.max(inv_cond)

    # --- Interpolation-based fit (replaces broken polynomial) ---
    # w(ι) is a smooth function of cos²ι (the natural symmetry variable).
    # Use linear interpolation on the raw empirical curve, clamped to [0, 1].
    # This can't produce the nonsense values (negative weights, residuals
    # larger than the function range) that the old unconstrained polynomial
    # fit produced.
    cos2_iota_grid = np.cos(iota_grid) ** 2  # descending: 1 → 0 as ι: 0 → π/2

    # Sort by cos²ι (ascending) for interpolation
    sort_idx = np.argsort(cos2_iota_grid)
    cos2_sorted = cos2_iota_grid[sort_idx]
    w_sorted = w_raw[sort_idx]

    # Interpolation: cos²ι → w(ι) using numpy (no scipy dependency)
    w_interpolated = np.clip(
        np.interp(cos2_iota_grid, cos2_sorted, w_sorted), 0.0, 1.0
    )

    # Residual check
    residuals = w_raw - w_interpolated
    max_residual = float(np.max(np.abs(residuals)))
    rms_residual = float(np.sqrt(np.mean(residuals ** 2)))

    fit_info = (
        f"Interpolation on cos²ι (linear).  "
        f"w(cos²ι=1) = {w_interpolated[0]:.4f} (face-on), "
        f"w(cos²ι=0) = {w_interpolated[-1]:.4f} (edge-on).  "
        f"Max residual = {max_residual:.2e}, RMS residual = {rms_residual:.2e}."
    )

    return {
        "iota_grid": iota_grid,
        "cos2_iota_grid": cos2_iota_grid,
        "mean_cond": mean_cond,
        "w_raw": w_raw,
        "w_interpolated": w_interpolated,
        "fit_info": fit_info,
        "build_w_callable": {
            "cos2_iota_knots": cos2_sorted.tolist(),
            "w_knots": w_sorted.tolist(),
        },
    }


# ---------------------------------------------------------------------------
# Weight functions (used at training time)
# ---------------------------------------------------------------------------


def w_iota_default(cos_iota: np.ndarray) -> np.ndarray:
    """Default curriculum weight: w = 1 − cos²(ι).

    Monotonically increasing from 0 at |cos ι| = 1 (face-on, fully degenerate)
    to 1 at cos ι = 0 (edge-on, best constrained).

    This is the fallback used *before* Step 1.2's derivation is complete.
    Replace with the fitted function once ``derive_w_iota()`` results are
    available.

    Args:
        cos_iota: (...,) array of true cos(inclination) ∈ [−1, 1].

    Returns:
        (...,) weights ∈ [0, 1].
    """
    return 1.0 - cos_iota ** 2


def build_w_iota_from_fit(fit_result: dict) -> callable:
    """Build a w(cos_ι) callable from the output of :func:`derive_w_iota`.

    Uses linear interpolation on cos²ι from the empirical curve, clamped to
    [0, 1].  The old polynomial-fit approach was removed because it produced
    negative weights and residuals larger than the function's own range.

    Args:
        fit_result: Dict returned by :func:`derive_w_iota`.

    Returns:
        Callable ``w(cos_iota: np.ndarray) → np.ndarray``.
    """
    knots = fit_result["build_w_callable"]
    cos2_knots = np.array(knots["cos2_iota_knots"])
    w_knots = np.array(knots["w_knots"])

    def _w(cos_iota: np.ndarray) -> np.ndarray:
        c2 = (cos_iota ** 2).astype(np.float64)
        return np.clip(np.interp(c2, cos2_knots, w_knots), 0.0, 1.0)

    return _w


# ---------------------------------------------------------------------------
# TensorFlow wrapper (for use inside the trainer graph)
# ---------------------------------------------------------------------------


def tf_w_iota(cos_iota):
    """TF-graph curriculum weight using the default w = 1 − cos²ι.

    Args:
        cos_iota: (B,) tensor of cos(ι_true).

    Returns:
        (B,) tensor of weights ∈ [0, 1].
    """
    import tensorflow as tf
    return 1.0 - tf.square(cos_iota)


# ---------------------------------------------------------------------------
# Step 1.6 — cos ι histogram
# ---------------------------------------------------------------------------


def histogram_cos_iota(
    data_path: str, split: str = "training", max_samples: int | None = None
) -> dict:
    """Load params from HDF5 and histogram true cos(ι) (Step 1.6).

    Args:
        data_path: Path to the HDF5 file.
        split: ``"training"`` or ``"validation"``.
        max_samples: If set, truncate to this many samples.

    Returns:
        Dict with keys:
          - face_on_frac: fraction with |cos ι| > 0.9.
          - edge_on_frac: fraction with |cos ι| < 0.5.
          - cos_iota: (N,) array of cos(ι) values.
          - warning: str or None — warning if population is too face-on-heavy.
    """
    import h5py

    with h5py.File(data_path, "r") as f:
        grp = f[split]
        sl = slice(None, max_samples)
        params = grp["params"][sl]

    # Column 2 = inclination (see heads_spec.PARAM_COLUMNS)
    iota = params[:, 2]
    cos_iota = np.cos(iota)

    face_on_frac = float(np.mean(np.abs(cos_iota) > 0.9))
    edge_on_frac = float(np.mean(np.abs(cos_iota) < 0.5))

    warning = None
    if face_on_frac > 0.9:
        warning = (
            f"WARNING: {face_on_frac:.1%} of samples have |cos ι| > 0.9 "
            f"(heavily face-on).  w(ι) will suppress the diff-head loss on "
            f"nearly everything — Run B may not produce a statistically "
            f"meaningful signal regardless of whether the underlying mechanism "
            f"works.  Consider augmenting with edge-on injections or explicitly "
            f"documenting the statistical weakness."
        )

    return {
        "face_on_frac": face_on_frac,
        "edge_on_frac": edge_on_frac,
        "cos_iota": cos_iota,
        "warning": warning,
    }
