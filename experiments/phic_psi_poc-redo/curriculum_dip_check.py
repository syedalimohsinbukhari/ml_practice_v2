"""Is the edge-on dip in the fitted curriculum weight w(ι) real, or a unit artifact?

Background (``UNDERSTANDING.md`` §3.8): ``curriculum.derive_w_iota`` sets
w(ι) ∝ 1 / mean_sky cond(J), with J the Jacobian of (φc, ψ) → (R, δ).
R is a raw projection sum over ``n_Phi`` samples (so it carries a factor of
about n_Phi / 2 and the sky amplitude), while δ is in radians.
A condition number is not invariant to rescaling one row of J, so the shape
of w(ι) can depend on an arbitrary normalisation of R.

This script recomputes the curve while changing *only* how R enters J:

* ``raw``      : R as shipped (row scale 1);
* ``R x0.01`` / ``R x100`` : R multiplied by a constant (a pure unit change);
* ``log R``    : ∂ log R instead of ∂ R, i.e. a dimensionless row.

If the edge-on dip were a property of the physics, it would not depend on
the unit choice.

Second finding checked here: the shipped orbital-phase grid is
``np.linspace(0, 2π, n_Phi)``, which includes both endpoints, so phase 0 and
2π are sampled twice and the cos/sin projections are not exactly orthogonal.
The ``periodic grid`` variants use ``endpoint=False`` to show the unit
conclusion does not depend on that flaw.

Closed form (hand-derived, checked here as Control 3).
With a = r cos α, b = r sin α and β = 2ψ − α, the projection gives
R = (N/2) r g(β), g = sqrt(k1² cos²β + k2² sin²β), and
δ = 2φc + atan2(k2 sin β, k1 cos β), valid on an exactly periodic grid.
Then ∂R/∂φc = 0 and ∂δ/∂φc = 2, and ∂/∂ψ = 2 ∂/∂β, so J = [[0, p], [2, q]] with
p = ∂R/∂ψ = 2·(N/2)·r·g'(β) and q = ∂δ/∂ψ = 2 k1 k2 / (k1² cos²β + k2² sin²β).
Its condition number κ obeys κ + 1/κ = (p² + q² + 4) / (2|p|).
(See ``closed_form_kappa`` for the exact expressions used.)
At face-on k1 = |k2|, g is constant, p = 0 and κ diverges; away from it p
carries the factor N r / 2 from R's raw units, which is why the scale of R
sets the curve's shape.

Positive controls (CLAUDE.md "positive-control rule"):
(1) the ``raw`` variant at the shipped settings must reproduce
    ``derive_w_iota``'s ``w_raw`` curve (same seed, same grid);
(2) every variant must give w = 0 at face-on, where the degeneracy is exact;
(3) the closed form must match the numerical κ on the periodic grid.
Verdicts are untrusted unless all three pass.

CPU-only, pure NumPy toy model.
Writes ``curriculum_dip_output/curriculum_dip_<YYYYMMDD>_<HHMMSS>.{log,md}``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime as _dt
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import curriculum as C  # noqa: E402

OUT_DIR = _HERE / "curriculum_dip_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PHI_C_REF = 0.5  # same fixed reference point as curriculum.derive_w_iota
PSI_REF = 0.3
N_PHI = 200
# Defaults match the prerequisite run quoted in the paper (200 sky draws, 200-point
# iota grid, UNDERSTANDING.md 3.8); override with --n-sky / --n-iota for quick runs.
N_SKY = 200
N_IOTA = 200
SEEDS = (42, 1, 2)
# (label, row_scale, log_R, periodic_grid)
VARIANTS = (
    ("raw R (as shipped)", 1.0, False, False),
    ("R x0.01", 0.01, False, False),
    ("R x100", 100.0, False, False),
    ("log R (dimensionless)", 1.0, True, False),
    ("raw R, periodic grid", 1.0, False, True),
    ("log R, periodic grid", 1.0, True, True),
)
REPORT_IDX = ()  # set in main() from N_IOTA
N_CF_DRAWS = 600
CF_TOL = 1e-3  # Control 3 pass threshold (max relative error, periodic grid)


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


def _cond(iota, a, b, row_scale, log_R, periodic=False, eps=1e-5, n_Phi=N_PHI):
    """Condition number of the (φc, ψ) → (R', δ) Jacobian, R' = scale·R or log R.

    Same central finite differences as ``curriculum._jacobian_condition_number``;
    only the R row (and, if ``periodic``, the phase grid) is changed.
    """
    Phi = np.linspace(0.0, 2.0 * np.pi, n_Phi, endpoint=not periodic)

    def f(pc, p):
        R, d = C.project_to_R_delta(C.detector_signal(p, pc, iota, a, b, Phi), Phi)
        return np.array([np.log(R) if log_R else R * row_scale, d])

    dpc = (f(PHI_C_REF + eps, PSI_REF) - f(PHI_C_REF - eps, PSI_REF)) / (2.0 * eps)
    dps = (f(PHI_C_REF, PSI_REF + eps) - f(PHI_C_REF, PSI_REF - eps)) / (2.0 * eps)
    J = np.array([[dpc[0], dps[0]], [dpc[1], dps[1]]])
    sv = np.linalg.svd(J, compute_uv=False)
    return 1e15 if sv[1] < 1e-15 else float(sv[0] / sv[1])


def closed_form_kappa(iota, a, b, row_scale=1.0, n_Phi=N_PHI):
    """κ of J = [[0, p], [2, q]] from the exact periodic-grid expressions.

    R = (N/2) r g(β), δ = 2φc + atan2(k2 sin β, k1 cos β), β = 2ψ − α, so
    ∂/∂ψ = 2 ∂/∂β and p = 2 ∂R'/∂β·scale, q = 2 ∂δ/∂β.
    """
    r = np.hypot(a, b)
    alpha = np.arctan2(b, a)
    beta = 2.0 * PSI_REF - alpha
    k1, k2 = C._k1(iota), C._k2(iota)
    den = k1**2 * np.cos(beta) ** 2 + k2**2 * np.sin(beta) ** 2
    g = np.sqrt(den)
    dg = (k2**2 - k1**2) * np.sin(beta) * np.cos(beta) / g
    p = 2.0 * row_scale * (n_Phi / 2.0) * r * dg
    q = 2.0 * k1 * k2 / den
    s = (p * p + q * q + 4.0) / (2.0 * abs(p))  # = κ + 1/κ
    return float((s + np.sqrt(s * s - 4.0)) / 2.0)


def closed_form_check(n_draws=N_CF_DRAWS, seed=0):
    """Relative error of the closed form vs numerical κ, on both phase grids."""
    rng = np.random.default_rng(seed)
    err = {True: [], False: []}  # keyed by periodic
    for _ in range(n_draws):
        iota = rng.uniform(0.05, np.pi / 2 - 0.05)
        ang = rng.uniform(0.0, 2.0 * np.pi)
        a, b = np.cos(ang), np.sin(ang)
        for scale in (1.0, 0.01, 100.0):
            cf = closed_form_kappa(iota, a, b, row_scale=scale)
            for periodic in (True, False):
                num = _cond(iota, a, b, scale, False, periodic=periodic)
                err[periodic].append(abs(cf / num - 1.0))
    return {k: np.array(v) for k, v in err.items()}


def curve(n_sky, seed, n_iota, row_scale, log_R, periodic=False):
    """w(ι) ∝ 1 / mean-over-sky cond, normalised to max 1 (as in derive_w_iota)."""
    rng = np.random.default_rng(seed)
    ab = C._random_sky_coefficients(n_sky, rng)
    iota = np.linspace(0.001, np.pi / 2 - 0.001, n_iota)
    mean_cond = np.array(
        [np.mean([_cond(i, a, b, row_scale, log_R, periodic) for a, b in ab]) for i in iota]
    )
    inv = 1.0 / np.maximum(mean_cond, 1e-12)
    return iota, inv / inv.max()


def _decreases(w, iota, tol=1e-9):
    """(count, largest size, iota of largest) of steps where w decreases along ι."""
    d = np.diff(w)
    bad = np.where(d < -tol)[0]
    if len(bad) == 0:
        return 0, 0.0, float("nan")
    k = bad[np.argmax(-d[bad])]
    return int(len(bad)), float(-d[k]), float(iota[k])


def main() -> None:
    global N_SKY, N_IOTA, REPORT_IDX
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-sky", type=int, default=N_SKY, help="sky-position draws per curve")
    ap.add_argument("--n-iota", type=int, default=N_IOTA, help="points on the iota grid")
    args = ap.parse_args()
    N_SKY, N_IOTA = args.n_sky, args.n_iota
    REPORT_IDX = tuple(int(round(x)) for x in np.linspace(0, N_IOTA - 1, 5))

    ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    stamp = _dt.now().astimezone()
    z = stamp.strftime("%z")
    iso = f"{stamp.strftime('%Y-%m-%d %H:%M')} {z[:3]}:{z[3:]}"
    log_path = OUT_DIR / f"curriculum_dip_{ts}.log"
    md_path = OUT_DIR / f"curriculum_dip_{ts}.md"
    tee = _Tee(str(log_path))
    sys.stdout = tee
    try:
        print("=" * 100)
        print("Curriculum-weight edge-on dip: real or unit artifact?")
        print(f"Run: {iso}   n_sky={N_SKY}  n_iota={N_IOTA}  seeds={SEEDS}")
        print("=" * 100)

        # ---- Positive control 1: reproduce derive_w_iota ---------------------
        print("\n[Control 1] raw-R variant vs curriculum.derive_w_iota (n_sky=50, n_iota=100, seed=42)")
        ref = C.derive_w_iota(n_sky_samples=50, n_iota_points=100, seed=42)
        _, mine = curve(50, 42, 100, 1.0, False)
        max_diff = float(np.max(np.abs(mine - ref["w_raw"])))
        c1_ok = max_diff < 1e-9
        print(f"  max |w_mine - w_raw(derive_w_iota)| = {max_diff:.3e}  ->  {'PASS' if c1_ok else 'FAIL'}")
        print(f"  w(edge-on): mine={mine[-1]:.4f}  derive_w_iota={ref['w_raw'][-1]:.4f}")

        # ---- Positive control 3: closed form vs numerical κ ------------------
        print(f"\n[Control 3] closed-form κ vs numerical κ ({N_CF_DRAWS} draws x 3 row scales, iota in [0.05, pi/2-0.05])")
        cf = closed_form_check()
        cf_p, cf_s = cf[True], cf[False]
        c3_ok = float(cf_p.max()) < CF_TOL
        print(f"  periodic grid : median={np.median(cf_p):.2e}  p99={np.percentile(cf_p, 99):.2e}  max={cf_p.max():.2e}  "
              f"->  {'PASS' if c3_ok else 'FAIL'} (tolerance {CF_TOL:g})")
        print(f"  shipped grid  : median={np.median(cf_s):.2e}  p99={np.percentile(cf_s, 99):.2e}  max={cf_s.max():.2e}  "
              f"(duplicated endpoint; informational)")

        # ---- Main sweep ------------------------------------------------------
        results = {}  # (label, seed) -> (iota, w)
        for label, scale, logr, periodic in VARIANTS:
            for seed in SEEDS:
                results[(label, seed)] = curve(N_SKY, seed, N_IOTA, scale, logr, periodic)
                print(f"  done: {label}, seed {seed}", flush=True)

        iota = results[(VARIANTS[0][0], SEEDS[0])][0]
        print("\n[Sweep] w(ι), normalised to max 1")
        print(f"{'variant / seed':36s}" + " ".join(f"{iota[i]:6.2f}" for i in REPORT_IDX) + "   decreases (n, max, at ι)")
        for label, *_ in VARIANTS:
            for seed in SEEDS:
                w = results[(label, seed)][1]
                n, mx, at = _decreases(w, iota)
                print(f"{label + ' seed=' + str(seed):36s}" + " ".join(f"{w[i]:6.3f}" for i in REPORT_IDX)
                      + f"   {n}, {mx:.3f}, {at:.3f}")

        # ---- Positive control 2: face-on is exactly degenerate ---------------
        faceon = {k: v[1][0] for k, v in results.items()}
        c2_ok = all(x < 1e-3 for x in faceon.values())
        print(f"\n[Control 2] w(face-on) < 1e-3 in every variant/seed: max = {max(faceon.values()):.2e}  ->  "
              f"{'PASS' if c2_ok else 'FAIL'}")

        controls_ok = c1_ok and c2_ok and c3_ok
        print(f"\nControls: {'all pass — verdicts below are trusted' if controls_ok else 'FAILED — verdicts below are UNTRUSTED'}")

        # ---- Derived statistics ---------------------------------------------
        def stat(label, seed):
            w = results[(label, seed)][1]
            n, mx, at = _decreases(w, iota)
            return dict(edge=float(w[-1]), peak_iota=float(iota[int(np.argmax(w))]),
                        n_dec=n, max_dec=mx, dec_at=at, peak_at_edge=bool(int(np.argmax(w)) == len(w) - 1))

        shipped_scales = [l for l, s, lg, pg in VARIANTS if not lg and not pg]
        raw_edges = [stat(l, 42)["edge"] for l in shipped_scales]
        L_LOG, L_LOGP, L_RAWP = "log R (dimensionless)", "log R, periodic grid", "raw R, periodic grid"
        L_RAW = VARIANTS[0][0]

        with open(md_path, "w") as f:
            f.write("# Curriculum-weight edge-on dip: real or unit artifact?\n\n")
            f.write(f"Run {iso}; auto-generated by `curriculum_dip_check.py`, regenerate rather than hand-edit.\n")
            f.write(f"Log: `{log_path.name}`.\n\n")
            f.write("## Question\n\n")
            f.write("`curriculum.derive_w_iota` falls from a peak near ι≈0.8 to a low value at edge-on (`UNDERSTANDING.md` §3.8).\n")
            f.write("The Jacobian it takes the condition number of has one row in raw-R units (a 200-sample projection sum) and one in radians.\n")
            f.write("A condition number changes when one row is rescaled, so the dip could be a unit artifact.\n")
            f.write("This run changes only how R enters the Jacobian (and, in two variants, the phase grid) and leaves everything else as shipped.\n\n")
            f.write("## Controls\n\n")
            f.write(f"- **Control 1** (raw-R variant reproduces `derive_w_iota`, n_sky=50, n_iota=100, seed 42): max |Δw| = {max_diff:.2e}, {'PASS' if c1_ok else 'FAIL'}.\n")
            f.write(f"- **Control 2** (w = 0 at face-on in every variant and seed): max w(face-on) = {max(faceon.values()):.2e}, {'PASS' if c2_ok else 'FAIL'}.\n")
            f.write(f"- **Control 3** (closed-form κ vs numerical κ on the periodic grid, {N_CF_DRAWS} draws x 3 row scales): median relative error {np.median(cf_p):.1e}, max {cf_p.max():.1e}, {'PASS' if c3_ok else 'FAIL'} (tolerance {CF_TOL:g}).\n")
            f.write(f"- Verdicts below are {'trusted' if controls_ok else '**UNTRUSTED** (a control failed)'}.\n\n")
            f.write("## Shipped phase grid vs the closed form\n\n")
            f.write("`np.linspace(0, 2π, n_Phi)` samples phase 0 and 2π twice, so the cos/sin projections are not exactly orthogonal.\n")
            f.write(f"Against the closed form the shipped grid's κ is off by a median {np.median(cf_s):.1e} (99th percentile {np.percentile(cf_s, 99):.1e}, max {cf_s.max():.1e}); the periodic grid's is off by a median {np.median(cf_p):.1e}.\n")
            f.write("This is a separate flaw from the unit dependence; the periodic-grid variants below show the unit conclusion does not rely on it.\n\n")
            f.write("## w(ι) by variant and seed\n\n")
            f.write(f"{N_SKY} sky draws, {N_IOTA}-point ι grid on [0.001, π/2−0.001], fixed reference (φc, ψ) = ({PHI_C_REF}, {PSI_REF}); w normalised to max 1.\n")
            f.write("`peak ι` is the argmax of w; `decreases` is the number of grid steps where w falls, the largest fall, and its ι.\n\n")
            f.write("| variant | seed | " + " | ".join(f"w(ι={iota[i]:.2f})" for i in REPORT_IDX) + " | peak ι | decreases (n; max; at ι) |\n")
            f.write("|---|---|" + "---|" * len(REPORT_IDX) + "---|---|\n")
            for label, *_ in VARIANTS:
                for seed in SEEDS:
                    w = results[(label, seed)][1]
                    s = stat(label, seed)
                    dec = "none" if s["n_dec"] == 0 else f"{s['n_dec']}; {s['max_dec']:.3f}; {s['dec_at']:.2f}"
                    f.write(f"| {label} | {seed} | " + " | ".join(f"{w[i]:.3f}" for i in REPORT_IDX)
                            + f" | {s['peak_iota']:.2f} | {dec} |\n")
            f.write("\n## Key numbers\n\n")
            f.write(f"- Edge-on w, seed 42, shipped grid, over the three raw-R scalings (x0.01, x1, x100): {min(raw_edges):.3f} to {max(raw_edges):.3f}.\n")
            f.write("- Peak ι of the raw-R curve moves with the scale factor (seed 42): "
                    + ", ".join(f"{l} → {stat(l, 42)['peak_iota']:.2f}" for l in shipped_scales) + ".\n")
            f.write(f"- Edge-on w, seed 42, raw R vs log R on the periodic grid: {stat(L_RAWP, 42)['edge']:.3f} vs {stat(L_LOGP, 42)['edge']:.3f}.\n")
            for lab in (L_LOG, L_LOGP):
                ss = [stat(lab, sd) for sd in SEEDS]
                f.write(f"- {lab}: edge-on w = " + ", ".join(f"{x['edge']:.3f}" for x in ss)
                        + f" (seeds {', '.join(map(str, SEEDS))}); peak at edge-on in all seeds: "
                        + f"{'yes' if all(x['peak_at_edge'] for x in ss) else 'no'}; decreases per seed: "
                        + ", ".join(f"{x['n_dec']} (max {x['max_dec']:.3f} at ι={x['dec_at']:.2f})" if x['n_dec'] else "0" for x in ss) + ".\n")
            f.write("\n## Reading (claim labels per CLAUDE.md)\n\n")
            dep = max(raw_edges) - min(raw_edges) > 0.2
            f.write(f"- **Verified** (this run, toy model): the shipped curve's shape depends on an arbitrary unit choice for R — {'its edge-on weight and peak position move under a pure rescaling' if dep else 'NOT SUPPORTED: the edge-on weight did not move by more than 0.2 under rescaling'}.\n")
            f.write("  A feature that moves under a pure unit change is not a property of the physics.\n")
            logr_edge_ok = all(stat(l, sd)["edge"] > 0.99 and stat(l, sd)["peak_at_edge"] for l in (L_LOG, L_LOGP) for sd in SEEDS)
            f.write(f"- **Verified** (this run): with a dimensionless R row the weight {'peaks at edge-on (w≈1) for every seed, on both phase grids' if logr_edge_ok else 'does NOT peak at edge-on in every seed and grid; see the table'}.\n")
            any_dec = any(stat(l, sd)["n_dec"] for l in (L_LOG, L_LOGP) for sd in SEEDS)
            if any_dec:
                f.write("  The log-R curve is not strictly monotone: the table lists the isolated decreases.\n")
                f.write("  Their cause was not isolated in this run; a likely one is a near-singular sky draw dominating the mean κ at that ι (inferred, not checked).\n")
            f.write(f"- **Verified** (Control 3): on an exactly periodic phase grid κ + 1/κ = (p² + q² + 4)/(2|p|) with p = 2·(N/2)·r·g'(β)·(row scale) and q = 2 k1 k2/(k1² cos²β + k2² sin²β), reproducing the numerical κ with max relative error {cf_p.max():.1e}.\n")
            f.write("  p carries the raw-R factor N r / 2, which is why the R scale sets the curve's shape; at face-on g is constant, p = 0 and κ diverges.\n")
            f.write("- **Inferred**: the edge-on dip in `derive_w_iota` is an artifact of the metric, not a real re-degradation near edge-on.\n")
            f.write("- **Not shown**: that the true weight is monotonic.\n")
            f.write("  The log-R curve is much more edge-concentrated than sin²ι, and other dimensionless normalisations were not compared, so this run does not pick a replacement weight.\n")
            f.write("- **Scope limits**: toy detector model only, one fixed reference (φc, ψ), no check against the real generator.\n")
            f.write("  Training used sin²ι throughout, so no trained model depends on this result.\n")
        print(f"\nSummary written: {md_path}")
    finally:
        sys.stdout = tee.stdout
        tee.close()


if __name__ == "__main__":
    main()
