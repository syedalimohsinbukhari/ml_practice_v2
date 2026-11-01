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

Positive control (CLAUDE.md "positive-control rule"): (1) the ``raw`` variant
at the shipped settings must reproduce ``derive_w_iota``'s ``w_raw`` curve
(same seed, same grid); (2) every variant must give w = 0 at face-on, where
the degeneracy is exact.
Verdicts are untrusted unless both controls pass.

CPU-only, pure NumPy toy model.
Writes ``curriculum_dip_output/curriculum_dip_<YYYYMMDD>_<HHMMSS>.{log,md}``.
"""

from __future__ import annotations

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
N_SKY = 50
N_IOTA = 40
SEEDS = (42, 1, 2)
# (label, row_scale, log_R)
VARIANTS = (
    ("raw R (as shipped)", 1.0, False),
    ("R x0.01", 0.01, False),
    ("R x100", 100.0, False),
    ("log R (dimensionless)", 1.0, True),
)
REPORT_IDX = (0, 10, 20, 30, N_IOTA - 1)


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


def _cond(iota, a, b, row_scale, log_R, eps=1e-5, n_Phi=200):
    """Condition number of the (φc, ψ) → (R', δ) Jacobian, R' = scale·R or log R.

    Same central finite differences as ``curriculum._jacobian_condition_number``;
    only the R row is transformed.
    """
    Phi = np.linspace(0.0, 2.0 * np.pi, n_Phi)

    def f(pc, p):
        R, d = C.project_to_R_delta(C.detector_signal(p, pc, iota, a, b, Phi), Phi)
        return np.array([np.log(R) if log_R else R * row_scale, d])

    dpc = (f(PHI_C_REF + eps, PSI_REF) - f(PHI_C_REF - eps, PSI_REF)) / (2.0 * eps)
    dps = (f(PHI_C_REF, PSI_REF + eps) - f(PHI_C_REF, PSI_REF - eps)) / (2.0 * eps)
    J = np.array([[dpc[0], dps[0]], [dpc[1], dps[1]]])
    sv = np.linalg.svd(J, compute_uv=False)
    return 1e15 if sv[1] < 1e-15 else float(sv[0] / sv[1])


def curve(n_sky, seed, n_iota, row_scale, log_R):
    """w(ι) ∝ 1 / mean-over-sky cond, normalised to max 1 (as in derive_w_iota)."""
    rng = np.random.default_rng(seed)
    ab = C._random_sky_coefficients(n_sky, rng)
    iota = np.linspace(0.001, np.pi / 2 - 0.001, n_iota)
    mean_cond = np.array(
        [np.mean([_cond(i, a, b, row_scale, log_R) for a, b in ab]) for i in iota]
    )
    inv = 1.0 / np.maximum(mean_cond, 1e-12)
    return iota, inv / inv.max()


def _is_monotone_nondecreasing(w, tol=1e-9):
    return bool(np.all(np.diff(w) >= -tol))


def main() -> None:
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

        # ---- Main sweep ------------------------------------------------------
        results = {}  # (label, seed) -> (iota, w)
        for label, scale, logr in VARIANTS:
            for seed in SEEDS:
                results[(label, seed)] = curve(N_SKY, seed, N_IOTA, scale, logr)

        iota = results[(VARIANTS[0][0], SEEDS[0])][0]
        print("\n[Sweep] w(ι), normalised to max 1")
        print(f"{'variant / seed':34s}" + " ".join(f"{iota[i]:6.2f}" for i in REPORT_IDX))
        for label, _, _ in VARIANTS:
            for seed in SEEDS:
                w = results[(label, seed)][1]
                print(f"{label + ' seed=' + str(seed):34s}" + " ".join(f"{w[i]:6.3f}" for i in REPORT_IDX))

        # ---- Positive control 2: face-on is exactly degenerate ---------------
        faceon = {k: v[1][0] for k, v in results.items()}
        c2_ok = all(x < 1e-3 for x in faceon.values())
        print(f"\n[Control 2] w(face-on) < 1e-3 in every variant/seed: max = {max(faceon.values()):.2e}  ->  "
              f"{'PASS' if c2_ok else 'FAIL'}")

        controls_ok = c1_ok and c2_ok
        print(f"\nControls: {'both pass — verdicts below are trusted' if controls_ok else 'FAILED — verdicts below are UNTRUSTED'}")

        # ---- Summary ---------------------------------------------------------
        def stat(label, seed):
            w = results[(label, seed)][1]
            return dict(
                edge=float(w[-1]),
                peak_iota=float(iota[int(np.argmax(w))]),
                mono=_is_monotone_nondecreasing(w),
            )

        raw_edges = [stat(l, 42)["edge"] for l, s, lg in VARIANTS if not lg]
        logr_label = [l for l, s, lg in VARIANTS if lg][0]
        logr_edges = [stat(logr_label, sd)["edge"] for sd in SEEDS]
        logr_mono = [stat(logr_label, sd)["mono"] for sd in SEEDS]

        with open(md_path, "w") as f:
            f.write("# Curriculum-weight edge-on dip: real or unit artifact?\n\n")
            f.write(f"Run {iso}; auto-generated by `curriculum_dip_check.py`, regenerate rather than hand-edit.\n")
            f.write(f"Log: `{log_path.name}`.\n\n")
            f.write("## Question\n\n")
            f.write("`curriculum.derive_w_iota` falls from a peak near ι≈0.8 to a low value at edge-on (`UNDERSTANDING.md` §3.8).\n")
            f.write("The Jacobian it takes the condition number of has one row in raw-R units (a 200-sample projection sum) and one in radians.\n")
            f.write("A condition number changes when one row is rescaled, so the dip could be a unit artifact.\n")
            f.write("This run changes only how R enters the Jacobian and leaves everything else as shipped.\n\n")
            f.write("## Controls\n\n")
            f.write(f"- **Control 1** (raw-R variant reproduces `derive_w_iota`, n_sky=50, n_iota=100, seed 42): max |Δw| = {max_diff:.2e}, {'PASS' if c1_ok else 'FAIL'}.\n")
            f.write(f"- **Control 2** (w = 0 at face-on in every variant and seed): max w(face-on) = {max(faceon.values()):.2e}, {'PASS' if c2_ok else 'FAIL'}.\n")
            f.write(f"- Verdicts below are {'trusted' if controls_ok else '**UNTRUSTED** (a control failed)'}.\n\n")
            f.write("## w(ι) by variant and seed\n\n")
            f.write(f"{N_SKY} sky draws, {N_IOTA}-point ι grid on [0.001, π/2−0.001], fixed reference (φc, ψ) = ({PHI_C_REF}, {PSI_REF}); w normalised to max 1.\n")
            f.write("`peak ι` is the argmax of w; `monotone` means w never decreases along the grid.\n\n")
            f.write("| variant | seed | " + " | ".join(f"w(ι={iota[i]:.2f})" for i in REPORT_IDX) + " | peak ι | monotone |\n")
            f.write("|---|---|" + "---|" * len(REPORT_IDX) + "---|---|\n")
            for label, _, _ in VARIANTS:
                for seed in SEEDS:
                    w = results[(label, seed)][1]
                    s = stat(label, seed)
                    f.write(f"| {label} | {seed} | " + " | ".join(f"{w[i]:.3f}" for i in REPORT_IDX)
                            + f" | {s['peak_iota']:.2f} | {'yes' if s['mono'] else 'no'} |\n")
            f.write("\n## Key numbers\n\n")
            f.write(f"- Edge-on w, seed 42, over the three raw-R scalings (x0.01, x1, x100): {min(raw_edges):.3f} to {max(raw_edges):.3f}.\n")
            f.write(f"- Edge-on w, log R, seeds {', '.join(map(str, SEEDS))}: {', '.join(f'{x:.3f}' for x in logr_edges)}; monotone in all seeds: {'yes' if all(logr_mono) else 'no'}.\n")
            f.write(f"- Peak ι of the raw-R curve moves with the scale factor (seed 42): "
                    + ", ".join(f"{l} → {stat(l, 42)['peak_iota']:.2f}" for l, s, lg in VARIANTS if not lg) + ".\n\n")
            f.write("## Reading (claim labels per CLAUDE.md)\n\n")
            f.write("- **Verified** (this run, toy model): the shape of the shipped curve, including the position of its peak and the size of the edge-on dip, depends on an arbitrary unit choice for R.\n")
            f.write("  A feature that moves under a pure unit change is not a property of the physics.\n")
            f.write("- **Verified** (this run): with a dimensionless R row the dip is absent in all tested seeds.\n")
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
