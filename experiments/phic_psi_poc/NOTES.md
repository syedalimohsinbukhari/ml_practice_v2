# φc/ψ Degeneracy PoC — Running Notes

Companion to `phic_psi_implementation_plan_v4.md` and
`phi_c_psi_degeneracy_poc.md`.

## Setup

- **Branch:** `poc/phic-psi-degeneracy` (off `master`)
- **Created:** 2026-07-16
- **Goal:** go/no-go signal on sum/difference-head + inclination-curriculum design

## Confirmed vs. Assumed

### Confirmed
- [Step 1.3] Backbone ingests raw whitened time-domain strain (N, 4096, 2) with
  detector order [h1, l1] — inherently encodes phase in temporal structure.
  Confirmed from `loader.py:29-35`.

- [Step 1.4] True inclination (column 2) is accessible via adding `inclination`
  to the active heads list. `TargetTransforms` encodes it as PERIODIC with
  period=2π, giving `y_true["inclination"] = [sin(ι), cos(ι)]`. Therefore
  `y_true["inclination"][:, 1]` = cos(ι_true). No extra data plumbing needed.

- [Step 1.5] Curriculum mechanism: static per-sample loss weight `w(ι_true)`
  throughout training (option a). No epoch-based scheduling.

### Still to verify
- [Step 1.6] cos ι histogram — population distribution check (needs HDF5 data, run on lab GPU machine)

### Resolved (2026-07-16)

- [Step 1.1] ✓ Sign/combination check complete.
  - **Well-constrained combo: combo_B (φc − 2ψ)**
  - Poorly-constrained combo: combo_A (φc + 2ψ) — this gets the curriculum weight.
  - Correlation ratio (well/poorly): 1.2×.
  - Sign flip at cos ι < 0: YES (well-constrained combo flips when cos ι changes sign).
  - Note: the weak 1.2× ratio suggests the degeneracy is nearly symmetric — both combos
    carry limited independent information at moderate inclinations. This is expected from
    the theoretical analysis.

- [Step 1.2] ✓ Curriculum weight derivation complete (analytical harness).
  - Polynomial fit: w(cos²ι) = 0.287 − 2.551·cos²ι − 3.022·cos⁴ι
  - w(ι≈0) = 0.000 (face-on, as expected)
  - w(ι≈π/2) = 0.231 (edge-on) — lower than expected; fit residual is large (1.32).
    **The default w=1−cos²ι fallback is likely better-behaved than this fit.**
    Recommend using the default for initial Run B and revisiting the derivation
    if results are ambiguous.
  - **Decision: use w_iota_default (1−cos²ι) for Run B.** The analytical fit is
    saved for reference but the large residual and low edge-on asymptote suggest
    the finite-difference Jacobian may need a smaller step size or more Phi samples.

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Subclass MultiHeadTrainer | Reuses log_var, metrics, train_step machinery |
| inclination as trained head | Simplest way to get cos(ι_true) at batch level |
| Both numpy + TF transform utils | numpy for validation, TF for graph ops |
| Circular loss (1−dot), not Huber | Provably isotropic; avoids per-component Huber anisotropy |
| Combo log_var clamps at 3.0 (default) | Don't double-suppress with w(ι) |
| Separate train_poc.py script | Doesn't modify main-branch train.py |
| TCN backbone | Current best performer per multi-architecture comparison |

## Run Log

### Run A (baseline) — TBD
- Date:
- Config:
- Result:

### Run B (PoC) — TBD
- Date:
- Config:
- Result:
