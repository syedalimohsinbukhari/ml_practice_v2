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
- None remaining — all prerequisite gates passed.

### Resolved (2026-07-16, rev2)

- [Step 1.1] ✓ Sign/combination check — **rev2: split by sign(cos ι).**
  - **cos ι > 0 (ι=π/4):** well-constrained = **combo_B** (φc−2ψ), ratio = 1.2×.
  - **cos ι < 0 (ι=3π/4):** well-constrained = **combo_A** (φc+2ψ), ratio = 1.2×.
  - **Sign flip: YES** — the well-constrained label depends on sign(cos ι).
  - **Trainer change:** `sign_dependent_combo=true` enables dynamic per-sample
    assignment at batch level (combo_B is good when cos ι ≥ 0, combo_A when cos ι < 0).
  - Note: the 1.2× ratio in each regime is still weak. The rev2 review flagged this —
    it could mean the degeneracy is genuinely near-symmetric, or that the
    correlation-based test statistic has limited sensitivity. Splitting by sign
    eliminated the dilution artifact (was mixing both regimes before).
  - **Config updated:** `well_constrained_combo: combo_B`, `sign_dependent_combo: true`.

- [Step 1.2] ✓ Curriculum weight derivation — **rev2: replaced broken polynomial fit.**
  - Old polynomial fit had residual 1.32 (4× the function range) and inconsistent
    endpoint values. Root cause: fitting an unconstrained 3-param polynomial to a
    sigmoid-like curve in cos²ι.
  - **Fix:** replaced with numpy linear interpolation on the raw empirical curve,
    clamped to [0, 1]. Residual = 0 (interpolation is exact at grid points).
  - w(cos²ι=1) = 0.0000 (face-on) — correct.
  - w(cos²ι=0) = 0.1212 (edge-on) — lower than the 1−cos²ι default would give (1.0).
    Edge-on asymptote is limited by how well (R,δ) can constrain (φc,ψ) even at
    favourable inclination — the analytical Jacobian says the condition number is
    still moderate at edge-on.
  - Improved Jacobian computation: eps=1e-5 (was 1e-6), n_Phi=200 (was 100).
  - **Decision unchanged: use w_iota_default (1−cos²ι) for Run B.** The empirical
    curve's low edge-on asymptote (0.12) may be a real physical signal or a Jacobian
    artifact — either way, the default is safer for an initial go/no-go run.

- [Step 1.6] ✓ cos ι histogram (run on lab GPU machine, 2026-07-16).
  - Face-on fraction (|cos ι| > 0.9): 28.7%.
  - Edge-on fraction (|cos ι| < 0.5): 32.7%.
  - Population is well-mixed — NO statistical power concern. Proceed.

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
| Sign-dependent combo label | Step 1.1 found flip at cos ι=0; dynamic per-sample assignment avoids label dilution |
| w=1−cos²ι default (not empirical fit) | Empirical fit gives edge-on w≈0.12 vs. default 1.0; default is safer for first go/no-go run |

## Run Log

### Run A (baseline) — TBD
- Date:
- Config:
- Result:

### Run B (PoC) — TBD
- Date:
- Config:
- Result:
