# φc/ψ Degeneracy PoC — Redo — Running Notes

Companion to [`redo_procedure.md`](redo_procedure.md) (step-by-step plan) and [`formulae_reference.md`](formulae_reference.md) (exact math). The original investigation at `experiments/phic_psi_poc/` used the wrong combo formula (`φc±2ψ`); this redo uses the corrected `2φc±2ψ` (see `experiments/phic_psi_poc/diagnostic_log.md`'s 2026-09-09 dated entry for the full evidence chain). Original preserved untouched, and separately snapshotted on `archive/phic-psi-poc-v1`.

## Setup

- **Branch:** `poc/phic-psi-degeneracy-redo` (off `master`)
- **Created:** 2026-09-13
- **Goal:** redo the φc/ψ degeneracy PoC under the corrected combo formula; determine whether the original's null result (network learns nothing) still holds, or whether it was an artifact of testing the wrong target.

## Step 1 — Prerequisite checks (2026-09-14)

Ran `prereq_checks.py` at full resolution (n_sky=200, n_iota_sweep=50/regime, n_iota_w=200, n_boot=5000 — same settings as the original's own run). CPU-only, completed in ~3 minutes (faster than the original's 30–70 min estimate — likely machine/load-dependent, not a methodology difference). Full output: [`prereq_checks_full_run.log`](prereq_checks_full_run.log).

### Step 1.1 — Sign/combination check ✓

- **Harness self-check passed**: `(R,δ)` constant at ι=0 along `2φc+2ψ=const` lines, max std(R)=1.16e-14, max std(δ)=5.87e-16 — essentially machine precision, confirms the corrected A.5 formula is implemented correctly before trusting anything downstream.
- **Sign flip: YES** — same qualitative structure as the original (well-constrained label depends on sign(cos ι)):
  - **cos ι > 0** (ι=π/4): well-constrained = **combo_B** (2φc−2ψ), ratio = **1.17×**, 95% CI = **[1.11, 1.24]**.
  - **cos ι < 0** (ι=3π/4): well-constrained = **combo_A** (2φc+2ψ), ratio = **1.19×**, 95% CI = **[1.12, 1.26]**.
- **Significance: YES for both regimes** — 95% CI excludes 1.0 in both. This is a materially different outcome from the original investigation, whose equivalent check (`experiments/phic_psi_poc/results.md`, rev3) found ratios of similar magnitude (~1.2×) that were *not* statistically distinguishable from 1.0 in the same reference-point test (only the deeper ι-sweep + bootstrap in that investigation's rev3 established significance). Here, even the simple two-point reference check clears significance directly.
- **ι-sweep trend** (100 points, both sign regimes): ratio grows toward face-on and shrinks toward edge-on in both regimes, exactly the physics-predicted shape — from ~1.6× near ι≈0.05 down to ~1.05–1.07× near ι≈π/2, mirrored on the cos ι<0 side. Matches the original's qualitative trend, with a similar peak magnitude (original: ~1.6× at ι≈0.1; here: ~1.6–1.65× near the same range) but a materially stronger, now-significant baseline.
- **Decision: `well_constrained_combo: combo_B`, `sign_dependent_combo: true`** — same config shape the original settled on (Step 1.1 rev2/rev3), now on firmer statistical footing given both reference points clear significance directly rather than needing the full bootstrap/sweep apparatus to establish it.

Sweep data: [`sweep_1_1_ratio_vs_iota.csv`](sweep_1_1_ratio_vs_iota.csv), [`sweep_1_1_ratio_vs_iota.png`](sweep_1_1_ratio_vs_iota.png), [`sweep_1_1_ratio_vs_iota.pdf`](sweep_1_1_ratio_vs_iota.pdf).

### Step 1.2 — Curriculum weight derivation ✓

- Same recipe as the original (Jacobian condition-number sweep → linear interpolation on cos²ι), now against the corrected A.5 antenna-pattern model.
- `w(cos²ι=1) = 0.0000` (face-on, fully suppressed — same as original).
- `w(cos²ι=0) = 0.1416` (edge-on) — close in magnitude to the original's 0.1212 (both well below the naive `1−cos²ι` default's edge-on value of 1.0), same qualitative shape (peaks at intermediate ι, ~0.9956 near ι≈0.789 here vs. the original's similar intermediate-peak behavior).
- **Decision: use `w_iota_default` (1−cos²ι) for the first training runs**, same reasoning as the original — the empirical fit's low edge-on asymptote may be real physics or a harness artifact either way; the default is the safer choice for a first go/no-go pass. Can switch to the fitted curve later if warranted.

### Step 1.3–1.5 — carried over unchanged, not re-run

Data representation, true-inclination batch access, and the static per-sample curriculum mechanism are not formula-dependent — see `redo_procedure.md` §1.3–1.5 for why these are cited from the original rather than re-verified.

### Step 1.6 — cos ι histogram ✓

- Face-on fraction (|cos ι| > 0.9): **28.7%**
- Edge-on fraction (|cos ι| < 0.5): **32.7%**
- Identical to the original's Step 1.6 result (same dataset, not formula-dependent) — population is well-mixed, no statistical power concern. Proceed.

## Design decisions (redo)

| Decision | Rationale |
|---|---|
| `well_constrained_combo: combo_B`, `sign_dependent_combo: true` | Step 1.1 (redone, 2026-09-14): significant sign-dependent effect, both regimes clear 95% CI |
| `w_iota_default` (1−cos²ι) for first runs | Step 1.2 (redone): same reasoning as original — empirical fit's edge-on asymptote ambiguous, default is the safer first pass |
| `activation="linear"` for PERIODIC heads, from run 1 | Known root cause (tanh saturation), established in the original investigation (`diagnostic_log.md` Run 3) — not rediscovered |
| `magnitude_penalty_lambda=0.01`, from run 1 | Known root cause (normalize_unit gradient pathology), established in the original (`tanh_to_linear_postmortem.md`, Run 6) — not rediscovered |
| `checkpoint_every_n=10` | New: satisfies CLAUDE.md's periodic-checkpoint requirement from the start (`src/gwml/training/callbacks.py`'s `PeriodicCheckpoint`) |
| Full 7-architecture Round-1-equivalent sweep | Per explicit instruction — redo the full down-select rather than assume the original's 4-model certified set still holds |

## Next steps

- [x] Step 0 — branches, folder, shared checkpoint-callback fix
- [x] Step 1.1/1.2/1.6 — prerequisite checks, redone under corrected formula
- [x] Step 2.2/2.3 — `transform_utils.py`/`curriculum.py`/`trainer.py` implemented and verified
- [x] Step 3 — `validation_script.py`, 29/29 checks pass
- [ ] Build config YAMLs for the full 7-architecture Round-1-equivalent sweep (baseline mode) + `config_poc.yaml` (poc mode, TCN) using this section's decisions
- [ ] Hand training off to the lab GPU machine (this machine is CPU-only per CLAUDE.md)
- [ ] Down-select re-validation once Round-1-equivalent results land
