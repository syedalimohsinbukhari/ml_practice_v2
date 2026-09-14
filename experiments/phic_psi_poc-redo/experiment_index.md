# φc/ψ Degeneracy PoC — Redo — Experiment Index

**Branch**: `poc/phic-psi-degeneracy-redo`
**Last updated**: 2026-09-14

---

## Core narrative

| File | Description |
|---|---|
| [`NOTES.md`](NOTES.md) | Running narrative log — Step 1 prerequisite-check results, design decisions, next steps. Read this first. |
| [`redo_procedure.md`](redo_procedure.md) | Step-by-step build plan for the redo, mirroring the original `planning_files/phic_psi_implementation_plan_v4.md`'s structure, with every step marked `[UNCHANGED]`/`[CORRECTED]`/`[NEW]` against the corrected `2φc±2ψ` combo formula. |
| [`formulae_reference.md`](formulae_reference.md) | Exact physics/loss formulae used at each step (vector algebra, combo construction, isotropic circular loss, magnitude penalty, toy antenna-pattern model, curriculum weight derivation, reconstruction branch-handling). Model architectures given as a bare outline only, no per-layer math. |
| [`comments.md`](comments.md) | External review of the redo package (2026-09-14) — confirms the core `2φc±2ψ` fix is internally consistent throughout, hand-verifies the A.5 self-check, and proposed the A.8 branch-ambiguity re-derivation. The branch-*count* part held up; the closed-form parity-filter part did not (see below). |

## Code

| File | Description |
|---|---|
| [`curriculum.py`](curriculum.py) | Step 1.2 toy antenna-pattern model + Jacobian-condition-number sweep + `derive_w_iota`, copied from `experiments/phic_psi_poc/curriculum.py` with `h_plus`/`h_cross` corrected to `2*phi_c` (A.5). |
| [`transform_utils.py`](transform_utils.py) | Vector algebra primitives, copied from `experiments/phic_psi_poc/transform_utils.py` with `double_angle`/`tf_double_angle` added (A.1.5) and `reconstruct_phic_psi` corrected for the `2φc±2ψ` combo (A.8) — brute-force only, no parity shortcut (see below). |
| [`trainer.py`](trainer.py) | `SumDiffTrainer`, copied from the original with one change: `_build_combo_vectors` doubles `z_phic` before combining, per A.2. Verified against synthetic TF tensors. |
| [`validation_script.py`](validation_script.py) | Standalone (no GPU/model) test suite for the transform/reconstruction math, §3. 29/29 checks pass. |
| [`prereq_checks.py`](prereq_checks.py) | Steps 1.1/1.2/1.6 empirical checks, corrected combo↔(φc,ψ) conversion and sweep ranges. Results in `NOTES.md` and `prereq_checks_full_run.log`. |
| [`train_poc.py`](train_poc.py), [`plot_poc.py`](plot_poc.py), [`evaluate_poc.py`](evaluate_poc.py), [`run_full.py`](run_full.py) | Training/plotting/evaluation orchestration, copied verbatim in logic from the original — no formula-specific code in any of them. |

## Analysis outputs

| File | Description |
|---|---|
| [`prereq_checks_full_run.log`](prereq_checks_full_run.log) | Full console output of the Step 1.1/1.2/1.6 run (2026-09-14, full resolution: n_sky=200, n_iota_sweep=50/regime, n_boot=5000). |
| [`sweep_1_1_ratio_vs_iota.csv`](sweep_1_1_ratio_vs_iota.csv), [`sweep_1_1_ratio_vs_iota.png`](sweep_1_1_ratio_vs_iota.png), [`sweep_1_1_ratio_vs_iota.pdf`](sweep_1_1_ratio_vs_iota.pdf) | Step 1.1's ι-sweep data and plot, corrected-formula version. |

## What's still open?

**Step 1 (prerequisite checks) is complete** (2026-09-14, see `NOTES.md`) — harness self-check passes, sign-flip confirmed and now *statistically significant* in both regimes (unlike the original's borderline result), `w(ι)` re-derived, population histogram reconfirmed unchanged. Decision: `well_constrained_combo: combo_B`, `sign_dependent_combo: true`.

**Code implementation (§2/§3) is complete**: `transform_utils.py`, `curriculum.py`, `trainer.py` all built, all numerically verified (5,000–20,000-trial checks where relevant); `validation_script.py` 29/29 pass.

**Resolved by the 2026-09-14 review** (`comments.md`): A.8's reconstruction branch-handling, previously flagged "re-derive before coding," now has confirmed branch counts — ψ is 2-fold ambiguous, φc is 4-fold ambiguous, exactly 4 of the naive 8 candidate pairs are always jointly consistent. Also noted, not a bug: A.5's toy `F_plus`/`F_cross` uses the opposite handedness convention (`e^{+2iψ}`) from some literature conventions — washes out, self-check passes regardless.

**Same-day self-correction (2026-09-14, after implementing `transform_utils.py`):** the review's proposed closed-form parity rule for picking the 4 consistent candidates without brute force (`k≡j mod 2`) was implemented, tested, and immediately falsified — a 20,000-trial numerical sweep showed the parity pattern is ~50/50 and data-dependent (depends on an integer `arctan2`'s mod-2π reduction destroys), not fixed. Retracted in `formulae_reference.md` §A.8 and `redo_procedure.md` §2.6, kept on the record rather than silently erased, per this repo's frozen-vs-living-docs convention. The reconstruction code brute-forces all 8 candidates every time — verified against 5,000 random trials, 100% recovery.

Next steps, in order:
1. Build config YAMLs for the full 7-architecture Round-1-equivalent sweep (baseline mode) + `config_poc.yaml` (poc mode, TCN), using `NOTES.md`'s Step 1 decisions.
2. Hand the full 7-architecture Round-1-equivalent training sweep (§4) to the lab GPU machine — this machine (T530) is CPU-only, per this repo's `CLAUDE.md`.
3. Run the down-select re-validation step (§4) and document, with a fresh comparison table, whether the original 4-model certified set (poc_a, poc_b, tcn, cnn_attention) still holds under the corrected formula.

The original (wrong-formula) investigation is untouched at `experiments/phic_psi_poc/`, and fully preserved as a static snapshot on the `archive/phic-psi-poc-v1` branch.
