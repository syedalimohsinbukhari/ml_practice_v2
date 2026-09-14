# φc/ψ Degeneracy PoC — Redo — Experiment Index

**Branch**: `poc/phic-psi-degeneracy-redo`
**Last updated**: 2026-09-14

---

## Files

| File | Description |
|---|---|
| [`redo_procedure.md`](redo_procedure.md) | Step-by-step build plan for the redo, mirroring the original `planning_files/phic_psi_implementation_plan_v4.md`'s structure, with every step marked `[UNCHANGED]`/`[CORRECTED]`/`[NEW]` against the corrected `2φc±2ψ` combo formula. |
| [`formulae_reference.md`](formulae_reference.md) | Exact physics/loss formulae used at each step (vector algebra, combo construction, isotropic circular loss, magnitude penalty, toy antenna-pattern model, curriculum weight derivation, reconstruction branch-handling). Model architectures given as a bare outline only, no per-layer math. |
| [`comments.md`](comments.md) | External review of the redo package (2026-09-14) — confirms the core `2φc±2ψ` fix is internally consistent throughout, hand-verifies the A.5 self-check, and proposed the A.8 branch-ambiguity re-derivation. The branch-*count* part held up; the closed-form parity-filter part did not (see below). |
| [`curriculum.py`](curriculum.py) | Step 1.2 toy antenna-pattern model + Jacobian-condition-number sweep + `derive_w_iota`, copied from `experiments/phic_psi_poc/curriculum.py` with `h_plus`/`h_cross` corrected to `2*phi_c` (A.5). |
| [`transform_utils.py`](transform_utils.py) | Vector algebra primitives, copied from `experiments/phic_psi_poc/transform_utils.py` with `double_angle`/`tf_double_angle` added (A.1.5) and `reconstruct_phic_psi` corrected for the `2φc±2ψ` combo (A.8) — brute-force only, no parity shortcut (see below). |

## What's still open?

This is a **planning-stage-only package** — no trainer/config code has been written and no training has run yet. `curriculum.py` and `transform_utils.py` (above) are the first two implementation files, built and numerically self-tested this session.

**Resolved by the 2026-09-14 review** (`comments.md`): A.8's reconstruction branch-handling, previously flagged "re-derive before coding," now has confirmed branch counts — ψ is 2-fold ambiguous, φc is 4-fold ambiguous, exactly 4 of the naive 8 candidate pairs are always jointly consistent. Also noted, not a bug: A.5's toy `F_plus`/`F_cross` uses the opposite handedness convention (`e^{+2iψ}`) from some literature conventions — washes out, self-check passes regardless.

**Same-day self-correction (2026-09-14, after implementing `transform_utils.py`):** the review's proposed closed-form parity rule for picking the 4 consistent candidates without brute force (`k≡j mod 2`) was implemented, tested, and immediately falsified — a 20,000-trial numerical sweep showed the parity pattern is ~50/50 and data-dependent (depends on an integer `arctan2`'s mod-2π reduction destroys), not fixed. Retracted in `formulae_reference.md` §A.8 and `redo_procedure.md` §2.6, kept on the record rather than silently erased, per this repo's frozen-vs-living-docs convention. The reconstruction code brute-forces all 8 candidates every time — verified against 5,000 random trials, 100% recovery.

Next steps, in order:
1. Implement `trainer.py` (SumDiffTrainer subclass) and the config YAMLs per `redo_procedure.md` (§2.4–§3.5), building on the now-complete `transform_utils.py`/`curriculum.py`.
2. Add the epoch-N periodic-checkpoint callback to `src/gwml/training/train.py`'s `_build_callbacks` (the one shared-file change this redo requires, per `redo_procedure.md` §0).
3. Run the CPU-only §1.1/§1.2 prerequisite checks (sign/combination re-confirmation via a redo `prereq_checks.py`, `w(ι)` re-derivation via `curriculum.py.derive_w_iota`) locally before any GPU work.
4. Hand the full 7-architecture Round-1-equivalent training sweep (§4) to the lab GPU machine — this machine (T530) is CPU-only, per this repo's `CLAUDE.md`.
5. Run the down-select re-validation step (§4) and document, with a fresh comparison table, whether the original 4-model certified set (poc_a, poc_b, tcn, cnn_attention) still holds under the corrected formula.

The original (wrong-formula) investigation is untouched at `experiments/phic_psi_poc/`, and fully preserved as a static snapshot on the `archive/phic-psi-poc-v1` branch.
