# φc/ψ Degeneracy PoC — Redo — Experiment Index

**Branch**: `poc/phic-psi-degeneracy-redo`
**Last updated**: 2026-09-13

---

## Files

| File | Description |
|---|---|
| [`redo_procedure.md`](redo_procedure.md) | Step-by-step build plan for the redo, mirroring the original `planning_files/phic_psi_implementation_plan_v4.md`'s structure, with every step marked `[UNCHANGED]`/`[CORRECTED]`/`[NEW]` against the corrected `2φc±2ψ` combo formula. |
| [`formulae_reference.md`](formulae_reference.md) | Exact physics/loss formulae used at each step (vector algebra, combo construction, isotropic circular loss, magnitude penalty, toy antenna-pattern model, curriculum weight derivation, reconstruction branch-handling). Model architectures given as a bare outline only, no per-layer math. |

## What's still open?

This is a **planning-stage-only package** — no redo code has been written and no training has run yet.

Next steps, in order:
1. Implement `transform_utils.py`, `curriculum.py`, `trainer.py`, and the config YAMLs per `redo_procedure.md` (§0–§3.5).
2. Add the epoch-N periodic-checkpoint callback to `src/gwml/training/train.py`'s `_build_callbacks` (the one shared-file change this redo requires, per `redo_procedure.md` §0).
3. Run the CPU-only §1.1/§1.2 prerequisite checks (sign/combination re-confirmation, `w(ι)` re-derivation) locally before any GPU work.
4. Hand the full 7-architecture Round-1-equivalent training sweep (§4) to the lab GPU machine — this machine (T530) is CPU-only, per this repo's `CLAUDE.md`.
5. Run the down-select re-validation step (§4) and document, with a fresh comparison table, whether the original 4-model certified set (poc_a, poc_b, tcn, cnn_attention) still holds under the corrected formula.

The original (wrong-formula) investigation is untouched at `experiments/phic_psi_poc/`, and fully preserved as a static snapshot on the `archive/phic-psi-poc-v1` branch.
