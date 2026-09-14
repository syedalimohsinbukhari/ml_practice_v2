# φc/ψ Degeneracy PoC — Redo — Step-by-Step Procedure

Companion to [`formulae_reference.md`](formulae_reference.md) (exact math, use as given, do not re-derive ad hoc) and the original investigation at `experiments/phic_psi_poc/` (kept intact, not modified — see `archive/phic-psi-poc-v1` for the exact pre-redo snapshot).

## Why this document exists

The original PoC's entire combo_A/combo_B design was built on the combination `φc ± 2ψ`. A reviewer flagged this as wrong. Investigation traced `coa_phase`'s actual injection convention through this dataset's generator (`src/gwml/gen_py_data_pipeline.md`, and the sibling generator scripts `ml-gw-search/mlgwsc-1/gen.py:202` / `ml-gw-search/extended_mass/gen.py:179`, which pass `coa_phase` straight into `pycbc.waveform.get_td_waveform`) and confirmed: `coa_phase` is PyCBC/LALSimulation's **orbital** reference phase, and for the dominant-(2,2)-mode-only approximant this dataset uses (IMRPhenomD — already confirmed in `experiments/phic_psi_poc/NOTES.md`'s v2 adversarial-review section), the strain depends on it through `e^{i·2·coa_phase}` — doubled, exactly like ψ already is. The correct combination is `2φc ± 2ψ = 2(φc±ψ)`. Full evidence chain: `experiments/phic_psi_poc/diagnostic_log.md`, dated entry 2026-09-09.

Every combo_A/combo_B result in the original investigation (Round 1, Runs 6–9b, λ-retune, verification Sections A–E) tested `φc±2ψ` — the wrong target. That null-result conclusion is therefore **unestablished**, not confirmed wrong, just no longer supported. This document is the build plan for redoing the investigation under the corrected formula.

**What this document is not:** it does not itself contain the redo's Python code, and no training has happened yet. It mirrors the original `planning_files/phic_psi_implementation_plan_v4.md`'s structure so the redo is traceable step-for-step against the original, and marks every step:

- **[UNCHANGED — reuse]**: the original conclusion/artifact still holds, cite and reuse it, don't redo the work.
- **[CORRECTED — redo needed]**: this step's original result depended on the wrong formula and must be rerun from scratch.
- **[NEW]**: didn't exist in the original plan, added for this redo specifically.

## 0. Isolation setup

- **Branch:** `poc/phic-psi-degeneracy-redo` (created off `master`, post-merge of `paper-helper`).
- **Folder:** `experiments/phic_psi_poc-redo/` (this directory).
- **[UNCHANGED]** Same isolation rule as the original: never modify `src/gwml/heads_spec.py`, `DEFAULT_HEADS`, or `MultiHeadTrainer` in place. Subclass/extend locally, same pattern `SumDiffTrainer` used in the original (`experiments/phic_psi_poc/trainer.py`).
- **[CORRECTED, per explicit decision]** The φc-doubling fix lives entirely inside this redo's own `trainer.py`/`curriculum.py` — `z_φc` gets self-complex-multiplied (`double_angle`, see formulae ref A.1.5) before combining with `z_ψ`. `src/gwml/heads_spec.py`'s `coa_phase` PERIODIC period stays `2π`, untouched. Why local, not global: `coa_phase`'s period=2π head declaration is correct as a *data encoding* choice (the injected value genuinely spans `[0,2π)`) — the physics correction belongs in how the PoC's own combo construction *uses* that vector, not in the head's own definition, which other experiments/heads elsewhere in the codebase may depend on unchanged.
- **[NEW] One shared-file exception, deliberately scoped and justified:** `src/gwml/training/train.py`'s `_build_callbacks` (train.py:86-136) currently only saves `best.weights.h5` (via `ModelCheckpoint(save_best_only=True)`) and `final.weights.h5` (saved separately at the end of `run_experiment`/`run_poc_experiment`) — no epoch-N snapshots exist anywhere in this codebase today. This repo's `CLAUDE.md` already mandates fixing this ("Training runs must save epoch-N snapshots... Hook: `_build_callbacks`... land this with the next training campaign") — this redo *is* that next campaign. Add a second callback (custom `on_epoch_end` checking `epoch % 10 == 0`, since Keras' `ModelCheckpoint` string `save_freq` only supports `"epoch"`, not an N-epoch stride) that writes `epoch_{epoch:03d}.weights.h5`. This is additive — existing best/final behavior is untouched — and since `train_poc.py` imports `_build_callbacks` directly from `gwml.training.train`, every future experiment gets it too, not just this redo.

## 1. Prerequisite checks

- **1.1 [CORRECTED]** Re-run the empirical sign/combination check. The original Step 1.1 tested which of `φc+2ψ`, `φc−2ψ` correlates more cleanly with ground truth, conditioned on `sign(cos ι)`, and found `combo_B` (φc−2ψ) well-constrained at `cos ι > 0`. **That result is void** — it validated the wrong candidate pair entirely, not just the wrong sign, so it cannot be reused even for direction. Rerun the same procedure testing `2φc+2ψ` vs `2φc−2ψ` instead.
  - *Why redo, not reuse:* the empirical check measured correlation strength for `φc+2ψ`-shaped candidates. A `2φc+2ψ`-shaped candidate is a different function of the same two labels; there's no way to infer its correlation from the old numbers.
- **1.2 [CORRECTED]** Re-derive the curriculum weight `w(ι)`. Fix `curriculum.py`'s `h_plus`/`h_cross` to `cos(2Φ+2φc)` / `sin(2Φ+2φc)` (formulae ref A.5), rerun the Jacobian condition-number sweep (`_jacobian_condition_number`, `derive_w_iota`), regenerate `sweep_1_1_ratio_vs_iota.{csv,png,pdf}` fresh. CPU-only, ~30–70 minutes per the original's own timing note. The harness self-check (R,δ constant along the degenerate line `2φc+2ψ=const` at ι=0) must pass under the corrected formula before anything downstream is trusted — this is the same discipline the original harness check used, just re-pointed at the right line in (φc,ψ)-space.
  - *Why redo, not reuse:* `w(ι)`'s shape comes from the condition number of the `(φc,ψ)→(R,δ)` Jacobian, which is built directly from `h_plus`/`h_cross`. Changing those functions changes the Jacobian, which changes `w(ι)`'s shape — the old `w=1−cos²ι` default and the old empirical fit were both derived against the wrong antenna-pattern model.
- **1.3 [UNCHANGED — reuse]** Data representation: raw whitened time-domain strain `(N, 4096, 2)`, detector order `[h1, l1]` — confirmed via `loader.py`, not formula-dependent. Cite the original confirmation directly, no rerun needed.
- **1.4 [UNCHANGED — reuse]** True inclination accessible at batch level via `y_true["inclination"][:, 1]` = cos(ι_true) — not formula-dependent.
- **1.5 [UNCHANGED — reuse]** Curriculum mechanism: static per-sample weight `w(ι_true)` throughout training, no epoch-based scheduling — this was a design decision independent of the combo formula, still the right call.
- **1.6 [CORRECTED — cheap rerun]** Re-histogram the `cos ι` population on the current dataset snapshot (face-on fraction, edge-on fraction) rather than assume the original 28.7%/32.7% figures still hold exactly. This one isn't formula-dependent either, but it's cheap (CPU-only, minutes) and worth reconfirming fresh rather than citing a run from two months prior — the dataset file itself hasn't been shown to be byte-identical since.

## 2. Code components

- **2.1 [UNCHANGED]** Reactivate the `polarization_angle` head (add to the active heads list) — trivial, no formula dependency.
- **2.2 [CORRECTED]** Vector transform utilities: reuse `normalize_unit`/`complex_mul`/`complex_mul_conj` from the original `transform_utils.py` unchanged — the underlying vector algebra was never wrong, only how it was applied. Add one new helper, `double_angle(z) = complex_mul(z, z)`, which turns a unit vector at angle `θ` into one at angle `2θ`. This is the mechanism for turning `z_φc` (angle φc, unchanged period-2π head) into a `2φc`-angle vector before combining with `z_ψ`.
- **2.3 [CORRECTED]** Combo construction (formulae ref A.2): `combo_A = complex_mul(double_angle(z_φc), z_ψ)` → angle `2φc+2ψ`; `combo_B = complex_mul_conj(double_angle(z_φc), z_ψ)` → angle `2φc−2ψ`. Same isotropic circular loss `L = 1 − dot(z_pred, z_true)` as the original — that loss function was never implicated in the bug, it operates correctly on whatever vectors it's given. combo_A/combo_B labels are provisional again pending 1.1's redone empirical result — **do not** carry forward `well_constrained_combo: combo_B` from the original config; it was determined for the wrong pair.
- **2.4 [UNCHANGED, but configured correctly from day 1]** `log_var_clamp`/`combo_log_var_clamp` defaults (3.0), uncertainty weighting (`weighting: uncertainty`) — same as the original's Run 6 fix, just present from the start instead of discovered partway through.
- **2.5 [UNCHANGED, but from day 1 — no rediscovery]** `activation="linear"` for all PERIODIC heads, and `magnitude_penalty_lambda=0.01` from the very first training run. The original investigation spent Runs 1–6 discovering, in order: tanh saturation at random init (root cause found in Run 3, `diagnostic_log.md`), then — after fixing that — a second pathology where `normalize_unit`'s gradient blows up/vanishes because the isotropic circular loss doesn't constrain `|v_raw|` (found in the post-fix retraining, `tanh_to_linear_postmortem.md`), fixed by the explicit magnitude penalty `λ·(|v_raw|−1)²` (Run 6). Both root causes are established facts now, not open questions — the redo starts with both fixes already in place and cites the original diagnostic runs as evidence, rather than re-running that debugging arc.
- **2.6 [CORRECTED — derived and verified, 2026-09-14]** Inference-time reconstruction. The original plan's branch-ambiguity analysis (`planning_files/phic_psi_implementation_plan_v4.md` §2.6/A.4) found "2-fold for φc, 4-fold for ψ" for `φc±2ψ`'s reconstruction — a property of that formula, not reusable here. Under `2φc±2ψ`, the fold counts swap: see `formulae_reference.md` §A.8 for the full re-derivation (reviewed and hand-verified 2026-09-14) — **ψ is 2-fold ambiguous** (candidates `ψ₀`, `ψ₀+π/2`), **φc is 4-fold ambiguous** (candidates `φc₀+k·π/2`, k=0..3), and — the part worth flagging explicitly for implementation — **the two ambiguities are not independently combinable**: only combinations where φc's branch index `k` and ψ's branch index `j` share parity (`k≡j mod 2`) reproduce both `combo_A` and `combo_B`, cutting the naive `4×2=8` candidate pairs down to the true 4. Write this parity rule directly into candidate generation as a closed-form filter (not brute force); keep the original's re-encode-and-check-against-both-combos filter alongside it as a cheap validation cross-check, same role it played in the original.

## 3. Standalone validation script

Same purpose as the original: confirm `combo_A`/`combo_B` reconstruct correctly under `2φc±2ψ`, confirm `w(ι)` behaves correctly at the extremes (→0 at ι=0, →max at ι=π/2) under the corrected antenna-pattern model, *before* wiring anything into the trainer. Note: the original plan referenced `invert_test2.py`/`joint_invert.py`/`degeneracy_check.py` as sources to reuse for this — per the `LINKED.md` audit, none of those three files actually exist anywhere in this repo. This redo's validation script is written from scratch, not adapted from files that were never committed.

## 3.5 Minor implementation notes

Carried over unchanged from the original's §3.5: reuse the same normalization epsilon (`1e-8`) already used in `normalize_unit`/`_magnitude_penalty`; reconfirm the true-ι batch plumbing still works exactly as before (it does — 1.4 is unchanged).

## 4. Training run plan

- **Round-1-equivalent [CORRECTED, full 7-architecture scope]:** the original Round 1 covered 5 backbones (tcn, cnn_baseline, cnn_attention, inception_time, resnet1d) plus poc_a/poc_b on tcn, all in baseline mode (circular loss on individual φc/ψ, no combo). This redo repeats that full 7-config sweep, with tanh→linear, the magnitude penalty, and periodic epoch checkpoints all present from the first run — there is no "Round 1 invalidated by tanh, redo Round 1" arc this time, because both fixes are already known and built in from day 1.
- **[NEW] Explicit down-select re-validation step.** The original investigation narrowed from 7 models to 4 (poc_a, poc_b, tcn, cnn_attention) before the magnitude-penalty/combo phase (Run 7 onward). The stated reasons at the time were Round-1 numbers for the other three — cnn_baseline mchirp R²=0.63, merger_time R²=0.24; inception_time merger_time R²=−0.001 — but `NOTES.md`'s own 2026-07-24 consistency-audit note flags these as probably stale: they predate the tanh→linear fix, and the closest surviving post-fix artifacts for the same architectures show markedly better numbers (cnn_baseline mchirp R²≈0.83, merger_time R²≈0.82; inception_time merger_time R²≈+0.0001). Since this redo runs the full 7-architecture sweep *with the fix already in place from run 1*, this step re-evaluates the down-select criteria on genuinely comparable numbers and **produces an explicit comparison table stating whether the same 4-model set still holds, changes, or should be reconsidered** — this is not assumed either way going in, per direct instruction.
- **Combo-phase runs:** magnitude-penalty (λ=0.01) training on whichever model set the down-select step confirms, using the corrected `combo_A`/`combo_B` construction from §2.3.
- **λ ablation/retune:** kept as a step, but reframed as **confirmatory**, not exploratory — the original λ=0/0.01/0.05/0.10 sweep and its "λ alone is insufficient" verdict for tcn/coa_phase and poc_a/pol_angle were about std_ratio stabilization, a mechanism orthogonal to which combination is degenerate. It's plausible the same λ values behave the same way under the corrected target; this step checks that rather than assuming it, but isn't expected to need a fresh multi-value sweep from scratch unless the corrected-formula results surprise us.

## 5. Evaluation checklist

Same structure as the original plan's §5 checklist:

- [ ] Sum-combo head (whichever of `combo_A`/`combo_B` 1.1 confirms well-constrained) trains to a sane R²/angular accuracy, not just "better than the other combo."
- [ ] Diff-combo head's R² is compared against the *baseline* run's raw φc/ψ individual-head R² (Run-1-equivalent, §4), not against zero — same reasoning the original used: zero is not automatically the right null.
- [ ] Residuals binned jointly by `cos ι` and by the inclination-head's own error, matching the original's inclination-noise-floor stratification method (`experiments/phic_psi_poc/inclination_stratification.py`, reusable as-is once pointed at the redo's checkpoints).
- [ ] Optional Run-C equivalent: a fixed `ι=0` dataset slice, confirming the diff-combo collapses to near-zero under guaranteed-exact degeneracy — same sanity-check role as the original's (never-run) Run C.
- [ ] All results, including negative/null ones, documented in this redo's own `NOTES.md` (to be created when runs actually start) — full-trajectory, not endpoint-only, per this repo's `CLAUDE.md` rule.

## 6. Explicitly out of scope for this redo

- No architecture changes beyond reactivating the `polarization_angle` head.
- No epoch-based curriculum scheduling (static per-sample `w(ι)` only, per §1.5).
- No antenna-pattern/sky-position refinement of `w(ι)` beyond the doubling fix itself.
- No modification of `src/gwml/heads_spec.py`'s `coa_phase` period (per §0's local-fix decision).
- No merge of `poc/phic-psi-degeneracy-redo` into `master`/`main` until an explicit go/no-go call, same rule the original PoC operated under.

## 7. Reference formulae

See [`formulae_reference.md`](formulae_reference.md) for the exact formulae used at each step above — use as given, do not re-derive ad hoc, same discipline the original plan's Appendix A required.
