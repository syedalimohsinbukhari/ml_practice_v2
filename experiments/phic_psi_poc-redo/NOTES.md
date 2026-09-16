# φc/ψ Degeneracy PoC — Redo — Running Notes

Companion to [`redo_procedure.md`](redo_procedure.md) (step-by-step plan) and [`formulae_reference.md`](formulae_reference.md) (exact math). The original investigation at `experiments/phic_psi_poc/` used the wrong combo formula (`φc±2ψ`); this redo uses the corrected `2φc±2ψ` (see `experiments/phic_psi_poc/diagnostic_log.md`'s 2026-09-09 dated entry for the full evidence chain). Original preserved untouched, and separately snapshotted on `archive/phic-psi-poc-v1`.

## Setup

- **Branch:** `poc/phic-psi-degeneracy-redo` (off `master`)
- **Created:** 2026-09-13
- **Goal:** redo the φc/ψ degeneracy PoC under the corrected combo formula; determine whether the original's null result (network learns nothing) still holds, or whether it was an artifact of testing the wrong target.

## Step 1 — Prerequisite checks (2026-09-14)

Ran `prereq_checks.py` at full resolution (n_sky=200, n_iota_sweep=50/regime, n_iota_w=200, n_boot=5000 — same settings as the original's own run). CPU-only, completed in ~3 minutes (faster than the original's 30–70 min estimate — likely machine/load-dependent, not a methodology difference). Full output: [`prereq_checks_output/prereq_checks_20260914_160834.log`](prereq_checks_output/prereq_checks_20260914_160834.log).

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

## Round 1 — full 7-architecture sweep, first result under corrected formula (2026-09-15)

All 7 configs trained on the lab GPU machine (80 epochs each). Runs (latest timestamp used where a config was rerun):

| Config | run_dir | val_r2_mchirp | val_r2_merger_time | val_r2_snr | φc/ψ signal (val, last-10 mean) |
|---|---|---|---|---|---|
| `config_baseline.yaml` (Run A, tcn) | `phic_psi_poc_redo_a/20260915_051948` | 0.961 | 0.916 | 0.787 | coa_phase=1.0155, pol_angle=1.0009 |
| `config_poc.yaml` (Run B, tcn) | `phic_psi_poc_redo_b/20260915_061008` | 0.960 | 0.921 | 0.785 | **combo_A=0.9999, combo_B=0.9895** |
| `config_tcn.yaml` | `phic_psi_redo_tcn/20260915_064106` | 0.961 | 0.919 | 0.790 | coa_phase=1.0093, pol_angle=1.0069 |
| `config_cnn_baseline.yaml` | `phic_psi_redo_cnn_baseline/20260915_054959` | 0.833 | 0.818 | 0.699 | coa_phase=1.0081, pol_angle=0.9995 |
| `config_cnn_attention.yaml` | `phic_psi_redo_cnn_attention/20260915_054256` | 0.922 | 0.904 | 0.720 | coa_phase=1.0012, pol_angle=1.0088 |
| `config_inception_time.yaml` | `phic_psi_redo_inception_time/20260915_055532` | 0.906 | **0.000** | 0.712 | coa_phase=1.0084, pol_angle=0.9940 |
| `config_resnet1d.yaml` | `phic_psi_redo_resnet1d/20260915_063256` | 0.897 | 0.870 | 0.757 | coa_phase=1.0063, pol_angle=1.0037 |

### Down-select re-validation — original 4-model certified set holds

Cross-checked against the original's own post-tanh-fix numbers (the closest apples-to-apples comparison, per its 2026-07-24 consistency-audit note): `cnn_baseline` mchirp/merger_time here (0.833/0.818) match the original's post-fix figures (≈0.83/0.82) almost exactly; `inception_time` merger_time here (0.000) matches the original's (≈+0.0001) almost exactly. This is a strong internal-consistency signal — the redo's baseline-mode pipeline reproduces the same architecture-level behavior the original found, on the same held-out weaknesses.

**Conclusion: the original 4-model certified set (poc_a, poc_b, tcn, cnn_attention) still holds.** `cnn_baseline` and `inception_time` remain measurably weaker (inception_time still cannot learn `merger_time` at all — an architecture limitation, not a formula artifact, reproduced independently here). `resnet1d` is intermediate (0.897/0.870/0.757) but doesn't clear tcn/cnn_attention. Carrying forward poc_a(tcn baseline)/poc_b(tcn poc)/tcn/cnn_attention for any further phase is justified by fresh evidence, not assumed.

### ⚠ Central result: the corrected combo formula does not, on its own, produce learning

`config_poc.yaml` (Run B, TCN, `2φc+2ψ`/`2φc−2ψ`) is the run that actually exercises the corrected formula. **`circular_loss_combo_A`/`circular_loss_combo_B` stay flat at ~0.98–1.01 across all 80 epochs, both train and validation** — full trajectory checked (min/max/mean/last-10-mean), no hidden mid-training excursion. Positive controls in the *same run* confirm this isn't a broken harness (mchirp R²=0.960, merger_time R²=0.921, snr R²=0.785 — all healthy).

**Directly compared against the original's equivalent run** (`runs/archive_phic_psi_poc_v1/phic_psi_poc_b/20260720_213202`, wrong formula `φc+2ψ`/`φc−2ψ`, Run 7, λ=0.01): `val_circular_loss_combo_A/B` last-10 means there were **0.9989/0.9913** — essentially indistinguishable from this redo's **0.9999/0.9895**. Fixing the combo formula did not change the outcome.

**This does not confirm the degeneracy is fundamental — it specifically fails to confirm the hypothesis that the original's null result was an artifact of testing the wrong combo pair.** Two things are worth flagging as not yet ruled out, both direct parallels to open threads the original investigation had to chase down before it could trust its own null result:

- **std_ratio for the raw coa_phase/polarization_angle vectors settles low** (val last-10 means: coa_phase≈0.64, polarization_angle≈0.25 — see `runs/phic_psi_poc_redo_b/20260915_061008/history.csv`), well below the healthy ~1.0 band, despite `magnitude_penalty_lambda=0.01` being active from epoch 0. The original hit exactly this kind of std_ratio pathology for specific head/model combinations and needed a dedicated λ-retune investigation (Runs 8–9b) before treating the result as interpretable — that diagnostic discipline (the `preregistration_lam_retune.md`-style Step-0 interpretability gate) hasn't been applied to this redo run yet.
- **The uncertainty weights are climbing, not the loss**: `weight_combo_A`/`weight_combo_B` (=exp(−log_var)) rise from ~1.1 to ~1.36 over training while circular loss never moves — the trainer is growing *more confident* in heads that aren't learning. This is the same log_var-runaway pattern `diagnostic_checks.py`'s Check 3 (`check_logvar_trajectory`) was built to catch in the original investigation.

**Verdict at this stage: UNINTERPRETABLE, not NULL** — same distinction the original's own preregistration framework insisted on (a std_ratio/log_var gate failing means the result can't be read as evidence either way yet, not that it's evidence of no learning). Next step, mirroring the original's own methodology rather than skipping ahead of it: run the redo's diagnostic/log_var-trajectory checks and, if the gate fails, a λ-retune pass, before drawing any conclusion about whether the corrected degeneracy hypothesis holds.

## Step 0 diagnostic — std_ratio gate result: FAIL, UNINTERPRETABLE (2026-09-15)

Ran [`diagnostic_logvar_gate.py`](diagnostic_logvar_gate.py) — the Step 0 gate from `experiments/phic_psi_poc/preregistration_lam_retune.md`, thresholds copied verbatim (`[0.5,2.0]` healthy band, <10% of last 40 epochs unhealthy, trend within ±0.005/ep). CPU-only, reads `history.csv` only, no model loading. Full output: [`diagnostic_output/diagnostic_logvar_gate_20260915_101447.{log,md}`](diagnostic_output/), [`diagnostic_output/logvar_gate_trajectories.{png,pdf}`](diagnostic_output/).

**Mechanical verdict: GATE FAILS on both heads, for both `poc_redo_b` (poc mode) and `poc_redo_a` (baseline mode):**

| Run | Head | frac unhealthy (last 40 ep) | trend/ep | final std_ratio | Gate |
|---|---|---|---|---|---|
| poc_redo_b | coa_phase | 0.225 | +0.00638 | 0.583 | FAIL |
| poc_redo_b | polarization_angle | 0.850 | −0.00402 | 0.251 | FAIL |
| poc_redo_a | coa_phase | 0.750 | −0.00721 | 0.407 | FAIL |
| poc_redo_a | polarization_angle | 1.000 | +0.00076 | 0.374 | FAIL |

**Per `preregistration_lam_retune.md`'s own decision table: `phic_psi_poc_redo_b`'s flat `circular_loss_combo_A`/`combo_B` result is UNINTERPRETABLE, not a confirmed null.** `|v|`-space (the raw φc/ψ vectors) hasn't stabilized under `magnitude_penalty_lambda=0.01`, so the flat loss can't yet be trusted as evidence the corrected formula fails too.

**Correction to the 2026-09-15 Round-1 write-up above:** that entry flagged `weight_combo_A`/`weight_combo_B` as "still climbing" as a second red flag. This script checked it mechanically (same late-window trend test) and that read doesn't hold up: `late_trend` for both is ≈+0.0003–0.0004, well inside the same ±0.005/ep threshold — the weight rises early (epochs 0–~10) then plateaus, it is *not* still rising in the diagnostic window. Leaving the original entry as written (frozen, dated) rather than editing it, per this repo's convention — this paragraph is the correction of record.

**New finding worth the emphasis it deserves: the SAME gate failure shows up in `poc_redo_a` (baseline mode), which never touches the combo transform at all.** This points at something upstream of the corrected-formula question entirely — most likely `magnitude_penalty_lambda=0.01` simply being insufficient for TCN at this head/mode combination, matching the original's *own* finding almost exactly ("tcn coa_phase still declining at λ=0.01, Run 7") before it needed a dedicated λ-retune (Runs 8–9b) to even attempt an interpretable read. The redo inherited this same open problem by keeping λ=0.01 "from day 1" rather than rediscovering it — expected, not a new bug, but it means the gate failure is very unlikely to be specific to the formula fix.

**Next: a λ-retune pass, mirroring the original's Runs 8–9b**, pre-registering the criterion first (same discipline `preregistration_lam_retune.md` itself insists on) before running it — not deciding the threshold after seeing results.

## λ=0.05 retune — Step 0 gate result: FAIL again, UNINTERPRETABLE (2026-09-15)

`config_lam005_retune_b.yaml`/`config_lam005_retune_a.yaml` trained on the lab GPU machine (80 epochs each, same seed/schedule as Round 1). Runs: `phic_psi_lam005_retune_b/20260915_154203`, `phic_psi_lam005_retune_a/20260915_175801`. Reran [`diagnostic_logvar_gate.py`](diagnostic_logvar_gate.py) with the `ROUNDS` dict extended to include this round (script already had the entry from day 1). Full output: [`diagnostic_output/diagnostic_logvar_gate_20260915_145042.{log,md}`](diagnostic_output/), plots updated in place at [`diagnostic_output/logvar_gate_trajectories.{png,pdf}`](diagnostic_output/).

**Mechanical verdict: GATE FAILS on all four readings again, at 5x the original penalty:**

| Run | Head | frac unhealthy (last 40 ep) | trend/ep | final std_ratio | Gate |
|---|---|---|---|---|---|
| lam005_retune_b | coa_phase | 1.000 | +0.00093 | 0.315 | FAIL |
| lam005_retune_b | polarization_angle | 0.725 | +0.00679 | 0.531 | FAIL |
| lam005_retune_a | coa_phase | 0.350 | −0.00561 | 0.522 | FAIL |
| lam005_retune_a | polarization_angle | 0.425 | +0.00383 | 0.572 | FAIL |

**Per `preregistration_lam_retune.md`'s own decision table: still UNINTERPRETABLE, not NULL.** The gate did not clear at λ=0.05 for either the primary run or its control — `poc_redo_b`'s `combo_A`/`combo_B` result cannot yet be read as evidence in either direction. Worth noting for the record: `lam005_retune_a`'s (control) frac-unhealthy numbers did improve somewhat over its own λ=0.01 baseline (coa_phase 0.750→0.350, polarization_angle 1.000→0.425) — the penalty is doing *something* — but neither head clears the <10% threshold, and `lam005_retune_b`'s coa_phase is worse on frac-unhealthy (0.225→1.000) even though its final std_ratio and trend aren't dramatically different. Not a case for reading tea leaves on partial improvement — the gate is binary per the preregistration, and it fails.

**Per the preregistration's pre-committed fallback order ("Gate fails at λ=0.05 → try λ=0.10 before drawing any conclusion"): built and CPU-verified `config_lam010_retune_b.yaml`/`config_lam010_retune_a.yaml`** (same structure, `magnitude_penalty_lambda: 0.10`, both wiring-checked end-to-end — forward pass, loss, gradient step, no `None` grads). This is the last fallback step the preregistration commits to; if it also fails, the preregistration's own instruction is to report "λ alone insufficient for this architecture/head, post-formula-fix" rather than open a new unplanned round.

## λ=0.10 retune — Step 0 gate result: FAIL, λ-retune line of attack exhausted (2026-09-16)

`config_lam010_retune_b.yaml`/`config_lam010_retune_a.yaml` trained on the lab GPU machine (80 epochs each). Runs: `phic_psi_lam010_retune_b/20260915_181803`, `phic_psi_lam010_retune_a/20260916_020049`. Reran [`diagnostic_logvar_gate.py`](diagnostic_logvar_gate.py) with a `"λ=0.10 (retune)"` entry appended to `ROUNDS` (per its own module docstring: "just appending one more entry"). Full output: [`diagnostic_output/diagnostic_logvar_gate_20260916_085902.{log,md}`](diagnostic_output/), plots updated in place.

**Mechanical verdict: 3 of 4 readings still FAIL — one, for the first time, PASSES:**

| Run | Head | frac unhealthy (last 40 ep) | trend/ep | final std_ratio | Gate |
|---|---|---|---|---|---|
| lam010_retune_b | coa_phase | **0.000** | +0.00011 | 0.930 | **PASS** |
| lam010_retune_b | polarization_angle | 0.950 | +0.00326 | 0.421 | FAIL |
| lam010_retune_a | coa_phase | 0.425 | −0.00453 | 0.546 | FAIL |
| lam010_retune_a | polarization_angle | 0.650 | +0.00220 | 0.559 | FAIL |

**Per `preregistration_lam_retune.md`'s own Scope section ("All four must pass... a single failing head/run keeps the whole round UNINTERPRETABLE — the combo depends on both source vectors jointly"): mechanically still UNINTERPRETABLE**, not a partial pass. Full trajectory checked for both new runs (not just the last-40-epoch gate window, per this repo's full-trajectory rule) — `val_circular_loss_combo_A`/`combo_B` in `lam010_retune_b` stay flat at 0.9833–1.0133 across all 80 epochs (last-10-mean 0.9963/1.0000), no hidden mid-training excursion; same flat-null pattern as every round so far.

**This is the last λ value `preregistration_lam_retune.md`'s fallback order commits to.** Per that document's own Step 0 section ("Gate fails at λ=0.10 too → report as 'λ alone insufficient for this architecture/head, post-formula-fix'... not counted toward a null tally or as counter-evidence"), the correct call here is exactly that: **λ alone (0.01 → 0.05 → 0.10) is insufficient to stabilize `|v|`-space for TCN's coa_phase/polarization_angle heads on this formula, in both poc and baseline mode.** This is a negative result about the retune mechanism, not about the corrected-formula degeneracy hypothesis itself — the central question stays formally unanswered under this line of attack, exactly as the preregistration anticipated it might.

**Correction to `diagnostic_logvar_gate.py`'s own generic FAIL message:** the script prints "Next step: next fallback lambda (0.10)" on any FAIL verdict — that text doesn't know when the round under test *is* the last committed fallback, and reading it literally here would incorrectly suggest a λ=0.15 (or similar) round should be tried next. Per the preregistration's own decision table, it should not be — trying further ad hoc λ values without a new pre-registration would be exactly the after-the-fact threshold-picking this repo's discipline exists to prevent. Flagging this here (frozen, dated) rather than editing the script's hardcoded string, same convention as the earlier `weight_combo_A/B` correction.

**One specific result worth flagging per-case, not averaged away:** `lam010_retune_b`'s coa_phase is the first reading across all three λ rounds (0.01/0.05/0.10) to clear the gate outright (frac_unhealthy=0.000, well inside the trend threshold). `polarization_angle` in the same run moved the opposite direction over training (first-10-epoch mean 0.885 → last-10-epoch mean 0.426, i.e. it got *worse*, not better, under λ=0.10). Both facts are reported as-is: one head in one run is healthy at λ=0.10; the other three readings are not; the pre-registered rule treats this as a single FAIL, and that reporting choice is followed here rather than reopened.

### ⚠ Correction (2026-09-16, same day) — the coa_phase "recovery" above does not hold up against the scatter plots; std_ratio alone is not a substitute for eyeballing predicted-vs-true

The claim in an earlier draft of this entry — that `lam010_retune_b`'s coa_phase clearing the gate was "a real recovery, not an artifact of the window" — does not survive a look at the training-time scatter plots already written to `runs/phic_psi_lam010_retune_b/20260915_181803/scatter/epoch_{0005,0080}.png` (per `_build_callbacks`' scatter-subset hook, no extra GPU work needed to view these). This is exactly the failure mode this repo's diagnostics are supposed to catch, caught one level too late — a good aggregate statistic (std_ratio≈0.93, "healthy") was taken at face value without checking the actual predicted-vs-true shape underneath it.

**What the scatter shows, at both epoch 5 and epoch 80 (same qualitative shape throughout — not something that emerged late):** `φc`'s "predicted (unwrapped) vs. true" panel is not a diagonal cloud — it is **two flat, horizontal bands** (≈−1.2 and ≈+5.2 rad, which are the same angle mod 2π), each covering roughly half the true-value range, with **no slope within either band**. The model is not predicting a per-sample phase at all; it is emitting a value close to one constant direction, and the branch-unwrapping logic (per `formulae_reference.md` A.8) is splitting that single constant into two apparent "bands" depending on which side of the true value it falls on. `val_std_ratio_coa_phase`'s climb from 0.62 (epoch 0) through a spike to 1.66 (epoch 10) down to 0.25 (epoch 20) and settling ≈0.90–0.93 from epoch 40 on is consistent with a **fixed, collapsed output whose population spread happens to converge near the true population's spread** — not with the model learning to track individual samples. This is a std_ratio false PASS: the gate's healthy-band criterion was calibrated to catch collapsed/underdispersed outputs, and it did not catch this one, because a bimodal-constant output can have "healthy" aggregate variance while carrying zero per-sample information.

**The reverse asymmetry also shows up, worth recording precisely because it cuts the other way:** `polarization_angle` (`ψ`) in the *same* run — which FAILS the gate (final std_ratio=0.421) — shows two-branch scatter clusters that each have a real positive slope tracking true `ψ` (visible at both epoch 5 and epoch 80, `.../scatter/epoch_0080.png`'s ψ panel), consistent with the pre-existing, already-documented 2-fold branch ambiguity in ψ reconstruction (`formulae_reference.md` A.8: "ψ is 2-fold ambiguous") rather than pure non-learning. `lam010_retune_a`'s (baseline-mode control) ψ panel shows the same two-branch-with-slope structure. This doesn't flip the mechanical Step 0 verdict — std_ratio<0.5 is still a FAIL by the pre-registered threshold, and that threshold is not renegotiated after seeing this — but it means "FAIL" here is not straightforwardly "no learning" either; the true picture is more mixed than either the PASS or the FAIL label alone conveys.

**This does not change the round's verdict.** The pre-registered rule (`preregistration_lam_retune.md`) is binary and mechanical by design specifically so a result like this — one metric looking better than it is, another looking worse than it is — doesn't get relitigated per-case after the fact. λ=0.10 still fails 3 of 4 readings by the letter of the criterion, and "λ alone insufficient, post-formula-fix" stands as the verdict of record. What this correction changes is confidence in what "healthy std_ratio" *means* on its own: it does not, by itself, certify that a head is producing a real per-sample regression — only a scatter plot does that. Recommending (not yet actioned) that any future Step 0-style gate add a mandatory scatter-plot check alongside the std_ratio number, rather than treating std_ratio as sufficient on its own — flagged in `experiment_index.md`'s "What's still open?" for follow-up.

## Redo-completeness audit and Section E/adversarial-review checks prepped (2026-09-16)

Explicit instruction: redoing the investigation under the corrected formula means redoing *every* check the original performed to reach its conclusions, including ones whose result isn't expected to change — a null result the redo hasn't independently re-earned isn't justified just because the original earned an equivalent-looking one under the wrong formula. Audited every `.py` file in `experiments/phic_psi_poc/` against this directory (full list and disposition in `experiment_index.md`'s "What's still open?").

**Ported, repointed at this redo's own certified 4-model set, and CPU-wiring-verified (config resolution, `best.weights.h5`/`transforms.json` presence, pure-math sanity — no GPU, no real predict call):**
- [`inclination_stratification.py`](inclination_stratification.py) — original's 2026-07-23 adversarial-review follow-up (Table 6.6/§6.7), face-on/mixed/edge-on `ang_MAE` bands for `coa_phase`/`polarization_angle`/`inclination`.
- [`inclination_control_stratification.py`](inclination_control_stratification.py) — companion control, same bands applied to `mchirp` to rule out the banding scheme itself injecting variance.
- [`snr_stratification.py`](snr_stratification.py) — original's Section E, same idea with SNR terciles instead of inclination bands.

None have been run — all three load real checkpoint weights and call `trainer.predict`, the same GPU/eval category as `evaluate_poc.py`, and this machine is CPU-only per `CLAUDE.md`. Queued for the lab GPU machine (`run_commands.md` Stage 2b — renumbered 2026-09-16 from an earlier "Stage 6" once it became clear this batch depends only on Stage 2, not on the λ-retune stages 3–5; see that file's dependency graph).

**Not yet ported — genuine gaps, not oversights, flagged for a priority call rather than silently skipped:** `bootstrap_ang_mae.py` (individual-head shuffle-null CI, ungated — distinct from the preregistration's combo-specific Steps 1–3, which never ran).

## Second batch ported (2026-09-16, same day): `perturbation_trace_standalone.py`, `analyse_predictions.py`, `diagnostic_checks.py`, `plot_certified_memorization.py` — and a real finding from the last one

**`perturbation_trace_standalone.py`** and **`analyse_predictions.py`**: ported, repointed, CPU-wiring-verified (no GPU/predict call run). Both directly relevant to this session's mode-collapse finding — `analyse_predictions.py`'s Table 4 "COLLAPSE" grade (circ_r > 0.9 and angular_mae > 0.5) is exactly the systematic version of the scatter-plot check that caught `lam010_retune_b`/coa_phase's std_ratio false-positive by hand. Also added `.pdf` saving alongside `.png` in `analyse_predictions.py`'s three plotting functions (the original only wrote `.png`) — bringing it in line with this repo's current figure convention, not a change to its logic.

**`diagnostic_checks.py`**: ported with two load-bearing corrections, not a plain copy. (1) Check 6 (gradient chain) originally replicated the combo construction as `complex_mul(z_phic_norm, z_psi_norm)` — the **wrong, pre-correction formula**, no doubling. Fixed to insert `tf_double_angle(z_phic_norm)` before combining, matching `trainer.py`'s actual `_build_combo_vectors`; verified the fix with a synthetic-tensor check (gradient flows through the new stage, `double_angle` genuinely doubles the angle: 0.3→0.6 rad). Porting this unfixed would have silently traced the wrong formula's gradient chain from inside the redo directory. (2) Checks 5/7 (tanh saturation, early-training saturation timing) are **not run** — not a redo of a check expected to reproduce, but structurally inapplicable: this redo uses `activation="linear"` for PERIODIC heads from day 1 (tanh saturation was the original's own Run 3 root cause, already fixed), so a "|value| > 0.99" saturation heuristic on an unbounded linear output would manufacture a false signal, not skip a redundant one. Function bodies kept for reference, not called from `main()`. Check 3's `runs` dict repointed at the redo's actual 7 Round-1 run directories (`phic_psi_poc_redo_{a,b}`, `phic_psi_redo_{tcn,cnn_baseline,cnn_attention,inception_time,resnet1d}`) — verified all 7 resolve to real `history.csv` files.

**`plot_certified_memorization.py`**: this one needed an actual definitional adaptation, not just a repoint. The original singled out exactly 2 models (poc_b, cnn_attention) because those were the two its null was *certified* on. This redo has certified nothing — Step 0 never cleared at any λ — so there's no principled "certified subset" to single out. Adapted to apply the same train-vs-val circular-loss check uniformly to all 4 down-select-confirmed models (poc_a, poc_b, tcn, cnn_attention) instead. This one is CPU-only (reads `history.csv`, no model loading) — **actually run**, not just wiring-verified. Output: [`diagnostic_output/redo_models_train_val_loss.{png,pdf}`](diagnostic_output/).

**Finding, previously hidden by the Round-1 summary table (which only ever reported *validation* circular loss — `NOTES.md`'s own Round-1 table header says "(val, last-10 mean)"): `cnn_attention` shows a large train/validation divergence on `coa_phase`/`polarization_angle` that `poc_a`/`poc_b`/`tcn` do not.**

| Model | Head | train last-10 mean | val last-10 mean | train–val gap |
|---|---|---|---|---|
| poc_a | coa_phase | 0.9680 | 1.0155 | +0.0475 |
| poc_a | polarization_angle | 0.9928 | 1.0009 | +0.0080 |
| poc_b | combo_A | 0.9845 | 0.9999 | +0.0153 |
| poc_b | combo_B | 0.9869 | 0.9895 | +0.0026 |
| tcn | coa_phase | 0.9787 | 1.0093 | +0.0306 |
| tcn | polarization_angle | 0.9800 | 1.0069 | +0.0269 |
| **cnn_attention** | **coa_phase** | **0.5631** | **1.0012** | **+0.4381** |
| **cnn_attention** | **polarization_angle** | **0.5367** | **1.0088** | **+0.4721** |

`cnn_attention`'s training-split circular loss drops to ~0.54–0.56 by epoch 80 (well below the 1.0 null) while its validation loss stays flat at ~1.0 throughout — a textbook memorization signature: the network found a way to reduce loss on the specific training samples' `coa_phase`/`ψ` values without learning anything that generalizes to held-out data. The other three down-selected models show train/val gaps an order of magnitude smaller (0.003–0.05), consistent with genuinely flat behavior on both splits, not memorization.

**This does not overturn any verdict already on record** — the Step 0 std_ratio gate that drives this redo's central decision table is about `poc_redo_b`'s `combo_A`/`combo_B`, not `cnn_attention`'s baseline-mode circular loss, and `cnn_attention`'s positive controls (mchirp/merger_time/snr) were and remain healthy, so its place in the down-select-confirmed 4-model set stands. **What it does change: `cnn_attention`'s validation-flat circular loss on `coa_phase`/`polarization_angle` should not be read as "the network never engaged with these targets" the way `poc_a`/`poc_b`/`tcn`'s flat-on-both-splits behavior can be — it engaged, memorized, and still failed to generalize.** Any future work that uses `cnn_attention` as a comparison point for these two heads specifically should carry this caveat. Exactly the kind of thing "redo every check, even the ones expected to reproduce" was for — this was not visible in any previously-reported number, because no previously-reported number was ever train-vs-val for these two heads on this model.

## Next steps

- [x] Step 0 — branches, folder, shared checkpoint-callback fix
- [x] Step 1.1/1.2/1.6 — prerequisite checks, redone under corrected formula
- [x] Step 2.2/2.3 — `transform_utils.py`/`curriculum.py`/`trainer.py` implemented and verified
- [x] Step 3 — `validation_script.py`, 29/29 checks pass
- [x] Config YAMLs for the full 7-architecture Round-1-equivalent sweep — verified end-to-end on CPU before handoff
- [x] Full 7-architecture sweep trained on the lab GPU machine (2026-09-15, see above)
- [x] Down-select re-validation — original 4-model certified set (poc_a, poc_b, tcn, cnn_attention) confirmed still holds, on fresh evidence
- [x] Step 0 std_ratio gate (`diagnostic_logvar_gate.py`) — **FAILS on both heads, both poc_redo_a and poc_redo_b.** Verdict: UNINTERPRETABLE, not NULL. Failure appears in baseline mode too, so it's very unlikely to be formula-specific — most likely λ=0.01 insufficient for TCN here, matching the original's own pre-retune finding.
- [x] λ-retune criterion pre-registered — [`preregistration_lam_retune.md`](preregistration_lam_retune.md), written 2026-09-15 before any retune exists. Primary tests: `poc_redo_b`'s `combo_A`/`combo_B` own angles (not individual φc/ψ — those have no direct supervision in poc mode). `poc_redo_a` retrained in parallel as a required control. Frozen from this point — dated addenda only if it needs revision later.
- [x] λ=0.05 configs built — [`config_lam005_retune_b.yaml`](config_lam005_retune_b.yaml) (primary, copied from `config_poc.yaml`), [`config_lam005_retune_a.yaml`](config_lam005_retune_a.yaml) (required control, copied from `config_baseline.yaml`). Both verified end-to-end on CPU (forward pass, loss, gradient step, no `None` grads).
- [x] λ=0.05 trained on the lab GPU machine and Step 0 gate rerun (2026-09-15) — **FAILS again, all four readings.** Verdict: still UNINTERPRETABLE. See "λ=0.05 retune" section above.
- [x] λ=0.10 fallback configs built — [`config_lam010_retune_b.yaml`](config_lam010_retune_b.yaml), [`config_lam010_retune_a.yaml`](config_lam010_retune_a.yaml), both CPU-verified (forward pass, loss, gradient step, no `None` grads). Last step the preregistration's fallback order commits to.
- [x] λ=0.10 trained on the lab GPU machine and Step 0 gate rerun (2026-09-16) — **FAILS, 3 of 4 readings** (one, `lam010_retune_b`/coa_phase, passes for the first time). Per the preregistration's own decision table: **λ alone insufficient for TCN coa_phase/polarization_angle, post-formula-fix.** λ-retune line of attack exhausted — see "λ=0.10 retune" section above.
- [ ] **Open decision point, not yet made:** the preregistration commits only through λ=0.10 and explicitly does not authorize a new ad hoc λ value after seeing this result. Steps 1–3 (bootstrap significance, effect size, SNR stratification) never ran — Step 0 never cleared at any tried λ. Whether to pursue a different stabilization mechanism (not λ alone), accept this as the redo's terminal state on this architecture, or something else, is a call for the user to make, not a threshold to invent post hoc.
- [x] Redo-completeness audit (2026-09-16) — every original `.py` checked against this directory; disposition in `experiment_index.md`.
- [x] `inclination_stratification.py`, `inclination_control_stratification.py`, `snr_stratification.py` ported, repointed at the redo's certified 4-model set, CPU-wiring-verified. Not yet run (GPU/eval) — queued for the lab GPU machine.
- [x] `perturbation_trace_standalone.py`, `analyse_predictions.py`, `diagnostic_checks.py` ported (with corrections — see the audit entry above), CPU-wiring-verified. Not yet run — GPU/eval work, queued for the lab GPU machine.
- [x] `plot_certified_memorization.py` ported and adapted (uniform 4-model application, not the original's 2-model "certified" framing), **actually run** (CPU-only) — found `cnn_attention`'s train/val memorization divergence on `coa_phase`/`polarization_angle`, see finding above.
- [x] `bootstrap_ang_mae.py` ported 2026-09-16, repointed at the redo's certified 4-model set, CPU-wiring-verified (config resolution + a statistical-logic sanity check: synthetic perfect-fit case reads significant at z≈24σ, synthetic independent-random case correctly reads not significant at p=0.68). Not yet run — GPU/eval work. **This closes the redo-completeness audit** — every original `.py` check now has a redo counterpart, except the per-λ training-launcher wrappers, which the redo's config-based workflow already covers directly.

**The redo folder produced its first full-sweep result on 2026-09-15** — infrastructure is fully validated (positive controls healthy across all 7 runs, periodic checkpoints, plot/eval pipeline all confirmed working), but the central question (does the corrected formula let the model learn where the wrong one couldn't) is not yet answered — it's gated on the diagnostic work above, not concluded from this one run. The λ=0.05 retune (2026-09-15) did not clear the gate; the λ=0.10 retune (2026-09-16), the preregistration's last committed fallback, did not clear it either (3 of 4 readings still fail). Per the preregistration's own decision table, this is reported as "λ alone insufficient for this architecture/head, post-formula-fix" — a negative result about the retune mechanism, not evidence for or against the corrected-formula degeneracy hypothesis itself, which remains formally unanswered under this line of attack.
