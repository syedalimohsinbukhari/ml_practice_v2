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

## Next steps

- [x] Step 0 — branches, folder, shared checkpoint-callback fix
- [x] Step 1.1/1.2/1.6 — prerequisite checks, redone under corrected formula
- [x] Step 2.2/2.3 — `transform_utils.py`/`curriculum.py`/`trainer.py` implemented and verified
- [x] Step 3 — `validation_script.py`, 29/29 checks pass
- [x] Config YAMLs for the full 7-architecture Round-1-equivalent sweep — verified end-to-end on CPU before handoff
- [x] Full 7-architecture sweep trained on the lab GPU machine (2026-09-15, see above)
- [x] Down-select re-validation — original 4-model certified set (poc_a, poc_b, tcn, cnn_attention) confirmed still holds, on fresh evidence
- [x] Step 0 std_ratio gate (`diagnostic_logvar_gate.py`) — **FAILS on both heads, both poc_redo_a and poc_redo_b.** Verdict: UNINTERPRETABLE, not NULL. Failure appears in baseline mode too, so it's very unlikely to be formula-specific — most likely λ=0.01 insufficient for TCN here, matching the original's own pre-retune finding.
- [ ] λ-retune pass mirroring the original's Runs 8–9b — pre-register the criterion first (same discipline `preregistration_lam_retune.md` insists on), then rerun the Step 0 gate before touching Steps 1–3
- [ ] Only once combo_A/combo_B's gate passes (clean NULL) or shows real learning does the central degeneracy question get an answer

**The redo folder produced its first full-sweep result on 2026-09-15** — infrastructure is fully validated (positive controls healthy across all 7 runs, periodic checkpoints, plot/eval pipeline all confirmed working), but the central question (does the corrected formula let the model learn where the wrong one couldn't) is not yet answered — it's gated on the diagnostic work above, not concluded from this one run.
