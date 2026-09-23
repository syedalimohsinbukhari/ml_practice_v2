# Original vs. Redo — Deep Comparison

This file is the deliverable specified by `deep_comparison_task.md`.
It traces every analogous check between `experiments/phic_psi_poc/` (original, wrong combo formula `φc±2ψ`) and `experiments/phic_psi_poc-redo/` (redo, corrected formula `2φc±2ψ`), with numbers pulled from both sides' artifacts.
A prior chat-level pass already established the headline: fixing the formula did not change the conclusion.
This document is the numbers-backed version of that headline, not a repeat of it.
Every number below cites its exact source file, and, where the source is a table/log, the row or line it came from.

## 1. Procedural / methodology parity

`redo_procedure.md` already tags every step `[UNCHANGED]`/`[CORRECTED]`/`[NEW]` against the original's own plan.
Checking those tags against everything that happened after the document was written (the Step 0 gate, the λ-retune, Stage 2b(i)) shows they still hold — the redo really did run every step it claimed to, and the frozen tags accurately describe what was reused vs. rebuilt.
The table below adds the fourth category the brief asks for: checks that exist only on the **original** side, with no redo counterpart.

| Step / check                                                                           | Tag                                                 | Original artifact                                                                             | Redo artifact                                                                                                                                                | Notes                                                                                                                                                                                                    |
|----------------------------------------------------------------------------------------|-----------------------------------------------------|-----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Step 1.1 sign/combo check                                                              | CORRECTED                                           | `experiments/phic_psi_poc/results.md` Run 4                                                   | `experiments/phic_psi_poc-redo/prereq_checks_output/prereq_checks_20260914_160834.log`                                                                       | Both rerun from scratch; redo's tags hold up — see §2 for numbers.                                                                                                                                       |
| Step 1.2 curriculum weight w(ι)                                                        | CORRECTED                                           | `experiments/phic_psi_poc/results.md` Run 2/4                                                 | same redo log, Step 1.2 section                                                                                                                              | Both keep the `1−cos²ι` default over the empirical fit, same reasoning.                                                                                                                                  |
| Step 1.3–1.5 (data repr., true-ι access, curriculum mechanism)                         | UNCHANGED                                           | `experiments/phic_psi_poc/NOTES.md` §"Confirmed"                                              | `experiments/phic_psi_poc-redo/redo_procedure.md` §1.3–1.5 (cites original, not rerun)                                                                       | Correctly not formula-dependent; redo cites rather than reruns.                                                                                                                                          |
| Step 1.6 cos ι histogram                                                               | CORRECTED (cheap rerun)                             | `experiments/phic_psi_poc/NOTES.md` Step 1.6                                                  | redo prereq log, Step 1.6                                                                                                                                    | Identical result both sides: 28.7%/32.7%.                                                                                                                                                                |
| tanh→linear fix, magnitude penalty                                                     | UNCHANGED (already known)                           | Discovered in `diagnostic_log.md` Runs 3–7                                                    | Present from day 1, `redo_procedure.md` §2.5                                                                                                                 | Redo correctly didn't rediscover either bug.                                                                                                                                                             |
| Round-1 7-architecture sweep                                                           | CORRECTED (full rerun)                              | `NOTES.md` "Round 1" + post-fix retrain                                                       | `NOTES.md` (redo) "Round 1 — full 7-architecture sweep"                                                                                                      | See §2 for the down-select numbers.                                                                                                                                                                      |
| Down-select re-validation                                                              | NEW                                                 | not performed as an explicit step in the original (down-select simply happened)               | `redo_procedure.md` §4 "Explicit down-select re-validation step"                                                                                             | Redo made this step explicit; original's equivalent decision was implicit and initially rested on stale numbers (see the `NOTES.md` 2026-07-24 consistency-audit note).                                  |
| Combo-phase training (λ=0.01)                                                          | CORRECTED                                           | Run 6/7                                                                                       | redo NOTES.md "Round 1" combo result                                                                                                                         | Same qualitative outcome (flat combo loss), see §2.                                                                                                                                                      |
| Step 0 std_ratio gate                                                                  | UNCHANGED (methodology), CORRECTED (target)         | `diagnostic_lam005_retune.py`/`diagnostic_lam010_retune.py`, gate embedded in those scripts   | `diagnostic_logvar_gate.py`, standalone                                                                                                                      | Redo generalized the gate to a dedicated always-rerunnable script and widened its scope from 2 to 4 readings per round (see §2, §3).                                                                     |
| λ-retune (0.05, 0.10)                                                                  | CORRECTED                                           | Run 9a/9b                                                                                     | redo NOTES.md "λ=0.05/0.10 retune"                                                                                                                           | Both closed with "λ alone insufficient"; see §2 for the numeric table.                                                                                                                                   |
| Bootstrap ang_MAE (individual heads)                                                   | UNCHANGED methodology, CORRECTED target/checkpoints | `bootstrap_ang_mae.py` → `bootstrap_output/bootstrap_ang_mae_20260721_093533.md`              | same script name, ported → `bootstrap_output/bootstrap_ang_mae_20260916_113955.md`                                                                           | See §2 for the full 12-row table; this is the check behind the inclination anomaly in §3.5.                                                                                                              |
| SNR stratification                                                                     | UNCHANGED methodology                               | `snr_stratification.py` → `snr_output/snr_stratification_20260721_094039.md`                  | ported → `snr_output/snr_stratification_20260916_113730.md`                                                                                                  | Same null pattern; see §2.                                                                                                                                                                               |
| Inclination stratification (+ control)                                                 | UNCHANGED methodology                               | `inclination_stratification.py`/`inclination_control_stratification.py`, 2026-07-23           | ported, run 2026-09-18                                                                                                                                       | See §2.                                                                                                                                                                                                  |
| `diagnostic_checks.py` full suite                                                      | CORRECTED (Check 6 fixed)                           | 7 checks, `diagnostic_output/diagnostic_checks_20260721_000331.log`                           | 5 checks (5/7 dropped as inapplicable — linear activation from day 1), `diagnostic_output/diagnostic_checks_20260916_113913.log`                             | Redo's Check 6 originally hardcoded the *pre-correction* formula and had to be fixed mid-port — a real bug caught, not a trivial repoint (`experiment_index.md` Code table, `diagnostic_checks.py` row). |
| `analyse_predictions.py`                                                               | UNCHANGED methodology                               | `analysis_output/analysis_report_20260720_234304.md`                                          | ported, `.pdf` saving added, `analysis_output/analysis_report_20260916_113847.md`                                                                            | See §2.                                                                                                                                                                                                  |
| `plot_certified_memorization.py`                                                       | CORRECTED (definitional adaptation)                 | Applied to the 2 *certified* models only (poc_b, cnn_attention)                               | Adapted to all 4 down-select-confirmed models, since the redo certified no null                                                                              | Genuinely new finding surfaced by the redo: `cnn_attention` memorizes on this pair of heads (§4).                                                                                                        |
| Perturbation trace (A.3)                                                               | UNCHANGED methodology, different resolution         | Full arc: first run → review → paired-stat rerun → `early` calibration → provisionally closed | `early` only, calibration FAILED, `final` never run                                                                                                          | Procedural asymmetry — see §3.4 and §4.                                                                                                                                                                  |
| **Adversarial AI review process** (4 rounds, 3 reviewers)                              | **original-only, no redo counterpart**              | `thesis/reviews/` (12 review files + checklist)                                               | none                                                                                                                                                         | Largest procedural gap — see below.                                                                                                                                                                      |
| **Thesis chapter** (`.md`/`.tex`, claim-to-artifact appendix)                          | **original-only, no redo counterpart**              | `thesis/chapter_phic_psi_degeneracy.{md,tex}`                                                 | none (`closing_summary.md` is the closest redo analogue, but it's a NOTES-style summary, not a polished, cited chapter)                                      | See §6.                                                                                                                                                                                                  |
| **Standalone arXiv paper**                                                             | **original-only, no redo counterpart**              | `paper/paper1/` (Overleaf-linked)                                                             | none                                                                                                                                                         | Downstream of the thesis chapter; doesn't exist for the redo because the chapter doesn't either.                                                                                                         |
| Sky-position deep dive (bootstrap, SNR-stratification, rose/Mollweide/histogram plots) | **redo-only, no original counterpart**              | none                                                                                          | `bootstrap_sky_separation.py`, `snr_stratification_sky_position.py`, `rose_plot_residuals.py`, `sky_position_mollweide.py`, `sky_angular_separation_hist.py` | Genuinely new discovery track, not a gap in the original — the original never had reason to look here (see §4).                                                                                          |
| Periodic epoch-N checkpoints                                                           | **NEW, and shared**                                 | never existed (`CLAUDE.md`'s own rule cites this exact gap)                                   | `redo_procedure.md` §0, lands the `CLAUDE.md`-mandated fix in the shared `src/gwml/training/train.py`                                                        | Additive; benefits both the redo and any future campaign, not redo-specific in effect.                                                                                                                   |

**On the two checks flagged in the brief as the obvious candidates:** yes, both are real, original-only gaps, and there is one more of the same kind (the paper).
No other original-only check turned up beyond the review process, the chapter, and the paper — everything else in `experiments/phic_psi_poc/`'s `.py` file list has a redo counterpart per the redo's own completeness audit (`experiment_index.md`, "Redo-completeness audit" section; `bootstrap_ang_mae.py` was the last gap it closed, 2026-09-16).

## 2. Numeric side-by-side table

### 2.1 Step 1.1 — sign/combination check

| Quantity                          | Original (`φc±2ψ`)                                                                                     | Redo (`2φc±2ψ`)                                            |
|-----------------------------------|--------------------------------------------------------------------------------------------------------|------------------------------------------------------------|
| Well-constrained combo, cos ι > 0 | combo_B, ratio 1.155× (95% CI [1.118, 1.195])                                                          | combo_B, ratio 1.17× (95% CI [1.11, 1.24])                 |
| Well-constrained combo, cos ι < 0 | combo_A, ratio 1.171× (95% CI [1.130, 1.216])                                                          | combo_A, ratio 1.19× (95% CI [1.12, 1.26])                 |
| Ratio at face-on (ι≈0.05–0.1)     | ≈1.56×                                                                                                 | ≈1.62–1.65×                                                |
| Ratio at edge-on (ι≈π/2)          | ≈1.05–1.07×                                                                                            | ≈1.04–1.07×                                                |
| Reference-point significance      | needed the full 100-pt sweep + bootstrap to establish (the raw 2-point check alone wasn't significant) | significant directly at the simple 2-point reference check |

Sources: `experiments/phic_psi_poc/results.md` ("Run 4 — deep sweep" table and "Summary of go/no-go signals") vs. `experiments/phic_psi_poc-redo/prereq_checks_output/prereq_checks_20260914_160834.log` (Step 1.1 section) and `NOTES.md` (redo) Step 1.1 entry.

### 2.2 Step 1.2 — curriculum weight w(ι)

| Quantity            | Original                | Redo                                     |
|---------------------|-------------------------|------------------------------------------|
| w(face-on, cos²ι=1) | 0.0000                  | 0.0000                                   |
| w(edge-on, cos²ι=0) | 0.1212                  | 0.1416                                   |
| Decision            | use `w=1−cos²ι` default | use `w=1−cos²ι` default (same reasoning) |

Sources: `experiments/phic_psi_poc/results.md` Run 2 vs. redo prereq log Step 1.2 section / `NOTES.md` (redo).

### 2.3 Round-1 down-select R² (7-architecture sweep, post-fix)

| Model          | Head           | Original (post-fix, cited in `NOTES.md` 2026-07-24 audit) | Redo (Round 1, 2026-09-15) |
|----------------|----------------|-----------------------------------------------------------|----------------------------|
| cnn_baseline   | mchirp R²      | ≈0.83                                                     | 0.833                      |
| cnn_baseline   | merger_time R² | ≈0.82                                                     | 0.818                      |
| inception_time | merger_time R² | ≈+0.0001                                                  | 0.000                      |

Sources: `experiments/phic_psi_poc/NOTES.md` 2026-07-24 "consistency audit" correction note vs. `experiments/phic_psi_poc-redo/NOTES.md` "Round 1" table.
Both sides independently land on the same 4-model certified set (poc_a, poc_b, tcn, cnn_attention); the redo explicitly re-validated this rather than assuming it (`redo_procedure.md` §4).

### 2.4 Positive controls (same 4 certified models, λ=0.01, same dataset)

| Head         | Metric   | poc_a orig / redo | poc_b orig / redo | tcn orig / redo | cnn_attention orig / redo |
|--------------|----------|-------------------|-------------------|-----------------|---------------------------|
| mchirp       | MAE      | 0.9773 / 1.0133   | 1.0242 / 1.0078   | 0.9505 / 0.9785 | 1.3632 / 1.4181           |
| mchirp       | R²       | 0.9594 / 0.9580   | 0.9569 / 0.9579   | 0.9629 / 0.9602 | 0.9263 / 0.9197           |
| merger_time  | R²       | 0.9141 / 0.9252   | 0.9190 / 0.9277   | 0.9206 / 0.9238 | 0.9089 / 0.9062           |
| snr          | R²       | 0.7850 / 0.7872   | 0.7825 / 0.7890   | 0.7841 / 0.7884 | 0.7551 / 0.7411           |
| sky_position | ang. MAE | 8.2° / 8.7°       | 10.0° / 10.3°     | 4.5° / 6.0°     | 3.3° / 4.4°               |

Sources: `experiments/phic_psi_poc/analysis_output/analysis_report_20260720_234304.md` vs. `experiments/phic_psi_poc-redo/analysis_output/analysis_report_20260916_113847.md`.
Positive controls reproduce closely across the two campaigns (same architecture, seed, and dataset file — `combined_repackaged.hdf`, referenced identically by both `config_baseline.yaml`s) — but not exactly, and `sky_position` degrades by a small, consistent amount for every model in the redo.
This small but consistent drift is the same phenomenon investigated in §3.5.

### 2.5 Combo/circular loss (poc_b / poc_redo_b), λ=0.01, epoch 79

| Quantity                                | Original (`φc+2ψ`/`φc−2ψ`) | Redo (`2φc+2ψ`/`2φc−2ψ`) |
|-----------------------------------------|----------------------------|--------------------------|
| val_circular_loss_combo_A, last-10 mean | 0.9989                     | 0.9999                   |
| val_circular_loss_combo_B, last-10 mean | 0.9913                     | 0.9895                   |

Sources: `experiments/phic_psi_poc/diagnostic_log.md` Run 7 section (Check 3 table) vs. `experiments/phic_psi_poc-redo/NOTES.md` "Round 1" central-result entry, which quotes both numbers side by side directly.
This is the single most direct "did the fix change anything" comparison in either investigation, and the two numbers are within 0.001–0.002 of each other.

### 2.6 std_ratio gate at each λ

**Original** — 2 primary targets only (tcn/coa_phase, poc_a/pol_angle):

| λ    | Model/head      | frac unhealthy (last 40 ep) | trend/ep | final std_ratio                                   | Gate              |
|------|-----------------|-----------------------------|----------|---------------------------------------------------|-------------------|
| 0.01 | tcn coa_phase   | 0.80 (32/40)                | −0.0078  | 0.34                                              | FAIL              |
| 0.01 | poc_a pol_angle | 0.75 (30/40)                | +0.0001  | 0.44                                              | FAIL              |
| 0.05 | tcn coa_phase   | 0.05                        | −0.00638 | (not tabulated at this λ; late plateau 0.58–0.62) | FAIL (trend)      |
| 0.05 | poc_a pol_angle | 0.35                        | +0.00718 | (late plateau 0.53–0.56)                          | FAIL              |
| 0.10 | tcn coa_phase   | 0.28                        | −0.00255 | oscillates [0.2,0.95]                             | FAIL (worse)      |
| 0.10 | poc_a pol_angle | 0.72                        | +0.00731 | crosses 0.5 only in last 11 epochs                | FAIL (much worse) |

**Redo** — 4 readings per round (both heads × poc_redo_b/poc_redo_a, since the combo depends on both source vectors jointly):

| λ    | Run/head      | frac unhealthy (last 40 ep) | trend/ep | final std_ratio | Gate                                                                                                      |
|------|---------------|-----------------------------|----------|-----------------|-----------------------------------------------------------------------------------------------------------|
| 0.01 | b / coa_phase | 0.225                       | +0.00638 | 0.583           | FAIL                                                                                                      |
| 0.01 | b / pol_angle | 0.850                       | −0.00402 | 0.251           | FAIL                                                                                                      |
| 0.01 | a / coa_phase | 0.750                       | −0.00721 | 0.407           | FAIL                                                                                                      |
| 0.01 | a / pol_angle | 1.000                       | +0.00076 | 0.374           | FAIL                                                                                                      |
| 0.05 | b / coa_phase | 1.000                       | +0.00093 | 0.315           | FAIL                                                                                                      |
| 0.05 | b / pol_angle | 0.725                       | +0.00679 | 0.531           | FAIL                                                                                                      |
| 0.05 | a / coa_phase | 0.350                       | −0.00561 | 0.522           | FAIL                                                                                                      |
| 0.05 | a / pol_angle | 0.425                       | +0.00383 | 0.572           | FAIL                                                                                                      |
| 0.10 | b / coa_phase | 0.000                       | +0.00011 | 0.930           | PASS (later found to be a std_ratio false-positive — collapsed constant, confirmed by scatter inspection) |
| 0.10 | b / pol_angle | 0.950                       | +0.00326 | 0.421           | FAIL                                                                                                      |
| 0.10 | a / coa_phase | 0.425                       | −0.00453 | 0.546           | FAIL                                                                                                      |
| 0.10 | a / pol_angle | 0.650                       | +0.00220 | 0.559           | FAIL                                                                                                      |

Sources: `experiments/phic_psi_poc/diagnostic_log.md` Run 9a/9b sections + `std_ratio_trajectories.md` vs. `experiments/phic_psi_poc-redo/diagnostic_output/diagnostic_logvar_gate_20260916_085902.md` (which itself tabulates all three rounds).
Both close with the same verdict language ("λ alone insufficient... post-formula-fix"), and both show the same qualitative shape — a near-miss/partial-improvement at 0.05, a worse or mixed result at 0.10 — even though the redo's broader 4-reading scope structurally cannot pass as easily as the original's 2-reading scope.

### 2.7 Bootstrap ang_MAE — all 4 models, all 3 periodic heads

| Head               | Model         | Original z, p (φc±2ψ checkpoints) | Redo z, p (2φc±2ψ checkpoints) |
|--------------------|---------------|-----------------------------------|--------------------------------|
| coa_phase          | poc_a         | −0.20, 0.579                      | −0.38, 0.649                   |
| coa_phase          | poc_b         | −1.25, 0.895                      | −0.39, 0.649                   |
| coa_phase          | tcn           | −0.56, 0.712                      | −1.18, 0.879                   |
| coa_phase          | cnn_attention | −2.43, 0.994                      | −1.97, 0.976                   |
| polarization_angle | poc_a         | +0.46, 0.324                      | −0.82, 0.793                   |
| polarization_angle | poc_b         | −0.05, 0.518                      | −0.63, 0.740                   |
| polarization_angle | tcn           | −0.33, 0.630                      | −0.50, 0.692                   |
| polarization_angle | cnn_attention | +0.07, 0.472                      | +1.33, 0.091                   |
| inclination        | poc_a         | +1.04, 0.152                      | **+2.00, 0.0223 ★**           |
| inclination        | poc_b         | +0.07, 0.464                      | **+2.58, 0.0052 ★**           |
| inclination        | tcn           | +1.21, 0.114                      | **+2.58, 0.0059 ★**           |
| inclination        | cnn_attention | +3.17, 0.0007 ★                  | **+3.58, 0.0003 ★**           |

Sources: `experiments/phic_psi_poc/bootstrap_output/bootstrap_ang_mae_20260721_093533.md` vs. `experiments/phic_psi_poc-redo/bootstrap_output/bootstrap_ang_mae_20260916_113955.md` — full per-row citation, both files' "Summary" tables.
`coa_phase`/`polarization_angle`: 0 of 8 significant on either side (identical qualitative conclusion — the formula fix changed nothing here, within noise).
`inclination`: 1 of 4 significant in the original, 4 of 4 in the redo — every model's z-score moved upward. This is investigated in §3.5; it is not explained by the formula fix (inclination's loss path never touches the combo construction on either side, see `inclination_loss_trace.md`).

### 2.8 SNR stratification — high-SNR tercile Δ vs. null

| Head               | Model         | Original Δ                            | Redo Δ                                       |
|--------------------|---------------|---------------------------------------|----------------------------------------------|
| coa_phase          | poc_a         | −0.0358                               | −0.0293                                      |
| coa_phase          | poc_b         | −0.0292                               | −0.0328                                      |
| coa_phase          | tcn           | **+0.0274 (only monotonic improver)** | **+0.0209 (only monotonic improver, again)** |
| coa_phase          | cnn_attention | −0.0356                               | −0.0293                                      |
| polarization_angle | poc_a         | +0.0001                               | −0.0004                                      |
| polarization_angle | poc_b         | +0.0047                               | +0.0039                                      |
| polarization_angle | tcn           | +0.0059                               | +0.0078                                      |
| polarization_angle | cnn_attention | +0.0063                               | +0.0109                                      |

Sources: `experiments/phic_psi_poc/snr_output/snr_stratification_20260721_094039.md` vs. `experiments/phic_psi_poc-redo/snr_output/snr_stratification_20260916_113730.md`.
Both sides single out `tcn`/coa_phase as the one monotonic-with-SNR improver, and both dismiss it for the identical reason — it's the model whose std_ratio never stabilized (§2.6), so its ang_MAE trend across any stratification is doubly uninterpretable, not evidence.
Neither side's high-SNR delta for any head/model clears the 0.10 rad materiality floor.

### 2.9 Inclination stratification — edge-on Δ vs. null

| Head               | Model         | Original edge-on Δ | Redo edge-on Δ |
|--------------------|---------------|--------------------|----------------|
| coa_phase          | poc_a         | +0.0241            | +0.0028        |
| coa_phase          | poc_b         | +0.0096            | +0.0123        |
| coa_phase          | tcn           | −0.0407            | −0.0457        |
| coa_phase          | cnn_attention | −0.0087            | +0.0039        |
| polarization_angle | poc_a         | −0.0227            | −0.0019        |
| polarization_angle | poc_b         | +0.0054            | +0.0043        |
| polarization_angle | tcn           | −0.0083            | −0.0240        |
| polarization_angle | cnn_attention | −0.0189            | −0.0063        |

Sources: `experiments/phic_psi_poc/inclination_output/inclination_stratification_20260723_130630.md` vs. `experiments/phic_psi_poc-redo/inclination_output/inclination_stratification_20260916_113601.md`.
No model on either side clears the 0.10 rad floor in either direction; `tcn`/coa_phase is again the largest-magnitude outlier on both sides, and again in the same (anti-null) direction, again attributable to its unresolved std_ratio pathology rather than to a real edge-on recovery.

### 2.10 Gradient-chain health (Check 6) — poc_b/poc_redo_b, real trained checkpoint

| Stage                                    | Original (Run 7)                               | Redo (Round 1) |
|------------------------------------------|------------------------------------------------|----------------|
| dL/d(combo_A_pred)                       | 0.425                                          | 0.4245         |
| dL/d(combo_B_pred)                       | (not separately logged in the excerpted table) | 0.3562         |
| dL/d(z_phic_raw) [model output]          | 0.328                                          | 0.5703         |
| dL/d(z_psi_raw) [model output]           | 0.470                                          | 0.5138         |
| dL/d(inclination_raw) [healthy baseline] | 0.704                                          | 0.7280         |

Sources: `experiments/phic_psi_poc/diagnostic_log.md` Run 7 diagnostics ("Check 6" block) vs. `experiments/phic_psi_poc-redo/diagnostic_output/diagnostic_checks_20260916_113913.log` lines 297–304.
Both show healthy, non-vanishing gradients at every stage — the redo's chain additionally passes through the corrected `tf_double_angle` stage (line 299, `dL/d(z_2phic_norm) [after double_angle, A.1.5] = 0.3752`), confirming the formula fix is live end-to-end on the real checkpoint, not just in the synthetic-tensor unit test.

### 2.11 Prediction-perturbation asymmetry (Check 4)

| Model/head         | Original mean_abs(Δ) (rel_change)                          | Redo mean_abs(Δ) (rel_change)  |
|--------------------|------------------------------------------------------------|--------------------------------|
| coa_phase          | 1.52e-02 (part of the reported 89× ratio)                  | 1.61e-02 (rel_change 2.13e-02) |
| polarization_angle | 1.37e-02                                                   | 1.33e-02 (rel_change 1.48e-02) |
| mchirp (control)   | (ratio denominator, ≈1.7e-04 scale per Run 7's 89× figure) | 3.28e-04 (rel_change 5.27e-04) |
| Asymmetry ratio    | ≈89×                                                       | ≈40× (2.13e-02 / 5.27e-04)     |

Sources: `experiments/phic_psi_poc/diagnostic_log.md` Run 7 diagnostics ("Check 4" block, and the "89×" figure quoted in "A — Gating checks" A.3) vs. `experiments/phic_psi_poc-redo/diagnostic_output/diagnostic_checks_20260916_113913.log` lines 274–281.
Same qualitative finding both sides — gradient reaches the periodic-head weights and moves them far more than the converged scalar control — with the asymmetry magnitude differing (89× vs. ≈40×) because the two checkpoints/campaigns differ (§3.5), not because either side's chain is broken.

### 2.12 Perturbation trace (A.3) — early-stage calibration

| Model         | Original early mchirp verdict (t) | Redo early mchirp verdict (t) |
|---------------|-----------------------------------|-------------------------------|
| poc_a         | AMBIGUOUS (t ≈ −5.2)              | AMBIGUOUS (t = −0.83)         |
| poc_b         | AMBIGUOUS (t ≈ −3.4)              | AMBIGUOUS (t = −2.24)         |
| tcn           | AMBIGUOUS (t ≈ −4.1)              | AMBIGUOUS (t = −4.92)         |
| cnn_attention | OSCILLATORY (t ≈ −8.5)            | AMBIGUOUS (t = −8.73)         |

Sources: `experiments/phic_psi_poc/diagnostic_log.md` "Calibration run adjudication" section vs. `experiments/phic_psi_poc-redo/perturbation_trace_output/perturbation_trace_early_20260916_114129.md`.
Both sides' geometry classifier fails calibration (mchirp never reads DIRECTIONAL early); both sides' underlying paired-statistic channel shows real, often strongly significant, mchirp learning at the same stage.
The two investigations diverge procedurally at this point — see §3.4/§4 — the original salvaged A.3 via the paired channel and closed it (provisionally); the redo parked it, unresolved.

## 3. Verdict / epistemic-status comparison

### 3.1 The original's verdict

The thesis chapter certifies a null on exactly 2 λ-matched, metric-healthy models (poc_b, cnn_attention; `thesis/chapter_phic_psi_degeneracy.md` §8.2–§9), and treats the other 5 configurations (poc_a, tcn, plus the 3 unmatched trunks) as corroborating rather than independently repeating that certified test.
This is a single, coherent verdict, reached by running the λ-sweep to its pre-registered stopping point and then treating the 2 clean-|v| models' flat loss + non-significant bootstrap + null SNR trend as the certified evidence.

### 3.2 The redo's two verdicts

The redo has two verdicts that do not automatically collapse into one, exactly as the brief anticipates.

**Verdict A — the gated λ-retune path.** Per `preregistration_lam_retune.md`'s own decision table, the Step 0 std_ratio gate never cleared at any of λ = 0.01/0.05/0.10 for `poc_redo_b`'s `combo_A`/`combo_B` (§2.6 above).
The one reading that nominally passed at λ=0.10 was subsequently shown, by scatter-plot inspection, to be a collapsed-constant false positive, not real learning (`NOTES.md` (redo), 2026-09-16 correction).
Per the preregistration's own language, this path's status is **UNINTERPRETABLE**, formally: neither null nor counter-evidence.
Steps 1–3 of that preregistration (bootstrap significance, effect-size floor, SNR-monotonicity, all applied to the combo angles themselves) never ran, because they are gated on Step 0 passing.

**Verdict B — the broader, ungated Stage 2b(i) evidence (2026-09-18).** Running `analyse_predictions.py`, `diagnostic_checks.py` (Checks 2/4/6), `bootstrap_ang_mae.py`, and the inclination/SNR stratification scripts directly against the certified 4-model set answers a different, wider question — not "does `poc_redo_b`'s combo construction stabilize," but "is `coa_phase`/`polarization_angle` recoverable at all, on these checkpoints" — and answers it with a formal null: 0 of 8 bootstrap tests significant (§2.7), a confirmed genuine mode collapse for poc_a/poc_b rather than a metric artifact (`analysis_output/analysis_report_20260916_113847.md` Health Check table), and a clean gradient chain ruling out an implementation bug (§2.10).

### 3.3 Reconciling the two

These two verdicts resist clean reconciliation for a real, structural reason, not a sloppy one: they are answers to two different, only partially overlapping questions.
Verdict A is scoped narrowly to whether `poc_redo_b`'s specific `combo_A`/`combo_B` vectors — the direct output of the corrected-formula combo construction — are in an interpretable state; it says nothing about `poc_a`, `tcn`, or `cnn_attention`'s baseline-mode heads, which have their own, separately-failing gate readings (§2.6, `a` rows).
Verdict B is scoped broadly across all 4 models and both individual periodic heads (reconstructed for poc_b via ground-truth-assisted branch disambiguation, see §5), and never checks the gate at all — it takes the checkpoint as given and asks whether its predictions carry recoverable signal.
A model can fail Verdict A's narrow interpretability test (its raw vectors are in an unhealthy state) while its predictions, taken as-is, still fail Verdict B's broader null test in the uninformative direction — there is no logical contradiction in "the instrument's diagnostic health check fails, and the same instrument's actual output is also demonstrably uninformative."
The redo's own `closing_summary.md` states this directly: Verdict A is "superseded in practical confidence, not contradicted, by" Verdict B.
This document agrees with that framing and does not attempt a tighter formal merge; the open decision (whether the user wants one preregistered statement covering both) is carried into §6 rather than decided here.

### 3.4 A third comparison point: how each side treated its own unresolved instrument

Both investigations built the same multi-step perturbation trace, and both hit the identical failure mode at calibration — the geometry classifier cannot tell a fast-learning head from a dead one (§2.12).
The original responded by re-founding A.3's closure on a different, non-pre-registered channel (the paired probe-loss statistic) that happened to carry its own passed control in the same run, and closed A.3 — first outright, then downgraded to "provisionally closed, pending replication" after the v3 adversarial review (`diagnostic_log.md`, "Dated addendum," and `thesis/chapter_phic_psi_degeneracy.md` §6.6).
The redo, facing the same calibration failure and the same available paired-statistic channel (which, per §2.12, also shows strongly significant mchirp learning for 2 of 4 models — tcn, cnn_attention), did not attempt the same salvage: `NOTES.md` (redo) explicitly recommends *not* running `final` and leaves A.3 parked as open follow-up work, not closed even provisionally.
This is a genuine methodological asymmetry between the two investigations' handling of the identical instrument failure, not a difference in what the instrument found — worth surfacing because it means the redo is, in this one respect, more conservative than the original was, and the original's own "provisionally closed, pending replication" status (which that replication never happened for, on either side) is arguably the more apt description for both.

### 3.5 The inclination anomaly — investigated, not just noted

The task brief flags, as needing investigation rather than a note, that inclination's bootstrap significance pattern moved from 1-of-4 models significant (original) to 4-of-4 (redo), even though inclination's training path is architecturally untouched by the combo-formula fix on both sides (`inclination_loss_trace.md`, reused verbatim by the redo — Huber loss on a raw two-vector, no `normalize_unit`, no combo construction, confirmed by code trace in the original and unchanged in the redo's own `trainer.py`).
Three candidate explanations were checked against the repo's own artifacts rather than assumed.

**Checked and ruled out: a different dataset.** Both `config_baseline.yaml` files point at the identical file path, `combined_repackaged.hdf` (`experiments/phic_psi_poc/config_baseline.yaml:14`, `experiments/phic_psi_poc-redo/config_baseline.yaml:21`), and only one file with that name exists under this repo (`find` from repo root turns up a single match plus one in the sibling `ml-gw-search` tree, unrelated to this experiment).
The redo's own Step 1.6 rerun reproduces the original's population statistics exactly (28.7%/32.7% face-on/edge-on fraction, both sides) — the dataset is the same file, not a regenerated one.

**Checked and ruled out: a shared-code change between the two campaigns.** `git log --since="2026-07-20" --until="2026-09-16" -- src/gwml/training/ src/gwml/heads_spec.py src/gwml/models/` returns exactly one commit, and it is the additive periodic-checkpoint callback (`redo_procedure.md` §0's own "New: shared-file exception," landed as commit `09a4980`, "Add periodic epoch-N checkpointing to `_build_callbacks`").
That change only adds a save-weights callback at 10-epoch intervals; it does not touch loss computation, data loading, or model construction for any head, and the redo's own text argues (correctly, and consistent with this check) that it is additive and leaves best/final behavior untouched.
No code path that could differentially affect inclination's Huber-loss training exists between the two campaigns.

**Not ruled out, and the most likely remaining explanation: run-to-run variation between two training campaigns roughly two months apart.** Positive-control metrics (§2.4) reproduce closely but not exactly between the two campaigns — mchirp/merger_time/snr R² values differ by 0.001–0.01 in either direction per model, and `sky_position` angular MAE is *consistently worse* in the redo for every one of the 4 models (8.2°→8.7°, 10.0°→10.3°, 4.5°→6.0°, 3.3°→4.4°).
This pattern — small, mixed-direction drift in the strong-signal heads, and a consistent same-direction drift in the two weak-signal heads (`sky_position` uniformly worse, `inclination` uniformly more significant) — is what would be expected if the two campaigns' actual trained weights differ slightly for a reason outside the code (GPU/cuDNN operation-ordering nondeterminism, or a library/driver version difference on the lab GPU machine between 2026-07-20 and 2026-09-15, which this repo does not log or pin), rather than nothing at all: a head with a strong, well-conditioned gradient signal (mchirp) is robust to small weight perturbations, while a head whose entire signal is a marginal, near-null population-level bias (inclination, sky_position) is exactly the kind of quantity a small, consistent perturbation to the shared trunk's learned features would move most visibly.
This explanation is offered as the best-supported reading of the available evidence, not confirmed — no environment/library manifest is checked into this repo for either campaign, so it cannot be verified directly.
**What this does not change:** the `coa_phase`/`polarization_angle` null is unaffected either way (§2.7) — the anomaly is confined to the sub-floor, already-flagged-as-imperfect inclination "noise floor" convention on both sides, and both investigations' own inclination stratification scripts (§2.9) already carry an explicit caveat that ι's failure mechanism is separate from and not fully independent of φc/ψ's (`thesis/chapter_phic_psi_degeneracy.md` §6.7's head-capacity-confound caveat, reused by the redo).

## 4. Asymmetric findings

| Finding                                                                                                                                               | Side                                                                                                                                                                                                                                          | Classification                                                                                                                 | Reasoning                                                                                                                                                                                                                                                                                                                                                                                                                           |
|-------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `cnn_attention` train/val memorization gap on coa_phase/pol_angle                                                                                     | Original found it for its 2 certified models specifically (`thesis/chapter_phic_psi_degeneracy.md` Fig. 8.1, §8.2); redo found the same pattern independently for all 4 down-select-confirmed models (`plot_certified_memorization.py`, redo) | Genuine discovery on both sides, methodological gap closed by the redo                                                         | Original's version was scoped to "certified models only" by definition; the redo's adaptation (forced by having no certified null yet) is what let it also catch `poc_a`/`tcn`/`cnn_attention` uniformly — not a new finding invalidated by scope, but the redo's version is strictly more complete.                                                                                                                                |
| `cnn_attention` sky-position REAL-RECOVERY-CONSISTENT result (Δ=12.58°, z=+23.23σ, SNR-monotonic)                                                     | Redo only                                                                                                                                                                                                                                     | Genuinely new discovery — the check (bootstrap + SNR-stratification on sky_position) did not exist on the original side at all | Original never ran `bootstrap_sky_separation.py`/`snr_stratification_sky_position.py`-equivalents; sky_position was treated only as a positive control (§2.4), never itself put through significance testing. Peripheral to the φc/ψ question but a real, well-evidenced result.                                                                                                                                                    |
| Inclination's small-but-real sub-floor signal (bootstrap significant, all models, both sides — though only 1/4 originally, 4/4 in the redo, per §3.5) | Both sides found *some* significant inclination signal; the redo found it universally                                                                                                                                                         | Genuinely new discovery in degree, not in kind                                                                                 | The original already had 1 significant model (cnn_attention, z=+3.17) and already flagged it as a population-level bias rather than pure noise (`thesis/chapter_phic_psi_degeneracy.md` §6.3); the redo's contribution is showing this generalizes to all 4 models and stating explicitly, for the first time on either side, that inclination is not a clean noise floor (`closing_summary.md`).                                   |
| poc_b's undisclosed ground-truth-assisted branch reconstruction for its individual-angle bootstrap test                                               | Present in both `bootstrap_ang_mae.py` runs; disclosed in neither's original write-up                                                                                                                                                         | Methodological gap, present symmetrically on both sides                                                                        | Not a redo-specific finding — it is a property of a script the redo ported verbatim from the original, inheriting the same undisclosed property. See §5.                                                                                                                                                                                                                                                                            |
| The perturbation-trace calibration-failure salvage (original closes A.3 via the paired-statistic channel; redo does not attempt the same salvage)     | Asymmetric response to a symmetric finding                                                                                                                                                                                                    | Procedural/judgment difference, not a new empirical result                                                                     | See §3.4.                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Adversarial AI review process, thesis chapter, arXiv paper                                                                                            | Original only                                                                                                                                                                                                                                 | Methodological gap — the redo simply never ran this phase                                                                      | Explicitly out of scope for the redo as scoped by its own `redo_procedure.md` (which is a build plan for the training/diagnostic phase, not the write-up phase); not an oversight, but a real asymmetry in how far each investigation's polish extends. See §6.                                                                                                                                                                     |
| The redo's Step 0 gate structurally covers 4 readings per round vs. the original's 2                                                                  | Redo only, by design                                                                                                                                                                                                                          | Population/config difference, not a fair like-for-like comparison                                                              | The redo's combo construction makes both source heads (coa_phase, polarization_angle) load-bearing for *both* combo_A and combo_B jointly, so its preregistration correctly requires all 4 to pass; the original's preregistration only ever had 2 pre-declared primary targets. This is a genuine, defensible difference in what "the gate" means on each side, not evidence either investigation was run more or less rigorously. |

## 5. The undisclosed methodological property (both investigations, corrected here)

Neither write-up (`diagnostic_log.md`, `NOTES.md`, the thesis chapter, the redo's `NOTES.md`) discloses that `poc_b`'s individual `coa_phase`/`polarization_angle` bootstrap numbers rest on ground-truth-assisted branch reconstruction.
Confirmed directly in the code on both sides: `bootstrap_ang_mae.py` calls `transforms.inverse(raw_pred)` generically for every model (`experiments/phic_psi_poc/bootstrap_ang_mae.py:200`, `experiments/phic_psi_poc-redo/bootstrap_ang_mae.py:216`), and for poc-mode models that inverse pass routes through the branch-disambiguation machinery documented in `experiments/phic_psi_poc-redo/formulae_reference.md` §A.8, which by construction needs the true angle to pick the correct branch among the naive 8 reconstruction candidates.
This is true in both `bootstrap_ang_mae.py` runs (`bootstrap_output/bootstrap_ang_mae_20260721_093533.md` and `bootstrap_output/bootstrap_ang_mae_20260916_113955.md`), flagged in neither's original reporting, and does not change either verdict — poc_b reads non-significant for `coa_phase`/`polarization_angle` on both sides regardless (§2.7) — because ground-truth-assisted branch selection can only bias a reconstructed prediction toward looking *more* accurate than a genuinely blind one, never less, so a non-significant result survives despite the bias rather than because of it.
It is stated explicitly here, on both investigations' behalf, per the task brief's instruction not to pass over it silently a second time.

## 6. Confidence synthesis

The strongest defensible physics conclusion, citing artifacts from both investigations, is this: for a two-detector, SNR-7–15, dominant-quadrupole-only population, point-estimate neural regression recovers no usable strain-only signal about either the coalescence phase or the polarization angle, individually or in their analytically well-motivated sum/difference combination, under either the originally-tested (`φc±2ψ`) or the corrected (`2φc±2ψ`) formula.
This rests on convergent evidence from two independent training campaigns, each spanning multiple architectures and head parameterizations: validation circular/combo loss pinned at the random-guessing value across every configuration tested on both sides (`experiments/phic_psi_poc/diagnostic_log.md` Run 7 Check 3 table; `experiments/phic_psi_poc-redo/NOTES.md` "Round 1" central-result entry, §2.5 above); a formal label-permutation bootstrap finding 0 of 8 φc/ψ tests significant on the original's checkpoints and 0 of 8 again on the redo's, most reading at or worse than chance (`bootstrap_output/bootstrap_ang_mae_20260721_093533.md`, `bootstrap_output/bootstrap_ang_mae_20260916_113955.md`, §2.7 above); no SNR-dependent or inclination-band-dependent recovery on either side beyond artifact scale (§2.8, §2.9); and, specific to the redo, a mechanism-level ruling-out of an implementation bug via a healthy end-to-end gradient chain on the real trained checkpoint (`diagnostic_output/diagnostic_checks_20260916_113913.log`, §2.10) plus a direct confirmation that the poc_a/poc_b collapse is a genuine training-dynamics outcome (near-delta-function output distributions), not a metric artifact (`analysis_output/analysis_report_20260916_113847.md` Health Check table).
The redo adds, and the original could not have produced, two qualifications to this conclusion that should travel with it: `cnn_attention` is capable of memorizing sample-specific phase/polarization structure in-training without any of it generalizing (both investigations' own memorization-gap checks, §4), so any future architecture search using `cnn_attention`-family models on these targets should watch for this specifically; and the "uninformative noise floor" convention this null's stratification checks lean on (inclination as a same-model calibration control) is itself carrying a small, real, population-level signal rather than being clean noise, on both investigations' own bootstrap evidence, most clearly in the redo (§2.7, §3.5).
Neither of these qualifications weakens the central null; both sharpen exactly what it does and does not claim.

## 7. Open items / recommended next steps

These are listed for the user to decide, not decided here, per the task brief.

1. **Should the redo get its own thesis chapter and adversarial-review pass**, now that its completeness audit is done (`experiment_index.md` "Redo-completeness audit" section) and Stage 2b(i) has substantially answered its central question (§3.2, Verdict B)? The original's chapter and 4 rounds of review (`thesis/reviews/`) represent a large fraction of that investigation's total effort; the redo currently has no equivalent artifact, only `NOTES.md`/`closing_summary.md`.
2. **Should `perturbation_trace_standalone.py final` be pursued** for the redo, given its `early` calibration failed and was never resolved (§2.12, §3.4)? The redo's own recommendation (`NOTES.md`, 2026-09-18) is that recalibrating the classifier's thresholds is real follow-up work, not a quick rerun, and that the central question is already well-answered by other means — but this is explicitly left as the user's call, not the redo's own conclusion.
3. **Is the preregistration reconciliation from §3.3 still an open decision?** The redo's `closing_summary.md` states the two verdicts (gated UNINTERPRETABLE vs. broader formal null) are not formally merged into one preregistered statement, and this document does not attempt that merge either — it explains why they resist a clean merge (§3.3) but leaves the decision of whether one is still wanted to the user.
4. **The inclination anomaly (§3.5)** is investigated as far as this repo's own artifacts allow (dataset identity confirmed, shared-code changes ruled out) but not conclusively resolved — the leading remaining hypothesis (GPU/library nondeterminism or an unpinned environment difference between the two training campaigns) cannot be checked further without an environment manifest neither campaign recorded. If the user wants this pinned down precisely, the next step is checking whatever TF/CUDA/cuDNN versions were active on the lab GPU machine on 2026-07-20 vs. 2026-09-15, not a rerun of any analysis script in this repo.
5. **The perturbation-trace procedural asymmetry (§3.4)** — whether the redo should adopt the original's paired-statistic salvage (which its own `early` data already partially supports for tcn/cnn_attention, per §2.12) or whether the original's provisional closure should itself be revisited given neither has ever run the fresh-holdout replication both investigations' own review cycles called for.
6. **`cnn_attention`'s sky-position result (§4)** is flagged REAL-RECOVERY-CONSISTENT but not mechanistically confirmed as cross-detector amplitude-ratio recovery specifically (`closing_summary.md`) — worth a targeted follow-up if the user wants to pursue this secondary finding on its own merits, independent of the φc/ψ question.
