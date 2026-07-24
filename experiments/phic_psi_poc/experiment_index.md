# φc/ψ Degeneracy PoC — Experiment Index

**Branch**: `poc/phic-psi-degeneracy`
**Last updated**: 2026-07-24

---

## Core narrative (read in this order)

| File | Description |
|------|-------------|
| [`experiment_summary_2026-07-22.md`](experiment_summary_2026-07-22.md) | Self-contained synthesis of the full investigation (Runs 1–9b) — best single entry point for onboarding; compresses `NOTES.md`/`diagnostic_log.md` |
| [`NOTES.md`](NOTES.md) | Running notes: design decisions, run log, setup, next steps |
| [`diagnostic_log.md`](diagnostic_log.md) | **Thesis reference**: chronological log of every diagnostic run, hypothesis tested, outcome, and wrong turns. Definitive record. |
| [`results.md`](results.md) | Detailed run-by-run results tables |
| [`tanh_to_linear_postmortem.md`](tanh_to_linear_postmortem.md) | Full postmortem of tanh→linear fix and normalize_unit pathology discovery (Runs 1–5) |
| [`inclination_loss_trace.md`](inclination_loss_trace.md) | Code-path trace: inclination uses Huber loss, no normalize_unit — separate failure mechanism |

---

## Verification — Sections A–E (2026-07-21)

| Section | File | Description |
|---------|------|-------------|
| Plan | [`run7_verification_plan.md`](run7_verification_plan.md) | Agreed 5-section verification plan (gating criteria before ι-conditioning) |
| Rebuttal (superseded) | [`run7_verification_rebuttal.md`](run7_verification_rebuttal.md) | Initial rebuttal — **superseded by actual verification results**. Retained for record. |
| **A.2** | [`std_ratio_trajectories.md`](std_ratio_trajectories.md) | Full epoch-by-epoch std_ratio trajectories for all 4 models, both periodic heads |
| **B** | [`poc_b_config_diff.md`](poc_b_config_diff.md) | poc_b vs poc_a config diff + collapse mechanism explanation |
| **C** | [`cnn_attention_config_diff.md`](cnn_attention_config_diff.md) | cnn_attention vs tcn config diff + outlier explanation |
| **D** | [`bootstrap_output/bootstrap_ang_mae_20260721_093533.md`](bootstrap_output/bootstrap_ang_mae_20260721_093533.md) | Bootstrap CI on ang_MAE: N=10,000 shuffles, all models, all periodic heads |
| — | (validation ordering check) | Data is i.i.d. (window variance ratio=0.99). Bootstrap shuffle-null is valid — no row-ordering confound. |
| **E** | [`snr_output/snr_stratification_20260721_094039.md`](snr_output/snr_stratification_20260721_094039.md) | SNR-stratified ang_MAE: tercile analysis, all models, all periodic heads |
| — (2026-07-23) | [`inclination_output/inclination_stratification_20260723_130630.md`](inclination_output/inclination_stratification_20260723_130630.md) | Adversarial-review follow-up: inclination-stratified ang_MAE (face-on/mixed/edge-on bands per §3, all models, all periodic heads). No cross-model edge-on-favoring recovery; every φ_c/ψ deviation falls within its own model's ι-control noise floor except tcn/φ_c (wrong-signed, already-flagged std_ratio instability). Written up as Table 6.6 / §6.7 in the thesis chapter. |
| — (2026-07-23) | [`inclination_output/inclination_control_stratification_20260723_140016.md`](inclination_output/inclination_control_stratification_20260723_140016.md) | v2-review follow-up: same face-on/mixed/edge-on bands applied to chirp mass (known-good, non-angular). R² spread 0.005–0.010, MAE spread 0.034–0.080 M_⊙ across bands for every model — banding itself does not inject variance at a scale relevant to the φ_c/ψ deviations. Written up as Table 6.7 / §6.7. |

---

## Run 8/9 — λ ablation and retune (2026-07-21 – 2026-07-22)

| Item | File | Description |
|------|------|-------------|
| Run 8 (λ=0 ablation) | [`assessment_lam0_ablation_2026-07-22.md`](assessment_lam0_ablation_2026-07-22.md) | Write-up: isolates Run 7's val-loss creep as a λ/log-var interaction artifact (3/4 clean); closes item F.1/F.2 |
| Run 8 outputs | [`lam0_ablation_output/`](lam0_ablation_output/) | Auto-generated report + trajectories, λ=0 |
| Pre-registration | [`preregistration_lam_retune.md`](preregistration_lam_retune.md) | Locked decision criteria for Run 9a/9b, written before either result existed — do not edit retroactively |
| Run 9a (λ=0.05) | [`lam005_retune_output/lam005_retune_report.md`](lam005_retune_output/lam005_retune_report.md), [`lam005_retune_output/diagnostic_lam005_retune_20260722_142705.md`](lam005_retune_output/diagnostic_lam005_retune_20260722_142705.md) | Both primary targets (tcn coa_phase, poc_a pol_angle) failed the Step 0 gate — close but unhealthy in the last 40 epochs |
| Run 9b (λ=0.10) | [`lam010_retune_output/lam010_retune_report.md`](lam010_retune_output/lam010_retune_report.md), [`lam010_retune_output/diagnostic_lam010_retune_20260722_171025.md`](lam010_retune_output/diagnostic_lam010_retune_20260722_171025.md) | Both primary targets failed the gate again, worse than λ=0.05 — verdict: λ alone insufficient, neither null nor counter-evidence |

Scripts: [`run_lam0_ablation.py`](run_lam0_ablation.py), [`run_lam005_retune.py`](run_lam005_retune.py),
[`run_lam010_retune.py`](run_lam010_retune.py) (each chains train→plot→evaluate→diagnostic, then overlays
the λ sweep); [`diagnostic_lam005_retune.py`](diagnostic_lam005_retune.py),
[`diagnostic_lam010_retune.py`](diagnostic_lam010_retune.py) (mechanical Step 0–3 gate evaluation, per
`preregistration_lam_retune.md`). Configs: [`config_lam0_ablation.yaml`](config_lam0_ablation.yaml),
[`config_lam0_ablation_tcn.yaml`](config_lam0_ablation_tcn.yaml),
[`config_lam005_retune.yaml`](config_lam005_retune.yaml),
[`config_lam005_retune_tcn.yaml`](config_lam005_retune_tcn.yaml),
[`config_lam010_retune.yaml`](config_lam010_retune.yaml),
[`config_lam010_retune_tcn.yaml`](config_lam010_retune_tcn.yaml).

Full narrative and verdict language: `NOTES.md` / `diagnostic_log.md`, Run 9a/9b sections;
compressed synthesis in [`experiment_summary_2026-07-22.md`](experiment_summary_2026-07-22.md) §3.

---

## Closing phase — A.3 trace, review cycle, thesis chapter (2026-07-23)

| Item | File | Description |
|------|------|-------------|
| Punch list | [`phic_psi_closing_punch_list.md`](phic_psi_closing_punch_list.md) | Reviewer-supplied closing checklist separating pre-finalization items from future work; all items dispositioned (see `diagnostic_log.md` same-date sections) |
| Trace script | [`perturbation_trace_standalone.py`](perturbation_trace_standalone.py) | Un-gated A.3 trace; `final` and `early` (calibration) stages, per-sample paired statistics, per-step batch-loss logging |
| Trace outputs | [`perturbation_trace_output/`](perturbation_trace_output/) | Auto-generated reports/logs: first run (091229), final-stage rerun with paired stats (095054), early-stage calibration (095357). **Reading note:** the `Verdict` column is the retired geometry classifier — read the numeric columns |
| Reviewer memo | [`reviewer_response_a3_2026-07-23.md`](reviewer_response_a3_2026-07-23.md) | Point-by-point response to the A.3 review (per-case statistics, calibration outcome, classifier retirement, mchirp-degradation explanation, MDE note) |
| Thesis chapter | [`thesis/chapter_phic_psi_degeneracy.md`](thesis/chapter_phic_psi_degeneracy.md), [`thesis/chapter_phic_psi_degeneracy.tex`](thesis/chapter_phic_psi_degeneracy.tex) | The chapter, one document in two formats (edit both in the same session); claim-to-artifact map in its appendix |
| Prose tooling | [`thesis/sentence_per_line.py`](thesis/sentence_per_line.py) | One-sentence-per-line reflow tool for prose files (see root `CLAUDE.md` conventions) |
| v4 review checklist | [`thesis/reviews/v4_correction_checklist.md`](thesis/reviews/v4_correction_checklist.md) | Item-by-item cross-check of all three v4 adversarial reviews against the chapter's then-current text; drove the Round 4 text edits (see below) |

---

## Analysis outputs

### Run 7 (magnitude penalty λ=0.01, 2026-07-20)

| File | Description |
|------|-------------|
| [`analysis_output/analysis_report_20260720_234304.md`](analysis_output/analysis_report_20260720_234304.md) | Consolidated prediction analysis: scalar heads, periodic heads, sky position, health check |
| [`analysis_output/scalar_heads_20260720_234304.csv`](analysis_output/scalar_heads_20260720_234304.csv) | mchirp, merger_time, snr per-model stats |
| [`analysis_output/periodic_heads_20260720_234304.csv`](analysis_output/periodic_heads_20260720_234304.csv) | coa_phase, pol_angle, inclination per-model circular stats |
| [`analysis_output/sky_position_20260720_234304.csv`](analysis_output/sky_position_20260720_234304.csv) | Sky position angular errors |
| [`analysis_output/health_check_20260720_234304.csv`](analysis_output/health_check_20260720_234304.csv) | Per-model per-head health grades |
| [`analysis_output/health_check_20260720_234304.png`](analysis_output/health_check_20260720_234304.png) | Health check heatmap |
| [`analysis_output/histogram_coa_phase_20260720_234304.png`](analysis_output/histogram_coa_phase_20260720_234304.png) | φc prediction distributions |
| [`analysis_output/histogram_polarization_angle_20260720_234304.png`](analysis_output/histogram_polarization_angle_20260720_234304.png) | ψ prediction distributions |
| [`analysis_output/histogram_inclination_20260720_234304.png`](analysis_output/histogram_inclination_20260720_234304.png) | ι prediction distributions |
| [`analysis_output/scatter_mchirp_20260720_234304.png`](analysis_output/scatter_mchirp_20260720_234304.png) | True vs predicted mchirp |
| [`analysis_output/scatter_merger_time_20260720_234304.png`](analysis_output/scatter_merger_time_20260720_234304.png) | True vs predicted merger_time |
| [`analysis_output/scatter_snr_20260720_234304.png`](analysis_output/scatter_snr_20260720_234304.png) | True vs predicted SNR |

### Pre-post comparison (tanh→linear fix)

| File | Description |
|------|-------------|
| [`pre_post_comparison.csv`](pre_post_comparison.csv) | Pre-fix vs post-fix metrics across all architectures |

### Step 1.1 prereq sweep

| File | Description |
|------|-------------|
| [`sweep_1_1_ratio_vs_iota.csv`](sweep_1_1_ratio_vs_iota.csv) | Combo well-constrained ratio vs inclination (grid sweep + bootstrap) |
| [`sweep_1_1_ratio_vs_iota.png`](sweep_1_1_ratio_vs_iota.png) | Ratio vs ι plot |

---

## Diagnostic outputs (chronological)

### Run 7 diagnostics (2026-07-21) — definitive

| File | Description |
|------|-------------|
| [`diagnostic_output/diagnostic_checks_20260721_000331.log`](diagnostic_output/diagnostic_checks_20260721_000331.log) | Checks 1–7 on Run 7 (magnitude penalty) checkpoints |

### Earlier diagnostic runs (historical)

| Run | Date | Log file | Key finding |
|-----|------|----------|-------------|
| 1 | 2026-07-18 12:50 | [`diagnostic_checks_20260718_125027.log`](diagnostic_output/diagnostic_checks_20260718_125027.log) | Checks 1–4. Data pipeline clean. Loss wiring bug found. Log-vars frozen. |
| 2 | 2026-07-18 13:17 | [`diagnostic_checks_20260718_131719.log`](diagnostic_output/diagnostic_checks_20260718_131719.log) | Check 4 smoking gun: φc/ψ Δ=0.00. Tanh saturation ruled out (Check 5). |
| 3 | 2026-07-18 13:41 | [`diagnostic_checks_20260718_134134.log`](diagnostic_output/diagnostic_checks_20260718_134134.log) | Check 6: **tanh saturation confirmed**. Root cause found. |
| 4 | 2026-07-18 | [`diagnostic_checks_20260718_140622.log`](diagnostic_output/diagnostic_checks_20260718_140622.log) | Check 7: saturation at init (step 0). Born dead. |

### Diagnostic plots

| File | Description |
|------|-------------|
| [`diagnostic_output/true_label_distributions.png`](diagnostic_output/true_label_distributions.png) | True label histograms (Check 1) |
| [`diagnostic_output/logvar_trajectories.png`](diagnostic_output/logvar_trajectories.png) | Log-var trajectories over training (Check 3) |
| [`diagnostic_output/combo_loss_trajectories.png`](diagnostic_output/combo_loss_trajectories.png) | Circular/combo loss trajectories (Check 3) |
| [`diagnostic_output/true_label_stats.csv`](diagnostic_output/true_label_stats.csv) | True label statistics |

---

## Configuration files

| File | Mode | Trunk | Description |
|------|------|-------|-------------|
| [`config_baseline.yaml`](config_baseline.yaml) | baseline | tcn | poc_a — circular loss on individual φc/ψ |
| [`config_poc.yaml`](config_poc.yaml) | poc | tcn | poc_b — combo heads + curriculum weighting |
| [`config_tcn.yaml`](config_tcn.yaml) | baseline | tcn | Plain TCN baseline |
| [`config_cnn_attention.yaml`](config_cnn_attention.yaml) | baseline | cnn_attention | CNN + transformer + attention pooling |
| [`config_cnn_baseline.yaml`](config_cnn_baseline.yaml) | baseline | cnn_baseline | Plain CNN baseline (not in Run 7) |
| [`config_inception_time.yaml`](config_inception_time.yaml) | baseline | inception_time | InceptionTime (not in Run 7) |
| [`config_resnet1d.yaml`](config_resnet1d.yaml) | baseline | resnet1d | ResNet1D (not in Run 7) |

---

## Scripts

### Training

| File | Description |
|------|-------------|
| [`train_poc.py`](train_poc.py) | Main training entry point — builds SumDiffTrainer, runs training |
| [`run_full.py`](run_full.py) | Batch runner: train → plot → evaluate for all configs |
| [`run_magnitude_fix.py`](run_magnitude_fix.py) | Run 7 launcher: chains train→plot→eval for poc_a, poc_b, TCN, CNN Attention |

### Analysis & diagnostics

| File | Description |
|------|-------------|
| [`analyse_predictions.py`](analyse_predictions.py) | Load all models, predict on validation, compute per-head stats, generate CSVs + markdown + PNGs |
| [`diagnostic_checks.py`](diagnostic_checks.py) | Seven deep diagnostic checks (true labels, loss wiring, log-var trajectory, gradient routing, logit saturation, gradient chain, init saturation timing) |
| [`bootstrap_ang_mae.py`](bootstrap_ang_mae.py) | Section D: bootstrap CI on ang_MAE (N=10,000 shuffles) |
| [`snr_stratification.py`](snr_stratification.py) | Section E: SNR-stratified ang_MAE (tercile analysis) |
| [`inclination_stratification.py`](inclination_stratification.py) | Adversarial-review follow-up: inclination-stratified ang_MAE (face-on/mixed/edge-on bands per §3) — **run 2026-07-23**; results in `inclination_output/`, written up as Table 6.6 / §6.7 |
| [`plot_certified_memorization.py`](plot_certified_memorization.py) | v2-review follow-up: train-vs-validation circular loss for the two certified models (poc_b, cnn_attention) specifically, read directly from each run's `history.csv` — no model loading, no GPU. Output: `diagnostic_output/certified_models_train_val_loss.png`, written up as Fig. 8.1 / §8.2 |
| [`inclination_control_stratification.py`](inclination_control_stratification.py) | v2-review follow-up: chirp-mass MAE/R² stratified by the same face-on/mixed/edge-on bands as `inclination_stratification.py`, to check whether banding itself injects spurious variance into a known-good non-angular head — **run 2026-07-23**; R² spread only 0.005–0.010 across bands for every model, ruling out banding-induced noise at a scale relevant to Table 6.6. Results in `inclination_output/`, written up as Table 6.7 / §6.7 |
| [`prereq_checks.py`](prereq_checks.py) | Step 1.1–1.6 prerequisite verification (combo ratio sweep, w(ι) derivation, cos ι histogram) |
| [`validation_script.py`](validation_script.py) | Manual validation utilities |

### Evaluation & plotting

| File | Description |
|------|-------------|
| [`evaluate_poc.py`](evaluate_poc.py) | Per-model evaluation on train/validation splits |
| [`plot_poc.py`](plot_poc.py) | Training history plots |
| [`scripts/evaluate.py`](scripts/evaluate.py) | Evaluation subprocess (called by run_full.py) |
| [`scripts/plot_run.py`](scripts/plot_run.py) | Plotting subprocess (called by run_full.py) |
| [`scripts/train.py`](scripts/train.py) | Training subprocess (called by run_full.py) |

### Shared plotting utilities

| File | Description |
|------|-------------|
| [`plot_style.py`](plot_style.py) | Shared matplotlib style (`update_style()`) + `SERIES_COLORS` semantic color dict for all `experiments/` plotting scripts. Deliberate duplicate of `src/gwml/evaluation/plot_style.py` (kept in sync by hand, not imported cross-package). Retrofitted into every plotting entry point 2026-07-23 — see `NOTES.md`'s "Plotting-style retrofit" entry for the full list of legend/color/grid fixes and which figures were actually regenerated vs. still pending a lab-machine rerun. |

---

## Supporting modules (in experiment directory)

| File | Description |
|------|-------------|
| [`trainer.py`](trainer.py) | `SumDiffTrainer` — extends MultiHeadTrainer with combo loss, curriculum weighting, magnitude penalty |
| [`curriculum.py`](curriculum.py) | `w(ι)` derivation (Jacobian condition number sweep), TF wrapper, cos ι histogram |
| [`transform_utils.py`](transform_utils.py) | `normalize_unit`, `complex_mul`, `combo_labels` — numpy + TF implementations |

---

## Quick-reference: key results at a glance

### Does the magnitude penalty prevent |v| drift?
**Yes**, for 2/4 models (poc_b, cnn_attention) at λ=0.01. poc_a pol_angle and tcn coa_phase
were retuned at λ=0.05 and λ=0.10 (Run 9a/9b) — **both still fail the std_ratio
interpretability gate at every λ tried, and get worse at λ=0.10.** Filed as "λ alone
insufficient," not resolved by tuning.
→ [`std_ratio_trajectories.md`](std_ratio_trajectories.md) (λ=0.01 baseline),
[`lam005_retune_output/`](lam005_retune_output/), [`lam010_retune_output/`](lam010_retune_output/)

### Does the circular loss decrease during training?
**No.** Flat at ~1.0 (random baseline) for all 80 epochs, all models, all heads.
→ [`diagnostic_output/diagnostic_checks_20260721_000331.log`](diagnostic_output/diagnostic_checks_20260721_000331.log) (Check 3)

### Is ang_MAE distinguishable from random?
**No** — 11/12 model×head combinations non-significant. coa_phase & pol_angle: 0/4 significant.
→ [`bootstrap_output/bootstrap_ang_mae_20260721_093533.md`](bootstrap_output/bootstrap_ang_mae_20260721_093533.md)

### Does higher SNR help?
**No** — no model shows SNR-dependent ang_MAE improvement on any periodic head.
→ [`snr_output/snr_stratification_20260721_094039.md`](snr_output/snr_stratification_20260721_094039.md)

### Is poc_b's worse collapse a config bug?
**No** — it's the curriculum+degeneracy interaction. The curriculum funnels gradient through one underdetermined combo channel.
→ [`poc_b_config_diff.md`](poc_b_config_diff.md)

### Is cnn_attention's lower circ_r evidence of phase learning?
**No** — it's a feature-variance artifact from learned attention pooling. ang_MAE is at baseline.
→ [`cnn_attention_config_diff.md`](cnn_attention_config_diff.md)

### What's still open?
**Nothing from the Run 7 verification battery, beyond A.3's own pending replication.** The
last item (A.3, the 89× asymmetry) was provisionally closed 2026-07-23 after a three-round
sequence: trace executed → review caught a failed mchirp positive control → paired statistics
+ an `early` calibration stage added → calibration FAILED its pre-stated criterion (the
displacement-geometry classifier labeled the fastest-learning head "noise-like" and was
retired), while the paired probe-loss channel passed its control in the same run and reads
every periodic head as null at both stages. Verdict: radial movement without angular learning;
tcn/coa_phase escalation branch extinguished (paired t = −0.20). Per the Round 3 v3 review pass
(below), the chapter now states this closure as *provisional, pending replication on a fresh
holdout set*, since the surviving channel was adopted post hoc rather than pre-registered —
the replication itself is not yet scheduled. Remaining threads are scoped future work
(architecture-level std_ratio fix, finer λ mini-sweep, ι-conditioning, and — as of the
2026-07-24 consistency audit, now also carried as chapter §8.5 items rather than only in
`NOTES.md`/this index — the synthetic poc_b ablation, the high-SNR/Fisher-bound spot-check,
the ridge-structure check, and dataset regeneration from a version-controlled script).
→ [`perturbation_trace_output/`](perturbation_trace_output/),
[`diagnostic_log.md`](diagnostic_log.md) (calibration adjudication, 2026-07-23)

**Item from the adversarial-review pass (2026-07-23) — now closed:** three AI reviewers (DeepSeek,
Kimi, Qwen) independently flagged that the chapter never stratifies φ_c/ψ ang_MAE by
inclination, despite the analytic degeneracy being inclination-dependent. `inclination_stratification.py`
(mirrors `snr_stratification.py`, bins on |cos ι| against the face-on/mixed/edge-on bands of §3)
ran on the lab machine same-day. Result: no model shows a cross-consistent, edge-on-favoring
recovery of either angle; every φ_c/ψ deviation falls at or below its own model's ι-control
noise floor except tcn/φ_c, which exceeds it but in the wrong (worse-than-null) direction on
the one head already flagged elsewhere as never achieving certified metric health. Written up
as Table 6.6 / new §6.7 in the chapter; §8.1's evidence list and §8.5's future-work list updated
accordingly. **Nothing from the adversarial-review pass remains open.**
Same pass also caught and fixed a real error in §6.4 — the two numbers themselves (0.027 rad
high-tercile-vs-null, 0.090 rad low-to-high swing) were both correct, but the text conflated
them as if reporting the same comparison; rewritten to state both correctly and explain, via
the low tercile's own anomalous departure from null, why the 0.090 rad swing still doesn't
read as a real effect — added a waveform-approximant provenance caveat (IMRPhenomD, dominant-mode-only, confirmed
by the author but not recorded anywhere in the repo/dataset), and flagged the inclination
prior as uniform-in-ι rather than the astrophysically correct uniform-in-cos ι.
→ [`thesis/reviews/`](thesis/reviews/) (the three review files), `thesis/chapter_phic_psi_degeneracy.{md,tex}` (revised)

**Round 2 (2026-07-23, same day):** three v2 adversarial reviews of the revised chapter
(`thesis/reviews/*_v2.md`) — Qwen: "accept as is" (one stale copy-edit, already fixed);
DeepSeek: "defensible" with six mostly-wording asks; Kimi: substantially harder, arguing the
null is "certified on two points in configuration space" and demanding new evidence.
**Addressed by text/analysis, no new training:**
- Verified from existing `history.csv` (no new run) that the memorization-gap argument in
  §8.2 only directly covers cnn_attention among the two certified models — poc_b (and poc_a,
  tcn) show *no* train/val gap at all. Wrote [`plot_certified_memorization.py`](plot_certified_memorization.py)
  (reads existing CSVs only) and added Fig. 8.1 + prose arguing this is not a capacity failure
  (same TCN trunk hits R²=0.97 on chirp mass) and, if anything, a cleaner null for poc_b.
- Reworded the §7.3 "exhausted" language (tripped up DeepSeek in both v1 and v2) to remove
  the friction entirely rather than re-argue it.
- Reframed §8.3's waveform-provenance bullet around reproducibility (Kimi/DeepSeek's actual
  concern) rather than only "it would sharpen the null" (which Qwen found sufficient but the
  other two did not).
- Added an honest "this is a diagnostic calibration, not a formal statistical test" caveat to
  §6.7, plus a sky-position-based rebuttal to Kimi's "shared upstream angular-pathology" concern.
  Prepared and then ran the cheap chirp-mass-by-inclination-band follow-up
  (`inclination_control_stratification.py`) rather than leaving it as future work: R² spread
  only 0.005–0.010 across bands for every model, ruling out banding-induced noise as an
  alternative explanation for the φ_c/ψ deviations — written up as Table 6.7 in §6.7.
- Softened "memorization" language (§5.5), tightened the certified-vs-corroborating framing (§1),
  reframed the perturbation trace as supporting-not-primary evidence (§9), added a Table 6.6
  caption clarification and an honest scope-limit note on the SNR-monotonicity dismissal (§6.3).
**Consciously deferred, not attempted — needs a scope decision, not a text fix:**
- Kimi's demand for a synthetic-data ablation (deterministic recoverable A/B combinations) to
  rule out the curriculum itself forcing poc_b's collapse — this is a new training experiment.
- Kimi's demand for a high-SNR (25–30) validation set or a Fisher-matrix bound, to distinguish
  "degeneracy" from "signal below the SNR 7–15 sensitivity floor" — new injections or a nontrivial
  new physics calculation.
- Kimi's demand to regenerate the dataset (or a small validation slice) from a version-controlled
  generator script and re-confirm the null — closes the provenance gap for good but is new data
  generation, not a chapter edit.
- Kimi's either/or on A.3: pre-register a fresh-holdout replication of the paired-statistic trace,
  or downgrade A.3 from "closed" to "open." Chose a middle path (Conclusion now states the
  perturbation trace is corroborating, not primary, evidence) rather than either extreme.
None of these four are done; they are the actual next-tier asks if the chapter needs to withstand
a harder examiner than this round.

**Round 3 (2026-07-23, same day):** a third Kimi-only review (`thesis/reviews/kimi_ai_adversarial_review_v3.md`;
DeepSeek and Qwen did not return for a v3 pass) raised 7 numbered issues, all framing/wording gaps
rather than requests for new numerical results — addressed entirely by text edits to
`thesis/chapter_phic_psi_degeneracy.{md,tex}`, no new runs.
**Addressed by text/framing:**
- §2.1 gained a scope paragraph stating ι-conditioning is a training-paradigm design choice
  motivated by §3's analytic structure, not a forced move necessitated by proof that ι is
  unlearnable (closes the "Step 0" logical-gap objection — the ι head failure, §5.4, is
  unresolved, not diagnosed as fundamental).
- §6.7 gained a caveat sentence on the head-capacity confound (ι, φ_c, ψ share one MLP head
  architecture) — acknowledged as plausible-but-unproven, not elevated to a new deferred
  experiment (user decision: caveat only).
- §6.2's poc_b framing softened to "consistent with the null but confounded by curriculum
  design"; §8.2 now states explicitly that the architecture-sufficiency argument rests
  primarily on cnn_attention, with poc_b as a distinct confounded consistency check.
- §6.6's "A.3 is closed" changed to "provisionally closed, pending replication on a fresh
  holdout set"; §9 extended to note the post-hoc channel selection means the trace cannot
  independently close any confound on its own.
- §6.3's p=0.0007 cnn_attention/inclination outlier now reported as a genuine anomaly that
  does not generalize to φ_c/ψ, rather than dismissed outright.
- §7.1 gained an explicit rationale for the coarse 4-point λ grid with a guaranteed-termination
  stopping rule.
- Editorial note reworded: the high-SNR spot-check and dataset-provenance items reframed from
  "deliberately not pursued" to "critical future work"; the A.3 item marked as now reflected in
  the chapter body.
- 10 stray `../path/....png` references in `.md` figure captions removed (`.tex` never had this
  issue).
**Consciously still deferred (re-confirmed, unchanged from Round 2):** the synthetic-ablation,
high-SNR/Fisher-bound, and dataset-regeneration items remain undone.
→ [`thesis/reviews/kimi_ai_adversarial_review_v3.md`](thesis/reviews/kimi_ai_adversarial_review_v3.md),
`thesis/chapter_phic_psi_degeneracy.{md,tex}` (revised), `NOTES.md` ("v3 adversarial review pass" entry)

**Round 4 (2026-07-24):** three v4 reviews (`thesis/reviews/*_v4.md`) — all three reviewers
returned, and unlike Round 2/3 the feedback was overwhelmingly tonal: DeepSeek ("thesis-ready,"
3 wording asks + an optional roadmap addition), Kimi (5 wording/framing demands, no new logical
gap), Qwen ("a masterwork," 4 defensive additions, one implying new analysis rather than text).
A background agent built a full item-by-item checklist cross-referencing all three reviews
against the chapter's then-current text before any edits, catching several items that were only
*partially* addressed despite looking fixed at a glance (`thesis/reviews/v4_correction_checklist.md`).
**Addressed by text only, no new numbers:** §1 roadmap sentence foreshadowing the
inclination-conditioning successor; §2.1 gained two sentences stating explicitly that no model
in this chapter takes ι as an input at any stage (ι is a control *output* head only) and pointing
the input-channel design at §8.5(i) by name — a first-pass wording that stated true (sin ι, cos ι)
would be fed at both training and inference time was caught by the user as reading like a
description of *this chapter's* models and corrected same-session;
§6.2 now states poc_b's combination-head losses explicitly (already-known §5.5 numbers,
relocated); §6.7 gained a carrier-phase-vs-envelope caveat on the sky-position analogy, a
Huber-vs-circular-loss caveat on the ι-noise-floor comparison, a downgraded "most likely to
break it" → actual ~1.05–1.08× predicted-advantage framing, and its head-capacity caveat moved
to the section's literal end; §8.2's cnn_attention/poc_b paragraph rewritten so their evidential
weight is not stated jointly, and its "statement about the data" claim now excludes the angular
heads' shared-MLP confound explicitly; §9's "every model" softened to "every configuration
tested."
**Deferred, flagged to user rather than applied silently:** Qwen's "ridge test" — a 2D joint
scatter of predicted (φ̂_c, ψ̂) pairs to rule out the network having learned the degenerate
manifold itself — needs GPU inference against saved checkpoints; no such script or per-sample
prediction artifact exists in the repo. User decision: defer to future work rather than run now
or add Qwen's suggested "(data not shown)" phrasing (which would have violated this project's
artifact-traceability rule). Added as new item (iv) in §8.5, explicitly inference-only/no-retraining.
→ [`thesis/reviews/v4_correction_checklist.md`](thesis/reviews/v4_correction_checklist.md),
`thesis/chapter_phic_psi_degeneracy.{md,tex}` (revised), `NOTES.md` ("v4 adversarial review pass" entry)

**Correction (2026-07-24, consistency audit):** this Round-4 entry and the corresponding
NOTES.md entry each recorded only the ridge test above as a deferred item, while the other
three deferred items (synthetic poc_b ablation, high-SNR/Fisher check, dataset regeneration —
named in the Round 2/3 entries above) lived only in the chapter's editorial note, not in §8.5
proper. All four are now §8.5 items (iv)–(vii) in both chapter formats, so the editorial note
can be removed at final submission without losing any of them.
