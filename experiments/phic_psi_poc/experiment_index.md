# φc/ψ Degeneracy PoC — Experiment Index

**Branch**: `poc/phic-psi-degeneracy`
**Last updated**: 2026-07-22

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
One item: the perturbation-trace **calibration run**. The trace itself executed
2026-07-23 ([`perturbation_trace_standalone.py`](perturbation_trace_standalone.py),
reading: coherent but dominantly *radial* raw-output drift — movement without angular
learning), but same-day review found its positive control failed (mchirp AMBIGUOUS at
all four converged checkpoints), so **A.3 is provisionally closed**, pending the
script's `early` stage (fresh init + ~1-epoch warmup, paired statistics) showing
mchirp directional-early/ambiguous-late. That run also re-adjudicates the nominal
tcn/coa_phase escalation trigger (net/sum 0.925, Δcirc −0.010).
→ [`perturbation_trace_output/`](perturbation_trace_output/),
[`diagnostic_log.md`](diagnostic_log.md) (A.3 closure section + review addendum, 2026-07-23)
