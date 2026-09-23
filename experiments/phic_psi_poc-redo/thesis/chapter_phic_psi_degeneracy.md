# Chapter — On the Recoverability of Coalescence Phase and Polarization Angle from Gravitational-Wave Strain: A Systematically Verified Null Result Under the Corrected Combination Formula

> **Draft status.** Chapter number, cross-references to other chapters, and citation keys (`[@…]`) are placeholders to be resolved against the thesis bibliography.
> All quantitative values are taken verbatim from this study's experiment record on branch `poc/phic-psi-degeneracy-redo` (primary sources: `NOTES.md`, `closing_summary.md`, and the auto-generated reports under `experiments/phic_psi_poc-redo/`); a claim-to-artifact map is given in the chapter appendix.

> **A note on evidentiary architecture.** A pre-registered metric-health gate was meant to license a clean, certified test of the degeneracy: it never passed — at any regularization strength tried, for either combination vector, in either training mode.
> The null defended below therefore rests on a different kind of evidence: a broad, convergent battery of checks run across all four down-selected models, not a narrow gated subset.
> This is disclosed up front because it is the single most important qualifier on this chapter's argument, not a detail to be buried in the Discussion.

## 1. Introduction

Deep-learning approaches to gravitational-wave parameter estimation have matured rapidly, from early proof-of-concept classifiers [@george2018deep] to amortized neural posterior estimators that approach the accuracy of stochastic-sampling pipelines [@gabbard2022vitamin; @dax2021dingo] at a fraction of their latency.
Most of this progress has concentrated on parameters that imprint strongly and non-degenerately on the observed strain: the chirp mass through the phase evolution [@cutler1994gw], the merger time through the temporal localization of the signal, the signal-to-noise ratio through overall amplitude, and the sky position through inter-detector time delays and antenna-pattern amplitudes.

Two extrinsic parameters have received far less attention as direct regression targets: the orbital phase at coalescence, φ_c, and the polarization angle, ψ.
For the dominant quadrupole (ℓ = 2, |m| = 2) mode of a quasi-circular compact binary, the two polarization states are

h₊ ∝ (1 + cos²ι) cos(2φ(t) + 2φ_c), h_× ∝ 2 cos ι sin(2φ(t) + 2φ_c),

and the detector response mixes them through antenna patterns that depend on 2ψ [@cutler1994gw; @sathyaprakash2009physics].
Both φ_c and ψ enter the strain doubled, so the analytically degenerate combination in the face-on limit is **2φ_c ± 2ψ**, with the sign set by the sign of cos ι.
The correction traces directly to how `coa_phase` is injected in this dataset's generator: it is PyCBC/LALSimulation's orbital reference phase, and for a dominant-mode-only waveform that orbital phase enters the strain through `e^{i·2·coa_phase}`, exactly like ψ already does.
In the face-on limit (cos ι → ±1) the signal becomes circularly polarized and depends on φ_c and ψ only through this single combination, so the individual angles are exactly unidentifiable there, and only partially disentangled as the inclination moves toward edge-on.
Bayesian samplers absorb this structure into broad, multimodal joint posteriors [@veitch2015lalinference]; a point-estimate regression network has no such refuge.

This chapter asks:

> **Can φ_c and ψ be recovered from strain alone by a deep neural network — individually, or through their better-conditioned sum/difference combinations — or is the degeneracy effectively exact for a realistic detection-scale population?**

The answer is a **null result**: across four down-selected architectures, two head parameterizations, three values of the magnitude-regularization coefficient, and a comprehensive diagnostic battery, no model performed measurably better than random guessing on either angle, while the same networks simultaneously recovered chirp mass, merger time, signal-to-noise ratio, and sky position with high fidelity from the same shared features.
No model here ever cleared the pre-registered metric-health gate that would have certified it as a clean test.
The null defended here instead rests on convergence across four independent lines of evidence — direct inspection of the prediction distributions, a code-level gradient-health trace ruling out an implementation bug, a formal label-permutation bootstrap, and population stratification by inclination and signal-to-noise ratio — applied uniformly to all four down-selected models, without excluding any on interpretability grounds.
This study additionally surfaced two findings: a genuine, SNR-monotonic partial sky-localization skill in one architecture (`cnn_attention`), unrelated to the φ_c/ψ question, and a small but statistically real sub-floor signal in the inclination control head, which had been assumed to be clean noise.

The chapter is organized as follows, with one section deliberately compressed.
Section 2 formalizes the problem and the circular-regression framework.
Section 3 presents the analytic prerequisite study.
Section 4 details the datasets, architectures, losses, and training protocol, including the combination construction.
Section 5 summarizes, briefly, two optimization pathologies in this training pipeline (tanh saturation, magnitude drift) and how they were fixed prior to the main sweep.
Section 6 presents the defended null result, its verification battery, and the two new findings.
Section 7 describes the pre-registered magnitude-penalty retune and its outcome.
Section 8 discusses interpretation, limitations, and future work; Section 9 concludes.

## 2. Problem Formulation

### 2.1 Targets and the degeneracy hypothesis

We consider a ten-parameter description of a compact binary coalescence signal injected into two-detector noise: chirp mass M, mass ratio q, inclination ι, coalescence phase φ_c, polarization angle ψ, declination, right ascension, injection time, merger time within the analysis window, and network signal-to-noise ratio.
Seven of these are used as supervised regression targets (mass ratio and injection time are excluded; §4.1).
The scientific targets are φ_c ∈ [0, 2π), period 2π, and ψ ∈ [0, π), period π.
Inclination, chirp mass, merger time, SNR, and sky position are retained as *control heads*: parameters trained through the identical pipeline whose success or failure calibrates what the network can extract from strain when information is present.

The **degeneracy hypothesis** under test is: *φ_c and ψ carry no strain-only recoverable signal for this population, in this architecture and loss family* — a point-estimating network can do no better than the optimal constant prediction, because for most of the population the likelihood constrains only the combination `2φ_c ± 2ψ`, and the remaining direction is unconstrained.
The alternative hypothesis motivating the experimental design is that even if φ_c and ψ are individually unrecoverable, their **sum and difference combinations** may be well-conditioned: a network parameterized in `2φ_c ± 2ψ` combination space, with a curriculum that weights each combination according to how well the current inclination constrains it, might learn the recoverable structure that a naive per-angle parameterization misses.

Conditioning on inclination ι — tested as future work in §8.5 — is treated throughout as a *training-paradigm design choice*, motivated by the analytic structure of §3 showing that the well-constrained combination is recoverable given ι, not as a forced move necessitated by a proof that ι itself is unrecoverable; §6.7 shows ι's own behavior is more complicated than clean noise — partial mode collapse, not a real per-sample signal — which still leaves this question open rather than resolving it.
No model in this chapter takes ι, or any function of it, as an input at any stage; it appears only as a control *output* head.

### 2.2 Circular regression framework

Periodic targets are regressed on the unit circle [@mardia2000circular].
An angle θ with period T is encoded as the two-vector u(θ) = (sin(2πθ/T), cos(2πθ/T)), so the ψ head, with T = π, physically encodes 2ψ.
Each periodic head outputs a raw two-vector v ∈ ℝ², projected onto the unit circle by `tf_normalize_unit`, û = v/sqrt(max(‖v‖²,ε)) — a clamp on the squared norm, for a stable gradient at v=0, rather than a floor on ‖v‖ itself — and trained with the cosine (circular) loss L_circ = 1 − cos Δθ.
For uniformly distributed targets and any fixed prediction, the expected circular loss is exactly 1; the null baselines are ang_MAE = π/2 ≈ 1.5708 rad for φ_c and ι, π/4 ≈ 0.7854 rad for ψ.
Two further diagnostic statistics recur throughout: the circular resultant circ_r = ‖mean(û)‖, which measures mode collapse (circ_r → 1 means all predictions concentrate at one angle), and std_ratio, the ratio of the standard deviation of a head's raw outputs to that of the encoded targets, a monitor of raw-magnitude health with a healthy band of [0.5, 2.0].
Non-periodic heads use a Huber loss on transformed targets, and the sky-position head uses a closed-form von Mises–Fisher negative log-likelihood on the unit sphere.
The inclination head keeps an architecturally distinct path — periodic two-vector encoding, but a Huber loss on the raw output, with no `tf_normalize_unit` projection and no circular loss anywhere in its path.

## 3. Analytic Prerequisite Study

The combination-space hypothesis was tested analytically, using a toy detector-response model in which `h_plus`/`h_cross` add `2φ_c` to the orbital-phase term `2Φ`, not `φ_c` singly (`formulae_reference.md` §A.5).
A required self-check — sweeping ψ at fixed `2φ_c + 2ψ`, confirming (R, δ) stays exactly constant at ι = 0 — passed to machine precision (max std(R) = 1.16×10⁻¹⁴, max std(δ) = 5.87×10⁻¹⁶; `prereq_checks_20260914_160834.log`), gating trust in everything that follows.

The 100-point inclination sweep (50 points per sign of cos ι, 200 sky positions, 5,000-draw bootstrap) confirms the expected combination structure:

- for cos ι > 0, combo **B** (`2φ_c − 2ψ`) is the better-constrained combination throughout; for cos ι < 0, combo **A** (`2φ_c + 2ψ`) wins — a clean sign flip driven by the sign of cos ι;
- the preference ratio is strongest face-on, ≈ **1.62–1.65×**, decaying monotonically toward ≈ **1.04–1.07×** at edge-on, never crossing 1.0;
- population-bootstrapped mean ratios are **1.17× (95% CI [1.11, 1.24])** for cos ι > 0 and **1.19× (95% CI [1.12, 1.26])** for cos ι < 0, with the simple two-point reference check clearing statistical significance directly.

The curriculum weight derivation (`formulae_reference.md` §A.7) gives w(face-on) = 0.0000 and w(edge-on) = 0.1416, with an intermediate-peak shape; training uses the analytically motivated fallback w(ι) = 1 − cos²ι = sin²ι rather than the fitted empirical curve.
Population balance: 28.7% face-on (|cos ι| > 0.9), 32.7% edge-on (|cos ι| < 0.5).

> **Figure 3.1.** Analytic check of the corrected 2φ_c–2ψ degeneracy structure, ratio of the strain correlation with the well-constrained combination to the poorly-constrained one, as a function of inclination. Source: `sweep_1_1_ratio_vs_iota.png`.

The prerequisite study's verdict is GO: the combination structure exists in the data, with the predicted sign behavior.
Whatever the networks below fail to learn, they do not fail for want of an analytically real target.

## 4. Methods

### 4.1 Dataset

All experiments use a single pre-generated dataset, `combined_repackaged.hdf` (referenced by `config_baseline.yaml`): 25,000 training and 5,000 validation samples, two-detector (H1/L1), network SNR uniform in [7,15].
The injection convention behind the combination formula of §1 is traced directly through the dataset's actual generator source code (`ml-gw-search/mlgwsc-1/gen.py:202`, `ml-gw-search/extended_mass/gen.py:179`), confirming PyCBC/LALSimulation's `phiRef` convention and IMRPhenomD, dominant-mode-only, from code rather than recollection; the dataset itself was not regenerated from a version-controlled generator, so a reproducibility gap remains open (§8.3, §8.5).

### 4.2 Trunk architectures

Five one-dimensional trunk architectures (`tcn`, `cnn_baseline`, `cnn_attention`, `inception_time`, `resnet1d`) were evaluated, all mapping the (4096, 2) input to a pooled feature vector shared by every head.
`mode: baseline` and `mode: poc` on the primary `tcn` trunk give the named configurations `poc_a` and `poc_b`.

### 4.3 Head parameterizations

**Baseline mode** uses independent circular-loss heads directly on φ_c and ψ.
**PoC mode** (`SumDiffTrainer`) constructs the two combination heads explicitly rather than treating φ_c and ψ as independent outputs.
The φ_c head's unit vector, **z**_φc, is first passed through a `double_angle` operation, **z**_2φc = complex_mul(**z**_φc, **z**_φc), which turns a unit vector at angle φ_c into one at angle 2φ_c (`formulae_reference.md` §A.1.5); the ψ head's own encoding already stores `2ψ` internally, since `polarization_angle` is declared with period π and `PERIODIC.transform_head` rescales by `2π/period` before taking sin/cos.
The two combination vectors are then **z**_A = complex_mul(**z**_2φc, **z**_ψ), angle `2φ_c + 2ψ`, and **z**_B = complex_mul_conj(**z**_2φc, **z**_ψ), angle `2φ_c − 2ψ` (`formulae_reference.md` §A.2).
The isotropic circular loss, the sign-dependent per-sample curriculum weighting, and the uncertainty-weighting scheme are fixed pipeline components, applied identically to every head.
The well-constrained combination assignment (combo_B for cos ι ≥ 0) is derived from this study's own prerequisite analysis (§3).

### 4.4 Magnitude penalty

Following the diagnosis in §5, all configurations add an explicit radial regularizer on the raw, pre-normalization periodic outputs, `L_mag = λ · Σ mean[(‖v‖ − 1)² ]`, present in every configuration from the first training run.
λ = 0.01 for the Round-1 sweep (§6), retuned to 0.05 and 0.10 in §7.

### 4.5 Training protocol

Adam, initial learning rate 10⁻³, `ReduceLROnPlateau` (factor 0.5, patience 5), batch size 128, 80 epochs, seed 42, uncertainty weighting with log-variance clamp ±3, Huber δ = 1.
Periodic epoch-10 checkpoints (`checkpoint_every_n: 10`) were saved throughout, not only best/final weights, per this repository's standing convention requirement.

### 4.6 Evaluation

Point predictions are scored by wrap-aware angular MAE, circ_r, and, for scalar heads, MAE/R².
Statistical significance is assessed by a label-permutation bootstrap (N = 10,000 shuffles), and population heterogeneity is probed by SNR-tercile stratification.
Two additional checks were added in response to a genuine new finding for one architecture (§6.8) rather than planned in advance: a formal shuffle-null bootstrap and SNR-tercile stratification for the sky-position head's own angular separation (`bootstrap_sky_separation.py`, `snr_stratification_sky_position.py`).

## 5. Inherited Diagnostic History and Present-Day Machinery Check

### 5.1 What was inherited, not rediscovered

Before any measurement of the degeneracy could be trusted, two genuine optimization pathologies in this training pipeline had to be found and fixed: tanh saturation at random initialization on the periodic head output layers (fixed by switching to a linear output activation, since `tf_normalize_unit` already projects onto the unit circle), and, once that was fixed, a second pathology in which the isotropic circular loss provides no restoring force on the raw output's magnitude, so `tf_normalize_unit`'s backward pass — which divides by that magnitude — either crushes or explodes the angular gradient as the magnitude drifts (fixed by the explicit magnitude penalty of §4.4).
Both are established facts about *this codebase's training mechanism*, fixed before the Round-1 sweep in §6 began; the sections that follow do not re-narrate the bug hunt that found them.

### 5.2 Verification on trained checkpoints

Confirming, on the Round-1 (λ = 0.01) checkpoints, that the machinery above is healthy is a prerequisite to trusting anything it produces.

**Labels are clean.** True φ_c/ψ/ι/right-ascension are uniform to circular resultant ≤ 0.013 (`true_label_distributions.png`).

**Gradients reach the periodic-head weights, end to end, through the corrected construction.** A full forward/backward trace on `poc_redo_b`'s real trained checkpoint shows healthy, non-vanishing gradient at every stage of the combination pipeline: dL/d(combo_A_pred) = 0.4245, dropping through the new `double_angle` stage (dL/d(z_2φc_norm) = 0.375) to dL/d(z_φc_raw) = 0.570 and dL/d(z_ψ_raw) = 0.514 — all comparable in order of magnitude to the healthy inclination-head baseline, 0.728 (`diagnostic_output/diagnostic_checks_20260916_113913.log`, lines 297–304).
Prediction perturbation after one optimizer step shows the periodic heads moving far more than the converged scalar control (coa_phase mean|Δ| = 1.61×10⁻², polarization_angle 1.33×10⁻², against mchirp's 3.28×10⁻⁴), confirming the gradient path is live, not disconnected.

**Positive controls recover well** (Fig. 5.4): chirp mass R² = 0.920–0.960 (poc_a 0.958, poc_b 0.958, tcn 0.960, cnn_attention 0.920), merger time R² = 0.906–0.928, SNR R² = 0.741–0.789, sky position 4.4°–10.3° mean angular error.
Inclination is deliberately excluded from this positive-control list for a documented, code-level reason — its Huber-loss, non-normalized path is architecturally distinct from φ_c/ψ's — not omitted quietly; its own, more nuanced result is presented on its own terms in §6.7.

> **Figure 5.1.** True label distributions (Check 1) — `diagnostic_output/true_label_distributions.png`.
> **Figure 5.2.** Combo/circular-loss trajectories for `poc_redo_b`, all 80 epochs — `diagnostic_output/combo_loss_trajectories.png`.
> **Figure 5.3.** Uncertainty-weighting exp(−s) trajectories per head — `diagnostic_output/logvar_trajectories.png`.
> **Figure 5.4.** Positive-control recovery (chirp mass, merger time, SNR) for the four Round-1 models — `analysis_output/scatter_{mchirp,merger_time,snr}_20260916_113847.png`.

The machinery this study depends on is verifiably healthy.
What it produces on the actual question of interest is presented next.

## 6. Results: A Defended Null, Under the Corrected Formula

### 6.1 Headline result

At epoch 79 (last-10-epoch mean), `poc_redo_b`'s combination-space circular loss reads combo_A = 0.9999, combo_B = 0.9895 — both essentially at the ceiling value of 1 expected under random guessing.

**Table 6.1 — Periodic-head recovery, four down-selected models, λ = 0.01, Round 1 (2026-09-15).** Source: `analysis_output/analysis_report_20260916_113847.md`.

| Head | Model | circ_r | ang_MAE (rad) | Null | Health grade |
|---|---|---|---|---|---|
| coa_phase | poc_a | 0.542 | 1.586 | 1.571 | XX |
| coa_phase | poc_b | 0.975 | 1.535 | 1.571 | **COLLAPSE** |
| coa_phase | tcn | 0.840 | 1.590 | 1.571 | XX |
| coa_phase | cnn_attention | 0.386 | 1.616 | 1.571 | XX |
| polarization_angle | poc_a | 0.972 | 0.801 | 0.785 | **COLLAPSE** |
| polarization_angle | poc_b | 1.000 | 0.801 | 0.785 | **COLLAPSE** |
| polarization_angle | tcn | 0.639 | 0.799 | 0.785 | ~ |
| polarization_angle | cnn_attention | 0.396 | 0.785 | 0.785 | ~ |

`poc_a` and `poc_b` are, by the systematic COLLAPSE grade (circ_r > 0.9 **and** ang_MAE > 0.5 — a near-constant output that is also wrong), genuinely mode-collapsed: `poc_b`'s `polarization_angle` predictions place 99.9% of 5,000 validation samples in one 10°-wide histogram bin.
`tcn` and `cnn_attention` fail differently — noisy and uninformative rather than collapsed — but ang_MAE sits at the null for both regardless of failure style.
A direct forward-pass snapshot on `poc_redo_b`'s real checkpoint makes the collapse concrete in three raw numbers rather than one aggregate statistic: three `combo_A_pred` angles from very different true angles (−1.45, −2.22, +1.76 rad) come back at 1.468, 1.496, 1.499 rad — nearly identical regardless of input (`diagnostic_output/diagnostic_checks_20260916_113913.log`).

### 6.2 Confound elimination

This section works through five confounds directly against the trained checkpoints: is the penalty active, does |v|-space invalidate the metrics, is the gradient path dead, is a sub-null MAE real signal, and does learning hide in loud events.
This study's own pre-registered interpretability gate (§7) never clears at any λ tried, so this section reports the *ungated* answers to these questions, run directly against the actual trained checkpoints rather than inferred from a passing gate.

Loss wiring is confirmed correct in both training modes by direct code trace against `trainer.py`'s actual loss-dispatch logic, not merely by inspecting the (misleadingly generic) `head_loss` dictionary listing.
The gradient path is confirmed live end-to-end (§5.2).
The collapse is confirmed genuine, not a metric artifact, by direct histogram inspection (§6.1).
Whether a sub-null MAE reflects real signal is answered formally in §6.3; whether learning concentrates in loud events, in §6.4.

### 6.3 Statistical significance

**Table 6.3 — Label-permutation bootstrap (N = 10,000 shuffles, 5,000 validation samples, one-sided).** z > 0 means observed ang_MAE beats the shuffled-null mean.
Source: `bootstrap_output/bootstrap_ang_mae_20260916_113955.md`.

| Model | φ_c: z, p | ψ: z, p | ι: z, p |
|---|---|---|---|
| poc_a | −0.38, 0.649 | −0.82, 0.793 | +2.00, 0.0223 |
| poc_b | −0.39, 0.649 | −0.63, 0.740 | +2.58, 0.0052 |
| tcn | −1.18, 0.879 | −0.50, 0.692 | +2.58, 0.0059 |
| cnn_attention | −1.97, 0.976 | +1.33, 0.091 | +3.58, 0.0003 |

**φ_c and ψ: 0 of 8 tests significant, most worse than chance.**

**ι: significant in all four models at the uncorrected α = 0.05 level — but this claim needs a correction this chapter's own record has not previously stated.** Applying a 12-test Bonferroni correction (threshold 0.05/12 ≈ 0.00417) to these four inclination tests: only `cnn_attention`'s p = 0.0003 clears it.
`poc_a` (p = 0.0223), `poc_b` (p = 0.0052), and `tcn` (p = 0.0059) do **not** survive the correction.
This chapter's own prior working record (`closing_summary.md`) reported "significant in all 4 models" without stating or applying this threshold; that statement is corrected here.
The honest summary is: inclination reads statistically distinguishable from the shuffled null in `cnn_attention` under multiple-comparisons correction, and distinguishable only at the uncorrected level in the other three models — but §6.7 shows this reflects partial mode collapse, not a recoverable per-sample signal, in every case with an effect size well below this chapter's own materiality floor.

### 6.4 SNR stratification

**Table 6.4 — High-SNR-tercile Δ vs. null (rad).** Source: `snr_output/snr_stratification_20260916_113730.md`.

| Head | poc_a | poc_b | tcn | cnn_attention |
|---|---|---|---|---|
| coa_phase | −0.0293 | −0.0328 | **+0.0209** | −0.0293 |
| polarization_angle | −0.0004 | +0.0039 | +0.0078 | +0.0109 |

`tcn`/coa_phase is the sole monotonic-with-SNR improver, and is dismissed for the same reason: it is the one model whose std_ratio never stabilizes at any λ tried (§7), so its ang_MAE trend is doubly uninterpretable as evidence of learning, not merely unconvincing on its own.
No head/model combination clears this chapter's 0.10 rad materiality floor in the high-SNR tercile.

### 6.5 Loss trajectory and failure signature

Across all 80 epochs, combination-space circular loss is flat within noise for every model.
The failure signature these checkpoints show is std_ratio settling low from early training (§7), not validation loss creeping upward late in training.

### 6.6 The perturbation-trace probe: parked, not closed

This study's calibration stage for the multi-step perturbation trace failed on its own pre-declared terms for **every one of the four models**: the mchirp positive control never reads DIRECTIONAL even under the favorable fresh-init-plus-warmup conditions designed to make it read that way (`perturbation_trace_output/perturbation_trace_early_20260916_114129.md`).
No salvage was attempted: `final` was never run, because its output would be untrusted by this study's own rule regardless.
The probe is stated here as **parked, not closed** — an open instrument-calibration problem, not a resolved confound — and is carried forward as future work (§8.5) rather than presented with any closure language.

### 6.7 Inclination: partial mode collapse, not a small real signal or a clean noise floor

Inclination was used throughout as a same-model noise-floor calibration for judging φ_c/ψ's own band-to-band deviations, on the premise that its apparent failure was clean noise.
That premise does not survive this study's bootstrap evidence (§6.3): ι is not uninformative in the sense of pure random guessing.
But neither is it a small, real, per-sample-recoverable signal, the reading originally given here — most clearly in `cnn_attention` (p = 0.0003, survives Bonferroni correction) and, at the uncorrected level only, in the other three models, with an effect size of roughly 0.02–0.04 rad, well below this chapter's 0.10 rad materiality floor either way.
A predicted-vs-true inclination scatter (`inclination_output/inclination_scatter_20260923_132228.{png,pdf}`) shows no diagonal trend toward either the true value or the confirmed exact ι↔2π−ι waveform-mirror value (verified directly against the real IMRPhenomD generator via `pycbc.waveform.get_td_waveform`, not assumed) in any of the four models.
Instead, predictions cluster into a handful of preferred angles largely independent of the true value — partial mode collapse, quantified by this head's own circular resultant (circ_r = 0.32–0.60 across the four models), well short of the circ_r > 0.9 full-collapse grade used elsewhere for φ_c/ψ, but far above the ≈0 that either clean noise or genuine recovery would give.
Tellingly, the ranking tracks each model's own φ_c/ψ collapse severity: `poc_a`/`poc_b` (COLLAPSE-graded on φ_c/ψ) show the highest inclination circ_r, `tcn`/`cnn_attention` (not collapsed there) the lowest — consistent with a shared training-dynamics origin across a model's periodic heads, not head-specific recoverable physics.
The earlier claim that this scatter showed "a genuine, if noisy, diagonal trend" was not backed by a saved, checkable artifact when originally written; this correction is grounded in the actual plot, generated for the first time during this pass.

This changes how the face-on/mixed/edge-on stratification below must be read.
It is no longer a clean noise-floor test; it is a descriptive comparison against a control that is itself partially collapsed, not cleanly random, and is presented on those terms rather than as a formal calibration.

**Table 6.7 — Edge-on Δ vs. null, φ_c/ψ (rad).** Source: `inclination_output/inclination_stratification_20260916_113601.md`.

| Head | poc_a | poc_b | tcn | cnn_attention |
|---|---|---|---|---|
| coa_phase | +0.0028 | +0.0123 | −0.0457 | +0.0039 |
| polarization_angle | −0.0019 | +0.0043 | −0.0240 | −0.0063 |

No model clears the 0.10 rad floor in either direction; the largest-magnitude deviation (`tcn`/coa_phase, −0.0457) is again the model with the unresolved std_ratio pathology, consistent with instability rather than recovery.
A companion check on a known-good, non-angular control (chirp mass, banded the same way) shows small band-to-band spread (MAE spread 0.051–0.081, R² spread 0.006–0.017 across the four models; `inclination_output/inclination_control_stratification_20260916_113639.md`) — ruling out the banding scheme itself as an alternative explanation for the deviations above, independent of the noise-floor reframing.

### 6.8 A new finding: `cnn_attention`'s partial sky-localization skill

`sky_position` was carried throughout as a positive control; here it is tested for significance directly.

**Table 6.8 — Sky-position angular separation, bootstrap and SNR stratification.** Sources: `bootstrap_output/bootstrap_sky_separation_20260918_151359.md`, `snr_output/snr_stratification_sky_position_20260918_153412.md`.

| Model | Bootstrap Δ vs. null (deg), z | High-SNR Δ vs. null (deg) | Monotonic with SNR? | Verdict |
|---|---|---|---|---|
| poc_a | 3.43, +6.54σ | +3.01 | No | small-but-real, sub-floor |
| poc_b | 3.03, +5.51σ | +2.90 | No | small-but-real, sub-floor |
| tcn | 2.86, +5.21σ | +2.53 | No | small-but-real, sub-floor |
| cnn_attention | **12.58, +23.23σ** | **+15.53** | **Yes** | **REAL-RECOVERY-CONSISTENT** |

Three models show the same small-but-real, sub-floor pattern already established for inclination.
`cnn_attention` clears the materiality floor outright, and — critically, since this exact model is independently known to memorize training-set phase/polarization structure without generalizing (§6.1, §8.2) — the SNR-stratification discriminator resolves in favor of genuine recovery, not a memorization-driven population bias: a hypothetical model that always guesses the population-mean direction scores only ≈89.4° (essentially at the null), ruling out the simplest shortcut, and the actual error shrinks monotonically with SNR (80.15° → 77.95° → 74.47°) with the high-SNR tercile alone clearing the floor by a wider margin than the pooled result.
The leading hypothesis for the mechanism is cross-detector H1/L1 amplitude-ratio information via the sky-dependent antenna pattern — an envelope cue, not absolute carrier phase, consistent with (not contradicting) the general caveat that positive-control success on timing/amplitude quantities does not establish carrier-phase sensitivity.
This finding is **peripheral to the φ_c/ψ question and does not affect the null defended above**; it is reported because it is the first result in either investigation to pass every check this chapter's own discipline requires of a positive finding — significant, floor-clearing, and monotonic with signal strength — and because a future architecture search on these targets should know this specific model's specific skill and specific failure mode.

## 7. The Pre-Registered λ Retune

### 7.1 Why pre-register

This study adopted a strict pre-registration discipline from the outset, after a small cautionary episode of its own: on one same-day instance, a `weight_combo_A/B` trajectory was misread by eye and had to be corrected with a mechanical check before any retune began (`NOTES.md`, 2026-09-15).
The criterion was written down — Step 0 std_ratio gate, Step 1 Bonferroni-corrected bootstrap on the combination angles themselves, Step 2 the same 0.10 rad effect-size floor, Step 3 SNR-monotonicity — before any λ = 0.05/0.10 result existed (`preregistration_lam_retune.md`).

Because the combination construction depends on **both** source vectors (φ_c and ψ) jointly, the gate requires **all four readings** — `poc_redo_b`'s coa_phase and polarization_angle, and `poc_redo_a`'s coa_phase and polarization_angle as the required control — to pass.
A single failing reading keeps the whole round uninterpretable.

### 7.2 Outcome across three λ values

**Table 7.2 — Step 0 gate, all three rounds, all four readings.** Source: `diagnostic_output/diagnostic_logvar_gate_20260916_085902.md`.

| λ | Reading | frac. unhealthy (last 40 ep) | trend/ep | final std_ratio | Gate |
|---|---|---|---|---|---|
| 0.01 | `poc_redo_b` / coa_phase | 0.225 | +0.00638 | 0.583 | FAIL |
| 0.01 | `poc_redo_b` / pol_angle | 0.850 | −0.00402 | 0.251 | FAIL |
| 0.01 | `poc_redo_a` / coa_phase | 0.750 | −0.00721 | 0.407 | FAIL |
| 0.01 | `poc_redo_a` / pol_angle | 1.000 | +0.00076 | 0.374 | FAIL |
| 0.05 | `poc_redo_b` / coa_phase | 1.000 | +0.00093 | 0.315 | FAIL |
| 0.05 | `poc_redo_b` / pol_angle | 0.725 | +0.00679 | 0.531 | FAIL |
| 0.05 | `poc_redo_a` / coa_phase | 0.350 | −0.00561 | 0.522 | FAIL |
| 0.05 | `poc_redo_a` / pol_angle | 0.425 | +0.00383 | 0.572 | FAIL |
| 0.10 | `poc_redo_b` / coa_phase | 0.000 | +0.00011 | 0.930 | PASS* |
| 0.10 | `poc_redo_b` / pol_angle | 0.950 | +0.00326 | 0.421 | FAIL |
| 0.10 | `poc_redo_a` / coa_phase | 0.425 | −0.00453 | 0.546 | FAIL |
| 0.10 | `poc_redo_a` / pol_angle | 0.650 | +0.00220 | 0.559 | FAIL |

\* The one nominal PASS does not survive inspection.
`poc_redo_b`'s coa_phase at λ = 0.10 clears every numeric threshold in the gate, but its own training-time scatter plot (`runs/phic_psi_lam010_retune_b/20260915_181803/scatter/epoch_{0005,0080}.png`) shows two completely flat horizontal bands at the same angle mod 2π — a literal two-point mode collapse, present from epoch 5 through epoch 80 — not real per-sample learning.
This is stated here as a worked example of the discipline this chapter is built on, not an embarrassment to omit: a std_ratio number in the healthy band is not, by itself, proof that a head is producing a real per-sample regression, only a scatter plot is.
std_ratio certifies raw-vector norm health, not angular diversity, and the two are not interchangeable: nothing about clearing the numeric threshold guarantees the scatter plot will look healthy.

With every reading at λ = 0.10 either failing outright or failing on inspection, the round closes: **λ alone is insufficient to stabilize this pipeline's |v|-space for these heads.** Steps 1–3 of the pre-registration never ran, because Step 0 never cleared, at any of the three λ values tried, for any of the twelve readings.

### 7.3 Verdict

Per the pre-registered decision table, applied mechanically: this is filed as **UNINTERPRETABLE**, not as a null result and not as counter-evidence, on its own narrow terms — a negative result about the retune mechanism specifically.
It is not, on its own, the basis for this chapter's headline conclusion; §6's broader, ungated battery is.
Section 8.1 reconciles why these two lines of evidence — one gated and inconclusive, one ungated and convergent — do not collapse into a single clean statement, rather than treating the gate's failure as either dispositive or irrelevant.

## 8. Discussion

### 8.1 Reconciling two lines of evidence that do not collapse into one

This chapter's own two lines of evidence do not collapse into one, and the tension between them is argued here rather than asserted away.
A pre-registered metric-health gate (§7) was meant to license a clean, certified test of the degeneracy: **no model, at any λ, ever passed it** (§7.2) — there is no certified subset of any size to build that argument on.

The null defended in §6 instead rests on a different, and in one respect broader, evidentiary base: four independent methods — direct collapse/histogram inspection, a code-level gradient-health trace, a formal bootstrap, and population stratification — run uniformly across all four down-selected models, without excluding any model on interpretability grounds.
This substitution is defended, not merely adopted, on two grounds.
First, it answers a different and in some ways more direct question than the gate ever did: not "is this specific combination vector's raw magnitude well-behaved," but "do these checkpoints' actual predictions, whatever their internal state, carry recoverable signal" — and the collapse/histogram and gradient-chain evidence in §6.1 and §5.2 rule out the specific failure mode (an implementation bug masquerading as a null) that a gate is partly meant to guard against, by a more direct method than gate-passage ever provided.
Second, this substitution is itself a **novel evidentiary structure that has not been adversarially reviewed** — a distinction this chapter states plainly rather than obscures.
A skeptical reader is entitled to ask whether an ungated battery, built after the pre-registered path proved terminal, is simply a second attempt dressed as independent confirmation.
The honest answer is that it is a different, not a repeated, attempt: it was not constructed to produce a particular answer (the collapse evidence in §6.1, in particular, would have looked identical whether or not it supported the null, since it is a direct read of the prediction distribution, not a test calibrated against a threshold chosen after seeing data), but it was constructed reactively, after the gated path's terminal outcome was known, and that fact should weigh on how much independent confirmatory weight a reader assigns it relative to a fully pre-registered test.
This chapter's own position is that the convergence of four qualitatively different methods, several of which (the gradient-chain trace, the raw histogram inspection) are not calibrated against any threshold at all and simply report what is there, makes the substitute battery trustworthy — but this is an argument for a reader to weigh, not a settled fact.

### 8.2 Sufficiency of the architecture pool

The five trunk families were chosen as five distinct inductive-bias hypotheses, not five samples from one family, and capacity is not the binding constraint.
The memorization-gap check was run uniformly across all four down-selected models, and the result is asymmetric: `cnn_attention`'s training-split circular loss falls to 0.54–0.56 by epoch 80 while validation stays flat at ≈1.0 (train–val gap +0.44 to +0.47), while `poc_a`, `poc_b`, and `tcn` show gaps an order of magnitude smaller (+0.003 to +0.05), consistent with genuinely flat behavior on both splits rather than found-but-ungeneralized structure (`NOTES.md`, "Second batch ported" table).
`cnn_attention` demonstrably has, and uses, spare capacity to fit sample-specific noise and still generalizes none of it to `coa_phase`/`polarization_angle` — the cleanest single demonstration that architecture is not the limiting factor — while it *simultaneously* shows genuine, generalizing skill on `sky_position` (§6.8), a pointed illustration that this model's capacity is neither uniformly wasted nor uniformly informative, but head-specific.

### 8.3 Threats to validity

Recorded in roughly descending order of concern.

- **No model here cleared the pre-registered metric-health gate, at any λ.** §8.1 argues why the broader battery is nonetheless trustworthy; a skeptical reader may reasonably weigh this differently.
- **`poc_b`'s individual coa_phase/polarization_angle bootstrap numbers rest on ground-truth-assisted branch reconstruction, not previously disclosed.** `bootstrap_ang_mae.py`'s `transforms.inverse` call, for poc-mode models, routes through the branch-disambiguation machinery of `formulae_reference.md` §A.8, which by construction needs the true angle to select the correct branch among the naive eight reconstruction candidates. It does not change the verdict — `poc_b` reads non-significant for both heads regardless, and ground-truth-assisted branch selection can only bias a reconstruction toward looking *more* accurate than a blind one, never less — but it is disclosed here explicitly, per this chapter's own citation-discipline standard.
- **ι's noise-floor comparison (§6.7) mixes two non-identically-distributed loss paths.** ι is trained with Huber loss on a raw two-vector; φ_c/ψ with circular loss on a normalized vector. The two heads' finite-sample fluctuation distributions are not strictly comparable even before accounting for the fact that ι itself now carries a small real signal — a second, independent reason (beyond the noise-floor reframing already applied) to read §6.7 as descriptive rather than as a formal statistical calibration.
- **The perturbation-trace probe (§6.6) is open, not closed**, and no salvage attempt was made.
- **Waveform-provenance and inclination-prior gaps carried forward unresolved** (§4.1, §8.5): the dataset was not regenerated from a version-controlled generator, and the inclination prior remains uniform in ι rather than the astrophysically correct uniform in cos ι — over-representing face-on systems and under-representing edge-on systems, which mildly favors a null result rather than protecting against one (face-on is where the degeneracy is strongest), though the null itself does not depend on this skew since §6.7's inclination-band stratification checks the edge-on band directly and still finds no recovery clearing the materiality floor there; it should have been isotropic and was not.
- **Design constraints**: single seed (42) per configuration, fixed 80-epoch budget, single dataset with a single noise realization per event.

### 8.4 Methodological lessons

Three lessons run through this chapter — aggregate metrics are not evidence until their mechanism is verified, controls must be validated at the code level, and pre-registration belongs inside engineering loops — confirmed sharply enough to add a fourth.

**A passing interpretability gate is not proof that its subject matter is what the gate measures.** The λ = 0.10 std_ratio false-positive (§7.2) demonstrated this directly: a checkpoint that cleared every numeric gate threshold was, on scatter-plot inspection, a literal two-point mode collapse.
std_ratio measures raw-vector norm health, not angular diversity; the corrective, as with every lesson in this arc, is mechanism inspection — a scatter plot, not a summary statistic.

### 8.5 Future work

Eight items remain open, scoped below.

(i) **Inclination conditioning** — the natural next step given §3's analytic structure, scoped as a training-paradigm choice rather than a claim that ι is unlearnable; if anything strengthened as a target given ι's newly-confirmed small real signal (§6.7).

(ii) **An architecture-level attack on the std_ratio instability**, or a finer, freshly pre-registered λ mini-sweep — this chapter's own λ-sweep (§7) is exhausted on its own terms; it is not this chapter's job to relitigate a closed, pre-registered sweep, only to note the same lever remains available.

(iii) **A posterior-estimation reformulation.**

(iv) **A ridge-structure check**: a 2D joint scatter of the down-selected models' predicted (φ̂_c, ψ̂) pairs, to test directly whether any model learned the degenerate combination manifold itself.
No such script or artifact exists anywhere in this repository.
It is affirmed here as open, explicitly, rather than left to lapse silently.

(v) **A synthetic-data ablation disentangling `poc_b`'s curriculum design from the physical degeneracy**, never run. `poc_b`'s collapse (§6.1) is consistent with, but does not on its own prove, unlearnability rather than a curriculum-induced rank-deficient-gradient artifact, and only a synthetic target with a known-recoverable combination would settle it.

(vi) **A sensitivity-floor check** — a high-SNR (25–30) validation set or a Fisher-matrix bound, to separate "degenerate" from "below this population's detection-scale sensitivity floor."
Never run; this is the single most consequential of the open items, since it bears directly on how strongly "effectively exact" can be claimed.

(vii) **Dataset regeneration from a version-controlled generator script.** Partially addressed (the injection convention is now traced to actual generator source, §4.1) but the dataset file itself was not regenerated; the partial progress is stated rather than claimed as a full resolution.

(viii) **Confirming `cnn_attention`'s sky-position mechanism** (§6.8). The cross-detector amplitude-ratio/antenna-pattern hypothesis was never directly tested, only found consistent with the SNR-dependence observed; a targeted follow-up (e.g. an ablation removing one detector, or a synthetic single-antenna-pattern control) could confirm or refute it.

## 9. Conclusion

We set out to determine whether the coalescence phase and polarization angle of compact-binary signals are recoverable from strain by direct neural regression, individually or in their physically motivated sum/difference combinations.
The answer is **no**.
Every configuration tested converged to the optimal constant or noise-like predictor at the analytic random-guessing error on both angles, while the same shared representation recovered chirp mass, merger time, signal-to-noise ratio, and — for one architecture — a genuine partial sky-localization skill, from the same features.
No model here passed the pre-registered metric-health gate; the null instead rests on convergence across direct collapse inspection, a code-level gradient-health trace, a formal permutation bootstrap, and population stratification, applied uniformly across every down-selected model rather than to a certified subset — a defensible basis for the conclusion, argued explicitly in §8.1, but one that has not yet been through independent adversarial scrutiny.
Two findings exist only because this study looked further than strictly necessary to answer its own central question: `cnn_attention`'s real, SNR-verified sky-localization skill, and inclination's small-but-real departure from the "clean noise floor" role this study's own methodology initially assumed for it.
Neither changes the central verdict; both sharpen, honestly, what it does and does not claim.
The degeneracy remains, on the evidence presented here, effectively exact for this population as a point-estimation problem.

---

## Appendix: Claim-to-artifact map

All paths relative to `experiments/phic_psi_poc-redo/`.

| Claim / table / figure | Artifact |
|---|---|
| Analytic prerequisite sweep (§3, Fig. 3.1) | `sweep_1_1_ratio_vs_iota.{csv,png}`, `prereq_checks_output/prereq_checks_20260914_160834.log`, `formulae_reference.md` §A.5/A.7 |
| Combination construction (§4.3) | `formulae_reference.md` §A.1.5/A.2; `trainer.py` |
| Label-distribution audit (Fig. 5.1) | `diagnostic_output/true_label_distributions.png` |
| Combo-loss trajectories (Fig. 5.2) | `diagnostic_output/combo_loss_trajectories.png` |
| Uncertainty-weighting trajectories (Fig. 5.3) | `diagnostic_output/logvar_trajectories.png` |
| Gradient-chain / prediction-perturbation numbers (§5.2) | `diagnostic_output/diagnostic_checks_20260916_113913.log` |
| Positive controls (Fig. 5.4, §5.2) | `analysis_output/analysis_report_20260916_113847.md`, `scatter_{mchirp,merger_time,snr}_20260916_113847.{png,pdf}` |
| Table 6.1 | `analysis_output/analysis_report_20260916_113847.md` |
| Table 6.3, Bonferroni correction (§6.3) | `bootstrap_output/bootstrap_ang_mae_20260916_113955.md` |
| Table 6.4 | `snr_output/snr_stratification_20260916_113730.md` |
| §6.6 (perturbation trace, parked) | `perturbation_trace_output/perturbation_trace_early_20260916_114129.md`, `closing_summary.md` |
| Table 6.7, inclination reframing (§6.7) | `inclination_output/inclination_stratification_20260916_113601.md`, `inclination_control_stratification_20260916_113639.md`; partial-collapse finding: `inclination_output/inclination_stratification_20260923_132228.md`, scatter `inclination_output/inclination_scatter_20260923_132228.{png,pdf}` |
| Table 6.8, sky-position finding (§6.8) | `bootstrap_output/bootstrap_sky_separation_20260918_151359.md`, `snr_output/snr_stratification_sky_position_20260918_153412.md` |
| Table 7.2, std_ratio false-positive (§7.2) | `diagnostic_output/diagnostic_logvar_gate_20260916_085902.md`; scatter PNGs at `runs/phic_psi_lam010_retune_b/20260915_181803/scatter/epoch_{0005,0080}.png` |
| Memorization-gap table (§8.2) | `diagnostic_output/redo_models_train_val_loss.{png,pdf}`; `NOTES.md` "Second batch ported" table |
| Undisclosed poc_b reconstruction property (§8.3) | `bootstrap_ang_mae.py` lines ~200/216; `formulae_reference.md` §A.8; `closing_summary.md` |

One number in this chapter is *derived* rather than printed in its source artifact, flagged here under this chapter's own traceability rule: the Bonferroni threshold 0.00417 in §6.3/§6.7 is 0.05/12, arithmetic on the twelve bootstrap tests, not a value stated in `bootstrap_ang_mae_20260916_113955.md` itself.

## Suggested citation keys (to resolve against thesis bibliography)

`abbott2016gw150914`, `cutler1994gw`, `sathyaprakash2009physics`, `veitch2015lalinference`, `george2018deep`, `gabbard2022vitamin`, `dax2021dingo`, `kendall2018multi`, `mardia2000circular`, `fisher1953dispersion`, `bai2018tcn`, `he2016resnet`, `fawaz2019inceptiontime`, `vaswani2017attention`, `nosek2018preregistration`.
See `references.bib` in this directory for the resolved entries.
