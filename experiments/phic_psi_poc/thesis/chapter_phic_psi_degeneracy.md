# Chapter — On the Recoverability of Coalescence Phase and Polarization Angle from Gravitational-Wave Strain: A Systematically Verified Null Result

> **Draft status.** Chapter number, cross-references to other chapters, and citation keys (`[@…]`) are placeholders to be resolved against the thesis bibliography. All quantitative values are taken verbatim from the experiment record on branch `poc/phic-psi-degeneracy` (primary sources: `diagnostic_log.md`, `NOTES.md`, and the auto-generated reports under `experiments/phic_psi_poc/`); a claim-to-artifact map is given in the chapter appendix.

> **Editorial note — remove before final submission.** This chapter has been through three rounds of adversarial AI review (`thesis/reviews/`: unsuffixed files are round 1, `*_v2.md` files are round 2, `kimi_ai_adversarial_review_v3.md` is round 3 — Kimi is the only reviewer to reach a third round). All three rounds' fixes and new analyses are folded into the text below (§ 2.1, § 6.2, § 6.6, § 6.7, § 7.1, § 8.2 Fig. 8.1, Table 6.7, and framing changes throughout § 1, § 6.3, § 8.3, § 9). Three items remain **critical future work that this text pass cannot resolve**, so a future review pass — human or AI — does not mistake the omission for an oversight: (1) a synthetic-data ablation to disentangle poc_b's curriculum design from the physical degeneracy; (2) a high-SNR (25–30) validation set or a Fisher-matrix bound, to rule out a sensitivity-floor confound distinct from degeneracy; (3) regenerating the dataset (or a small slice of it) from a version-controlled generator script, to close the waveform-provenance gap for good. A fourth item raised in round 2 — either a fresh-holdout pre-registered replication of the A.3 perturbation trace, or a formal downgrade of A.3 from "closed" to "open" — is now reflected directly in the chapter body: § 6.6 states A.3's closure as provisional, pending that replication, rather than asserting closure outright. Each of the three remaining items is a genuine, well-posed research task, not a text fix — and each was judged, at this point in the project, to cost more in new training/data-generation effort than the marginal certainty it would buy, weighed against finishing the thesis. They are recorded as scoped future work in `experiment_index.md` and `NOTES.md` (2026-07-23, "Round 2" and "Round 3" adversarial review pass entries) rather than silently dropped.

## 1. Introduction

Deep-learning approaches to gravitational-wave parameter estimation have matured rapidly, from early proof-of-concept classifiers [@george2018deep] to amortized neural posterior estimators that approach the accuracy of stochastic-sampling pipelines [@gabbard2022vitamin; @dax2021dingo] at a fraction of their latency.
Most of this progress has concentrated on parameters that imprint strongly and non-degenerately on the observed strain: the chirp mass through the phase evolution [@cutler1994gw], the merger time through the temporal localization of the signal, the signal-to-noise ratio through overall amplitude, and the sky position through inter-detector time delays and antenna-pattern amplitudes.

Two extrinsic parameters have received far less attention as direct regression targets: the orbital phase at coalescence, φ_c, and the polarization angle, ψ.
There is a physical reason for this neglect.
For the dominant quadrupole (ℓ = 2, |m| = 2) mode of a quasi-circular compact binary, the two polarization states are

h₊ ∝ (1 + cos²ι) cos(2φ(t) + 2φ_c), h_× ∝ 2 cos ι sin(2φ(t) + 2φ_c),

and the detector response mixes them through antenna patterns that depend on 2ψ [@cutler1994gw; @sathyaprakash2009physics].
In the face-on limit (cos ι → ±1) the signal becomes circularly polarized and depends on φ_c and ψ only through a single combination — effectively φ_c ± 2ψ, with the sign set by the sign of cos ι — so the individual angles are exactly unidentifiable there, and only partially disentangled as the inclination moves toward edge-on.
Bayesian samplers absorb this structure into broad, multimodal joint posteriors [@veitch2015lalinference]; a point-estimate regression network has no such refuge.

This chapter asks the question directly:

> **Can φ_c and ψ be recovered from strain alone by a deep neural network — individually, or through their better-conditioned sum/difference combinations — or is the degeneracy effectively exact for a realistic detection-scale population?**

The answer we converge on is a **null result**: across five trunk architectures, two head parameterizations, four values of a magnitude-regularization coefficient, and roughly nine full training campaigns, no model ever performed measurably better than random guessing on either angle, while the same networks simultaneously recovered chirp mass, merger time, signal-to-noise ratio, and sky position with high fidelity from the same shared features.
All seven configurations land at the null; the distinction between "certified" and "corroborating" is about what else can be ruled out, not about whose ang_MAE counts.
For two λ-matched configurations, poc_b and cnn_attention, the pre-declared magnitude-health interpretability gate (§ 8.2) confirms no unresolved metric-health caveat could be inflating or masking that number — these are the fully certified base.
The remaining five show the identical outcome but carry an open metric-interpretability question of their own (§ 8.2, § 8.3), so they corroborate the same result under progressively relaxed certification rather than serving as independent repeats of the certified test.

A null result of this kind is only as strong as the effort spent trying to break it.
The scientific contribution of this chapter is therefore twofold.
First, the null itself, defended against an explicit list of confounds: data-pipeline faults, loss-wiring faults, two genuine and initially invisible optimization pathologies (a saturating output activation and a gradient-attenuating normalization layer), architecture-specific artifacts, statistical flukes, and population heterogeneity.
Second, a methodological account of *how* the null was established, because the investigation was repeatedly misled by aggregate metrics before converging on a mechanism-first diagnostic discipline and, ultimately, a pre-registered decision procedure with mechanically evaluated success criteria [@nosek2018preregistration].
Ruling out artifacts is not preamble to the result here; it *is* the result.

The chapter is organized conventionally, with one deliberate exception.
Section 2 formalizes the problem and the circular-regression framework.
Section 3 presents an analytic prerequisite study that motivated the sum/difference reparameterization.
Section 4 details the datasets, architectures, losses, and training protocol.
Section 5 — the exception — narrates the diagnostic arc: the two optimization pathologies that had to be found and fixed before any measurement of the degeneracy could be trusted.
Section 6 presents the defended null result and its verification battery.
Section 7 describes the pre-registered magnitude-penalty retune and its outcome.
Section 8 discusses interpretation, limitations, and future work; Section 9 concludes.

## 2. Problem Formulation

### 2.1 Targets and the degeneracy hypothesis

We consider a ten-parameter description of a compact binary coalescence signal injected into two-detector noise: chirp mass ℳ, mass ratio q, inclination ι, coalescence phase φ_c, polarization angle ψ, declination, right ascension, injection time, merger time within the analysis window, and network signal-to-noise ratio.
Of these, seven are used as supervised regression targets (mass ratio and injection time are excluded; § 4.1).
The scientific targets are φ_c ∈ [0, 2π), with period 2π, and ψ ∈ [0, π), with period π.
Inclination ι, chirp mass, merger time, SNR, and sky position are retained as *control heads*: parameters trained through the identical pipeline whose success or failure calibrates what the network can extract from strain when information is present.

The **degeneracy hypothesis** under test is: *φ_c and ψ carry no strain-only recoverable signal for this population, in this architecture and loss family* — i.e., a point-estimating network can do no better than the optimal constant (or constant-per-mode) prediction, because for most of the population the likelihood constrains only a combination of the two angles, and the remaining direction is unconstrained.

The alternative hypothesis motivating the experimental design is that even if φ_c and ψ are individually unrecoverable, their **sum and difference combinations** may be well-conditioned: face-on systems constrain φ_c ∓ 2ψ (sign by cos ι), so a network parameterized in combination space, with a curriculum that weights each combination according to how well the current inclination constrains it, might learn the recoverable structure that a naive per-angle parameterization misses.

One scope note applies to the whole chapter and is stated here rather than left implicit.
Conditioning on inclination ι — tested as future work in § 8.5 — is treated throughout as a *training-paradigm design choice*, motivated by the analytic structure of § 3 showing that the well-constrained combination is recoverable given ι, not as a forced move necessitated by a proof that ι itself is unrecoverable.
The inclination head's own failure (§ 5.4) is unresolved, not diagnosed as fundamental: it could reflect a genuine unlearnability of ι from strain under this architecture, or a fixable loss-path choice — § 5.4 traces it to a Huber loss on a raw two-vector, unlike the successful von Mises–Fisher parameterization used for sky position.
Whether ι can be predicted by an independent pipeline is therefore a separate, open question that this chapter does not resolve and does not need to resolve in order to motivate testing ι-conditioning as a training paradigm; § 8.5 excludes inference-time ι estimation from scope for exactly this reason, not because ι is assumed to be fundamentally unrecoverable.

### 2.2 Circular regression framework

Periodic targets are regressed on the unit circle [@mardia2000circular].
An angle θ with period T is encoded as the two-vector

**u**(θ) = (sin(2πθ/T), cos(2πθ/T)),

so the ψ head, with T = π, physically encodes 2ψ.
Each periodic head outputs a raw two-vector **v** ∈ ℝ², projected onto the unit circle by a normalization layer, `normalize_unit`:

**û** = **v** / max(‖**v**‖, ε), ε = 10⁻⁸,

and trained with the cosine (circular) loss

L_circ = 1 − **û**_pred · **u**_true = 1 − cos Δθ.

For uniformly distributed targets and any fixed prediction, the expected circular loss is exactly 1; the expected wrap-aware angular mean absolute error (ang_MAE) is T/4 — that is, **π/2 ≈ 1.5708 rad for φ_c and ι, and π/4 ≈ 0.7854 rad for ψ**.
These are the null baselines against which every result in this chapter is judged.

Two further diagnostic statistics recur throughout.
The **circular resultant** of a set of predicted angles, circ_r = ‖mean(**û**)‖, measures mode collapse: circ_r → 1 means all predictions concentrate at one angle (the empirical baseline for our uniform targets is circ_r ≈ 0.005).
The **std_ratio** of a head is the ratio of the standard deviation of its raw outputs to that of the encoded targets, accumulated over an epoch; because the targets lie on the unit circle, std_ratio is a direct monitor of raw-magnitude health, with values far from unity signalling the pathology analyzed in § 5.4.
We adopt [0.5, 2.0] as the healthy band.

Non-periodic heads use a Huber loss (δ = 1) on transformed targets (log–z-scored chirp mass, z-scored SNR, sigmoid-bounded merger time), and the sky-position head uses a closed-form von Mises–Fisher negative log-likelihood on the unit sphere [@fisher1953dispersion].
All heads share a single trunk and are combined by learned homoscedastic uncertainty weighting [@kendall2018multi]: each head h carries a trainable log-variance s_h, clamped to |s_h| ≤ 3, contributing exp(−s_h)·L_h + s_h to the total loss.

## 3. Analytic Prerequisite Study

Before any training, the combination-space hypothesis was tested analytically on the dataset itself, using a toy detector-response model: for a sweep of inclinations, we computed the correlation between strain-derived quantities and each candidate combination, A = φ_c + 2ψ and B = φ_c − 2ψ, and formed the ratio of the better-correlated ("well-constrained") to the worse-correlated combination.

The definitive 100-point inclination sweep (50 points per sign of cos ι, 200 sky positions, 5,000-draw bootstrap) gave an unambiguous, sign-symmetric structure (Fig. 3.1):

- for cos ι > 0, combo **B** is the better-constrained combination at **50 of 50** inclination points; for cos ι < 0, combo **A** wins 50 of 50 — a clean sign flip, as the circular-polarization picture predicts;
- the preference ratio is strongest face-on, ≈ **1.56×**, decaying monotonically toward ≈ **1.05–1.08×** at edge-on, never crossing 1.0 (minimum 1.049 at ι = 1.62);
- population-bootstrapped mean ratios are **1.155× (95% CI [1.118, 1.195])** for cos ι > 0 and **1.171× (95% CI [1.130, 1.216])** for cos ι < 0 — modest but statistically unambiguous.

> **Figure 3.1.** Analytic check of the φ_c–ψ degeneracy structure: ratio of the strain correlation with the well-constrained combination to the poorly-constrained one, as a function of inclination, for cos ι > 0 (left; combo B = φ_c − 2ψ favored) and cos ι < 0 (right; combo A = φ_c + 2ψ favored). Face-on systems prefer the expected combination by ≈ 1.56×; the preference shrinks to ≈ 1.05–1.08× at edge-on but never inverts. Bootstrap means: 1.155× (95% CI [1.118, 1.195]) and 1.171× (CI [1.130, 1.216]).

The same study produced the curriculum weight: a Jacobian-conditioning derivation (finite-difference Jacobian of the angle-to-response map, condition number averaged over sky positions, interpolated on cos²ι) yielded an empirical weight with w(face-on) = 0.0000 and w(edge-on) = 0.1212 after an earlier polynomial fit was rejected for producing negative weights (maximum residual 1.32, w(face-on) = −5.29 from the fit against 0.000 from the raw data).
The trained runs use the analytically motivated fallback

w(ι) = 1 − cos²ι = sin²ι,

which shares the fitted weight's essential feature (zero weight where the degeneracy is exact, maximal weight edge-on) while avoiding the fit's pathologies; the fitted variant was derived and validated but never wired into training, a caveat noted in § 8.3.
Population balance was also confirmed: 28.7% of samples are near-face-on (|cos ι| > 0.9) and 32.7% near-edge-on (|cos ι| < 0.5), so neither regime is vanishingly rare.

The prerequisite study's gate verdict was GO: the combination structure exists in the data, with the predicted sign behavior.
Whatever the networks subsequently failed to learn, they did not fail for want of an analytically real target.

## 4. Methods

### 4.1 Dataset

All experiments use a single pre-generated HDF5 dataset of simulated two-detector observations: **25,000 training and 5,000 validation samples**, each a pair of 4,096-sample noisy strain series for the H1 and L1 detectors — a **2 s window at 2,048 Hz** — stacked as a (4096, 2) input tensor.
The merger is placed at 1.6–1.8 s (80–90% of the window).
All injections use the dominant quadrupole (ℓ = 2, |m| = 2) mode only (IMRPhenomD), with no higher-order modes — the condition under which the face-on degeneracy argument of § 1 is exact rather than approximate.
This detail is not preserved as HDF5 metadata or in any generation script retained in this repository (the generator itself lives outside the tracked pipeline); it is recorded here from the authors' knowledge of how the dataset was produced, an honest provenance gap flagged in § 8.3.
Marginal parameter distributions (verified empirically; § 5.2, Check 1): chirp mass 8.84–43.45 M_⊙, mass ratio 0.20–1.00, network SNR uniform in **[7, 15]**, and — critically — φ_c, ψ, ι, and right ascension uniform over their supports, with declination following the cos-weighted sky prior.
**Note on the inclination prior:** ι is uniform in ι itself (0 to π), not in cos ι — the astrophysically correct isotropic choice — which over-represents edge-on systems relative to a real population; because edge-on is where the § 3 degeneracy is weakest, this does not bias the null toward a self-serving conclusion, but it is a real limit on astrophysical realism and is revisited as a design constraint in § 8.3.
Strain is fed to the model already whitened and otherwise raw: PSD-whitening is applied at dataset-generation time, and the only further transform before the trunk is the batch-normalization layer that starts every architecture.
Targets are transformed per § 2.2; transform statistics are fitted on the training split only.
No data augmentation or SNR-dependent sample weighting is used.

### 4.2 Trunk architectures

Five one-dimensional trunk architectures were evaluated, all mapping the (4096, 2) input to a pooled feature vector shared by every head:

| Trunk | Body | Readout | Feature dim |
|---|---|---|---|
| `tcn` (primary) | Conv stem (64 ch, k=7, stride 2) + 10 residual TCN blocks, dilations 1–512, 64 filters, receptive field ≈ full window [@bai2018tcn] | GAP ⊕ GMP | 128 |
| `cnn_baseline` | 4 × [Conv1D(k=16) → BN → ReLU → MaxPool(4)], filters 32–128 | GAP | 128 |
| `cnn_attention` | 5 strided conv stages → 2 transformer encoder blocks (d=128, 4 heads) [@vaswani2017attention] | learned attention pooling | 128 |
| `inception_time` | Conv stem (stride 4) + 6 Inception modules, kernels {9, 19, 39} [@fawaz2019inceptiontime] | GAP | 128 |
| `resnet1d` | ResNet-style, 3 stages, filters 64/128/256, dilated second blocks [@he2016resnet] | GAP ⊕ GMP | 512 |

Each head is a two-layer MLP (64 hidden units, ReLU) on the shared features, with a linear 2-unit output for periodic heads (§ 5.3 explains why *linear*, not bounded, matters).
The five trunks were not chosen as a random sample of model space but as five distinct *inductive-bias families* — full-receptive-field dilated temporal convolution, plain hierarchical convolution, attention-based sequence modeling, explicit multi-scale kernels, and deep residual feature learning — so that each represents a different hypothesis about how phase information might be encoded in the strain; the sufficiency of this pool for a null claim is argued in § 8.2.

### 4.3 Head parameterizations under test

Two parameterizations of the φ_c/ψ problem were crossed with the trunks:

- **baseline mode** — independent circular-loss heads directly on φ_c and ψ, each with its own uncertainty weight.
- **poc mode** (`SumDiffTrainer`) — the φ_c and ψ heads are retained as raw outputs, unit-normalized, and combined by complex multiplication into the two combination vectors, **z**_A = **z**_φc ⊗ **z**_2ψ (angle φ_c + 2ψ) and **z**_B = **z**_φc ⊗ conj(**z**_2ψ) (angle φ_c − 2ψ). The circular loss is applied to the *combinations*, per-sample weighted by the sign-dependent curriculum: with cos ι ≥ 0, combo B receives weight 1 and combo A receives w(ι) = sin²ι; the roles swap for cos ι < 0. The individual φ_c/ψ uncertainty weights are removed and replaced by per-combo weights.

Named configurations referenced throughout: **poc_a** = baseline mode on the TCN trunk; **poc_b** = poc mode on the TCN trunk; `tcn`, `cnn_baseline`, `cnn_attention`, `inception_time`, `resnet1d` = baseline mode on the named trunk.
(poc_a and `tcn` are identical in trunk and mode and differ only in run identity — a deliberate seed-free replication pair.)

### 4.4 Magnitude penalty

Following the diagnosis in § 5.4, all main-line configurations add an explicit radial regularizer on the *raw, pre-normalization* periodic outputs:

L_mag = λ · Σ_{v ∈ {z_φc, z_ψ}} mean[(‖**v**‖ − 1)²],

the standard companion of cosine-similarity losses in metric learning. λ values of 0 (ablation), 0.01 (main runs), 0.05, and 0.10 (pre-registered retune) were tested.

### 4.5 Training protocol

All runs share: Adam, initial learning rate 10⁻³, `ReduceLROnPlateau` (factor 0.5, patience 5), batch size 128, **80 epochs**, seed 42, no early stopping, uncertainty weighting with log-variance clamp ±3, Huber δ = 1.
Configuration differences are confined to the trunk, the loss mode, and λ, with three caveats recorded for honesty: the `cnn_baseline`, `inception_time`, and `resnet1d` configs omit the magnitude penalty (λ = 0) so the five-architecture comparison is not λ-matched; the two CNN configs use a smaller learning-rate floor (10⁻⁷ vs 10⁻⁶); and `resnet1d` alone uses a 5-epoch warmup.
None of these differences touches the four models (poc_a, poc_b, tcn, cnn_attention) on which the verified null (§ 6) rests, which are λ-matched at 0.01.

### 4.6 Evaluation

Point predictions on the 5,000-sample validation set are scored by wrap-aware angular MAE (§ 2.2), circ_r, and, for scalar heads, MAE/R².
Statistical significance of any apparent sub-null ang_MAE is assessed by a label-permutation bootstrap (§ 6.3): the observed ang_MAE is compared against the distribution obtained from **10,000 random permutations** of the true labels against fixed predictions, which preserves both marginals while destroying any input–label association.
Population heterogeneity is probed by SNR-tercile stratification (§ 6.4).
A seven-check diagnostic suite (§ 5.2) instruments the data pipeline, loss wiring, uncertainty-weight trajectories, gradient routing, output saturation, the full gradient chain through the circular-loss pipeline, and early-training dynamics.

## 5. The Diagnostic Arc: Making the Measurement Trustworthy

The headline result of this chapter is that nothing was learned about φ_c and ψ.
The first result obtained, however, was that *something appeared to be learned* — and the road from the first to the second is where most of the experimental effort was spent.
We narrate it explicitly, because two genuine optimization pathologies had to be discovered, mechanistically understood, and fixed before flat loss curves could be read as evidence about physics rather than as engineering failure; and because the record of misreadings corrected along the way is the strongest argument that the final reading is not itself one of them.

### 5.1 Round 1: apparent success, actual collapse

The first full campaign (seven configurations across the five trunks) produced a beguiling number: validation R² = **0.754** for ψ in the poc-mode model.
It was an artifact.
Every φ_c prediction across the TCN variants had collapsed to a single constant (315°; ψ collapsed to 67.5°), with circular resultant circ_r = 1.000.
For a *periodic* target with uniform truth and constant prediction, the expected "R²" computed on wrapped residuals is not 0 but exactly

R²_null = 1 − (π²/48)/(π²/12) = 0.75,

so 0.754 was the signature of total mode collapse, not of learning.
Meanwhile the same models' chirp-mass heads reached R² > 0.95 — the trunk was healthy; the periodic heads specifically were dead.
This was the first of several instances in which an aggregate metric, read naively, pointed in exactly the wrong direction, and it set the tone for everything that followed: *no metric was thereafter trusted until the mechanism producing it was inspected.*

### 5.2 The diagnostic suite

Seven mechanism-level checks were built incrementally during the bug hunt and re-run against every subsequent campaign: (1) true-label distribution audit; (2) loss-wiring verification; (3) uncertainty-weight and loss trajectories; (4) gradient routing — do the periodic head weights actually change after a training step; (5) output-saturation dump; (6) end-to-end gradient-chain trace through normalization, complex multiplication, curriculum weighting, and uncertainty scaling; (7) initialization-time saturation timing.

Check 1 immediately ruled out the data pipeline: true φ_c/ψ/ι are uniform to circ_r ≤ 0.013 with no duplicated or quantized values (Fig. 5.1).
Check 2 caught a real but benign wiring fault (stale Huber-loss registrations for φ_c/ψ alongside the poc-mode combo losses — never invoked, but removed).
Check 4 then produced the first smoking gun: after one optimizer step, every head's predictions moved *except* φ_c and ψ (mean absolute prediction change 0.15–0.81 for scalar heads, **0.00** for both angles), while — the crucial control — the inclination head, which shares the trunk, the two-vector encoding, and the head architecture, moved healthily.
Whatever was killing φ_c/ψ was in their final layer, not in the trunk or the data.

> **Figure 5.1.** True label distributions after HDF5 load, before target transforms, for training (top, n = 25,000) and validation (bottom, n = 5,000). φ_c, ψ, ι, and right ascension are uniform over their supports (circular resultant ≤ 0.013); declination follows the non-uniform sky prior. The prediction collapse observed in training is not a data-pipeline artifact.

### 5.3 Pathology I: output saturation at initialization

Check 6's forward-pass dump found the mechanism.
Every sample's raw φ_c output was exactly (−1.000000, +1.000000) — norm √2 = 1.4142 — and had been from **step 0**: the periodic heads used a tanh output activation, and at random initialization the pre-activation logits were already large enough to saturate it.
At saturation tanh′ ≈ 0, so although the gradient flowing *into* the output was healthy (‖∂L/∂z_raw‖ ≈ 0.31), the gradient reaching the weights was numerically zero (kernel gradient 0.000000; bias 3.7 × 10⁻¹⁰).
Adam's momentum still moved the weights (kernel change of 2.46 after a step) without ever changing the predictions — the model was, in the log's phrase, *born dead*.
The constant √2 also retro-explained the frozen std_ratio = 1.4142 seen in every earlier trajectory.

The fix is disproportionately small for the damage caused: the tanh was **redundant** — `normalize_unit` already projects onto the unit circle — so the periodic output activations were changed to linear, and all seven configurations retrained.
Two lessons carried forward.
First, the saturation was invisible to every aggregate metric and to a logit-magnitude heuristic (an earlier check had "ruled out" saturation from an estimated logit scale; the direct dump proved the estimate wrong) — only inspection of actual forward-pass values settled it.
Second, a redundant nonlinearity that is merely useless in expectation can be fatal in the tail of initialization space.

### 5.4 Pathology II: normalization-induced magnitude drift

The retrained models still did not learn the angles — but now for a different, equally mechanical reason.
The backward pass of **û** = **v**/‖**v**‖ scales angular gradients by 1/‖**v**‖, and the circular loss, computed entirely on **û**, is *blind* to ‖**v**‖: nothing in the objective anchors the raw magnitude.
Freed from tanh's box, ‖**v**‖ drifted unboundedly — by epoch 79 the φ_c head's std_ratio reached **103.7** (gradients attenuated ~100×) and ψ's ≈ 13.
The pathology had been introduced *by an earlier correct fix*: replacing an anisotropic Huber-on-vector loss (whose minimum at ‖v‖ = 1 had implicitly regularized magnitude) with the isotropic circular loss silently discarded that side effect.
The remedy (§ 4.4) is the standard explicit penalty λ(‖**v**‖ − 1)², adopted at λ = 0.01.

One misreading from this period required formal retraction and is retained in the record.
The inclination head appeared to fail "identically" to φ_c/ψ and was briefly cited as key evidence that the collapse was a shared `normalize_unit` bug rather than physics.
A code-level trace disproved this: the inclination head is trained with a *Huber* loss on its raw two-vector — no `normalize_unit` in its path at all — so its failure has a different (still unresolved) mechanism, plausibly convergence to the dataset-mean vector as Huber's optimal constant predictor, and is evidence *neither for nor against* the φ_c/ψ mechanism.
The control that seemed to prove the point was not a valid control; only code inspection, not metric comparison, revealed this.

### 5.5 Run 7: healthy machinery, flat objective

With linear outputs and the λ = 0.01 penalty, the four λ-matched models (poc_a, poc_b, tcn, cnn_attention) were retrained and re-instrumented.
The machinery now verifiably works: raw magnitudes contract from initialization norms of ~100–250 to ‖**v**‖ ≈ 1 by epoch ~40; gradient norms at the periodic head kernels are 0.58–1.37 (previously 0.0); predictions move after an optimizer step (mean |Δ| ≈ 1.4–1.5 × 10⁻²); and the epoch-79 std_ratio sits at order unity for most model–head pairs (poc_b 0.85/0.67, cnn_attention 0.64/0.62, poc_a 0.69/0.44, tcn 0.34/0.62 for φ_c/ψ respectively).

And yet: **validation circular loss for every periodic head, in every model, sat at the random-guessing value ≈ 1.0 for all 80 epochs** (Fig. 5.2) — poc_a φ_c 0.995 → 1.020, ψ 0.990 → 1.006; tcn 0.995 → 1.016 and 0.992 → 1.006; poc_b's *combination* losses likewise (combo A 0.999 → 0.999, combo B 1.006 → 0.991).
Crucially, the loss stayed flat even during epochs 40–79 when ‖**v**‖ ≈ 1 and gradient flow was demonstrably healthy — the strongest mechanistic evidence that the remaining failure is not an optimization artifact.
Higher-capacity trunks did *reduce training* circular loss substantially (to ≈ 0.49–0.60 for cnn_baseline, cnn_attention, resnet1d) with validation pinned at 1.0 throughout: they memorized the training set's phase labels, the behavior expected of a flexible model given a target that is unpredictable from its input.
("Memorized" here means overfitting to sample-specific structure without generalization, not verbatim lookup — with 25,000 continuous targets, exact memorization is implausible; the diagnostic content of the pattern is the train/validation gap itself, not the mechanism by which training loss falls.)

> **Figure 5.2.** Circular-loss (1 − cos Δθ) trajectories for φ_c and ψ (combo A/B for poc_b) across all seven architectures. Higher-capacity models memorize the training set — training loss falls as low as ≈ 0.49 — while validation loss stays pinned at the random-guessing value ≈ 1.0 in every case: the signature of a target unpredictable from the input, not of an optimization failure.

The learned uncertainty weighting corroborates this reading from a different direction (Fig. 5.3): the well-learned scalar heads rapidly acquire large effective weights (exp(−s) > 20 for chirp mass and merger time), while the periodic heads' weights stay near unity — the multi-task machinery correctly identifies them as uninformative rather than starving them of gradient.

> **Figure 5.3.** Uncertainty-weighting trajectories exp(−s_h) per head for all seven architectures. Scalar heads that learn (chirp mass, merger time) acquire large weights; the φ_c/ψ heads (and poc_b's combo heads, flat at ≈ 1.3) remain near unity.

### 5.6 Positive controls

The same checkpoints recover the control parameters well (Fig. 5.4): chirp mass R² = 0.926–0.963 (MAE 0.95–1.36 M_⊙), merger time R² = 0.909–0.921, SNR R² = 0.755–0.785, and sky position to 3.3°–10.0° mean angular error (median 0.0°) — with cnn_attention, the *worst* periodic-head model by ang_MAE, achieving the *best* sky localization (3.3°).
Inclination, though nominally a control head (§ 2.1), is deliberately absent from this list: § 5.4 traced its failure to an independent, unresolved mechanism (a Huber loss on its raw two-vector, with no `normalize_unit` in its path) unrelated to the φ_c/ψ pathologies, so its own null result neither corroborates nor undermines the claim below — it is excluded from the positive-control set for a documented, code-level reason, not omitted quietly.
Information that is present in the strain is extracted by this pipeline; the two phase angles specifically are not.

> **Figure 5.4.** Chirp-mass recovery on the validation set for the four Run 7 models (R² = 0.93–0.96). Together with merger time, SNR, and sky position, these positive controls establish that the shared trunk and training pipeline learn strain-derived parameters when the information is present.

## 6. Results: A Defended Null

### 6.1 Headline result

Table 6.1 collects the final validation angular MAE for every architecture, before and after the activation fix, against the analytic null.
The values are not merely *close to* the null; they bracket it within a few hundredths of a radian in both directions, across seven architectures, two head parameterizations, and every λ.

One scope note belongs here rather than in the Discussion, because it bears on how to read every number below.
These are *point* estimates under circular loss, and circular loss cannot distinguish "no information" from "a genuinely multimodal or ridge-like true posterior that no single-valued output can express" — a sampler retains a meaningful joint posterior over (φ_c, ψ) even where the degeneracy is exact (§ 8.1).
The claim defended below is about point-estimation regression specifically, not about the information-theoretic content of the strain.

**Table 6.1 — Validation angular MAE (rad) against the random-guessing null.** ("Pre-fix" = tanh outputs, Round 1; "post-fix" = linear outputs with λ = 0.01 where configured, Run 7 era.
Dashes: no pre-fix run.
Source: `pre_post_comparison.csv`.)

| Head (null) | TCN | ResNet1D | CNN base | CNN attn | Inception | poc_a | poc_b |
|---|---|---|---|---|---|---|---|
| φ_c (1.571) pre | 1.572 | 1.564 | 1.577 | — | 1.556 | — | 1.572 |
| φ_c post | 1.579 | 1.578 | 1.578 | 1.577 | 1.573 | 1.580 | 1.572 |
| ψ (0.785) pre | 0.787 | 0.787 | 0.787 | — | 0.792 | 0.779 | 0.779 |
| ψ post | 0.786 | 0.786 | 0.793 | 0.790 | 0.780 | 0.780 | 0.791 |
| ι (1.571) post | 1.544 | 1.588 | 1.570 | 1.559 | 1.571 | 1.560 | 1.573 |

The ι row is carried for diagnostic completeness, not as a positive control: as § 5.4 and § 5.6 establish, ι fails through a mechanism disjoint from the φ_c/ψ pathologies, so its presence at the null here is neither confirming nor disconfirming evidence for the degeneracy claim.

What distinguishes models is not accuracy but *failure style*, visible in the prediction distributions (Fig. 6.1): poc_a and tcn collapse most predictions onto one narrow mode (circ_r = 0.85–0.86); poc_b collapses almost totally (circ_r = 0.989); cnn_attention spreads its predictions broadly (circ_r = 0.43 for φ_c, 0.17 for ψ) — varied wrong answers instead of one constant wrong answer — with ang_MAE identical to everyone else's.
Output diversity is an architectural artifact; information recovery is uniformly nil.

> **Figure 6.1.** Predicted φ_c distributions (color) against the uniform truth (grey) for the four Run 7 models. Every model collapses to one or a few preferred angles (circ_r = 0.43–0.99) while angular MAE sits at the null π/2 ≈ 1.571 — optimal constant-prediction behavior for an unidentifiable target.

### 6.2 Confound elimination

A five-section verification plan was executed against the Run 7 checkpoints under an explicit gating rule: nothing downstream counts as evidence about the degeneracy until the interpretability of the metrics themselves is established.
Table 6.2 summarizes.

**Table 6.2 — Verification battery.**

| § | Confound | Check | Outcome |
|---|---|---|---|
| A.1 | Penalty silently inactive | config + code-path inspection | λ = 0.01 confirmed active in all four configs |
| A.2 | Raw-magnitude drift invalidates metrics | full 80-epoch std_ratio trajectories | 2/4 models fully healthy (poc_b, cnn_attention); tcn φ_c still declining (0.34, −0.0078/epoch), poc_a ψ stable-but-low (0.44) → flagged, pursued in §7 |
| A.3 | Gradient path dead | perturbation + gradient-chain trace | path healthy; an 89× prediction-sensitivity asymmetry initially flagged, closed by the calibrated standalone trace's paired-statistic channel (§ 6.6) |
| B | poc_b config bug | config diff vs poc_a | four intentional keys only; worse collapse *explained* by curriculum (below) |
| C | cnn_attention hides real learning | config diff + architecture trace | outlier metrics are attention-pooling variance artifacts; ang_MAE at null |
| D | Sub-null MAE is real signal | 10,000-permutation bootstrap | 11/12 model×head tests non-significant; φ_c and ψ: 0/8 |
| E | Learning hides in loud events | SNR-tercile stratification | no model–head pair shows a monotonic-and-material SNR trend |
| — | Ordering-confounded bootstrap | row-order audit | data i.i.d. (window variance ratio 0.991); artifact bound 0.013 rad |

Two of these deserve narrative attention.

**The poc_b anomaly is consistent with the null but confounded by curriculum design.** The combination-space model collapsed *more* severely than its baseline twin (circ_r 0.989 vs 0.848).
The config diff shows only the four intended poc-mode keys; the mechanism is the curriculum itself.
For near-face-on samples w(ι) ≈ 0 suppresses the poorly-constrained combination, so the model effectively trains one combination — a single constraint on the two-dimensional (φ_c, ψ) — and the optimal response to a single unlearnable constraint under circular loss is a sharp constant, which is what poc_b produces (a single peak carrying ~42% of all predictions).
The design would have exploited a breakable degeneracy; its collapse into the constant predictor is what that design does when the degeneracy is not breakable.
(Cost: a mild, predictable transfer-interference tax on chirp mass, MAE 0.977 → 1.024.)
This reading is the best mechanistic account available, but it is not the only one: the curriculum's near-face-on suppression of one combination could itself produce rank-deficient gradients independent of whether the underlying target is truly unlearnable, so poc_b is treated below (§ 8.2) as a distinct, curriculum-confounded consistency check rather than as an independent repeat of the certified null.

**The cnn_attention outlier is an artifact with a testable signature.** Its lower circ_r comes from learned attention pooling producing higher-variance features than fixed GAP/GMP readouts — spreading predictions without informing them (ang_MAE 1.597, *worst* of the four).
Its nominally significant inclination result is dissected below.

### 6.3 Statistical significance

**Table 6.3 — Label-permutation bootstrap (N = 10,000 shuffles, 5,000 validation samples, one-sided).** z > 0 means observed ang_MAE beats the shuffled-null mean.
Source: `bootstrap_output/bootstrap_ang_mae_20260721_093533.md`.

| Model | φ_c: z, p | ψ: z, p | ι: z, p |
|---|---|---|---|
| poc_a | −0.20, 0.579 | +0.46, 0.324 | +1.04, 0.152 |
| poc_b | −1.25, 0.895 | −0.05, 0.518 | +0.07, 0.464 |
| tcn | −0.56, 0.712 | −0.33, 0.630 | +1.21, 0.114 |
| cnn_attention | −2.43, 0.994 | +0.07, 0.472 | **+3.17, 0.0007** |

Eleven of twelve tests are non-significant; **all eight φ_c/ψ tests fail**, six of them on the *worse-than-shuffled* side, and every one of the eight observed values falls inside its own null 95% confidence interval (`bootstrap_output/bootstrap_ang_mae_20260721_093533.md` reports the null CI directly for each; e.g. tcn/φ_c: observed 1.578 against null CI [1.561, 1.588]).
Across the eight φ_c/ψ tests the null-CI half-widths range ≈ 0.002–0.023 rad, so this is not merely "not significant" — it is a comparatively tight bound against even a small real effect for most model–head pairs.
The single nominal detection — cnn_attention on inclination — is a genuine statistical outlier (z = +3.17, p = 0.0007), and it *does* survive Bonferroni correction across the twelve tests (threshold p < 0.0042; observed p = 0.0007 clears it) — we do not dismiss it on multiple-comparisons grounds.
We do not treat it as counter-evidence to the degeneracy claim, because it fails the same test this chapter applies to every other candidate signal: whether the effect grows with SNR.
Its effect size is Δ = 0.038 rad (2.2°), and that effect is uniform across SNR terciles (1.526/1.548/1.527) — flat, not increasing with signal strength.
This is not an ad hoc criterion invoked only here: it is the identical logic of the SNR stratification (§ 6.4) and of Step 3 of the pre-registered λ-retune gate (§ 7.1), applied consistently — a per-event phase-recovery effect must be stronger in louder events, while a population-level output-distribution bias need not be, and the inclination detection has the signature of the latter.
One honest scope limit: the SNR-monotonicity criterion was pre-registered as Step 3 of the § 7.1 λ-retune gate specifically, not as a general-purpose rule for the bootstrap battery; applying the same logic here is principled — it is the same physical argument, not a different one invented for this case — but it was not itself locked in writing before this particular test, and the reader should weigh the dismissal accordingly rather than as a pre-registered result.
Read on its own terms rather than folded into the degeneracy verdict, this is a genuine anomaly — possibly a population bias in this model's attention-pooling readout, possibly a real but weak effect specific to inclination — and the honest report is that it does not generalize to either φ_c or ψ, where all eight tests land at or worse than the shuffled null.
A row-ordering audit additionally bounded the largest possible ordering artifact at 0.013 rad and confirmed the permutation null is valid (validation data i.i.d.; window variance ratio 0.991).

### 6.4 SNR stratification

If weak phase information were present but diluted, it would concentrate in loud events.
It does not.
Across terciles (SNR [7.0, 9.6] / [9.7, 12.3] / [12.3, 15.0], n ≈ 1,666 each): every ψ result lies within 0.007 rad of the null in every tercile for every model; three of four models are non-monotonic in φ_c; the single monotonic case (tcn: 1.633 → 1.557 → 1.543) drops by 0.090 rad from low to high SNR — larger than the 0.038-rad artifact scale established in § 6.3, and large enough that, read in isolation, it could look like a genuine SNR-dependent trend.
Two facts argue against that reading.
First, the more diagnostic comparison is not tercile-to-tercile but each tercile against the null (1.5708 rad): the low tercile sits *worse* than random guessing (1.633, Δ = −0.062 vs null), and even the high tercile clears the null by only 0.027 rad — itself beneath the artifact scale — so the 0.090-rad swing is mostly the low tercile's anomalous departure *from* the null, not the high tercile's departure *toward* a genuine signal; a real degeneracy-breaking effect should make high-SNR predictions better than random, not merely less bad than an anomalously poor low-SNR estimate.
Second, tcn/φ_c is the one model whose std_ratio never converged to the healthy band (§ 7) at λ = 0.01, so its ang_MAE trend across any stratification is doubly uninterpretable as evidence of learning.

### 6.5 Run 8: the λ = 0 ablation

One loose thread survived the battery: Run 7's validation circular losses *rose* slightly over training (+0.014 to +0.025).
A rising loss could mean the magnitude penalty was actively corrupting predictions — an engineering artifact worth fixing, not more null evidence — so the two TCN-trunk baselines were re-run with λ = 0 (Fig. 6.2).

**Table 6.4 — Endpoint drift of validation circular loss.**

| Model | Head | Δ at λ=0 | Δ at λ=0.01 | Reading |
|---|---|---|---|---|
| poc_a | φ_c | +0.0010 | +0.0249 | artifact of λ (resolved) |
| poc_a | ψ | +0.0002 | +0.0163 | artifact of λ (resolved) |
| tcn | φ_c | +0.0011 | +0.0204 | artifact of λ (resolved) |
| tcn | ψ | **+0.0072** | +0.0136 | persists at half size — unexplained, small |

Three of four drift signals vanish with the penalty off: the creep was a penalty/uncertainty-weighting interaction, not anti-learning.
The fourth (tcn ψ) persists at +0.0072 on a loss of ~1.0 and is recorded as unexplained-but-small rather than forced into the majority pattern.
The ablation also confirmed it was genuinely an off-state — without the penalty, std_ratio diverged to 73–83 — and final MAEs at λ = 0 landed on the same nulls as every other run (φ_c ≈ 1.579; ψ ≈ 0.780–0.785).

> **Figure 6.2.** λ = 0 ablation for poc_a and tcn: validation/training circular loss and validation std_ratio for φ_c (top) and ψ (bottom); λ = 0 (solid) vs the λ = 0.01 references (dashed). Removing the penalty eliminates the upward validation-loss drift (e.g. +0.0010 vs +0.0249 for poc_a φ_c) but lets ‖v‖ diverge (std_ratio > 140 at peak). Circular loss remains at ≈ 1.0 in every configuration.

### 6.6 Closing the last open probe: the 89× sensitivity asymmetry

Verification item A.3 left one mechanistic loose end: after a single optimizer step, the periodic heads' predictions moved with relative change ~1.6 × 10⁻², some 89× more than the converged chirp-mass head's 1.8 × 10⁻⁴ — movement a one-step snapshot cannot classify as slow learning or noise.
The multi-step trace built to resolve this was gated behind the pre-registered stability gate and therefore never ran during the retune; after the retune closed, it was decoupled into a standalone script and executed against the Run 7 checkpoints (25 consecutive gradient steps on a fixed 128-sample batch, with predictions tracked on a disjoint 512-sample probe).

**Table 6.5 — Standalone multi-step perturbation trace, at the converged checkpoints ("final") and after a fresh initialization plus ~1-epoch warmup ("early").** net/sum is the ratio of net displacement to summed per-step displacements (random-walk reference 1/√25 = 0.20). Δ is the paired per-sample probe change over the 25 steps with its paired standard error's t statistic — circular loss for φ_c/ψ, transformed-target MSE for mchirp. Source: `perturbation_trace_output/`.

| Stage | Model | φ_c: net/sum, Δ (t) | ψ: net/sum, Δ (t) | mchirp: net/sum, Δ (t) |
|---|---|---|---|---|
| final | poc_a | 0.30, −0.018 (−0.9) | 0.63, +0.043 (+0.8) | 0.04, +0.021 (+4.2) |
| final | poc_b | 0.39, −0.013 (−1.0) | 0.23, +0.004 (+1.7) | 0.04, +0.040 (+9.0) |
| final | tcn | 0.92, −0.010 (−0.2) | 0.36, +0.050 (+1.6) | 0.03, +0.017 (+4.3) |
| final | cnn_attention | 0.41, −0.044 (−1.1) | 0.34, +0.023 (+0.7) | 0.19, +0.065 (+4.4) |
| early | poc_a | 0.16, −0.017 (−1.6) | 0.14, +0.004 (+0.1) | 0.09, **−0.158 (−5.2)** |
| early | poc_b | 0.08, −0.006 (−0.7) | 0.09, +0.001 (+0.1) | 0.13, **−0.129 (−3.4)** |
| early | tcn | 0.24, −0.002 (−0.0) | 0.15, +0.035 (+0.6) | 0.10, **−0.113 (−4.1)** |
| early | cnn_attention | 0.42, +0.001 (+0.0) | 0.37, −0.031 (−0.6) | 0.29, **−0.646 (−8.5)** |

Two observations stand on arithmetic alone.
First, the asymmetry is real and now explained: the periodic heads' raw outputs move coherently and far (net/sum up to 0.93 against the 0.20 random-walk reference), while the converged scalar control barely moves net (net/sum 0.03–0.16; its positive cosine similarity is attributable to Adam momentum, which correlates consecutive steps for every head).
Second, the coherent movement is dominantly *radial*, not angular: the circular loss depends only on the predicted angle, and its change over 25 steps is bounded by |Δ| ≤ 0.051 on a loss of ≈ 1.0 while the raw displacement is a large fraction of the output norm — large coherent raw movement with near-zero angular consequence is precisely the ‖**v**‖-magnitude dynamics of § 5.4, and the two most directional cases (tcn/φ_c, poc_a/ψ) are exactly the two heads with the known std_ratio pathologies.

The first run of the trace was reviewed before acceptance, and the review reshaped the instrument itself; the sequence is reported in order because each step gates the next.
The review raised two objections: the positive control had failed — mchirp, the best-learned head in the investigation (R² ≈ 0.96 throughout), read AMBIGUOUS at every converged checkpoint, so the classifier could not distinguish a demonstrably learned head from a dead one — and the probe-loss deltas lacked the correct *paired* statistic (the first run recorded only means, and a marginal-SE comparison is invalid for a before/after change on the same fixed samples).
Both were remediated: per-sample paired statistics were added to the instrument, and a calibration stage was defined with a pre-stated criterion — rerun the trace near the start of training (fresh initialization plus ~1-epoch warmup, since per-epoch checkpoints were never saved), where a genuinely learning mchirp *must* read directional, with failure defined to retire the classifier.

**The calibration failed its criterion — and in failing, validated the measurement that matters.**
Early-stage mchirp never read directional (net/sum 0.09–0.29) *while its probe MSE was collapsing at t = −3.4 to −8.5*, the strongest learning signal in the entire table: the displacement-geometry classifier labels the fastest-learning head "noise-like."
The mechanism is instructive — learning is not rigid drift; per-step prediction deltas decorrelate as different samples' errors are corrected — and the verdict is mechanical: per the pre-stated tree, the geometry classifier is retired and no directional/ambiguous verdict from either run is used.
The same run, however, constitutes a *passed* positive control for the instrument's other channel: the paired probe-loss statistic detects early mchirp learning decisively in all four models, and on that validated channel every periodic head is null at both stages (early |t| ≤ 1.6, final |t| ≤ 1.7, signs mixed across models and heads).
The early stage is the sharper result: a within-run, stage-matched, positive-controlled contrast in which the same instrument, over the same 25 steps, watches mchirp learn steeply while φ_c and ψ sit at the random baseline.
The paired statistic also extinguishes the one nominal escalation trigger from the first run — tcn/φ_c, directional at net/sum 0.92 with an apparently decreasing probe loss, measures −0.0097 ± 0.0491 (t = −0.20) — and the radial explanation of the original 89× asymmetry stands on arithmetic independent of the retired classifier: raw periodic outputs move by a large fraction of their norm while the angular loss moves by at most |0.051|, and the two largest raw movers are the two known std_ratio-pathology heads.

Two secondary observations complete the account.
First, the final-stage mchirp deltas are significantly *positive* — probe MSE worsens by +0.017 to +0.065 (t = +4.2 to +9.0) — which is not a probe artifact but the expected signature of resuming training on a single fixed batch: at convergence the full-data gradient is ≈ 0 while the batch gradient is not, so 25 repeated steps move the head along batch-specific directions at the expense of the held-out optimum.
Three pieces of in-hand evidence support this reading over a data-loading quirk: the probe pipeline is stage-independent yet yields strongly negative deltas early and positive ones late; the gradient batch is disjoint from the probe by construction, so the degradation is genuine out-of-batch generalization loss; and the effect appears only on the head with a real strain-to-target mapping — a converged optimum to be dragged away from.
Read as an unplanned second control, it strengthens the final-stage null: the paired channel demonstrably has the power to detect a real, if artifactual, effect at the converged stage — and φ_c/ψ still show nothing against that backdrop.
Second, on sensitivity: the per-combo minimum detectable effect (≈ 2× the paired SE) spans ≈ 0.005–0.10 loss units across the twelve cases, so the individual φ_c/ψ point estimates (up to |0.050|) are non-significant partly because the single-batch probe design yields wide standard errors — the final-stage per-combo null is "not rejected and bounded at the few-percent level," not "excluded to arbitrary precision."
This is adequate for A.3's mechanistic role: the degeneracy verdict itself is carried independently by the N = 10,000 label-permutation bootstrap (§ 6.3) and the SNR stratification (§ 6.4), not by this trace.

**A.3 is provisionally closed, pending replication on a fresh holdout set**: the 89× asymmetry was radial movement without angular learning.
The disposition honors the pre-stated decision tree in order — the calibration criterion failed, so the classifier it governed is retired — and the closure is then re-founded, transparently post hoc, on the surviving channel, which carries its own passed control from the same run; that residual caveat is recorded in § 8.3 rather than hidden.

### 6.7 Inclination stratification: does the degeneracy weaken edge-on?

The one population slice never separately reported in the original verification battery is inclination itself.
The analytic study of § 3 predicts the φ_c/ψ degeneracy is exact face-on and only partially breakable edge-on, so a recoverable signal, if present, should concentrate there.
`inclination_stratification.py` was run against the four λ-matched Run 7 checkpoints, splitting the 5,000-sample validation set into the same face-on (|cos ι| > 0.9, n = 1,442), mixed (0.5 ≤ |cos ι| ≤ 0.9, n = 1,918), and edge-on (|cos ι| < 0.5, n = 1,640) bands used throughout § 3.

**Table 6.6 — Inclination-stratified ang_MAE, edge-on band vs null (rad).** Positive Δ = edge-on ang_MAE *better* than the random-guessing null (ang_MAE < null); negative Δ = *worse* than null. Source: `inclination_output/inclination_stratification_20260723_130630.md`.

| Model | φ_c edge-on Δ vs null | ψ edge-on Δ vs null | Same-model ι-band noise floor (max \|Δ\|) |
|---|---|---|---|
| poc_a | +0.0241 | −0.0227 | 0.0569 |
| poc_b | +0.0096 | +0.0054 | 0.0215 |
| tcn | −0.0407 | −0.0083 | 0.0225 |
| cnn_attention | −0.0087 | −0.0189 | 0.0944 |

No model shows a cross-consistent, edge-on-favoring recovery of either angle.
Of the eight φ_c/ψ deviations, the largest in the *hoped-for* direction is poc_a/φ_c (+0.0241 rad, and the only one of the eight monotonic across all three bands: 1.6042 → 1.5623 → 1.5467); the largest in the *anti* direction is tcn/φ_c (−0.0407 rad).

Neither is evidence of recovered signal, for the same reason § 6.4 already established for the SNR axis, extended here with a model-matched calibration.
Because inclination is itself a labeled quantity, each model's already-diagnosed, uninformative ι head (§ 5.4) provides a same-model, same-binning noise floor for exactly this stratification: ι carries no information about φ_c/ψ by construction, so its own band-to-band swings measure how much a completely uninformative head fluctuates from finite-band sampling alone at these N.
That floor is 0.0569 rad for poc_a, 0.0215 for poc_b, 0.0225 for tcn, and 0.0944 for cnn_attention (the last consistent with its already-diagnosed high-variance attention-pooling readout, § 6.2) — and every φ_c/ψ deviation above falls at or below its own model's floor, with one exception.
tcn/φ_c's −0.0407 rad edge-on deviation exceeds tcn's own 0.0225-rad floor, but in the *worse-than-null* direction and on the one head this chapter already flags as never achieving certified metric health (std_ratio never converged, § 6.2, § 7): an unstable head producing an oversized swing, in the direction a real signal could not produce, is consistent with its documented instability, not a new finding.

One honest limit on this argument, raised on review: the ι-noise-floor comparison is a *diagnostic calibration*, not a formal statistical test, and it assumes ι's band-to-band fluctuation is independent sampling noise rather than a symptom shared with φ_c/ψ's own failure — e.g. a common-cause pathology upstream of any individual loss (the shared trunk, the shared batch-normalization front end, or the two-layer MLP head design) that makes *any* angular target unrepresentable regardless of which loss trains it.
Two facts argue against a shared-cause reading, short of a fully independent statistical proof.
First, § 5.4's code-level trace already rules out one specific common cause: ι's failure runs through a Huber loss on its raw two-vector with no `normalize_unit` in the path at all, so it cannot share the tanh-saturation or magnitude-drift mechanisms diagnosed for φ_c/ψ — there is no single bug common to all three periodic heads, only three code paths independently inspected.
Second, the sky-position head — genuinely directional (a von Mises–Fisher likelihood on the unit sphere, § 2.2), and dependent on inter-detector phase information in a way analogous to φ_c/ψ — is recovered well (3.3°–10.0° mean error, § 5.6): the shared trunk demonstrably carries usable directional information into a head architecture in the same family when that information is present, which weighs against an architecture-wide inability to represent angular targets.
Neither point converts the ι-noise-floor comparison into a formal test; it remains a calibration heuristic, and a stronger version — stratifying a known-good scalar control by the same inclination bands to confirm banding itself does not inject spurious metric variance — is reported next rather than left as future work.
A further caveat, distinct from the two points above: the independence assumption is plausible but not proven against a head-capacity confound specifically, since ι, φ_c, and ψ all share the same two-layer MLP head architecture, so a shared limitation in mapping the trunk's representation to angular targets — rather than a shared loss-path bug, which is already ruled out — could in principle produce correlated failure across all three periodic heads.
The code-level and sky-position arguments above bound this concern but do not rule it out, and it is not tested further here.

**Table 6.7 — Chirp-mass MAE/R² across the same face-on/mixed/edge-on bands (scalar-control check).** Source: `inclination_output/inclination_control_stratification_20260723_140016.md`.

| Model | face-on R² | mixed R² | edge-on R² | R² spread | MAE spread (M_⊙) |
|---|---|---|---|---|---|
| poc_a | 0.9639 | 0.9566 | 0.9626 | 0.0073 | 0.0460 |
| poc_b | 0.9612 | 0.9525 | 0.9605 | 0.0087 | 0.0801 |
| tcn | 0.9646 | 0.9549 | 0.9616 | 0.0097 | 0.0508 |
| cnn_attention | 0.9265 | 0.9219 | 0.9272 | 0.0053 | 0.0343 |

`inclination_control_stratification.py`, prepared alongside this revision, reruns the same three bands against chirp mass, a head we already know is well-learned and non-angular (§ 5.6).
R² stays within [0.92, 0.97] for every model, with band-to-band R² spread of only 0.005–0.010 and MAE spread of 0.034–0.080 M_⊙ against a baseline MAE of 0.95–1.37 M_⊙: the banding scheme itself does not inject variance at a scale that could explain the φ_c/ψ deviations in Table 6.6.
This directly answers the weaker of the two circularity concerns above — whether merely slicing the validation set this way manufactures metric noise — and it does not.
It does not answer the stronger concern (a possible common-cause pathology specific to angular representations, which a non-angular control cannot rule out by construction); that remains bounded only by the § 5.4 code-level trace and the sky-position argument above, not eliminated.

This closes the gap the adversarial reviews correctly identified.
The axis along which the analytic study predicts the degeneracy should weaken was tested directly, and no model recovers a signal there that exceeds its own uninformative control's sampling noise.
The null of § 6 is not confined to the face-on-dominated aggregate; it holds, band by band, across the one population split most likely to break it.

## 7. The Pre-Registered λ Retune

### 7.1 Why pre-register an engineering sweep

After § 6, two model–head pairs remained *formally uninterpretable* rather than cleanly null: tcn/φ_c (std_ratio still declining at λ = 0.01) and poc_a/ψ (stable but below the healthy band).
The obvious next step — raise λ and see whether the numbers "look better" — is precisely the kind of after-the-fact eyeballing that had already misled this investigation three times (the R² = 0.75 collapse artifact; an endpoint-only std_ratio summary that concealed a mid-training crash-and-recovery; the Run 8 drift artifact).
The grid itself — four points, {0, 0.01, 0.05, 0.10}, with a stopping rule guaranteeing termination after the fourth — was chosen deliberately coarse, and the reason is worth stating rather than leaving implicit: the working hypothesis going in was that λ was not the binding constraint on these two heads (the degeneracy evidence of § 6 already pointed at the target itself, not at the magnitude penalty), so the sweep was designed to test that hypothesis cheaply rather than to search exhaustively for an optimum, and an open-ended tuning loop was exactly the kind of after-the-fact flexibility this chapter's pre-registration discipline exists to foreclose.
The decision criteria were therefore **locked in writing before any retune ran** [@nosek2018preregistration], with the verdict computed mechanically by the diagnostic scripts, not by inspection:

- **Scope.** Exactly two primary tests: tcn/φ_c and poc_a/ψ. All else exploratory, with promotion to "primary" after the fact explicitly forbidden.
- **Step 0 — interpretability gate.** std_ratio healthy iff < 10% of the final 40 epochs fall outside [0.5, 2.0] *and* the linear trend over those epochs is within ±0.005/epoch. Failure at λ = 0.05 → try λ = 0.10; failure at both → report "λ alone insufficient," counted **neither as null nor as counter-evidence**.
- **Step 1 — significance.** The § 6.3 bootstrap, Bonferroni-corrected for two tests (p < 0.025).
- **Step 2 — effect-size floor.** Δang_MAE ≥ 0.10 rad (≈ 5.7°) — set ≈ 3× above the largest known metric artifact (0.038 rad) and ≈ 8× above the ordering bound (0.013 rad).
- **Step 3 — SNR consistency.** Improvement monotonic non-decreasing across terciles, with the high-SNR tercile independently clearing the floor.

Only a result passing all four steps would count as counter-evidence to the degeneracy hypothesis.

### 7.2 Outcome

**Table 7.1 — Pre-registered gate results.** Source: `lam005_retune_output/`, `lam010_retune_output/`.

| λ | Test | frac. unhealthy (last 40 ep) | trend /epoch | Gate |
|---|---|---|---|---|
| 0.05 | tcn / φ_c | 0.05 ✓ | −0.0064 ✗ | **FAIL** |
| 0.05 | poc_a / ψ | 0.35 ✗ | +0.0072 ✗ | **FAIL** |
| 0.10 | tcn / φ_c | 0.28 ✗ | −0.0026 ✓ | **FAIL** (worse) |
| 0.10 | poc_a / ψ | 0.73 ✗ | +0.0073 ✗ | **FAIL** (much worse) |

At λ = 0.05 both failures were near-misses with a shared shape — each trajectory settles into a healthy plateau (0.58–0.62 and 0.53–0.56 respectively) only in the last 15–20 epochs, and the 40-epoch window still contains the climb into that band (Fig. 7.1).
At λ = 0.10 the failures were not near-misses and diverged in character: poc_a/ψ crashes through epochs ~30–48 and is still recovering at epoch 80 (crossing 0.5 only in the final 11 epochs), while tcn/φ_c oscillates across [0.2, 0.95] with no convergence at all (Fig. 7.2).
The exploratory pairs mostly regressed too: λ = 0.10 helped none of the four TCN-trunk head/model combinations and worsened three.
Throughout all of it, circular loss stayed at 1.004–1.008 — flat at the null, as in every preceding run.

> **Figure 7.1.** Run 9a, λ = 0.05, overlaid on λ = 0 and λ = 0.01 references. The penalty holds std_ratio at order unity, but neither pre-registered target passes the stability gate, and validation circular loss remains at ≈ 1.0.

> **Figure 7.2.** Run 9b, λ = 0.10, same layout. The stronger penalty fails the gate more severely (poc_a/ψ 73% of late epochs unhealthy) while circular loss stays at the random-guessing plateau.

### 7.3 Verdict

Per the pre-registered decision table — applied as written, not re-derived after seeing the data — the result is: **λ alone is insufficient to stabilize std_ratio for either primary target at the four pre-registered values {0, 0.01, 0.05, 0.10}**; per the pre-registered stopping rule, this sweep is terminated, and any further pursuit is architecture-level or a fresh, separately pre-registered sweep (below).
This is a claim about the four-point pre-registered sweep and its stopping rule, not about the λ dimension as a whole: the response to λ is *peaked*, not monotonic — λ = 0.05 produced near-misses on both targets while λ = 0.10 regressed on every head/model combination — so a finer, freshly pre-registered mini-sweep over λ ∈ [0.02, 0.08] plausibly contains a stability optimum that this four-point sweep was never positioned to find.
It was not pursued here for reasons of time and is recorded as future work (§ 8.5) rather than claimed to be ruled out.
The two pairs are filed as *uninterpretable*, not as additional nulls and not as counter-evidence.
Steps 1–3 never executed for any configuration, at any λ — which is itself a summary statistic of the whole investigation: **no result ever cleared even the first gate of a pre-declared path to counter-evidence.**

Two disciplined refusals are worth recording.
The gate's 40-epoch window is arguably miscalibrated for the plateau learning-rate schedule's late settling — a real concern, flagged during Run 9a — but revising a criterion between pre-registered arms, with results in hand, would reintroduce exactly the flexibility pre-registration exists to remove; the question is documented for future gates instead.
And the near-miss λ = 0.05 trajectories *look* healthy at endpoint — the temptation to wave them through was declined for the same reason.

## 8. Discussion

### 8.1 What the evidence supports

The degeneracy hypothesis — φ_c and ψ carry no strain-only recoverable signal for this population in this architecture/loss family — is the best-supported reading of nine training campaigns:

1. validation circular loss flat at the random value ≈ 1.0 in every run, every model, every λ, with healthy gradients and unit-scale magnitudes over the second half of training in the clean models;
2. 0 of 8 φ_c/ψ bootstrap tests significant (11 of 12 overall), with the single scalar-head detection explained as a population bias;
3. no SNR-dependent improvement anywhere — the direction any real strain-derived effect must point;
4. the combination-space model, purpose-built to exploit the analytically confirmed conditioning structure of § 3, collapsing *into* the constant predictor exactly as it should if the residual per-sample information is nil;
5. positive controls learned to high fidelity from the same features throughout;
6. no model recovers an edge-on signal exceeding its own uninformative ι-control's sampling noise (§ 6.7) — the one population axis along which the analytic study predicts the degeneracy should weaken shows no cross-model, edge-on-favoring trend either.

Two points sharpen what is and is not being claimed.
First, the claim is *conditional*: it applies to point-estimating regression on this SNR-7–15, two-detector, dominant-mode population, under circular loss — not to Bayesian posterior recovery (a sampler still recovers a meaningful *joint* posterior over (φ_c, ψ)), not to louder signals, not to alternative data representations (e.g. frequency-domain or time–frequency inputs), and not to configurations with inclination supplied as side information (§ 8.5).
Second, the analytic study of § 3 makes the null *informative* rather than merely disappointing: a combination-conditioning ratio of only ~1.16× population-averaged — concentrated face-on, where the curriculum then down-weights it to zero — quantifies how little marginal structure there was for a per-event point estimator to find.
The networks did not miss an obvious target; they confirmed, at scale, how small the target is.

### 8.2 Sufficiency of the architecture pool

A natural objection to any architecture-crossed null is: *why only seven models?* The answer is that architecture count is the wrong axis for defending this particular claim, and the design reflects that deliberately.

A comparative claim ("architecture X is best for task Y") grows stronger with breadth, and every untried model is a hole in it.
A null of the form "the input–label relationship carries no recoverable signal" does not: it grows stronger with *diversity of inductive bias* and with *mechanism-level evidence*, both of which saturate quickly.
On the first, the five trunk families (§ 4.2) span qualitatively different hypotheses about how phase information could be encoded — long-range dilated convolution, local hierarchical convolution, content-based attention, explicit multi-scale filtering, and deep residual composition — rather than five samples from one family.
On the second, the decisive observation is that **capacity was demonstrably not the binding constraint**: the higher-capacity trunks drove *training* circular loss down to ≈ 0.49–0.60 while validation loss stayed pinned at 1.0 (Fig. 5.2).
That is precisely what added capacity does against an uninformative target — it memorizes.
An eighth architecture buys another memorization curve, not information absent from the strain; once the failure is localized in the data by flat validation loss under certified-healthy gradients, in agreement with the analytic degeneracy of § 3, architecture search has no remaining expected payoff.
In one sentence: *the null is a statement about the data, not about the hypothesis class, and the memorization gap is the evidence that the hypothesis class was not the limit.*

This memorization-gap argument was illustrated in aggregate (Fig. 5.2, all seven architectures); an adversarial reviewer correctly asked whether it holds for the two models the null is actually *certified* on, individually, rather than for the pool as a whole.
It does, but asymmetrically, and the asymmetry is worth reporting rather than folding into the aggregate.
cnn_attention's own training/validation circular loss (Fig. 8.1) reproduces the pool-level pattern exactly: training loss falls to 0.60 (φ_c) and 0.52 (ψ) by epoch 79 while validation stays at 1.01/1.00 — this model demonstrably has, and uses, spare capacity to fit sample-specific structure, and still generalizes none of it.
poc_b does not: its combo-A/combo-B training loss tracks validation almost exactly throughout training (0.988/0.998 vs 0.999/0.991 at epoch 79), the same flat-everywhere pattern shown by the other two TCN-trunk configurations, poc_a and tcn (train 0.98–0.97 vs val 1.02–1.01 at epoch 79), regardless of head parameterization.
This is not evidence that the TCN trunk lacks capacity: the identical trunk reaches chirp-mass R² = 0.97 (train) / 0.96 (val) on poc_a — ample nonlinear capacity when a learnable target exists.
Read together, the two certified models fail in two different, individually informative ways: cnn_attention finds and fits sample-specific structure it cannot generalize (the overfitting-without-learning pattern already qualified in § 5.5); poc_b's optimizer finds no exploitable structure even in-sample, on a trunk independently proven capable of high-capacity fits elsewhere.
Neither pattern is consistent with a capacity-starved null; the second is, if anything, the cleaner of the two.
To be explicit about evidential weight: this section's architecture-sufficiency argument rests primarily on cnn_attention's memorization gap, the cleanest single demonstration that spare capacity finds and cannot generalize structure from an uninformative target.
poc_b (§ 6.2) is a different, curriculum-confounded consistency check, not an independent repeat of the same certification — its flat training loss could in principle reflect the curriculum's near-face-on suppression of one combination rather than the target's unlearnability alone, so it corroborates from the optimizer side without carrying the same evidential weight as cnn_attention's memorization gap.

> **Figure 8.1.** Train vs. validation circular loss for the two certified models specifically, generated directly from each run's `history.csv` (`plot_certified_memorization.py`, no model loading or GPU use). cnn_attention (right column) shows the memorization gap (train ↓ to 0.60/0.52, val flat ≈ 1.0); poc_b (left column) shows no gap at all — train tracks validation throughout, the same pattern shown by the other two TCN-trunk configurations (poc_a, tcn).

The progressive narrowing of the pool — seven configurations, then four, then two — likewise reflects criteria, not attrition.
The seven were never seven independent tests but five trunks plus the two head parameterizations on the primary trunk; the cut to four (poc_a, poc_b, tcn, cnn_attention) selected the λ-matched subset after the magnitude penalty was introduced (§ 4.5), the remaining three trunks retaining evidential value as unmatched corroboration; and the cut to the two *certified-clean* models (poc_b, cnn_attention) was imposed by the pre-declared std_ratio interpretability gate (§ 6.2), not by preference among results.
Each restriction is documented and criterion-based, and the excluded configurations agree with the included ones at every point where they can be compared.

For the same reason, the deliberate decision *not* to expand the architecture pool in future work (§ 8.5) is scoping rather than omission: the evidence localizes the failure in the information available to the network, so the productive next lever is conditioning information (ι-conditioning), not a broader model zoo.
A single modern-family spot-check (e.g. a state-space sequence model at λ = 0.01) would be a cheap robustness supplement, but per unit of compute, seed replication of an existing configuration addresses a larger residual risk (§ 8.3) than any additional architecture.

### 8.3 Threats to validity

We record the limitations explicitly, in roughly descending order of concern.

- **Two configurations remain uninterpretable.** tcn/φ_c and poc_a/ψ never achieved certified metric health at any λ; the clean degeneracy test rests on poc_b and cnn_attention (plus the ablation and pre/post-fix consistency of every other configuration). The null is multiply corroborated, but its *certified* base is narrower than its table count.
- **The A.3 closure rests on a within-run-validated channel.** The trace's pre-stated displacement-geometry classifier failed its positive control at calibration and was retired (§ 6.6); the closure rests on the paired probe-loss statistic, whose own control passed emphatically in the same run (early mchirp |t| = 3.4–8.5) but which was added on review rather than pre-registered. The evidence is positive-controlled; the channel choice is post hoc — recorded here rather than hidden.
- **Waveform-model provenance is a reproducibility gap, not merely a caveat.** All injections are understood to use the dominant quadrupole (ℓ = 2, |m| = 2) mode only (IMRPhenomD, no higher-order modes) — the condition under which the face-on degeneracy argument of § 1 is exact — but this is not preserved as metadata in the HDF5 dataset, nor in any generation script retained in this repository; it is recorded here from the authors' knowledge of how the dataset was produced.
The primary concern this raises is reproducibility, not the direction of the null: a reader cannot regenerate or independently verify the dataset from what is provided, and on review the honest position is that this is not acceptable practice, not a footnote — any follow-up work should regenerate the dataset from a version-controlled script that records its own configuration before relying further on this result.
Separately, and only as a secondary point, the direction of the risk is asymmetric: if this recollection were wrong and higher-order modes were in fact present, the degeneracy would be only approximately exact even face-on, which would *sharpen* rather than weaken the null (a network failing to exploit even a HOM-broken degeneracy is stronger evidence of no recoverable signal) — but this does not excuse the provenance gap, which is a threat to the chapter's reproducibility independent of which way it would cut if resolved.
- **Residual anomalies.** tcn/ψ's +0.0072 validation-loss drift at λ = 0 remains unexplained (small, on a loss of ~1.0); the inclination head's failure mechanism (Huber path, no normalization) is documented as distinct but is itself unresolved; and a sky-position degradation specific to the `SumDiffTrainer` runs (8.2°–10.0° vs 3.3°–4.5° for the plain baselines) was flagged in the record but never investigated — a known open issue, out of scope for this chapter.
- **Design constraints.** Single seed (42) per configuration — replication is across architectures and λ values, not across seeds; the five-trunk comparison is not λ-matched (three trunks trained at λ = 0), though the four models carrying the verified null are matched; the curriculum used the analytic sin²ι weight, not the Jacobian-fitted variant it was validated against; fixed 80-epoch budget with no early stopping; a single simulated dataset with a single noise realization per event; and the inclination prior is uniform in ι itself rather than in cos ι, the astrophysically correct isotropic choice, which over-represents edge-on systems relative to reality — since edge-on is where the degeneracy is weakest, this deviation does not appear to bias the claim in a self-serving direction, but it should have been isotropic and was not.
- **Metric-versioning note.** Two evaluation passes over the Run 7 checkpoints differ by up to 0.026 rad in periodic ang_MAE (both straddling the null; no verdict affected). Tables 6.1 and 6.3 use the self-consistent bootstrap/stratification set.

None of these, alone or jointly, provides a mechanism by which a real, useful φ_c/ψ signal could have hidden from all of § 6 — but they bound the strength of the claim honestly.

### 8.4 Methodological lessons

Three transferable lessons emerge from the diagnostic arc, and they are the part of this chapter most likely to outlive its specific numbers.

**Aggregate metrics are not evidence until their mechanism is verified.** Every major error in this investigation was a plausible reading of a true number: R² = 0.75 (mode collapse in disguise), an endpoint std_ratio of 0.62 (concealing a crash to 0.07 and recovery), rising validation loss (a regularizer interaction), a "significant" p = 0.0007 (population bias).
The correctives were never better metrics but mechanism inspection: forward-pass dumps, gradient-chain traces, full trajectories, code-level loss-path audits.
For null results especially, the epistemic burden inverts: a *flat* loss is only informative once the gradient path that would have moved it is certified live.

**Controls must be validated at the code level.** The inclination head's coincident failure was, for a time, load-bearing evidence for a shared-mechanism (hence fixable, hence non-physical) explanation.
It shared the encoding, the architecture, and the failure signature — but not, it turned out, the loss path.
A control that resembles the treatment in every visible metric can still differ in the one line of code that matters.

**Pre-registration has a place inside engineering loops, not just at study level.** The λ retune's gates were written down precisely because the investigators had watched themselves misread results three times.
The payoff was concrete: a near-miss that endpoint-eyeballing would have waved through was held to the written criterion, a mid-stream temptation to recalibrate the gate was resisted and documented instead, and the final filing — "neither null nor counter-evidence" — is a category that post-hoc reasoning almost never produces but that honest accounting sometimes requires.

### 8.5 Future work

Three successors are scoped, in priority order.
(Two items originally listed here as immediate checks have since run and are no longer future work: the inclination-stratified breakdown, § 6.7, and the scalar-control chirp-mass-by-inclination-band check, Table 6.7.)
(i) **Inclination conditioning**: supply true (sin ι, cos ι) as an auxiliary input — the analytic structure of § 3 says the well-constrained combination is knowable *given* ι, so this tests whether the degeneracy is breakable with side information; a full implementation plan exists (train-time truth, with inference-time ι estimation explicitly out of scope).
(ii) An **architecture-level attack on the std_ratio instability** for the two uninterpretable pairs (the pre-registered λ-sweep's named next lever), or an explicit decision to close that thread; alongside it, a finer freshly pre-registered λ mini-sweep over [0.02, 0.08], motivated by the peaked λ-response observed in § 7.3.
(iii) A **posterior-estimation reformulation**: the natural follow-on from "point estimation is hopeless" is not resignation but a conditional-density head (e.g. a von Mises mixture over the well-constrained combination), which the present chapter's null both motivates and baselines.

## 9. Conclusion

We set out to determine whether the coalescence phase and polarization angle of compact-binary signals are recoverable from strain by direct neural regression, individually or in their physically motivated sum/difference combinations.
After eliminating two genuine optimization pathologies that initially masqueraded as (and then obscured) the answer — tanh saturation at initialization, and magnitude drift induced by a norm-blind circular loss — and after subjecting the resulting flat learning curves to a verification battery spanning configuration audits, permutation tests, population stratification, an ablation, and a pre-registered two-arm regularization sweep, the answer is: **no**.
Every model, at every capacity and every regularization strength, converged to the optimal constant predictor, at the analytic random-guessing error, while learning four other parameters well from the same shared representation — and no result at any point cleared even the first gate of a pre-declared path to counter-evidence.
The fully certified base of that claim is two λ-matched configurations (§ 8.2); the wider table of seven architectures corroborates without independently repeating the same test.
The primary evidence is the label-permutation bootstrap (§ 6.3), the SNR stratification (§ 6.4), and the inclination stratification (§ 6.7), all pre-specified or run against fixed, pre-existing checkpoints; the perturbation trace (§ 6.6) is corroborating mechanistic detail, not primary evidence, since its closure rests on a channel adopted after its pre-registered instrument failed calibration — the degeneracy verdict does not depend on it.
Because that channel was adopted post hoc, the trace cannot independently close any confound on its own; it corroborates a verdict established elsewhere, pending the fresh-holdout replication that would let A.3's closure stand on its own pre-registered footing.
The degeneracy is effectively exact for this population as a point-estimation problem.
The value of the chapter lies as much in the demonstrated discipline — mechanism before metrics, code-validated controls, pre-registered criteria inside the engineering loop — as in the null it certifies, and both carry directly into the inclination-conditioned and posterior-based formulations that follow.

---

## Appendix: Claim-to-artifact map

All paths relative to `experiments/phic_psi_poc/`.

| Claim / table / figure | Artifact |
|---|---|
| Chronological record (Runs 1–9b) | `diagnostic_log.md`, `NOTES.md`, `experiment_summary_2026-07-22.md` |
| Analytic prerequisite sweep (§3, Fig. 3.1) | `sweep_1_1_ratio_vs_iota.{csv,png}`, `results.md`, `prereq_checks.py` |
| Label-distribution audit (Fig. 5.1) | `diagnostic_output/true_label_distributions.png`, `true_label_stats.csv` |
| tanh postmortem (§5.3–5.4) | `tanh_to_linear_postmortem.md`, `diagnostic_output/diagnostic_checks_20260718_*.log` |
| Inclination loss-path correction (§5.4) | `inclination_loss_trace.md` |
| Run 7 trajectories (Figs. 5.2, 5.3) | `diagnostic_output/{combo_loss,logvar}_trajectories.png`, `diagnostic_checks_20260721_000331.log` |
| Positive controls (Fig. 5.4, §5.6) | `analysis_output/` CSVs, scatter PNGs, `analysis_report_20260720_234304.md` |
| Table 6.1 | `pre_post_comparison.csv` |
| std_ratio correction (§6.2 A.2) | `std_ratio_trajectories.md` |
| poc_b / cnn_attention config diffs (§6.2 B, C) | `poc_b_config_diff.md`, `cnn_attention_config_diff.md` |
| Bootstrap (Table 6.3) | `bootstrap_output/bootstrap_ang_mae_20260721_093533.md` |
| SNR stratification (§6.4) | `snr_output/snr_stratification_20260721_094039.md` |
| Run 8 ablation (Table 6.4, Fig. 6.2) | `lam0_ablation_output/`, `assessment_lam0_ablation_2026-07-22.md` |
| Perturbation trace (Table 6.5, § 6.6) | `perturbation_trace_standalone.py`, `perturbation_trace_output/` |
| Inclination stratification (Table 6.6, § 6.7) | `inclination_stratification.py`, `inclination_output/inclination_stratification_20260723_130630.{log,md}` |
| Certified-model memorization gap (Fig. 8.1, § 8.2) | `plot_certified_memorization.py`, `diagnostic_output/certified_models_train_val_loss.png`, source `runs/phic_psi_{poc_b,cnn_attention}/20260720_*/history.csv` |
| Scalar-control inclination stratification (Table 6.7, § 6.7) | `inclination_control_stratification.py`, `inclination_output/inclination_control_stratification_20260723_140016.{log,md}` |
| Pre-registration (§7.1) | `preregistration_lam_retune.md` |
| Runs 9a/9b (Table 7.1, Figs. 7.1–7.2) | `lam005_retune_output/`, `lam010_retune_output/` |
| Verification plan / superseded rebuttal | `run7_verification_plan.md`, `run7_verification_rebuttal.md` |

## Suggested citation keys (to resolve against thesis bibliography)

`abbott2016gw150914` (first detection), `cutler1994gw` (waveform/PE foundations), `sathyaprakash2009physics` (GW physics review), `veitch2015lalinference` (Bayesian PE), `george2018deep` (first DL for GW), `gabbard2022vitamin`, `dax2021dingo` (neural posterior estimation), `kendall2018multi` (uncertainty weighting), `mardia2000circular` (circular statistics), `fisher1953dispersion` (vMF), `bai2018tcn`, `he2016resnet`, `fawaz2019inceptiontime`, `vaswani2017attention` (architectures), `nosek2018preregistration` (pre-registration).
