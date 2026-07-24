# Adversarial Review: "On the Recoverability of Coalescence Phase and Polarization Angle from Gravitational-Wave Strain"

## Executive Assessment

**Overall Verdict:** This is a methodologically rigorous chapter that convincingly demonstrates a specific null result *under narrow conditions*. However, the claim is significantly over-stated in ways that the author's own evidence undermines. The chapter's primary contribution is less the null result itself (which is physically expected) and more the diagnostic methodology—but the author frames it as a substantive physics discovery while simultaneously hedging with seven caveats that collectively undermine the generality of the conclusion.

**Score (as thesis chapter):** Strong B+ with potential for A- if claims are recalibrated downward and the methodological contributions are foregrounded over the physics claims.

---

## Major Substantive Critiques

### 1. The Central Claim is Trivial Relative to Its Framing

**The Problem:** The chapter frames the null result as a surprising finding requiring explanation, but the physical argument in Section 1 already provides the theoretical expectation: in the face-on limit, φ_c and ψ are exactly degenerate. For a population dominated by moderate inclinations, the degeneracy is *partially* broken but remains severe. That a point-estimation network fails to recover individually identifiable estimates is precisely what Bayesian samplers also find—they recover multimodal joint posteriors, not point estimates.

**The Evidence Against the Claim's Novelty:**
- Section 3 finds the combination preference ratio is only ~1.16× population-averaged—this is *tiny*.
- The curriculum design explicitly assigns zero weight to the well-constrained combination at face-on, exactly where the signal is strongest.
- For a point estimator, the optimal strategy *is* the constant predictor—the network is doing exactly what statistical decision theory predicts.

**The Over-Statement:** The conclusion says the degeneracy is "effectively exact for this population"—but "effectively exact" means something different from "the signal-to-noise ratio is too low and the inclination distribution too face-on for a point estimator to exploit." The evidence supports the latter; the former is a philosophical claim about identifiability that the network architecture and training regime cannot establish.

**Suggested Recalibration:** The abstract should state: "We find that for a two-detector, SNR 7-15 population under circular loss, point-estimation networks cannot recover φ_c or ψ individually. This is consistent with the known degeneracy structure and suggests that posterior estimation or inclination-conditioned architectures are necessary for these parameters."

---

### 2. The Architecture Pool is Insufficient for the Claim

**The Claim:** The author states "the null is a statement about the data, not about the hypothesis class" (Section 8.2) and that the architecture pool is diverse enough to establish this.

**The Counter-Argument:**
- Five trunks is not "diverse" in any meaningful sense—they're all one-dimensional temporal convolutions or attention variants. None are:
  - Frequency-domain architectures (which might exploit the phase structure differently)
  - Complex-valued networks (which natively represent phase)
  - Hybrid time-frequency architectures
  - Architectures specifically designed for periodic target regression
  - Architectures with explicit physical inductive biases (e.g., waveform model layers)
- The author notes that higher-capacity trunks memorize training data (training loss drops to 0.49-0.60 while validation stays at 1.0). This is evidence of *capacity to memorize*, not evidence that *no architecture could learn*. An architecture with appropriate inductive bias might generalize differently.
- The "attention" trunk (cnn_attention) performs *worst* on the periodic heads but best on sky localization—this suggests architecture choice matters for what is learned, undermining the claim that all architectures would fail.

**The Counter-Evidence:** The author's own Table 6.1 shows variation across architectures (φ_c ranges 1.572-1.580; ψ ranges 0.780-0.793). While all are near the null, the variation is real and systematic, suggesting the network *does* have some sensitivity to the angles but cannot convert it into accurate predictions.

**Suggested Recalibration:** The architecture claim should be narrowed: "Across five temporal-convolutional architectures, none recovered the angles." Not "across architectures generally."

---

### 3. The Bootstrap Test is Underpowered for the Claim

**The Method:** The author uses a label-permutation bootstrap (10,000 shuffles) and finds no significant p-values for φ_c/ψ.

**The Problem:** The test asks: "Is the model's performance significantly different from random guessing?" This is the wrong null hypothesis. The relevant question is: "Is there any information about φ_c/ψ in the strain that the network could exploit?" The permutation test is a **necessary but not sufficient** condition—a model could perform better than random guessing and still be practically useless (e.g., recovering angles to ±30°).

**The Evidence:** The author's own SNR stratification (Section 6.4) finds a monotonic improvement for tcn/φ_c (1.633 → 1.543), a 0.09 rad improvement. This is *larger than the 0.038 rad artifact scale* the author established. The author dismisses it as "beneath the 0.038-rad artifact scale" (it's not; it's 2.4× larger) and because the model's std_ratio was unstable. But:

1. The std_ratio instability is the author's own criterion, not a statistical argument
2. A monotonic SNR trend *is* evidence of information extraction, however weak
3. The author establishes artifact bounds (0.038 rad) and then selectively applies them

**The Inconsistency:** The author states "no SNR-dependent improvement anywhere" (Section 8.1) but their own data show otherwise—tcn/φ_c improves monotonically from low to high SNR. The author's explanation (std_ratio instability) is a *methodological* criticism, not a refutation of the signal.

**Suggested Recalibration:** The claim should be: "Any information present is too weak to yield practically useful estimates given the current architecture and training regime." The absolute "no information" claim is not supported.

---

### 4. The "Certified Clean" Model Restriction is Suspect

**The Sequence:**
- Seven architectures trained → four with λ-matched → two "certified clean" (poc_b, cnn_attention)
- The two clean models are the ones with the *least* favorable results
- The excluded models show hints of information extraction (tcn/φ_c SNR trend)
- The author then rests the primary null on the two clean models

**The Problem:** This is a post-hoc selection that discards the only evidence that might challenge the null. The criterion for "clean" (std_ratio in [0.5, 2.0] with stable trend) is the author's *own diagnostic criterion*—not a physical condition. The tcn/φ_c model with std_ratio=0.34 at epoch 79 is "uninterpretable" by the author's diagnostic definition, but the evidence of SNR-dependent improvement is real.

**The Counter-Argument:** The author argues that "the evidence localizes the failure in the information available to the network" (Section 8.2). But the tcn/φ_c SNR trend *is* evidence of information extraction—it's precisely the pattern one would expect if weak information is present. That the author's diagnostic criterion flags the model as "uninterpretable" means either:
1. The criterion is too strict, or
2. The criterion masks a real signal

**Suggested Recalibration:** The null should be stated as: "Of the models that passed our interpretability criteria, none showed practically useful angle recovery. However, one model with marginal interpretability showed a weak SNR-dependent trend, suggesting the null may not hold for all architectures."

---

### 5. The Perturbation Trace Closure is Post-Hoc

**The Sequence (Section 6.6):**
- Verification item A.3 left an open probe: an 89× prediction-sensitivity asymmetry
- A multi-step trace was designed with a pre-stated decision tree
- The calibration criterion failed (mchirp, the positive control, never read directional)
- The classifier was retired; the closure was re-founded on a "surviving channel"
- The author notes this is "transparently post hoc" and records the caveat

**The Problem:** This is the chapter's mechanism-level evidence, and it's now based on a test that:
1. Wasn't the pre-registered test (since the classifier was retired)
2. Had its criterion adjusted on review
3. Only works if the reader accepts the post-hoc re-foundation

**The Honesty:** To the author's credit, this is documented. But the conclusion "A.3 is closed" is over-stated. A more honest framing would be: "The original instrument was retired when its calibration failed. A secondary, internally-controlled instrument (the paired probe-loss statistic) shows no angle recovery while detecting chirp-mass learning. This supports the null but is not a pre-registered result."

**The Epistemic Status:** Post-hoc re-foundation is a form of researcher degrees of freedom. While the author documents it, the claim that this "closes" the probe is not justified—the probe is now in the same category as the pre-registered λ-sweep: it *would* have been a strong result if it had been pre-specified, but since it wasn't, it's a post-hoc observation.

**Suggested Recalibration:** Acknowledge that the perturbation trace is a post-hoc exploratory analysis that supports the null but does not constitute a pre-specified test. The primary evidence remains the bootstrap tests and SNR stratification.

---

### 6. The Pre-Registered λ Retune Claims Exhaustion

**The Claim (Section 7.3):** "The λ-sweep {0, 0.01, 0.05, 0.10} is exhausted; any further pursuit is architecture-level."

**The Problem:** The author notes "the response to λ is peaked, not monotonic—λ=0.05 produced near-misses on both targets while λ=0.10 regressed." This means there is a peak somewhere between 0.05 and 0.10 (or between 0.01 and 0.05, given λ=0.05 was the best). The author concludes the sweep is exhausted because it failed the pre-registered gates, but:

1. The gates are binary (pass/fail), not continuous
2. The near-miss at λ=0.05 suggests a finer sweep might yield a pass
3. The author acknowledges this in Section 8.5 ("a finer freshly pre-registered mini-sweep ... plausibly contains a stability optimum")

**The Inconsistency:** The author claims the sweep is "exhausted" while simultaneously acknowledging that the λ dimension is not exhausted and suggesting a finer future sweep. This is a contradiction—"exhausted" means there's nothing more to try; "suggesting a finer sweep" means there is.

**The Reality:** The pre-registered sweep's stopping rule was: try 0.05, if fail try 0.10, if fail report "λ alone insufficient." This is what the author did. But the *substantive* conclusion that the λ dimension is exhausted is false—the author's own data suggest a peak at some intermediate value.

**Suggested Recalibration:** State: "The pre-registered sweep failed to stabilize either primary target. The peaked response suggests a finer sweep could succeed, but this was not pursued and is left as future work."

---

### 7. The Positive Controls Are Misleading

**The Evidence:** The chirp mass, merger time, SNR, and sky position are recovered well (R² 0.91-0.96 for chirp mass).

**The Problem:** The positive controls are evidence that the *pipeline works*, but they don't establish that the *architecture is appropriate for the targets*. Chirp mass and SNR are amplitude/phase-evolution parameters that are trivially extracted from the time-frequency structure. φ_c and ψ are phase *offsets* that require precise timing of the waveform's starting phase. These are fundamentally different inference problems:

- Chirp mass: determined by the frequency evolution over the entire waveform (~100 cycles of phase)
- φ_c: determined by the phase at a single instant (the merger)
- ψ: determined by the relative amplitude in the two polarization states

The positive controls demonstrate that the network can extract *integrated* properties of the signal, not that it can extract *instantaneous* phase information. The failure of the periodic heads could be due to the network not *representing* phase information at sufficient temporal precision.

**The Counter-Evidence:** The sky-position head, which depends on inter-detector timing and antenna patterns, is recovered well (mean 3.3°). This suggests the network *does* represent phase information (since sky position depends on the strain phase across detectors). So why doesn't it recover φ_c? One possibility is that the network is learning phase *differences* between detectors (which determine sky position) but not the absolute phase (which determines φ_c). This is a representational issue, not a data-information issue.

**The Implication:** The null result may be an architecture representation failure, not a physics claim about the information content of the data.

**Suggested Recalibration:** The positive controls establish that the network can learn amplitude and integrated phase properties, not that it can learn instantaneous phase offsets. The failure may be representational.

---

### 8. The Claim About Sky Position is Over-Stated

**The Evidence:** Sky position recovery: mean angular error 3.3°-10.0° across models.

**The Problem:** The author uses this as a positive control, but sky position is also subject to degeneracies. For a two-detector network, the sky position is determined by:
- Arrival time difference (determines declination/right ascension partially)
- Amplitude ratio in the two detectors (determines the rest)

Both of these are *coarse-grained* properties of the signal—they don't require precise phase extraction. The fact that sky position is recovered well doesn't establish that the network can extract phase to the precision needed for φ_c/ψ.

**The Implication:** The positive controls are not as strong as the author claims.

---

### 9. Missing: Sensitivity Analysis on Inclination

**The Omission:** The degeneracy is strongest at face-on (cos ι ≈ ±1) and weakest at edge-on. The author's analytic study shows this. But the network training and evaluation are on a population with a broad inclination distribution (including 28.7% near-face-on). The null result could be driven entirely by face-on systems.

**The Missing Analysis:** What happens if the network is evaluated *only* on edge-on systems (|cos ι| < 0.5)? The author's SNR stratification (Section 6.4) shows a weak φ_c improvement for tcn, but there's no inclination-stratified analysis. The author has the inclination labels (they're in the dataset) but doesn't report results by inclination tercile.

**The Problem:** This is a major gap. If the network recovers φ_c and ψ for edge-on systems but not face-on ones, that would *confirm* the physical degeneracy explanation. But the author doesn't test this.

**The Counter-Evidence:** The poc_b model's collapse (circ_r=0.989) is attributed to the curriculum down-weighting face-on combinations. But if edge-on systems have information, the model should recover them—it doesn't. However, the author doesn't report the results separately for edge-on systems, so we don't know if there's a subset with non-null performance.

**Suggested Addition:** Add an inclination-stratified analysis analogous to the SNR stratification. Report ang_MAE for face-on, edge-on, and intermediate systems separately.

---

### 10. Missing: Bayesian Posterior Comparison

**The Omission:** The author notes in Section 8.1 that "a sampler still recovers a meaningful joint posterior over (φ_c, ψ)" but doesn't compare to the network's predictions. This is a missed opportunity for a direct comparison:

- Does the network's posterior (or point estimate) fall within the Bayesian credible interval?
- Does the network's marginal distribution over (φ_c, ψ) match the true posterior?
- Does the network capture the degeneracy structure (the well-constrained combination)?

**The Value:** A comparison to Bayesian inference would establish whether the network's failure is a point-estimation artifact or a genuine inability to capture the posterior structure. The author has the necessary tools (the data generation process could be used to produce posterior samples) but doesn't do this analysis.

**The Significance:** If the network also fails to capture the joint posterior structure, that's a stronger null. If it captures the joint posterior but fails on point estimates, that's a point-estimation limitation, not a data-information limitation.

**Suggested Addition:** Add a brief comparison to Bayesian posterior recovery (even an approximate one). Show the network's (φ_c, ψ) distribution and compare to the true distribution.

---

## Minor Critiques and Presentation Issues

### 11. Over-Loaded Terminology

The chapter uses terms like "well-constrained" and "poorly-constrained" interchangeably with the analytic study's "better-correlated combination" and "worse-correlated combination." This blurs the distinction between:
- The *analytic degeneracy* (which is about the data generation process)
- The *network's ability to learn* (which is about the optimization problem)
- The *practical utility* (which is about the accuracy of estimates)

**Suggestion:** Use distinct terms for each concept.

### 12. The Table Formatting is Self-Serving

Table 6.1 combines pre-fix and post-fix results, with dashes for missing runs. This makes the table look like a comprehensive comparison when it isn't—the pre-fix runs are from a broken pipeline and shouldn't be in the same table.

**Suggestion:** Separate pre-fix and post-fix results into different tables or clearly mark them.

### 13. The Claim-to-Artifact Map is Over-Wrought

The appendix is useful but includes 21 separate artifact references. This suggests the investigation was chaotic (which is acknowledged in the narrative) but also that the chapter is trying to pre-empt criticism by showing how thoroughly everything was documented.

**Suggestion:** Focus on the key artifacts and references; the rest can be in the supplementary materials.

### 14. The Narrative is Too Long

The chapter is ~8,000 words of text plus tables/figures. The narrative of the "diagnostic arc" is interesting but takes up ~3,000 words. A more concise version would be stronger.

**Suggestion:** Move the full diagnostic narrative to a supplement and summarize it in the main text.

### 15. The Figures Are Not Referenced Properly

The chapter references figures but the text doesn't always say what they show. For example, Figure 5.4 is referenced as showing chirp-mass recovery but the text says "Figure 5.4 — Chirp-mass recovery on the validation set" without describing the plot.

**Suggestion:** Add one-sentence captions that describe what the figure shows.

---

## Summary of Major Issues

| Issue | Severity | Suggested Fix |
|-------|----------|---------------|
| Claim is trivial relative to framing | High | Recalibrate claim to be about point-estimation under specific conditions, not "effectively exact" degeneracy |
| Architecture pool is insufficient for broad claim | High | Narrow claim to temporal-convolutional architectures; acknowledge representational limitations |
| Bootstrap test is underpowered | Medium | Acknowledge that "no significant difference" is not "no information"; show effect sizes |
| "Certified clean" restriction excludes evidence | High | Report results for all models; note that excluded models show weak SNR trends |
| Perturbation trace closure is post-hoc | Medium | Acknowledge exploratory nature of post-hoc analysis |
| λ-sweep doesn't exhaust the λ dimension | Medium | Acknowledge peaked response suggests finer sweep could succeed |
| Positive controls are mismatched | Medium | Note that chirp mass and SNR are integrated properties, not instantaneous phase |
| Missing inclination stratification | High | Add inclination-tercile analysis |
| Missing Bayesian posterior comparison | High | Add comparison to Bayesian posterior recovery |
| Over-loaded terminology | Low | Use distinct terms for distinct concepts |

---

## Verdict and Recommendations

**For Acceptance (with Major Revisions):**

The chapter has substantial methodological merit and the null result is credible *within its carefully defined scope*. However, the author over-states the generality of the conclusion and selectively excludes evidence that might challenge it. The following revisions are necessary:

1. **Recalibrate the central claim** from "φ_c and ψ are unrecoverable" to "φ_c and ψ are not recoverable by point-estimation networks on this SNR population, consistent with the known degeneracy structure and suggesting that posterior estimation or inclination conditioning is necessary."

2. **Add inclination-stratified analysis** to test whether edge-on systems show information recovery.

3. **Add Bayesian posterior comparison** to establish whether the failure is specific to point estimation or is a more general inability to capture the structure.

4. **Acknowledge the limitations of the architecture pool** and frame the claim as conditional on temporal-convolutional architectures.

5. **Report results for all models** in the main text, not just the "certified clean" ones. Acknowledge the weak SNR trends in the excluded models.

6. **Recalibrate the λ-sweep conclusion** from "exhausted" to "not resolved; finer sweep needed."

7. **Foreground the methodological contribution** over the physics claim—the diagnostic methodology is the chapter's strongest contribution.

**For the Thesis as a Whole:**

This chapter is a case study in how to do a null result well (diagnostic discipline, pre-registration, mechanism inspection) and how *not* to over-claim the result. The methodological lessons (Section 8.4) are valuable and should be highlighted. The physics claim needs to be toned down significantly.

**Bottom Line:** The chapter is publishable with major revisions. Without revisions, the over-claiming would be a significant weakness in the thesis defense.
