This is a methodologically serious chapter with a strong narrative arc, but its central claim—a defended null on φ_c and ψ recoverability—rests on a narrower evidential base than the prose admits, and several control structures are shakier than presented. Below is an adversarial review structured by severity.

---

### 1. The Inclination Head Failure Undermines the Control Architecture

The strongest evidence for a *physical* null (as opposed to an engineering failure) is the positive-control structure: the same trunk learns chirp mass, merger time, SNR, and sky position, so the pipeline works. **But the inclination head also fails** (Table 6.1: ι ang_MAE ≈ 1.54–1.59 vs. null 1.571), and the chapter concedes its mechanism is “unresolved” (§5.4, §8.3).

This is not a minor footnote. Inclination is an angular parameter that *is* physically encoded in the strain amplitude through the antenna-pattern mixing of h₊ and h×. If the network cannot learn ι, the most natural explanation is not “physics makes ι unlearnable” but that **the circular-regression framework or head design has a systematic blind spot for angular targets**—one that happens to afflict φ_c and ψ as well. The chapter’s retraction of the inclination head as a “valid control” (§5.4) is honest, but honesty does not restore its epistemic function. Without a certified-learned angular control, the argument that “the trunk works, therefore the periodic heads fail because of degeneracy” commits the fallacy of selective controls. A skeptical reader is left wondering whether the entire angular head family (circular loss, two-vector encoding, `normalize_unit`) is simply mismatched to the signal structure, especially given that the scalar heads use different loss mechanics (Huber, vMF).

**Recommendation:** Either fix the inclination head and show it learns (which would rehabilitate the control structure), or downgrade the claim from “the degeneracy is exact” to “the angular head family fails on all periodic targets tested, for reasons that may include degeneracy.”

---

### 2. The “Certified Null” Rests on Only Two Models

The chapter repeatedly invokes “seven architectures, two head parameterizations, four λ values” (§1, §8.2), but the **verified, interpretable null** is explicitly narrowed to *two* model–head pairs: **poc_b and cnn_attention** (§6.2, §7, §8.3). The other two primary candidates (tcn/φ_c and poc_a/ψ) are filed as “uninterpretable” due to std_ratio pathology; the remaining three trunks are λ-mismatched and play only a corroborating role.

This is a severe shrinkage of the evidentiary base. A null result is only as strong as the diversity of attempts to break it, and here the attempt is essentially: one TCN variant with combination heads, and one attention CNN with baseline heads. The defense—that “the memorization gap shows capacity was not the limit” (§8.2)—is clever but circular in this context: the memorization gap is observed across the pool, but the *certified* healthy-gradient, healthy-magnitude, pre-registered evidence comes from two points in configuration space. A reviewer will not accept “architecture diversity” as a defense when the pre-registered gates themselves disqualify half the primary models.

**Recommendation:** Be explicit in the abstract and conclusion that the defended null is certified for **two** healthy configurations, with the remainder providing consistency checks but not independent verification. Do not let “seven architectures” appear in the same sentence as “verified null” without this qualification.

---

### 3. The Combination-Space Collapse May Be an Engineering Artifact, Not Physics

The poc_b model (SumDiffTrainer) is the most theoretically motivated configuration: it directly parameterizes the well-constrained combination and is weighted by the analytically derived curriculum. It collapses *more* severely than the baseline (circ_r = 0.989 vs. 0.848). The chapter interprets this as supporting evidence: “the design would have exploited a breakable degeneracy; its collapse into the constant predictor is what that design does when the degeneracy is not breakable” (§6.2).

This is post-hoc reasoning. An equally valid interpretation is that **the complex-multiplication head and the sin²ι curriculum create a pathological optimization landscape**. The curriculum gives zero weight to the poorly constrained combination at face-on, which means the model sees a single loss surface with a narrow valley; combined with the coupling between φ_c and ψ through complex multiplication, this is a recipe for mode collapse regardless of whether the combination carries signal. The baseline mode, for all its naivety, at least decouples the two heads. That the coupled, curriculum-regularized model collapses *more* is not evidence that the degeneracy is unbreakable; it is evidence that the poc_b head design is fragile.

Moreover, the analytic prerequisite (§3) shows a conditioning ratio of only **1.16× population-averaged** (1.56× at best, face-on). This is a weak signal. The chapter correctly notes this is “modest,” but then seems surprised that a point estimator cannot extract it. A skeptical reader is not surprised at all: a 1.16× correlation advantage is below the sample-noise floor for N=25,000 at SNR 7–15. The analytic study actually predicts failure; it does not motivate the combination head as a viable target.

**Recommendation:** Treat poc_b as a *sensitivity test* that failed, not as supporting evidence for the null. Acknowledge that the curriculum and complex multiplication may introduce optimization artifacts that are inseparable from the physical degeneracy in this design.

---

### 4. The SNR Regime Is Too Restricted for a General Null

All injections have network SNR uniform in **[7, 15]** (§4.1). This is a low-to-moderate SNR regime. The chapter frames its result as conditional—“for this population, in this architecture and loss family” (§2.1)—but the conclusion (§9) escalates to “the degeneracy is effectively exact for this population as a point-estimation problem,” and §8.1 says the claim is “not to louder signals” as if that is a minor caveat.

For gravitational-wave parameter estimation, SNR is the dominant control on parameter resolution. At SNR ~25–30 (e.g., GW150914-like), the phase evolution is measured with vastly higher precision, and the antenna-pattern amplitude differences between detectors become much more informative for ι and ψ. The chapter’s analytic study (§3) does not include an SNR sweep; the stratification (§6.4) shows no monotonic trend, but the *range* is only 7–15. A null over [7,15] does not license a null over [7,50].

**Recommendation:** Either (a) explicitly scope the conclusion to “low-to-moderate SNR,” or (b) add a high-SNR spot-check (even a small validation set at SNR 20–30) to test whether the degeneracy tightens. Absent (b), the chapter risks publishing a result that is immediately suspected to be an SNR-floor effect.

---

### 5. The A.3 Perturbation Trace Closure Is Methodologically Contaminated

§6.6 describes a sophisticated perturbation trace intended to close the “89× sensitivity asymmetry” loose end. The pre-registered displacement-geometry classifier (net/sum ratio) **failed its calibration**—it labeled a rapidly learning mchirp head as “noise-like.” The chapter’s response is to retire the classifier and rely instead on the **paired probe-loss statistic**, which was “added on review rather than pre-registered” (§6.6, §8.3).

This is a serious procedural problem. The pre-registered decision tree was: if the classifier fails calibration, do not use it. The chapter follows this rule. But the replacement channel—the paired statistic—was not part of the pre-registered protocol. Using it to close A.3 is **post-hoc instrument selection**, exactly the kind of flexibility that pre-registration is meant to prevent. The chapter is transparent about this (§8.3: “the channel choice is post hoc”), but transparency does not validate the choice. A reviewer will ask: if the paired statistic had also failed its control, would another channel have been promoted? The fact that the paired statistic “passed” (early mchirp |t| = 3.4–8.5) is comforting, but the decision to trust it was made after seeing that it passed.

**Recommendation:** Either (a) relegate the A.3 closure to “suggestive but not definitive” and let the null rest on the bootstrap + SNR stratification alone, or (b) pre-register a replication of the paired-statistic trace with a fresh holdout set and a locked decision rule.

---

### 6. The Bootstrap “Significance” Framework Is Underpowered for Tiny Effects

The label-permutation bootstrap (§6.3, Table 6.3) is well-executed, but its interpretation is strained. With N = 5,000 validation samples and 10,000 permutations, the test is precise, but the **minimum detectable effect** is bounded by the metric artifact scale (~0.038 rad). The chapter sets a pre-registered effect-size floor of 0.10 rad (§7.1), which is sensible, but then treats non-significance at the 0.025 level as evidence of nullity.

In high-dimensional null-result settings, non-significance is not evidence of absence unless power is reported. The chapter does not report power calculations. With ang_MAE noise dominated by the uniform target distribution, the standard error of the mean is ~0.02 rad; detecting a 0.10 rad effect would require ~16 samples at α=0.05, so N=5,000 is massively overpowered for that threshold. But the observed effects are near zero with tight CIs—this is genuinely strong evidence. The chapter should say so explicitly: the CIs exclude effects >0.05 rad, not just “p > 0.025.”

More troubling is the handling of the **cnn_attention inclination “detection”** (p = 0.0007, Table 6.3). The chapter dismisses it because the effect is “uniform across SNR terciles” and below the Bonferroni threshold. But Bonferroni is conservative, and the p-value *does* survive it (0.0007 < 0.0042). The dismissal relies on an *additional* post-hoc criterion (SNR monotonicity) that was not part of the pre-registered battery. If SNR monotonicity is required for validity, it should have been stated before the test. A reviewer will flag this as cherry-picking.

**Recommendation:** Report 95% confidence intervals on all ang_MAE differences from null, not just p-values. For the inclination anomaly, either treat it as a genuine (if small) misfit of the null for that head, or apply a uniformly pre-registered dismissal criterion.

---

### 7. Frequency Domain and Data Augmentation Are Conspicuous Absences

The chapter notes that the claim is conditional on “raw” strain inputs (§8.1), and that frequency-domain representations are future work. But for **coalescence phase**, the frequency domain is the natural representation: φ_c appears as a phase offset in the Fourier transform, and the ψ–φ_c degeneracy manifests as a frequency-dependent mixing of plus and cross polarizations. A time-domain TCN must learn to implicitly Fourier-transform the signal to access this information. The fact that the TCN fails is not surprising if the inductive bias is wrong for the task.

Similarly, no data augmentation (time shifts, phase rotations, detector recoloring) is used. For periodic targets, augmentation is standard practice to prevent collapse to dataset biases. The chapter’s models *do* collapse to narrow modes (Fig. 6.1, circ_r up to 0.99)—exactly the behavior augmentation is designed to break.

**Recommendation:** Add at least one frequency-domain or time-frequency spot-check (e.g., Q-transform input to the same TCN trunk), and one augmentation ablation, to rule out representation bias. Without this, the null is vulnerable to the objection that the network simply cannot see the signal in the chosen basis.

---

### 8. The “Memorization” Interpretation of Training Loss Is Unverified

Higher-capacity models achieve training circular loss ≈ 0.49–0.60 while validation loss stays at ~1.0 (§5.5, Fig. 5.2). The chapter calls this “memorization of the training set’s phase labels.” But with 25,000 training samples and continuous circular targets, pure memorization is implausible. A more likely explanation is that the model learns **spurious correlations** between strain morphology and phase (e.g., specific mass-ratio–inclination–sky-position subpopulations that accidentally correlate with φ_c or ψ in this particular training draw). If so, the training-loss drop is not “memorization” but **overfitting to finite-sample fluctuations**, which regularization or more data might cure.

The distinction matters. “Memorization” implies the target is unlearnable in principle; “overfitting to spurious structure” implies the target is weakly learnable but the model lacks inductive bias or data. The chapter does not test this distinction—e.g., by training on 2× data and checking whether the training/validation gap shrinks.

**Recommendation:** Soften the “memorization” language to “training-set overfitting without generalization,” and note that this pattern is consistent with either an uninformative target or a target with SNR below the network’s effective sample complexity.

---

### 9. Minor but Sharp Issues

- **§8.1, Bayesian samplers:** The chapter says Bayesian samplers recover “a meaningful joint posterior over (φ_c, ψ).” If the degeneracy is exact, the joint posterior is a *ridge* (perfect correlation), not a tight posterior. The phrasing risks misleading readers into thinking samplers succeed where neural networks fail; they succeed only in mapping the degeneracy, not breaking it. Clarify.
- **§5.1, R² artifact:** The initial R² = 0.754 misreading is a powerful cautionary tale, but its inclusion also raises questions about the investigators’ initial statistical literacy. Frame it as a discovered pathology of circular metrics, not as a near-mistake.
- **§4.5, protocol heterogeneity:** The five trunks are not trained with identical protocols (different LR floors, warmup, λ values). The chapter notes this honestly, but a reviewer will ask why the λ-matched subset is not the *only* reported comparison. The unmatched trunks add noise, not diversity.
- **Figure captions:** Many figures are referenced by relative paths (`../diagnostic_output/...`). For a thesis chapter, these should be embedded or referenced by figure number, not file path.

---

### Summary Verdict

This chapter is **methodologically self-aware and unusually honest about its blind alleys**, which is its greatest strength. However, the adversarial reading reveals that the **“defended null” is defended well against optimization artifacts but poorly against scope limitations and control failures**. The inclination head failure removes the angular positive control. The certified evidence narrows to two models. The SNR ceiling is low. The combination-space result is confounded by its own curriculum. And the A.3 closure relies on a post-hoc statistical channel.

**If I were the external examiner or referee, I would not reject the chapter**, but I would demand:
1. A downgraded claim: “We find no evidence of recoverability under these conditions” rather than “The degeneracy is effectively exact.”
2. Explicit front-matter acknowledgment that the certified null rests on **two** configurations.
3. Either a fixed inclination control or its removal from the control narrative.
4. A high-SNR spot-check or a hard scope boundary at SNR ≤ 15.

The prose is clear, the diagnostic discipline is admirable, and the pre-registration is a genuine contribution to practice. But the physics claim is currently stronger than the statistics allow.
