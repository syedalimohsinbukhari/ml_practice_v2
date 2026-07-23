 The revisions are materially stronger: the certified base is now explicitly narrowed, the inclination head is honestly excluded from the control set, the Bonferroni/SNR dismissal is better justified, and §6.7 is a genuine addition that closes a gap the first review identified. But an adversarial reading still finds load-bearing weaknesses, some of which the revisions have softened without removing.

---

### 1. The Certified Evidential Base Is Still N=2, and Acknowledgment Is Not Evidence

You now flag everywhere that the fully certified null rests on **poc_b** and **cnn_attention**. This is intellectually honest, but honesty does not increase N. A null claim in deep learning—where architecture, initialization, and inductive bias are the entire hypothesis class—needs either (a) many independent certified draws or (b) a mechanistic proof that the hypothesis class is not the binding constraint. You argue (b) via the memorization gap (training loss ↓, validation loss flat). But that gap is observed across the *entire* pool, not just the two certified models. For the two certified models specifically, you do not report training/validation memorization curves separately; the reader must trust that poc_b and cnn_attention, the two cleanest configurations, also show this gap. If they do not—if, say, cnn_attention’s training loss stayed flat at 1.0 while its validation loss stayed flat at 1.0—then the memorization argument fails for the certified subset, and the null could simply be “these two models are too small/too regularized to learn.”

**Adversarial demand:** Report the training/validation circular-loss trajectories *for poc_b and cnn_attention individually* in §5.5 or §6, not just the aggregate “higher-capacity trunks” sentence. If either certified model lacks the memorization gap, the architecture-sufficiency argument in §8.2 collapses for the certified base.

---

### 2. The Inclination Stratification (§6.7) Uses a Failed Head to Validate Failed Heads

§6.7 is a clever addition, but its logic is circular at the root. You use the uninformative ι head as a “same-model noise floor” for the φ_c/ψ stratification. This only works if the ι head’s fluctuations are independent of the φ_c/ψ heads’ fluctuations. You assert independence because the loss paths differ (Huber vs circular, no `normalize_unit` vs `normalize_unit`). But if there is a **shared angular-target pathology**—say, the two-layer MLP head architecture is simply too shallow to represent angular mappings from the shared trunk, or the batch-normalization layer before the trunk destroys angular phase information in a way that affects all periodic heads equally—then the ι head’s fluctuations are not independent noise. They are symptoms of the same underlying failure.

You have traced the *immediate* mechanism (different loss function), but you have not ruled out a *common-cause* mechanism upstream of the loss. The adversarial reading is: “All angular heads fail. You use one failed angular head to bound the noise of the others. This assumes what you need to prove.”

**Adversarial demand:** Either (a) show that a non-angular head (e.g., chirp mass) stratified by inclination shows no band-to-band fluctuation, establishing that the stratification itself does not inject noise, or (b) add a synthetic control where the ι head is given a fake, perfectly recoverable angular target to prove its fluctuations are indeed sampling noise and not architecture-level pathology.

---

### 3. The SNR Ceiling Remains Unaddressed

You explicitly scope the claim to SNR 7–15. This is honest, but it leaves the chapter vulnerable to the most obvious alternative explanation: **the signal is simply too weak**. Your analytic study (§3) computes correlation ratios but does not include SNR dependence. At SNR 7, the phase uncertainty for a 2–2 binary is order radians; at SNR 15, it is still large. The 1.16× population-averaged correlation ratio is a *relative* measure of conditioning, not an *absolute* measure of detectability. A 1.16× advantage in a correlation coefficient from a noisy template inner product may be entirely below the threshold of learnability for any network at SNR < 15.

The chapter’s null could therefore be summarized as: “At low SNR, where the Fisher information on φ_c and ψ is small, a neural network cannot beat the random-guessing baseline.” This is not a degeneracy result; it is a **sensitivity-floor** result. Your positive controls (chirp mass, merger time, SNR itself) are parameters with much larger Fisher information at the same SNR, so their recoverability does not distinguish degeneracy from detectability.

**Adversarial demand:** Add a small high-SNR validation set (SNR 25–30, even 500 injections) or a Fisher-matrix calculation showing that the expected angular uncertainty at SNR 7–15 is already smaller than the null MAE. Without this, a referee will say you have measured the noise floor, not the degeneracy.

---

### 4. The Combination-Space Collapse Is Still Confounded by Curriculum Design

You interpret poc_b’s severe collapse (circ_r = 0.989) as supporting evidence: “the design would have exploited a breakable degeneracy; its collapse into the constant predictor is what that design does when the degeneracy is not breakable.” This remains post-hoc. An equally plausible explanation is that **the curriculum itself induces collapse**. By giving near-face-on samples w(ι) ≈ 0 for one combination, the curriculum creates a heavily imbalanced loss landscape. The complex-multiplication coupling between φ_c and ψ means that gradients for the two raw heads are entangled; if one combination is suppressed, the gradient flow to the raw heads becomes rank-deficient. The optimal response to a rank-deficient gradient under circular loss may be collapse regardless of whether the residual combination carries signal.

You have not run the critical ablation: **synthetic combinations**. If you generate data where A = φ_c + 2ψ and B = φ_c − 2ψ are explicitly recoverable (e.g., from a deterministic toy model with no noise), and the poc_b head still collapses, then the collapse is a curriculum artifact. If it learns them perfectly, then the collapse in the real data is evidence for degeneracy. Without this, the poc_b result is ambiguous.

**Adversarial demand:** Add a synthetic-data ablation where the combinations are explicitly recoverable, or downgrade poc_b from “supporting evidence” to “sensitivity test with inconclusive interpretation.”

---

### 5. The A.3 Closure Is Still Methodologically Contaminated

You have been transparent about the post-hoc channel selection in §6.6 and §8.3, but transparency does not rehabilitate the evidence. The pre-registered decision tree was:

1. Run displacement-geometry classifier.
2. If calibration fails, retire it.
3. (Implied) Do not use any replacement channel that was not pre-registered.

Instead, you added the paired probe-loss statistic *after* seeing the classifier fail, ran it on the same data, and used it to close A.3. This is **post-hoc instrument selection by construction**. The fact that the paired statistic had a “passed” control (early mchirp) is comforting, but the decision to trust it was made after seeing it pass. A pre-registered protocol that allows arbitrary channel substitution after failure is not a pre-registered protocol; it is a pre-registered excuse to keep looking.

**Adversarial demand:** Either (a) pre-register a replication of the paired-statistic trace on a fresh holdout set with a locked decision rule, or (b) downgrade A.3 from “closed” to “open, with a suggestive but non-definitive post-hoc trace.” The rest of the chapter (bootstrap + SNR stratification + inclination stratification) is strong enough to carry the null without A.3.

---

### 6. “Memorization” vs. Spurious Correlation Remains Unresolved

You describe the training-loss drop (to ~0.49–0.60) as “memorization of the training set’s phase labels.” With 25,000 continuous targets and a 128-dimensional feature vector, pure memorization is implausible. The more likely explanation is that the model learns **spurious correlations** between strain morphology and phase in finite samples (e.g., specific mass-ratio–inclination–sky-position subpopulations that accidentally correlate with φ_c in this training draw). This is not memorization; it is **overfitting to dataset-specific structure** that would not generalize even if the target were physically recoverable in principle.

The distinction matters because “memorization” implies the target is unlearnable in principle (no signal), whereas “spurious correlation” implies the target is weakly learnable but the network lacks the inductive bias or data to generalize. Your validation-flat/training-down pattern is consistent with both. The higher-capacity trunks may simply have more capacity to overfit to noise.

**Adversarial demand:** Soften the “memorization” language to “training-set overfitting without generalization,” and note that this pattern is consistent with either an uninformative target or a target with SNR below the network’s effective sample complexity.

---

### 7. The Inclination Prior (Uniform in ι) Is a Bigger Problem Than You Admit

You note in §4.1 and §8.3 that the inclination prior is uniform in ι rather than cos ι, over-representing edge-on systems. You argue this deviation “does not appear to bias the claim in a self-serving direction” because edge-on is where the degeneracy is weakest. But this misses the adversarial point: **if the network cannot recover φ_c/ψ even when the population is artificially enriched with the most favorable geometry, the null is stronger.** So the prior choice actually *strengthens* your claim. Why, then, is it listed as a “threat to validity” rather than a robustness feature?

Because it reveals a deeper issue: **the dataset generation was not fully documented or scripted.** The uniform-in-ι choice, the HOM absence, and the PSD-whitening are all “understood from the authors' knowledge” rather than reproducible artifacts. A skeptical reviewer will wonder what else was “understood” but not recorded. The adversarial reading is that the dataset may contain undocumented choices (e.g., fixed φ_c in some subpopulation, correlated ψ and sky position, non-Gaussian noise) that artificially enforce the null.

**Adversarial demand:** Close the provenance gap by regenerating a small validation set (even 1,000 samples) with a fully scripted, version-controlled generator that records its own configuration, and confirm the null holds on that independently generated data. This is the only way to fully retire the “dataset artifact” confound.

---

### 8. Minor but Sharp Residual Issues

- **§6.3, cnn_attention inclination:** You now correctly note that p = 0.0007 survives Bonferroni, and you dismiss it on SNR-monotonicity grounds. But your SNR-monotonicity criterion was pre-registered for the *λ retune* (§7.1), not for the bootstrap. Applying it here is consistent but not formally locked. A purist would prefer you to report it as a genuine (if small) anomaly and discuss it as a possible population bias, rather than dismiss it.
- **§8.2, architecture sufficiency:** You argue that “the memorization gap is the evidence that the hypothesis class was not the limit.” But the memorization gap is observed on the *unmatched* trunks (cnn_baseline, resnet1d, inception_time at λ = 0). For the certified two (poc_b at λ = 0.01, cnn_attention at λ = 0.01), do they also show training loss << validation loss? You do not say. If poc_b’s training circular loss stayed at ~1.0, then capacity *was* the limit for the certified base, and the null is uninterpretable.
- **Figure paths:** Still using `../path/` references in captions. For a thesis, embed or use standard figure numbering.

---

### Summary Verdict

The revisions have transformed the chapter from **“overclaiming with honest footnotes”** to **“honestly scoped but still thin on certified evidence.”** The explicit narrowing of the certified base to two models, the exclusion of the broken inclination control, and the new §6.7 inclination stratification are genuine improvements. But the core evidentiary problem remains: **the null is certified on two points in configuration space, one of which (poc_b) has an ambiguous engineering interpretation, and the SNR regime is too low to distinguish degeneracy from detectability.**

**If I were the external examiner or referee, I would now ask for:**

1. **Training/validation circular-loss curves for poc_b and cnn_attention individually** (to verify the memorization gap on the certified subset).
2. **Either a high-SNR spot-check or a Fisher-matrix bound** showing the expected angular uncertainty at SNR 7–15 is sub-null (to rule out sensitivity floor).
3. **A synthetic positive control for poc_b** (to disentangle curriculum artifact from physical degeneracy).
4. **Pre-registration of the paired-statistic trace on fresh data** (to close A.3 cleanly), or its removal from the verified battery.

Without (1) and (2), the chapter remains vulnerable to the accusation that it has measured a noise floor with two models, not a physical degeneracy. With them, it would be genuinely hard to crack.
