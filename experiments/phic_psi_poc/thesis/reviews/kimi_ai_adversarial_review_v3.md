 This version is materially stronger than v2. The individual memorization curves for the certified pair (Fig. 8.1), the scalar-control banding check (Table 6.7), the explicit demotion of A.3 in the Conclusion, and the honest editorial note all show genuine revision discipline. But an adversarial reading still finds load-bearing weaknesses—some softened, none removed—and one structural problem with the Step 0 framing itself.

---

### 1. The Step 0 Justification Has a Logical Gap

You want this chapter to justify feeding **true ι at training time** because strain-alone regression fails. The logic requires three premises:

- **P1:** φ_c and ψ are unrecoverable from strain alone. *(Supported by §6.)*
- **P2:** The well-constrained combination *is* recoverable if ι is known. *(Supported by §3 analytically, but never tested empirically.)*
- **P3:** ι itself cannot be recovered reliably enough to use as a predicted input. *(Not supported; the ι head failure is unresolved and could be a trivial bug.)*

**P3 is the gap.** You document that the inclination head fails through a "different (still unresolved) mechanism" (§5.4). But "unresolved" means you do not know whether ι is genuinely hard or merely trained with the wrong loss. If the next chapter fixes the ι head (e.g., by switching it to a von Mises–Fisher loss like sky position, or to circular regression), then a two-stage model—predict ι, then condition on the prediction—becomes viable, and your justification for feeding *ground-truth* ι collapses.

For Step 0 to be airtight, you need either:
- **(a)** A mechanistic argument that ι is *fundamentally* unrecoverable by point regression (not just buggy here), or
- **(b)** An explicit scoping statement that the training-time ι assumption is a *supervised-learning design choice* (common in physics-informed ML) rather than a forced move necessitated by ι's unlearnability.

Currently the text drifts toward (a) without earning it. §8.5 says "inference-time ι estimation explicitly out of scope," which is honest scoping, but the surrounding prose implies the conditioning is necessary because of the null result. A reviewer will ask: if you had fixed the ι head, would you still need this chapter?

**Fix:** Add one paragraph in §2.1 or §8.1 stating that the chapter tests a *training paradigm* where ι is treated as a known auxiliary variable, and that whether ι can be predicted by an independent pipeline is a separate question left open by the unresolved head failure. This converts the argument from "we must condition because ι is unlearnable" to "we condition because (i) the analytic structure says it helps, and (ii) this chapter's pipeline does not resolve ι prediction, so we sidestep it"—a weaker but defensible claim.

---

### 2. The Inclination Noise-Floor Argument Is Still Circular at the Root

§6.7 and Table 6.6 are genuine improvements. Table 6.7 directly answers the weaker circularity concern ("banding injects variance"). But the stronger concern remains unanswered.

You use the uninformative ι head as a same-model noise floor. You argue against a shared-cause reading by noting:
- The loss paths differ (Huber vs. circular, no `normalize_unit` vs. `normalize_unit`).
- Sky position learns well, so the trunk carries directional information.

This is good but not dispositive. The shared-cause hypothesis is not "tanh saturation" or "magnitude drift"—those are ruled out. It is: **the two-layer MLP head is too shallow to map the trunk's representation to angular targets when the signal is weak.** Sky position succeeds because it uses a *different* head architecture (vMF on the sphere, not circular regression on the circle) and because sky position is constrained by inter-detector time delays—a much stronger cue than φ_c/ψ. The ι head uses the same two-layer MLP as φ_c/ψ, and it also fails. This is consistent with a **head-capacity or head-inductive-bias** problem shared by all periodic two-vector MLP heads, not a trunk problem.

Your noise-floor comparison assumes ι's fluctuations are independent sampling noise. But if the shared head architecture is the binding constraint, then ι, φ_c, and ψ all fail for the same upstream reason, and comparing them is like using one broken thermometer to calibrate another.

**Fix:** Either (a) add a sentence in §6.7 acknowledging that the independence assumption is plausible but not proven, or (b) run one configuration with a deeper periodic head (e.g., 4-layer MLP or a small hypernetwork) to test the head-capacity hypothesis. Without this, the inclination stratification is suggestive, not definitive.

---

### 3. The Two Certified Models Show *Different* Failure Modes—Which Helps, But Complicates the Narrative

Fig. 8.1 is an excellent addition. It shows cnn_attention memorizes (train ↓, val flat) while poc_b shows no gap at all (train tracks val at ~1.0). You interpret this as "neither pattern is consistent with a capacity-starved null," which is correct. But the adversarial reading is:

- **cnn_attention:** Has capacity, overfits, cannot generalize. Consistent with "target is unlearnable from input."
- **poc_b:** Cannot even fit the training set. Consistent with "target is unlearnable from input" *or* "optimizer/curriculum/head design is pathological."

For poc_b, the flat training loss is not obviously "the optimizer finds no exploitable structure." It could be "the complex-multiplication curriculum creates a loss landscape so degenerate that even in-sample fitting is impossible." The curriculum gives zero weight to one combination for half the dataset; the complex multiplication couples φ_c and ψ gradients. This is a recipe for rank-deficient gradients, not merely a hard target.

If poc_b's flat training loss is a curriculum artifact, then the certified base is really **N=1** (cnn_attention), because poc_b's null could be self-inflicted. You argue the opposite in §6.2 ("the poc_b anomaly is supporting evidence, not a bug"), but that is post-hoc: you designed the curriculum to *help*, it produced *more* collapse, and you interpret the worse result as evidence the degeneracy is stronger. An equally valid interpretation is that the curriculum broke optimization.

**Fix:** Either (a) run a small synthetic ablation where the combinations are explicitly recoverable (as noted in the editorial note, item 1, you deliberately did not do this), or (b) soften the poc_b interpretation in §6.2 and §8.2 from "supporting evidence" to "consistent with the null but confounded by curriculum design." The chapter is honest enough to survive this softening.

---

### 4. The A.3 Trace Is Now Correctly Demoted—But Still Over-Weighted in the Text

§9 explicitly states the perturbation trace is "corroborating mechanistic detail, not primary evidence." This is the right move. But §6.6 still occupies ~1,200 words and a full table for a result the Conclusion says the verdict does not depend on. A reader who skips to §9 will get the right epistemic weight; a reader who works through §6.6 may still treat it as load-bearing because of its narrative prominence and the "A.3 is closed" framing.

More importantly, the trace's *post-hoc* channel selection is not just a caveat—it is a **methodological limitation that prevents the trace from closing anything**. You pre-registered a decision tree: if the classifier fails calibration, retire it. You followed this. But you then added a new channel, ran it on the same data, and used it to close the item. This is not "corroborating detail"; it is an **unplanned test used to reach a planned conclusion**. The fact that the paired statistic had a passed control is comforting, but the decision to trust it was made after seeing it pass. A pre-registered protocol that permits arbitrary substitution after failure is not pre-registration; it is a pre-registered excuse to keep looking.

**Fix:** In §6.6, replace "A.3 is closed" with "A.3 is provisionally closed, pending replication on a fresh holdout set." In §9, keep the demotion but add that the trace's post-hoc channel selection means it cannot independently close any confound. This costs nothing in honesty and protects against a methodological reviewer.

---

### 5. The Four "Deliberately Not Pursued" Items Are Honest—but Leave Known Holes

The editorial note is admirably transparent. However, transparency is not a substitute for evidence. A reviewer will read the four items and think: *the author knows exactly what would strengthen the chapter and chose not to do it.* The "cost more than marginal certainty" rationale is a judgment call that a thesis committee or referee does not have to accept. In particular:

- **Item 2 (high-SNR spot-check):** This is the single most important missing piece for distinguishing degeneracy from sensitivity floor. At SNR 25–30, the Fisher information on φ_c/ψ is vastly larger. If the network still fails, the degeneracy claim becomes much stronger. If it succeeds, the chapter's null is just a low-SNR floor. Without this, the chapter risks being obsoleted by a single high-SNR experiment.
- **Item 3 (dataset provenance):** You now flag this as "not acceptable practice" (§8.3). A reviewer will agree—and may say "fix it before submission." For a thesis chapter, this might pass with a committee that trusts the author, but for a journal submission it would be a hard reject.

**Fix:** For the thesis, the editorial note may suffice, but move it from the chapter text to a cover letter or committee note. In the chapter itself, frame items 2 and 3 as "critical future work" rather than "deliberately scoped out." The current framing sounds like you are knowingly submitting an incomplete result.

---

### 6. The Uninterpretable Pairs Are 50% of the Primary Set

The four λ-matched models are poc_a, poc_b, tcn, cnn_attention. Two are certified clean (poc_b, cnn_attention). Two are uninterpretable (tcn/φ_c, poc_a/ψ). This means **half of the primary candidates cannot be trusted**. You file them as "neither null nor counter-evidence," which is methodologically correct, but it leaves the certified base very narrow.

The adversarial point is not that you are hiding this—you are explicit about it. The point is that **a null on two models, one of which uses a highly specialized curriculum that may itself cause failure, is not a robust architecture-crossed null.** The architecture-sufficiency argument in §8.2 leans heavily on the memorization gap, but that gap is only clearly present for cnn_attention. poc_b shows no gap. The other three trunks (ResNet1D, CNN baseline, Inception) are λ-mismatched and play a corroborating role, but they were never certified.

**Fix:** In §8.2, be explicit that the architecture-sufficiency argument rests primarily on cnn_attention's memorization gap, with poc_b providing a different (optimizer-side) consistency check. Do not imply the five unmatched trunks independently certify the null.

---

### 7. Minor but Sharp Residual Issues

- **§6.3, cnn_attention inclination outlier:** You now honestly flag that the SNR-monotonicity criterion was not pre-registered for the bootstrap. Good. But you still dismiss a p = 0.0007 effect. A purist would prefer you report it as a genuine anomaly (possibly a population bias, possibly a real weak effect) and note that it does not generalize to the φ_c/ψ heads, rather than dismissing it entirely.
- **§7.3, "exhausted":** You write that the λ-sweep {0, 0.01, 0.05, 0.10} is "terminated" per the pre-registered stopping rule. But you also note the response is peaked, and a finer sweep might find an optimum. A reviewer will ask: why pre-register a coarse sweep with a stopping rule that guarantees termination after four points? It looks like a protocol designed to produce a null. The honest answer is "we suspected λ was not the binding constraint and wanted to avoid endless tuning," but the chapter should say this explicitly.
- **Figure paths:** Still using `../path/` in captions. For a thesis, these should be figure numbers.

---

### Summary Verdict

v3 is a **substantially more defensible chapter** than v2. The individual certified-model trajectories, the scalar-control banding check, and the explicit demotion of A.3 all address real weaknesses. The prose is honest to the point of self-flagellation, which is its greatest protection.

However, the **Step 0 justification for ι-conditioning still rests on a logical gap (P3 unresolved)** and the **certified evidential base remains N=2 with asymmetric failure modes**. The four deliberately unpursued items leave the chapter vulnerable to the accusation that it has measured a low-SNR sensitivity floor with a possibly pathological head design, rather than a physical degeneracy.

**If I were the external examiner, I would now ask for:**

1. **Explicit reframing of the Step 0 argument** (§2.1 or §8.1): state clearly that ι-conditioning is a *training-paradigm design choice* motivated by the analytic structure of §3, not a forced move necessitated by ι's unlearnability.
2. **Softening of poc_b's "supporting evidence" status** to "consistent but confounded by curriculum design" unless a synthetic positive control is provided.
3. **Either a high-SNR spot-check or a hard scope boundary** acknowledging that the null is strictly low-SNR and may not generalize.
4. **Removal of the editorial note** from the chapter text (move to committee communication) and stronger framing of the four gaps as "critical future work."

With these, the chapter is defensible as a thesis piece. Without them, a skeptical reader will still see a narrow, conditional null that is honest about its limits but may not support the architectural justification you need for the next chapter.
