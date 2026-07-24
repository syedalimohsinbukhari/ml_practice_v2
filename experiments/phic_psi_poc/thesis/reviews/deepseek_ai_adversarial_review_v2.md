# Additional Review: Chapter on φ_c/ψ Degeneracy (Revised Version)

## Overview

The revised chapter has substantially addressed the major critiques from the previous adversarial review. The addition of inclination stratification (Section 6.7), the calibrated language around the certified evidential base, and the acknowledgment of waveform-model provenance gaps represent meaningful improvements. The chapter is now **defensible** as a thesis chapter, though some residual issues remain.

---

## What Has Been Fixed (Acknowledged)

| Previous Critique | Response in Revised Version | Status |
|-------------------|----------------------------|--------|
| Missing inclination stratification | **Section 6.7** added: full edge-on analysis with ι-control noise floor | ✓ Addressed |
| Over-claiming generality | **Abstract & Section 8.1**: "point-estimating regression on this SNR-7–15, two-detector, dominant-mode population" | ✓ Addressed |
| Claim-to-artifact map reference | No change needed | ✓ |
| Architecture pool concerns | **Section 8.2**: strengthened defense with memorization gap argument | ✓ Partially addressed |
| Positive controls mismatched | **Section 5.6**: explicit exclusion of inclination from positive controls with code-level reason | ✓ Addressed |
| Waveform model provenance | **Section 8.3**: caveat added that HOM status is asserted but not recorded in metadata | ✓ Addressed |
| Inclination prior issue | **Section 8.3**: note added that ι is uniform rather than cos-weighted | ✓ Addressed |

---

## Remaining Concerns

### 1. The "Certified" vs "Corroborating" Distinction is Still Ambiguous

**The Text:**
> "The fully certified evidential base... is two λ-matched configurations, poc_b and cnn_attention; the remaining five corroborate under progressively relaxed certification, not as independent repeats of the same test."

**The Problem:** This creates a hierarchical evidence structure that is difficult to interpret. The reader is left wondering: Is the null result supported by two models, or by seven? The answer appears to be "two, with five weaker corroborations." But the chapter's main tables and figures present all seven as though they are co-equal evidence.

**Suggestion:** Be more explicit about what "certified" means in the context of the claim. If the claim is "no model recovered φ_c/ψ," then all seven are evidence. If the claim is "under clean optimization dynamics, no recovery occurs," then only the two certified models count. The distinction should be made earlier and more sharply.

**Recommended Revision:** Add a sentence in the Introduction: "The fully certified evidential base is two models. The remaining five provide corroborating evidence under conditions that are independently informative but do not repeat the same certified test."

---

### 2. Inclination Stratification: The ι-Control Noise Floor is Clever but Over-Interpreted

**The Logic (Section 6.7):**
- The ι head is known to be uninformative (§ 5.4)
- Its band-to-band swings establish a "noise floor" for what finite-band sampling can produce
- If φ_c/ψ deviations exceed this floor, they might be signal; if they don't, they're sampling noise

**The Problem:** This is a *calibration* argument, not a statistical one. The ι head is uninformative by construction, but it is also trained with a different loss (Huber vs circular). The noise floor derived from the ι head's fluctuations is a useful heuristic but not a formal statistical test. The author acknowledges this by noting tcn/φ_c exceeds its floor but in the "wrong" direction—but the logic here is slippery.

**The Substance:** The tcn/φ_c edge-on deviation is -0.0407 rad, which is *larger* than the tcn ι-noise floor (0.0225 rad). The author dismisses it because (a) it's in the "worse-than-null" direction and (b) tcn/φ_c is the one uninterpretable model. These are reasonable dismissals, but they're not statistical arguments—they're diagnostic arguments. The reader must accept the author's diagnostic framework to accept the dismissal.

**Suggestion:** Frame this less as a formal test and more as a diagnostic consistency check: "Applying the same diagnostic logic as § 6.4, the edge-on band shows no cross-model, edge-on-favoring trend. The single case that exceeds its model's noise floor does so in the opposite direction on a model flagged as unstable."

---

### 3. The λ-Sweep "Exhaustion" Claim Remains Self-Contradictory

**The Text (Section 7.3):**
> "the λ-sweep {0, 0.01, 0.05, 0.10} is exhausted"
> *immediately followed by*: "A finer, freshly pre-registered mini-sweep over λ ∈ [0.02, 0.08] plausibly contains a stability optimum"

**The Problem:** This is a direct contradiction. A sweep cannot be "exhausted" if a finer sweep might contain a solution. The author is using "exhausted" to mean "we stopped per our pre-registered rules," not "the λ dimension is exhausted." But the word "exhausted" implies the latter.

**Suggestion:** Change to: "Per the pre-registered stopping rule, the λ-sweep is terminated. The peaked response suggests a finer sweep could succeed, which is recorded as future work (§ 8.5)." This is honest and avoids the contradiction.

---

### 4. The Waveform-Provenance Gap is More Serious Than the Chapter Treats It

**The Text (Section 8.3):**
> "This is not preserved as metadata... it is recorded here from the authors' knowledge... If this recollection were wrong and higher-order modes were in fact present, the degeneracy would be only approximately exact even face-on, which would *sharpen* rather than weaken the null"

**The Problem:** The author argues the gap is harmless because if HOMs were present, the null would be *stronger* (the network failed even when the degeneracy was only approximate). This is logically valid but misses the point: the issue is **reproducibility**, not the direction of the effect. A reader cannot reproduce or verify the dataset generation. This is a threat to the chapter's scientific value, not just to the null's strength.

**The Recommended Action:** The author should either:
- (a) Regenerate the dataset with a script that records its own configuration (preferred), or
- (b) Explicitly state that the dataset is not reproducible from the information provided, and that this is a limitation of the study

The current framing ("it would sharpen the null") is true but beside the point.

**Suggestion:** Add: "We regard this as an unacceptable provenance gap and recommend that any follow-up work regenerate the dataset with scripted configuration recording before relying on this result."

---

### 5. The Perturbation Trace Closure Remains Post-Hoc

**The Text (Section 6.6):**
> "The disposition honors the pre-stated decision tree in order — the calibration criterion failed, so the classifier it governed is retired — and the closure is then re-founded, transparently post hoc, on the surviving channel"

**The Problem:** The author is honest about this, but the chapter still uses the perturbation trace as evidence in the conclusion. The reader is told the trace "closes" the probe, but only after a post-hoc re-foundation. This is a weaker form of evidence than the pre-specified tests.

**Suggestion:** In the Conclusion, frame the perturbation trace as a *supporting* but not *primary* piece of evidence. The primary evidence remains the bootstrap tests and SNR stratification. The trace is a mechanistic sanity check, not a confirmatory test.

---

## Minor Issues

### 6. Table 6.6 Caption is Unclear

The caption says "edge-on band vs null" and then shows values like "+0.0241." It's not immediately clear whether positive means "better than null" or "worse than null." The text explains it ("largest in the hoped-for direction"), but the caption should be self-contained.

**Suggestion:** Add to caption: "Positive Δ indicates performance better than random guessing (ang_MAE < null)."

---

### 7. The "Astrophysically Correct Prior" Note is Buried

**The Text (Section 8.3):**
> "the inclination prior is uniform in ι itself rather than in cos ι, the astrophysically correct isotropic choice, which over-represents edge-on systems relative to reality"

**The Problem:** This is a significant design limitation—the dataset is not astrophysically realistic in a key dimension. But it's buried in the "Threats to validity" section. It should be more prominent, possibly in Section 4.1 where the dataset is described.

**Suggestion:** Add a note in Section 4.1: "Note: the inclination prior is uniform in ι (0 to π), not in cos ι, which over-represents edge-on systems. Since edge-on is where the degeneracy is weakest, this does not bias the null toward a self-serving conclusion, but it does limit astrophysical realism."

---

### 8. The "Dominant Mode Only" Claim is Repeated Without Supporting Evidence

**The Text:** "All injections use the dominant quadrupole (ℓ = 2, |m| = 2) mode only (IMRPhenomD, with no higher-order modes)" appears in Section 4.1 and again in Section 8.3.

**The Problem:** The author notes this is asserted from knowledge, not recorded. Repeating it twice doesn't make it more verified.

**Suggestion:** Say it once in Section 4.1 with the caveat, and reference back to it in Section 8.3 rather than restating it.

---

## Summary Assessment

| Dimension | Score (1-5) | Comments |
|-----------|-------------|----------|
| **Methodological rigor** | 5 | Diagnostic discipline is exemplary |
| **Honesty about limitations** | 5 | Caveats are explicit and well-documented |
| **Strength of evidence for null** | 4 | Strong within defined conditions; architecture pool limitation remains |
| **Clarity of claims** | 3 | "Certified" vs "corroborating" is ambiguous; λ "exhaustion" is contradictory |
| **Reproducibility** | 3 | Waveform provenance gap is a real issue |
| **Contribution to field** | 3 | Methodology > physics finding; null is expected, not surprising |

**Overall Verdict: A defensible thesis chapter with clear methodological contributions.**

The chapter has addressed the previous review's major critiques. The inclination stratification is the most important addition and substantially strengthens the argument. The remaining concerns are primarily about clarity and framing, not about the substance of the result.

**Recommended Final Actions Before Submission:**

1. Revise the "exhausted" language in Section 7.3 to avoid contradiction
2. Clarify the "certified" vs "corroborating" distinction in the Introduction
3. Address the waveform-provenance gap more directly in Section 8.3
4. Move the inclination-prior caveat to Section 4.1
5. Ensure Table 6.6 captions are self-contained
6. Frame the perturbation trace as supporting, not primary, evidence in the Conclusion

With these minor revisions, the chapter is ready for thesis submission.
