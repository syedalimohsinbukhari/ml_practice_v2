# Final Review: Chapter on φ_c/ψ Degeneracy (V4)

## Executive Summary

**This chapter is now thesis-ready.** The three rounds of adversarial review have produced a methodologically rigorous, honestly caveated, and logically coherent chapter that successfully establishes its null result and, crucially, **provides the exact "Step 0" justification needed for the next chapter's inclination-conditioning experiment.**

The V4 changes—particularly the clarification in § 2.1 that conditioning on inclination is a *training-paradigm design choice*, the explicit flagging of the poc_b curriculum as a confound rather than a pure replication, the provisional framing of A.3 pending replication, and the clarification of evidential weight in § 8.2—have addressed the last remaining ambiguities.

---

## The "Step 0" Justification: Now Explicit and Bulletproof

**Your framing in § 2.1 is perfect:**

> "Conditioning on inclination ι ... is treated throughout as a *training-paradigm design choice*, motivated by the analytic structure of § 3 showing that the well-constrained combination is recoverable given ι, not as a forced move necessitated by a proof that ι itself is unrecoverable."

**The logical chain is now explicit:**

| Step | What the Chapter Establishes |
|------|------------------------------|
| **1** | § 3: The well-constrained combination is known *given* sign(cos ι) |
| **2** | § 6: Unconditioned networks fail universally—no architecture recovers φ_c/ψ |
| **3** | § 6.7: Even edge-on (where degeneracy is weakest) shows no signal |
| **4** | § 5.6: The pipeline works (positive controls succeed); the failure is target-specific |
| **5** | § 8.5: The logical next step is to condition on ι to test whether the degeneracy is breakable with side information |

**Conclusion for your thesis narrative:**
> *"Since strain alone cannot recover φ_c/ψ, but the waveform equations show identifiability conditional on ι, the only remaining hypothesis is that the network needs ι as an explicit input. We therefore test this hypothesis in the next chapter."*

This is clean, defensible, and exactly what a thesis examiner wants to see.

---

## What V4 Fixed (Relative to V3)

| Issue | V3 Status | V4 Fix |
|-------|-----------|--------|
| Ambiguity about why inclination conditioning is justified | Implicit in § 8.5 | **§ 2.1 now explicit**: it's a training-paradigm design choice, not a forced move |
| poc_b treated as independent repeat of null | § 6.2 called it "supporting evidence" | **§ 6.2 now flags it as curriculum-confounded**; § 8.2 clarifies it carries less evidential weight than cnn_attention |
| A.3 closure asserted outright | "A.3 is closed" | **Now "provisionally closed, pending replication"**; § 6.6 and § 9 both frame it as corroborating, not primary |
| cnn_attention vs poc_b evidential weight | Implicit | **§ 8.2 now explicit**: "this section's architecture-sufficiency argument rests primarily on cnn_attention's memorization gap" |
| Shared-head-capacity confound for inclination stratification | Not discussed | **§ 6.7 now notes**: a shared head-capacity limitation could in principle produce correlated failure across all periodic heads—honest caveat added |

---

## Remaining Items (All Scoped as Future Work)

The editorial note at the top now correctly lists three items that remain **critical future work** that this text pass cannot resolve:

1. **Synthetic-data ablation** for poc_b's curriculum vs. physical degeneracy
2. **High-SNR (25–30) validation set** or Fisher-matrix bound
3. **Dataset regeneration** with version-controlled script to close the provenance gap

**Assessment:** These are honest, well-scoped future work items. None undermines the chapter's core claim. The provenance gap (item 3) is the most serious, and you've now framed it correctly as a *reproducibility* issue rather than a *directionality* issue.

---

## Minor Residual Issues (Non-Blocking)

### 1. The "Shared Head Capacity" Caveat in § 6.7 is Slightly Awkward

**The Text:**
> "a shared limitation in mapping the trunk's representation to angular targets — rather than a shared loss-path bug, which is already ruled out — could in principle produce correlated failure across all three periodic heads"

**The Problem:** This is a valid caveat, but it's buried in the middle of a paragraph and reads like an afterthought. It's a genuine limitation—a shared head architecture could be the bottleneck—and it deserves clearer placement.

**Recommendation:** Move this caveat to the **end of § 6.7**, as a clear "limit of this argument" statement, rather than embedding it mid-paragraph.

---

### 2. The "Conditioning" Language in § 2.1 Could Be Even Sharper

**The Text:**
> "treated throughout as a *training-paradigm design choice*"

**The Problem:** "Training-paradigm design choice" is a bit abstract. The reader might wonder: "Does this mean you feed ι during training but drop it at inference? Or do you feed it at all times?"

**Recommendation:** Add one clarifying sentence in § 2.1:

> "Specifically, we feed true (sin ι, cos ι) as explicit input channels to the model at both training and inference time—this is a test of the conditional identifiability of the problem, not a shortcut that leaks information only during training."

This removes any ambiguity about what "conditioning" means.

---

### 3. Table 6.6 Caption: "Positive Δ" Definition

**The Caption:**
> "Positive Δ = edge-on ang_MAE *better* than the random-guessing null"

**Assessment:** This is now clear. No issue.

---

## Overall Verdict

| Dimension | Score |
|-----------|-------|
| **Methodological rigor** | 5/5 |
| **Honesty about limitations** | 5/5 |
| **Strength of evidence for null** | 4/5 (strong within defined scope) |
| **Clarity of claims** | 5/5 (now explicit) |
| **"Step 0" justification for next chapter** | 5/5 (now bulletproof) |
| **Reproducibility** | 3/5 (provenance gap is the single serious limitation) |

**Final Verdict: READY FOR THESIS**

The chapter does exactly what it needs to do:
1. Establishes a defended null result for φ_c/ψ under point estimation
2. Documents the diagnostic methodology with exemplary honesty
3. Provides the logical justification for the next chapter's inclination-conditioning experiment

**One final recommendation before submission:**

Add a **one-sentence roadmap** at the end of the Introduction (or early in § 2.1) that explicitly tells the reader *why* this null result matters for the thesis as a whole:

> "This null result—if established—has a direct architectural implication: the analytic structure of the degeneracy shows that conditioning on inclination should break the unidentifiability. We therefore test a conditional architecture in the following chapter, using true inclination as an auxiliary input."

This gives the reader the narrative thread from the start.

---

**The chapter is otherwise ready. Submit with confidence.**
