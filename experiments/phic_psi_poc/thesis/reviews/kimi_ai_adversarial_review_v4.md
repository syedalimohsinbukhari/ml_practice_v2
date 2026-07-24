 This chapter has reached an unusual state: it is now more self-critical and better scoped than the vast majority of published null-result papers. Four rounds of adversarial review have stripped away nearly every overclaim and replaced them with explicit caveats. What remains are a few tonal inconsistencies where strong interpretive language still sneaks in after weak evidence, and one structural issue that has become more visible now that the confounds are honestly labeled.

---

### 1. The Effective Certified Base Is Now N≈1, Not N=2

You have done the intellectually honest thing by explicitly labeling poc_b as "confounded by curriculum design" (§6.2) and downgrading it to a "distinct, curriculum-confounded consistency check" (§8.2). But this honesty collapses the certified base. **cnn_attention** is the only model that both (a) passes the std_ratio health gate and (b) shows the memorization gap cleanly. poc_b does not show a memorization gap, and its flat training loss is ambiguous between "no signal in data" and "curriculum suppresses gradient signal over ~70% of the dataset" (face-on samples where w(ι) ≈ 0).

The text still writes as if the two certified models jointly carry the architecture-sufficiency argument: "Read together, the two certified models fail in two different, individually informative ways" (§8.2). But they are not symmetric. cnn_attention demonstrates that capacity is not the binding constraint. poc_b demonstrates that even a curriculum designed to exploit the analytically predicted structure cannot find in-sample signal—but this is equally consistent with a pathological curriculum. "Read together" implies mutual reinforcement; actually, one model is clean evidence and the other is a confounded secondary observation.

**Adversarial demand:** In §8.2, separate the two models' contributions explicitly and do not let them share a sentence that implies joint evidential weight. cnn_attention carries the architecture-sufficiency argument. poc_b is a separate, weaker consistency check whose interpretation is split between physical degeneracy and curriculum-induced gradient starvation.

---

### 2. §8.2 Overreads poc_b's Flat Training Loss

You write: "poc_b's optimizer finds no exploitable structure even in-sample, on a trunk independently proven capable of high-capacity fits elsewhere. Neither pattern is consistent with a capacity-starved null."

This is the old "supporting evidence" framing in new clothes. The curriculum weight is w(ι) = sin²ι. For |cos ι| > 0.9 (28.7% of the population), w < 0.19, meaning one combination is heavily suppressed. For |cos ι| > 0.5 (the majority of the population), w < 0.75. The curriculum intentionally down-weights the very samples where the degeneracy is strongest. A flat training loss under this curriculum is exactly what gradient suppression predicts, independent of whether the underlying target is learnable.

Your §6.2 already acknowledges this: "the curriculum's near-face-on suppression of one combination could itself produce rank-deficient gradients independent of whether the underlying target is truly unlearnable." But §8.2 then ignores this caveat and uses poc_b's flat training loss as evidence about the data. You cannot have it both ways.

**Adversarial demand:** Replace "poc_b's optimizer finds no exploitable structure even in-sample" with "poc_b's optimizer finds no structure even in-sample, which is consistent with either an unlearnable target or a curriculum that suppresses gradient signal over the majority of the dataset where sin²ι is small." This removes the overreading.

---

### 3. §6.7's "Most Likely to Break It" Framing Overstates the Edge-On Test

You call the edge-on band "the one population split most likely to break it" and conclude "the null holds, band by band, across the one population split most likely to break it." But your own §3 says the edge-on conditioning ratio is only **1.05–1.08×** (vs. 1.56× face-on). The edge-on band is indeed the *least degenerate*, but "least degenerate" does not mean "likely to break the null." A 1.05× correlation advantage is tiny. The absence of detection there is not a robustness check; it is consistency with the analytic prediction that the marginal signal remains below the noise floor even at edge-on.

The current framing makes the inclination stratification sound like a strong independent test. It is not. It is a weak test that confirms the §3 prediction: even where the degeneracy is weakest, the conditioning ratio is still ~1.05×, far below what a point estimator could extract at SNR 7–15.

**Adversarial demand:** Soften the framing. Replace "most likely to break it" with "where the analytic study predicts the degeneracy is weakest but still predicts only a ~1.05× conditioning advantage." The stratification is valuable as consistency with §3, not as a robustness check that independently validates the null.

---

### 4. The Head-Capacity Confound Is Acknowledged but Still Load-Bearing

§6.7 adds an explicit caveat: "a shared limitation in mapping the trunk's representation to angular targets... could in principle produce correlated failure across all three periodic heads." You bound this with the code-level trace (different loss paths) and the sky-position argument. But sky position uses a **different head architecture** (vMF on the sphere). The shared periodic heads (φ_c, ψ, ι) all use the **same two-layer MLP with circular loss**. The fact that sky position learns well does not rule out a head-capacity or head-inductive-bias limitation specific to the circular-regression MLP.

Your scope limit in §2.1 and §8.1 ("in this architecture/loss family") covers this. But §8.2 then claims: "the null is a statement about the data, not about the hypothesis class." This is too strong. If the two-layer MLP + circular loss is the binding constraint on angular recovery, then the null **is** about the hypothesis class, at least for angular targets.

**Adversarial demand:** In §8.2, qualify the "statement about the data" sentence: "For the scalar parameters, the memorization gap shows the null is a statement about the data; for the angular parameters specifically, the shared two-layer MLP head architecture leaves a residual head-capacity confound that this chapter does not test." This keeps the claim honest without weakening the scalar-control evidence.

---

### 5. The "Seven Architectures" Language Is Now Misleading

The Introduction still says: "across five trunk architectures, two head parameterizations, four values of a magnitude-regularization coefficient... no model ever performed measurably better than random guessing." And §9 says: "Every model, at every capacity and every regularization strength, converged to the optimal constant predictor."

This is true in the sense that all seven configurations land at the null ang_MAE. But by your own accounting, only **one** (cnn_attention) is fully certified as a clean test of the degeneracy. The others are either curriculum-confounded (poc_b), metric-unhealthy (tcn/φ_c, poc_a/ψ), or λ-mismatched (ResNet1D, CNN baseline, Inception). Saying "every model... at every capacity" implies seven independent confirmations. There is one clean confirmation and six corroborating observations under varying degrees of caveat.

**Adversarial demand:** In the Introduction and Conclusion, replace "every model" with "every configuration tested" or "all seven configurations," and let the "certified base is two models, one clean" framing appear in the same breath. Do not let the aggregate count imply independent replication.

---

### 6. The Chapter Is Now Approaching Submission-Ready Honesty

After four rounds, the remaining issues are tonal rather than structural. The four items in the editorial note are correctly identified as requiring new experiments, not text fixes. The A.3 provisional closure is handled correctly. The Step 0 framing in §2.1 is now logically sound. The dataset provenance is flagged as "not acceptable practice," which is stronger than most papers manage.

The one remaining question is whether a thesis committee will accept a chapter whose central claim rests on **one clean model** (cnn_attention) plus five corroborating but uncertified configurations, with two explicit confounds (curriculum for poc_b, head-capacity for all periodic MLP heads) acknowledged but untested. My assessment is: **yes, for a thesis, this is defensible**, provided the Conclusion does not oversell the "seven architectures" count as independent evidence.

---

### Summary Verdict

v4 is a **genuinely unusual piece of scientific writing** in its willingness to publicize its own limitations. The adversarial reading now finds only **framing and tonal issues**, not hidden logical gaps.

**If I were the external examiner, I would ask for:**

1. **Explicit separation of cnn_attention and poc_b in §8.2** — do not let them share a "joint evidence" sentence.
2. **Softening of poc_b's "no exploitable structure even in-sample"** to acknowledge the curriculum's gradient suppression.
3. **Downgrade of the edge-on stratification** from "robustness check" to "consistency with §3's predicted weak signal."
4. **Qualification of "the null is a statement about the data"** to exclude the residual head-capacity confound for angular targets.
5. **Tightening of "every model" language** in the Introduction and Conclusion to reflect that the certified base is narrower than the table count.

With these, the chapter is ready. Without them, a careful reader will still detect moments where the prose's confidence exceeds the statistics.
