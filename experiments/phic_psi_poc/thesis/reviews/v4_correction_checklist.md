# V4 Adversarial Review — Correction Checklist

Source reviews: `deepseek_ai_adversarial_review_v4.md`, `kimi_ai_adversarial_review_v4.md`, `qwen_ai_adversarial_review_v4.md`.
Cross-checked against `chapter_phic_psi_degeneracy.md` as it stands on branch `poc/phic-psi-degeneracy` (line numbers below refer to this file; the `.tex` mirror is not separately checked — any text fix here must be mirrored there in the same edit session per project convention).

All three reviewers agree the chapter is fundamentally sound; every item below is either a wording/tonal tightening or a request for a new artifact.
None of the three reviews asks for a structural rewrite or identifies a logical gap.

---

## 1. Pure wording/tonal fixes (safe to apply directly)

These require no new data, no new script run, and no new number — only edits to existing prose, all traceable to facts already established elsewhere in the chapter.

- **D1** — Move the §6.7 head-capacity caveat to the literal end of the section.
- **D2** — Add one explicit sentence to §2.1 clarifying that true (sin ι, cos ι) will be fed at both training *and* inference time.
- **D4** — Add a one-sentence roadmap at the end of §1 (or early §2.1) foreshadowing the inclination-conditioning chapter.
- **K1** — In §8.2, stop letting cnn_attention and poc_b share a sentence that implies joint/symmetric evidential weight.
- **K2** — Soften "poc_b's optimizer finds no exploitable structure even in-sample" to acknowledge the curriculum-suppression reading in the same sentence, not two paragraphs later.
- **K3** — Downgrade "the one population split most likely to break it" (§6.7 close) to language matching the actual ~1.05× predicted advantage.
- **K4** — Qualify "the null is a statement about the data, not about the hypothesis class" (§8.2) to exclude the angular-target head-capacity confound already conceded in §6.7.
- **K5** — Replace "every model"/"seven architectures" framing in §1 and §9 with "every configuration tested," paired with the certified-base caveat in the same breath (largely already done in §1; §9 still uses the unqualified phrase).
- **Q1** — Add an explicit sentence in §6.2 stating the poc_b combination-head losses also sat at the null (fact already exists in §5.5; this is a relocation/cross-reference, not new data).
- **Q2** — Add a parenthetical to §6.7's sky-position argument distinguishing envelope/time-delay information from absolute carrier phase, and tying in chirp mass as the existing phase-sensitivity proof.
- **Q4** — Add an explicit note that the ι noise floor (Huber loss, raw vector) and φ_c/ψ (circular loss) are not strictly identically distributed, tightening the existing "diagnostic calibration, not formal test" hedge.

## 2. Items requiring new analysis/data before they could honestly enter the chapter

- **Q3 — the "ridge test."** Qwen's suggested fix text is: *"the 2D joint distribution of the predicted (φ̂_c, ψ̂) pairs shows no ridge-like correlation (data not shown)."*
  **Do not add this as a text claim.** No artifact for it currently exists: `analysis_output/periodic_heads_20260720_234304.csv` holds only aggregate summary statistics (circular_r, angular_mae, peak locations), not per-sample prediction pairs, and no script or plot producing a joint (φ̂_c, ψ̂) scatter was found anywhere in the repo (checked `analysis_output/`, `inclination_output/`, and all `*.py` for "joint"/"scatter"/"ridge"). Reconstructing it requires reloading a trained checkpoint and running inference — a GPU operation this machine cannot do locally, per project convention. The phrase "(data not shown)" is also precisely the kind of unfalsifiable claim the project's artifact-traceability rule exists to forbid.
  **Flagged for the human to decide: skip / defer to future work (§8.5) / actually run it on the lab GPU machine.** If run, it would need its own `<type>_output/` directory, `.log`, summary `.md`, and appendix row per the standard artifact conventions before any sentence about it could enter the chapter.

## 3. Mild tensions between reviewers

- **Kimi (K2) vs. Qwen (Q1).** Kimi wants poc_b's evidential language *softened* — "no exploitable structure even in-sample" should immediately concede the curriculum-suppression alternative, not assert unlearnability outright. Qwen's suggested §6.2 addition is a *stronger*-sounding claim about the same model: "the network... failed to extract even the analytically well-constrained combination from the strain." Applying Qwen's exact suggested wording right after applying Kimi's hedge would partially undo it. Resolution: if Q1 is applied, its wording should inherit K2's hedge (e.g., "...also sat at the null, consistent with either an unlearnable target or curriculum-suppressed gradient signal — see §8.2") rather than Qwen's more definitive phrasing verbatim.
- **Deepseek vs. Kimi, in degree only (not a real conflict).** Deepseek's overall verdict is "submit with confidence," treating the current hedges (including the exact §8.2/§6.7 sentences Kimi objects to) as already sufficient — Deepseek's own "What V4 Fixed" table cites the §6.7 shared-head-capacity caveat and §8.2 evidential-weight sentence as fixes, not problems. Kimi reads the same passages as still overclaiming. This isn't a contradiction so much as two reviewers drawing the sufficiency line at different points; K1–K4's edits are strict supersets of what Deepseek already signed off on, so applying Kimi's sharper wording will not contradict Deepseek's assessment, only exceed it.

---

## Full itemized checklist

### D1 — Reposition the §6.7 head-capacity caveat
- [x] **Source:** deepseek (sole)
- **Resolved:** the caveat now stands alone as its own paragraph at the literal end of §6.7, after Table 6.7 and the "closes the gap" paragraph (merged with K3's fix in the same pass, as suggested).
- **Demand:** The head-capacity confound caveat is "buried in the middle of a paragraph and reads like an afterthought"; move it to the end of §6.7 as a clear "limit of this argument" statement.
- **Objected text:** *"a shared limitation in mapping the trunk's representation to angular targets — rather than a shared loss-path bug, which is already ruled out — could in principle produce correlated failure across all three periodic heads"* — chapter line 436.
- **Target section:** §6.7 (lines 408–456).
- **Current status: partially addressed.** The caveat exists verbatim (lines 436–437), but it sits *before* Table 6.7 and the section's closing paragraph (lines 439–455), which ends instead on the stronger note "the null holds, band by band, across the one population split most likely to break it" (line 455) — i.e., the section's actual last word undercuts rather than reinforces the caveat, the opposite of what Deepseek asked for.
- **Fix type:** (a) pure text edit — move/restate the existing sentence after Table 6.7's discussion, as the section's final sentence. Consider merging with K3's fix to line 455 in the same pass, since both touch the section's ending.

### D2 — Sharpen "conditioning" language in §2.1
- [x] **Source:** deepseek (sole)
- **Resolved (revised after user review):** first pass added a sentence stating true (sin ι, cos ι) is fed at both training and inference time, sourced from §8.5(i) — user caught that this reads as describing *this chapter's* training, when in fact no model reported here ever takes ι as an input at any stage (ι is a control *output* head only, § 2.2).
Replaced with two sentences that state that directly and point the input-channel design at §8.5(i) by name, instead of importing operational detail about the future experiment into the current chapter's problem formulation.
- **Demand:** Add a clarifying sentence: *"we feed true (sin ι, cos ι) as explicit input channels to the model at both training and inference time — this is a test of the conditional identifiability of the problem, not a shortcut that leaks information only during training."*
- **Objected text:** *"treated throughout as a training-paradigm design choice"* — chapter line 60.
- **Target section:** §2.1 (lines 48–63).
- **Current status: not addressed in §2.1.** §2.1 (lines 59–62) states the scope note about ι-conditioning being a training-paradigm choice but never states *when* true ι is fed. The detail exists only in §8.5(i), line 589: *"a full implementation plan exists (train-time truth, with inference-time ι estimation explicitly out of scope)"* — which, read carefully, implies true ι is used at inference too (only *estimating* ι at inference is out of scope), consistent with Deepseek's proposed sentence.
- **Fix type:** (a) pure text edit — the fact is already established in §8.5; this is echoing it earlier for clarity, not new data.

### D4 — Add a one-sentence roadmap to the Introduction
- [x] **Source:** deepseek (sole)
- **Resolved:** added a two-sentence paragraph after the twofold-contribution paragraph in §1, foreshadowing the ι-conditioning successor chapter and cross-referencing §3/§8.5.
- **Demand:** Add near the end of §1 (or early §2.1): *"This null result... has a direct architectural implication: the analytic structure of the degeneracy shows that conditioning on inclination should break the unidentifiability. We therefore test a conditional architecture in the following chapter..."*
- **Objected text:** N/A — this is an addition, not an objection to existing text. §1 currently ends at line 45 with the chapter roadmap ("Section 9 concludes") and never states why the null matters for the thesis as a whole.
- **Target section:** §1 (Introduction, lines 7–45).
- **Current status: not addressed.** Grepped for "architectural implication," "We therefore test," "following chapter," "next chapter" — no hits anywhere in the chapter. The forward-pointing motivation exists only in §8.5(i) (future work) and is never previewed in §1.
- **Fix type:** (a) pure text edit — draws only on already-established §3/§8.5 content.

### K1 — Separate cnn_attention and poc_b's evidential weight in §8.2
- [x] **Source:** kimi (sole, but thematically linked to K2)
- **Resolved:** the "Read together... the cleaner of the two" framing is removed; the paragraph now leads with "kept separate below rather than read together" and states cnn_attention carries the argument alone before poc_b is discussed (applied together with K2 in the same rewrite, since both touch this passage).
- **Demand:** Do not let the two certified models "share a sentence that implies joint evidential weight"; cnn_attention carries the architecture-sufficiency argument, poc_b is a separate, weaker, confounded check.
- **Objected text:** *"Read together, the two certified models fail in two different, individually informative ways: cnn_attention finds and fits sample-specific structure it cannot generalize... poc_b's optimizer finds no exploitable structure even in-sample, on a trunk independently proven capable of high-capacity fits elsewhere. Neither pattern is consistent with a capacity-starved null; the second is, if anything, the cleaner of the two."* — chapter lines 541–542.
- **Target section:** §8.2 (lines 524–554).
- **Current status: partially addressed.** The objected sentence is still present verbatim (lines 541–542) — including "Read together" and "the second is, if anything, the cleaner of the two," both of which read as joint/comparative framing. Immediately after (lines 543–544) the text *does* add an explicit asymmetry statement: "this section's architecture-sufficiency argument rests primarily on cnn_attention's memorization gap... poc_b... is a different, curriculum-confounded consistency check... without carrying the same evidential weight." So the explicit separation Kimi wants exists, but bolted on *after* the joint-framing sentence rather than replacing it — a reader stops at "Read together... individually informative ways" before reaching the correction.
- **Fix type:** (a) pure text edit — restructure so the separation precedes or replaces the "read together" framing rather than following it.

### K2 — Soften poc_b's "no exploitable structure" claim
- [x] **Source:** kimi (sole)
- **Resolved:** "poc_b's optimizer finds no structure even in-sample" is now immediately followed, in the same sentence pair, by the curriculum-suppression mechanism (w(ι)=sin²ι, w<0.19 for |cos ι|>0.9) rather than reading as an unqualified claim about the data.
- **Demand:** Replace *"poc_b's optimizer finds no exploitable structure even in-sample"* with *"poc_b's optimizer finds no structure even in-sample, which is consistent with either an unlearnable target or a curriculum that suppresses gradient signal over the majority of the dataset where sin²ι is small."*
- **Objected text:** Chapter line 541 (same sentence as K1) and line 542: *"Neither pattern is consistent with a capacity-starved null."*
- **Target section:** §8.2 (line 541), cross-referencing §6.2 (line 309) where the curriculum-confound caveat already exists for poc_b in a different context.
- **Current status: partially addressed.** §6.2 already carries the caveat Kimi wants echoed: *"the curriculum's near-face-on suppression of one combination could itself produce rank-deficient gradients independent of whether the underlying target is truly unlearnable"* (line 309). §8.2 line 544 also now has a version of it: *"its flat training loss could in principle reflect the curriculum's near-face-on suppression of one combination rather than the target's unlearnability alone."* But the specific sentence Kimi quotes (line 541) is unchanged and still reads as an unqualified claim about the data at the point it's made — exactly Kimi's "you cannot have it both ways" objection.
- **Fix type:** (a) pure text edit — the caveat text to borrow already exists twice in the chapter (§6.2 line 309, §8.2 line 544); this is consolidation/reordering, not new analysis.

### K3 — Downgrade "most likely to break it" framing
- [x] **Source:** kimi (sole)
- **Resolved:** replaced with explicit "~1.05–1.08× conditioning advantage against face-on's ~1.56×" language and a sentence clarifying the stratification corroborates §3's prediction rather than independently stress-testing the null.
- **Demand:** Replace "most likely to break it" with language reflecting the actual ~1.05× predicted conditioning advantage at edge-on; the stratification is consistency-with-§3, not an independent robustness check.
- **Objected text:** *"The null of § 6 is not confined to the face-on-dominated aggregate; it holds, band by band, across the one population split most likely to break it."* — chapter line 455 (exact phrase match).
- **Target section:** §6.7, closing paragraph (lines 453–455).
- **Current status: not addressed.** Confirmed via grep — line 455 is the only occurrence of "most likely to break it" in the file; no softened variant exists elsewhere.
- **Fix type:** (a) pure text edit — the ~1.05–1.08× figure is already established in §3 (line 98); this is just aligning §6.7's rhetoric with it.

### K4 — Qualify "the null is a statement about the data, not about the hypothesis class"
- [x] **Source:** kimi (sole)
- **Resolved:** sentence rescoped to "the trunk architecture" plus a new sentence explicitly carving out the angular-target head-capacity confound, cross-referencing §6.7.
- **Demand:** In §8.2, qualify this claim to exclude angular targets specifically, given the shared two-layer-MLP head-capacity confound conceded in §6.7: *"For the scalar parameters, the memorization gap shows the null is a statement about the data; for the angular parameters specifically, the shared two-layer MLP head architecture leaves a residual head-capacity confound that this chapter does not test."*
- **Objected text:** *"In one sentence: the null is a statement about the data, not about the hypothesis class, and the memorization gap is the evidence that the hypothesis class was not the limit."* — chapter line 534.
- **Target section:** §8.2 (line 534), with the relevant caveat currently living in §6.7 (lines 436–437) and the general scope limiter in §2.1 (line 55) / §8.1 (line 510) ("in this architecture and loss family").
- **Current status: not addressed at the point of the claim.** The generic scope qualifier "in this architecture/loss family" exists at lines 55 and 510, and the specific head-capacity confound is conceded in §6.7 (lines 436–437, see D1). But §8.2 line 534 itself is not qualified — the "hypothesis class" sentence reads unconditionally, and a reader of §8.2 alone would not see the angular-specific carve-out Kimi wants. This is the same underlying fact as D1's caveat; it just isn't cross-referenced into §8.2.
- **Fix type:** (a) pure text edit — add one clause pointing back to §6.7's existing caveat.

### K5 — Tighten "every model"/"seven architectures" language
- [x] **Source:** kimi (sole)
- **Resolved:** §9's "Every model" changed to "Every configuration tested"; §1 already had the equivalent fix from an earlier round.
- **Demand:** In the Introduction and Conclusion, replace "every model"/aggregate-count phrasing with "every configuration tested," and keep the certified-base framing in the same breath, so the aggregate count doesn't imply seven independent replications.
- **Objected text:** §1: *"across five trunk architectures, two head parameterizations, four values of a magnitude-regularization coefficient... no model ever performed measurably better than random guessing"* (line 26). §9: *"Every model, at every capacity and every regularization strength, converged to the optimal constant predictor"* (line 597).
- **Target section:** §1 (Introduction) and §9 (Conclusion).
- **Current status: mostly addressed in §1, not in §9.** §1 already carries the exact fix Kimi wants — lines 27–29 immediately follow the "no model ever" sentence with: *"All seven configurations land at the null; the distinction between 'certified' and 'corroborating' is about what else can be ruled out... For two λ-matched configurations, poc_b and cnn_attention, the pre-declared magnitude-health interpretability gate... confirms... these are the fully certified base."* This is presumably a holdover fix from the V3 round (editorial note cites "framing changes throughout § 1... § 9"). §9, however, still opens with the unqualified "Every model, at every capacity and every regularization strength, converged..." (line 597) before the certified-base caveat two sentences later (line 598: *"The fully certified base of that claim is two λ-matched configurations..."*). The substance is present in both places; only the literal word "every" (vs. "every configuration tested") persists as the residual nitpick Kimi is pointing at.
- **Fix type:** (a) pure text edit — word substitution only; no numbers change.

### Q1 — State poc_b's combination-head MAE explicitly in §6.2
- [x] **Source:** qwen (sole)
- **Resolved:** added a sentence in §6.2 restating the §5.5 combo A/B loss numbers, phrased with K2's hedge ("did not merely fail to decouple... also failed to show measurable in-sample progress") rather than Qwen's more absolute wording, per the tension noted below.
- **Demand:** Add to §6.2: *"the MAE of the combination heads themselves (Combo A and Combo B) also sat exactly at the random-guessing null baseline... The network did not merely fail to decouple the angles; it failed to extract even the analytically well-constrained combination from the strain."* Framed against the adversarial trap that the network might have learned the combination perfectly and the null MAE is a parameterization artifact of decoding back to individual angles.
- **Objected text:** §6.2's poc_b discussion (lines 304–309) never states the combo losses' own values, only individual-angle behavior and circ_r.
- **Target section:** §6.2 (lines 284–313); the underlying fact is stated in §5.5 (line 235).
- **Current status: already substantively addressed, but not in §6.2 as requested.** Line 235: *"poc_b's combination losses likewise (combo A 0.999 → 0.999, combo B 1.006 → 0.991)"* — this is exactly the number Qwen's adversarial trap needs to be defused, and it already refutes the "learned the ridge, decoding artifact" objection (a perfectly learned combination would show combo loss near 0, not ~1.0). It just isn't restated or cross-referenced in §6.2, where a reader focused on the poc_b collapse narrative would look for it.
- **Fix type:** (a) pure text edit — cross-reference or restate the existing §5.5 numbers in §6.2; no new data needed. Per the tension noted above (§3), phrase this with K2's hedge rather than Qwen's more absolute "failed to extract even..." wording.

### Q2 — Carrier-phase vs. envelope caveat on the sky-position analogy
- [x] **Source:** qwen (sole)
- **Resolved:** added two sentences to §6.7 naming the envelope/carrier-phase distinction and tying chirp mass in as the actual phase-sensitivity control.
- **Demand:** Add a parenthetical to §6.7 acknowledging sky position is constrained by amplitude modulation/time delays, not absolute carrier phase, so its success doesn't trivially prove carrier-phase sensitivity — mitigated by noting chirp-mass recovery (a genuine phase-evolution feature) as evidence the trunk isn't phase-blind.
- **Objected text:** *"the sky-position head — genuinely directional (a von Mises–Fisher likelihood on the unit sphere, § 2.2), and dependent on inter-detector phase information in a way analogous to φ_c/ψ — is recovered well"* — chapter line 434.
- **Target section:** §6.7 (lines 432–437).
- **Current status: not addressed.** Grepped for "carrier phase," "amplitude modulation," "antenna pattern" — the only hit is the unrelated §1 mention of antenna patterns (line 18); no such caveat appears in §6.7. The chirp-mass-as-phase-evidence point is also never explicitly connected to the sky-position argument anywhere in the chapter.
- **Fix type:** (a) pure text edit — chirp-mass R² is already an established artifact (§5.6, Fig. 5.4); this only requires drawing an inferential connection already licensed by existing numbers.

### Q3 — The "ridge test": 2D joint scatter of predicted (φ̂_c, ψ̂)
- [x] **Source:** qwen (sole)
- **Resolved:** put to the user directly (AskUserQuestion) rather than decided unilaterally. **User chose: defer to future work.** Added as new item (iv) in §8.5, marked inference-only/no-retraining and explicitly not claimed as done; not run, and Qwen's "(data not shown)" phrasing was not used anywhere in the chapter.
- **Demand:** Either produce a 2D scatter of predicted φ̂_c vs. ψ̂ to check for a ridge-like correlation (which would indicate the network learned the degenerate manifold rather than nothing), or state in §6.1 that this was checked and found absent.
- **Objected text:** N/A — a proposed addition, not an objection to existing text. Nothing about a joint prediction-pair distribution currently appears anywhere in the chapter (§6.1, lines 256–283, discusses only marginal circ_r per angle).
- **Target section:** §6.1 (lines 256–283).
- **Current status: not addressed, and not addressable by a text edit.** No artifact exists. `analysis_output/periodic_heads_20260720_234304.csv` contains only aggregate statistics (circular_r, angular_mae, peak locations) per model/head, not per-sample prediction pairs; no script or output anywhere in the repo (`analysis_output/`, `inclination_output/`, all `*.py`) computes or plots a joint (φ̂_c, ψ̂) distribution.
- **Fix type: (b) NEW ANALYSIS REQUIRED.** Qwen's own suggested text — *"(data not shown)"* — must not be used verbatim; it is an unfalsifiable claim of exactly the kind the project's artifact-traceability rule forbids. **This item is flagged for the human to decide**: skip it, defer it explicitly to §8.5 future work, or run it for real on the lab GPU machine (requires reloading a checkpoint and running inference — cannot be done on this machine). If run, it needs the standard `ridge_output/` (or similarly named) directory, timestamped log, summary `.md`, PNG+PDF figure pair, and a new claim-to-artifact appendix row before any sentence about it enters the chapter.

### Q4 — ι noise-floor loss-function mismatch
- [x] **Source:** qwen (sole)
- **Resolved:** added a sentence to §6.7 naming the Huber-vs-circular-loss distinction and its consequence for the two heads' finite-sample fluctuation distributions.
- **Demand:** Explicitly name that ι is trained with Huber loss on a raw 2D vector while φ_c/ψ use circular loss on a normalized vector, so their finite-sample fluctuation distributions are not strictly identical — tighten the existing "diagnostic calibration, not formal test" hedge with this specific mechanism.
- **Objected text:** *"the ι-noise-floor comparison is a diagnostic calibration, not a formal statistical test, and it assumes ι's band-to-band fluctuation is independent sampling noise..."* — chapter lines 430–431.
- **Target section:** §6.7 (lines 426–437); the Huber/circular-loss distinction is already established in §5.4 (line 61: *"a Huber loss on a raw two-vector, unlike the successful von Mises–Fisher parameterization"*) and §2.2/§5.6.
- **Current status: partially addressed.** The generic "diagnostic calibration, not a formal test" hedge (line 430) covers the spirit of the objection, and the underlying fact (ι's Huber-on-raw-vector loss path) is documented elsewhere in the chapter (§5.4). But the specific statistical point — that the two heads' finite-sample fluctuation distributions are not identically distributed *because of the loss mismatch* — is never stated in §6.7 itself.
- **Fix type:** (a) pure text edit — connects two already-established facts (§5.4's loss-path trace, §6.7's calibration hedge); no new data.

### Q5 — "Oracle Conditioning" note (§8.5 future-work framing)
- [x] **Source:** qwen (sole)
- **Resolved:** confirmed no-op for this chapter; no action taken (advice is for the successor chapter's introduction or defense slides).
- **Demand:** N/A for this chapter — Qwen explicitly frames this as advice to keep in reserve for the defense presentation or the *next* chapter's introduction, not a chapter-text fix: *"I highly recommend taking the... arguments we formulated and keeping them in your back pocket for your defense presentation slides or the introductory text of the next chapter."*
- **Objected text:** None — this references §8.5(i) approvingly, no objection.
- **Target section:** N/A (out of scope for this chapter).
- **Current status: no chapter action needed.** Not a correction to `chapter_phic_psi_degeneracy.md`; note for whoever drafts the next chapter's introduction.
- **Fix type:** N/A.

---

## Summary count

- Deepseek: 3 actionable items (D1, D2, D4), all pure text edits, all sole-sourced.
- Kimi: 5 actionable items (K1–K5), all pure text edits, all sole-sourced; K1/K2 are related (both re-scope poc_b's evidential weight in §8.2) but distinct asks.
- Qwen: 3 actionable items (Q1, Q2, Q4) as pure text edits + 1 item (Q3) requiring new GPU analysis, flagged for human decision; Q5 is a no-op for this chapter.
- **Total: 11 pure-text items, 1 new-analysis item requiring a human decision, 1 no-op.**
- No reviewer identified a currently-false or currently-unsupported claim in the chapter; every item is a hedge, a cross-reference, or an omission, consistent with the user's read that V4 feedback is "mostly tonal."
