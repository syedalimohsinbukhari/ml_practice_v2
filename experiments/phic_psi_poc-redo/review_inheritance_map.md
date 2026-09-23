# Review-Inheritance Map

**Purpose:** answer one question — if a new thesis chapter is written for the redo, does it need a fresh 4-round adversarial review, or does the redo's methodology already satisfy most of what the original's three reviewers (DeepSeek, Kimi, Qwen) demanded across their 4 rounds?

**Method:** every distinct concern across all 11 files in `experiments/phic_psi_poc/thesis/reviews/` (plus `v4_correction_checklist.md`) was extracted and classified against the redo's actual artifacts by four independent audits — three full independent passes plus one separate, unhurried single-shot comprehensive pass — then reconciled by hand below.
Where the four audits agreed, that consensus is reported directly.
Where they disagreed, the disagreement is stated explicitly and resolved with a specific tie-breaking argument, not averaged away.
Two claims below were independently verified against the underlying repo artifacts (the Bonferroni arithmetic in §2, and the scatter-plot image already inspected earlier in this investigation) rather than taken on any single audit's word.

## 1. Headline answer

**No, the redo cannot skip adversarial review of one specific thing: its evidentiary architecture.**
The original chapter's entire results/discussion/conclusion structure rests on "2 models (poc_b, cnn_attention) passed a Step-0 metric-health gate and are therefore certified; the other 5 configurations corroborate without independently repeating the test."
In the redo, that gate **never passed once** — at none of λ=0.01/0.05/0.10, for either combo vector, in either training mode (`experiments/phic_psi_poc-redo/diagnostic_output/diagnostic_logvar_gate_20260916_085902.md`).
There is no certified subset in the redo of any size, so every reviewer concern that leans on "certified vs. corroborating," "N=2," or "seven architectures implies seven replications" has no target left to inherit a resolution from — a new chapter needs an entirely fresh argument for why its actual evidentiary base (the broader, ungated Stage 2b(i) battery run across all 4 down-select-confirmed models) is trustworthy, since nobody has ever reviewed that specific substitute argument.
All four audits converged on this as the single most consequential finding, independently of each other, using different phrasing (§3 below has the details, including where one audit's initial classification label needed to be overridden).

**Everything else is much more tractable.** A meaningful fraction of the original reviewers' sharpest objections are already satisfied by the redo's methodology, by construction, because the redo was built after those lessons existed rather than before (§4).
A second meaningful fraction are concrete physics/ablation gaps that neither investigation ever closed — these need a research decision (run it, or explicitly re-defer it with a stated reason), not a review decision (§5).
A third fraction is pure editorial/drafting discretion with no artifact to check against yet (§6).

## 2. Two findings verified directly, not taken on any single audit's word

**(a) The redo's own "inclination is significant in all 4 models" claim does not survive the same multiple-comparisons correction the original chapter had to apply to itself once already.**
`closing_summary.md` reports inclination bootstrap p-values of 0.0223 (poc_a), 0.0052 (poc_b), 0.0059 (tcn), and 0.0003 (cnn_attention) as "significant... in every model tested" without stating or applying a Bonferroni correction.
Applying the same 12-test correction the original chapter's own Bonferroni fix used (`experiments/phic_psi_poc/diagnostic_log.md` line ~999, "0.0007 < 0.0042, so it does survive correction") to the redo's 12 analogous tests gives a threshold of 0.05/12 ≈ 0.00417 — verified directly: only cnn_attention's p=0.0003 clears it; poc_a, poc_b, and tcn's p-values (0.0223, 0.0052, 0.0059) do not.
Only one of the four independent audits caught this specific arithmetic gap.
It should be stated explicitly in any new chapter, exactly as the original had to correct itself once for the analogous mistake in the other direction.

**(b) The original's own certified poc_b model shows a literal mode collapse on its own scatter plot, undisclosed in any repo document.**
Confirmed by direct visual inspection earlier in this investigation (`runs/archive_phic_psi_poc_v1/phic_psi_poc_b/20260720_213202/scatter/epoch_0080.png`): poc_b's `φc` panel is two completely flat horizontal bands at the same angle mod 2π, despite a "healthy" std_ratio of 0.85 that the original chapter used as part of poc_b's certification.
This is not new evidence against the null (the chapter already reports and explains this exact collapse via `circ_r≈0.99` and the curriculum-suppression mechanism in `poc_b_config_diff.md`) — but it is direct, previously-uninspected proof that "std_ratio healthy" was never actually equivalent to "not mode-collapsed," a distinction no reviewer round ever caught for the original's own certified pair.
Three of the four audits independently flagged the general version of this concern (std_ratio vs. scatter-plot trust) as `NEW_UNADDRESSED`, and specifically noted that the redo's own `NOTES.md` (2026-09-16) rediscovered the identical failure mode at λ=0.10 and wrote a recommendation ("any future Step 0-style gate add a mandatory scatter-plot check") that is explicitly **not yet enforced anywhere** — a stated intention, not a closed gap.

## 3. The certification/evidentiary-architecture cluster (top priority, all four audits agree)

Every concern below collapses into one structural fact: the redo's Step-0 gate is 0-for-12 across all three λ rounds, so "certified vs. corroborating" has no analogue to inherit.

| Concern | Raised by | Reconciled classification | Cross-audit agreement |
|---|---|---|---|
| "Certified null rests on only 2 models" / ambiguous whether the null is backed by 2 or 7 | Kimi (R1§2, R2§1, R3§3, R4§1), DeepSeek (R1§4, R2§1) | **NEW_UNADDRESSED** | 4/4, though one audit initially labeled sub-items `STRUCTURAL_INHERITED (moot)` — overridden below |
| "Seven architectures"/"every model" language implies independent replication it doesn't have | Kimi (R4§5) | **NEW_UNADDRESSED** | 4/4 |
| Half the primary λ-matched set is "uninterpretable," undermining a cross-architecture null | Kimi (R3§6) | **NEW_UNADDRESSED** (worse in the redo: effectively 100%, not 50%, of gated readings fail) | 4/4 |
| Pre-registration "loophole": Step 0 failing means the physics test never ran; a cynical reviewer says only normalization instability was proven | Qwen (R1§3B) | **NEW_UNADDRESSED — single highest-priority item** | 2/4 direct; 2/4 initially called this `STRUCTURAL_INHERITED`, both flagging their own low confidence |
| §8.2 overreads poc_b's flat loss / poc_b and cnn_attention share unequal evidential weight | Kimi (R3§3, R4§2) | **NEW_UNADDRESSED** | 4/4 |

**Resolving the one real disagreement (the pre-registration-loophole row):** two audits argued this is `STRUCTURAL_INHERITED` because the redo built a parallel, ungated evidential path (Stage 2b(i): collapse histograms, gradient-chain health check, formal bootstrap, stratification) specifically to answer "is this physics or instability" independent of Step 0 — and that path is arguably more rigorous than the original's gate-then-bootstrap pipeline, since it establishes the null via several independently-convergent methods rather than one interpretability precondition.
That argument has merit and is not dismissed here.
But both of those audits flagged their own conclusion as low-confidence, and the other two audits' objection is decisive: whatever the merits of the substitute argument, **it is a novel evidentiary structure that no reviewer has ever seen or attacked.**
The question "does a Step-0 gate failing at every λ, replaced by an ungated battery invented after the fact, actually satisfy this objection" is itself an adversarial-review-shaped question — one a new chapter has to argue and defend, not one that resolves itself by construction.
This is why the overall answer in §1 is "cannot skip review of the evidentiary architecture specifically," even though the redo has strong raw material to make that defense.

## 4. Concerns already satisfied by the redo's methodology, by construction

These do not need a fresh review round — the redo's practices, built after the original review cycles existed, already satisfy the sharpest form of each concern. All four audits agree on this cluster with high confidence.

| Concern | Raised by | Why it's satisfied |
|---|---|---|
| A.3 perturbation-trace closure is post-hoc instrument selection (classifier retired mid-decision-tree, closure re-founded on an unplanned channel) | Kimi (R1§5, R2§5, R3§4), DeepSeek (R1§5, R2§5) | The redo's `early` calibration failed identically (mchirp reads AMBIGUOUS, all 4 models) and, unlike the original, the redo did **not** attempt the same salvage — `final` was explicitly never run, and `closing_summary.md` states its output "would be formally untrusted by this batch's own rule anyway." This satisfies the sharpest possible form of the demand: full transparency that the probe is parked, not provisionally closed. |
| λ-sweep "exhausted" language is self-contradictory alongside admitting a finer sweep might succeed | DeepSeek (R1§6, R2§3), Kimi (R3§7, re: the coarse-grid-looks-designed-to-fail variant) | `preregistration_lam_retune.md` never uses "exhausted"; it states its fallback order and termination condition in writing *before* any λ=0.05/0.10 result existed, and reports the outcome as "λ alone insufficient for this architecture/head... no further λ trial is authorized" — precise, non-contradictory language by design, not retrofitted after criticism. |
| Missing inclination-stratified analysis (face-on vs. edge-on) | DeepSeek (R1§9), Qwen (R1§1C) | `inclination_stratification.py`/`inclination_control_stratification.py` are standard, from-day-one-ported checks in the redo's own completeness audit, run unprompted by any fresh review round. |
| Memorization-gap argument (§8.2) only verified in aggregate / for unmatched trunks, not for the certified pair individually | Kimi (R2§1, R2§8, R4§1) | `plot_certified_memorization.py` was explicitly adapted to run per-model, uniformly across all 4 down-select-confirmed models — a stronger, more granular answer than the two-model check Kimi asked for, done because the redo certified no null (not retrofitted after a review demand). |

## 5. Concrete physics/ablation gaps — never closed on either side, need a research decision not a review decision

This is the cluster where the loosest of the four audits initially called several items `DEFERRED_BOTH` ("already named as future work on both sides, so no urgency").
That classification does not survive a strict check: the original's four explicit §8.5 deferred items (synthetic ablation, high-SNR/Fisher bound, dataset regeneration, ridge test) are **not** re-listed anywhere in the redo's own "still open" tracking (`closing_summary.md`'s "What's still explicitly open" section lists only: `perturbation_trace_standalone.py final`, verdict reconciliation, `cnn_attention`/sky_position mechanism, and the GPU-nondeterminism check for the inclination anomaly — none of the original four).
Two of the four audits caught this directly; the reconciled classification below uses their stricter reading, because it changes the practical implication: these items were **silently dropped**, not **actively re-deferred**, and a new chapter needs to either address them or explicitly re-defer them with a stated reason, rather than assume "deferred on both sides" status that was never actually re-established.

| Concern | Raised by | Status | Reconciled classification |
|---|---|---|---|
| poc_b's collapse may be a curriculum/optimization artifact (rank-deficient gradients from `w(ι)=sin²ι` suppression), not physical unlearnability — needs a synthetic-data ablation with deterministically recoverable combinations | Kimi (R1§3, R2§4, R3§3, R4§2), persistent across all 4 rounds | Original explicitly deferred (§8.5 item v); redo never re-lists it; no ablation script exists in either directory | **NEW_UNADDRESSED — silently dropped** |
| SNR range (7–15) too restricted to distinguish "degeneracy" from "signal below sensitivity floor" — needs a high-SNR (25–30) spot-check or Fisher-matrix bound | Kimi (R1§4, R2§3, R3§5) | Original explicitly deferred (§8.5 item vi); redo used the identical SNR-7–15 dataset, no Fisher bound computed | **NEW_UNADDRESSED — silently dropped** |
| Dataset-generation choices undocumented/unscripted; demands regenerating the dataset (or a slice) from a version-controlled generator | DeepSeek (R2§4), Kimi (R2§7, R3§5), Qwen (R1§1B) | Original explicitly deferred (§8.5 item vii); redo made real, partial progress — traced `coa_phase`'s injection convention through actual generator source (`formulae_reference.md` §A.6, `gen.py:202`/`gen.py:179`), stronger than the original's "recorded from authors' knowledge" — but the dataset file itself (`combined_repackaged.hdf`) is byte-identical, not regenerated | **PARTIAL — genuine provenance improvement, reproducibility gap still open** |
| "Ridge test": 2D joint scatter of predicted (φ̂c, ψ̂) to check whether the network learned the degenerate manifold itself rather than nothing | Qwen (V4, item Q3) | Original explicitly deferred (§8.5 item iv, after the user declined a "(data not shown)" shortcut); redo never mentions it; no joint two-angle scatter exists anywhere in the redo (`rose_plot_residuals.py` is a per-head residual plot, answers a different question) | **NEW_UNADDRESSED — silently dropped** |

## 6. Concerns re-run with a directly analogous check, and what changed

These are cases where the redo did rerun the original reviewer's underlying check — but in three cases the result changed enough that citing "we already checked this" would be misleading without restating the new number.

| Concern | Raised by | What the redo found |
|---|---|---|
| Inclination-noise-floor argument (§6.7) is circular — assumes ι's fluctuations are independent noise, doesn't rule out a shared head-capacity pathology across all periodic heads | Kimi (R2§2, R3§2, R4§4), DeepSeek (R2§2) | **Reopened, not resolved.** The redo's own bootstrap shows ι is *not* noise — it's significant in all 4 models (subject to the Bonferroni caveat in §2a), a small real signal. This undermines the "known-uninformative noise floor" technique's core assumption rather than confirming it; the shared-head-capacity hypothesis itself was never tested (no deeper-MLP-head ablation exists on either side). A new chapter cannot cite the original's noise-floor argument as precedent without rebuilding it. |
| Sky-position/positive-control success is coarse-grained (timing/amplitude), doesn't prove fine carrier-phase sensitivity | DeepSeek (R1§8), Qwen (V4, item Q2) | **Confirmed, self-aware.** The redo's own new `cnn_attention`/sky-position finding attributes the effect to cross-detector amplitude-ratio/antenna-pattern information — explicitly *not* carrier phase — directly conceding, not contradicting, exactly this caveat. |
| "Memorization" framing for the train/val loss gap is unverified — could be overfitting to spurious correlation, a weaker and different claim; recommends a 2×-data disambiguation test | Kimi (R1§8, R2§6) | **Reconfirmed at greater scope, terminology unresolved.** The redo re-ran this across all 4 models (not 2) and found the identical asymmetric pattern (`cnn_attention` gap ≈0.44–0.47; the other three ≈0.003–0.05). The empirical pattern holds up; the "memorization" vs. "overfitting-to-spurious-correlation" terminology question, and the 2×-data test that would settle it, remain untested on both sides. |
| Unphysical inclination prior (uniform in ι, not cos ι) | DeepSeek (R2§7), Kimi (R2§7), Qwen (R1§2C) | **Reconfirmed, unfixed.** Step 1.6 reproduces identical population statistics (28.7%/32.7% face-on/edge-on). The prior itself was never changed on either side — the redo's only improvement is disclosing it plainly and early (Step 1.6, day one) rather than it taking a review round to surface, as happened originally. |

## 7. Untested physics/representation critiques, unchanged from the original

No artifact on either side bears on these; they transfer to a new chapter exactly as they stood.

- Architecture pool (5 trunks, all temporal-convolutional/attention families) is insufficient diversity for a general "no architecture" claim; no frequency-domain, complex-valued, or physics-informed architecture was tried (DeepSeek R1§2, Kimi R1§7).
- Positive controls (chirp mass, SNR) are integrated/amplitude properties, not proof the network can extract instantaneous carrier phase (DeepSeek R1§7).
- Missing Bayesian posterior comparison to distinguish a point-estimation artifact from genuine inability to capture a degenerate/multimodal posterior structure (DeepSeek R1§10, Qwen R1§2B) — the general form of the ridge-test gap in §5.
- Raw strain representation may consume network capacity on implicit preprocessing rather than phase extraction (Qwen R1§2A) — one audit flagged that the redo's own docs describe the input as "raw *whitened*" strain, so Qwen's original premise (unwhitened raw strain) may not match either investigation's actual pipeline; worth a factual clarification in any new chapter regardless of that nuance.

## 8. Editorial / drafting-only concerns

A large fraction of the original 4 rounds — roughly a third of all concerns raised — were prose, structure, or notation issues with no artifact to check against, because no redo chapter exists yet to check: terminology consistency ("well-constrained" vs. "correlated," `circ_r` naming), table/figure caption clarity, appendix and narrative length, the placement of the §6.7 head-capacity caveat, a one-sentence roadmap to the ι-conditioning successor chapter, and similar.
These require no research and no adversarial review — they are quality-control items for whoever drafts the new chapter's text, resolvable by attention during drafting rather than by checking any current repo artifact.

## 9. Bottom line

A fresh chapter for the redo needs one targeted, real review round — not a full 4-round×3-reviewer cycle — scoped to the evidentiary-architecture question in §3 (why the ungated Stage 2b(i) battery should be trusted as the redo's central evidence, given its gated path never passed) plus honest disclosure of the two verified findings in §2.
Everything in §4 can be cited as already-satisfied without re-litigation.
Everything in §5 needs an explicit decision — run the ablation/spot-check/regeneration/ridge-test, or state plainly in the new chapter that it remains open and why — rather than silently inheriting a "deferred" status neither side actually re-affirmed.
Section §6's three items need their restated numbers cited accurately (especially the inclination-Bonferroni point), not just a pointer back to the original's precedent.
