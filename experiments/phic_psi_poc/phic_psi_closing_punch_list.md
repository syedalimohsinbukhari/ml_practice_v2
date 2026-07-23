# φc/ψ Degeneracy Investigation — Closing Punch List

Status: 9 runs deep, degeneracy hypothesis well-supported, pre-registered λ-sweep
exhausted per its own stopping rule. This list separates what should be done
before treating the chapter as final vs. what's legitimately future work.

---

## A. Do before finalizing — cheap, no new training runs

- [ ] **Un-gate and run the A.3 multi-step perturbation trace standalone.**
      It's currently trapped behind the λ-retune scripts' Step 0 gate and has
      never executed. Decouple it and run against the existing Run 7 λ=0.01
      checkpoint (or whichever is most representative). This is the one open
      empirical question left that could still change the write-up — it's
      meant to resolve whether the ~89× coa_phase/pol_angle vs. mchirp
      rel_change asymmetry (Run 7, Check 4) is directional-but-slow learning
      or noise. Cheap; no reason to leave it unrun going into the chapter.

- [ ] **Write one explicit sentence deciding the λ-sweep question, rather than
      letting "exhausted" stand implicitly.** The data (λ=0.05 near-passing,
      λ=0.10 regressing on every combo) show a peaked, non-monotonic
      relationship — not a flat "nothing works" result. Given the chapter is
      close to done, the practical call is almost certainly: *note this
      explicitly as a finer optimum likely existing between 0.02–0.08, state
      that it wasn't pursued further given time constraints, and flag it as
      future work* — rather than asserting the λ dimension was exhaustively
      ruled out. This is a documentation fix, not a request to run more
      training.

- [ ] **Documentation sweep** (per `doc_update_sweep_handoff_2026-07-22.md`) —
      bring `assessment_lam0_ablation_2026-07-22.md` and any other stale files
      in line with the Run 9a/9b outcome before anything gets cited from them.

- [ ] **One explicit paragraph on scope of the conclusion.** Make sure the
      chapter states the finding as "φc/ψ carry no strain-only recoverable
      signal extractable by this architecture/loss family (CNN/TCN/attention
      trunks, circular loss, sum/diff reparameterization)" — not an
      unqualified claim that the information is absent from the data under
      any method. This distinction has been maintained carefully throughout
      the investigation; worth double-checking it survives into the actual
      chapter text.

---

## B. Explicitly document as known-but-unresolved (no action needed, just say so)

- [ ] **Inclination's separate failure mechanism** (Huber loss, no
      `normalize_unit`, traced but not chased to a resolution) — state clearly
      that it was ruled out as a confound for the φc/ψ question and left there;
      not itself explained.
- [ ] **`sky_position` degradation specific to `SumDiffTrainer`** — flagged,
      never investigated. State as a known open issue, out of scope for this
      chapter.
- [ ] **40-epoch/±0.005-per-epoch gate window vs. the plateau LR schedule's
      settling time** — the Run 9a near-miss raised this; correctly not
      retroactively changed. Worth one sentence noting it as a methodological
      question for any future λ-sensitivity work, not resolved here.

---

## C. Explicitly deferred to future work (name them, don't chase them now)

- [ ] A finer, freshly pre-registered λ mini-sweep (0.02–0.08) — plausible
      given the peaked pattern in A above, not pursued further here.
- [ ] Architecture-level fix for `tcn` coa_phase / `poc_a` pol_angle
      std_ratio instability — named as "the next lever" by the diagnostic
      scripts themselves; not scoped or started.
- [ ] ι-conditioning experiments (true inclination as auxiliary model input)
      — the original next major phase. Given where this chapter lands, this
      reads as the start of a *new* investigation, not a tail end of this one.

---

## Suggested framing for the chapter's conclusion

Something close to: *"Across nine training runs spanning five architectures,
two head-parameterization schemes, and a four-point magnitude-penalty sweep,
φc and ψ — individually and in sum/diff combination — show no statistically
significant, SNR-monotonic, effect-size-meaningful recoverable signal beyond
what a shuffled-label null model achieves. This result survived deliberate
attempts to falsify it: pre-registered decision criteria, ablations isolating
engineering artifacts from physical signal, and cross-architecture replication.
The conclusion is scoped to the model class tested; whether inclination as an
explicit input, alternative data representations, or non-regression estimation
methods (e.g. full posterior inference) can recover this information remains
open."*

That framing gets you a defensible, appropriately-hedged result without
needing anything further from A/B/C above except the two cheap items in
Section A.
