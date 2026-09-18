# Task brief: deep comparison — original φc/ψ PoC vs. the redo

This file is instructions for a fresh Claude Code session with no memory of the work that produced it.
Paste this file's content (or point the session at it) to start the task.
`CLAUDE.md` will load automatically for any session in this repo — read it before starting, its conventions apply to everything below.

## Context

This repo contains two related investigations into a suspected φc/ψ (coalescence-phase / polarization-angle) degeneracy in a gravitational-wave parameter-estimation network:

- **`experiments/phic_psi_poc/`** — the original investigation.
  Its entire `combo_A`/`combo_B` design was built on the combination `φc±2ψ`.
  A reviewer later flagged this as physically wrong: the injection convention actually makes `φc` enter the waveform doubled too, so the correct combination is `2φc±2ψ`.
  Full evidence chain for the fix: `experiments/phic_psi_poc/diagnostic_log.md`, dated entry 2026-09-09.
- **`experiments/phic_psi_poc-redo/`** — the redo, under the corrected `2φc±2ψ` formula.
  `redo_procedure.md` in that directory already maps every step of the redo against the original's own plan, tagged `[UNCHANGED]`/`[CORRECTED]`/`[NEW]` — read it first, it's the procedural skeleton for this task.

A prior chat session already did an informal, chat-level comparison and reported the headline result: **fixing the formula did not change the conclusion.**
Both investigations land on a null result for `coa_phase`/`polarization_angle` recoverability.
The redo assembled broader and more rigorous evidence than the original (a formal ungated bootstrap significance test across all 4 certified models, plus a gradient-chain check that rules out an implementation bug as the cause of an observed mode collapse) and surfaced two findings the original's methodology could not have caught (`cnn_attention`'s train/validation memorization divergence, and `inclination` carrying a small-but-real sub-floor signal rather than being a clean noise floor).

**That headline is not the task.** The task is to turn it into a rigorous, numbers-backed, side-by-side deep comparison — tracing every analogous check between the two investigations, not just re-confirming the headline conclusion.

## Goal / deliverable

Produce a new file, `experiments/phic_psi_poc-redo/original_vs_redo_comparison.md`, containing the deep comparison.
Register it in `experiment_index.md` in the same change that creates it, per this repo's own indexing convention.
Every numeric claim in the comparison must cite its exact source (file path, and the specific row/value/line it came from) on **both** sides — same claim-to-artifact discipline the original's own thesis chapter already holds itself to.

## Required reading

### Original side

- `experiments/phic_psi_poc/diagnostic_log.md` — the primary historical record, long, read in full rather than sampling.
- `experiments/phic_psi_poc/NOTES.md`
- `experiments/phic_psi_poc/experiment_index.md`
- `experiments/phic_psi_poc/results.md`
- `experiments/phic_psi_poc/preregistration_lam_retune.md`
- `experiments/phic_psi_poc/thesis/chapter_phic_psi_degeneracy.md` — the polished write-up, including its adversarial-review responses (multiple reviewers are referenced by name in the redo's own docs — Kimi, DeepSeek, Qwen).
- Original output directories (`bootstrap_output/`, `snr_output/`, `inclination_output/`, `analysis_output/` or equivalents) for the original's own numeric results.

### Redo side

- `experiments/phic_psi_poc-redo/redo_procedure.md` — already tags every step against the original's plan; verify its tags are still accurate given everything that happened after it was written (the Step 0 gate, the λ-retune, the Stage 2b(i) batch), don't take it as final without checking.
- `experiments/phic_psi_poc-redo/NOTES.md` — full narrative; the 2026-09-15 through 2026-09-18 entries carry the bulk of the redo's own findings.
- `experiments/phic_psi_poc-redo/experiment_index.md`
- `experiments/phic_psi_poc-redo/preregistration_lam_retune.md`
- `experiments/phic_psi_poc-redo/run_commands.md`
- Redo output directories: `bootstrap_output/`, `snr_output/`, `inclination_output/`, `analysis_output/`, `diagnostic_output/`, `perturbation_trace_output/`.

## Known anchor facts

Established already; cite their sources rather than re-deriving them, but do verify each against the artifact before restating it as fact.

- Combo formula: original used `φc±2ψ` (wrong), redo uses `2φc±2ψ` (corrected) — evidence in `diagnostic_log.md`'s 2026-09-09 entry.
- Same 4 certified/down-selected architectures on both sides: `poc_a`, `poc_b`, `tcn`, `cnn_attention`.
- Both sides ran a λ ablation (original: 0/0.01/0.05/0.10; redo: 0.01/0.05/0.10) reaching the same qualitative "λ alone insufficient" conclusion.
- The redo's Step 0 std_ratio gate never cleared at any λ, so its preregistered path stayed formally UNINTERPRETABLE.
- The redo's broader Stage 2b(i) batch (2026-09-18) gives a formal, ungated bootstrap null for `coa_phase`/`polarization_angle` across all 4 models, plus a gradient-chain check confirming the observed mode collapse isn't an implementation bug.
- Two findings exist only on the redo side: `cnn_attention`'s train/validation memorization divergence, and `inclination`'s small but statistically significant sub-floor bias.
- Neither investigation's write-up (`diagnostic_log.md`, `NOTES.md`, the thesis chapter, the redo's `NOTES.md`) discloses that `poc_b`'s individual `coa_phase`/`polarization_angle` bootstrap numbers rest on ground-truth-assisted branch reconstruction (`transforms.inverse`'s use of `formulae_reference.md` A.8) — true in both `bootstrap_ang_mae.py` runs, flagged in neither's original reporting. Didn't change either verdict (both read non-significant for `poc_b` on those two heads), but is a real, undisclosed methodological property worth stating explicitly in the deliverable rather than passing over silently a second time.
- **Needs investigation, not just noting:** `inclination`'s bootstrap significance pattern changed between the two investigations in a way the formula correction cannot explain (inclination is a separate head, untouched by the combo construction). Original: only `cnn_attention` significant (z=+3.17, p=0.0007); `poc_a`/`poc_b`/`tcn` not significant (z=+1.04/+0.07/+1.21, p=0.11–0.46). Redo: **all four** significant (z=+2.00/+2.58/+2.58/+3.58, p=0.0003–0.022) — source: `experiments/phic_psi_poc/bootstrap_output/bootstrap_ang_mae_20260721_093533.md` vs. `experiments/phic_psi_poc-redo/bootstrap_output/bootstrap_ang_mae_20260916_113955.md`. Every model's z-score moved upward, not just one — track down whether this is checkpoint/seed variance, a data-pipeline difference between the two investigations, or something else, rather than assuming it's noise.

## Required comparison dimensions

This is what makes the deliverable "deep" rather than a repeat of the brief chat-level summary already given.

1. **Procedural/methodology parity.**
   Use `redo_procedure.md` as the backbone.
   Report what's `[UNCHANGED]`, `[CORRECTED]`, `[NEW]` — and add a category it doesn't have: checks that exist in the **original** with no redo counterpart at all (the adversarial review process and the thesis chapter are the obvious candidates; find out if there are others).
   Flag these explicitly rather than omitting them silently.
2. **Numeric side-by-side table.**
   For every check that exists on both sides — Step 1.1's sign/combo ratios, the `w(ι)` derivation, Round-1 down-select R² values, the λ-retune's std_ratio gate numbers at each λ, bootstrap p-values per head/model, inclination/SNR stratification deltas, and any gradient/perturbation-trace results the original produced — pull the actual numbers from both sides' artifacts into one table.
   Do not summarize as "similar" or "different" without showing the numbers side by side.
3. **Verdict/epistemic-status comparison.**
   The original certified a null on 2 specific models, in thesis-chapter language.
   The redo has two verdicts that don't automatically collapse into one: the gated λ-retune path (UNINTERPRETABLE) and the broader Stage 2b(i) evidence (a formal null across 4 models).
   Explicitly reconcile these, or explain precisely why they resist clean reconciliation.
4. **Asymmetric findings.**
   For everything either side found that the other didn't, classify it: methodological gap (one side simply never ran the check), genuinely new discovery (the check existed on both sides but only fired on one), or population/config difference (not a fair comparison to begin with).
5. **Confidence synthesis.**
   Given everything now known on both sides, write one paragraph stating the strongest defensible physics conclusion about φc/ψ recoverability from strain, citing exactly which artifacts (both investigations) support it.
6. **Open items / recommended next steps.**
   Should the redo get its own thesis chapter and adversarial-review pass now that its completeness audit is done?
   Should `perturbation_trace_standalone.py final` be pursued (its `early` calibration failed and was never resolved)?
   Is the preregistration reconciliation from point 3 above still an open decision?
   List these plainly rather than deciding them — they're for the user.

## Format and conventions

- One sentence per line for prose, per this repo's `CLAUDE.md` — this applies to the deliverable file itself, not just to this brief.
- Every number needs a citation: file path plus the specific value/row it came from.
- Use a real markdown table for the numeric side-by-side comparison, not prose paragraphs.
- Add the new file's row to `experiment_index.md` in the same change that creates the file.
- Do not edit either side's frozen documents (`preregistration_lam_retune.md`, dated historical snapshots) — cite them, never modify them.
- This is a read-only synthesis task over existing artifacts.
- Do not run training, evaluation, or GPU scripts — the machine this repo is normally worked on is CPU-only per `CLAUDE.md`, and nothing in this task requires new computation.

## Out of scope

- No new training runs, no new diagnostic scripts, no new checkpoints.
- No changes to either investigation's own recorded conclusions or verdicts — this task documents and reconciles, it does not relitigate what either investigation already decided.
