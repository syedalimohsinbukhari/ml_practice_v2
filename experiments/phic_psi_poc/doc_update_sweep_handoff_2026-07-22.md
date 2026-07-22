# Doc Update Sweep — Handoff (2026-07-22)

Drop this file into a fresh conversation as the starting context. It exists
because the doc-update task (propagate the Run 9a/9b λ-retune outcome across
every file that references it) got interrupted mid-sweep in a conversation
that was getting overloaded with unrelated history — this file is the clean
restart point, not a summary of that conversation.

## Task

Sweep `experiments/phic_psi_poc/*.md` for stale references to the λ-retune
status and update them now that both pre-registered retune runs are
complete and closed. Docs-only — no training/data-loading code runs here;
everything executes on the user's lab GPU machine (see "Ground rules"
below).

## Already done (committed — do not redo)

Commit `08f407c` on branch `poc/phic-psi-degeneracy` already updated:
- `NOTES.md` — added "Run 9a" and "Run 9b" sections, updated the
  pre-ι-conditioning checklist
- `diagnostic_log.md` — same, plus more detailed raw-trace commentary

Treat these two files as the current source of truth for exact wording and
tables; other files should be brought into line with them, not the other
way around.

## The result to propagate (verbatim facts — reuse, don't re-derive)

Pre-registered decision procedure: `preregistration_lam_retune.md`, locked
2026-07-22 before either retune ran. Two primary pre-declared targets only:
tcn/coa_phase and poc_a(baseline)/polarization_angle. Step 0 is a std_ratio
interpretability gate (< 10% of last 40 epochs outside [0.5, 2.0] AND
|trend| < 0.005/epoch); Steps 1–3 (bootstrap significance, effect-size
floor, SNR-monotonicity) only run if Step 0 passes.

**Run 9a (λ=0.05, 2026-07-22):**

| Model | Head | frac unhealthy (last 40 ep) | trend/ep | Gate |
|---|---|---|---|---|
| tcn | coa_phase | 0.05 | −0.00638 | FAIL |
| poc_a (baseline) | polarization_angle | 0.35 | +0.00718 | FAIL |

Both close — each settles into a stable healthy plateau only in the back
~15–20 epochs; the 40-epoch gate window also catches the earlier climb.
Verdict: UNINTERPRETABLE for both (mechanical, Steps 1–3 didn't run).

**Run 9b (λ=0.10, 2026-07-22):**

| Model | Head | frac unhealthy (last 40 ep) | trend/ep | Gate | vs λ=0.05 |
|---|---|---|---|---|---|
| tcn | coa_phase | 0.28 | −0.00255 | FAIL | worse (0.05→0.28) |
| poc_a (baseline) | polarization_angle | 0.73 | +0.00731 | FAIL | much worse (0.35→0.73) |

Not a near-miss this time: poc_a pol_angle crashes hard early then climbs
slowly, crossing 0.5 only in the last ~11 of 80 epochs; tcn coa_phase
oscillates in [0.2, 0.95] with no clear convergence. Non-primary combos
mostly regressed too.

**Verdict language to reuse verbatim (per the pre-registered decision
table, not re-derived after the fact):** "λ alone is insufficient to
stabilize std_ratio for tcn coa_phase or poc_a polarization_angle... filed
as neither a null result nor counter-evidence for the degeneracy
hypothesis... the λ-sweep branch (0, 0.01, 0.05, 0.10) is exhausted for
these two heads; the next lever, if pursued, is architecture-level, not a
further λ value."

## Candidate files found via grep sweep

Command used (run from `experiments/phic_psi_poc/`):

```bash
grep -rl "λ retune\|lam010\|lam005\|ι-conditioning gate\|Still open\|tcn coa_phase\|pol_angle λ\|std_ratio problem" --include="*.md" .
```

Results and status:

- **`assessment_lam0_ablation_2026-07-22.md` — confirmed needs updating.**
  Section 4 table marks "tcn coa_phase λ retune" and "poc_a pol_angle λ
  check" as "Still open" — change to "Resolved — λ alone insufficient (Run
  9a/9b)". Section 5's recommendation items 1–2 describe hypothetical
  outcomes ("if it starts to learn, that's the first real counter-evidence…")
  that are now moot — replace with the actual outcome. Item 3 (perturbation
  trace) currently reads "no retraining needed, just run it" — that's now
  wrong; the trace is gated behind Step 0 passing, which never happened, so
  it remains blocked by design, not merely unscheduled.

- **`experiment_index.md` — confirmed needs updating.** "Last updated:
  2026-07-21" header is stale. The "Quick-reference" section's "Does the
  magnitude penalty prevent |v| drift?" line says "poc_a pol_angle and tcn
  coa_phase need λ tuning" — tuning was tried (0.05, 0.10) and didn't
  resolve it; update to point at Run 9a/9b. The "What's still open?" line
  says "Four small items... λ=0 ablation, multi-step perturbation trace,
  tcn λ retune, poc_a pol_angle λ check" — λ=0 ablation (Run 8) and both λ
  retunes (Run 9a/9b) are done; only the perturbation trace remains, and
  it's blocked, not just pending.

- **`run7_verification_rebuttal.md` — not yet checked.** `experiment_index.md`
  itself already labels this "superseded by actual verification results...
  retained for record" — read it before touching anything; if it's
  intentionally a frozen historical snapshot, leave it alone even if it
  mentions the retune as forward-looking. Don't rewrite history docs to
  match later outcomes.

- **`std_ratio_trajectories.md` — not yet checked.** This is the Section
  A.2 gating-check artifact that originally identified the two std_ratio
  problems the retune was meant to fix. Same judgment call as above: check
  whether it's framed as a dated snapshot (leave it, maybe add a pointer
  forward to Run 9a/9b) or as a living "still open" tracker (update it).

- **`lam005_retune_output/lam005_retune_report.md`,
  `lam010_retune_output/lam010_retune_report.md` — do not hand-edit.**
  Auto-generated by `run_lam005_retune.py` / `run_lam010_retune.py`'s
  `compare_trajectories()`; they regenerate from `history.csv` on every
  run and are already correct as of the last execution.

- **`preregistration_lam_retune.md` — do not edit retroactively.** This is
  the locked pre-commitment document; by design it stays exactly as written
  before either result existed. If a completion note is genuinely useful,
  append a dated addendum at the bottom — never touch the criteria
  themselves.

## Ground rules (carried over from this project)

- Docs-only. Never run `train_poc.py` / `evaluate_poc.py` / any retune
  script — the user runs everything on the lab GPU machine themselves.
- Report the pre-registered mechanical verdict as-is. Do not soften it,
  and do not retroactively adjust the gate window or thresholds now that
  the outcome is known — that exact temptation was flagged and explicitly
  avoided in the Run 9a/9b write-ups.
- Only stage/commit `experiments/phic_psi_poc/**` files relevant to this
  sweep. The repo has an unrelated untracked `deeplearning_models/`
  directory at the repo root — never stage it.
- Branch: `poc/phic-psi-degeneracy`. Match the existing commit-message
  tone (descriptive, present-tense, no fluff — see `git log` on this
  branch for examples).
- Ask before committing rather than committing proactively — the user has
  consistently asked to review/commit doc batches explicitly.

## Suggested first step

1. Read `assessment_lam0_ablation_2026-07-22.md`, `experiment_index.md`,
   `run7_verification_rebuttal.md`, `std_ratio_trajectories.md` in full.
2. Re-run the grep sweep with a broader pattern to catch anything missed —
   e.g. also search for `"UNINTERPRETABLE"`, `"gate FAIL"`, `"need λ
   tuning"`, `"still declining"` to find other stale forward-looking
   language.
3. For each file, decide: living doc (update to current state) vs. frozen
   historical snapshot (leave alone, at most append a forward pointer).
   `experiment_index.md`'s own description of each file is a good signal
   for which bucket it's in.
4. Apply edits, then show the diff for review before committing.

If a fuller picture of the whole investigation (not just this sweep) is
needed for context, see `experiment_summary_2026-07-22.md` in this same
directory.
