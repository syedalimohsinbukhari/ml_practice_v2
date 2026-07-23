# Project conventions

Each rule below was paid for by a specific failure in the φc/ψ investigation (see `experiments/phic_psi_poc/diagnostic_log.md`); do not relax them without an equally specific reason.

## Prose formatting (thesis / documentation)

- All prose files (`.tex`, `.md` — thesis chapters, experiment write-ups, notes) use **one sentence per line** (semantic line breaks). Never hard-wrap sentences across lines and never put multiple sentences on one line.
- This applies both when writing new prose and when editing existing prose: if a paragraph you touch is not in this format, reflow it.
- Protected content is exempt: LaTeX environments (`table`, `tabular`, `equation`, `figure`), Markdown tables, headings, blockquote captions, and code blocks keep their own layout.
- For bulk conversion use `experiments/phic_psi_poc/thesis/sentence_per_line.py` (`python3 sentence_per_line.py <file>`): joins hard-wrapped paragraphs, splits after sentence-ending punctuation (protecting abbreviations like e.g., i.e., Fig., vs., et al.), treats `\item` and `\paragraph` as new-line anchors, and leaves protected environments untouched.

## Full-trajectory analysis

- Any claim about a training metric must be judged on the **entire** `history.csv` trajectory, never on endpoint-only values — an endpoint std_ratio summary once hid a mid-training crash to 0.07 and recovery.
- Behavioral probes of a model (gradient traces, perturbation traces) must be calibrated across training stages, not run only against final checkpoints — a converged checkpoint makes a well-learned head look identical to a dead one on short probes; a probe whose positive control fails at the stage under test is untrusted.

## Output artifacts: naming, logs, gitignore

- Every analysis/diagnostic script writes into its own `<type>_output/` directory and produces a console-tee'd `<type>_<YYYYMMDD>_<HHMMSS>.log` plus a summary `.md` with the same stem.
- When a new log type is created, add `!**/<type>_*.log` to `.gitignore` **in the same change** — the global `*.log` ignore silently excludes it otherwise (existing negations: `diagnostic_`, `bootstrap_`, `snr_`, `perturbation_`).
- Key numbers used in any decision must appear in the summary `.md`, not only in the log — verdict-relevant statistics buried in logs stalled a review once.
- Auto-generated reports are never hand-edited; regenerate them from their script.

## Experiment index

- Each experiment directory keeps an `experiment_index.md` as the single master index: one row per file with a one-line description, plus a current "What's still open?" section.
- A new file gets its index row in the same change that creates it; a status change (run closed, item reopened) sweeps the index in the same change.

## Thesis chapter discipline

- The chapter lives in the experiment's `thesis/` directory; treat it as a first-class artifact, edited carefully, never casually.
- Every quantitative claim must trace to a repo artifact; keep the claim-to-artifact appendix current — no number enters the chapter without a file it can be checked against.
- The `.md` and `.tex` versions are one document in two formats: an edit to one is incomplete until mirrored in the other, in the same session.
- When an experiment item changes status (closed / reopened / provisional), the chapter's affected sections (threats-to-validity, future work, verification tables) update in the same sweep as the record files.

## Training runs: periodic checkpoints

- Training runs must save epoch-N snapshots (e.g. every 10 epochs) in addition to `best`/`final` weights — instrument calibration against early training was blocked once because only best/final existed. Hook: `_build_callbacks` in `src/gwml/training/train.py`; land this with the next training campaign.

## Diagnostics: positive-control rule

- Every new diagnostic instrument must include a head known to work (e.g. `mchirp`, R² ≈ 0.96) as a positive control, and its verdicts on the heads under test are untrusted until the control reads correctly at the same stage/configuration.

## Frozen vs living docs

- Preregistrations and dated historical snapshots are never rewritten — dated addenda only; pre-committed criteria are never touched after results exist.
- Living docs (`NOTES.md`, `diagnostic_log.md`, `experiment_index.md`) are updated in place; a status change is propagated to every doc that states it, in one sweep.
- Exceptions are reported per-case, never averaged away: "3/4 resolved, 1/4 persists," not "mostly resolved."

## Compute & git ground rules

- Never run training/eval/GPU scripts locally — this machine (T530) is CPU-only; prepare scripts for the user to run on the lab GPU machine and wait for their output.
- Scripts that iterate multiple models on GPU must chunk large forward passes (batched predict) and free each model before the next (`del` + `tf.keras.backend.clear_session()` + `gc.collect()`) — four accumulated models once exhausted a 22 GB pool.
- Ask before committing; never stage the untracked `deeplearning_models/` directory.
