# Project conventions (q_value work)

Derived from `WORKING_CONVENTIONS.md`; adopted: §1–5, 9–11, 13–17.
Dropped for now: §6 (generated artifacts), §7 (per-unit notes and index), §8 (frozen vs living docs), §12 (code style) — no need has appeared yet in the q_value work; add each when one does.
Not adopted: the φc/ψ-era rules, which live on branch `poc/phic-psi-degeneracy-redo` and are preserved verbatim in `CLAUDE_phic_psi_backup.md`.
Files in the repo root that predate this file (`PLAN.md`, `q_head_action_plan.md`, `phase2_5_3_*.md`, `q_head_run_comparison.md`), the carryover snapshot `PHIC_PSI_CARRYOVER.md`, `src/gwml/gen_py_data_pipeline.md`, and the `experiments/` directory (including `experiments/q_recovery/`) are not governed by these conventions until triaged.

## Working documents (§1)

- Root-level `BUGS.md`, `HANDOFF.md`, and `VERIFY.md` carry state between sessions; do not leave state only in chat.
- `PLAN.md` is not part of this trio; the existing one predates this file and is left as is.
- Read `HANDOFF.md` first when resuming.

## Timestamps (§2)

- Every dated entry in any working document carries date and time with an offset, ISO 8601, e.g. `2026-10-19T10:36Z`, never a bare date.

## Bug and observation log (§3)

- IDs are project-global and never reused: `BUG-NNN` for a defect, `OBS-NNN` for something surfaced by discussion or judgment.
- Status key: `OPEN`, `FIXED` (changed, not yet tested), `VERIFIED` (re-run or recomputed, not just read), `CLOSED: <reason>`, `DEFERRED`.
- Newest entry at the bottom; the index table at the top changes in the same edit as the entry's status.
- A fixed entry states the resolution first, then keeps the original diagnosis underneath, labeled; a wrong fix is reverted and marked, not deleted.
- Log each finding or fix when it happens, not batched for later (§17).

## Verify queue (§4)

- Status key: `PENDING`, `CONFIRMED`, `FAILED`, `STALE`.
- Each entry has what changed, what to check, and why it matters; it cites the originating bug ID and names who or what closes it (e.g. "user, on the lab GPU machine").
- An unconfirmed entry is an open question, not a passed check.
- Training runs, full evaluations, and long outputs are deferred to this queue rather than pulled into context.

## Verify, don't assume (§5, §16)

- A clean build or exit code 0 shows the output is well-formed, not that its content is correct; inspect the output.
- Judge any training metric over the full `history.csv` trajectory, never the endpoint alone.
- A new diagnostic is untrusted until it reads a known-good case correctly.
- State what result would falsify a check before running it.
- A fix gets a regression test where a suite exists for that code; otherwise say it is covered only by an ad-hoc check.
- An ambiguous or surprising result is a decision point for the user, not something to resolve quietly.
- Label non-trivial claims **verified** (name the check), **inferred**, **assumed** (state the reasoning), or **unknown**.

## Scope, subagents, destructive operations (§9, §10)

- Work stays inside this machine's working tree and the related repositories listed under "Related repositories" below; both are in scope and not out of jurisdiction.
- Root searches at a specific subdirectory, never a parent or home.
- Check `hostname` and `pwd` at session start and use the matching row; a host that is not in the table is a decision point for the user, not something to guess.
- Do not spawn subagents by default; when one is spawned, restate this boundary in its prompt and verify its actual diffs.
- Confirm with the user before: `rm` outside the working tree or `rm -rf` without a listed path, overwriting a non-generated file wholesale, force-push or history rewrite on a pushed branch, branch or tag deletion, and mass renames or refactors beyond the current change.
- Test: if undoing an operation needs anything other than `git checkout` or a script re-run, confirm first.

## Git (§11)

- Ask before committing; a yes covers that commit only.
- Never commit secrets; check file contents, not names.
- Run `git status` after any broad `git add`.
- Never stage the untracked `deeplearning_models/` directory.
- Amend only unpushed commits.
- A commit that addresses a logged item cites its ID inside a bullet, e.g. `(BUG-042)`, never in the subject.
- Keep the required attribution trailer on every commit.
- Commit message format, from `[q_value-minor-11]` and `[dev-minor-2]` on: the subject is the bare `[<branch>-minor-N]` label, then a blank line, then one bullet per change, then a blank line and the trailers.
- Each bullet is one sentence on a single physical line, with no hard wrapping and no continuation lines.
- The same format applies to commits in the related repositories (`ml-gw-search` uses its own `[dev-minor-N]` numbering); earlier messages keep their older wrapped layout and are not rewritten.
- Check the message with `git show -s --format=%B` after committing, because an IDE reword can change the hash (not the label) of an unpushed commit.

## Prose (§13)

- Prose files (`.md`, `.tex`) use one sentence per line; reflow a paragraph you touch.
- Tables, code blocks, and headings keep their own layout.
- Captions and status notes describe; interpretation belongs in the surrounding narrative.

## Determinism (§14)

- One named master-seed constant; derive per-run seeds from it and thread one generator through downstream calls.
- Never re-seed inside a loop over runs meant to be independent.
- To check determinism, temporarily shrink the workload and revert it before finishing.

## Session start and end (§15, §17)

- On resume: read `HANDOFF.md`, re-derive the environment (hostname, working directory, virtualenv, GPU presence), run `git log` since the handoff's timestamp, then open changed `BUGS.md` entries and `PENDING` `VERIFY.md` entries.
- After each major unit of work and at session end: update `HANDOFF.md` (landed, mid-flight, next, and the working-tree state as "committed as `<sha>`" or "uncommitted: `<files>`"), sweep `VERIFY.md`, update `BUGS.md` statuses.

## Compute (carried over from the φc/ψ era; verify before relying)

- Do not run training, evaluation, or GPU scripts without the user's go-ahead; prepare them for the lab GPU machine and wait for the output.
- Scripts that loop over models on GPU chunk large forward passes and free each model before the next (`del`, `tf.keras.backend.clear_session()`, `gc.collect()`).
- Earlier notes call "the local machine" a "T530, CPU-only"; that was true of only one of several machines, so read the "Machines" table instead of relying on it.

## Machines

The repo is worked on from more than one machine, each with its own checkout path.
Paths are absolute; the `.venv` sits inside the working tree.

| Host | Working tree | GPU | Role | Verified |
|------|--------------|-----|------|----------|
| `thinkpad-t530` | `/home/syedalimohsinbukhari/PycharmProjects/ml_practice_v2` | none (`nvidia-smi` absent), 8 CPUs | edits, docs, code-only work; no training | 2026-10-19T20:27Z |
| `sarlws5-Precision-7820-Tower` | `/mnt/DRIVE1/PycharmProjects/ml_practice_v2` | Quadro P620, 40 CPUs | workstation | 2026-10-19T10:36Z |
| unrecorded | unrecorded | unrecorded | third machine | — |

- Add a row (host, path, GPU, role, verification time) the first time a session runs on a new machine.
- The lab GPU machine named in `VERIFY.md` entries is not yet tied to a row; record which host it is.

## Related repositories

Two sibling repositories are part of this work and are in scope for reading and editing.
They are separate git repositories: a commit in one never covers the others, and the §11 rule to ask before committing applies to each.

| Repository | Path on `thinkpad-t530` | Branch | Role | Verified |
|------------|-------------------------|--------|------|----------|
| `ml-gw-search` | `/home/syedalimohsinbukhari/PycharmProjects/ml-gw-search` | `dev` | data generation (`mlgwsc-1/gen.py`, `re_package_hdf.py`); has its own `.venv` with pycbc | 2026-10-19T20:36Z |
| `gw_research_paper` | `/home/syedalimohsinbukhari/PycharmProjects/gw_research_paper` | `main` | the research paper, an Overleaf git repo (LaTeX) | 2026-10-19T20:36Z |

- Paths for these two on the other machines are unrecorded; add them to this table when a session runs there.
- `ml-gw-search` has its own `HANDOFF.md` and `SEEDING.md` (with its own master seed, separate from this repository's); read them before working there.
- `ml-gw-search`:
  - `origin` is the upstream author's GitHub repository, so never push to it.
  - It holds the 1.9 GB `combined_repackaged.hdf`, which is untracked; read its structure with `h5py` metadata only and never load it wholesale.
  - Generation is CPU-heavy pycbc work; do not run it without the user's go-ahead, and not on `thinkpad-t530`.
- `gw_research_paper`:
  - `origin` is Overleaf, so a push publishes to the paper's Overleaf project; never push without the user's explicit go-ahead.
  - Check `git status` before editing, because `main.tex` may carry the user's uncommitted edits; do not overwrite or stage them.
  - Edit `.tex` and `.bib` sources only, never the build products (`main.pdf`, `*.aux`, `*.bbl`, and similar, which `.gitignore` excludes).
