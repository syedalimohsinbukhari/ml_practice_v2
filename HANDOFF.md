# Handoff

Written 2026-10-19T10:36Z; last extended 2026-10-19T20:41Z.

## Environment

- At least three machines work on this repo; their hosts and working-tree paths are in the "Machines" table in `CLAUDE.md`.
- Last session ran on `thinkpad-t530` (2026-10-19T20:27Z), working tree `/home/syedalimohsinbukhari/PycharmProjects/ml_practice_v2`, `.venv` active, 8 CPUs, no GPU.
- The earlier 2026-10-19T10:36Z session ran on `sarlws5-Precision-7820-Tower` (`/mnt/DRIVE1/PycharmProjects/ml_practice_v2`, Quadro P620, 40 CPUs).
- In scope alongside this tree on `thinkpad-t530`: `/home/syedalimohsinbukhari/PycharmProjects/ml-gw-search` (data generation, branch `dev`) and `/home/syedalimohsinbukhari/PycharmProjects/gw_research_paper` (the Overleaf paper repo, branch `main`); see "Related repositories" in `CLAUDE.md`.
- Branch `q_value`; the commit before the latest is `d61fd66` (`[q_value-minor-8]`), and `git log -2` shows the current pair.
- Re-derive host and path each session rather than trusting this; record the path of any machine not yet in the table.

## Current state

- New q_value work starts here; the φc/ψ experiment is concluding on branch `poc/phic-psi-degeneracy-redo`.
- `CLAUDE.md` was rebuilt from `WORKING_CONVENTIONS.md`; the previous φc/ψ-era version is preserved verbatim in `CLAUDE_phic_psi_backup.md`.
- `BUGS.md` holds 9 bugs and 17 observations: BUG-001 to 006 and OBS-001 to 013 opened 2026-10-19T11:37Z from the `q_recovery` open items, the carryover, and the doc reconciliation; BUG-007 to 009 and OBS-014 to 017 opened 2026-10-19T20:40Z from the read of the `ml-gw-search` generator.
  `VERIFY.md` holds 7 `PENDING` checks keyed by the first group; the generator items have no `VERIFY.md` entries yet, because none has a fix to check.
- Commits `f50dfae` (`PeriodicCheckpoint`, epoch-N snapshots every `train.checkpoint_every_n` epochs, default 10) and `0ace363` (carryover from the φc/ψ branch) landed from the other machine.
  The convention commit (`[q_value-minor-5]`) was rebased on top of them and its message numbering fixed.
- `PHIC_PSI_CARRYOVER.md` is a frozen snapshot of what the φc/ψ redo established; its source of truth is `experiments/phic_psi_poc-redo/qvalue_carryover.md` on the φc/ψ branch.
- `experiments/q_recovery/` (written for this branch, 2026-07-27) recaps the 2026-07-13 to 07-15 q investigation and lists its open items.
- The non-convention docs (`PLAN.md`, `README.md`, `docs/`, the four q documents) were reconciled with the code on 2026-10-19; the four q documents carry a dated status addendum and their original text is unchanged.
  The pipeline docs also describe the periodic checkpoints and a `std_ratio` caveat for two-component heads.
- All q results are from 2026-07-14 runs, before the carryover's findings; they have not been re-measured.

## Open decisions

- Drop `coa_phase` from the default head set (carryover §2 recommends it; configs still list it).
- Restore the periodic-checkpoint rule in `CLAUDE.md` (it exists only in `CLAUDE_phic_psi_backup.md`).
- The seed count and materiality floor for the head-inclusion rule are the author's to fix (carryover §3).

## Decided

- 2026-10-19T20:30Z: OBS-012 closed; a T530 is in use, and the three machines are tracked by host and path in the "Machines" table in `CLAUDE.md`.
  The third machine's host and path, and which machine is the "lab GPU machine", are still unrecorded.

- 2026-10-19T20:24Z: the non-convention md files (the four q documents, `PHIC_PSI_CARRYOVER.md`, `experiments/q_recovery/`) are kept as they are; no archive or deletion.

## Next

- First task, per the user (2026-10-19T20:36Z): data regeneration with the `ml-gw-search` generator.
  The generator review is logged as BUG-007 to 009 and OBS-014 to 017 (unseeded generation, mislabeled `PARAM_COLUMNS`, float32 `injection_time`, noise-only half dropped by repackaging, real-noise whitening mismatch, unrecorded generation arguments, two diverging pipeline notes).
  Settle with the user what the regeneration should change (parameters, noise source, sample counts, seed handling) before any generation is run; generation must not run on `thinkpad-t530`.

- Work the `PENDING` items in `VERIFY.md`; the decisions that need you (OBS-002, 007, 008, 011) are listed under "Open decisions" above and in `BUGS.md`.
- Decide the first q_value task and log it here.

## Working tree

- Committed as `[q_value-minor-5]` (the convention files), `[q_value-minor-6]` (the doc update plus the earlier refresh) and `[q_value-minor-7]` (`BUGS.md`, `VERIFY.md`, `HANDOFF.md`: the q_recovery and carryover items logged), on top of `0ace363`; `git log -3` shows them.
- Committed as `[q_value-minor-8]` (`CLAUDE.md` "Machines" table, `BUGS.md` OBS-012 closed, the earlier refresh).
- Committed as `[q_value-minor-9]` (`CLAUDE.md` "Related repositories" section, `BUGS.md` generator findings, this refresh).
- Uncommitted: none.
- The sibling repos have their own working-tree state, untouched by this session: `ml-gw-search` has untracked `re_package_hdf.py`, `mlgwsc-1/gen_py_data_pipeline.md`, `mlgwsc-1/__pycache__/` and `combined_repackaged.hdf`; `gw_research_paper` has an uncommitted `main.tex`.
- The user must run `git push --force-with-lease origin q_value` (history was rewritten to fix commit messages); another machine holding the old commits needs `git fetch` then `git reset --hard origin/q_value`.
