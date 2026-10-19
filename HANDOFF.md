# Handoff

Written 2026-10-19T10:36Z; last extended 2026-10-19T11:30Z.

## Environment

- Host `sarlws5-Precision-7820-Tower`, 40 logical CPUs, Quadro P620 present (`nvidia-smi -L`).
- Branch `q_value`, based on `0ace363`.
- Re-derive this each session rather than trusting it.

## Current state

- New q_value work starts here; the φc/ψ experiment is concluding on branch `poc/phic-psi-degeneracy-redo`.
- `CLAUDE.md` was rebuilt from `WORKING_CONVENTIONS.md`; the previous φc/ψ-era version is preserved verbatim in `CLAUDE_phic_psi_backup.md`.
- `BUGS.md` and `VERIFY.md` are still empty skeletons.
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
- Whether the "T530, CPU-only" statements in `PLAN.md`, `README.md`, and `docs/pipeline/testing.md` still hold on another machine.

## Next

- Turn the open items in `experiments/q_recovery/experiment_index.md` and the carryover into `BUGS.md` and `VERIFY.md` entries.
- Decide the first q_value task and log it here.

## Working tree

- Committed as `[q_value-minor-5]` (the convention files) and `[q_value-minor-6]` (the doc update plus this refresh), on top of `0ace363`; `git log -2` shows them.
