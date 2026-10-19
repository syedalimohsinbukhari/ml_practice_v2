# Handoff

Written 2026-10-19T10:36Z; last extended 2026-10-19T10:36Z.

## Environment

- Host `sarlws5-Precision-7820-Tower`, 40 logical CPUs, Quadro P620 present (`nvidia-smi -L`).
- Branch `q_value`, HEAD `21552e2`.
- Re-derive this each session rather than trusting it.

## Current state

- New q_value work starts here; the φc/ψ experiment is concluding on branch `poc/phic-psi-degeneracy-redo`.
- `CLAUDE.md` was rebuilt from `WORKING_CONVENTIONS.md`; the previous φc/ψ-era version is preserved verbatim in `CLAUDE_phic_psi_backup.md`.
- `BUGS.md` and `VERIFY.md` are empty skeletons.
- Existing root files (`PLAN.md`, `q_head_action_plan.md`, `q_head_run_comparison.md`, `phase2_5_3_*.md`) and `experiments/` are untouched and not yet triaged against the new conventions.

## Next

- Triage the pre-existing root `.md` files and `experiments/` (keep, absorb into the new docs, or archive).
- Decide the first q_value task and log it here.

## Working tree

- Committed as `[q_value-minor-5]` (the `git log -1` subject after HEAD `21552e2`): `CLAUDE.md` (rewritten), `CLAUDE_phic_psi_backup.md`, `BUGS.md`, `HANDOFF.md`, `VERIFY.md`.
- `.mcp.json` was already untracked before this session.
