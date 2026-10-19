# Handoff

Written 2026-10-19T10:36Z; rewritten 2026-10-19T22:14Z (the earlier version had grown by appending and contradicted itself).

## Read first

| File | What it tells you |
|------|-------------------|
| `CLAUDE.md` | the working rules, the machines table (hosts and paths), the related repositories and their cautions, the commit-message format |
| `HANDOFF.md` (this file) | where the work stands, the dated decisions, what is open, what is next |
| `BUGS.md` | every defect and observation (`BUG-001` to `009`, `OBS-001` to `024`), with status |
| `VERIFY.md` | checks that still need a run, who closes them, and what would falsify them |
| `GENERATOR_OVERHAUL.md` | what the current data generator lacks (G-01 to G-24), the target output file, and decisions D-01 to D-12 |
| `SEEDING.md` | this repository's seeding scheme for training and tests (its own master seed) |
| `../ml-gw-search/HANDOFF.md` and `../ml-gw-search/SEEDING.md` | the generator repository's state, and its separate seeding scheme and master seed |
| `PHIC_PSI_CARRYOVER.md` | frozen snapshot of what the φc/ψ redo established (source: `experiments/phic_psi_poc-redo/qvalue_carryover.md` on its branch) |
| `experiments/q_recovery/` | recap of the 2026-07-13 to 07-15 q investigation and its open items |

## Environment

- At least three machines work on this repository; their hosts and working-tree paths are in the "Machines" table in `CLAUDE.md`.
- Last session: `thinkpad-t530`, working tree `/home/syedalimohsinbukhari/PycharmProjects/ml_practice_v2`, `.venv` active, 8 CPUs, no GPU; training, evaluation, and data generation do not run here.
- Earlier session: `sarlws5-Precision-7820-Tower` (`/mnt/DRIVE1/PycharmProjects/ml_practice_v2`, Quadro P620, 40 CPUs).
- In scope beside this tree on `thinkpad-t530`: `/home/syedalimohsinbukhari/PycharmProjects/ml-gw-search` (data generation, branch `dev`) and `/home/syedalimohsinbukhari/PycharmProjects/gw_research_paper` (the Overleaf paper repository, branch `main`).
- The third machine's host and path, and which machine is the "lab GPU machine", are still unrecorded.
- Re-derive host and path each session rather than trusting this; record the path of any machine not yet in the table.

## Where we stand

**The q_value project.**
- The branch is `q_value`; the φc/ψ experiment is concluding on `poc/phic-psi-degeneracy-redo`.
- All q results are from 2026-07-14 runs, before the carryover's findings, and none has been reproduced (OBS-006).
- The conventions in `CLAUDE.md` were rebuilt from `WORKING_CONVENTIONS.md`; the φc/ψ-era version is preserved in `CLAUDE_phic_psi_backup.md`.
- `VERIFY.md` has 8 `PENDING` checks; the test suite has no recorded run (OBS-013).

**The first task: regenerate the dataset with the `ml-gw-search` generator.**
- The generator is part of a published paper, and its backend (`gen.py`, `whiten`, `regularize_psd`) is frozen; all new work wraps it (`GENERATOR_OVERHAUL.md` section 0).
- The target dataset: `IMRPhenomD`, spinless, legacy priors, Gaussian aLIGO design-PSD noise, H1 and L1, 4096-sample windows at 2048 Hz.
- Counts: training 100,000, validation 20,000, test 10,000, half of each noise-only, in 13 shards of 10,000, then a merge step.
- Layout: the injection rows keep `h1`, `l1`, `params` so the loader here works unchanged, and the noise-only rows go in `noise_h1`, `noise_l1`; the SNR is baked per sample in (7, 15), and the strain plus the unit-SNR waveform are stored.
- Nothing is implemented yet: no generator code has been written, and the generator itself is unchanged except for the `PARAM_COLUMNS` label list in `train.py`.

**Seeding.**
- Each repository has its own, unrelated master seed and its own `SEEDING.md`; no code implements either scheme yet.
- Generator: `3021456217`, key `(dataset_id, split, shard, purpose)`, one process per shard, pycbc's global state seeded once per shard.
- This repository: `1153279169`, training key `(0, seed_index, purpose)` and test key `(1, name_key, purpose)`; `seed_index` replaces the `42` literals (OBS-023, OBS-024).

**The head-inclusion rule (OBS-008).**
- Decided: M = 10 seeds, N = 8, K = all five trunks.
- The per-head floors were left to Claude and are proposals awaiting the user's confirmation.
- K = all conflicts with BUG-001 (see "Open decisions").

## Decisions, dated (user unless stated)

| When (2026-10-19) | Decision | Recorded in |
|-------------------|----------|-------------|
| 20:24Z | The non-convention markdown files (the four q documents, `PHIC_PSI_CARRYOVER.md`, `experiments/q_recovery/`) are kept as they are | this file |
| 20:30Z | OBS-012 closed: a T530 is in use; machines are tracked in `CLAUDE.md` | `BUGS.md`, `CLAUDE.md` |
| 20:36Z | First task: data regeneration | this file |
| 21:04Z | Legacy priors (D-02); Gaussian aLIGO noise (D-07); SNR baked at generation (D-05) | `GENERATOR_OVERHAUL.md` |
| 21:10Z | `IMRPhenomD` spinless (D-01); the counts and noise-only halves (D-03, D-04); 13 shards plus a merge; a new `re_package_hdf_with_noise.py` with the original unchanged | `GENERATOR_OVERHAUL.md` |
| 21:20Z | The generator internals are frozen; the earlier "configurable priors" (D-02) and "parameterized merger window" (D-08) are superseded: nothing is made configurable | `GENERATOR_OVERHAUL.md` section 0 |
| 21:31Z | Store the strain plus the unit-SNR waveform (the smallest form); the float32 cast of `injection_time` stays and is documented (BUG-009) | `GENERATOR_OVERHAUL.md`, `BUGS.md` |
| 21:36Z | The freeze-and-wrap plan is confirmed (seeded driver, pinned backend hashes); the `PARAM_COLUMNS` list in the sibling's `train.py` is edited (BUG-008); noise-only rows in separate datasets so the loader is unchanged (D-12) | `GENERATOR_OVERHAUL.md` section 7, `BUGS.md` |
| 21:53Z | `dataset_id` is in the generation key (D-09); training seeds derive from the master and the tests are seeded the same way (D-10) | `SEEDING.md` |
| 21:56Z | The config key replacing `train.seed` is `seed_index` | `SEEDING.md` |
| 22:06Z | Each repository has its own master seed; M = 10, N = 8, K = all five trunks; the floors are delegated to Claude | both `SEEDING.md`, `BUGS.md` OBS-008 |

## Open decisions

- **Confirm the OBS-008 floors, and settle K = all against BUG-001:** the InceptionTime `merger_time` head is dead (R² ≈ 0 in all five recorded runs), so that trunk's positive control fails and, under the rule, no head can clear it.
  Options: fix BUG-001 first, lower K to 4, or leave `inception_time` out of the campaign.
- Whether the test split confirms the floors (the rule says "validated recovery", i.e. the validation split).
- Whether a trunk belongs in the training key (proposed: no, for paired comparisons).
- Drop `coa_phase` from the default head set (OBS-007; the carryover recommends it, the configs still list it).
- Restore the periodic-checkpoint rule in `CLAUDE.md` (OBS-011; it exists only in `CLAUDE_phic_psi_backup.md`).
- Close `q_recovery` item 4 as done, or re-run (OBS-002).

## Next

1. Settle the open decisions above, starting with K against BUG-001.
2. Implement the generator wrap-around in `ml-gw-search`, one step at a time with the user's go-ahead: the seeded driver, `re_package_hdf_with_noise.py`, the merge step, backend hash pinning, provenance attributes; then run the generator's determinism check on a shrunken workload on a machine other than `thinkpad-t530`.
3. Implement the training and test seeding here: a `seeding` module, the `seed_index` config key, removal of the `make_dataset` default seed, run metadata (seed index, derived seeds, master seed), and a `conftest.py` fixture; then run the suite on the lab machine (to be queued in `VERIFY.md` with the change).
4. After the first generated file: write the contract check against `heads_spec.py` (this closes BUG-005 and the `VERIFY.md` BUG-008 check), and review the loader's memory use (the strain array grows from about 0.8 GB to about 1.6 GB for the 50,000 injection rows).
5. Work the remaining `PENDING` items in `VERIFY.md`, and re-measure the q results (OBS-006).

## Working tree

- Committed here as `[q_value-minor-5]` to `[q_value-minor-12]`; `git log` shows them (their hashes changed when messages were reworded in the IDE, the labels did not).
- Uncommitted here: none of the tracked files; `[q_value-minor-12]` carries `SEEDING.md` (rewritten for training and tests), `CLAUDE.md`, `BUGS.md`, `GENERATOR_OVERHAUL.md`, `README.md`, and this rewrite.
- Untracked and deliberately not staged: `deeplearning_models/`, `experiments/paper/`, `experiments/phic_psi_poc-redo/`.
- `ml-gw-search`: the `PARAM_COLUMNS` edit is committed on `dev` as `[dev-minor-2]`, unpushed (its only remote is the upstream author's, so nothing there is backed up); the new `SEEDING.md` and `HANDOFF.md` are committed on `dev` as `[dev-minor-3]`, also unpushed, and `re_package_hdf.py`, `mlgwsc-1/gen_py_data_pipeline.md`, `mlgwsc-1/__pycache__/`, and `combined_repackaged.hdf` stay untracked.
- `gw_research_paper` has an uncommitted `main.tex` that these sessions did not touch.
- The user must run `git push --force-with-lease origin q_value` (history was rewritten to fix commit messages); another machine holding the old commits needs `git fetch` then `git reset --hard origin/q_value`.
