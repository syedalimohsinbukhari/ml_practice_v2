# Handoff

Written 2026-10-19T10:36Z; rewritten 2026-10-19T22:14Z (the earlier version had grown by appending and contradicted itself); environment, generator status, decisions, next steps, and working tree updated 2026-10-22T06:52Z; generator status, `VERIFY.md` count, and next steps updated 2026-10-24T21:55Z.

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
- Last session (2026-10-22): `sarlws5-Precision-7820-Tower`, working tree `/mnt/DRIVE1/PycharmProjects/ml_practice_v2` (Quadro P620, 40 CPUs); the sibling is at `/mnt/DRIVE1/PycharmProjects/ml-gw-search`, with its own `.venv` (Python 3.10, pycbc 2.9.0, numpy 2.2.6).
- Earlier session (2026-10-19): `thinkpad-t530`, working tree `/home/syedalimohsinbukhari/PycharmProjects/ml_practice_v2`, 8 CPUs, no GPU; training, evaluation, and data generation do not run there.
- In scope beside this tree on `thinkpad-t530`: `/home/syedalimohsinbukhari/PycharmProjects/ml-gw-search` (data generation, branch `dev`) and `/home/syedalimohsinbukhari/PycharmProjects/gw_research_paper` (the Overleaf paper repository, branch `main`).
- The third machine's host and path, and which machine is the "lab GPU machine", are still unrecorded.
- Re-derive host and path each session rather than trusting this; record the path of any machine not yet in the table.

## Where we stand

**The q_value project.**
- The branch is `q_value`; the φc/ψ experiment is concluding on `poc/phic-psi-degeneracy-redo`.
- All q results are from 2026-07-14 runs, before the carryover's findings, and none has been reproduced (OBS-006).
- The conventions in `CLAUDE.md` were rebuilt from `WORKING_CONVENTIONS.md`; the φc/ψ-era version is preserved in `CLAUDE_phic_psi_backup.md`.
- `VERIFY.md` has 7 `PENDING` checks (OBS-025 is `CONFIRMED`, 2026-10-24T21:55Z); the test suite has no recorded run (OBS-013).

**The first task: regenerate the dataset with the `ml-gw-search` generator.**
- The generator is part of a published paper, and its backend (`gen.py`, `whiten`, `regularize_psd`) is frozen; all new work wraps it (`GENERATOR_OVERHAUL.md` section 0).
- The target dataset: `IMRPhenomD`, spinless, legacy priors, Gaussian aLIGO design-PSD noise, H1 and L1, 4096-sample windows at 2048 Hz.
- Counts: training 100,000, validation 20,000, test 10,000, half of each noise-only, in 13 shards of 10,000, then a merge step.
- Layout: the injection rows keep `h1`, `l1`, `params` so the loader here works unchanged, and the noise-only rows go in `noise_h1`, `noise_l1`; the SNR is baked per sample in (7, 15), and the strain plus the unit-SNR waveform are stored.
- Landed in `ml-gw-search` (2026-10-22; the backend files are unchanged): `[dev-minor-6]` the seeded shard driver `mlgwsc-1/gen_seeded.py` and the determinism check `mlgwsc-1/check_determinism.py` (steps 1 to 4 of `SEEDING.md`); `[dev-minor-7]` `re_package_hdf_with_noise.py`; `[dev-minor-8]` `merge_hdf_shards.py`. Details and what was checked are in the sibling's `HANDOFF.md`.
- Determinism: 9 of 9 checks passed in three runs at 10 injections + 5 noise-only per shard, after pinning pycbc's `numpy` FFT backend; under the FFTW default the waveforms differed by about 1e-13 in a few elements between processes (OBS-025). The check at shard scale passed for one shard (below).
- Checked only on a 10-injection shard and on synthetic shards copied from it (repackaging and merge); nothing at full size.
- Landed here (uncommitted): the contract check, `src/gwml/contract.py` (imports only `gwml.heads_spec`), `scripts/check_contract.py FILE [--full] [--max-rows N]`, and `tests/test_contract.py` (42 tests, all passed on this machine; only that file was run, not the suite).
  Known-good: a real repackaged shard (40 of 40) and a merged file of real data (44 of 44) pass; the legacy `combined_repackaged.hdf` fails, as a known-bad control; `--full` fails a small shard on the D-03 counts.
  Not yet run on a full-size merged file, and `structure.md` is still not written (BUG-005 stays `OPEN`).
- Determinism check, all six steps (2026-10-22T07:26Z): 17 of 17 passed in three runs on real small shards from separate processes; the contract check passed 68 of 68 on a merged file of them.
- Determinism check, step 7 added (2026-10-25 02:23 +05:00): the lag-aware cross-correlation of all noise series across H1 and L1, injected and noise-only rows, three shards, and two splits passed with a shifted-copy positive control; 19 of 19 checks passed (steps 1 to 7).
- One full-size shard has been generated (2026-10-25 02:29 +05:00, training shard 0, 5,000 injections + 5,000 noise-only, real 202.23 s, 492 MB), and a second generation in a fresh process is bit-identical in `noises`, `waveforms`, and `waveform_params` (OBS-025 `VERIFIED`, `VERIFY.md` `CONFIRMED`).
  The files are in `/mnt/DRIVE1/gen_out/fullsize_test/` and `/mnt/DRIVE1/gen_out/fullsize_repeat/`, outside both repositories.
- The full-size shard 0 was repackaged (2026-10-24T22:38Z, 4.80 s, 1.97 GB peak): strain = noise + snr × unit waveform to 7.6e-06, SNR in (7.002, 14.999), noise-only rows and parameters equal to the source, and `scripts/check_contract.py` passed 40 of 40 (details in `../ml-gw-search/HANDOFF.md`).
- All 13 shards were generated (2026-10-24T23:01Z, seven processes at a time, 3:24 each, 2.65 to 2.97 GB peak each, into `/mnt/DRIVE1/gen_out/raw/`) and repackaged (63.3 s in all, into `/mnt/DRIVE1/gen_out/repackaged/`, 6.0 GB): 0 duplicated noise rows among 260,000, the lag-aware cross-correlation across the shards peaked at 0.088, strain = noise + snr × unit waveform to 7.6e-06 in every file, the SNR over 65,000 rows spans 7.000 to 15.000, and `scripts/check_contract.py` passed 40 of 40 on each file (details in `../ml-gw-search/HANDOFF.md`).
  Not yet done: the full merge (about 6 GB; time and memory unknown) and `scripts/check_contract.py --full` on the merged file.

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

| When (2026-10-22) | Decision | Recorded in |
|-------------------|----------|-------------|
| 05:42Z to 05:47Z | The fork `syedalimohsinbukhari/ml-gw-search` becomes `origin` of the sibling (`dev` and `main` track it), the author's repository becomes the fetch-only `upstream`, and the diverged `dev` histories are merged | `CLAUDE.md`, sibling `HANDOFF.md` |
| 06:02Z | Pin pycbc's `numpy` FFT backend in the driver, because of the FFTW difference of about 1e-13 between processes | `BUGS.md` OBS-025 |

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
2. Finish the generator wrap-around in `ml-gw-search` with the user's go-ahead: merge the 13 repackaged shards (generation and repackaging are done) and check the merged file.
3. Implement the training and test seeding here: a `seeding` module, the `seed_index` config key, removal of the `make_dataset` default seed, run metadata (seed index, derived seeds, master seed), and a `conftest.py` fixture; then run the suite on the lab machine (to be queued in `VERIFY.md` with the change).
4. After the first full merged file: run `scripts/check_contract.py --full` on it, decide whether to write `structure.md` or drop its stale references (BUG-005), and review the loader's memory use (the strain array grows from about 0.8 GB to about 1.6 GB for the 50,000 injection rows).
5. Work the remaining `PENDING` items in `VERIFY.md`, and re-measure the q results (OBS-006).

## Working tree

- Committed here as `[q_value-minor-5]` to `[q_value-minor-14]`; `git log` shows them (hashes changed when messages were reworded in the IDE, the labels did not). `[q_value-minor-13]` is the `CLAUDE.md` remote-layout update; `[q_value-minor-14]` carries `BUGS.md` (OBS-025, BUG-008 `VERIFIED`), `VERIFY.md`, and the previous version of this file.
- Uncommitted here: this file's 2026-10-22T06:52Z update, and whatever the contract check adds (listed at the end of the session).
- Untracked and deliberately not staged: `deeplearning_models/`, `experiments/paper/`, `experiments/phic_psi_poc-redo/`.
- `ml-gw-search` (on `sarlws5-Precision-7820-Tower`): `dev` tracks `origin` (the user's fork); `[dev-minor-11]` and earlier are pushed, `[dev-minor-12]` (the `verbose` pass-through in `gen_seeded.py` and the full-size results in its `HANDOFF.md`) is committed and not yet pushed; `upstream` is the author's repository, fetch-only. The history there is described in its own `HANDOFF.md`.
- `gw_research_paper` has an uncommitted `main.tex` that these sessions did not touch.
- The user must run `git push --force-with-lease origin q_value` (history was rewritten to fix commit messages); another machine holding the old commits needs `git fetch` then `git reset --hard origin/q_value`.
