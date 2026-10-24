# Verify queue

Status key: `PENDING`, `CONFIRMED`, `FAILED`, `STALE`.
Entries are keyed by the `BUGS.md` ID they check (no separate numbering).
Each entry has three parts (what changed, what to check, why it matters), plus who or what closes it and what result would falsify the expectation.
Training runs and full test output are deferred here so they are not pulled into the conversation.
Format rules are in `CLAUDE.md` and `WORKING_CONVENTIONS.md` §4.

The first seven entries were opened on 2026-10-19T11:37Z; later entries carry their own date.
Triage: an entry untouched after three sessions goes to the top of the next handoff.

| ID | Check | Closer | Status |
|----|-------|--------|--------|
| OBS-013 | Run `pytest -m "not slow"`, then `pytest` | user, lab GPU machine | PENDING |
| BUG-006 | `PeriodicCheckpoint` writes `checkpoints/epoch_NNN.weights.h5` | user, lab GPU machine | PENDING |
| OBS-006 | Reproduce the 2026-07-14 q numbers on current code, with positive controls | user, next q training campaign on the lab GPU machine | PENDING |
| BUG-001 | InceptionTime `merger_time` epoch-0 logit check | user, lab GPU machine | PENDING |
| BUG-002 | `resnet1d` follow-up run with q-head regularization and `q: 1.0` clamp | user, lab GPU machine (config change first) | PENDING |
| OBS-001 | `cnn_attention` clamp-only ablation | user, lab GPU machine | PENDING |
| OBS-009 | `sky_position` with/without test across seeds | user, lab GPU machine | PENDING |
| OBS-025 | A full shard (5,000 injections + 5,000 noise-only) generated twice in fresh processes with the pinned `numpy` FFT backend is identical in `noises`, `waveforms`, `waveform_params` | user, workstation or lab machine, with the user's go-ahead | CONFIRMED |
| BUG-005 | `scripts/check_contract.py --full` passes on the first full merged file | user or agent, on the first full merged file | PENDING |
| BUG-008 | `columns` attribute of a generated file equals the first nine `heads_spec` names | user or agent, on the first generated file, off `thinkpad-t530` | CONFIRMED |

## Entries

### OBS-013 — Test suite

Status: `PENDING`. Closer: user, on the lab GPU machine.

- **What changed:** since the last (unrecorded) suite state, the vMF head, linear periodic heads, `PeriodicCheckpoint`, and the doc-reconciled configs landed.
- **What to check:** `pytest -m "not slow"` passes; then `pytest` (includes the slow overfit-one-batch test per trunk).
  Record the pass and fail counts and any failing test names here.
- **Falsified by:** any failure, especially in `test_heads_spec.py`, `test_losses.py`, or `test_run_dirs.py`.
- **Why:** a failure after the head-set and loss changes would mean documented behavior no longer matches code; the slow overfit tests are the only ones that catch sign and scale bugs in losses.
- **Lighter check the agent can run itself:** none here (the tests need the TensorFlow stack and the HDF5; this was not attempted).

### BUG-006 — `PeriodicCheckpoint`

Status: `PENDING`. Closer: user, on the lab GPU machine.

- **What changed:** commit `f50dfae` added the callback; no test and no smoke coverage exist.
- **What to check:** run the smoke config with `train.checkpoint_every_n: 1` (the default of 10 never fires in a 2-epoch run) and confirm `runs/smoke/<timestamp>/checkpoints/epoch_001.weights.h5` and `epoch_002.weights.h5` exist and load.
  Separately, confirm that a normal run with the default writes `epoch_010.weights.h5`, `epoch_020.weights.h5`, and so on.
- **Falsified by:** a missing file, a zero-byte file, or `load_weights` failing on a snapshot.
- **Why:** these snapshots exist so that instrument calibration against early training is possible; a silent failure would only be noticed when such a calibration is attempted.
- **Suggested follow-up:** a unit test in `tests/test_callbacks.py` and `checkpoint_every_n: 1` in `smoke.yaml` (not done).

### OBS-006 — Reproducing the q numbers

Status: `PENDING`. Closer: user, next q training campaign on the lab GPU machine.

- **What changed:** after the 2026-07-14 runs, periodic heads became linear, `sky_position` and the vMF loss were added, and the default head set changed; the q-specific config settings were removed.
- **What to check:** with a config that includes `q` and the Phase 2/2.5/3 settings (clamp, q-head regularization, warmup where used), re-run at least `tcn` and `cnn_baseline`.
  Compare final-epoch `val_r2_q` and the full `history.csv` trajectory against the recorded 0.260 and 0.151, and read `mchirp`, `merger_time`, and `snr` as positive controls in the same run.
- **Falsified by:** `val_r2_q` outside the run-to-run spread of the old numbers (the spread itself is unknown, since only one seed was run), or any positive control reading well below its recorded range.
- **Why:** every q conclusion in this repo rests on pre-change runs; if they do not reproduce, the q documents describe a different model.
- **Note:** use the seed discipline in `CLAUDE.md` (separate seed streams; record seeds), because the old runs used seed 42.

### BUG-001 — InceptionTime `merger_time`

Status: `PENDING`. Closer: user, on the lab GPU machine.

- **What changed:** nothing yet; this is the diagnostic that was never run.
- **What to check:** an epoch-0 pre-sigmoid logit histogram for `merger_time` on InceptionTime (the `logits_train_vs_val.png` protocol from `scripts/evaluate.py`, but against an initial or epoch-1 checkpoint, which `PeriodicCheckpoint` now makes available), compared with `tcn`.
  If the logits are saturated from the start, try warmup; if not, look at whether the parallel-kernel blocks can encode merger time.
- **Falsified by:** healthy-looking, spread-out logits at epoch 0 (the dead-sigmoid explanation fails).
- **Why:** a head that cannot fit its training data in five of five runs is a defect, not noise.

### BUG-002 — `resnet1d` follow-up

Status: `PENDING`. Closer: user, on the lab GPU machine, after a config change.

- **What changed:** nothing yet; `configs/resnet1d.yaml` has no q head, so the follow-up needs `q` in `model.heads`, `head_cfg.per_head.q: {hidden_units: 32, dropout: 0.3, l2: 1.0e-4}`, and `loss.log_var_clamp: {default: 3.0, q: 1.0}`, with `warmup_epochs: 5` kept.
- **What to check:** `val_r2_q` over the whole trajectory (not the endpoint), `val_std_ratio_q` staying above 0, and the train/val gap compared with 0.999 / 0.100.
- **Falsified by:** `val_std_ratio_q` returning to 0 (the tighter clamp re-killing the sigmoid) or a gap still near 0.9.
- **Why:** the earlier attempt to fix overfitting with the tighter clamp had starved this trunk's q head, so the combination is not known to be safe.

### OBS-001 — Clamp-only ablation on `cnn_attention`

Status: `PENDING`. Closer: user, on the lab GPU machine.

- **What changed:** nothing yet.
- **What to check:** `cnn_attention` with `log_var_clamp: {default: 3.0, q: 1.0}` and no q-head regularization; compare `val_r2_q` trajectory with the 0.204 run (`20260714_113745`).
- **Falsified by:** a clamp-only result well below 0.204 (then the regularization carried a real share of the gain).
- **Why:** it settles which of the two fixes did the work before further stacking.

### OBS-009 — `sky_position` with and without

Status: `PENDING`. Closer: user, on the lab GPU machine.

- **What changed:** nothing yet.
- **What to check:** train with and without `sky_position` on several seeds (the number is open under OBS-008) and compare the three positive controls (`mchirp`, `merger_time`, `snr`).
- **Falsified by:** a control difference larger than the seed-to-seed spread.
- **Why:** it answers whether a weak head harms the shared trunk, and also the carryover's open question about pruned head sets.

### BUG-008 — `columns` attribute of a generated file

Status: `CONFIRMED` (2026-10-22T06:04Z, agent, a 10-injection shard written on `sarlws5-Precision-7820-Tower`). Opened 2026-10-19T21:36Z. Closer: the first generated file, produced on a machine other than `thinkpad-t530` with the user's go-ahead.

- **What changed:** the `PARAM_COLUMNS` list in the sibling's `mlgwsc-1/train.py` (uncommitted in the sibling).
- **What to check:** in a file written by `Dataset.save` (or by the shard driver), `f['training']['waveform_params'].attrs['columns']` equals `mchirp, q, inclination, coa_phase, polarization_angle, declination, ra, injection_time, merger_time`, in that order, and matches the first nine keys of `heads_spec.PARAM_COLUMNS`.
- **Falsified by:** a missing attribute, a different name, or a different order.
- **Why:** the comparison so far read the source of the list, not a written file; the list is only metadata, so a mismatch would go unnoticed until someone relies on the label.
- **Lighter check already done (2026-10-19T21:36Z):** the list compared with `heads_spec.py` by an AST read; the old list does not match, the new one does.

### OBS-025 — Waveform reproducibility at shard scale

Status: `CONFIRMED` (2026-10-24T21:55Z, for training shard 0 only). Opened 2026-10-22T06:04Z. Closer: a full-size run on the workstation or the lab machine, with the user's go-ahead.

- **What changed:** `gen_seeded.py` pins pycbc's `numpy` FFT backend (uncommitted in the sibling at the time of writing).
- **What to check:** generate one full shard (`--counts 5000 5000`) twice in fresh processes and compare `noises`, `waveforms`, and `waveform_params` for exact equality, as step 1 of `check_determinism.py` does (it takes `--counts`).
- **Falsified by:** any differing element, however small.
- **Why:** under the FFTW default about 3 of 5 comparisons differed at 10 injections; the pinned backend was clean in 8 comparisons at that size, which does not cover 5,000 injections.
  The time per full shard is unknown and has not been measured.
- **Result (2026-10-24T21:55Z, verified; workstation, with the user's go-ahead):** training shard 0 (`--counts 5000 5000`, dataset 0) was generated twice in fresh processes, into `/mnt/DRIVE1/gen_out/fullsize_test/` and `/mnt/DRIVE1/gen_out/fullsize_repeat/`.
  `noises` (10000, 2, 4096), `waveforms` (5000, 2, 4096), and `waveform_params` (5000, 9) are bit-identical by `np.array_equal`, and all provenance attributes match apart from time and argv.
  Each run took about 202 s (real 202.23 s and 201.72 s) at about 50 samples per second, on about one core.
  Scope: one shard; the other 12 are covered by the small-scale checks only, and both files record `git_dirty = True` at `cdcc102` (the uncommitted `verbose` pass-through, since committed as `[dev-minor-12]` in `ml-gw-search`, which leaves the arrays unchanged in a 10 + 5 test).

### BUG-005 — Contract check on a full merged file

Status: `PENDING`. Opened 2026-10-22T06:55Z. Closer: the first full merged dataset file (13 shards), with the user's go-ahead for generating it.

- **What changed:** `src/gwml/contract.py`, `scripts/check_contract.py`, and `tests/test_contract.py` were added (uncommitted at the time of writing).
- **What to check:** `python scripts/check_contract.py <merged file> --full` reports every check passed, including the D-03 counts and 10 + 2 + 1 shards; then run `pytest tests/test_contract.py`.
- **Falsified by:** any failed check.
  A failure may mean the file or the contract is wrong; that is a decision for the user, not something to adjust until it passes.
- **Done so far (2026-10-22T07:26Z):** it passed 68 of 68 on a merged file of three real 10-injection shards (training 0 and 1, validation 0) and 40 of 40 on a repackaged shard.
- **Why:** the check has been exercised only on tiny real shards and synthetic mutations; nothing at full size, and its prior bounds (`PARAM_BOUNDS`) come from the frozen generator's priors by reading, not from a full file.
