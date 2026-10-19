# Verify queue

Status key: `PENDING`, `CONFIRMED`, `FAILED`, `STALE`.
Entries are keyed by the `BUGS.md` ID they check (no separate numbering).
Each entry has three parts (what changed, what to check, why it matters), plus who or what closes it and what result would falsify the expectation.
Training runs and full test output are deferred here so they are not pulled into the conversation.
Format rules are in `CLAUDE.md` and `WORKING_CONVENTIONS.md` §4.

All entries below were opened on 2026-10-19T11:37Z.
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
