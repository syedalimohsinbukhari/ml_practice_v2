# Bugs and observations

Status key: `OPEN`, `FIXED`, `VERIFIED`, `CLOSED: <reason>`, `DEFERRED`.
IDs are project-global (`BUG-NNN`, `OBS-NNN`), assigned in discovery order, never reused.
Newest entry at the bottom.
Verify-queue entries in `VERIFY.md` are keyed by the ID they check.
Format rules are in `CLAUDE.md` and `WORKING_CONVENTIONS.md` §3.

BUG-001 to BUG-006 and OBS-001 to OBS-013 were opened on 2026-10-19T11:37Z from `experiments/q_recovery/experiment_index.md` (open items 1–8), `PHIC_PSI_CARRYOVER.md`, and the 2026-10-19 doc reconciliation.
BUG-007 onward and OBS-014 onward carry their own dates; they come from the 2026-10-19 read of the sibling generator repo `ml-gw-search` (paths in `CLAUDE.md`, "Related repositories").
Claim labels follow `CLAUDE.md`: verified / inferred / assumed / unknown.

| ID      | Description                                                                                                                   | Status                                                             |
|---------|-------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| BUG-001 | InceptionTime `merger_time` head is dead (R² ≈ 0) in every recorded run                                                       | OPEN                                                               |
| BUG-002 | `resnet1d` overfits q after the dead-sigmoid revival (train 0.999, val 0.100)                                                 | OPEN                                                               |
| BUG-003 | `StdRatio` pools both components of two-component heads, so a collapsed `coa_phase` can read healthy                          | OPEN                                                               |
| BUG-004 | Stale code comments: `heads.py` docstring says tanh; `run_full.py` docstring is a copy of `run_all.py`'s                      | OPEN                                                               |
| BUG-005 | `structure.md` is referenced by docs, tests, and `heads_spec.py`, but is not in the repo                                      | OPEN                                                               |
| BUG-006 | `PeriodicCheckpoint` has no test, and `configs/smoke.yaml` (2 epochs, default every 10) never triggers it                     | OPEN                                                               |
| OBS-001 | Gain on `cnn_attention` q (−0.09 → +0.204) is clamp and q-head regularization combined; the clamp-only ablation was never run | OPEN                                                               |
| OBS-002 | `q_recovery` open item 4 ("combined re-run not done") conflicts with `phase2_5_3_run_results.md`                              | OPEN                                                               |
| OBS-003 | Input-level augmentation for the `q_high × mchirp_low` regime was never attempted                                             | DEFERRED                                                           |
| OBS-004 | Ensemble / auxiliary classifier for q (Phase 3.3)                                                                             | DEFERRED                                                           |
| OBS-005 | Clamp-value sweep (1.5 / 2.0) on `tcn` and `cnn_baseline`                                                                     | DEFERRED                                                           |
| OBS-006 | All q numbers predate the current code and the carryover findings; none has been reproduced                                   | OPEN                                                               |
| OBS-007 | `coa_phase` is in every default config; the carryover recommends dropping it                                                  | OPEN                                                               |
| OBS-008 | The head-inclusion rule: M = 10, N = 8, K = all five trunks decided; the per-head floors are proposed and await the user's confirmation | OPEN |
| OBS-009 | Whether keeping `sky_position` helps or hurts the other heads is untested                                                     | DEFERRED                                                           |
| OBS-010 | The action plan's "OLD" rows are the attempt-1 runs                                                                           | VERIFIED                                                           |
| OBS-011 | `CLAUDE.md` lacks the periodic-checkpoint rule that `f50dfae` cites                                                           | OPEN                                                               |
| OBS-012 | "ThinkPad T530, CPU-only" statements may not match the machine(s) in use                                                      | CLOSED: T530 confirmed in use; machines now tracked in `CLAUDE.md` |
| OBS-013 | The test suite has no recorded run (`pytest -m "not slow"`, then full)                                                        | OPEN                                                               |
| BUG-007 | `ml-gw-search` data generation is unseeded, so the dataset cannot be regenerated bit-for-bit                                  | OPEN                                                               |
| BUG-008 | `PARAM_COLUMNS` in `ml-gw-search` `train.py` mislabels the columns `gen.py` writes (`mass1`/`mass2` versus `mchirp`/`q`)      | VERIFIED |
| BUG-009 | `injection_time` is stored as float32, which resolves only 128 s at GPS ≈ 1.2e9                                               | OPEN                                                               |
| OBS-014 | The repackaged dataset drops the noise-only samples and carries no column labels or file attributes                           | OPEN                                                               |
| OBS-015 | With `--real-noise-file`, `gen.py` pairs whitened waveforms with unwhitened real noise                                        | OPEN                                                               |
| OBS-016 | The generation arguments and the step that produced `combined_repackaged.hdf` are not recorded anywhere                       | OPEN                                                               |
| OBS-017 | Two diverging copies of `gen_py_data_pipeline.md` (sibling 90 lines, `src/gwml/` 58 lines)                                    | OPEN                                                               |
| OBS-018 | The generator output omits parameters that shaped the data (spins for XPHM, approximant, per-sample PSD index, whitening constants) | OPEN                                                         |
| OBS-019 | `gen.py` can leave a readable file with no `validation` group and no completion marker; all samples are held in memory, with no resume | OPEN                                                    |
| OBS-020 | `gen.py` imports `Dataset` from `train.py`, so generating data needs torch and the network code; the sibling has no tests | OPEN                                                                 |
| OBS-021 | Legacy priors and window to settle: skewed `q`, angles on [0, 2π], merger time fixed in [1.6, 1.8] s with a stale comment | OPEN                                                                 |
| OBS-022 | The generator writes no held-out test split, and how the q_value test set is formed is unknown                                | OPEN                                                               |
| OBS-023 | Training uses one `42` literal for weight init and data order, `make_dataset` defaults to `seed=0`, and a run does not record its seed | OPEN |
| OBS-024 | The tests seed their randomness with 11 scattered literals (`0` to `3`) | OPEN |
| OBS-025 | The seeded driver reproduces noises and params bit-for-bit across processes, but `waveforms` differ in about 1 element per 10 injections (about 1e-13) under pycbc's default FFTW backend | FIXED |

## Entries

### BUG-001 — InceptionTime `merger_time` head is dead

Opened 2026-10-19T11:37Z. Status: `OPEN`.
Origin: `q_recovery` item 7; first recorded in `phase2_5_3_revision_plan.md` concern 5.

- **Evidence (verified, `history.csv`, final epoch of 80):** `r2_merger_time` / `val_r2_merger_time` is 0.005 / −0.002 (`runs/inception_time/20260714_121546`), 0.001 / −0.000 (`20260714_162758`), 0.002 / 0.000 (`20260715_052319`), 0.002 / −0.001 (`20260715_134649`), 0.003 / 0.000 (`20260715_144802`).
- **Contrast (verified, same files):** `tcn` `20260715_150727` reads 0.919 / 0.920 and `resnet1d` `20260715_150100` reads 0.981 / 0.878 on the same head, so it is not the head or the data.
- **Impact:** the head is dead in all five recorded InceptionTime runs, including the three from 2026-07-15 under the current head set.
  The earlier plan note that the loss weight sits at the clamp ceiling rules out a weighting cause (per `phase2_5_3_run_results.md`; not re-read here).
- **Diagnosis so far (assumed):** same dead-from-the-start signature as `resnet1d`'s q sigmoid; cause unknown.
- **Next:** see `VERIFY.md` BUG-001.

### BUG-002 — `resnet1d` overfits q after the revival

Opened 2026-10-19T11:37Z. Status: `OPEN`.
Origin: `q_recovery` item 6; `phase2_5_3_run_results.md` "What's still broken".

- **Evidence (verified, `runs/resnet1d/20260714_164144/history.csv`):** final `r2_q` 0.999, `val_r2_q` 0.100, peak validation 0.170 at epoch 12.
- **Context:** the revival came from reverting the clamp to 3.0 and adding 5 warmup epochs (dead run `20260714_114721`: `val_r2_q` −5.738).
- **Proposed fix, not applied:** `per_head.q: {hidden_units: 32, dropout: 0.3, l2: 1e-4}` and `log_var_clamp: {default: 3.0, q: 1.0}`, warmup kept.
  The expected effect in the source document ("val_r2 rises to ~0.15–0.20") is a prediction, not a result.
- **Config state:** `configs/resnet1d.yaml` no longer has a q head or any q setting, so this needs a q-including config before it can be tested.
- **Next:** see `VERIFY.md` BUG-002.

### BUG-003 — `StdRatio` pools two-component heads

Opened 2026-10-19T11:37Z. Status: `OPEN`.
Origin: `PHIC_PSI_CARRYOVER.md` §4.

- **Evidence (verified by code read, `src/gwml/training/losses.py:92-120`):** `update_state` flattens `y_true` and `y_pred` with `reshape(-1)`, so a (sin, cos) head's variance mixes both components.
  The carryover reports the consequence (a collapsed head reads |sin(θ₀ − π/4)|, not 0); I did not reproduce the number here.
- **Impact:** `coa_phase` is in every default config, so its `std_ratio_coa_phase` in `history.csv` can look healthy while the head is collapsed.
  The caveat is now in `docs/pipeline/training.md` and `evaluation.md`; the metric itself is unchanged.
- **Next:** decide whether to compute `StdRatio` per component; no code change made.

### BUG-004 — Stale code comments

Opened 2026-10-19T11:37Z. Status: `OPEN`.

- `src/gwml/models/heads.py` module docstring says "tanh for sin/cos pairs"; `heads_spec.py` declares `activation="linear"` for periodic heads since commit `6839b47` (verified by reading both).
- `scripts/run_full.py` docstring describes `run_all.py configs/resnet1d.yaml`, but the script takes only `--split` and loops over every non-smoke config (verified by reading).
- No behavior is affected.

### BUG-005 — `structure.md` missing

Opened 2026-10-19T11:37Z. Status: `OPEN`.

- Referenced (verified by grep) by `README.md` (marked "not present" since 2026-10-19), `docs/pipeline/data.md`, `docs/pipeline/testing.md`, `tests/test_data_integrity.py` (docstring and `test_shapes_match_structure_md`), `tests/test_heads_spec.py` (`test_every_spec_column_matches_structure_md`), and a comment in `src/gwml/heads_spec.py`.
- `git log --all` shows it has never been in the repository (verified).
- Next: restore it from wherever it lives, or write the HDF5 layout down from `combined_repackaged.hdf` and drop the stale references.
- **Update 2026-10-22T06:55Z:** the contract of the regenerated file now exists as code, `src/gwml/contract.py`, with `scripts/check_contract.py` and `tests/test_contract.py`; the references to `structure.md` in `README.md`, `docs/pipeline/`, `heads_spec.py`, and the two tests are unchanged, so this stays `OPEN` until the file is written or the references are dropped.

### BUG-006 — `PeriodicCheckpoint` is untested

Opened 2026-10-19T11:37Z. Status: `OPEN`.
Origin: commit `f50dfae`.

- **Evidence (verified):** `grep -rn 'PeriodicCheckpoint\|checkpoint_every' tests` finds nothing.
  `configs/smoke.yaml` has `epochs: 2` and no `checkpoint_every_n`, so the default of 10 never fires and the smoke run does not exercise the callback.
  `runs/tcn/20260715_150727` predates the callback and has no `checkpoints/` directory.
- **Impact:** the `epoch_NNN.weights.h5` snapshots are unverified; a silent failure would only show up when a later calibration needs them.
- **Next:** see `VERIFY.md` BUG-006.

### OBS-001 — Clamp versus q-head regularization, unseparated

Opened 2026-10-19T11:37Z. Status: `OPEN`.
Origin: `phase2_5_3_revision_plan.md` concern 2.

- `cnn_attention` went from `val_r2_q` −0.090 (attempt 1) to +0.204 (attempt 2) with the clamp added while q-head regularization was already present, so the split is unknown (verified from `history.csv` and the run index).
- Inferred only: `tcn`, `inception_time`, and `cnn_baseline` reached positive `val_r2_q` with the clamp alone, which points at the clamp.
- Next: see `VERIFY.md` OBS-001.

### OBS-002 — `q_recovery` item 4 conflicts with the run results

Opened 2026-10-19T11:37Z. Status: `OPEN`.

- `experiments/q_recovery/experiment_index.md` item 4 says the combined re-run of all five configs "is not done".
- `phase2_5_3_run_results.md` reports five runs on 2026-07-14 16:18–16:48 with the Phase 2.5, 3.1, and 3.2 changes (oversampling, q-head regularization, clamp revert plus warmup, `q_tokens` branch); the five run directories exist and their final `val_r2_q` values match the document (verified).
- Run directories store `transforms.json` and weights but not the config, so which settings each run used cannot be confirmed from the repo (verified by listing; inferred that the document is right).
- Decision for you: close item 4 as done, or re-run to obtain stored configs.

### OBS-003 — Targeted input augmentation, never attempted

Opened 2026-10-19T11:37Z. Status: `DEFERRED`.
Origin: `q_recovery` item 1; `q_head_action_plan.md` Phase 2 step 11.

- Jitter, noise, or SNR variation aimed at the `q_high × mchirp_low` cell (the worst cell in every run: R² −15 to −22 after oversampling, per `phase2_5_3_run_results.md`).
- Oversampling at 2× cut that cell's MAE 12–18% on four trunks but did not move full-set R².

### OBS-004 — Ensemble or auxiliary classifier for q

Opened 2026-10-19T11:37Z. Status: `DEFERRED`.
Origin: `q_recovery` items 2 and 8; Phase 3.3.
Contingent on augmentation and oversampling plateauing.

### OBS-005 — Clamp-value sweep

Opened 2026-10-19T11:37Z. Status: `DEFERRED`.
Origin: `q_recovery` item 5.
Optional: q clamp 1.5 and 2.0 on `tcn` and `cnn_baseline`, to find the best per-head gradient budget.

### OBS-006 — q numbers predate the current code

Opened 2026-10-19T11:37Z. Status: `OPEN`.

- Every q run is from 2026-07-14.
  Since then the periodic heads changed from tanh to linear (`6839b47`), `sky_position` and the vMF loss were added, and the default head set changed (`8b2eb41`).
- Whether the 2026-07-14 numbers reproduce on the current code is unknown.
- By the project's positive-control rule, any new q result needs `mchirp`, `merger_time`, and `snr` reading correctly in the same configuration (the carryover reports validation R² of about 0.92–0.96, 0.90–0.92, and 0.75–0.79 for them on the φc/ψ branch).
- Next: see `VERIFY.md` OBS-006.

### OBS-007 — `coa_phase` in the default heads

Opened 2026-10-19T11:37Z. Status: `OPEN`.

- The carryover (§2) recommends dropping it: circular loss and angular MAE sit at the null for all four models (verified on the φc/ψ branch, not here).
- All six `configs/*.yaml` still list it.
- Decision for you.
  Dropping it also removes the `StdRatio` exposure in BUG-003 for the default set.

### OBS-008 — Head-inclusion rule

Opened 2026-10-19T11:37Z. Status: `OPEN` (the numbers below are set; closes when the user confirms the floors and the K question).
Origin: `PHIC_PSI_CARRYOVER.md` §3.

- **Decided 2026-10-19T22:06Z (user):** M = 10 training seeds per trunk, N = 8 of those must clear the floor, K = all five trunks (`cnn_attention`, `cnn_baseline`, `inception_time`, `resnet1d`, `tcn`; `smoke` is a test configuration).
  The seeds are `seed_index` 0 to 9 (`SEEDING.md`).
- **Floors, proposed 2026-10-19T22:06Z by Claude on the user's delegation ("you can decide"):**
  - Scalar heads (`mchirp`, `q`, `merger_time`, `snr`, and `cos ι` or `|cos ι|` if used in place of `inclination`): validation R², averaged over the last five epochs of `history.csv`, at least 0.50, and validation `std_ratio` at least 0.50 so the head is not collapsed.
  - Angle heads (`inclination` as an angle, `coa_phase`, `polarization_angle`) and `sky_position`: validation error (angular MAE, or mean great-circle separation) at most 0.75 of the same error for a shuffled-pairs baseline on the same validation set, which is about 67.5° for sky position against the ≈90° null.
  - Rationale: a material head has errors about a quarter below the chance predictor; R² 0.50 corresponds to an RMSE about 29% below the mean predictor.
- **Positive controls, in the same run:** `mchirp` validation R² at least 0.87, `merger_time` at least 0.85, and `snr` at least 0.70 (the lowest recorded values, 0.921, 0.901, and 0.749 from the φc/ψ branch, minus 0.05; measured there, assumed similar on the new dataset).
  A run whose controls fail is untrusted and none of its heads counts.
- **Counting:** a head clears a trunk if it passes the floor, with the controls reading correctly, in at least 8 of that trunk's 10 seeds; it is a goal of the sister paper if it clears all five trunks.
- **Fixed in advance:** these values are set before any training and are changed only by a dated addendum; until the first campaign run starts the user may still amend the floors.
- **Cost (inferred from run start times; see the answer in the session of 2026-10-19T22:06Z):** 10 seeds × 5 trunks is 50 runs of 80 epochs, about 5 h at the current 25,000 training rows and about 10 h at the planned 50,000 injection rows, one GPU, sequential.
- **Consequence to decide (verified from BUG-001):** the InceptionTime `merger_time` head is dead (validation R² ≈ 0 in all five recorded runs), so its control fails and, under the rule above, no head can clear `inception_time`; with K = all, no head can qualify unless BUG-001 is fixed first, or K is lowered to 4, or `inception_time` is left out of the campaign.
- **Also open:** the rule says "validated recovery" (the validation split, as in the carryover); whether the planned test split is used for a final confirmation is not decided.
- **Original text:** N seeds, M seeds, K trunks, and the per-head materiality floor are the author's to fix before any training and not to change afterward.

### OBS-009 — Effect of `sky_position` on the other heads

Opened 2026-10-19T11:37Z. Status: `DEFERRED`.
Origin: `PHIC_PSI_CARRYOVER.md` §2–3.
The carryover treats sky position as a weak result (mean separation 86.7°–87.1° against a ≈90° null for three models; 77.5° for `cnn_attention`; verified on the φc/ψ branch) and recommends the with/without test on several seeds.
Whether a weak head harms the shared trunk is unknown here.
Next: see `VERIFY.md` OBS-009.

### OBS-010 — "OLD" rows in the action plan

Opened 2026-10-19T11:37Z. Status: `VERIFIED`.

- In `q_head_action_plan.md`, the "Before/after table" rows marked OLD for `cnn_attention` and `cnn_baseline` are the attempt-1 runs (`20260714_104523`, `20260714_104107`), not the pre-fix runs (`20260714_093319`, `20260714_091740`).
- Verified by recomputing final and peak `val_r2_q` from `history.csv` for all four runs on 2026-10-19.
- Resolution: a dated status addendum in the document; the original table is deliberately unchanged.
- Original diagnosis: the table's peaks (0.237 at epoch 19 and 0.146 at epoch 6) match attempt 1, which is how the mislabel was found.

### OBS-011 — Periodic-checkpoint rule missing from `CLAUDE.md`

Opened 2026-10-19T11:37Z. Status: `OPEN`.
`f50dfae` implements the rule and cites `CLAUDE.md`, but the rule exists only in `CLAUDE_phic_psi_backup.md`.
Decision for you: restore a short rule in `CLAUDE.md`.

### OBS-012 — "T530, CPU-only" statements

Opened 2026-10-19T11:37Z. Status: `CLOSED: T530 confirmed in use; machines now tracked in CLAUDE.md`, closed 2026-10-19T20:30Z.

- **Resolution:** a T530 is still in use.
  A session on 2026-10-19T20:27Z ran on `thinkpad-t530` with 8 CPUs and no `nvidia-smi` (verified), so the "T530, CPU-only" statements are correct for that machine.
  At least three machines work on this repo, so "the local machine" in those documents is ambiguous; the per-machine hosts and paths are now in the "Machines" table in `CLAUDE.md`.
  `PLAN.md`, `README.md`, and `docs/pipeline/testing.md` are left unchanged.
- **Original diagnosis:**
  - `PLAN.md`, `README.md`, and `docs/pipeline/testing.md` call the local machine a ThinkPad T530 with no GPU.
  - On 2026-10-19T10:36Z the host was `sarlws5-Precision-7820-Tower` with a Quadro P620 (verified).
  - Whether a T530 is still in use was unknown, so the statements were left unchanged.

### OBS-013 — No recorded test run

Opened 2026-10-19T11:37Z. Status: `OPEN`.
Origin: `q_recovery` item 3.
The action plan records `pytest -m "not slow"` as never run, and nothing since shows otherwise.
The code has changed substantially since (vMF head, linear periodic heads, `PeriodicCheckpoint`), so the suite's current state is unknown.
Next: see `VERIFY.md` OBS-013.

### BUG-007 — Generation is unseeded

Opened 2026-10-19T20:40Z. Status: `OPEN`.
Origin: read of `ml-gw-search/mlgwsc-1/gen.py`.

- **Evidence (verified, code read):**
  - `gen.py:39` creates `np_gen = np.random.default_rng()` with no seed, and the argument parser (`gen.py:281-297`) has no seed option.
  - The injection parameters (masses, angles, spins, injection time, merger placement) all draw from `np_gen`.
  - The Gaussian noise comes from `pycbc.noise.gaussian.frequency_noise_from_psd(psd)` called without a seed (`gen.py:111`); I did not read pycbc's default RNG behavior.
  - `train.py`'s `Dataset.__init__` also creates an unseeded `np.random.default_rng()` for the per-epoch SNR draw.
  - Only `re_package_hdf.py:47` seeds anything (`default_rng(132354)`, a literal, for the SNR column).
- **Impact:** the current 25,000 / 5,000 dataset cannot be reproduced bit-for-bit, and a regeneration will differ from it even with identical arguments.
  This contradicts the `CLAUDE.md` determinism rule (one named master seed, per-run seeds derived from it).
  The paper already names this gap: `004_methods.tex` (comment, line 6) and `008_discussion.tex` (line 45) list "dataset regeneration from a version-controlled generator" as open.
- **Update 2026-10-19T20:55Z (verified, pycbc source read in the sibling `.venv`):** seeding `np_gen` alone is not enough.
  `frequency_noise_from_psd(psd, seed=None)` draws from the legacy global `numpy.random` (a seed only calls `numpy.random.seed`), and the sky draw `UniformSky.rvs()` (`gen.py:204`) uses the global `numpy.random.uniform` and has no seed parameter.
  Passing one seed per call also gives identical noise on every call, including identical H1 and L1 noise within a sample (verified).
- **Decision 2026-10-19T20:55Z (user):** per-file keys `SeedSequence(MASTER_SEED, spawn_key=(file_index, stream, purpose))`, one `np.random.default_rng` threaded per stream, and one `numpy.random.seed` call per stream for pycbc (option A).
  The scheme and the fresh `MASTER_SEED` are recorded in `SEEDING.md`; since 2026-10-19T22:06Z they live in the generator's own `ml-gw-search/SEEDING.md`, and this repository has a separate master seed for training and tests.
  Revised 2026-10-19T21:46Z for the shard design: the key is now `(dataset_id, split, shard, purpose)`, with one process per 10,000-sample shard, and the earlier per-file key is superseded (`ml-gw-search/SEEDING.md`).
- **Next:** implement the scheme in `ml-gw-search` (a master-seed constant, a file-index CLI option, the per-stream seeding), then run the determinism check in `ml-gw-search/SEEDING.md` on a shrunken workload on a non-T530 machine with the user's go-ahead; no code change made yet.

### BUG-008 — `PARAM_COLUMNS` mislabels the written columns

Opened 2026-10-19T20:40Z. Status: `VERIFIED` (2026-10-22T06:04Z); fixed 2026-10-19T21:36Z.
Origin: read of `ml-gw-search/mlgwsc-1/train.py` and `gen.py`.

- **Resolution (2026-10-19T21:36Z, user's decision):** the list in the sibling's `mlgwsc-1/train.py` was edited (uncommitted in the sibling) to `mchirp, q, inclination, coa_phase, polarization_angle, declination, ra, injection_time, merger_time`, the first nine keys of `heads_spec.PARAM_COLUMNS`.
  Checked by an AST comparison with a known-bad control: the old list does not match `heads_spec.py`, the new list does.
  Behavior is unaffected: `PARAM_COLUMNS` is used only for log messages, the `columns` attribute written by `Dataset.save`, and `NUM_PARAMS = len(...)` (verified, grep).
  The docstring and help text in `train.py` that still describe the regressed columns (for example "3 → mass1, mass2, merger_time_in_window") were already inconsistent with the column order and were left untouched.
  The sibling has no test suite, so this is covered only by the ad-hoc comparison; the attribute in a written file is queued in `VERIFY.md`.
  **Verified 2026-10-22T06:04Z:** in a shard written by `gen_seeded.py` (10 injections, `sarlws5-Precision-7820-Tower`), the `columns` attribute of `waveform_params` equals the first nine names of `heads_spec.PARAM_COLUMNS` in order, and the file-level `columns` attribute equals it too; the control is the earlier AST comparison, where the old list did not match.
- **Original diagnosis:**
- **Evidence (verified, code read):**
  - `train.py:24-34` defines `PARAM_COLUMNS = ['mass1', 'mass2', 'inclination', 'coa_phase', 'pol_angle', 'declination', 'right_ascension', 'merger_time_gps', 'merger_time_in_window']`, and `Dataset.save()` writes it as the `columns` attribute of `waveform_params`.
  - `gen.py:258-268` writes `[mchirp, q, inclination, coa_phase, pol_angle, declination, right_ascension, injection_time, merger_time_in_window]`, with `q = m2/m1`.
  - Columns 0 and 1 are therefore chirp mass and mass ratio, not the two masses; the other seven names match.
  - Column 7 is called `merger_time_gps` but holds the injection time (GPS seconds, provenance only).
- **Impact:** any `gen.py` output carries a `columns` attribute that contradicts its contents.
  The repackaged file has no labels at all (OBS-014), so the real meaning lives only in `gen.py` and in this repo's `heads_spec.py`.
- **Update 2026-10-19T21:03Z (verified, code read):** `src/gwml/heads_spec.py` (`PARAM_COLUMNS`) assigns `mchirp, q, inclination, coa_phase, polarization_angle, declination, ra, injection_time, merger_time, snr` to columns 0 to 9, which agrees with `gen.py`.
  The defect is confined to the sibling's own `PARAM_COLUMNS` and the `columns` attribute it writes; this repository's schema is correct.
- **Next:** confirm the attribute in the first generated file (`VERIFY.md` BUG-008).

### BUG-009 — `injection_time` quantized to 128 s

Opened 2026-10-19T20:40Z. Status: `OPEN`.

- **Evidence:**
  - `gen.py:210` draws the injection time uniformly in [1238166018, 1253977218], and `gen.py:268` stores the parameter row as `float32` (verified, code read).
  - `np.spacing(np.float32(1.24e9))` is 128.0, and the same at both ends of the range (verified, computed 2026-10-19T20:38Z).
  - `combined_repackaged.hdf` stores `params` as float64, but the values were rounded to float32 first (inferred from the generation path).
- **Impact:** the column is documented as provenance only, so training is probably unaffected.
  Any use that needs better than 128 s, such as matching an injection to an O3a data segment, cannot rely on it.
- **Update 2026-10-19T21:03Z:** every `injection_time` in `combined_repackaged.hdf` is exactly float32-representable (verified, computed).
  Verified 2026-10-19T21:16Z (pycbc source read): `project_wave` defaults to `method='lal'` and applies time-varying antenna patterns and Doppler shifts from the waveform epoch, so the GPS time influences the data and the stored time can differ by up to 64 s from the time actually used; "provenance only" understates it.
  `heads_spec.py` already excludes `injection_time` as a head.
- **Decision 2026-10-19T21:31Z (user):** the cast is inside the generator code, which is not to be changed, so it stays; the 128 s quantization is documented in the output contract instead.
  Because `project_wave` uses the epoch (verified above), the contract must not describe the column as provenance only.
- **Next:** write that note into the contract; no code change.

### OBS-014 — Repackaged dataset: no noise class, no labels

Opened 2026-10-19T20:40Z. Status: `OPEN`.
Origin: read of `ml-gw-search/re_package_hdf.py` and the structure of `combined_repackaged.hdf`.

- **Evidence (verified):**
  - `re_package_hdf.py` truncates `noises` to `noises[:n_wv]` and writes only the injection samples.
  - The HDF (listed with `h5py`, metadata only, 2026-10-19) has `training` (25,000 rows) and `validation` (5,000 rows), each with `h1`, `l1`, `h1_waveform`, `l1_waveform` (float32, 4096 samples) and `params` (float64, 10 columns).
  - There are no file attributes and no dataset attributes.
  - The `snr` column is appended as the tenth, drawn uniformly in [7, 15] with seed 132354, whereas `train.py` defaults to (5, 15) and redraws every epoch.
- **Impact:** a consumer needing a noise class has to take it from elsewhere, and column meaning has to be known from outside the file (BUG-008).
  Whether this is intended for the current model work is the user's call.
- **Decision 2026-10-19T21:10Z (user):** the regenerated dataset keeps a noise-only class, half of every split, written by a new `re_package_hdf_with_noise.py`; the original `re_package_hdf.py` stays unchanged.
  The layout of the noise-only rows (separate datasets or interleaved with a label) is open (`GENERATOR_OVERHAUL.md` D-12).
- **Next:** settle D-12, then implement; labels and `sample_rate` go in as attributes (`GENERATOR_OVERHAUL.md` section 4).

### OBS-015 — Real-noise path pairs whitened and unwhitened data

Opened 2026-10-19T20:40Z. Status: `OPEN`.
Origin: `mlgwsc-1/gen_py_data_pipeline.md` caveat, checked against the code.

- **Evidence:** `slice_real_noise.py:72` calls the slicer with `white=False` (verified, grep), and `HDFNoiseGetter.__next__` returns those slices unchanged (`gen.py:77-79`, verified, code read), while the waveform is whitened (`gen.py:242-245`).
- **Inferred, not tested:** the stored noise is then colored and the waveform whitened, so the sum is inconsistent.
- **Impact:** none for the Gaussian-noise default; it matters only if the regeneration uses `--real-noise-file`.
- **Next:** if real noise is chosen, whiten it in the generator, then confirm on a small run that noise and waveform have matching spectra.

### OBS-016 — Generation provenance not recorded

Opened 2026-10-19T20:40Z. Status: `OPEN`.

- **Evidence (verified):**
  - `ml-gw-search/out/` is empty and `out.hdf`, which `re_package_hdf.py` reads, does not exist there.
  - The script writes `out_repackaged.hdf`, but the dataset is named `combined_repackaged.hdf`, so a step outside the repository produced it.
  - The HDF shows 2 detectors, 4096-sample windows, and 25,000 / 5,000 injections, but the `gen.py` arguments (approximant, noise source, seed, sample counts) are not stored in the file or anywhere I found.
  - The two copies of `combined_repackaged.hdf` (sibling and this repository) are byte-identical (`cmp`, 2026-10-19T20:38Z).
  - In `ml-gw-search`, `re_package_hdf.py` is untracked and the HDF is neither tracked nor ignored.
- **Impact:** the existing dataset cannot be described from the repositories alone, which is the reproducibility gap the paper lists.
- **Update 2026-10-19T20:55Z:** the required provenance for a regenerated file is now specified in `ml-gw-search/SEEDING.md` ("What each generated file records"): master seed, file index and keys, `gen.py` arguments, generator git SHA, library versions, UTC time, host, and column names, stored as HDF attributes.
- **Next:** implement that recording in the generator, and track `re_package_hdf.py`.

### OBS-017 — Two diverging pipeline notes

Opened 2026-10-19T20:40Z. Status: `OPEN`.

- `ml-gw-search/mlgwsc-1/gen_py_data_pipeline.md` (90 lines) adds a `re_package_hdf.py` section after line 58; `src/gwml/gen_py_data_pipeline.md` (58 lines) is otherwise the same and predates it (verified, `diff`).
- `CLAUDE.md` leaves the `src/gwml/` copy ungoverned until triaged.
- Next: decide which copy is the source of truth, or point one at the other.

### OBS-018 — Parameters that shaped the data are not saved

Opened 2026-10-19T21:03Z. Status: `OPEN`.
Origin: `GENERATOR_OVERHAUL.md` G-04.

- **Evidence (verified, code read):**
  - For `IMRPhenomXPHM`, `gen.py` draws and uses both spin vectors but the parameter row has no spin columns, and the approximant is not stored.
  - The noise getter returns the PSD and whitening PSD for each sample, and `generate_dataset` drops them, so the PSD index per sample is lost with `--psd-file` or real noise.
  - The whitening constants (`segment_duration`, `max_filter_length`, `low_frequency_cutoff`) are module constants and are not recorded.
- **Impact:** the file cannot be described from its contents, and spin effects cannot be studied.
- **Next:** decide with the user what to record (`GENERATOR_OVERHAUL.md` D-01); no code change made.

### OBS-019 — Incomplete files look complete

Opened 2026-10-19T21:03Z. Status: `OPEN`.
Origin: `GENERATOR_OVERHAUL.md` G-12.

- **Evidence (verified, code read, `gen.py:341-355`):** the training group is saved, then the validation group is generated and appended.
  Every sample is accumulated in lists and stacked before saving, so there is no incremental write and no resume.
- **Impact:** a failure during the validation step leaves a readable file without a `validation` group and with nothing marking it incomplete.
- **Next:** write under a temporary name, rename on success, and set a `complete` attribute last; no code change made.

### OBS-020 — Generation is coupled to the training stack; no tests

Opened 2026-10-19T21:03Z. Status: `OPEN`.
Origin: `GENERATOR_OVERHAUL.md` G-18 and G-19.

- **Evidence (verified, imports read and listing):** `gen.py` imports `Dataset` from `train.py`, which imports `torch` and the network code from `apply.py`, only to write an HDF.
  Scripts import each other as top-level modules, and the sibling repository has no tests.
- **Impact:** data generation needs the whole training environment, and the determinism check in `ml-gw-search/SEEDING.md` can only be an ad-hoc check unless a test is added.
- **Next:** decide where the generator code lives (`GENERATOR_OVERHAUL.md` D-11); no code change made.

### OBS-021 — Legacy priors and window to settle

Opened 2026-10-19T21:03Z. Status: `OPEN`.
Origin: `GENERATOR_OVERHAUL.md` G-13 to G-15.

- **Evidence (verified from code and from the 30,000 stored rows):**
  - Component masses are uniform on [10, 50] with `m1 = max`; observed `q` runs 0.2027 to 0.99998 with mean 0.650, and 7.6% of training samples have `q ≥ 0.95`.
  - `inclination`, `coa_phase`, and `pol_angle` are uniform on [0, 2π]; observed inclination maximum is 6.283.
  - The merger time lies in [1.6, 1.8] s of a 2 s window (observed min 1.6, max 1.8), and the code comment says "between 0.5 s and 0.7 s", which does not match the arithmetic.
- **Inferred:** the physically isotropic inclination prior is `cos ι` uniform on `ι` in [0, π].
- **Impact:** these are design choices, not defects, but they shape the q result (the worst cell is `q_high × mchirp_low`, OBS-003) and the paper's mirror-degeneracy discussion.
- **Next:** the user decides whether to keep or change them (`GENERATOR_OVERHAUL.md` D-02, D-08).

### OBS-022 — No held-out test split

Opened 2026-10-19T21:03Z. Status: `OPEN`.
Origin: `GENERATOR_OVERHAUL.md` G-11.

- **Evidence (verified, code read):** `gen.py` writes only `training` and `validation`, and `load_arrays` reads a `split` name from the file.
- **Unknown:** how the test set for the q_value work is meant to be formed.
- **Decision 2026-10-19T21:10Z (user):** the regenerated dataset gets a test split (10,000 samples: 5,000 injections and 5,000 noise-only), alongside training (100,000) and validation (20,000); see `GENERATOR_OVERHAUL.md` D-03.
- **Next:** implement it in the generator; the loader here does not yet read a `test` split.

### OBS-023 — Training seed handling

Opened 2026-10-19T21:53Z. Status: `OPEN`.
Origin: `SEEDING.md` D-10.

- **Evidence (verified, grep and code read):**
  - `train.seed: 42` is set in all six `configs/*.yaml` (`cnn_attention` line 36, `cnn_baseline`, `inception_time`, `resnet1d`, `tcn` line 34, `smoke` line 35), and `train.py:147` defaults to 42.
  - `train.py:148` passes it to `keras.utils.set_random_seed`, and `train.py:185` passes the same number to `make_dataset(seed=...)`, so weight initialisation and data order share one number.
  - `make_dataset` in `loader.py:87` defaults to `seed: int = 0`.
  - Nothing else reads or writes the seed, so a run does not record which seed it used.
- **Decision 2026-10-19T21:53Z (user, D-10):** derive the training seeds from the master seed with a replicate index in the configs and separate streams for initialisation and data order (`SEEDING.md`, "Training seeds"); `make_dataset` can take the derived value (checked with `tf.data`).
- **Next:** implement it, and have a run record `seed_index`, the derived seeds, and the master seed; no code change made.

### OBS-024 — Test seeds

Opened 2026-10-19T21:53Z. Status: `OPEN`.
Origin: `SEEDING.md` D-10.

- **Evidence (verified, grep):** `np.random.default_rng` or `keras.utils.set_random_seed` is called with a literal `0`, `1`, `2`, or `3` in `conftest.py:49`, `test_registry.py:25`, `test_losses.py` (lines 21, 108, 121), `test_heads_spec.py` (lines 53, 86, 105, 125), and `test_overfit.py:35`; several tests share the same literal.
- **Decision 2026-10-19T21:53Z (user):** the tests are seeded with the same methodology, keyed by test name (`SEEDING.md`, "Test seeds").
- **Risk:** reseeding changes each test's data, and tolerance-based and overfit tests may be sensitive to it; a failure after reseeding is a decision point for the user, not a reason to search for a seed that passes.
- **Next:** implement a `conftest.py` fixture, then run the suite on the lab machine (to be queued in `VERIFY.md` with the change); no code change made.

### OBS-025 — Waveforms are not bit-reproducible across processes under FFTW

Opened 2026-10-22T06:02Z. Status: `FIXED` (2026-10-22T06:04Z); verified only on a shrunken workload, the at-scale check is queued in `VERIFY.md`.
Origin: first run of `ml-gw-search/mlgwsc-1/check_determinism.py` (2026-10-22, `sarlws5-Precision-7820-Tower`).

- **Resolution (2026-10-22T06:04Z, user's decision):** `gen_seeded.py` now sets pycbc's `numpy` FFT backend before generating (`pin_fft_backend`), checks that `pycbc.fft.npfft` is the active module, and records it in the `fft_backend` attribute.
  Reran `check_determinism.py` three times with 10 injections and 5 noise-only samples per shard: 9 of 9 checks passed each time (steps 1 to 4); steps 5 and 6 were added and the whole check passed 17 of 17 in three runs on 2026-10-22.
  Weak power: under the FFTW default about 3 of 5 comparisons failed at this size, so 8 clean comparisons (3 here, 5 in the probe) are unlikely by chance but do not cover a 5,000-injection shard.
  The `numpy` backend is a run-time pycbc setting, not an edit to the frozen backend, and it shifts the arithmetic by about 1e-13 relative to the FFTW default the paper used.
- **Original diagnosis:**
- **Evidence (verified, check output and array comparison):**
  - Step 1 and the in-process-versus-file check failed: two fresh processes with the same key (3 injections + 3 noise-only) differ in 1 element of 24,576 of `waveforms`, by 5.68e-14.
  - `noises` and `waveform_params` were bit-identical, and the provenance attributes matched.
  - Steps 2 to 4 passed, and A, B, C, then A again in one process gave identical copies of A.
- **Probe (verified; scratch script `fft_probe.py` in the session scratchpad, not in a repository; 10 injections, 6 fresh processes per variant, each compared with the variant's first run):**
  - Default pycbc backend (FFTW): 3 of 5 runs differ from run 0 in 1 waveform element, by up to 2.3e-13; noises and params identical.
  - `numpy` FFT backend: 0 of 5 runs differ.
  - FFTW at measure level 0 (`ESTIMATE`): 3 of 5 runs differ, in 1 to 2 elements.
  - Control: the default and `numpy` backends differ from each other in 2 of 81,920 waveform elements, by 2.3e-13, so the variant did change the arithmetic.
- **Inferred, not tested:** the cause is FFTW choosing different plans in different processes, and since `ESTIMATE` did not remove it the cause is more likely memory alignment than timing-based `MEASURE` planning.
  Whether `set_measure_level(0)` took effect was not checked.
- **Size:** the largest difference is far below the noise (noise values reach 131), and the parameters are identical.
- **Decision (user, 2026-10-22T06:04Z):** pin the `numpy` FFT backend in the driver; the alternatives were a tolerance on `waveforms` (noises and params stay bitwise) or a test of the alignment guess.
- **Also found in the same run (fixed):** the driver crashed writing attributes because `torch.__version__` is a `str` subclass that h5py cannot store; `write_provenance` now coerces strings to `str`.
