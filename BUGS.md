# Bugs and observations

Status key: `OPEN`, `FIXED`, `VERIFIED`, `CLOSED: <reason>`, `DEFERRED`.
IDs are project-global (`BUG-NNN`, `OBS-NNN`), assigned in discovery order, never reused.
Newest entry at the bottom.
Verify-queue entries in `VERIFY.md` are keyed by the ID they check.
Format rules are in `CLAUDE.md` and `WORKING_CONVENTIONS.md` §3.

All entries below were opened on 2026-10-19T11:37Z from `experiments/q_recovery/experiment_index.md` (open items 1–8), `PHIC_PSI_CARRYOVER.md`, and the 2026-10-19 doc reconciliation.
Claim labels follow `CLAUDE.md`: verified / inferred / assumed / unknown.

| ID | Description | Status |
|----|-------------|--------|
| BUG-001 | InceptionTime `merger_time` head is dead (R² ≈ 0) in every recorded run | OPEN |
| BUG-002 | `resnet1d` overfits q after the dead-sigmoid revival (train 0.999, val 0.100) | OPEN |
| BUG-003 | `StdRatio` pools both components of two-component heads, so a collapsed `coa_phase` can read healthy | OPEN |
| BUG-004 | Stale code comments: `heads.py` docstring says tanh; `run_full.py` docstring is a copy of `run_all.py`'s | OPEN |
| BUG-005 | `structure.md` is referenced by docs, tests, and `heads_spec.py`, but is not in the repo | OPEN |
| BUG-006 | `PeriodicCheckpoint` has no test, and `configs/smoke.yaml` (2 epochs, default every 10) never triggers it | OPEN |
| OBS-001 | Gain on `cnn_attention` q (−0.09 → +0.204) is clamp and q-head regularization combined; the clamp-only ablation was never run | OPEN |
| OBS-002 | `q_recovery` open item 4 ("combined re-run not done") conflicts with `phase2_5_3_run_results.md` | OPEN |
| OBS-003 | Input-level augmentation for the `q_high × mchirp_low` regime was never attempted | DEFERRED |
| OBS-004 | Ensemble / auxiliary classifier for q (Phase 3.3) | DEFERRED |
| OBS-005 | Clamp-value sweep (1.5 / 2.0) on `tcn` and `cnn_baseline` | DEFERRED |
| OBS-006 | All q numbers predate the current code and the carryover findings; none has been reproduced | OPEN |
| OBS-007 | `coa_phase` is in every default config; the carryover recommends dropping it | OPEN |
| OBS-008 | The head-inclusion rule's N, M, K and per-head floor are unset | OPEN |
| OBS-009 | Whether keeping `sky_position` helps or hurts the other heads is untested | DEFERRED |
| OBS-010 | The action plan's "OLD" rows are the attempt-1 runs | VERIFIED |
| OBS-011 | `CLAUDE.md` lacks the periodic-checkpoint rule that `f50dfae` cites | OPEN |
| OBS-012 | "ThinkPad T530, CPU-only" statements may not match the machine(s) in use | CLOSED: T530 confirmed in use; machines now tracked in `CLAUDE.md` |
| OBS-013 | The test suite has no recorded run (`pytest -m "not slow"`, then full) | OPEN |

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

### OBS-008 — Head-inclusion rule unset

Opened 2026-10-19T11:37Z. Status: `OPEN`.
Origin: `PHIC_PSI_CARRYOVER.md` §3.
N seeds, M seeds, K trunks, and the per-head materiality floor are the author's to fix before any training and not to change afterward.

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
