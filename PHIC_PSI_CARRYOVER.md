> Snapshot copy of `experiments/phic_psi_poc-redo/qvalue_carryover.md` from branch `poc/phic-psi-degeneracy-redo`, commit `73793c4`.
> The source of truth is that file; do not edit this copy, add dated addenda instead.

# Carry-over for `q_value`: what the φc/ψ redo established

**Written**: 2026-10-19 16:03 +05:00, from branch `poc/phic-psi-degeneracy-redo` at `8c00565`.
**Revised**: 2026-10-19 16:11 +05:00 (draft, before first commit): sky-position stance, inclusion rule and per-head claim ledger added (§3).
This is a dated snapshot, not a living document: change it with dated addenda, not in place.
Labels follow the claim vocabulary: **verified** (a specific check was run), **inferred**, **assumed**, **unknown**.
Explanations live in `UNDERSTANDING.md`; evidence lives in `NOTES.md` and the `*_output/` directories of this experiment.

## 1. Bottom line

- **The shared code is the same on both branches (verified).** Outside `experiments/` and `runs/`, `q_value` (at `21552e2`) and this branch differ only in `CLAUDE.md`, `.gitignore`, `uv.lock`, `WORKING_CONVENTIONS.md`, the `PeriodicCheckpoint` callback (`src/gwml/training/callbacks.py`, `train.py`) and one new note, `src/gwml/gen_py_data_pipeline.md`. Models, heads, losses, the trainer, evaluation and `configs/` are identical.
- **The corrected φc/ψ math exists only in this experiment's own code (verified).** The `2φc ± 2ψ` combination construction, circular loss with unit-circle normalization, magnitude penalty and curriculum weight live in `experiments/phic_psi_poc-redo/{trainer,transform_utils,curriculum}.py`, not in `src/gwml`. `q_value` does not inherit them. They only matter if φc/ψ combinations return to the head set.
- **What changes for `q_value` is a set of decisions and some stale text, not shared code.**

## 2. Decisions for the head set

| Item                                                                                                  | Evidence                                                                                                                                                                                                                            | Suggested action                                                                                                 |
|-------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| `coa_phase` is in every default config (`heads: [mchirp, merger_time, snr, sky_position, coa_phase]`) | φc is not recoverable by point regression on this population: circular loss and ang_MAE sit at the null for all four models (verified; paper §6, `bootstrap_output/`)                                                               | Drop it from the defaults                                                                                        |
| `polarization_angle`                                                                                  | Same null as φc (verified)                                                                                                                                                                                                          | Keep out                                                                                                         |
| `inclination`                                                                                         | Partial mode collapse (circ_r 0.32–0.60), not clean noise and not recoverable (verified). The target is also mirror-ambiguous: $\iota$ and $2\pi-\iota$ give bit-identical IMRPhenomD waveforms (verified, `UNDERSTANDING.md` §4.1) | If needed, use $\cos\iota$ or $\|\cos\iota\|$, not $\iota$ itself (inferred)                                     |
| `sky_position`                                                                                        | A weak control: mean separation 86.7°, 87.0°, 87.1°, 77.5° against a ≈90° null; only `cnn_attention` clears the materiality floor (verified; `analysis_report_20261019_125442.md`, `bootstrap_sky_separation_20260918_151359.md`)   | Not a goal of the sister paper: keep as an attempted, secondary result (§3); do not use it as a positive control |
| `mchirp`, `merger_time`, `snr`                                                                        | The healthy positive controls here: validation R² 0.921–0.961, 0.901–0.921, 0.749–0.794 (verified, same report)                                                                                                                     | Re-measure on the new dataset (assumed to be similar, not verified)                                              |

Caveat: removing the dead heads from the shared trunk was never tested, so the controls need re-measuring on the pruned head set (unknown); the with/without test in §3 addresses this.
The null is for point-estimating regression on a dominant-mode, SNR 7–15 population; it is not a statement about posterior estimators.

## 3. Sky position, and a rule for what counts as a goal

**Stance (a recommendation, not a result).** Do not make sky position a goal of the sister paper; report it as an attempted, secondary result that cites this experiment.

**Evidence (verified).**
- Mean great-circle separation is 86.7°, 87.0° and 87.1° for three models and 77.5° for `cnn_attention`, against a ≈90° null (`analysis_report_20261019_125442.md`, `bootstrap_sky_separation_20260918_151359.md`).
- The separation histograms show the uniform-on-sphere shape for all four models, with no hidden accurate subpopulation (`sky_mollweide_output/`).
- `cnn_attention`'s skill is one model, one seed, with an untested mechanism (paper §6.8).
- The new dataset keeps the same generator, detectors (H1, L1), SNR range and window, so the same weakness is expected there (inferred).

**A likely reason (inferred, not tested).**
- With two detectors, the arrival-time difference confines the source to a ring around the baseline; amplitude and phase differences only partly break that.
- A unimodal von Mises–Fisher head cannot represent a ring, and a single point on a ring is a poor estimate.
- That puts sky position in the same class as φc/ψ: a partly degenerate target where point regression is the wrong tool and posterior methods are the right one. Making it a goal would invite the objection the φc/ψ paper itself raises about point-estimate heads.

**Inclusion rule (to be set before any training, and not changed after).**
- A head counts as a goal of the sister paper only if its validated recovery clears a stated materiality floor in at least N of M training seeds and in at least K trunks, and its positive control reads correctly in the same configuration.
- N, M, K and the per-head floor are **open**: the author fixes them in advance.
- Under such a rule, sky position in this experiment (floor cleared in 1 of 4 models, 1 seed) would not qualify.

**One cheap test.**
- Train with and without the sky head, on several seeds, and compare the three healthy controls (`mchirp`, `merger_time`, `snr`).
- This tells whether a weak head harms the others on the shared trunk, which is also the open pruned-head question in §2.

### Per-head claim ledger

"Cite" means quote the result with its scope: point regression, dominant mode, SNR 7–15, the dataset as generated, seed 42, two distinct trunks.
"Re-establish" means measure again on `q_value`'s own data and seeds before asserting it there.

| Head                              | Claim                                                                                                                          | Status                                                                                      | Source                                                                                              | On `q_value`                                                                                        |
|-----------------------------------|--------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| `sky_position`                    | Weak on this dataset: 86.7°, 87.0°, 87.1°, 77.5° against a ≈90° null                                                           | verified                                                                                    | `analysis_report_20261019_125442.md`, `bootstrap_sky_separation_20260918_151359.md`                 | Re-establish on the new dataset with several seeds (similar result expected: inferred)              |
| `sky_position`                    | `cnn_attention` has real partial skill: 12.58° better than shuffled (z = +23.23), error falling from 80.15° to 74.47° with SNR | verified for this seed and dataset; mechanism unknown                                       | `bootstrap_sky_separation_20260918_151359.md`, `snr_stratification_sky_position_20260918_153412.md` | Re-establish across seeds before any claim; the H1/L1 amplitude-ratio mechanism is untested         |
| `sky_position`                    | The weakness is the two-detector ring constraint                                                                               | inferred                                                                                    | reasoning only (§3)                                                                                 | Leave as a hypothesis unless a ring-aware or posterior head is tried                                |
| `sky_position`                    | Keeping it neither helps nor hurts the other heads                                                                             | unknown                                                                                     | none                                                                                                | Re-establish: the with/without test (§3)                                                            |
| `inclination` (raw ι)             | Partial mode collapse (circ_r 0.32–0.60), neither clean noise nor recoverable                                                  | verified                                                                                    | `inclination_stratification_20260923_132228.md`, `inclination_scatter_20260923_132228.{png,pdf}`    | Cite with scope                                                                                     |
| `inclination` (raw ι)             | The target is mirror-ambiguous: ι and 2π − ι give identical waveforms                                                          | verified in an earlier session (`UNDERSTANDING.md` §4.1; not rerun today)                   | `NOTES.md` pycbc check                                                                              | Cite; rerun is optional                                                                             |
| `cos ι` or `\|cos ι\|`            | Recoverable                                                                                                                    | **unknown, untested**                                                                       | none                                                                                                | **Re-establish: this is the fresh claim for ι.** A cheap test; judge it by the inclusion rule above |
| `coa_phase`, `polarization_angle` | Not recoverable by point regression on this population                                                                         | verified (4 models, 2 trunks, seed 42; ungated evidence)                                    | paper §6, `bootstrap_ang_mae_20260916_113955.md`                                                    | Cite with scope; replication on the new dataset is optional                                         |
| `coa_phase`, `polarization_angle` | Not recoverable by any method, including posterior estimators                                                                  | not claimed (unknown); the paper argues posterior methods handle degeneracy by construction | paper §8.3                                                                                          | Do not claim                                                                                        |

## 4. Stale or at-risk items on `q_value`

- **Activation text (verified).** `PLAN.md` lists tanh for `coa_phase`, and the `src/gwml/models/heads.py` docstring says "tanh for sin/cos pairs". `heads_spec.py` declares `activation="linear"` for the periodic heads, so the code is linear and the text is stale.
- **`std_ratio` (verified).** For scalar heads it is `std(pred)/std(true)` and means what `q_value`'s docs say. For two-vector heads it pools both components, so a collapsed head reads $|\sin(\theta_0-\pi/4)|$ and can look healthy (`UNDERSTANDING.md` §2.6).
- **Uncertainty weighting (verified math).** For fixed $L_h$ the optimum is $s_h^*=\log L_h$, so weights track $1/L_h$ and the ±3 clamp caps them at $e^3\approx20.1$. The q-head docs saw all four heads saturate at 20.09; that is the clamp-ceiling coupling noted in `losses.py` (`UNDERSTANDING.md` §2.9).
- **A buggy sky metric exists in one script (verified).** `experiments/phic_psi_poc/analyse_predictions.py` dots (dec, ra) arrays as unit vectors and reports a 0.0° median. Shared `src/gwml/evaluation/metrics.py` computes it correctly. Do not reuse the experiment script.

## 5. What to bring over, and what to leave

- **Bring:** commit `09a4980` (`PeriodicCheckpoint`, 2 files, 29 lines; not on `master`) by cherry-pick.
- **Bring:** `src/gwml/gen_py_data_pipeline.md`, by checking out the file, not the commit (it was added in `adf834c`, a mixed commit of run files). It documents the `gen.py` noise/waveform/whitening pipeline.
- **Move:** `experiments/q_recovery/` (3 files). It was written for `q_value` ("Branch: `q_value`", compiled 2026-07-27) and carries the open-items list for the q work, but it exists only on this branch.
- **Leave:** `runs/` (about 1.7 GB, including 110 weight files) and `experiments/phic_psi_poc-redo/` (about 53 MB).
- **Decide yourself:** `uv.lock` (two package version bumps) and `.gitignore` (two lines).
- **Not needed:** `CLAUDE.md`; `q_value` gets its own rewrite.

## 6. Process lessons to apply

- **Seeds.** This experiment used seed 42 almost everywhere and an unseeded dataset, so its null holds across configurations, not across seeds. Use separate seed streams, a seeded generator, and record seeds with the data.
- **Positive controls.** A diagnostic's verdict is untrusted until a head known to work reads correctly at the same stage.
- **Full trajectories.** Judge training metrics over the entire `history.csv`, not the endpoint.
- **Cross-check numbers.** A control number from one script should be checked against a second script that uses the shared helpers (the sky-position table was wrong for weeks).

## 7. Pointers

`UNDERSTANDING.md` §2.5–2.9 (baselines, diagnostics, vMF, Huber for inclination, weighting), §4 (ι prior and the inclination head), §5 (controls); `NOTES.md` (dated entries from 2026-09-30); the paper in the sibling repo `../gw_research_paper`.
