# Generator overhaul

Written 2026-10-19T21:03Z.
Status: gap inventory, draft output contract, and decisions; no generator code has been written (see the status summary below).
The generator lives in the sibling repository `ml-gw-search` (`mlgwsc-1/gen.py`, `re_package_hdf.py`; paths in `CLAUDE.md`, "Related repositories").
This file is the single list of what the current generator lacks, what the regenerated file must contain, and what is still to be decided.
Seeding of the generator is specified separately in `ml-gw-search/SEEDING.md` (its own master seed, since 2026-10-19T22:06Z); training and test seeding is in this repository's `SEEDING.md`.

Claim labels follow `CLAUDE.md`: **verified** (name the check), **inferred**, **assumed**, **unknown**.
Gap IDs (`G-nn`) are local to this file; the project-wide log is `BUGS.md`, and each gap cites its entry where one exists.

## Status summary (2026-10-19T22:12Z)

- **Settled:**
  - The generator backend is frozen; all new work wraps it (section 0, section 7).
  - The dataset: `IMRPhenomD`, spinless, legacy priors, Gaussian aLIGO noise, H1 and L1, 4096 samples at 2048 Hz, 100,000 / 20,000 / 10,000 with half noise-only, in 13 shards plus a merge (D-01 to D-08, D-11).
  - The layout: injection rows in `h1`, `l1`, `params` so the loader is unchanged, noise-only rows in separate datasets, the strain plus the unit-SNR waveform stored, one SNR per sample baked in (D-05, D-12).
  - Seeding: a separate master seed per repository, `dataset_id` in the generation key, training seeds from `seed_index` (D-09, D-10; `ml-gw-search/SEEDING.md` and this repository's `SEEDING.md`).
- **Done:** the `PARAM_COLUMNS` label list in the sibling's `train.py` (BUG-008, committed there as `[dev-minor-2]`).
- **Not built:** the seeded driver, `re_package_hdf_with_noise.py`, the merge step, the backend hash pinning, the provenance attributes, the contract check; no generator code has run.
- **Open:** the head-inclusion floors and K = all against BUG-001 (`BUGS.md` OBS-008, which concerns the later training campaign, not the generation); the decisions are listed in `HANDOFF.md`.
- **Next:** build the wrap-around one step at a time with the user's go-ahead (`HANDOFF.md`, "Next").

## 0. Constraint: the generator internals are frozen (user, 2026-10-19T21:20Z)

The generator is part of a published, proven paper (the MLGWSC-1 submission and the thesis), and the user is reluctant to change its internals.
The one change already made, from `mass1` / `mass2` to `mchirp` / `q` labels, was accepted because it does not alter the backend.

Verified 2026-10-19T21:20Z (`git diff -w` against the last upstream commit `a33ba6b`, whose `gen.py` is identical to the thesis-release commit `0589eb0`):

- The current `mlgwsc-1/gen.py` differs from the upstream version only by formatting and import order, the collection of the nine label values in `generate_dataset` (passed on as `waveform_params`), and two file attributes (`sample_length`, `sample_rate`).
- The noise, waveform, projection, merger placement, SNR normalisation, and whitening code is unchanged.
- `whiten` and `regularize_psd` in `apply.py` are identical to upstream apart from whitespace.

Implications, confirmed by the user on 2026-10-19T21:36Z (section 7 sets them out):

- Nothing that shapes the data distribution or its numerics is edited or parameterized: priors, merger window, noise, whitening, SNR normalisation, and the constants stay as they are.
- New work goes in new files beside the generator (a seeded driver, shard packaging, merge, checks), calling the existing functions unchanged.
- Gaps G-13 to G-16 and G-22 to G-24 become documented properties of the paper's generator, not items to fix.
  A failed verification would be reported to the user as a finding about the published backend, not corrected on the spot.

## 1. What this repository consumes today

- `src/gwml/data/loader.py` (`load_arrays`) reads `f[split]` with datasets `h1`, `l1`, `params`, and returns strain `(N, 4096, 2)` float32 and params `(N, 10)` float64, detector order `[h1, l1]` (verified, code read).
- Only the splits `training` and `validation` are known to be read; `configs/tcn.yaml` names `combined_repackaged.hdf` as the data path (verified, grep).
- `src/gwml/heads_spec.py` (`PARAM_COLUMNS`) is the effective schema of the ten `params` columns: `mchirp, q, inclination, coa_phase, polarization_angle, declination, ra, injection_time, merger_time, snr` at indices 0 to 9 (verified, code read).
  It agrees with what `gen.py` writes for columns 0 to 8, so the mislabel in BUG-008 is confined to the sibling's own `PARAM_COLUMNS`.
- `heads_spec.py` cites a `structure.md` that does not exist (BUG-005); the overhaul should replace it with a real output contract (section 4).

## 2. Gap inventory

### A. Reproducibility and provenance

**G-01 Generation is unseeded** (BUG-007; verified, code read).
`np_gen = np.random.default_rng()` has no seed or CLI option; `train.py`'s `Dataset` also has an unseeded `rng`.
The existing dataset cannot be regenerated bit-for-bit.
Target: the scheme in `ml-gw-search/SEEDING.md`.

**G-02 pycbc draws use global state** (BUG-007; verified, pycbc source read).
The noise draw (`frequency_noise_from_psd`) and the sky draw (`UniformSky.rvs`) use the legacy global `numpy.random`, so seeding `np_gen` alone leaves them random.
Target: one `numpy.random.seed` call per stream, never `seed=` inside the sample loop (`ml-gw-search/SEEDING.md`).

**G-03 No provenance in the output** (OBS-016; verified).
The generator writes two file attributes, `sample_length` and `sample_rate`, and nothing else.
The repackaged file has no attributes at all.
Missing: the seeds and keys, the arguments, the approximant, the noise source, the PSD, the whitening configuration, the library versions, the generator git SHA, the time and host of generation.
The step that produced `combined_repackaged.hdf` is not in any repository: `out/` is empty, and the script writes `out_repackaged.hdf`, not `combined_repackaged.hdf`.

**G-04 Parameters that shaped the data are not saved** (verified, code read).
- For `IMRPhenomXPHM`, both spin vectors are drawn and used, but the parameter row has no spin columns; the approximant itself is not stored.
- The PSD used for each sample is returned by the noise getter and then dropped, so with `--psd-file` or real noise the PSD index per sample is lost.
- The whitening constants (`segment_duration = 0.5`, `max_filter_length = 512`, `low_frequency_cutoff = 20`) are module constants and are not recorded.

**G-05 `injection_time` is stored as float32** (BUG-009).
Verified: every value in `combined_repackaged.hdf` is exactly float32-representable (computed 2026-10-19), and float32 spacing at GPS ≈ 1.24e9 is 128 s.
Verified (pycbc source read, 2026-10-19T21:16Z): `Detector.project_wave` defaults to `method='lal'`, which calls `lalsimulation.SimDetectorStrainREAL8TimeSeries` with the waveform epoch, so the antenna patterns and Doppler shifts change with the GPS time.
The stored `injection_time` therefore influences the data, it is not provenance only, and a value rounded by up to 64 s cannot reproduce the exact projection.
The `injection_time` head is already excluded in `heads_spec.py`, so the column is not a training target.
Decision (user, 2026-10-19T21:31Z): the float32 cast is inside the generator code and stays; the quantization of `injection_time` is documented in the contract, not corrected.
The driver cannot correct it from outside `gen.py`.

**G-06 The column labels are wrong in the sibling** (BUG-008).
The sibling's `PARAM_COLUMNS` lists `mass1, mass2` for columns 0 and 1 and `merger_time_gps` for column 7; the data holds `mchirp, q` and the injection time.
The repackaged file carries no labels, so meaning lives only in `heads_spec.py`.
Target: labels written as a dataset attribute, matching `heads_spec.py`, and checked against it when the file is loaded.
Status (2026-10-19T21:36Z): the list in the sibling's `train.py` was edited and compared with `heads_spec.py` by an AST read (old list: no match, new list: match); the attribute in a written file is queued in `VERIFY.md`.

### B. Output file content and format

**G-07 Two-step pipeline with hardcoded paths** (verified, code read).
`gen.py` writes raw `noises`, `waveforms`, `waveform_params`; the untracked `re_package_hdf.py` then reads a hardcoded `./out.hdf` and writes a hardcoded `out_repackaged.hdf` in the loader's layout.
The step that merged or renamed that output into `combined_repackaged.hdf` is unrecorded.
Target: one tracked step that writes the consumer layout directly, with provenance, without hardcoded paths.

**G-08 Noise-only samples are discarded** (OBS-014; verified).
Repackaging keeps `noises[:n_wv]`, so the output has only injection samples (25,000 training and 5,000 validation).
Whether the regenerated file needs a noise-only class is open (section 5, D-04).

**G-09 SNR handling is split and unlabeled** (verified, code read).
- `gen.py` stores unit-network-SNR whitened waveforms; `train.py` applies a fresh SNR per epoch from (5, 15).
- `re_package_hdf.py` bakes one SNR per sample from (7, 15) with the literal seed `132354`, and appends it as an unlabeled tenth column.
- Observed in the data: `snr` min 7.00001, max 14.9995, mean 10.98 (verified).
- The convention (`data = noise + snr × unit-SNR waveform`, both whitened) is stored nowhere in the file.

**G-10 Layout, dtype, and units are implicit** (verified).
Detectors are separate datasets (`h1`, `l1`, `h1_waveform`, `l1_waveform`), so the detector order lives in dataset names; `params` is float64 though the source values are float32; units and ranges are not stored.

**G-11 No held-out test split** (verified, code read).
`gen.py` writes only `training` and `validation`, and the loader reads those.
Unknown: how the final test set is meant to be formed for the q_value work.

**G-12 Incomplete files look complete** (verified, code read).
`main()` saves the training group, then generates the validation group and appends it.
All samples are held in memory until saving, with no incremental write and no resume.
A failure during validation leaves a readable file with no `validation` group and no completion marker.
Target: write to a temporary name and rename on success, plus a final `complete` attribute.

### C. Design choices to settle (not defects)

**G-13 Mass prior** (verified from code and data).
Component masses are uniform on [10, 50] with `m1 = max`, `m2 = min`, which makes `q` skewed: observed q min 0.2027, max 0.99998, mean 0.650; 7.6% of training samples have `q ≥ 0.95` and 5.4% have `q < 0.3`.
Observed `mchirp` range 8.84 to 43.45.
The worst-performing cell in every q run was `q_high × mchirp_low` (OBS-003), so coverage there is a design lever.

**G-14 Angle priors** (verified from code and data).
`inclination`, `coa_phase`, and `pol_angle` are all uniform on [0, 2π]; observed inclination max is 6.283.
Inferred: the physically isotropic prior is `cos ι` uniform with `ι` in [0, π]; the current prior folds each inclination onto a mirrored one, which the paper already discusses (`006_results.tex`, comment at line 88).
`heads_spec.py` treats polarization with period π, so the [0, 2π] draw is redundant for that head.
Whether to keep the legacy priors or switch is the user's call.

**G-15 Merger-time window is narrow, hardcoded, and its comment is stale** (verified).
The merger is placed uniformly in `[sample_duration − 0.4, sample_duration − 0.2]` seconds, so for 2 s windows `merger_time` lies in [1.6, 1.8] s, and the data show min 1.6, max 1.8.
The code comment says "between 0.5 s and 0.7 s", which does not match the arithmetic (1 s windows give 0.6 to 0.8 s).
The `merger_time` head therefore predicts within a 0.2 s range of a 2 s window.

**G-16 Noise source and PSD** (verified, code read).
The default is whitened Gaussian noise from `aLIGOZeroDetHighPower`, the same PSD for every detector, stationary.
The real-noise path pairs whitened waveforms with unwhitened real noise (OBS-015, inferred from code) and does not save the PSD index (G-04).

**G-17 Detectors and sample length.**
The existing data use H1 and L1 and 4096-sample (2 s, 2048 Hz) windows (verified, HDF shapes); `gen.py` defaults to one detector and 2048 samples, so both are command-line choices that must be recorded.

### D. Engineering

**G-18 Generation is coupled to the training stack** (verified, imports read).
`gen.py` imports `Dataset` from `train.py`, which imports `torch` and the network code from `apply.py`, only to write an HDF.
Target: a generator that needs only numpy, h5py, and pycbc.

**G-19 No tests, no package structure** (verified, listing).
The sibling has no test suite, and scripts import each other as top-level modules (`from apply import …`).
The determinism check in `ml-gw-search/SEEDING.md` would be an ad-hoc check unless a test is added.

**G-20 Artifacts and scripts untracked** (verified, `git status`).
`re_package_hdf.py`, the pipeline note, and the 1.9 GB dataset are untracked in the sibling, and `.gitignore` does not exclude `*.hdf`.

**G-21 Documentation** (OBS-017, BUG-005).
Two diverging copies of the pipeline note exist, and the output contract the code cites (`structure.md`) was never written.
Target: one contract document, generated or checked against the file, so it cannot drift.

### E. Conventions not yet verified end to end (found while reading the generation path, 2026-10-19T21:16Z)

**G-22 SNR scale versus noise scale** (unknown).
The unit-SNR waveform is normalised with `pycbc.filter.sigmasq` against the unwhitened PSD, while the noise comes from `frequency_noise_from_psd` and both are whitened by the same regularised PSD.
Nothing in the generator checks that `noise + snr × waveform` then has a matched-filter SNR of `snr` against that noise.
A known-good check is proposed in section 6.

**G-23 Long waveforms are cut at the window start** (inferred; no taper read).
The projected strain is sliced to the 4608-sample colored window ending 0.2 to 0.4 s after the merger, so inspirals longer than the 1.7 to 1.9 s before the merger are cut abruptly at the slice start, and the normalising SNR is computed on that truncated waveform.
By a leading-order chirp-time estimate (not checked against pycbc), this affects chirp masses below about 18 M☉.
The first 0.125 s of the whitened series is trimmed, which hides only part of the edge.

**G-24 `merger_time` is a geocentric position** (inferred from `project_wave`'s documented time shift).
The label is `time_placement − 0.125 s`, the geocentric merger time in the whitened window; each detector sees the merger up to about 10 ms earlier or later, which is not small next to the 0.058 s spread of the label.
The meaning should be stated in the contract.

## 3. What the current dataset is (verified, 2026-10-19)

`combined_repackaged.hdf` (1.97 GB, byte-identical copies in both repositories): groups `training` (25,000) and `validation` (5,000), each with `h1`, `l1`, `h1_waveform`, `l1_waveform` as float32 `(N, 4096)` and `params` as float64 `(N, 10)`.
No attributes at the file or dataset level.
Parameter ranges are in G-13, G-14, G-15, and G-09; `injection_time` runs 1.23817e9 to 1.25398e9.

## 4. Draft output contract (for the regenerated file)

Firm items follow from the gaps above; items marked "open" depend on section 5.

| Item              | Content                                                                                                                                                                                                        | Status                               |
|-------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------|
| Splits            | `training` 100,000, `validation` 20,000, `test` 10,000; half of each is noise-only                                                                                                                             | firm (D-03)                          |
| Strain            | noise + scaled waveform per detector; layout `(N, L)` per detector or `(N, L, det)`                                                                                                                            | open (D-06)                          |
| Waveform | the strain, plus the whitened unit-network-SNR waveform per detector (the scaled waveform and the pure noise are derived from `snr`); the baked `snr` is a labeled column | firm (D-05, storage form decided 2026-10-19T21:31Z) |
| Noise-only class | present, half of every split, in separate datasets (`noise_h1`, `noise_l1`) so that `h1`, `l1`, `params` stay injection-only and the loader is unchanged | firm (D-04, D-12 decided 2026-10-19T21:36Z) |
| `params`          | `(N, 10)` or more columns; float64; an attribute `columns` that matches `heads_spec.py`                                                                                                                        | firm                                 |
| Extra columns     | spins (for `IMRPhenomXPHM`)                                                                                                                                                                                    | open (D-01)                          |
| File attributes   | schema version; sample length and rate; detectors; approximant; noise source and PSD; `low_frequency_cutoff`; whitening constants; SNR range and the `data = noise + snr × waveform` convention; whitened flag | firm                                 |
| Seed attributes   | `master_seed`, `file_index`, the `(file_index, stream, purpose)` keys                                                                                                                                          | firm (`ml-gw-search/SEEDING.md`)                  |
| Run attributes    | generator arguments, git SHA, library versions, UTC time, host                                                                                                                                                 | firm                                 |
| Completion        | written under a temporary name, renamed on success, with a `complete` attribute                                                                                                                                | firm                                 |
| Contract document | one file describing the layout, checked against a generated file                                                                                                                                               | firm                                 |

## 5. Decisions to settle

| ID   | Question                                                                                             | Status                                                                                                                                                                                                                                                                                                 |
|------|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| D-01 | Which approximant, and are spins recorded?                                                           | **decided 2026-10-19T21:10Z (user):** `IMRPhenomD`, spinless; the approximant is still recorded as a file attribute                                                                                                                                                                                    |
| D-02 | Mass prior and angle priors: keep the legacy ones (G-13, G-14) or change them? | **decided 2026-10-19T21:04Z (user), narrowed 2026-10-19T21:20Z by the freeze:** keep the legacy priors exactly; they are not made configurable |
| D-03 | Splits and counts: sample counts per split, a test split, and sharding into files                    | **decided 2026-10-19T21:10Z (user):** training 100,000 (50,000 signal + noise, 50,000 noise only), validation 20,000 (10,000 + 10,000), test 10,000 (5,000 + 5,000); 10,000-sample shards (5,000 + 5,000 each) generated in parallel, 13 in all (10 training, 2 validation, 1 test), then a merge step |
| D-04 | Keep a noise-only class?                                                                             | **decided 2026-10-19T21:10Z (user):** yes, half of every split; written by a new `re_package_hdf_with_noise.py`, with the original `re_package_hdf.py` left unchanged; the layout of the noise-only rows is open (D-12)                                                                                |
| D-05 | SNR: baked at generation (reproducible) or applied at training time (fresh per epoch), and its range | **decided 2026-10-19T21:04Z (user):** bake one SNR per sample at generation and store the unit-SNR waveform; the range stays (7, 15). **Storage form decided 2026-10-19T21:31Z (user):** the smallest form, strain plus the unit-SNR waveform plus the `snr` column (two float32 arrays per injection sample, not three) |
| D-06 | Detectors, sample length, and the strain layout | **decided 2026-10-19T21:10Z (user):** H1 and L1, 4096-sample windows at 2048 Hz, datasets `h1` / `l1` / `params` as the loader expects, recorded in the file attributes (as run-time values, not new settings) |
| D-07 | Noise source: Gaussian aLIGO design PSD or real noise (OBS-015 applies)                              | **decided 2026-10-19T21:04Z (user):** Gaussian aLIGO design PSD; the real-noise path is not part of this overhaul                                                                                                                                                                                      |
| D-08 | Merger-time window: keep [1.6, 1.8] s or widen and parameterize | **decided 2026-10-19T21:20Z (freeze, confirmed 21:36Z):** keep [1.6, 1.8] s; it is not parameterized and is documented as a property of the generator |
| D-09 | A dataset-version level in the seed key (`ml-gw-search/SEEDING.md`, open items) | **decided 2026-10-19T21:53Z (user):** yes, `dataset_id` is the first element of the generation key `(dataset_id, split, shard, purpose)` |
| D-10 | How training-side seeds relate to the master seed; the seed count and materiality floor (OBS-008) | **decided 2026-10-19T22:06Z (user):** each repository has its own, separate master seed; training seeds are derived from this repository's master with `seed_index` (M = 10), and the tests are seeded the same way, keyed by test name (this repository's `SEEDING.md`, OBS-023, OBS-024); N = 8, K = all five trunks, floors proposed (OBS-008) |
| D-12 | Layout of the noise-only rows in the merged file, and what the consumer does with them (section 5b) | **decided 2026-10-19T21:36Z (user):** the loader works unchanged, so the injection rows keep `h1`, `l1`, `params` exactly as now and the noise-only rows go in separate datasets (`noise_h1`, `noise_l1`) in the same group |
| D-11 | Where the generator code and the output contract live | **decided 2026-10-19T21:36Z (user, with the plan):** new code in `ml-gw-search` (the frozen files untouched), the output contract and its check in this repository |

### 5a. Consequences of the decisions so far (2026-10-19T21:10Z)

- **Size (verified arithmetic from the current file, 65,616 bytes per injection sample; a noise-only sample needs only its strain, 32,768 bytes):**
  130,000 samples come to about 6.4 GB, of which training is about 4.9 GB and each 10,000-sample shard about 0.49 GB.
  If the noise-only rows also carried an all-zero waveform array, the total would be about 8.5 GB.
- **Scaled and unit waveform (decided, D-05, 2026-10-19T21:31Z):** the scaled waveform is `snr × unit waveform`, so storing both would add about 50% for the injection rows, and about 33% for the whole file with its noise-only half (8.5 GB instead of 6.4 GB), for no new information.
  The user chose the smallest form: store the strain and the unit-SNR waveform, and derive the scaled waveform and the pure noise (`strain − snr × unit`) from the labeled `snr` column.
  The two-array forms tie in size (strain plus unit waveform, or strain plus scaled waveform); the unit waveform is kept because it is the generator's own output and allows another SNR to be re-applied.
  The derived noise carries float32 rounding error, which is negligible next to the noise amplitude; this repository reads no waveform array, only `h1`, `l1`, and `params`.
- **Generation work (inferred from `gen.py`):** only the injections need pycbc waveform generation and whitening of a signal, so the heavy work is 65,000 injections, half of what 130,000 injections would be; the noise-only half is whitened Gaussian noise only.
  No timing has been recorded (unknown), and generation must not run on `thinkpad-t530`.
- **Memory:** a 10,000-sample shard holds about 0.5 GB in lists, and about twice that at the `np.stack` step, which is what makes sharding workable (G-12); one 100,000-sample group in a single process would not be.
- **Loader (verified, `loader.py` docstring and arithmetic):** `load_arrays` reads a whole split eagerly, with a docstring assumption of about 800 MB of training strain (25,000 samples); 100,000 rows are about 3.3 GB of strain, or about 1.6 GB for only the 50,000 injection rows, before the `astype` copy.
- **Seeds (`ml-gw-search/SEEDING.md` revised 2026-10-19T21:46Z):** a shard holds one split, the key is `(dataset_id, split, shard, purpose)` with the shard counted within its split, and each shard is generated by its own process.
- **Comparability:** with legacy priors, IMRPhenomD, and Gaussian noise, the signal half is a larger draw from the same distribution as `combined_repackaged.hdf`.

### 5b. Noise-only rows and the consumer (D-12, decided 2026-10-19T21:36Z)

`load_arrays` returns every row of a split together with `params`, and the training code builds targets from `params`.
Noise-only rows have no physical parameters, so they cannot sit among the injection rows without the consumer filtering them out.
Two layouts are possible:

- **Separate datasets (proposal):** keep `h1`, `l1`, `params` for the injection rows exactly as now, so the loader works unchanged, and put the noise-only rows in their own datasets (for example `noise_h1`, `noise_l1`) in the same group.
- **Interleaved with a label:** one set of datasets with a per-row class label, `params` set to NaN for noise-only rows, and the loader (and every head's loss) changed to select the signal rows.

The first layout is the smaller change on the consumer side; the second suits a classifier that wants one array.

Decision (2026-10-19T21:36Z, user): the loader works unchanged, so the separate-datasets layout is used.

## 6. Verification plan

- The determinism check in `ml-gw-search/SEEDING.md`, on a shrunken workload, on a machine other than `thinkpad-t530`, with the user's go-ahead.
- SNR consistency (G-22): for a batch of whitened noise-only samples and whitened unit-SNR waveforms from the same chain, the summed squared waveform divided by the per-sample noise variance should be about 1.
  Falsified by: a ratio that differs from 1 by more than the batch spread.
  The check must first read a known case correctly (a white-noise series with a waveform of known matched-filter SNR).
- Known-good case for the new generator: with the legacy settings, the parameter ranges and per-column summary in section 3 and G-09 to G-15 are reproduced in distribution (a regenerated file cannot match bit-for-bit).
  Falsified by: a range or fraction outside the statistical spread of a same-size draw.
- Contract check: a script loads a generated file and asserts the layout, the column labels against `heads_spec.py`, and the required attributes; a file missing any of them fails.
- Open items stay in `BUGS.md` until verified; unconfirmed items go to `VERIFY.md`.

## 7. Revised approach under the freeze (proposed 2026-10-19T21:20Z; confirmed by the user 2026-10-19T21:36Z)

### Frozen: not edited, not parameterized

`mlgwsc-1/gen.py` (the noise getters, `generate_dataset`, the constants, the priors, the merger placement, the SNR normalisation), `apply.whiten`, and `apply.regularize_psd`.
Their blob hashes are recorded in every generated file, and a test pins them, so a change has to be deliberate.

### Added beside them, in `ml-gw-search`, without editing `gen.py`

| Need                                        | Mechanism                                                                                                                                                                                                                                                                  | Touches the backend?                                                            |
|---------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| Seeding (G-01, G-02)                        | A new driver rebinds the module global `gen.np_gen` before calling `generate_dataset` (the function reads it as a global at call time, by code read), and calls `numpy.random.seed` once per shard for pycbc, per `ml-gw-search/SEEDING.md`                                             | No: the draws come from the same distributions                                  |
| One split per 10,000-sample shard           | The driver repeats the few orchestration lines of `gen.main()` (detectors, colored length, `GaussianNoiseGetter`) and calls `generate_dataset` unchanged; `gen.main()` itself cannot produce a shard without a validation group, because `np.stack` of an empty list fails | No, but the driver can drift from `main()`; a check compares them on a tiny run |
| Provenance (G-03, G-04)                     | Attributes written to each shard after `Dataset.save`; constants read from the `gen` module at run time                                                                                                                                                                    | No                                                                              |
| SNR baking and noise-only rows (G-08, G-09) | The new `re_package_hdf_with_noise.py`; `re_package_hdf.py` stays unchanged                                                                                                                                                                                                | No                                                                              |
| Merge, completion marker (G-07, G-12)       | A new tracked merge step writing to a temporary name and renaming on success                                                                                                                                                                                               | No                                                                              |
| Contract check (G-21)                       | A script in this repository comparing a generated file with `heads_spec.py`                                                                                                                                                                                                | No                                                                              |

### Decisions affected

- **D-02:** legacy priors only; nothing is made configurable.
- **D-08:** the merger window is not parameterized; [1.6, 1.8] s is documented as a property of the generator.
- **D-11:** the new code goes in `ml-gw-search`, and `gen.py`, `apply.py`, and the thesis scripts are not edited.
- **G-18:** the torch coupling stays, because refactoring it would touch the paper's files; the sibling's `.venv` already has torch.
- **G-06 (decided and done 2026-10-19T21:36Z, user):** `PARAM_COLUMNS` in the sibling's `train.py` was edited, as a label-level change (BUG-008, now `FIXED`; uncommitted in the sibling); the list now matches the first nine keys of `heads_spec.PARAM_COLUMNS`, and the stale docstring and help text in `train.py` that still name the old columns were left untouched.
- **G-05 (decided 2026-10-19T21:31Z, user):** the `np.float32` cast stays in `gen.py`, and the quantization of `injection_time` is documented, not fixed.
- **G-22 to G-24:** documented, and G-22 is verified without changing anything.
