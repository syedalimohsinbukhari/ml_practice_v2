# Data pipeline

## Source file

`combined_repackaged.hdf` (see `structure.md`): `training/` (25,000 samples)
and `validation/` (5,000), each with `h1`, `l1` (whitened signal+noise,
(N, 4096) float32), `h1_waveform`, `l1_waveform` (clean signal, unused for
now), and `params` (N, 10) float64. The window is 2 s at 2048 Hz; the merger
sits 1.6–1.8 s in; every window contains a signal (SNR 7–15).

## Loading — `gwml/data/loader.py`

`load_arrays(path, split, max_samples=None)` reads a whole split into RAM
(~800 MB for training strain) and stacks the detectors channel-wise:

- strain `(N, 4096, 2)` float32, channel order **[h1, l1]** (asserted by
  `tests/test_data_integrity.py`)
- params `(N, 10)` float64, columns per `PARAM_COLUMNS` in `heads_spec.py`

`max_samples` truncates the split — used by `configs/smoke.yaml` and the
overfit tests.

The strain is fed to the model **raw**: normalization is the
`BatchNormalization` layer every trunk starts with (`input_bn`), so saved
models are self-contained. (Alternative kept in mind: a `Normalization` layer
adapted on the training split — fixed statistics at inference; one-line swap
in the trunks if BatchNorm's moving averages ever misbehave.)

## Target transforms — `gwml/data/transforms.py`

`TargetTransforms(heads)` executes whatever the head registry declares
(`gwml/heads_spec.py` is the single source of truth — see `training.md`'s
"Head registry" section; summary):

| kind | heads | forward | inverse |
|------|-------|---------|---------|
| `LOG_ZSCORE` | mchirp | `(log v − μ)/σ`, μ/σ from **training split only** | `exp(vσ + μ)` |
| `ZSCORE` | snr | `(v − μ)/σ` | `vσ + μ` |
| `UNIT_AFFINE` | q, merger_time, declination | fixed documented bounds → [0, 1] | affine back |
| `PERIODIC` | ra, coa_phase, inclination, polarization_angle | angle → `(sin θ, cos θ)`, θ = 2π·v/period | `atan2` → [0, period) |

- `fit(train_params)` computes the z-score stats; affine/periodic transforms
  are stateless (fixed constants from the spec).
- `to_json` / `from_json` persist `{"heads": [...], "stats": {...}}` to the
  run directory, so evaluation always reconstructs the same head list and
  statistics — never re-fit transforms on validation data.
- `signed_error(head, true, pred)` / `abs_error(...)` compute physical-unit
  residuals, **wrap-aware** for periodic heads (0.05 rad vs 6.23 rad is an
  error of 0.1, not 6.18). Every reported MAE/RMSE goes through these.

Why these choices: MSE in log-space ≈ fractional error, which is what matters
for mchirp — so no custom loss is needed there. The (sin, cos) encoding keeps
the wraparound discontinuity out of the loss entirely; `polarization_angle`
uses period π because strain is invariant under ψ → ψ+π.

## Dataset construction

`make_dataset(strain, params, transforms, batch_size, shuffle, seed,
snr_weight_alpha)` builds `tf.data.Dataset.from_tensor_slices` over
`(x, y_dict)` or `(x, y_dict, sample_weight)`:

- `y_dict` maps head name → `(N, dim)` float32 (dim 2 for periodic heads);
  head names match the model's output layer names, so Keras routes by dict.
- `shuffle` uses a full-length buffer with the config seed,
  `reshuffle_each_iteration=True`. Validation datasets are not shuffled.
- `snr_weight_alpha` (config `loss.snr_weight_alpha`, default null) attaches
  per-sample weights `(SNR/10)^α` that the trainer applies to every head's
  loss equally — the "don't burn capacity on unrecoverable labels"
  experiment; leave off by default.

## Targeted oversampling (Phase 3.1)

`run_experiment` supports `data.augmentation.oversample` — a dict mapping
subset names to integer duplication factors. Before transforms are fitted,
rows in the named subsets are duplicated the specified number of times.
Subset masks come from `build_subset_masks()` in `loader.py` (the same
function used by `DiagnosticSubsetsCallback`). This targets the
`q_high × mchirp_low` cell (~11% of data, val_r2_q ≈ −22 across all
trunks) — the physically hardest and most data-starved regime.

```yaml
data:
  augmentation:
    oversample:
      q_high_mchirp_low: 2   # duplicate these rows 1 extra time
```

Oversampling happens before `TargetTransforms.fit()`, so z-score statistics
are computed on the augmented distribution (which is the desired behavior —
the augmented set is the effective training population).
