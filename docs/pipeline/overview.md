# Pipeline overview

End-to-end flow from HDF5 to trained model and evaluation artifacts. Each
stage links to its detailed doc.

```
combined_repackaged.hdf                                  [data.md]
        │  load_arrays(path, split)
        ▼
strain (N, 4096, 2) float32        params (N, 10) float64
        │                                  │
        │                 TargetTransforms(heads).fit(train_params)
        │                                  │  transform()
        │                                  ▼
        │                  targets: {head: (N, dim) float32}
        │                                  │
        └────────────── make_dataset ──────┘
                             │  (x, y_dict[, snr sample_weight])
                             ▼
                    tf.data.Dataset (shuffle/batch/prefetch)
                             │
                             ▼
        build_model(trunk, trunk_cfg, head_cfg, heads)   [../models/*.md]
          trunk registry ──► trunk (starts with input BatchNorm)
          heads_spec     ──► one named output head per active head
                             │
                             ▼
        MultiHeadTrainer(base, loss_cfg, heads)          [training.md]
          Huber per head · uncertainty/fixed weighting
          collapse safeties: r2/std_ratio metrics, log-var clamp
                             │  fit(callbacks=...)
                             ▼
        runs/<name>/                                     [training.md]
          transforms.json      best.weights.h5   final.weights.h5
          history.csv          diagnostics.csv   scatter/epoch_*.png   tb/
                             │
                             ▼
        scripts/evaluate.py                              [evaluation.md]
          metrics_<split>.csv  scatter_<split>.png  residuals_snr_<split>.png
```

## Module map

| module | responsibility |
|--------|----------------|
| `gwml/heads_spec.py` | **single source of truth** for every possible output head: params column, transform, dim, activation, loss binding |
| `gwml/data/loader.py` | HDF5 → numpy → `tf.data`; optional SNR sample weights |
| `gwml/data/transforms.py` | executes the spec's transforms; JSON persistence; wrap-aware error helpers |
| `gwml/models/registry.py` | trunk name → builder lookup |
| `gwml/models/trunks/*` | the five trunk architectures (docs in `../models/`) |
| `gwml/models/heads.py` | attaches the active heads to any trunk's feature vector |
| `gwml/training/losses.py` | `MultiHeadTrainer`: multi-task loss, collapse safeties, custom train/test steps |
| `gwml/training/callbacks.py` | `LiveScatterCallback`, `DiagnosticSubsetsCallback`, `WarmupLR` |
| `gwml/training/train.py` | YAML config → full experiment (`run_experiment`) |
| `gwml/evaluation/` | physical-unit metrics and plots |

## Design invariants

1. **The YAML picks *what*, the code defines *how*.** Configs choose trunk,
   heads, epochs, LR; how a head is transformed, activated, and penalized is
   frozen in `heads_spec.py` and cannot be overridden per experiment.
2. **Everything reported to a human is in physical units.** Normalized space
   exists only between `TargetTransforms.transform()` and `inverse()`.
   Training-log metrics (`mae_*`, `r2_*`, `std_ratio_*`) are the exception —
   they live in normalized space for speed; the diagnostics CSV, scatter PNGs,
   and evaluation CSVs are all physical.
3. **Models are self-contained.** Normalization is the in-model input
   BatchNorm; a checkpoint plus its YAML plus `transforms.json` fully
   reproduces predictions in physical units.
4. **A trunk without its doc doesn't enter the zoo**, and pipeline changes
   should keep these docs current.

## Typical session (lab machine)

```bash
pytest -m "not slow"                      # quick suite, ~a minute
pytest                                    # + overfit-one-batch per trunk
python scripts/train.py configs/smoke.yaml    # pre-flight: callbacks fire
python scripts/train.py configs/resnet1d.yaml
python scripts/evaluate.py configs/resnet1d.yaml --split validation
```
