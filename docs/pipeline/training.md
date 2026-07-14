# Training pipeline

## Head registry — `gwml/heads_spec.py`

The single source of truth. Each `HeadSpec` binds, for one physical
parameter: params column, transform kind, output dim, activation, and loss
key. Valid heads: `mchirp`, `q`, `merger_time`, `snr` (the defaults), plus
`inclination`, `coa_phase`, `polarization_angle`, `declination`, `ra`.
`injection_time` is deliberately not a head (absolute GPS time is not
learnable from a 2 s window) — requesting it raises with the valid list.

Experiment YAMLs choose *which* heads (`model.heads`); they cannot override
the per-head bindings. That is intentional: it makes "periodic parameter
paired with a plain loss on raw angles" unrepresentable.

## Model assembly

`build_model(trunk, trunk_cfg, head_cfg, heads)`:

1. `build_trunk(trunk, trunk_cfg)` — registry lookup; every trunk returns
   `(input (4096, 2), pooled features)` and starts with `input_bn`
   (see `../models/<trunk>.md` for each architecture).
2. `attach_heads(...)` — per active head: `Dense(hidden_units, relu)` →
   optional `Dropout` → `Dense(spec.dim, spec.activation)`, layer named after
   the head. `head_cfg.bounded: false` downgrades sigmoid/tanh outputs to
   linear (an A/B lever; bounded stays on by default). `hidden_units`,
   `dropout`, and `l2` all default globally but can be overridden for one
   head via `head_cfg.per_head.<name>` — e.g. to shrink/regularize a head
   that overfits faster than its trunk-mates.

## `MultiHeadTrainer` — `gwml/training/losses.py`

A `keras.Model` subclass wrapping the functional model with a custom
multi-task loss (custom `train_step`/`test_step`; weights-only
checkpointing, the model is rebuilt from YAML).

**Per-head loss** — resolved from `HeadSpec.loss` (currently Huber for every
head; periodic heads are (sin, cos) pairs so Huber on the pair is correct).
Unknown loss keys fail at construction, not mid-training.

**Head balancing** (`loss.weighting`):

- `uncertainty` (default) — Kendall, Gal & Cipolla (2018):
  `total = Σ_h exp(−s_h)·L_h + s_h` with one learnable log-variance `s_h`
  per active head. The learned weights are logged as `weight_<head>`; they
  tell you which parameters the data actually constrains.
- `fixed` — `loss.fixed_weights: {head: w}`, default 1.0 each.

**Mean-collapse safeties** (GW regression's classic failure — every head
predicting the target mean with ~0 variance):

| safety | always on? | mechanism |
|--------|-----------|-----------|
| `r2_<head>` metric | yes | `keras.metrics.R2Score` per head; collapse ⇒ → 0 |
| `std_ratio_<head>` metric | yes | exact epoch-level `std(pred)/std(true)` (custom `StdRatio`); collapse ⇒ → 0 |
| log-var clamp | yes | `s_h` constrained to ±`loss.log_var_clamp` (default 3.0) so a head's effective weight never drops below `exp(−3) ≈ 5%` — uncertainty weighting cannot entrench collapse |
| variance penalty | opt-in | `loss.variance_penalty: λ` adds `λ·(std(pred)−std(true))²` per head per batch |
| LR warmup | opt-in | `optim.warmup_epochs: N` ramps LR linearly, dodging the early mean-prediction minimum |
| LiveScatter PNGs | yes | visual: a collapsed head is a horizontal band at the target mean |

## Callbacks — `gwml/training/callbacks.py`

Built by `train.py` in this order:

1. **`LiveScatterCallback`** — every `train.diagnostics_every_n` epochs,
   predicts on the first `train.scatter_subset` validation samples and writes
   `runs/<name>/scatter/epoch_XXXX.png`: pred-vs-true per head, physical
   units, y=x diagonal, wrap-aware MAE annotated.
2. **`DiagnosticSubsetsCallback`** — same cadence, appends to
   `runs/<name>/diagnostics.csv`. Rows: one per (epoch, subset); subsets are
   `full`, SNR terciles, mchirp halves, merger-time halves, q terciles
   (`q_low`/`q_mid`/`q_high`), and the q-tercile × mchirp-half cross-tab
   (`q_{low,high}_mchirp_{low,high}`) that isolates the near-equal-mass /
   low-chirp-mass regime. Columns: `epoch, subset, n, mae_<head>…,
   r2_<head>…, std_ratio_<head>…` — all physical units, wrap-aware. Errors
   should order cleanly by SNR; if they don't, be suspicious.
3. **`ModelCheckpoint`** — `best.weights.h5` on `val_loss`, weights only.
4. **`CSVLogger`** (`history.csv`) and **TensorBoard** (`tb/`).
5. **LR schedule** — `optim.schedule.type: plateau` (default;
   `ReduceLROnPlateau` with `factor`/`patience`/`min_lr`) or `step`
   (`step_epochs`/`gamma`). With warmup: plateau gets a `WarmupLR` callback
   that ramps then *stops touching the optimizer*; step folds warmup into
   the schedule function (two callbacks setting LR every epoch would fight).
6. **`EarlyStopping`** — only if `train.early_stopping: true`
   (`early_stopping_patience`, restores best weights).

## Config reference (YAML)

| key | default | meaning |
|-----|---------|---------|
| `name` / `run_dir` | — / `runs/<name>` | experiment id, output base; training writes each run to a timestamped child like `runs/<name>/20260713_153012` |
| `data.path` | — | HDF5 file |
| `data.batch_size` | 128 | batch size |
| `data.max_samples` | null | truncate splits (smoke/debug) |
| `model.trunk` | — | registry name (see `../models/`) |
| `model.trunk_cfg` | `{}` | trunk knobs (per-trunk doc) |
| `model.heads` | core four | active output heads |
| `model.head_cfg.hidden_units` | 64 | per-head MLP width |
| `model.head_cfg.bounded` | true | sigmoid/tanh output activations |
| `model.head_cfg.dropout` | 0.0 | dropout after the hidden Dense, applied to every head |
| `model.head_cfg.l2` | 0.0 | L2 kernel regularizer on both Dense layers, every head |
| `model.head_cfg.per_head.<name>` | — | overrides `hidden_units`/`dropout`/`l2` for one head only |
| `loss.huber_delta` | 1.0 | Huber transition point |
| `loss.weighting` | uncertainty | `uncertainty` \| `fixed` |
| `loss.fixed_weights` | 1.0 each | only used with `fixed` |
| `loss.log_var_clamp` | 3.0 | bound on \|s_h\|; accepts `{default: 3.0, <head>: tighter}` to stop one head's weight from riding every other head's clamp ceiling |
| `loss.variance_penalty` | 0.0 | λ of the spread penalty |
| `loss.snr_weight_alpha` | null | per-sample `(SNR/10)^α` weights |
| `optim.lr` | 1e-3 | Adam learning rate |
| `optim.warmup_epochs` | 0 | linear LR warmup |
| `optim.schedule` | plateau | see callbacks above |
| `train.epochs` | 50 | max epochs |
| `train.seed` | 42 | `keras.utils.set_random_seed` |
| `train.deterministic_ops` | false | `enable_op_determinism` (slower, strict) |
| `train.diagnostics_every_n` | 5 | cadence of callbacks 1–2 |
| `train.scatter_subset` | 1000 | samples in the live scatter |
| `train.early_stopping` | false | off by default |
| `train.verbose` | 2 | Keras fit verbosity |

## Run directory contents

```
runs/<name>/
├── transforms.json      # active heads + fitted stats — needed for inference
├── best.weights.h5      # best val_loss checkpoint (weights only)
├── final.weights.h5     # last epoch
├── history.csv          # per-epoch: loss, mae_*, r2_*, std_ratio_*, weight_*, val_*
├── diagnostics.csv      # per-subset physical-unit metrics (see above)
├── scatter/epoch_*.png  # live scatter grids
└── tb/                  # TensorBoard logs
```

To reload: `build_trainer(cfg)` → call once on a dummy batch to build
variables → `load_weights` (this is what `scripts/evaluate.py` does).
