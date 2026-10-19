# Training pipeline

## Head registry — `gwml/heads_spec.py`

The single source of truth. Each `HeadSpec` binds, for one physical
parameter: params column, transform kind, output dim, activation, and loss
key.
Valid heads: `mchirp`, `q`, `merger_time`, `snr`, `inclination`, `coa_phase`, `polarization_angle`, and `sky_position`.
The defaults (`DEFAULT_HEADS`) are `mchirp`, `merger_time`, `snr`, `sky_position`, `coa_phase`; `q` was shelved from the defaults (commit `8b2eb41`, 2026-07-15) but stays registered and can be re-added via `model.heads`.
`sky_position` is one joint head over (declination, RA): a 3-D unit vector on the sphere (`SPHERICAL_UNIT_VECTOR`) trained with a closed-form von Mises-Fisher NLL (`loss: "vmf"`).
It replaced the earlier independent `ra` and `declination` heads, which are no longer in the registry.
`injection_time` is deliberately not a head (absolute GPS time is not
learnable from a 2 s window) — requesting it raises with the valid list.

Experiment YAMLs choose *which* heads (`model.heads`); they cannot override
the per-head bindings. That is intentional: it makes "periodic parameter
paired with a plain loss on raw angles" unrepresentable.

## Model assembly

`build_model(trunk, trunk_cfg, head_cfg, heads)`:

1. `build_trunk(trunk, trunk_cfg)` — registry lookup; every trunk returns
   `(input (4096, 2), pooled features)` and starts with `input_bn`
   (see `../models/<trunk>.md` for each architecture). A trunk may optionally
   return a third element — a dict of extra feature tensors — which heads can
   connect to via `head_cfg.per_head.<name>.branch` (used by `cnn_attention`
   to give q pre-pooling transformer features; see Phase 3.2).
2. `attach_heads(...)` — per active head: `Dense(hidden_units, relu)` →
   optional `Dropout` → `Dense(spec.dim, spec.activation)`, layer named after
   the head. `head_cfg.bounded: false` downgrades sigmoid outputs to
   linear (an A/B lever; bounded stays on by default).
   Only `UNIT_AFFINE` heads use sigmoid; `PERIODIC` heads are linear on the (sin, cos) pair since 2026-07-18 (commit `6839b47`).
   `sky_position` emits two flat outputs, `sky_position_mu_raw` (3) and `sky_position_kappa_raw` (1), consumed jointly by the vMF loss.
   `hidden_units`,
   `dropout`, `l2`, `sigmoid_bias`, and `branch` all default globally but can
   be overridden for one head via `head_cfg.per_head.<name>`:
   - `hidden_units`/`dropout`/`l2` — regularize a head that overfits
   - `sigmoid_bias` — nudge the output bias away from zero for sigmoid heads
     (useful for reviving a saturated/dead sigmoid; Phase 2.5)
   - `branch` — connect this head to a named extra feature tensor from the
     trunk instead of the global pooled features (Phase 3.2)

   The `per_head` overrides, `sigmoid_bias`, and `branch` machinery is retained in the code (the `per_head` overrides are covered by `tests/test_heads_spec.py`), but no config in `configs/` currently uses it.
   The q-specific settings from Phase 2/2.5/3 (`log_var_clamp: {q: 1.0}`, q-head regularization, `q_tokens` branch, `q_high_mchirp_low` oversampling) were removed from the configs when q was shelved; `resnet1d.yaml` keeps `warmup_epochs: 5`.
   Results and rationale are in `q_head_action_plan.md`, `q_head_run_comparison.md`, and `phase2_5_3_run_results.md`.

## `MultiHeadTrainer` — `gwml/training/losses.py`

A `keras.Model` subclass wrapping the functional model with a custom
multi-task loss (custom `train_step`/`test_step`; weights-only
checkpointing, the model is rebuilt from YAML).

**Per-head loss** — resolved from `HeadSpec.loss`: Huber for every head except `sky_position`, which uses the vMF NLL (`vmf_nll_loss`).
Periodic heads are (sin, cos) pairs, so Huber on the pair is correct.
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
| `r2_<head>` metric | yes | `keras.metrics.R2Score` per head; collapse ⇒ → 0; not tracked for the vMF `sky_position` head |
| `std_ratio_<head>` metric | yes | exact epoch-level `std(pred)/std(true)` (custom `StdRatio`); collapse ⇒ → 0; not tracked for `sky_position`. `StdRatio` flattens its inputs, so for a two-component (sin, cos) head it pools both components and a collapsed head need not read 0 — see `PHIC_PSI_CARRYOVER.md` §4 |
| `kappa_<head>` metric | yes, vMF head only | mean predicted concentration κ for `sky_position`, in place of `r2`/`std_ratio`/`mae` |
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
   **`PeriodicCheckpoint`** (commit `f50dfae`) — also writes `checkpoints/epoch_NNN.weights.h5` every `train.checkpoint_every_n` epochs (default 10), so early-training weights survive alongside best and final.
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
| `data.augmentation.oversample.<subset>` | — | duplication factor for named subsets before transforms are fitted (Phase 3.1); implemented but not enabled in any current config |
| `model.trunk` | — | registry name (see `../models/`) |
| `model.trunk_cfg` | `{}` | trunk knobs (per-trunk doc) |
| `model.heads` | `DEFAULT_HEADS` (`mchirp`, `merger_time`, `snr`, `sky_position`, `coa_phase`) | active output heads |
| `model.head_cfg.hidden_units` | 64 | per-head MLP width |
| `model.head_cfg.bounded` | true | sigmoid output activations on `UNIT_AFFINE` heads (set false to make them linear) |
| `model.head_cfg.dropout` | 0.0 | dropout after the hidden Dense, applied to every head |
| `model.head_cfg.l2` | 0.0 | L2 kernel regularizer on both Dense layers, every head |
| `model.head_cfg.sigmoid_bias` | 0.0 | bias initializer for sigmoid output heads (Phase 2.5) |
| `model.head_cfg.per_head.<name>` | — | overrides `hidden_units`/`dropout`/`l2`/`sigmoid_bias`/`branch` for one head only |
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
| `train.checkpoint_every_n` | 10 | cadence of the periodic `checkpoints/epoch_NNN.weights.h5` snapshots |
| `train.scatter_subset` | 1000 | samples in the live scatter |
| `train.early_stopping` | false | off by default |
| `train.verbose` | 2 | Keras fit verbosity |

## Run directory contents

```
runs/<name>/
├── transforms.json      # active heads + fitted stats — needed for inference
├── best.weights.h5      # best val_loss checkpoint (weights only)
├── final.weights.h5     # last epoch
├── checkpoints/         # epoch_NNN.weights.h5 every train.checkpoint_every_n epochs
├── history.csv          # per-epoch: loss, mae_*, r2_*, std_ratio_*, weight_*, val_*
├── diagnostics.csv      # per-subset physical-unit metrics (see above)
├── scatter/epoch_*.png  # live scatter grids
└── tb/                  # TensorBoard logs
```

To reload: `build_trainer(cfg)` → call once on a dummy batch to build
variables → `load_weights` (this is what `scripts/evaluate.py` does).
