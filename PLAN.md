# GW Parameter Estimation — Multi-Head DL Plan

## Objective

Regress four physical parameters from dual-detector whitened strain using a single
shared-trunk, multi-head network. Multiple trunk architectures are pluggable behind a
common interface so they can be benchmarked against each other (locally on CPU for
smoke tests, on the lab GPU machine for real training).

## Data (from `combined_repackaged.hdf`)

| dataset         | training                                                         | validation |
|-----------------|------------------------------------------------------------------|------------|
| samples         | 25,000                                                           | 5,000      |
| strain          | `h1`, `l1` — (N, 4096) float32, whitened signal+noise            | same       |
| clean waveforms | `h1_waveform`, `l1_waveform` (held in reserve, not used for now) | same       |
| params          | (N, 10) float64                                                  | same       |

- Window is 4096 samples with merger at 1.6–1.8 s → **2 s @ 2048 Hz**.
- Every window contains a signal (SNR 7–15); this is pure regression, no detection head.

### Model input
Stack the two detectors channel-wise: **(4096, 2)** float32.
The strain is already whitened but has std ≈ 32. Normalization happens **inside the
model**: every trunk starts with a `BatchNormalization` layer directly on the input
(the common pattern in GW deep-learning literature). The model therefore ingests raw
strain — no external scaling stats to carry around, and a saved model is fully
self-contained.

*Alternative kept in mind:* a `keras.layers.Normalization` layer adapted once on the
training split gives fixed (non-batch-dependent) statistics at inference, which can be
slightly more stable than BatchNorm's moving averages. Both are one line behind the
same trunk interface; we start with BatchNorm and can A/B this cheaply.

### Targets (4 heads)

| head          | raw range    | transform for training                                                          | head activation   |
|---------------|--------------|---------------------------------------------------------------------------------|-------------------|
| `mchirp`      | 8.84 – 43.45 | log, then z-score (MSE in log-space ≈ fractional error — no custom loss needed) | linear            |
| `q`           | 0.20 – 1.00  | affine map [0.15, 1.05] → [0, 1] (padded off the sigmoid asymptote)             | sigmoid (bounded) |
| `merger_time` | 1.6 – 1.8 s  | affine map [1.6, 1.8] → [0, 1]                                                  | sigmoid (bounded) |
| `snr`         | 7.0 – 15.0   | z-score                                                                         | linear            |

Bounded sigmoid heads guarantee physical predictions for the two range-limited
targets and remove an early-epoch failure mode; toggleable per config.

**Head registry (`src/gwml/heads_spec.py`):** every possible head — the core
four above plus `inclination`, `coa_phase`, `polarization_angle`,
`declination`, `ra` — is defined once as a `HeadSpec` binding its params
column, transform, output dimension, activation, and loss. Experiment YAMLs
select *which* heads are active via `model.heads: [...]`; they cannot override
*how* a head is handled — that binding lives in code so a periodic parameter
can never be silently paired with a plain loss. Periodic heads are (sin, cos)
pairs of the angle scaled to the head's period (π for `polarization_angle`,
since strain is invariant under ψ → ψ+π), so standard Huber on the pair is
correct and the inverse recovers the angle via atan2; all reported errors are
wrap-aware. `injection_time` is deliberately absent from the registry —
absolute GPS time is not learnable from a 2 s window.

All transforms are computed on training data only, saved as JSON, and inverted at
evaluation time so metrics are reported in physical units.

## Repository layout

```
src/gwml/
├── data/
│   ├── loader.py        # HDF5 → tf.data.Dataset; loads splits into RAM (~800 MB)
│   └── transforms.py    # input scaling + per-target transform/inverse, JSON persistence
├── models/
│   ├── registry.py      # @register("name") → build_trunk(cfg) lookup
│   ├── heads.py         # attach_heads(features, head_cfg) → multi-output Model
│   └── trunks/
│       ├── cnn_baseline.py    # small plain CNN — pipeline sanity check
│       ├── resnet1d.py        # strided/dilated residual blocks (George & Huerta style)
│       ├── inception_time.py  # parallel multi-kernel blocks for the sweeping chirp
│       ├── tcn.py             # exponentially dilated temporal conv net
│       └── cnn_attention.py   # conv front-end + transformer encoder + attention pool
├── training/
│   ├── losses.py        # per-head Huber, configurable loss weights
│   ├── callbacks.py     # LiveScatterCallback, DiagnosticSubsetsCallback (see below)
│   └── train.py         # fit loop: callbacks, checkpoints, CSV/TensorBoard logs
└── evaluation/
    ├── metrics.py       # per-head MAE/RMSE in physical units
    └── plots.py         # pred-vs-true scatter, residual-vs-SNR/mchirp,
                          # pre-sigmoid logit histograms (train vs val)

docs/models/             # one MD file per trunk: what it is + selection rationale
├── cnn_baseline.md
├── resnet1d.md
├── inception_time.md
├── tcn.md
└── cnn_attention.md

docs/pipeline/           # pipeline documentation (kept current with the code)
├── overview.md          # end-to-end flow, module map, design invariants
├── data.md              # loading, transforms, dataset construction
├── training.md          # heads/losses/callbacks + full YAML config reference
├── evaluation.md        # evaluate script, physics sanity checks, caveats
└── testing.md           # test suite map and workflow rules

configs/                 # one YAML per experiment (model, lr, batch, loss weights)
├── smoke.yaml           # tiny end-to-end run; pre-flight check before GPU launches
└── <trunk>.yaml         # one per trunk architecture

tests/                   # pytest suite (run on the lab machine, not locally)
scripts/
├── train.py             # python scripts/train.py configs/resnet1d.yaml
├── plot_run.py          # history.csv + diagnostics.csv -> summary PNGs
├── evaluate.py          # loads checkpoint, writes metrics + plots for a split
└── run_all.py           # train -> plot_run -> evaluate, one config, fail-fast
```

## Model contract (the pluggable part)

Every trunk implements one function and registers itself:

```python
@register("resnet1d")
def build_trunk(cfg) -> tuple[keras.Input, tf.Tensor]:
    """Returns (input tensor (4096, 2), pooled feature vector (B, F)).

    A trunk may optionally return a third element: a dict[str, KerasTensor]
    of extra feature tensors accessible to individual heads via
    ``head_cfg.per_head.<name>.branch``. ``cnn_attention`` uses this to
    expose per-token transformer features before attention pooling so q can
    branch from finer-grained representations (Phase 3.2).
    """
```

`heads.py` takes the feature vector and attaches one small MLP per active
head (Dense `hidden_units` → Dense `spec.dim`, 1 for scalar heads like
`mchirp`/`q`/`merger_time`/`snr`, 2 for periodic sin/cos pairs), named after
the head. All heads share `head_cfg`'s global `hidden_units`/`dropout`/`l2`
by default, but any one head can override them via
`head_cfg.per_head.<name>` — used to cut capacity on a head that overfits
faster than its trunk-mates (e.g. `q` on `cnn_attention`) without touching
the others. Per-head overrides also support `branch` (connects the head to a
named extra feature tensor from the trunk) and `sigmoid_bias` (nudges the
output bias away from zero at init — useful for reviving a saturated sigmoid).
Swapping architectures is a one-line config change; heads, losses, and
evaluation never change.

### Trunk candidates

1. **cnn_baseline** — 4 conv blocks, ~100k params. Exists to prove the pipeline.
2. **resnet1d** — residual blocks, stride-2 downsampling, dilation in later blocks;
   the expected workhorse.
3. **inception_time** — parallel kernels (e.g. 9/19/39) per block; suited to the
   chirp's time-varying frequency content.
4. **tcn** — dilated convs give a large receptive field cheaply.
5. **cnn_attention** — conv downsampling to ~128 steps, 2 transformer encoder
   blocks, attention pooling. Heaviest; GPU-targeted.

All trunks end in global average pooling (optionally concat with global max pool) so
the feature dimension is architecture-defined but the head interface is constant.

### Per-model documentation

Each trunk ships with a companion file in `docs/models/<name>.md` containing:

- **What it is** — the architecture described block by block, with tensor shapes.
- **Why it's here** — the rationale for selecting/creating it for chirp signals
  (receptive field vs. the signal's duration, how it handles the time-varying
  frequency, parameter count, compute cost).
- **Provenance** — the papers/uses it draws from (e.g. George & Huerta 2018 for the
  CNN family, Ismail Fawaz et al. 2020 for InceptionTime, Bai et al. 2018 for TCN).
- **Knobs** — which config fields the builder honors (depth, filters, kernel sizes).

Writing the MD file is part of "done" for each trunk in milestone 3 — a trunk without
its doc doesn't get merged into the zoo.

## Training setup

- **Per-head loss:** Huber (delta in config).
- **Head balancing:** uncertainty-weighted multi-task loss (Kendall, Gal & Cipolla
  2018) **on by default** — each head gets a learnable log-variance `s_i`, total loss
  `Σ exp(-s_i)·L_i + s_i`. The network balances the four heads itself, and the learned
  `s_i` are logged per epoch as a diagnostic of which parameters the data constrains.
  Fixed hand-set weights remain available as the config fallback.
- **SNR-weighted sample loss:** optional config switch (off by default) — weight each
  sample by `(SNR/10)^alpha` to down-weight events whose labels are barely
  recoverable. An experiment, not a default.
- **Targeted oversampling:** `data.augmentation.oversample.<subset>: <factor>`
  duplicates rows in named subsets (from `build_subset_masks()` in
  `loader.py`) before transforms are fitted. Used to up-weight the
  `q_high × mchirp_low` cell (~11% of data, val_r2_q ≈ −22 across all
  trunks). No architecture changes — pure array-level duplication before
  dataset construction. See `q_head_action_plan.md` Phase 3.1.
- **Not doing:** GradNorm/gradient-balancing schemes — overkill for four well-behaved
  regression heads. Periodic losses (`1 − cos(Δ)` or sin/cos pairs) only become
  relevant if/when angle heads are added; plain MSE on wrapped angles is silently
  wrong (noted for the extensions).
- **Mean-collapse safeties** (GW regression often collapses every head to the
  target mean with ~0 variance):
  - *Always on:* per-head R² (`r2_<head>`) and `std(pred)/std(true)`
    (`std_ratio_<head>`) tracked as epoch metrics and per-subset in the
    diagnostics CSV — collapse reads as r2 → 0, std_ratio → 0, no eyeballing
    needed. The learnable log-variances are clamped to ±`log_var_clamp`
    (default 3.0) so uncertainty weighting can never push a head's effective
    weight below exp(−3) ≈ 5% and entrench collapse. `log_var_clamp` accepts
    a per-head override (`{default: 3.0, <head>: tighter}`) — needed because
    a head whose *train* loss shrinks fastest (whether from real learning or
    overfitting) can otherwise ride the same weight ceiling every other head
    eventually reaches, erasing the scheme's per-head signal entirely; see
    `q_head_action_plan.md`.
  - *Config-optional (off by default):* `loss.variance_penalty` adds
    `λ·(std(pred) − std(true))²` per head to directly punish shrinking
    prediction spread; `optim.warmup_epochs` ramps the LR linearly to dodge
    the early-training mean-prediction local minimum.
  - `LiveScatterCallback` remains the visual check (a collapsed head is a
    horizontal band at the target mean).
- **Optimizer:** Adam. Batch 64–256 (config), checkpointing best-val per experiment
  under `runs/<name>/`.
- **Determinism:** every config carries a `seed`; training calls
  `keras.utils.set_random_seed(seed)` (optionally `enable_op_determinism` for strict
  runs) so architecture comparisons are apples-to-apples.
- Everything driven by the YAML config → moving to the GPU machine changes nothing
  but the config (batch size, epochs).

### Callbacks (all config-toggled, in `training/callbacks.py`)

1. **`LiveScatterCallback`** (custom) — every N epochs (config, e.g. 5), run the model
   on a fixed validation subset and save a 2×2 pred-vs-true scatter grid (one panel
   per head, in physical units, with the y=x diagonal and per-head MAE annotated) to
   `runs/<name>/scatter/epoch_XXXX.png`. Lets you watch each head converge — or
   collapse to predicting the mean — without waiting for training to finish.
2. **`DiagnosticSubsetsCallback`** (custom) — a few fixed subsets carved from the
   validation split once at startup and evaluated every N epochs, with per-head MAE
   logged to CSV/TensorBoard per subset:
   - SNR terciles (low 7–9.7, mid 9.7–12.3, high 12.3–15) — errors should order
     cleanly by SNR;
   - mchirp low/high halves — checks mass-range bias;
   - merger-time early/late halves — checks time-localization bias;
   - q terciles (`q_low`/`q_mid`/`q_high`) and their cross-tab with mchirp
     low/high (`q_{low,high}_mchirp_{low,high}`) — isolates the
     near-equal-mass / low-chirp-mass regime, which was both the
     worst-performing and most data-scarce cell in every run tested so far
     (see `q_head_action_plan.md`).
3. **LR schedule** — `ReduceLROnPlateau` on val loss as the default "decrease after a
   while" mechanism (factor, patience, min_lr in config); a fixed step-decay schedule
   available as the config alternative.
4. **`EarlyStopping`** — optional, off by default, enabled per-config
   (monitor val loss, restore best weights).
5. **`ModelCheckpoint` + `CSVLogger` + TensorBoard** — best-val weights, per-epoch
   metric history, and curves under `runs/<name>/`.

## Testing

Written in this repo, **executed on the lab GPU machine** (local box is a ThinkPad
T530 — nothing heavier than a syntax check runs here). Needs `pytest` added to
dependencies.

1. **Overfit-one-batch** (`tests/test_overfit.py`, marked `slow`) — train each trunk
   on ~32 fixed samples until loss ≈ 0. If it can't memorize 32 samples, the bug is
   in the pipeline/loss/normalization, not the architecture.
2. **Smoke config** (`configs/smoke.yaml`) — the real training script on ~256 samples
   for 2 epochs, exercising every callback (scatter PNG written, diagnostics CSV
   written, checkpoint saved). Run before every long GPU training launch.
3. **Unit tests** — transform round-trips (`inverse(transform(y)) ≈ y` per head),
   registry (every trunk builds; outputs named heads with shape (B, 1)), loss
   machinery (finite losses, learnable log-vars present, SNR weights monotonic),
   and HDF5 integrity (shapes, no NaN/Inf, params within documented ranges;
   skipped automatically if the file is absent).
4. Quick suite: `pytest -m "not slow"`. Full suite (overfit tests included):
   `pytest`.

## Evaluation

- Per-head MAE and RMSE in **physical units** on the validation split.
- Pred-vs-true scatter per head; residuals binned by true SNR (errors should shrink
  as SNR grows — a good physics sanity check) and by mchirp.
- A single comparison table across trunk architectures.

## Milestones

1. **Scaffold + data** — package layout, loader, transforms; verify shapes and plot one
   sample (strain + overlaid clean waveform).
2. **Baseline end-to-end** — cnn_baseline trained a few epochs on CPU; confirms the
   whole loop (data → model → inverse-transformed metrics) and exercises the full
   callback suite (LiveScatter PNGs and diagnostic-subset CSVs actually produced).
3. **Model zoo** — implement the remaining four trunks behind the registry, each with
   its `docs/models/<name>.md`.
4. **Experiment configs** — one YAML per trunk, train/evaluate scripts finalized.
5. **GPU handoff** — full training runs in the lab; comparison report.

## Later extensions (explicitly out of scope now)

- Auxiliary denoising decoder using `*_waveform` targets (regularizes the trunk).
- Heteroscedastic heads (mean + σ, Gaussian NLL) for per-event error bars.
- Sky-position and orientation-angle heads with periodic encodings.
- Time-shift augmentation (shift window, update merger_time label accordingly).