# Plan: Feed cos(ι) as Model Input Instead of Predicting It

> **STATUS: ON HOLD** — pending retraining with `activation: linear` for
> PERIODIC heads.  The first-round results cited below (φc R² ≤ 0.007)
> were caused by tanh saturation at initialization, not by the φc/ψ
> degeneracy.  See `diagnostic_log.md` for the full investigation (7
> diagnostic checks across 4 runs).  This plan should only proceed if
> φc/ψ remain unlearnable after the tanh→linear fix is applied and all
> models are retrained.

## Context

The phic_psi PoC currently treats inclination as an **output head** — the model
tries to predict `[sin(ι), cos(ι)]` from `(N, 4096, 2)` strain. True ι is
extracted from `y_true["inclination"][:, 1]` inside the trainer for curriculum
weighting, but the model's trunk never sees ι. The model has no way to learn
ι-conditioned features.

Results from the first round confirm this: across 5 architectures + poc_a/poc_b
on TCN, **φc R² ≤ 0.007 everywhere** — the degeneracy is universal. The
hypothesis is that without ι as an input, the trunk can't learn features that
disambiguate φc from ψ.

**Proposed change:** feed true `[sin(ι), cos(ι)]` (the full PERIODIC encoding
already produced by `TargetTransforms`) as a second model input alongside
strain. Remove inclination as an output head. The trunk still processes only
strain, but its pooled features are concatenated with the inclination vector
before the heads, so every head can learn ι-conditioned representations.

Using the full `(N, 2)` vector rather than just `cos(ι)` alone adds only 1
extra feature dimension (e.g., TCN goes from 128 → 130) and gives the model a
proper normalized unit-vector representation that may be easier to learn from
than a raw scalar.

At training time we have true ι → feed it. At inference time we don't → that's
out of scope for this PoC (per plan Sec 6). First confirm whether ι
conditioning helps at all.

---

## Current vs. Proposed Data Flow

### Current

```
HDF5 → load_arrays → (strain, params)
       TargetTransforms(heads) → {mchirp, ..., inclination: (sin ι, cos ι)}
       make_dataset → (strain, targets_dict)
       model(strain) → {mchirp, ..., inclination}
       SumDiffTrainer._build_combo_vectors → reads y_true["inclination"][:, 1]
```

### Proposed

```
HDF5 → load_arrays → (strain, params)
       TargetTransforms(transform_heads) → {mchirp, ..., inclination: (sin ι, cos ι)}
       make_dataset → (strain, targets_dict)
       ds.map → ((strain, incl_vec), targets_dict)          ← inclination vector added as input
       model(strain, incl_vec) → {mchirp, ..., coa_phase, polarization_angle}
                                                            ← NO inclination output
       SumDiffTrainer._build_combo_vectors → still reads y_true["inclination"][:, 1]
```

---

## Files to Modify

| File | Change |
|------|--------|
| 7 `config_*.yaml` files | Remove `inclination` from heads list |
| `train_poc.py` | Split head lists, build 2-input model, map dataset, conditioned callbacks |
| `trainer.py` | **No changes needed** — reads `y_true["inclination"][:, 1]` from targets dict |
| `run_full.py` `_evaluate` | Pass `(strain, cos_iota)` to predict |
| `NOTES.md` | Document the design change |

**No `src/gwml/` files modified.**

---

## Key Design Decisions

### 1. Reuse `attach_heads` with two inputs

`attach_heads(inputs, features, heads, cfg)` at `src/gwml/models/heads.py:27`
accepts `inputs` as either a single `keras.Input` or a list — it passes it
directly to `keras.Model(inputs=inputs)`. This means we can:

```python
# Build standard trunk — returns (strain_input, features, [extra_features])
result = build_trunk(trunk_name, trunk_cfg)
strain_input, features = result[0], result[1]
extra_features = result[2] if len(result) == 3 else None

# Add inclination vector input [sin(ι), cos(ι)]
incl_input = keras.Input(shape=(2,), name="inclination")

# Condition features
conditioned = Concatenate()([features, incl_input])

# Reuse attach_heads — pass both inputs
base = attach_heads(
    inputs=[strain_input, incl_input],
    features=conditioned,
    heads=model_heads,
    cfg=head_cfg,
    extra_features=extra_features,
)
# → keras.Model(inputs=[strain_input, incl_input], outputs={...})
```

No manual head-building. No replication of `attach_heads` logic.

### 2. Keep inclination in targets dict

The trainer's `_build_combo_vectors` reads `y_true["inclination"][:, 1]` for
curriculum weighting. We keep inclination in the dataset's targets dict (via
`transform_heads`) even though the model no longer outputs it. The dataset
restructuring map does NOT delete inclination from targets — it only reads the
cos component to add as input:

```python
def _add_cos_iota_to_input(strain, targets):
    cos_iota = targets["inclination"][:, 1]   # read cos component
    return (strain, cos_iota), targets          # keep all targets intact
```

### 3. Separate transform_heads from model_heads

- `transform_heads` = `model_heads + ["inclination"]` — used for data pipeline
- `model_heads` = config heads (no inclination) — used for model outputs

`make_dataset` uses `transform_heads` so inclination is in the targets dict.
The model is built with `model_heads` so it doesn't output inclination.
`transforms.json` is saved with `model_heads` so evaluation doesn't expect it.

### 4. Conditioned callback subclasses (no `call` override needed)

`DiagnosticSubsetsCallback` and `LiveScatterCallback` call
`model.predict(strain)` internally. Rather than overriding `call` (which gives
approximate diagnostics with cos_iota=0), we create thin subclasses that pass
`(strain, cos_iota_val)` to predict:

```python
class _ConditionedLiveScatterCallback(LiveScatterCallback):
    def __init__(self, strain, params, transforms, out_dir, every_n=5,
                 batch_size=256):
        super().__init__(strain, params, transforms, out_dir, every_n, batch_size)
        iota = params[:, PARAM_COLUMNS["inclination"]]
        self._incl_vec = np.stack([np.sin(iota), np.cos(iota)], axis=-1).astype(np.float32)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n:
            return
        pred = self.model.predict(
            (self.strain, self._incl_vec), batch_size=self.batch_size, verbose=0
        )
        pred_physical = self.transforms.inverse(pred)
        scatter_grid(
            self.true_physical, pred_physical, self.transforms.heads,
            self.out_dir / f"epoch_{epoch + 1:04d}.png",
            title=f"epoch {epoch + 1}",
        )


class _ConditionedDiagnosticSubsetsCallback(DiagnosticSubsetsCallback):
    def __init__(self, strain, params, transforms, csv_path, every_n=5,
                 batch_size=256):
        super().__init__(strain, params, transforms, csv_path, every_n, batch_size)
        iota = params[:, PARAM_COLUMNS["inclination"]]
        self._incl_vec = np.stack([np.sin(iota), np.cos(iota)], axis=-1).astype(np.float32)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n:
            return
        pred = self.model.predict(
            (self.strain, self._incl_vec), batch_size=self.batch_size, verbose=0
        )
        pred_physical = self.transforms.inverse(pred)
        # Identical logic to parent, just using the conditioned predict result
        heads = self.transforms.heads
        write_header = not self.csv_path.exists()
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "epoch", "subset", "n",
                    *[f"mae_{h}" for h in heads],
                    *[f"r2_{h}" for h in heads],
                    *[f"std_ratio_{h}" for h in heads],
                ])
            for name, mask in self.subsets.items():
                maes, r2s, ratios = [], [], []
                for h in heads:
                    t = np.ravel(self.true_physical[h][mask])
                    p = np.ravel(pred_physical[h][mask])
                    res = signed_error(h, t, p)
                    maes.append(float(np.mean(np.abs(res))))
                    ss_tot = float(np.sum((t - t.mean()) ** 2))
                    ss_res = float(np.sum(res ** 2))
                    r2s.append(1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan)
                    ratios.append(float(p.std() / (t.std() + 1e-12)))
                writer.writerow([
                    epoch + 1, name, int(mask.sum()),
                    *[f"{v:.6g}" for v in (*maes, *r2s, *ratios)],
                ])
```

And a `_build_callbacks_conditioned` that wires them up with the same LR
schedule, ModelCheckpoint, CSVLogger, and TensorBoard as the original
`_build_callbacks`. This is ~50 lines of boilerplate copied from
`src/gwml/training/train.py` — acceptable for PoC isolation.

**Why this over a `call` override:** Diagnostic metrics use the correct
cos_iota per sample rather than a zero-padded default. No fragile
single-tensor detection logic.

### 5. Final evaluation uses true cos_iota

`run_full.py`'s `_evaluate` loads params from HDF5, computes true cos_iota, and
passes both inputs to predict:

```python
iota = params[:, 2]       # PARAM_COLUMNS["inclination"]
cos_iota = np.cos(iota).astype(np.float32)
trainer((strain[:1], cos_iota[:1]))           # build variables
trainer.load_weights(weights)
raw_pred = trainer.predict((strain, cos_iota), batch_size=256)
```

---

## Step-by-Step Implementation

### Step 1: Config files — remove `inclination` from heads

In all 7 `config_*.yaml` files in `experiments/phic_psi_poc/`, change:
```yaml
heads: [ mchirp, merger_time, snr, sky_position, coa_phase,
         polarization_angle, inclination ]
```
to:
```yaml
heads: [ mchirp, merger_time, snr, sky_position, coa_phase,
         polarization_angle ]
```

Files: `config_poc.yaml`, `config_baseline.yaml`, `config_tcn.yaml`,
`config_cnn_baseline.yaml`, `config_cnn_attention.yaml`,
`config_inception_time.yaml`, `config_resnet1d.yaml`

### Step 2: `trainer.py` — add `call` override

Add to `SumDiffTrainer` class in `experiments/phic_psi_poc/trainer.py`:

```python
def call(self, x, training=False, **kwargs):
    """Handle both two-input (strain, cos_iota) and single-input (strain) calls.

    Callbacks (DiagnosticSubsetsCallback, LiveScatterCallback) call
    ``model.predict(strain)`` with only strain.  We detect single-tensor
    input and pad cos_iota = 0 (edge-on, the least constraining default)
    so the model graph still executes.
    """
    if isinstance(x, (list, tuple)):
        return super().call(x, training=training, **kwargs)
    import tensorflow as tf
    cos_iota = tf.zeros((tf.shape(x)[0],), dtype=tf.float32)
    return super().call((x, cos_iota), training=training, **kwargs)
```

### Step 3: `train_poc.py` — core changes

**3a.** Change `_REQUIRED_HEADS` (line 44):
```python
# Before:
_REQUIRED_HEADS = ("coa_phase", "polarization_angle", "inclination")
# After:
_REQUIRED_HEADS = ("coa_phase", "polarization_angle")
```

**3b.** Rewrite `build_sumdiff_trainer` (lines 63-100) to build a 2-input model
using `build_trunk` + `attach_heads` (see Key Decision 1 above for the
pseudocode).

**3c.** Add dataset restructuring helper:
```python
def _add_inclination_to_input(*args):
    """Transform batched dataset to add inclination vector as second input.

    Handles both ``(strain, targets)`` and ``(strain, targets, sample_weight)``
    dataset structures.  The full ``inclination`` target ``(B, 2)`` vector
    ``[sin(ι), cos(ι)]`` is passed through as the second model input.
    Does NOT remove inclination from targets — the trainer still needs it.
    """
    if len(args) == 2:
        strain, targets = args
        return ((strain, targets["inclination"]), targets)
    elif len(args) == 3:
        strain, targets, sample_weight = args
        return ((strain, targets["inclination"]), targets, sample_weight)
    raise ValueError(f"Expected 2 or 3 args, got {len(args)}")
```

Applied via ``ds.map()`` after ``make_dataset()``:

**3d.** In `run_poc_experiment`:

- Build two head lists:
  ```python
  model_heads = _ensure_required_heads(cfg["model"].get("heads"))
  transform_heads = list(model_heads) + ["inclination"]
  ```

- Use `transform_heads` for `TargetTransforms` and `make_dataset`

- Save `transforms.json` with `model_heads` only:
  ```python
  import copy
  transforms_eval = TargetTransforms(heads=model_heads)
  # Copy stats from full transforms (LOG_ZSCORE/ZSCORE means/stds)
  transforms_eval.stats = copy.deepcopy(transforms.stats)
  transforms_eval.to_json(run_dir / "transforms.json")
  ```

- Apply dataset map after `make_dataset`:
  ```python
  train_ds = make_dataset(...)
  train_ds = train_ds.map(_add_inclination_to_input)
  val_ds = make_dataset(...)
  val_ds = val_ds.map(_add_inclination_to_input)
  ```

- Existing callback and training logic unchanged.

### Step 4: `run_full.py` `_evaluate` — pass inclination vector to predict

The `_evaluate` function inside `run_full.py` is the evaluation entry point
(there is no separate `evaluate_poc.py`).

```python
# Build inclination vector for the evaluation split
iota = params[:, 2]  # PARAM_COLUMNS["inclination"]
incl_vec = np.stack([np.sin(iota), np.cos(iota)], axis=-1).astype(np.float32)

# Build variables with two inputs
trainer((strain[:1], incl_vec[:1]))

# Predict with both inputs
raw_pred = trainer.predict((strain, incl_vec), batch_size=256, verbose=0)
```

Rest of `_evaluate` (scatter_grid, residual plots, logit hist) unchanged —
they use `raw_pred` and `transforms.heads` which only has `model_heads`.

### Step 5: `NOTES.md` — document

Add entry explaining the decision: inclination moved from output to input,
rationale, what changed, caveats.

---

## What Does NOT Change

- **SumDiffTrainer._build_combo_vectors** — still reads
  `y_true["inclination"][:, 1]` for curriculum weighting (inclination is still
  in the targets dict)
- **SumDiffTrainer._other_heads_loss** — iterates over `self.head_names` which
  no longer includes inclination, so no loss is computed for it (correct)
- **SumDiffTrainer._patch_log_vars** — only operates on coa_phase and
  polarization_angle (unchanged)
- **curriculum.py, transform_utils.py, validation_script.py, prereq_checks.py**
  — no model-building code in these files
- **All `src/gwml/` files** — zero modifications

---

## Verification

1. **Syntax check:**
   ```bash
   python -c "import ast; ast.parse(open('experiments/phic_psi_poc/train_poc.py').read())"
   python -c "import ast; ast.parse(open('experiments/phic_psi_poc/trainer.py').read())"
   ```

2. **Config validation:**
   ```bash
   for c in experiments/phic_psi_poc/config_*.yaml; do
       python -c "import yaml; c=yaml.safe_load(open('$c')); print(c['name'], c['model']['heads'])"
   done
   ```
   All should show 6 heads (no inclination).

3. **Smoke test** on GPU machine: run `config_baseline.yaml` for 1-2 epochs.
   Verify:
   - Model summary shows two inputs: `strain (None, 4096, 2)` and
     `inclination (None, 2)`
   - No inclination in output heads
   - Dataset map runs without errors
   - Training loss decreases
   - Callbacks produce valid scatter and diagnostics

4. **Full evaluation check:** run `_evaluate` on a completed run, verify
   `metrics_validation.csv` has 6 heads, no inclination.

5. **Compare against previous results:** If φc R² improves above noise (> 0.02),
   the degeneracy is breakable with ι conditioning.  (Note: this threshold was
   set before the tanh saturation bug was discovered; it may need re-evaluation
   once all models are retrained with `activation: linear` for PERIODIC heads.)

---

## Open Questions / Risks

1. **cos_iota = 0 fallback for callbacks means diagnostic metrics are approximate.**
   Acceptable for a PoC — final evaluation uses true cos_iota. If this is a
   concern, alternatives: (a) store val_cos_iota on the trainer and index by
   batch during predict, or (b) modify callbacks. Both are more invasive.

2. **`transforms.stats` sharing:**
   `TargetTransforms.stats` is a dict keyed by head name for LOG_ZSCORE/ZSCORE
   params. PERIODIC heads like inclination have no stats entries. Copying the
   full stats dict to the model_heads transform should be safe — the extra keys
   (if any) are ignored. Verify the keys match.

3. **Inference-time ι:** out of scope for this PoC. If ι conditioning helps,
   we'd need a separate ι estimator network before the main model. That's a
   future design problem.

4. **Feature dimension increase:** Concatenating `[sin(ι), cos(ι)]` (2 values) to
   features (e.g., 128 for TCN) is a negligible increase (130 vs 128). No
   meaningful impact on model size or training speed.