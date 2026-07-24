# Inclination Head: Code-Path Trace

**Question:** Does the `inclination` head's loss computation pass through
`normalize_unit`?

**Answer: No.** Inclination uses standard Huber loss directly on the raw
`Dense(2, activation="linear")` output. `normalize_unit` is only applied to
`coa_phase` and `polarization_angle` inside `_build_combo_vectors`, and only
in the SumDiffTrainer's circular-loss path.

---

## 1. Head specification

`src/gwml/heads_spec.py:99-100`
```python
HeadSpec("inclination", 2, TransformKind.PERIODIC, label=r"$\iota$ [rad]",
         dim=2, activation="linear", period=_TWO_PI),
```

- `dim=2` → outputs a (sin ι, cos ι) pair.
- `activation="linear"` → no squashing, no tanh, no normalization.
- Default loss: `"huber"` (line 57: `loss: str = "huber"`).

## 2. Head construction (model graph)

`src/gwml/models/heads.py:99-104`
```python
outputs[spec.name] = layers.Dense(
    spec.dim, activation=activation,
    kernel_regularizer=regularizer,
    bias_initializer=bias_init,
    name=spec.name
)(x)
```

- `spec.dim = 2`, `activation = "linear"`.
- The output `y_pred["inclination"]` is a raw (N, 2) tensor — **no
  `Lambda` normalization, no `normalize_unit`, no `tanh`**.  Just a bare
  `Dense(2, activation="linear")`.

## 3. Loss function binding

`src/gwml/training/losses.py:135-147`
```python
self.huber = keras.losses.Huber(delta=cfg.get("huber_delta", 1.0))
loss_table = {"huber": self.huber, "vmf": vmf_nll_loss}
self.head_loss = {}
for h in self.head_names:
    kind = HEAD_SPECS[h].loss       # → "huber" for inclination
    self.head_loss[h] = loss_table[kind]
```

- `HEAD_SPECS["inclination"].loss` defaults to `"huber"` (head spec line 57).
- `self.head_loss["inclination"] = keras.losses.Huber(delta=1.0)`.
- Huber loss on (sin, cos) pairs is magnitude-sensitive: its minimum is at
  `v_pred = v_true`, which has `|v| = 1`. It implicitly penalizes `|v| ≠ 1`.

## 4. Loss computation at train time

In the SumDiffTrainer, inclination is handled by `_other_heads_loss`:

`experiments/phic_psi_poc/trainer.py:347-367`
```python
def _other_heads_loss(self, y_true, y_pred, sample_weight) -> tf.Tensor:
    for h in self.head_names:
        if h in self._SUMDIFF_SOURCE_HEADS:   # ("coa_phase", "polarization_angle")
            continue                           # ← inclination NOT skipped
        if HEAD_SPECS[h].loss == "vmf":
            ...
        else:
            head_loss = self.head_loss[h](
                y_true[h], y_pred[h], sample_weight=sample_weight
            )
        # → head_loss = Huber()(y_true["inclination"], y_pred["inclination"])
```

- `y_pred["inclination"]` is the raw `Dense(2, linear)` output.
- `y_true["inclination"]` is the target `(sin ι, cos ι)` pair from
  `TargetTransforms` (period=2π, so |true| = 1 by construction).
- The loss is `Huber(δ=1.0)(v_pred, v_true)` — **no `normalize_unit`,
  no `complex_mul`, no circular loss.**  This is a standard regression
  loss on the (sin, cos) representation.

## 5. Where `normalize_unit` IS used (φc and ψ only)

`experiments/phic_psi_poc/trainer.py:277-280`
```python
z_phic_pred = tf_normalize_unit(z_phic_pred)   # ← coa_phase only
z_psi_pred = tf_normalize_unit(z_psi_pred)     # ← polarization_angle only
```

`_build_combo_vectors` normalizes φc and ψ predictions to build combo vectors.
Inclination's **prediction** (`y_pred["inclination"]`) is never passed to this
method at all. The method only reads `y_true["inclination"][:, 1]` (line 293)
to extract `cos(ι_true)` for curriculum weighting — that's the true label, not
the model's prediction.

## 6. What this means

| Head               | Output layer      | Loss function | Uses `normalize_unit`? | |v| regularized? |
|--------------------|-------------------|---------------|------------------------|-----------------|
| coa_phase          | Dense(2, linear)  | 1−cosΔθ       | **Yes** (in trainer)   | No (before fix) |
| polarization_angle | Dense(2, linear)  | 1−cosΔθ       | **Yes** (in trainer)   | No (before fix) |
| inclination        | Dense(2, linear)  | Huber         | **No**                 | **Yes** (Huber)  |

- **φc and ψ**: the magnitude penalty (`λ·(|v|−1)²`) addresses a real bug —
  the isotropic `1−cosΔθ` loss is blind to |v|, and `normalize_unit`'s
  backward pass divides by |v|, crushing or exploding the angular gradient
  when |v| drifts.

- **Inclination**: Huber loss is already magnitude-sensitive (`‖v−v_true‖²`
  penalizes |v| ≠ 1). If inclination is mode-collapsed (MAE = π/2), the
  cause is NOT the `normalize_unit` |v| drift. It's something else — possibly
  a weak gradient signal from strain → inclination, an optimizer issue, or
  the model learning the dataset mean (0.637, 0) as the optimal constant
  predictor under Huber loss when the true mapping is unlearnable.

## 7. Verdict

The earlier claim stands: **inclination does not use `normalize_unit`**.
The reviewer's argument that "inclination failing identically proves it's a
shared mechanism bug (normalize_unit)" is not supported by the code. The two
failure modes have different mechanisms and may have different root causes.
