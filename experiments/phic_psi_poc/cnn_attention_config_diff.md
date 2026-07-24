# Section C — cnn_attention config diff and outlier investigation

**Date**: 2026-07-21
**Context**: Run 7 verification plan Section C — why is cnn_attention the consistent outlier?

---

## 1. Config diff: cnn_attention vs tcn/poc_a

| Setting | cnn_attention | tcn | poc_a (baseline) |
|---------|--------------|-----|------------------|
| `model.trunk` | `cnn_attention` | `tcn` | `tcn` |
| `model.trunk_cfg` | `{}` | `{}` | `{}` |
| `model.heads` | same 7 | same 7 | same 7 |
| `model.head_cfg` | `{hidden: 64, bounded: true}` | identical | identical |
| `loss.mode` | `baseline` | `baseline` | `baseline` |
| `loss.*` (all keys) | identical | identical | identical |
| `optim.*` (all keys) | identical | identical | identical |
| `optim.schedule.min_lr` | `1.0e-7` | `1.0e-6` | `1.0e-6` |
| `train.*` (all keys) | identical | identical | identical |

**The only functional difference is the trunk architecture.** The `min_lr` difference (1e-7 vs 1e-6) is a red herring — with a plateau schedule starting at 1e-3 and factor 0.5, it takes ~10 halvings to reach 1e-6, which at patience=5 is 50 plateau epochs. Neither model is likely hitting min_lr.

---

## 2. Architectural differences: cnn_attention vs TCN

### TCN trunk (`src/gwml/models/trunks/tcn.py`)

```
Input (4096, 2)
  → BatchNorm → Stem Conv (stride=2) → (2048, 64)
  → 10× TCN blocks (dilations 1,2,4,...,512, filters=64)
  → GAP + GMP concatenate → features (128,)
```

- 64 filters throughout
- Dilated convolutions for receptive field
- **Fixed pooling**: GlobalAveragePooling + GlobalMaxPooling concatenated
- Returns: `(inputs, features)` — 2-tuple, no extra features

### CNN Attention trunk (`src/gwml/models/trunks/cnn_attention.py`)

```
Input (4096, 2)
  → BatchNorm
  → 5× Conv1D (filters=[32,64,128,128,128], kernel=7, stride=2 each)
    → each: Conv → BatchNorm → ReLU
  → Dense projection → dim=128
  → PositionalEmbedding (learned)
  → 2× Transformer encoder blocks (dim=128, heads=4, ff_dim=256, dropout=0.1)
  → Branch point: tokens (B, T, 128) returned as q_tokens
  → AttentionPooling (learned score over time) → features (128,)
```

Returns: `(inputs, features, {"q_tokens": tokens})` — **3-tuple** with per-token transformer output.

### Key architectural differences

| Property | TCN | CNN Attention |
|----------|-----|---------------|
| Max filters | 64 | 128 |
| Model dim | 64 | 128 |
| Receptive field | Dilated conv (global) | Conv strides + transformer self-attention (global) |
| Pooling | Fixed GAP+GMP | **Learned attention pooling** |
| Per-token features | Not exposed | Exposed as `q_tokens` |
| Parameters | Lower | Higher (transformer + deeper convs) |

---

## 3. Is q_tokens the explanation? No.

The `q_tokens` branch point exists so heads can read from per-token transformer outputs (before attention pooling) rather than the globally-pooled features. This is accessed via `per_head.<head_name>.branch` in the config (`heads.py:64–70`):

```python
branch_name = overrides.get("branch")
if branch_name and branch_name in extra_features:
    ef = extra_features[branch_name]
    # 3-D tensors are GAP-pooled first
    if len(ef.shape) == 3:
        ef = layers.GlobalAveragePooling1D(name=f"{spec.name}_branch_gap")(ef)
    head_features = ef
```

**However**, the config has `head_cfg: { hidden_units: 64, bounded: true }` — no `per_head` section exists. **No head uses the q_tokens branch.** Every head reads from the same globally-pooled `features` as in TCN.

The q_tokens mechanism is present but unused in this experiment. The cnn_attention advantage comes from the trunk's baseline feature quality, not from per-head branching.

---

## 4. What actually explains cnn_attention's outlier behavior?

### The data

From `analysis_report_20260720_234304.md`:

| Metric | cnn_attention | tcn | poc_a | poc_b |
|--------|:---:|:---:|:---:|:---:|
| coa_phase circ_r | **0.43** | 0.86 | 0.85 | 0.99 |
| pol_angle circ_r | **0.17** | 0.88 | 0.49 | 0.99 |
| coa_phase ang_MAE | 1.605 | 1.598 | 1.570 | 1.541 |
| pol_angle ang_MAE | 0.791 | 0.802 | 0.784 | 0.801 |
| mchirp MAE | 1.363 | 0.951 | 0.977 | 1.024 |
| mchirp R² | 0.926 | **0.963** | **0.959** | 0.957 |
| snr MAE | 0.880 | 0.837 | 0.831 | 0.835 |
| snr R² | 0.755 | **0.784** | **0.785** | 0.783 |
| sky_position ang MAE | **3.3°** | 4.5° | 8.2° | 10.0° |
| val_loss (epoch 79) | **−1.53** | −3.27 | −3.79 | −3.16 |

### The pattern

cnn_attention is **not** uniformly better. It's:

- **Better at producing spread**: Lowest circ_r (most varied predictions) on both periodic heads
- **Worse at scalar regression**: mchirp R² of 0.926 vs 0.963 for tcn — a meaningful gap
- **Better at sky_position**: 3.3° vs 4.5° for tcn (and dramatically better than poc_a/poc_b at 8-10°)
- **Different loss scale**: val_loss at −1.53 vs −3.16 to −3.79

The loss-scale difference is the clearest clue. The `val_loss` includes uncertainty-weighted terms `exp(−s)·loss + s`. A less negative total loss means the `s` (log_var) terms are contributing differently — likely the log_vars initialized or converged to different values because the trunk features have different statistics.

### The explanation: learned attention pooling → higher-variance trunk features

The TCN's GAP+GMP pooling is fixed and deterministic — it treats every time step equally (GAP) or takes the single maximum (GMP). The CNN attention's `AttentionPooling` layer learns a per-sample softmax over time steps:

```python
weights = softmax(score(tanh(score_hidden(x))))  # (B, T, 1)
output = sum(x * weights, axis=1)                 # weighted average over time
```

This is a strictly more expressive readout. Different samples can weight different time regions differently. The result: **higher-variance features in the trunk output space.** This higher variance propagates to the head outputs:

- **Periodic heads**: Higher variance → more spread in predictions → lower circ_r. But the ang_MAE is still at baseline (1.605 rad ≈ π/2) — the predictions are *varied* but not *correct*.
- **Scalar heads (mchirp)**: Higher variance → potentially noisier features → slightly worse R² than TCN (0.926 vs 0.963). TCN's more constrained pooling acts as a regularizer for well-learned scalars.
- **Sky position**: Benefits from the attention mechanism (locating the source on the sky depends on subtle timing/amplitude modulations that per-sample attention weighting helps with). This is the one genuinely better result.

### The val_loss scale

The less-negative val_loss (−1.53 vs −3.79) is consistent with different log_var convergence. The uncertainty weighting loss is `exp(−s)·angular_loss + s`. If the periodic heads' log_vars drift to different values because the trunk features have different statistics, the `+ s` term shifts the loss baseline. This is a calibration artifact, not a sign that cnn_attention is "learning better" — the ang_MAE confirms it's not.

---

## 5. Verdict

**cnn_attention's outlier behavior is architectural, not config-driven.** There is no hidden config difference to apply to the other models. The mechanism:

1. **Learned attention pooling** produces higher-variance trunk features than TCN's fixed GAP+GMP
2. Higher-variance features → more spread in periodic head predictions → lower circ_r
3. But ang_MAE stays at random baseline → spread ≠ signal
4. The same higher variance slightly degrades scalar regression (mchirp R² drops) while helping sky_position
5. The val_loss scale difference is a log_var calibration artifact

**This does not weaken the degeneracy case.** cnn_attention's lower circ_r looks healthier at a glance, but the ang_MAE tells the real story: 1.605 rad for coa_phase — essentially identical to the random baseline of π/2 = 1.571 rad, and actually *worse* than poc_a's 1.570. The model produces varied wrong answers instead of one constant wrong answer. That's a difference in output diversity, not in information recovery.

### Testable prediction

If the attention pooling were replaced with GAP+GMP (matching TCN's pooling), cnn_attention's circ_r should rise toward TCN levels. The spread is a pooling artifact, not phase signal.

To test this without retraining: check whether the variance of the trunk features (pre-head) is higher for cnn_attention than for TCN on identical inputs. If yes, the circ_r difference is explained by feature statistics, not phase learning.