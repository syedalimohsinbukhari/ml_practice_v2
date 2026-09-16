# Bootstrap CI on ang_MAE — Periodic Heads (Redo)

**Generated**: 20260916_113955
**N bootstrap**: 10000
**Validation samples**: 5000
**Null hypothesis**: Shuffling true labels (destroying strain→angle association) does not change ang_MAE.
**Test**: One-sided — is observed ang_MAE significantly *below* the null distribution?

## Summary

| Model | Head | Observed | Null mean | Null 95% CI | z (σ) | p | Significant? |
|-------|------|----------|-----------|-------------|-------|---|-------------|
| poc_a (baseline) | coa_phase | 1.5736 | 1.5695 | [1.5480, 1.5904] | -0.38 | 0.6494 | no (worse) |
| poc_a (baseline) | polarization_angle | 0.7850 | 0.7836 | [0.7802, 0.7870] | -0.82 | 0.7931 | no (worse) |
| poc_a (baseline) | inclination | 1.5494 | 1.5701 | [1.5498, 1.5900] | +2.00 | 0.0223 | ★ YES (better) |
| poc_b (PoC) | coa_phase | 1.5716 | 1.5703 | [1.5643, 1.5765] | -0.39 | 0.6494 | no (worse) |
| poc_b (PoC) | polarization_angle | 0.7837 | 0.7837 | [0.7834, 0.7839] | -0.63 | 0.7400 | no (worse) |
| poc_b (PoC) | inclination | 1.5448 | 1.5728 | [1.5516, 1.5944] | +2.58 | 0.0052 | ★ YES (better) |
| tcn | coa_phase | 1.5853 | 1.5768 | [1.5629, 1.5909] | -1.18 | 0.8791 | no (worse) |
| tcn | polarization_angle | 0.7881 | 0.7856 | [0.7760, 0.7952] | -0.50 | 0.6920 | no (worse) |
| tcn | inclination | 1.5400 | 1.5699 | [1.5472, 1.5927] | +2.58 | 0.0059 | ★ YES (better) |
| cnn_attention | coa_phase | 1.5936 | 1.5702 | [1.5473, 1.5935] | -1.97 | 0.9759 | no (worse) |
| cnn_attention | polarization_angle | 0.7779 | 0.7858 | [0.7741, 0.7973] | +1.33 | 0.0908 | no (better) |
| cnn_attention | inclination | 1.5290 | 1.5723 | [1.5490, 1.5961] | +3.58 | 0.0003 | ★ YES (better) |

## Interpretation

- **p < 0.05, z > 0**: ang_MAE is significantly better than random → evidence of learning.
- **p ≥ 0.05**: ang_MAE is not distinguishable from random → no evidence of learning.
- **z < 0**: ang_MAE is *worse* than random — model does worse than guessing (usually a collapse or bias artifact).

## Per-model details

### poc_a (baseline)

**coa_phase**: observed=1.5736, null_mean=1.5695±0.0108, z=-0.38σ, p=0.6494

**polarization_angle**: observed=0.7850, null_mean=0.7836±0.0017, z=-0.82σ, p=0.7931

**inclination**: observed=1.5494, null_mean=1.5701±0.0103, z=+2.00σ, p=0.0223

### poc_b (PoC)

**coa_phase**: observed=1.5716, null_mean=1.5703±0.0032, z=-0.39σ, p=0.6494

**polarization_angle**: observed=0.7837, null_mean=0.7837±0.0001, z=-0.63σ, p=0.7400

**inclination**: observed=1.5448, null_mean=1.5728±0.0109, z=+2.58σ, p=0.0052

### tcn

**coa_phase**: observed=1.5853, null_mean=1.5768±0.0072, z=-1.18σ, p=0.8791

**polarization_angle**: observed=0.7881, null_mean=0.7856±0.0049, z=-0.50σ, p=0.6920

**inclination**: observed=1.5400, null_mean=1.5699±0.0116, z=+2.58σ, p=0.0059

### cnn_attention

**coa_phase**: observed=1.5936, null_mean=1.5702±0.0119, z=-1.97σ, p=0.9759

**polarization_angle**: observed=0.7779, null_mean=0.7858±0.0059, z=+1.33σ, p=0.0908

**inclination**: observed=1.5290, null_mean=1.5723±0.0121, z=+3.58σ, p=0.0003

