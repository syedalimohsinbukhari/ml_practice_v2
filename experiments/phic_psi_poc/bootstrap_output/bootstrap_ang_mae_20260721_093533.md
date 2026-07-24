# Bootstrap CI on ang_MAE — Periodic Heads

**Generated**: 20260721_093533
**N bootstrap**: 10000
**Validation samples**: 5000
**Null hypothesis**: Shuffling true labels (destroying strain→angle association) does not change ang_MAE.
**Test**: One-sided — is observed ang_MAE significantly *below* the null distribution?

## Summary

| Model | Head | Observed | Null mean | Null 95% CI | z (σ) | p | Significant? |
|-------|------|----------|-----------|-------------|-------|---|-------------|
| poc_a (baseline) | coa_phase | 1.5693 | 1.5679 | [1.5542, 1.5816] | -0.20 | 0.5793 | no (worse) |
| poc_a (baseline) | polarization_angle | 0.7795 | 0.7820 | [0.7713, 0.7930] | +0.46 | 0.3240 | no (better) |
| poc_a (baseline) | inclination | 1.5609 | 1.5713 | [1.5518, 1.5910] | +1.04 | 0.1521 | no (better) |
| poc_b (PoC) | coa_phase | 1.5727 | 1.5702 | [1.5662, 1.5740] | -1.25 | 0.8947 | no (worse) |
| poc_b (PoC) | polarization_angle | 0.7798 | 0.7798 | [0.7776, 0.7820] | -0.05 | 0.5178 | no (worse) |
| poc_b (PoC) | inclination | 1.5713 | 1.5722 | [1.5492, 1.5950] | +0.07 | 0.4640 | no (better) |
| tcn | coa_phase | 1.5779 | 1.5741 | [1.5608, 1.5875] | -0.56 | 0.7115 | no (worse) |
| tcn | polarization_angle | 0.7845 | 0.7834 | [0.7771, 0.7896] | -0.33 | 0.6296 | no (worse) |
| tcn | inclination | 1.5560 | 1.5656 | [1.5502, 1.5809] | +1.21 | 0.1139 | no (better) |
| cnn_attention | coa_phase | 1.5968 | 1.5683 | [1.5455, 1.5911] | -2.43 | 0.9937 | no (worse) |
| cnn_attention | polarization_angle | 0.7842 | 0.7847 | [0.7726, 0.7969] | +0.07 | 0.4722 | no (better) |
| cnn_attention | inclination | 1.5337 | 1.5718 | [1.5481, 1.5955] | +3.17 | 0.0007 | ★ YES (better) |

## Interpretation

- **p < 0.05, z > 0**: ang_MAE is significantly better than random → evidence of learning.
- **p ≥ 0.05**: ang_MAE is not distinguishable from random → no evidence of learning.
- **z < 0**: ang_MAE is *worse* than random — model does worse than guessing (usually a collapse or bias artifact).

## Per-model details

### poc_a (baseline)

**coa_phase**: observed=1.5693, null_mean=1.5679±0.0069, z=-0.20σ, p=0.5793

**polarization_angle**: observed=0.7795, null_mean=0.7820±0.0055, z=+0.46σ, p=0.3240

**inclination**: observed=1.5609, null_mean=1.5713±0.0100, z=+1.04σ, p=0.1521

### poc_b (PoC)

**coa_phase**: observed=1.5727, null_mean=1.5702±0.0020, z=-1.25σ, p=0.8947

**polarization_angle**: observed=0.7798, null_mean=0.7798±0.0011, z=-0.05σ, p=0.5178

**inclination**: observed=1.5713, null_mean=1.5722±0.0117, z=+0.07σ, p=0.4640

### tcn

**coa_phase**: observed=1.5779, null_mean=1.5741±0.0068, z=-0.56σ, p=0.7115

**polarization_angle**: observed=0.7845, null_mean=0.7834±0.0032, z=-0.33σ, p=0.6296

**inclination**: observed=1.5560, null_mean=1.5656±0.0079, z=+1.21σ, p=0.1139

### cnn_attention

**coa_phase**: observed=1.5968, null_mean=1.5683±0.0117, z=-2.43σ, p=0.9937

**polarization_angle**: observed=0.7842, null_mean=0.7847±0.0063, z=+0.07σ, p=0.4722

**inclination**: observed=1.5337, null_mean=1.5718±0.0120, z=+3.17σ, p=0.0007

