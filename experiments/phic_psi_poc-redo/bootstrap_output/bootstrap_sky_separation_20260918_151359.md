# Bootstrap CI on Sky-Position Angular Separation (Redo)

**Generated**: 20260918_151359
**N bootstrap**: 10000
**Validation samples**: 5000
**Null hypothesis**: shuffling true sky positions does not change mean angular separation.
**Test**: one-sided — is observed separation significantly *below* the null distribution?

Companion to `sky_angular_separation_hist.py` (2026-09-18), which found the separation distribution matches a uniform-on-sphere null shape (no bimodal structure) but was descriptive only. This is the formal significance test for that same quantity.

## Summary

| Model | Observed (deg) | Null mean (deg) | Null 95% CI | z (σ) | p | Significant? |
|-------|-----------------|-------------------|-------------|-------|---|-------------|
| poc_a (baseline) | 86.6918 | 90.1200 | [89.0993, 91.1617] | +6.54 | 0.0000 | ★ YES (better) |
| poc_b (PoC) | 87.0208 | 90.0517 | [88.9699, 91.1449] | +5.51 | 0.0000 | ★ YES (better) |
| tcn | 87.1099 | 89.9657 | [88.9004, 91.0504] | +5.21 | 0.0000 | ★ YES (better) |
| cnn_attention | 77.5206 | 90.0998 | [89.0355, 91.1664] | +23.23 | 0.0000 | ★ YES (better) |

## Interpretation

- **p < 0.05, z > 0**: mean angular separation is significantly lower than random → evidence of localization.
- **p ≥ 0.05**: not distinguishable from random placement.
- **z < 0**: separation is *worse* than random (collapse/bias artifact).
