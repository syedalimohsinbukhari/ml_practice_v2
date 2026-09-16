# SNR-Stratified ang_MAE — All Periodic Heads (Redo)

**Generated**: 20260916_113730
**N terciles**: 3
**Validation samples**: 5000

## Key question

If the degeneracy is breakable, the highest-SNR events (loudest, best-measured) should show lower ang_MAE than the lowest-SNR events. If even the high-SNR tercile sits at the random baseline, the signal genuinely isn't there — this is physics, not engineering.

### coa_phase (null = 1.5708 rad)

| Model | ALL | Low SNR | Mid SNR | High SNR | Improves? | High-SNR vs null |
|-------|-----|---------|---------|----------|-----------|------------------|
| poc_a (baseline) | 1.5736 | 1.5686 | 1.5521 | 1.6001 | NO | -0.0293 |
| poc_b (PoC) | 1.5716 | 1.5382 | 1.5729 | 1.6036 | NO | -0.0328 |
| tcn | 1.5853 | 1.6321 | 1.5739 | 1.5499 | YES ↓ | +0.0209 |
| cnn_attention | 1.5936 | 1.6036 | 1.5771 | 1.6001 | partial | -0.0293 |

### polarization_angle (null = 0.7854 rad)

| Model | ALL | Low SNR | Mid SNR | High SNR | Improves? | High-SNR vs null |
|-------|-----|---------|---------|----------|-----------|------------------|
| poc_a (baseline) | 0.7850 | 0.7859 | 0.7833 | 0.7858 | partial | -0.0004 |
| poc_b (PoC) | 0.7837 | 0.7865 | 0.7832 | 0.7815 | YES ↓ | +0.0039 |
| tcn | 0.7881 | 0.7902 | 0.7964 | 0.7776 | partial | +0.0078 |
| cnn_attention | 0.7779 | 0.7835 | 0.7757 | 0.7745 | YES ↓ | +0.0109 |

### inclination (null = 1.5708 rad)

| Model | ALL | Low SNR | Mid SNR | High SNR | Improves? | High-SNR vs null |
|-------|-----|---------|---------|----------|-----------|------------------|
| poc_a (baseline) | 1.5494 | 1.5332 | 1.5731 | 1.5419 | NO | +0.0289 |
| poc_b (PoC) | 1.5448 | 1.5439 | 1.5471 | 1.5433 | partial | +0.0275 |
| tcn | 1.5400 | 1.5296 | 1.5621 | 1.5284 | partial | +0.0424 |
| cnn_attention | 1.5290 | 1.5345 | 1.5123 | 1.5402 | NO | +0.0306 |

## Verdict

A positive result would be: high-SNR ang_MAE noticeably below null and below low-SNR ang_MAE, monotonically improving with SNR.

Anything else — flat across terciles, high-SNR at or above null, no monotonic trend — is consistent with the degeneracy hypothesis.
