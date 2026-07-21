# SNR-Stratified ang_MAE — All Periodic Heads

**Generated**: 20260721_094039
**N terciles**: 3
**Validation samples**: 5000

## Key question

If the degeneracy is breakable, the highest-SNR events (loudest, best-measured) should show lower ang_MAE than the lowest-SNR events. If even the high-SNR tercile sits at the random baseline, the signal genuinely isn't there — this is physics, not engineering.

### coa_phase (null = 1.5708 rad)

| Model | ALL | Low SNR | Mid SNR | High SNR | Improves? | High-SNR vs null |
|-------|-----|---------|---------|----------|-----------|------------------|
| poc_a (baseline) | 1.5693 | 1.5290 | 1.5723 | 1.6066 | NO | -0.0358 |
| poc_b (PoC) | 1.5727 | 1.5469 | 1.5711 | 1.5999 | NO | -0.0292 |
| tcn | 1.5779 | 1.6330 | 1.5573 | 1.5434 | YES ↓ | +0.0274 |
| cnn_attention | 1.5968 | 1.5967 | 1.5873 | 1.6064 | NO | -0.0356 |

### polarization_angle (null = 0.7854 rad)

| Model | ALL | Low SNR | Mid SNR | High SNR | Improves? | High-SNR vs null |
|-------|-----|---------|---------|----------|-----------|------------------|
| poc_a (baseline) | 0.7795 | 0.7746 | 0.7786 | 0.7853 | NO | +0.0001 |
| poc_b (PoC) | 0.7798 | 0.7820 | 0.7767 | 0.7807 | partial | +0.0047 |
| tcn | 0.7845 | 0.7845 | 0.7895 | 0.7795 | partial | +0.0059 |
| cnn_attention | 0.7842 | 0.7722 | 0.8013 | 0.7791 | NO | +0.0063 |

### inclination (null = 1.5708 rad)

| Model | ALL | Low SNR | Mid SNR | High SNR | Improves? | High-SNR vs null |
|-------|-----|---------|---------|----------|-----------|------------------|
| poc_a (baseline) | 1.5609 | 1.5634 | 1.5572 | 1.5621 | partial | +0.0087 |
| poc_b (PoC) | 1.5713 | 1.5442 | 1.5804 | 1.5894 | NO | -0.0186 |
| tcn | 1.5560 | 1.5543 | 1.5364 | 1.5774 | NO | -0.0066 |
| cnn_attention | 1.5337 | 1.5260 | 1.5480 | 1.5271 | NO | +0.0437 |

## Verdict

A positive result would be: high-SNR ang_MAE noticeably below null and below low-SNR ang_MAE, monotonically improving with SNR.

Anything else — flat across terciles, high-SNR at or above null, no monotonic trend — is consistent with the degeneracy hypothesis.
