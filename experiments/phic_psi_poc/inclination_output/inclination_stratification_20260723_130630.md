# Inclination-Stratified ang_MAE — All Periodic Heads

**Generated**: 20260723_130630
**Bands**: face-on |cos(iota)| > 0.9, edge-on |cos(iota)| < 0.5, mixed in between (matches thesis chapter Section 3)
**Validation samples**: 5000

## Key question

The chapter's analytic prerequisite study finds the phi_c-psi degeneracy is exact face-on and only partially breakable edge-on. If any recoverable signal survives in this population, ang_MAE should improve — face-on to edge-on — below the null. If it stays flat at the null in every band, the degeneracy is exact across the tested population, not just at face-on.

### coa_phase (null = 1.5708 rad)

| Model | ALL | face-on | mixed | edge-on | Improves face-on->edge-on? | Edge-on vs null |
|-------|-----|---------|-------|---------|------------------------------|------------------|
| poc_a (baseline) | 1.5693 | 1.6042 | 1.5623 | 1.5467 | YES | +0.0241 |
| poc_b (PoC) | 1.5727 | 1.5681 | 1.5858 | 1.5612 | YES | +0.0096 |
| tcn | 1.5779 | 1.5527 | 1.5681 | 1.6115 | NO | -0.0407 |
| cnn_attention | 1.5968 | 1.6588 | 1.5649 | 1.5795 | YES | -0.0087 |

### polarization_angle (null = 0.7854 rad)

| Model | ALL | face-on | mixed | edge-on | Improves face-on->edge-on? | Edge-on vs null |
|-------|-----|---------|-------|---------|------------------------------|------------------|
| poc_a (baseline) | 0.7795 | 0.7832 | 0.7523 | 0.8081 | NO | -0.0227 |
| poc_b (PoC) | 0.7798 | 0.7917 | 0.7707 | 0.7800 | YES | +0.0054 |
| tcn | 0.7845 | 0.7915 | 0.7714 | 0.7937 | NO | -0.0083 |
| cnn_attention | 0.7842 | 0.7834 | 0.7676 | 0.8043 | NO | -0.0189 |

### inclination (null = 1.5708 rad)

| Model | ALL | face-on | mixed | edge-on | Improves face-on->edge-on? | Edge-on vs null |
|-------|-----|---------|-------|---------|------------------------------|------------------|
| poc_a (baseline) | 1.5609 | 1.5872 | 1.5139 | 1.5926 | NO | -0.0218 |
| poc_b (PoC) | 1.5713 | 1.5493 | 1.5867 | 1.5728 | NO | -0.0020 |
| tcn | 1.5560 | 1.5613 | 1.5483 | 1.5604 | YES | +0.0104 |
| cnn_attention | 1.5337 | 1.4764 | 1.5348 | 1.5828 | NO | -0.0120 |

## Verdict

A positive result would be: edge-on ang_MAE noticeably below null and below face-on ang_MAE, consistent with the analytic degeneracy weakening away from face-on (thesis chapter Section 3).

Anything else — flat across bands, edge-on at or above null, no improvement toward edge-on — is consistent with the degeneracy hypothesis holding across the entire tested population, not just face-on.
