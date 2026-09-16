# Inclination-Stratified ang_MAE — All Periodic Heads (Redo)

**Generated**: 20260916_113601
**Bands**: face-on |cos(iota)| > 0.9, edge-on |cos(iota)| < 0.5, mixed in between (matches original thesis chapter Section 3)
**Validation samples**: 5000

## Key question

The chapter's analytic prerequisite study finds the phi_c-psi degeneracy is exact face-on and only partially breakable edge-on. If any recoverable signal survives in this population, ang_MAE should improve — face-on to edge-on — below the null. If it stays flat at the null in every band, the degeneracy is exact across the tested population, not just at face-on. This is the redo's own run against the corrected `2φc±2ψ` formula's checkpoints, not a reuse of the original's (wrong-formula) result.

### coa_phase (null = 1.5708 rad)

| Model | ALL | face-on | mixed | edge-on | Improves face-on->edge-on? | Edge-on vs null |
|-------|-----|---------|-------|---------|------------------------------|------------------|
| poc_a (baseline) | 1.5736 | 1.5886 | 1.5670 | 1.5680 | YES | +0.0028 |
| poc_b (PoC) | 1.5716 | 1.5669 | 1.5863 | 1.5585 | YES | +0.0123 |
| tcn | 1.5853 | 1.5348 | 1.5965 | 1.6165 | NO | -0.0457 |
| cnn_attention | 1.5936 | 1.6472 | 1.5763 | 1.5669 | YES | +0.0039 |

### polarization_angle (null = 0.7854 rad)

| Model | ALL | face-on | mixed | edge-on | Improves face-on->edge-on? | Edge-on vs null |
|-------|-----|---------|-------|---------|------------------------------|------------------|
| poc_a (baseline) | 0.7850 | 0.7926 | 0.7773 | 0.7873 | YES | -0.0019 |
| poc_b (PoC) | 0.7837 | 0.7956 | 0.7771 | 0.7811 | YES | +0.0043 |
| tcn | 0.7881 | 0.7889 | 0.7692 | 0.8094 | NO | -0.0240 |
| cnn_attention | 0.7779 | 0.7805 | 0.7641 | 0.7917 | NO | -0.0063 |

### inclination (null = 1.5708 rad)

| Model | ALL | face-on | mixed | edge-on | Improves face-on->edge-on? | Edge-on vs null |
|-------|-----|---------|-------|---------|------------------------------|------------------|
| poc_a (baseline) | 1.5494 | 1.5241 | 1.5535 | 1.5670 | NO | +0.0038 |
| poc_b (PoC) | 1.5448 | 1.4910 | 1.5527 | 1.5827 | NO | -0.0119 |
| tcn | 1.5400 | 1.5212 | 1.5404 | 1.5561 | NO | +0.0147 |
| cnn_attention | 1.5290 | 1.4970 | 1.5271 | 1.5593 | NO | +0.0115 |

## Verdict

A positive result would be: edge-on ang_MAE noticeably below null and below face-on ang_MAE, consistent with the analytic degeneracy weakening away from face-on (thesis chapter Section 3).

Anything else — flat across bands, edge-on at or above null, no improvement toward edge-on — is consistent with the degeneracy hypothesis holding across the entire tested population, not just face-on.
