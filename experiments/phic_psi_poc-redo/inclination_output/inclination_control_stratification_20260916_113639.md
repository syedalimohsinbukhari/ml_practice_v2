# Inclination-Stratified Chirp-Mass MAE/R^2 -- Scalar-Control Check (Redo)

**Generated**: 20260916_113639
**Bands**: face-on |cos(iota)| > 0.9, edge-on |cos(iota)| < 0.5, mixed in between (matches original thesis chapter Section 3 / inclination_stratification.py)
**Validation samples**: 5000

## Key question

The original thesis chapter's Section 6.7 uses the known-uninformative inclination head as a same-model noise floor for the phi_c/psi inclination stratification. This script checks the companion question, on the redo's own checkpoints: does the face-on/mixed/edge-on banding scheme itself inject spurious MAE/R^2 variance, even into a head (chirp mass) we already know is well-learned and non-angular? If the spread here is small relative to the phi_c/psi deviations in inclination_stratification.py's output, banding-induced noise is not a plausible alternative explanation for those deviations.

| Model | ALL MAE | face-on MAE | mixed MAE | edge-on MAE | MAE spread | R^2 spread |
|-------|---------|-------------|-----------|-------------|------------|------------|
| poc_a (baseline) | 1.0116 | 0.9688 | 1.0498 | 1.0044 | 0.0810 | 0.0113 |
| poc_b (PoC) | 1.0040 | 0.9955 | 1.0310 | 0.9801 | 0.0509 | 0.0069 |
| tcn | 0.9782 | 0.9494 | 1.0000 | 0.9781 | 0.0506 | 0.0064 |
| cnn_attention | 1.4201 | 1.3963 | 1.4561 | 1.3988 | 0.0598 | 0.0166 |

## Verdict

Small band-to-band MAE/R^2 spread here, relative to the phi_c/psi edge-on deviations in inclination_stratification.py's output, supports treating those deviations as head-specific (iota-noise-floor-scale) rather than an artifact of slicing the validation set into unequal, differently-composed bands.
