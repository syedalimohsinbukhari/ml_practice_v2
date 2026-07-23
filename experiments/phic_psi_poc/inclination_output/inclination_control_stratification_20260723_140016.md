# Inclination-Stratified Chirp-Mass MAE/R^2 -- Scalar-Control Check

**Generated**: 20260723_140016
**Bands**: face-on |cos(iota)| > 0.9, edge-on |cos(iota)| < 0.5, mixed in between (matches thesis chapter Section 3 / inclination_stratification.py)
**Validation samples**: 5000

## Key question

Section 6.7 of the thesis chapter uses the known-uninformative inclination head as a same-model noise floor for the phi_c/psi inclination stratification. This script checks the companion question: does the face-on/mixed/edge-on banding scheme itself inject spurious MAE/R^2 variance, even into a head (chirp mass) we already know is well-learned and non-angular? If the spread here is small relative to the phi_c/psi deviations in Table 6.6, banding-induced noise is not a plausible alternative explanation for those deviations.

| Model | ALL MAE | face-on MAE | mixed MAE | edge-on MAE | MAE spread | R^2 spread |
|-------|---------|-------------|-----------|-------------|------------|------------|
| poc_a (baseline) | 0.9701 | 0.9486 | 0.9946 | 0.9603 | 0.0460 | 0.0073 |
| poc_b (PoC) | 1.0363 | 1.0007 | 1.0808 | 1.0155 | 0.0801 | 0.0087 |
| tcn | 0.9817 | 0.9577 | 1.0085 | 0.9714 | 0.0508 | 0.0097 |
| cnn_attention | 1.3661 | 1.3451 | 1.3794 | 1.3690 | 0.0343 | 0.0053 |

## Verdict

Small band-to-band MAE/R^2 spread here, relative to the phi_c/psi edge-on deviations in Table 6.6, supports treating those deviations as head-specific (iota-noise-floor-scale) rather than an artifact of slicing the validation set into unequal, differently-composed bands.
