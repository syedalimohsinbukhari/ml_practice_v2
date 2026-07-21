> **⚠️ UPDATE (2026-07-20):** Run 7 re-ran poc_a, poc_b, cnn_attention, and tcn with updated configs.
> Scalar heads (mchirp, merger_time, snr) remain stable. Raw periodic-head metrics also unchanged —
> all models remain at random-guess level for coa_phase, polarization_angle, and inclination,
> consistent with the φc/ψ degeneracy being effectively exact without inclination input.
> poc_b shows ~8% mchirp regression; cnn_attention shows modest scalar improvements.
> The `analyse_phic_distributions.py` script still needs to be re-run on Run 7 outputs for
> true sky-position angular errors and circular statistics. See **[phic_psi_run7_comparison.md](phic_psi_run7_comparison.md)** for full analysis.
>
> **This document (below) reflects Run 5/6 (July 17-18, 2026).**

====================================================================================================
TABLE 1: SCALAR HEADS — MAE / R² / bias / std_ratio
====================================================================================================

--- mchirp ---

| model            | MAE    | R²     | bias    | std_ratio |
|------------------|--------|--------|---------|-----------|
| poc_a (baseline) | 0.9925 | 0.9581 | 0.0269  | 0.9955    |
| poc_b (PoC)      | 0.9142 | 0.9657 | 0.1146  | 0.9893    |
| tcn              | 0.9960 | 0.9597 | 0.1313  | 1.0003    |
| cnn_baseline     | 3.1839 | 0.6275 | -2.6287 | 0.7290    |
| cnn_attention    | 1.4293 | 0.9120 | -0.0780 | 0.9625    |
| inception_time   | 1.3717 | 0.9101 | -0.0606 | 0.9215    |
| resnet1d         | 1.7693 | 0.8835 | -0.5114 | 0.8774    |

--- merger_time ---

| model            | MAE    | R²      | bias    | std_ratio |
|------------------|--------|---------|---------|-----------|
| poc_a (baseline) | 0.0127 | 0.9210  | -0.0022 | 0.9955    |
| poc_b (PoC)      | 0.0121 | 0.9275  | 0.0009  | 0.9890    |
| tcn              | 0.0126 | 0.9187  | 0.0036  | 0.9873    |
| cnn_baseline     | 0.0407 | 0.2409  | 0.0396  | 0.8615    |
| cnn_attention    | 0.0134 | 0.8910  | 0.0057  | 0.9718    |
| inception_time   | 0.0492 | -0.0009 | -0.0007 | 0.0208    |
| resnet1d         | 0.0173 | 0.8329  | 0.0110  | 0.9729    |

--- snr ---

| model            | MAE    | R²     | bias    | std_ratio |
|------------------|--------|--------|---------|-----------|
| poc_a (baseline) | 0.8333 | 0.7860 | 0.0878  | 0.9196    |
| poc_b (PoC)      | 0.8266 | 0.7868 | -0.0070 | 0.9428    |
| tcn              | 0.8230 | 0.7901 | 0.0490  | 0.9282    |
| cnn_baseline     | 1.0102 | 0.6829 | 0.0208  | 0.9520    |
| cnn_attention    | 0.9470 | 0.7209 | 0.3667  | 0.9728    |
| inception_time   | 0.9639 | 0.7057 | 0.2515  | 0.9430    |
| resnet1d         | 1.4232 | 0.4067 | 1.1916  | 0.7642    |


====================================================================================================
TABLE 2: SKY POSITION — angular error
====================================================================================================

| model            | angular MAE | angular median |
|------------------|-------------|----------------|
| poc_a (baseline) | 12.9°       | 0.0°           |
| poc_b (PoC)      | 12.7°       | 0.0°           |
| tcn              | 3.2°        | 0.0°           |
| cnn_baseline     | 3.0°        | 0.0°           |
| cnn_attention    | 3.6°        | 0.0°           |
| inception_time   | 7.5°        | 0.0°           |
| resnet1d         | 2.2°        | 0.0°           |


====================================================================================================
TABLE 3: PERIODIC HEADS — circular concentration (r) and angular MAE
        r → 0 = uniform/random, r → 1 = perfectly concentrated
====================================================================================================

--- coa_phase (φc [0,2π)) ---

| model            | circ_r | circ_mean | ang_MAE | structured? | peaks                  |
|------------------|--------|-----------|---------|-------------|------------------------|
| poc_a (baseline) | 1.0000 | 315.0°    | 1.5974  | YES         | 315°(1.00)             |
| poc_b (PoC)      | 1.0000 | 315.0°    | 1.5974  | YES         | 315°(1.00)             |
| tcn              | 1.0000 | 315.0°    | 1.5974  | YES         | 315°(1.00)             |
| cnn_baseline     | 0.9924 | 38.7°     | 1.5676  | YES         | 45°(0.60)              |
| cnn_attention    | 0.8788 | 56.9°     | 1.5620  | YES         | 45°(0.72), 135°(0.05)  |
| inception_time   | 0.1183 | 254.4°    | 1.5366  | YES         | 285°(0.18), 115°(0.12) |
| resnet1d         | 1.0000 | 225.0°    | 1.5747  | YES         | 225°(1.00)             |

--- polarization_angle (ψ [0,π)) ---

| model            | circ_r | circ_mean | ang_MAE | structured? | peaks      |
|------------------|--------|-----------|---------|-------------|------------|
| poc_a (baseline) | 1.0000 | 67.5°     | 0.7946  | YES         | 68°(1.00)  |
| poc_b (PoC)      | 1.0000 | 67.5°     | 0.7946  | YES         | 68°(1.00)  |
| tcn              | 1.0000 | 112.5°    | 0.8000  | YES         | 112°(1.00) |
| cnn_baseline     | 0.9998 | 111.9°    | 0.8001  | YES         | 112°(0.98) |
| cnn_attention    | 1.0000 | 112.5°    | 0.8000  | YES         | 112°(1.00) |
| inception_time   | 1.0000 | 157.6°    | 0.7761  | YES         | 158°(1.00) |
| resnet1d         | 1.0000 | 112.5°    | 0.8000  | YES         | 112°(1.00) |

--- inclination (ι [0,π]) ---

| model            | circ_r | circ_mean | ang_MAE | structured? | peaks                           |
|------------------|--------|-----------|---------|-------------|---------------------------------|
| poc_a (baseline) | 0.6948 | 5.1°      | 1.5739  | YES         | —                               |
| poc_b (PoC)      | 0.2304 | 269.9°    | 1.5175  | YES         | 205°(0.06)                      |
| tcn              | 0.4795 | 333.1°    | 1.5652  | YES         | 345°(0.08)                      |
| cnn_baseline     | 0.9893 | 28.6°     | 1.6062  | YES         | 35°(0.49)                       |
| cnn_attention    | 0.3858 | 22.2°     | 1.5229  | YES         | 155°(0.04)                      |
| inception_time   | 0.2498 | 157.4°    | 1.5499  | YES         | 145°(0.18), 325°(0.07)          |
| resnet1d         | 0.2783 | 50.1°     | 1.5744  | YES         | 45°(0.06), 75°(0.05), 95°(0.04) |


====================================================================================================
TABLE 4: QUICK HEALTH CHECK
        ✓ = well predicted   ~ = marginal   ✗ = dead/random
====================================================================================================

| model            | mchirp | merger_time | snr | sky_position | coa_phase | polarization_angle | inclination |
|------------------|--------|-------------|-----|--------------|-----------|--------------------|-------------|
| poc_a (baseline) | ✓      | ✓           | ✓   | ✓            | ✓         | ✓                  | ✓           |
| poc_b (PoC)      | ✓      | ✓           | ✓   | ✓            | ✓         | ✓                  | ~           |
| tcn              | ✓      | ✓           | ✓   | ✓            | ✓         | ✓                  | ~           |
| cnn_baseline     | ~      | ✗           | ✓   | ✓            | ✓         | ✓                  | ✓           |
| cnn_attention    | ✓      | ✓           | ✓   | ✓            | ✓         | ✓                  | ~           |
| inception_time   | ✓      | ✗           | ✓   | ✓            | ✗         | ✓                  | ~           |
| resnet1d         | ✓      | ✓           | ~   | ✓            | ✓         | ✓                  | ~           |


Done.
Run `experiments/phic_psi_poc/analyse_predictions.py` for detailed φc histogram data.
