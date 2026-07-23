# Standalone Multi-Step Perturbation Trace (A.3) — stage: final

25 consecutive gradient steps on one fixed batch. Random-walk reference net/sum = 0.200; directional drift approaches 1.0. Probe Δ is a per-sample paired statistic on a fixed disjoint 512-sample probe (circular loss for periodic heads, transformed-target MSE for mchirp).

| Model | Head | mean cos-sim | net/sum | probe Δ ± SE | t | Verdict |
|---|---|---|---|---|---|---|
| poc_a (baseline) | coa_phase | +0.929 | 0.302 | -0.0177 ± 0.0205 | -0.86 | AMBIGUOUS |
| poc_a (baseline) | polarization_angle | +0.964 | 0.626 | +0.0426 ± 0.0519 | +0.82 | DIRECTIONAL (coherent drift) |
| poc_a (baseline) | mchirp | +0.602 | 0.038 | +0.0208 ± 0.0050 | +4.18 | AMBIGUOUS |
| poc_b (poc) | coa_phase | +0.955 | 0.390 | -0.0132 ± 0.0132 | -1.00 | AMBIGUOUS |
| poc_b (poc) | polarization_angle | +0.801 | 0.234 | +0.0039 ± 0.0023 | +1.72 | AMBIGUOUS |
| poc_b (poc) | mchirp | +0.544 | 0.035 | +0.0404 ± 0.0045 | +8.99 | AMBIGUOUS |
| tcn | coa_phase | +0.941 | 0.923 | -0.0097 ± 0.0491 | -0.20 | DIRECTIONAL (coherent drift) |
| tcn | polarization_angle | +0.816 | 0.356 | +0.0504 ± 0.0312 | +1.61 | AMBIGUOUS |
| tcn | mchirp | +0.591 | 0.028 | +0.0169 ± 0.0039 | +4.28 | AMBIGUOUS |
| cnn_attention | coa_phase | +0.632 | 0.407 | -0.0439 ± 0.0400 | -1.10 | DIRECTIONAL (coherent drift) |
| cnn_attention | polarization_angle | +0.603 | 0.335 | +0.0232 ± 0.0337 | +0.69 | AMBIGUOUS |
| cnn_attention | mchirp | +0.502 | 0.186 | +0.0647 ± 0.0148 | +4.37 | AMBIGUOUS |

Escalation rule (unchanged): a DIRECTIONAL verdict on a periodic head with a decreasing probe circular loss that is significant under the paired statistic is a mechanistic hint of slow learning and should be escalated, not filed.
