# Standalone Multi-Step Perturbation Trace (A.3)

25 consecutive gradient steps on one fixed batch, Run 7 (λ=0.01) checkpoints. Random-walk reference net/sum = 0.200; directional drift approaches 1.0.

| Model | Head | mean cos-sim | net/sum | Verdict |
|---|---|---|---|---|
| poc_a (baseline) | coa_phase | +0.933 | 0.288 | AMBIGUOUS |
| poc_a (baseline) | polarization_angle | +0.960 | 0.586 | DIRECTIONAL (coherent drift) |
| poc_a (baseline) | mchirp | +0.602 | 0.038 | AMBIGUOUS |
| poc_b (poc) | coa_phase | +0.956 | 0.394 | AMBIGUOUS |
| poc_b (poc) | polarization_angle | +0.801 | 0.233 | AMBIGUOUS |
| poc_b (poc) | mchirp | +0.544 | 0.035 | AMBIGUOUS |
| tcn | coa_phase | +0.943 | 0.925 | DIRECTIONAL (coherent drift) |
| tcn | polarization_angle | +0.869 | 0.394 | AMBIGUOUS |
| tcn | mchirp | +0.591 | 0.029 | AMBIGUOUS |
| cnn_attention | coa_phase | +0.658 | 0.399 | AMBIGUOUS |
| cnn_attention | polarization_angle | +0.641 | 0.324 | AMBIGUOUS |
| cnn_attention | mchirp | +0.511 | 0.164 | AMBIGUOUS |

Interpretation: if coa_phase/pol_angle are OSCILLATORY while mchirp is DIRECTIONAL, the 89x single-step rel_change asymmetry (Run 7, A.3) was movement without learning — large steps that cancel — and the item closes as consistent with the null. A DIRECTIONAL verdict on a periodic head with decreasing probe circular loss would instead be the first mechanistic hint of slow learning and should be escalated, not filed.
