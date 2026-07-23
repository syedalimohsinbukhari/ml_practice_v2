# Standalone Multi-Step Perturbation Trace (A.3) — stage: early

Calibration run: fresh init + 200 warmup steps (~1 epoch) on the training split, then the standard trace. A working instrument must read mchirp as DIRECTIONAL here; if it does, the AMBIGUOUS mchirp verdicts at the converged checkpoints are a convergence effect and the final-stage table is interpretable. If mchirp stays AMBIGUOUS even here, the trace methodology itself is unsound and no verdict from it should be used.

25 consecutive gradient steps on one fixed batch. Random-walk reference net/sum = 0.200; directional drift approaches 1.0. Probe Δ is a per-sample paired statistic on a fixed disjoint 512-sample probe (circular loss for periodic heads, transformed-target MSE for mchirp).

| Model | Head | mean cos-sim | net/sum | probe Δ ± SE | t | Verdict |
|---|---|---|---|---|---|---|
| poc_a (baseline) | coa_phase | +0.580 | 0.158 | -0.0173 ± 0.0111 | -1.56 | AMBIGUOUS |
| poc_a (baseline) | polarization_angle | +0.725 | 0.140 | +0.0044 ± 0.0499 | +0.09 | AMBIGUOUS |
| poc_a (baseline) | mchirp | +0.522 | 0.091 | -0.1583 ± 0.0304 | -5.20 | AMBIGUOUS |
| poc_b (poc) | coa_phase | +0.754 | 0.079 | -0.0059 ± 0.0091 | -0.65 | AMBIGUOUS |
| poc_b (poc) | polarization_angle | +0.758 | 0.085 | +0.0013 ± 0.0207 | +0.06 | AMBIGUOUS |
| poc_b (poc) | mchirp | +0.477 | 0.128 | -0.1292 ± 0.0382 | -3.38 | AMBIGUOUS |
| tcn | coa_phase | +0.709 | 0.240 | -0.0018 ± 0.0405 | -0.04 | AMBIGUOUS |
| tcn | polarization_angle | +0.828 | 0.152 | +0.0345 ± 0.0573 | +0.60 | AMBIGUOUS |
| tcn | mchirp | +0.530 | 0.096 | -0.1132 ± 0.0276 | -4.10 | AMBIGUOUS |
| cnn_attention | coa_phase | +0.487 | 0.421 | +0.0008 ± 0.0374 | +0.02 | DIRECTIONAL (coherent drift) |
| cnn_attention | polarization_angle | +0.551 | 0.369 | -0.0309 ± 0.0486 | -0.64 | AMBIGUOUS |
| cnn_attention | mchirp | +0.089 | 0.286 | -0.6457 ± 0.0760 | -8.50 | OSCILLATORY (noise-like, random-walk scale) |

Escalation rule (unchanged): a DIRECTIONAL verdict on a periodic head with a decreasing probe circular loss that is significant under the paired statistic is a mechanistic hint of slow learning and should be escalated, not filed.
