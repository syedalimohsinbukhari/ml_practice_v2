# Standalone Multi-Step Perturbation Trace (A.3, Redo) — stage: early

Calibration run: fresh init + 200 warmup steps (~1 epoch) on the training split, then the standard trace. A working instrument must read mchirp as DIRECTIONAL here; if it does, the AMBIGUOUS mchirp verdicts at the converged checkpoints are a convergence effect and the final-stage table is interpretable. If mchirp stays AMBIGUOUS even here, the trace methodology itself is unsound and no verdict from it should be used.

25 consecutive gradient steps on one fixed batch. Random-walk reference net/sum = 0.200; directional drift approaches 1.0. Probe Δ is a per-sample paired statistic on a fixed disjoint 512-sample probe (circular loss for periodic heads, transformed-target MSE for mchirp).

| Model | Head | mean cos-sim | net/sum | probe Δ ± SE | t | Verdict |
|---|---|---|---|---|---|---|
| poc_a (baseline) | coa_phase | +0.875 | 0.294 | +0.0368 ± 0.0604 | +0.61 | AMBIGUOUS |
| poc_a (baseline) | polarization_angle | +0.782 | 0.164 | -0.0371 ± 0.0436 | -0.85 | AMBIGUOUS |
| poc_a (baseline) | mchirp | +0.457 | 0.090 | -0.0264 ± 0.0319 | -0.83 | AMBIGUOUS |
| poc_b (poc) | coa_phase | +0.890 | 0.343 | +0.0334 ± 0.0494 | +0.68 | AMBIGUOUS |
| poc_b (poc) | polarization_angle | +0.724 | 0.280 | -0.0378 ± 0.0558 | -0.68 | AMBIGUOUS |
| poc_b (poc) | mchirp | +0.470 | 0.111 | -0.0723 ± 0.0323 | -2.24 | AMBIGUOUS |
| tcn | coa_phase | +0.827 | 0.292 | +0.0515 ± 0.0586 | +0.88 | AMBIGUOUS |
| tcn | polarization_angle | +0.908 | 0.145 | +0.0620 ± 0.0517 | +1.20 | AMBIGUOUS |
| tcn | mchirp | +0.521 | 0.068 | -0.1313 ± 0.0267 | -4.92 | AMBIGUOUS |
| cnn_attention | coa_phase | +0.886 | 0.432 | -0.0157 ± 0.0281 | -0.56 | DIRECTIONAL (coherent drift) |
| cnn_attention | polarization_angle | +0.907 | 0.622 | +0.0486 ± 0.0517 | +0.94 | DIRECTIONAL (coherent drift) |
| cnn_attention | mchirp | +0.489 | 0.296 | -0.5325 ± 0.0610 | -8.73 | AMBIGUOUS |

Escalation rule (unchanged): a DIRECTIONAL verdict on a periodic head with a decreasing probe circular loss that is significant under the paired statistic is a mechanistic hint of slow learning and should be escalated, not filed.
