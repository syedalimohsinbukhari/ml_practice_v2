# Step 0 Diagnostic — std_ratio Gate, across λ rounds

Answers whether `poc_redo_b`'s `circular_loss_combo_A`/`combo_B` result is interpretable at each λ tried, using the std_ratio gate from [`preregistration_lam_retune.md`](preregistration_lam_retune.md) (thresholds copied verbatim, not re-derived).

| λ round | Variant | Head | frac unhealthy (last 40 ep) | trend/ep | final std_ratio | Gate |
|---|---|---|---|---|---|---|
| λ=0.01 (Round 1) | b | coa_phase | 0.225 | +0.00638 | 0.583 | FAIL |
| λ=0.01 (Round 1) | b | polarization_angle | 0.850 | -0.00402 | 0.251 | FAIL |
| λ=0.01 (Round 1) | a | coa_phase | 0.750 | -0.00721 | 0.407 | FAIL |
| λ=0.01 (Round 1) | a | polarization_angle | 1.000 | +0.00076 | 0.374 | FAIL |
| λ=0.05 (retune) | b | coa_phase | 1.000 | +0.00093 | 0.315 | FAIL |
| λ=0.05 (retune) | b | polarization_angle | 0.725 | +0.00679 | 0.531 | FAIL |
| λ=0.05 (retune) | a | coa_phase | 0.350 | -0.00561 | 0.522 | FAIL |
| λ=0.05 (retune) | a | polarization_angle | 0.425 | +0.00383 | 0.572 | FAIL |

**Overall verdict — round under test (λ=0.05 (retune)), all 4 readings must pass: FAIL — UNINTERPRETABLE**

Earlier rounds are shown for context, not re-adjudicated. See log for the descriptive weight_combo_A/B trajectory.
