# Step 0 Diagnostic — std_ratio Gate (phic_psi_poc_redo_b)

Answers whether the flat `circular_loss_combo_A`/`combo_B` result (`NOTES.md`, 2026-09-15) is interpretable, using the std_ratio gate from [`preregistration_lam_retune.md`](../phic_psi_poc/preregistration_lam_retune.md) (thresholds copied verbatim, not re-derived).

| Run | Head | frac unhealthy (last 40 ep) | trend/ep | final std_ratio | Gate |
|---|---|---|---|---|---|
| poc_redo_b (poc, tcn) | coa_phase | 0.225 | +0.00638 | 0.583 | FAIL |
| poc_redo_b (poc, tcn) | polarization_angle | 0.850 | -0.00402 | 0.251 | FAIL |
| poc_redo_a (baseline, tcn) | coa_phase | 0.750 | -0.00721 | 0.407 | FAIL |
| poc_redo_a (baseline, tcn) | polarization_angle | 1.000 | +0.00076 | 0.374 | FAIL |

**Overall verdict (poc_redo_b, both heads must pass): FAIL — UNINTERPRETABLE**

See log for the descriptive weight_combo_A/B trajectory and the baseline-mode (poc_redo_a) comparison.
