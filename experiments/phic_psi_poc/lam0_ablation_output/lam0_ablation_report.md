# λ=0 Ablation — Circular Loss Drift

**Question**: Does the upward drift in circular loss disappear at λ=0?

| Model | Head | Metric | λ=0 start | λ=0 end | λ=0 Δ | λ=0.01 Δ | Verdict |
|-------|------|--------|-----------|----------|-------|-----------|--------|
| poc_a (baseline) | coa_phase | val circ loss | 1.0048 | 1.0058 | +0.0010 | +0.0249 | drift ABSENT at λ=0 |
| poc_a (baseline) | coa_phase | train circ loss | 0.9967 | 0.9940 | -0.0027 | -0.0236 | drift ABSENT at λ=0 |
| poc_a (baseline) | coa_phase | val std_ratio | 21.8032 | 82.9722 | +61.1690 | +0.3136 | expected (|v| diverges without penalty) |
| poc_a (baseline) | polarization_angle | val circ loss | 0.9959 | 0.9962 | +0.0002 | +0.0163 | drift ABSENT at λ=0 |
| poc_a (baseline) | polarization_angle | train circ loss | 0.9998 | 0.9824 | -0.0174 | -0.0214 | decreases at λ=0 (opposite direction) |
| poc_a (baseline) | polarization_angle | val std_ratio | 18.9412 | 10.0275 | -8.9136 | -1.8620 | expected (|v| diverges without penalty) |
| tcn | coa_phase | val circ loss | 1.0046 | 1.0057 | +0.0011 | +0.0204 | drift ABSENT at λ=0 |
| tcn | coa_phase | train circ loss | 0.9968 | 0.9941 | -0.0026 | -0.0310 | drift ABSENT at λ=0 |
| tcn | coa_phase | val std_ratio | 21.2772 | 73.1470 | +51.8698 | -0.2467 | expected (|v| diverges without penalty) |
| tcn | polarization_angle | val circ loss | 0.9958 | 1.0030 | +0.0072 | +0.0136 | PERSISTS — penalty NOT the cause |
| tcn | polarization_angle | train circ loss | 0.9997 | 0.9791 | -0.0206 | -0.0231 | decreases at λ=0 (opposite direction) |
| tcn | polarization_angle | val std_ratio | 19.1829 | 7.7483 | -11.4346 | -0.7301 | expected (|v| diverges without penalty) |

![trajectories](lam0_ablation_trajectories.png)

### Interpretation

- **PERSISTS**: drift occurs even without the penalty → penalty is NOT the cause.
- **STOPPED**: drift disappears at λ=0 → penalty (or its interaction with log_var uncertainty weighting) IS the cause. Tunable, not degeneracy evidence.
- **std_ratio drift expected**: without penalty, |v| diverges. Early epochs (before |v| drifts far) are the clean comparison window.
