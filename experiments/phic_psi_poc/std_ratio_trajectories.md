# std_ratio Full Trajectories — Run 7 (Magnitude Penalty λ=0.01)

**Generated**: 2026-07-21
**Data**: Run 7 checkpoints (`runs/phic_psi_{model}/20260720_*/history.csv`)
**Context**: Rebuttal to `run7_verification_rebuttal.md` Section A.2, which relied on endpoint-only (epoch 0, epoch 79) summaries. Full trajectories reveal dynamics the two-point summary missed.

---

## coa_phase (φc)

### Full trajectory — validation

| Epoch | poc_a | poc_b | tcn | cnn_attention |
|-------|-------|-------|-----|---------------|
| 0 | 0.3786 ⚠ | 1.9493 | 0.5863 | 0.9745 |
| 5 | 2.7083 ⚠ | 1.5108 | 2.4009 ⚠ | 1.1415 |
| 10 | 0.8218 | 1.0279 | 0.9496 | 0.9633 |
| 15 | 1.2953 | 0.7341 | 0.5873 | 0.9526 |
| 20 | 0.2803 ⚠ | 0.7512 | 0.5218 | 0.8642 |
| 25 | 0.4791 ⚠ | 0.7386 | 0.4407 ⚠ | 0.7441 |
| 30 | 0.4519 ⚠ | 0.2908 ⚠ | 0.4766 ⚠ | 0.7096 |
| 35 | 0.7166 | 0.8117 | 0.6247 | 0.6738 |
| 40 | 0.7635 | 0.7039 | 0.6525 | 0.6485 |
| 45 | 0.5304 | 0.8218 | 0.7383 | 0.6501 |
| 50 | 0.4107 ⚠ | 0.8464 | 0.3499 ⚠ | 0.6448 |
| 55 | 0.4831 ⚠ | 0.8673 | 0.3331 ⚠ | 0.6432 |
| 60 | 0.3892 ⚠ | 0.8618 | 0.3538 ⚠ | 0.6431 |
| 65 | 0.7214 | 0.8636 | 0.3627 ⚠ | 0.6428 |
| 70 | 0.7179 | 0.8606 | 0.3440 ⚠ | 0.6427 |
| 75 | 0.7540 | 0.8558 | 0.3562 ⚠ | 0.6424 |
| **79** | **0.6922** | **0.8509** | **0.3397 ⚠** | **0.6423** |

⚠ = outside healthy range [0.5, 2.0]

### Summary statistics

| Model | Start (ep 0) | End (ep 79) | Min | Max | Late trend (ep 40–79) | Epochs < 0.5 in last 40 |
|-------|-------------|------------|-----|-----|----------------------|--------------------------|
| poc_a | 0.38 | 0.69 | 0.28 | 3.86 | −0.0018/ep | 4/40 (10%) |
| poc_b | 1.95 | 0.85 | 0.24 | 3.72 | +0.0037/ep | 0/40 (0%) ✅ |
| tcn | 0.59 | **0.34** | 0.26 | 5.64 | **−0.0078/ep** | **32/40 (80%)** ❌ |
| cnn_attention | 0.97 | 0.64 | 0.27 | 1.59 | −0.0002/ep | 0/40 (0%) ✅ |

### Verdict

- **poc_b, cnn_attention**: Stable, healthy. Flat late-epoch trend, never below 0.5 in second half. ✅
- **poc_a**: Mostly healthy — 4/40 late epochs dip below 0.5, trend is very slightly negative. Acceptable. ✅
- **tcn**: **Not stabilized.** 80% of late epochs below threshold, still declining at −0.008/ep. Cannot be used as evidence for degeneracy until λ is re-tuned for this architecture. ❌

---

## polarization_angle (ψ)

### Full trajectory — validation

| Epoch | poc_a | poc_b | tcn | cnn_attention |
|-------|-------|-------|-----|---------------|
| 0 | 2.3031 ⚠ | 1.4924 | 1.3543 | 2.0307 ⚠ |
| 5 | 0.6033 | 1.1646 | 1.0716 | 1.3562 |
| 10 | 1.0652 | 2.3630 ⚠ | 1.7038 | 1.1677 |
| 15 | 1.7234 | 0.3943 ⚠ | 0.3853 ⚠ | 1.1150 |
| 20 | 0.9800 | 0.4983 ⚠ | 0.4140 ⚠ | 0.9688 |
| 25 | 1.0160 | 0.5114 | 0.3890 ⚠ | 0.8432 |
| 30 | 0.3299 ⚠ | 0.3826 ⚠ | 0.1861 ⚠ | 0.7684 |
| 35 | 0.6759 | 0.4846 ⚠ | 0.0819 ⚠ | 0.7107 |
| 40 | 0.4371 ⚠ | 0.6425 | 0.0725 ⚠ | 0.6484 |
| 45 | 0.4809 ⚠ | 0.6638 | 0.1288 ⚠ | 0.6319 |
| 50 | 0.5019 | 0.7031 | 0.4483 ⚠ | 0.6264 |
| 55 | 0.3834 ⚠ | 0.6657 | 0.5850 | 0.6261 |
| 60 | 0.5393 | 0.6700 | 0.6586 | 0.6239 |
| 65 | 0.6022 | 0.6692 | 0.6342 | 0.6233 |
| 70 | 0.6260 | 0.6662 | 0.6922 | 0.6233 |
| 75 | 0.4321 ⚠ | 0.6683 | 0.6087 | 0.6224 |
| **79** | **0.4411 ⚠** | **0.6676** | **0.6242** | **0.6221** |

⚠ = outside healthy range [0.5, 2.0]

### Summary statistics

| Model | Start (ep 0) | End (ep 79) | Min | Max | Late trend (ep 40–79) | Epochs < 0.5 in last 40 |
|-------|-------------|------------|-----|-----|----------------------|--------------------------|
| poc_a | 2.30 | **0.44** | 0.25 | 2.56 | +0.0001/ep | **30/40 (75%)** ❌ |
| poc_b | 1.49 | 0.67 | 0.33 | 2.36 | +0.0006/ep | 0/40 (0%) ✅ |
| tcn | 1.35 | 0.62 | 0.07 | 2.63 | **+0.0138/ep** | 12/40 (30%) |
| cnn_attention | 2.03 | 0.62 | 0.43 | 2.03 | −0.0007/ep | 0/40 (0%) ✅ |

### Verdict

- **poc_b, cnn_attention**: Stable, healthy. Flat late-epoch trend, never below 0.5 in second half. ✅
- **poc_a**: **Systematically below 0.5** for 75% of late epochs. Trend is flat (+0.0001/ep), so it's stable — but at the wrong value. λ may need tuning for this head/model combination, or there's an additional force pulling |v| below 1.0 in this specific configuration. ⚠️
- **tcn**: **Recovered strongly.** Crashed to 0.07 at epoch 40, then the magnitude penalty pulled it to 0.62 by epoch 60 where it stabilized (+0.014/ep, sharply positive). This is the magnitude penalty *working as designed* — the late-epoch value is healthy and stable. The two-point summary (1.35→0.62) completely obscures this recovery. ✅ (post-recovery)

---

## Key findings the endpoint-only summary missed

### 1. tcn coa_phase is not "borderline" — it's still declining

The rebuttal characterized tcn coa_phase at 0.34 as "borderline" and a "λ-tuning issue." The full trajectory confirms this but more strongly: 32 of the last 40 epochs are below 0.5, and the late-epoch trend is −0.008/ep with no sign of plateau. This model has **not converged** in |v|-space for coa_phase and should not be cited as evidence for or against the degeneracy until λ is tuned.

### 2. poc_a polarization_angle is stable but at the wrong value

At 0.44 final, with 30/40 late epochs below 0.5 and a flat trend (+0.0001/ep), this head has settled into a stable equilibrium — just not at |v| ≈ 1.0. The magnitude penalty at λ=0.01 appears insufficient to overcome whatever is pulling |v| below unity in this specific configuration. This is a quantitative tuning issue, not a framework failure, but it means this model/head doesn't provide a clean test of the degeneracy either.

### 3. tcn polarization_angle recovery is a success story

The epoch 40 value was 0.07 — catastrophic collapse. By epoch 60 it recovered to 0.66 and stabilized. This is direct evidence that the magnitude penalty works: it took ~40 epochs to accumulate enough gradient signal to overcome the initial |v| collapse, but once it did, it corrected the problem and maintained stability. The endpoint-only summary (epoch 0: 1.35 → epoch 79: 0.62) reads as "mild decline" — the reality is "catastrophic collapse → successful recovery."

### 4. Only 2 of 4 models are clean on both heads

| Model | coa_phase | pol_angle | Fully healthy? |
|-------|-----------|-----------|----------------|
| poc_a | ✅ (mostly) | ❌ (stable but low) | No |
| poc_b | ✅ | ✅ | **Yes** ✅ |
| tcn | ❌ (still declining) | ✅ (recovered) | No |
| cnn_attention | ✅ | ✅ | **Yes** ✅ |

---

## Impact on the degeneracy argument

The two models that are genuinely healthy in |v|-space (poc_b, cnn_attention) **still show the same flat circular loss** as the two with std_ratio caveats. The core finding — circular loss stays at ~1.0 for all 80 epochs — holds across the board.

However, the rebuttal's claim of "3/4 models healthy" (treating tcn coa_phase 0.34 as "borderline" and poc_a pol_angle 0.44 as "near the healthy range") doesn't hold up against the full trajectory. The honest summary is:

- **2/4 models** (poc_b, cnn_attention) have clean, stable std_ratios on both heads → these provide a clean degeneracy test
- **poc_a** has one head systematically low → λ may need tuning, partially confounded
- **tcn** has one head still declining → not converged, cannot be used as evidence

The degeneracy conclusion survives this correction — the two clean models show the same flat circular loss — but the evidence base is narrower than the rebuttal presented.

---

## Recommendation

1. **Re-tune λ for tcn** (try 0.05–0.10 for coa_phase specifically) before using tcn as supporting evidence.
2. **Re-tune λ for poc_a polarization_angle** — the flat-below-0.5 pattern suggests λ=0.01 is slightly too weak for this specific head.
3. **Proceed with poc_b and cnn_attention as the clean degeneracy test** — both are fully healthy in |v|-space and both show flat circular loss at ~1.0.
4. After λ-tuning for tcn and poc_a, re-check whether the circular loss remains flat — if it does even with healthy |v| across all four models, that strengthens the case considerably.