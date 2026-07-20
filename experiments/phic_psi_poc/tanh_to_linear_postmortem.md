# Tanh → Linear Fix: Post-Mortem Analysis

**Date:** 2026-07-20 (revised 2026-07-20 after review)
**Branch:** `poc/phic-psi-degeneracy`
**Question:** Did the tanh→linear activation fix for PERIODIC heads resolve the φc/ψ mode collapse?

---

## Executive Summary

**No — but for a different reason than initially concluded.** The tanh→linear fix
successfully removed the tanh-saturation bottleneck (confirmed by std_ratio
changing from exactly √2 to other values). However, removing tanh unmasked a
**second bug**: the `normalize_unit` layer creates a gradient pathology where
the raw prediction magnitude |v| drifts without bound, attenuating or exploding
gradients through the 1/|v| factor in normalize_unit's backward pass.

**This is an engineering bug, not a physics limit.** The cleanest evidence:
**inclination also fails identically** (MAE = π/2, mode-collapsed), and
inclination is NOT part of the φc-ψ degenerate pair. There's no physical reason
ι should be unrecoverable from strain. But inclination uses the same
`normalize_unit` → circular loss mechanism as φc and ψ — and its |v| collapses
to 0.069× the unit circle, producing unstable gradients. The failure pattern
splits perfectly along mechanism lines (every PERIODIC head broken, every
non-periodic head healthy), not along physics lines.

**The φc-ψ degeneracy hypothesis remains UNTESTED.** The combo heads (poc_b)
also don't learn, but they inherit the same |v| pathology from their upstream
normalized inputs — their failure is fully explained by the unresolved
engineering bug. The degeneracy may be real, may be surmountable, or may be
irrelevant — we can't distinguish any of those possibilities until |v| is
stabilized.

**The fix is standard and low-risk:** add an explicit magnitude penalty
`λ·(|v_raw| − 1)²` alongside the angular loss. This is the standard companion
to cosine-distance losses in metric learning — it replaces the implicit
magnitude regularization that was accidentally provided by the old Huber loss
and discarded when we correctly switched to the isotropic 1−cosΔθ loss.

---

## 1. Pre-Fix vs Post-Fix: Final Validation Metrics

### 1.1 coa_phase (φc) — MAE (radians)

Null expectation for constant prediction vs uniform on [0, 2π): **π/2 ≈ 1.571 rad**

| Architecture   | Pre-Fix (tanh) | Post-Fix (linear) | Δ        |
|----------------|---------------|-------------------|----------|
| TCN            | 1.572         | 1.579             | +0.007   |
| ResNet1D       | 1.564         | 1.578             | +0.014   |
| CNN Baseline   | 1.577         | 1.578             | +0.001   |
| CNN Attention  | —             | 1.577             | —        |
| InceptionTime  | 1.556         | 1.573             | +0.017   |

**Verdict:** Zero improvement. All values hover at the null expectation π/2.

### 1.2 polarization_angle (ψ) — MAE (radians)

Null expectation for constant prediction vs uniform on [0, π): **π/4 ≈ 0.785 rad**

| Architecture   | Pre-Fix (tanh) | Post-Fix (linear) | Δ        |
|----------------|---------------|-------------------|----------|
| TCN            | 0.787         | 0.786             | −0.001   |
| ResNet1D       | 0.787         | 0.786             | −0.001   |
| CNN Baseline   | 0.787         | 0.793             | +0.006   |
| CNN Attention  | —             | 0.790             | —        |
| InceptionTime  | 0.792         | 0.780             | −0.012   |

**Verdict:** Zero improvement. All values hover at the null expectation π/4.

### 1.3 inclination (ι) — THE KEY CONTROL

Null expectation: **π/2 ≈ 1.571 rad**

| Architecture   | Pre-Fix (tanh) | Post-Fix (linear) |
|----------------|---------------|-------------------|
| TCN            | 1.556         | 1.544             |
| ResNet1D       | —             | 1.588             |
| CNN Baseline   | —             | 1.570             |
| CNN Attention  | —             | 1.559             |
| InceptionTime  | —             | 1.571             |
| poc_a          | —             | 1.560             |
| poc_b          | 1.528         | 1.573             |

**This is the single most important row in the table.** Inclination is NOT part
of the φc-ψ degenerate pair. It has no sum/diff structure, no combo transform
applies to it, it's just an ordinary PERIODIC head predicting an angle from
strain. There is no physical reason it should be unrecoverable.

Yet it fails identically to φc and ψ — MAE pinned at the null expectation,
mode-collapsed across all architectures.

**This is strong evidence that the root cause is a shared mechanism bug, not a
physics limit.** Inclination shares exactly one mechanism with φc and ψ:
`normalize_unit` → circular loss. Every head that uses this mechanism is
broken; every head that doesn't (mchirp, merger_time, snr) is healthy.

### 1.4 Scalar (non-PERIODIC) heads — healthy

| Architecture   | mchirp R² (post) | snr R² (post) | merger_time R² (post) |
|----------------|-----------------|---------------|----------------------|
| TCN            | 0.960           | 0.792         | 0.920                |
| poc_a          | 0.958           | 0.786         | 0.921                |
| poc_b          | 0.966           | 0.787         | 0.928                |

The split is clean: **PERIODIC heads (normalize_unit) → dead. Scalar heads
(no normalize_unit) → healthy.** This pattern is about mechanism, not physics.

---

## 2. What the Tanh→Linear Fix Actually Changed

### 2.1 std_ratio — the diagnostic that caught the next bug

The std_ratio = std(raw_prediction) / std(normalized_prediction) measures how far
the raw Dense outputs are from the unit circle. A healthy model should have
std_ratio ≈ 1.0 (raw predictions naturally lie near the unit circle).

| Head          | Pre-Fix std_ratio | Post-Fix std_ratio (epoch 79) | Trend            |
|---------------|-------------------|-------------------------------|------------------|
| coa_phase     | **1.414** (√2)    | **103.7**                     | Growing (21→104) |
| pol_angle     | **1.414** (√2)    | **13.2**                      | Stable ~13       |
| inclination   | —                 | **0.069**                     | Shrinking        |

**Interpretation:**

- **Pre-fix:** std_ratio = √2 means every prediction was exactly (±1, ±1) —
  the tanh output was hard-saturated at the corners. Zero gradient through tanh.
  This was correctly diagnosed and fixed.

- **Post-fix coa_phase:** With tanh removed, |v| grows without bound (21→104
  over training). normalize_unit divides by |v|, so the gradient is attenuated
  by **~100×**. The model receives essentially no training signal for φc.

- **Post-fix inclination:** |v| collapses to ~0.07× the unit circle.
  normalize_unit divides by a tiny number, amplifying the gradient by
  **~14×**. Updates are noisy and unstable — the model cannot converge.

**Both directions are the same underlying problem:** the isotropic loss
`1−cosΔθ` is computed *after* `normalize_unit`, so it is completely blind to
|v|. Nothing in the loss gives the optimizer any reason to keep |v| near 1.
Once |v| drifts even slightly (ordinary weight-update noise, no restoring force),
normalize_unit's backward pass — which multiplies by 1/|v| — either crushes or
explodes the gradient, and the drift accelerates.

### 2.2 How this bug was introduced

The original Huber-on-vector loss (`‖v_pred − v_true‖²`) was magnitude-sensitive
by construction — its minimum is exactly at v_pred = v_true, which has |v| = 1.
It implicitly regularized |v| even though nobody designed it to. Swapping to the
isotropic `1−cosΔθ` loss was the correct fix for the directional anisotropy
problem, but it silently discarded that magnitude-regularizing side effect.

The fix needs a partner: an explicit magnitude penalty to replace the implicit
one that was removed.

### 2.3 R² trajectory (TCN, validation)

| Epoch | coa_phase R²  | pol_angle R² | inclination R² |
|-------|---------------|--------------|----------------|
| 0     | −615          | −1,662       | −0.47          |
| 10    | −5,572        | −12,047      | −0.10          |
| 40    | −14,662       | −459         | −0.01          |
| **79**| **−20,897**   | **−180**     | **−0.002**     |

- **coa_phase:** R² diverges to −20,897 — the model gets progressively WORSE
  as |v| grows and kills the gradient entirely.
- **pol_angle:** R² improves slightly (−1,662 → −180) but stays catastrophic.
  |v| ≈ 13 is less extreme than 104, so some weak signal gets through, but not
  enough for convergence.
- **inclination:** R² → 0 (predicting the mean). |v| ≈ 0.07 creates unstable
  gradient amplification that prevents any coherent learning.

---

## 3. The Combo Heads: Why poc_b Doesn't Test the Degeneracy Yet

The PoC's central hypothesis was that combo_A = φc+2ψ and combo_B = φc−2ψ
would be learnable even if individual φc and ψ are degenerate.

### 3.1 Combo circular loss (poc_b, validation)

| Epoch | combo_A circ_loss | combo_B circ_loss |
|-------|-------------------|-------------------|
| 65    | 0.9945            | 0.9921            |
| 70    | 0.9944            | 0.9921            |
| 75    | 0.9943            | 0.9923            |
| **79**| **0.9943**        | **0.9923**        |

Random baseline = 1.0. The combo heads don't learn.

### 3.2 Why this doesn't refute the degeneracy hypothesis

The combo heads are built by normalizing z_φc and z_ψ *first*, then applying
`complex_mul` (per Appendix A.2 of the implementation plan). The |v| pathology
enters upstream — z_φc and z_ψ already have crushed/exploded gradients from
normalize_unit BEFORE the combo transform ever sees them. `complex_mul` then
operates on these already-degraded signals.

A flat circular loss on the combos tells you the gradient never meaningfully
reached the heads that feed them. It tells you nothing about whether the
combinations are physically well-constrained. **This result is fully explained
by the unresolved |v| bug** — no stronger degeneracy hypothesis is needed.

### 3.3 What WOULD test the degeneracy

Once |v| is stabilized (std_ratio ≈ 1 for all PERIODIC heads), THEN:

- If combo heads learn (circ_loss drops below random baseline) while individual
  φc/ψ still don't → degeneracy confirmed, combo approach works.
- If combo heads also don't learn → degeneracy is stronger than the 1.2× ratio
  suggested, or the signal genuinely isn't extractable.
- If individual φc/ψ DO learn → the degeneracy was never the problem.

We cannot distinguish among these until the engineering is clean.

---

## 4. Cross-Architecture Consistency

All five architectures (TCN, ResNet1D, CNN Baseline, CNN Attention,
InceptionTime) show the same pattern:

| Head                | All architectures? | Value                | Uses normalize_unit? |
|---------------------|-------------------|----------------------|----------------------|
| mchirp              | ✓ Learned         | R² = 0.63–0.97       | No                   |
| merger_time         | ✓ Learned         | R² = 0.24–0.93       | No                   |
| snr                 | ✓ Learned         | R² = 0.41–0.79       | No                   |
| sky_position        | ✓ Learned         | Varies               | No                   |
| coa_phase           | ✗ Dead            | MAE = π/2 (all)      | **Yes**              |
| polarization_angle  | ✗ Dead            | MAE = π/4 (all)      | **Yes**              |
| inclination         | ✗ Dead            | MAE = π/2 (all)      | **Yes**              |

The split is not "degenerate parameters vs non-degenerate parameters."
The split is **"heads that use normalize_unit vs heads that don't."**

---

## 5. Updated Hypothesis Status

| Hypothesis                          | Status           | Evidence                                              |
|-------------------------------------|------------------|-------------------------------------------------------|
| Data pipeline bug                   | Ruled out        | Check 1 ×4                                            |
| Loss wiring bug                     | Fixed            | Check 2 confirmed ×3                                  |
| Tanh saturation on PERIODIC heads   | **FIXED**        | std_ratio ≠ √2; linear activation verified            |
| PERIODIC encoding broken            | Ruled out        | Same encoding for all angular heads                   |
| Disconnected computation graph      | Ruled out        | Gradient reaches all heads post-fix                   |
| normalize_unit gradient attenuation | **ACTIVE**       | \|v\| diverges → gradient ∝ 1/\|v\|; inclination fails |
| φc-ψ degeneracy confirmed           | **UNTESTED**     | Cannot test until \|v\| is stabilized                 |
| Combo heads break degeneracy        | **UNTESTED**     | Combos inherit upstream \|v\| pathology               |
| ι-conditioning needed               | **PREMATURE**    | Fix the mechanism before evaluating the physics       |

---

## 6. Corrected Next Steps

### 6.1 Immediate: fix the normalize_unit magnitude pathology

The fix is small, standard, and has a known-good track record in metric
learning / embedding literature:

```
L = w(ι) · (1 − cosΔθ)  +  λ · (|v_raw| − 1)²
```

This keeps everything already validated:
- Isotropic angular loss (1−cosΔθ) — fixes directional anisotropy ✓
- Sign-conditioned combo labels — from Step 1.1 ✓
- Linear activation on PERIODIC heads — prevents tanh saturation ✓

And adds the missing piece: an explicit restoring force that gives the
optimizer a reason to keep |v| near 1, preventing the 1/|v| factor in
normalize_unit's backward pass from crushing or exploding the angular gradient.

`λ` is a standard hyperparameter — small enough not to dominate the angular
signal, large enough to stop the runaway drift. Start with λ = 0.01–0.1 and
tune based on whether std_ratio stabilizes near 1.

### 6.2 Test strategy

1. **Implement the magnitude penalty** in the SumDiffTrainer / loss pipeline.
2. **Retrain only poc_a and poc_b on TCN** (keeps cost down; if the fix
   works, it will work here first).
3. **Track std_ratio over the full training run** — the success criterion is
   std_ratio stabilizing near 1.0 (± some tolerance), not diverging to 104
   or collapsing to 0.07.
4. **Only then interpret R²/MAE.** A poor R² on coa_phase *after* |v| is
   well-behaved is clean evidence about the physics. A poor R² while |v| is
   exploding is just another engineering artifact.
5. **If PERIODIC heads start learning:** expand to CNN Attention and the
   remaining architectures; then evaluate the degeneracy hypothesis properly.
6. **If PERIODIC heads still don't learn with healthy |v|:** then — and only
   then — consider ι-conditioning, alternative representations, or the
   possibility that the degeneracy is fundamental.

### 6.3 Things NOT to do yet

- Do NOT conclude the degeneracy is fundamental — it's untested.
- Do NOT pivot to ι-conditioning — fix the known mechanism bug first.
- Do NOT explore alternative data representations — one bug at a time.
- Do NOT write up the degeneracy as a thesis finding — the evidence isn't clean.

---

## 7. Data Quality Notes

### 7.1 Training vs validation MAE discrepancy (TCN post-fix)

The training history records training MAE for coa_phase inflating from 20→78
while validation MAE stays at ~1.57. **This is likely the same |v| bug.** If
the training MAE is computed on raw (pre-normalize) predictions while
validation MAE uses the wrapped angle, then as |v| grows, the raw MAE inflates
while the angular MAE stays pinned at π/2. Not a separate metric discrepancy —
another symptom of |v| runaway.

### 7.2 sky_position degradation in poc_a/poc_b

Pre-fix analysis noted poc_a/poc_b sky_position angular MAE (12.9°) was 4×
worse than plain TCN (3.2°), despite all using the TCN backbone. This anomaly
persists post-fix and may indicate a transforms.json save/load issue in the
SumDiffTrainer pipeline. Tracked separately — not related to the |v| pathology.

---

## Appendix: Run Directory Reference

### Pre-fix (tanh activation)
| Config              | Run ID            |
|---------------------|-------------------|
| phic_psi_tcn        | 20260718_005033   |
| phic_psi_resnet1d   | 20260718_004319   |
| phic_psi_cnn_attn   | 20260717_235646   |
| phic_psi_cnn_base   | 20260718_000305   |
| phic_psi_inception  | 20260718_000749   |
| phic_psi_poc_a      | 20260717_233435   |
| phic_psi_poc_b      | 20260718_002136   |

### Post-fix (linear activation)
| Config              | Run ID            |
|---------------------|-------------------|
| phic_psi_tcn        | 20260718_153301   |
| phic_psi_resnet1d   | 20260718_152543   |
| phic_psi_cnn_attn   | 20260718_143902   |
| phic_psi_cnn_base   | 20260718_144514   |
| phic_psi_inception  | 20260718_145002   |
| phic_psi_poc_a      | 20260718_141645   |
| phic_psi_poc_b      | 20260718_150353   |

---

*Generated from `runs/phic_psi_*/` validation metrics and training histories.*
*See also: `NOTES.md`, `results.md`, `diagnostic_log.md`*
