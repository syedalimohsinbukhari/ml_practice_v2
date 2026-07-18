# Diagnostic Log — φc/ψ Mode Collapse Investigation

Chronological record of every diagnostic run, hypothesis tested, and outcome.
Failures and wrong turns documented alongside successes — all thesis material.

See `NOTES.md` for design decisions and overall plan.
See `diagnostic_checks.py` for the current diagnostic script.

---

## Run 1 — 2026-07-18 12:50

**Script version:** Checks 1-4 only (Check 5 not yet written).
**Log:** `diagnostic_output/diagnostic_checks_20260718_125027.log`

**Working hypothesis at time of run:** The mode collapse is either a data
pipeline bug (true labels collapsed), a loss-wiring bug (Huber still active),
or a degeneracy problem (combo transform not helping).

### Check 1: True label distribution

**Result: CLEAN.** circ_r ≈ 0.005 for all parameters, max bin ≈ 5% (uniform
baseline). The mode collapse is a training phenomenon, not a data pipeline bug.

**Hypothesis ruled out:** "True labels are collapsed in the data pipeline."

### Check 2: Loss function verification

**Result: BUG FOUND.** `_patch_log_vars` removed coa_phase/pol_angle from
`self.log_vars` but left stale huber registrations in `self.head_loss` for
poc_b. These huber functions were never called (`_other_heads_loss` correctly
skips `_SUMDIFF_SOURCE_HEADS`), so this was cosmetic — no double-gradient.

**Fix applied:** `self.head_loss.pop(h, None)` added alongside
`self.log_vars.pop(h, None)` in `_patch_log_vars`.

**Hypothesis confirmed:** Kimi's warning about individual losses was valid.
Not functionally impactful (never called), but a spec violation.

### Check 3: Log-var trajectory

**Result: FROZEN.** Circular loss at 1.0 (random expectation for 1−cosΔθ with
random unit vectors) from epoch 0 to 79. Never budged. coa_phase R² = −1.988
frozen across all 80 epochs in poc_a (baseline — no combo machinery at all).
This proved the degeneracy hypothesis hadn't actually been tested — the heads
never trained.

**Key insight:** The same freeze appears in both poc_a (plain Huber on
individual heads) AND poc_b (circular loss on combos). This means it's NOT
specific to the combo-transform design — something more fundamental is wrong.

**Hypothesis formed:** This looks like tanh-saturation in the PERIODIC head
output layers. If |pre-activation logit| > 3, tanh' ≈ 0 and gradient vanishes.

### Check 4: Gradient routing

**Result: INCONCLUSIVE (script bug).** 0 coa_phase/pol_angle weights found
because Keras returns flat names (`kernel`, `bias` — no layer prefix) and the
substring filter `"coa_phase" in "kernel"` is always False.

combo_A/combo_B log_vars DO get gradient (norm ≈ 0.097, 0.095), confirming
the loss graph is wired at the log_var level.

**Bug identified:** Name-matching filter broken. Needs layer-based lookup.

**Decision:** Add Check 5 (tanh saturation) before re-running.

---

## Run 2 — 2026-07-18 13:17

**Script version:** Checks 1-5 (Check 5 added, Check 4 with prediction
perturbation test, logging hardened).
**Log:** `diagnostic_output/diagnostic_checks_20260718_131719.log`

### Check 2: Confirmed fixed

poc_b shows `head_loss` without coa_phase/pol_angle:
```
head_loss keys: ['mchirp', 'merger_time', 'snr', 'sky_position', 'inclination']
✓ individual coa_phase / pol_angle losses correctly removed
```

Bug closed. No stale huber registrations remain.

### Check 4: Prediction perturbation — SMOKING GUN

Despite the weight-name filter still being broken, the prediction-perturbation
test directly measures whether each head's output changes after a gradient step:

| Head | mean\|Δ\| | Changes? |
|------|:---------:|:--------:|
| mchirp | 0.81 | YES |
| merger_time | 0.15 | YES |
| snr | 0.72 | YES |
| inclination | 0.46 | YES |
| **coa_phase** | **0.00** | **NO** |
| **polarization_angle** | **0.00** | **NO** |

This is decisive: the gradient reaches all other heads on the same trunk
(including inclination, which uses the same PERIODIC sin/cos encoding), but
does NOT reach coa_phase/polarization_angle specifically.

**Key control:** Inclination uses the same PERIODIC encoding, same tanh
activation, same Dense layer structure, same trunk — but gets healthy gradient
(mean|Δ| = 0.46). This rules out "the PERIODIC encoding itself is broken."

**What's unique to coa_phase/pol_angle:** They participate in the circular
loss + combo transform chain: normalize_unit → complex_mul →
1−cosΔθ → log_var weighting. Inclination does not.

**Weight-name bug persists:** All 121+114 weight names are flat (`kernel`,
`bias`, `gamma`, `beta`). Name-matching can't work. Needs layer lookup
(as Check 5 does successfully).

### Check 5: Pre-tanh logit saturation — RULED OUT

| Head | Init est_logit_mag | Trained est_logit_mag | Saturated? |
|------|:---:|:---:|:---:|
| coa_phase | 0.48 | 0.51 | No (threshold > 3) |
| polarization_angle | 0.52 | 0.53 | No |
| mchirp (control) | 0.37 | 0.28 | No |

Kernel norms change ~3% over training (1.93 → 1.99 for coa_phase). Weights
ARE moving, just far too slowly to affect predictions in 80 epochs.

**Hypothesis ruled out:** Tanh saturation. The pre-activations are healthy.
My prior hypothesis was wrong.

**Revised hypothesis:** Severely attenuated gradient through the circular
loss → combo transform → head output chain. The gradient exists (combo
log_vars get ~0.1 norm) but is orders of magnitude too weak by the time it
reaches the head output layers.

### Check 3 note: STALE

Check 3 reads training-history CSVs from the old runs. Numbers are
byte-identical to Run 1. Does not reflect the Check 2 fix. Needs fresh
retraining to produce new data.

---

## Current State of Hypotheses

| Hypothesis | Status | Evidence |
|-----------|--------|----------|
| Data pipeline bug (labels collapsed) | **Ruled out** | Check 1, two runs |
| Loss wiring bug (Huber still active) | **Fixed** | Check 2 Run 2 confirmed |
| Tanh saturation | **Ruled out** | Check 5: est_logit_mag ≈ 0.5 |
| PERIODIC encoding broken | **Ruled out** | Inclination uses same encoding, gets gradient |
| Disconnected computation graph | **Ruled out** | combo log_vars get gradient, other heads change |
| **Attenuated gradient through combo chain** | **ACTIVE** | Check 4: coa_phase/psi Δ=0.00, all others change |

---

## Next Diagnostic: Gradient Chain Instrument (Check 6)

Measure gradient norm at each stage of the combo loss pipeline on a single
batch:
```
y_pred["coa_phase"] → normalize_unit → complex_mul → combo_A_pred
                    → 1−cosΔθ loss → log_var weighting → total_loss
```

Use inclination's gradient magnitude at equivalent stages as healthy baseline.
Goal: find exactly which operation attenuates the gradient.

---

## Open Bugs in Diagnostic Script

1. **Check 4 weight-name filter:** Uses substring matching on flat Keras names
   (`kernel`, `bias`). Needs layer-based lookup (same pattern as Check 5).
   Non-blocking — prediction-perturbation test works around it.

2. **Check 3 stale:** Reads old training CSVs. Cannot produce new data until
   models are retrained.

---

## Files Tracking This Investigation

| File | Purpose |
|------|---------|
| `diagnostic_checks.py` | Current diagnostic script (5 checks, 6th pending) |
| `diagnostic_output/*.log` | Console output from each diagnostic run |
| `diagnostic_output/*.png` | Plots (true labels, logvar trajectories, combo loss) |
| `diagnostic_output/*.csv` | True label statistics |
| `analyse_predictions.py` | Cross-model prediction distribution analysis |
| `NOTES.md` | Design decisions, run log, overall plan |
| `plan_iota_conditioning.md` | Design plan for ι-conditioning (on hold) |

---

---

## Run 3 — 2026-07-18 13:41

**Script version:** Checks 1-6 (Check 4 fixed, Check 6 added).
**Log:** `diagnostic_output/diagnostic_checks_20260718_134134.log`

### Check 4: NOW WORKING — but reveals paradox

Layer-based lookup finds the weights. But gradient norms show:
- coa_phase kernel: **0.000000** (ZERO gradient)
- coa_phase bias: 3.68e-10 (ZERO)
- pol_angle kernel/bias: both 0.00 (ZERO)

Yet weight deltas: kernel Δ = 2.46, bias Δ = 0.014 — both change!

**Paradox resolved (later, by Check 6):** Adam optimizer momentum carries weight movement
even after tanh saturation kills instantaneous gradient. Motion without learning.

Prediction perturbation unchanged: coa_phase/pol_angle = 0.00, all other heads change.

### Check 5: STILL SAYS "OK" — but is wrong

`est_logit_mag ≈ 0.5` flagged "ok" against the 3.0 threshold. This estimate uses
`kernel_norm × assumed_input_scale / sqrt(input_dim)` — it assumed trunk feature
magnitude ≈ 2.0. The actual trunk features are clearly much larger, pushing
pre-activation logits past saturation.

**Lesson:** Check 5's estimate is approximate and was wrong. Check 6's actual
measured forward pass overrides it.

### Check 6: SMOKING GUN — CONFIRMED TANH SATURATION

Forward-pass dump for first 3 samples at epoch 80:
```
Sample 0: z_phic_raw = [-1.000000, +1.000000]  norm=1.4142 (=sqrt(2))
Sample 1: z_phic_raw = [-1.000000, +1.000000]  norm=1.4142
Sample 2: z_phic_raw = [-1.000000, +1.000000]  norm=1.4142
```

Every sample produces exactly `[-1, +1]` for φc and `[+1, -1]` for ψ.
This is complete, hard tanh saturation — outputs pinned to corners where
tanh'(x) = 0 numerically.

**This explains the std_ratio ≈ 1.414 observed in Check 3**: that's √2, the
norm of a saturated tanh output at (±1, ±1).

Gradient chain from Check 6:
```
dL/d(combo_A_pred)     = 0.415  OK
dL/d(combo_B_pred)     = 0.352  OK
dL/d(z_phic_norm)      = 0.642  OK  
dL/d(z_psi_norm)       = 0.533  OK
dL/d(z_phic_raw)       = 0.308  OK  ← gradient at post-tanh output is healthy
dL/d(z_psi_raw)        = 0.176  OK
dL/d(inclination_raw)  = 0.694  OK  ← healthy baseline
```

The gradient with respect to the post-tanh *output* is fine (0.308). But
the chain rule requires multiplying by d(tanh)/d(pre-activation) ≈ 0 at
saturation. Healthy gradient into the output, zero gradient into the
weights that produced it — textbook dead-tanh.

### Resolution of all contradictions

| Evidence | Explanation |
|----------|-------------|
| Check 4: kernel Δ=2.46 but grad=0 | Adam momentum coasting, not gradient-driven |
| Check 4: pred Δ=0 despite weight change | tanh'(sat)=0, output frozen regardless of weight |
| Check 5: est_logit≈0.5 "ok" | Estimate wrong; actual trunk features much larger |
| Check 6: raw outputs = ±1 | Direct proof of saturation |
| Check 3: std_ratio=1.414 | √2 = norm of saturated (±1,±1) output |
| Inclination gets gradient | Random init happened to not saturate this head |

### Root cause confirmed

**Tanh saturation** on PERIODIC head output layers. The `tanh` activation
creates a saturation bottleneck. `normalize_unit` already projects outputs
onto the unit circle — tanh is redundant and harmful.

### Next steps

- [x] Check 6 complete — root cause confirmed
- [ ] Check 7: instrument early training steps to determine WHEN saturation
      occurs (init vs drift)
- [ ] Fix: change `activation="tanh"` → `"linear"` for PERIODIC heads
- [ ] Retrain all models
- [ ] Re-run full diagnostics to test degeneracy hypothesis

### Hypothesis status (updated)

| Hypothesis | Status | Evidence |
|-----------|--------|----------|
| Data pipeline bug | Ruled out | Check 1 ×3 |
| Loss wiring bug | Fixed | Check 2 confirmed ×2 |
| Tanh saturation | **CONFIRMED** | Check 6: raw outputs ±1.0, Check 3: std_ratio=√2 |
| PERIODIC encoding broken | Ruled out | Inclination trains (different init luck) |
| Disconnected graph | Ruled out | Gradient reaches post-tanh output (0.308) |
| Attenuated gradient through combo | Superseded | Gradient is healthy until tanh kills it |

---

*Last updated: 2026-07-18 14:00*
*Next: Check 7 (early-training saturation timing) + tanh→linear fix + retrain*
