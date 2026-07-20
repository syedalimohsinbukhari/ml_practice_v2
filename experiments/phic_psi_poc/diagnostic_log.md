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

| Head                   | mean\|Δ\| | Changes? |
|------------------------|:---------:|:--------:|
| mchirp                 |   0.81    |   YES    |
| merger_time            |   0.15    |   YES    |
| snr                    |   0.72    |   YES    |
| inclination            |   0.46    |   YES    |
| **coa_phase**          | **0.00**  |  **NO**  |
| **polarization_angle** | **0.00**  |  **NO**  |

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

| Head               | Init est_logit_mag | Trained est_logit_mag |     Saturated?     |
|--------------------|:------------------:|:---------------------:|:------------------:|
| coa_phase          |        0.48        |         0.51          | No (threshold > 3) |
| polarization_angle |        0.52        |         0.53          |         No         |
| mchirp (control)   |        0.37        |         0.28          |         No         |

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

| Hypothesis                                  | Status        | Evidence                                         |
|---------------------------------------------|---------------|--------------------------------------------------|
| Data pipeline bug (labels collapsed)        | **Ruled out** | Check 1, two runs                                |
| Loss wiring bug (Huber still active)        | **Fixed**     | Check 2 Run 2 confirmed                          |
| Tanh saturation                             | **Ruled out** | Check 5: est_logit_mag ≈ 0.5                     |
| PERIODIC encoding broken                    | **Ruled out** | Inclination uses same encoding, gets gradient    |
| Disconnected computation graph              | **Ruled out** | combo log_vars get gradient, other heads change  |
| **Attenuated gradient through combo chain** | **ACTIVE**    | Check 4: coa_phase/psi Δ=0.00, all others change |

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

1. **Check 4 weight-name filter:** Fixed in Run 3 — now uses layer-based
   lookup via `trainer.base.get_layer()`, same pattern as Check 5.

2. **Check 3 stale:** Reads old training CSVs. Superseded — root cause
   confirmed independently via Checks 6 (forward-pass dump) and 7
   (saturation at init). Not a blocking issue.

---

## Files Tracking This Investigation

| File                        | Purpose                                              |
|-----------------------------|------------------------------------------------------|
| `diagnostic_checks.py`      | Current diagnostic script (5 checks, 6th pending)    |
| `diagnostic_output/*.log`   | Console output from each diagnostic run              |
| `diagnostic_output/*.png`   | Plots (true labels, logvar trajectories, combo loss) |
| `diagnostic_output/*.csv`   | True label statistics                                |
| `analyse_predictions.py`    | Cross-model prediction distribution analysis         |
| `NOTES.md`                  | Design decisions, run log, overall plan              |
| `plan_iota_conditioning.md` | Design plan for ι-conditioning (on hold)             |

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

| Evidence                                | Explanation                                       |
|-----------------------------------------|---------------------------------------------------|
| Check 4: kernel Δ=2.46 but grad=0       | Adam momentum coasting, not gradient-driven       |
| Check 4: pred Δ=0 despite weight change | tanh'(sat)=0, output frozen regardless of weight  |
| Check 5: est_logit≈0.5 "ok"             | Estimate wrong; actual trunk features much larger |
| Check 6: raw outputs = ±1               | Direct proof of saturation                        |
| Check 3: std_ratio=1.414                | √2 = norm of saturated (±1,±1) output             |
| Inclination gets gradient               | Random init happened to not saturate this head    |

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

| Hypothesis                        | Status        | Evidence                                         |
|-----------------------------------|---------------|--------------------------------------------------|
| Data pipeline bug                 | Ruled out     | Check 1 ×3                                       |
| Loss wiring bug                   | Fixed         | Check 2 confirmed ×2                             |
| Tanh saturation                   | **CONFIRMED** | Check 6: raw outputs ±1.0, Check 3: std_ratio=√2 |
| PERIODIC encoding broken          | Ruled out     | Inclination trains (different init luck)         |
| Disconnected graph                | Ruled out     | Gradient reaches post-tanh output (0.308)        |
| Attenuated gradient through combo | Superseded    | Gradient is healthy until tanh kills it          |

---

---

## Run 4 — 2026-07-18 (Check 7 only)

**Script version:** Check 7 added (early-training saturation timing).
**Log:** `diagnostic_output/diagnostic_checks_20260718_<timestamp>.log`

### Check 7: SATURATION AT INIT — confirmed

```
Step 0 (before any training)
  Sample 0: z_phic_raw = [-1.000000, -1.000000]  norm=1.4142  SATURATED
  Sample 1: z_phic_raw = [-1.000000, -1.000000]  norm=1.4142  SATURATED
  Sample 2: z_phic_raw = [-1.000000, -1.000000]  norm=1.4142  SATURATED

  Any |value| > 0.99 across all 8 samples:
    coa_phase:          SATURATED
    polarization_angle: SATURATED
```

Saturation is immediate — before any training step, the random-init weights
already push pre-activation logits past tanh's flat region. The model is
born dead.

**Conclusion:** Fix is `activation: linear` only. No gradient clipping needed
(no spike to protect against). The init variance on PERIODIC head output
layers is simply too large for tanh.

### All hypotheses resolved

| Hypothesis               | Status                | Evidence                                           |
|--------------------------|-----------------------|----------------------------------------------------|
| Data pipeline bug        | Ruled out             | Check 1 ×4                                         |
| Loss wiring bug          | Fixed                 | Check 2 confirmed ×3                               |
| Tanh saturation          | **CONFIRMED AT INIT** | Check 6: raw outputs ±1; Check 7: saturated step 0 |
| PERIODIC encoding broken | Ruled out             | Inclination trains when not saturated              |
| Disconnected graph       | Ruled out             | Gradient reaches post-tanh output (0.308)          |
| Attenuated gradient      | Superseded            | Gradient healthy until tanh kills it               |
| Init-variance vs drift   | **INIT**              | Check 7: saturated before first gradient step      |

### Bug closed. Ready for fix phase.

- [x] Root cause: tanh saturation at random init for PERIODIC heads
- [x] Mechanism: init variance on coa_phase/pol_angle output layers too large
- [x] Fix determined: `activation="tanh"` → `"linear"` in heads_spec.py
- [x] Apply fix (2026-07-18): `activation="tanh"` → `"linear"` in heads_spec.py
      for inclination, coa_phase, polarization_angle
- [ ] Retrain all models (7 configs)
- [ ] Re-run full diagnostics to test degeneracy hypothesis

### Fix applied

Changed in `src/gwml/heads_spec.py`:
```
inclination:         activation="tanh" → "linear"
coa_phase:           activation="tanh" → "linear"
polarization_angle:  activation="tanh" → "linear"
```

`normalize_unit` in the combo loss pipeline already projects outputs onto the
unit circle. `tanh` was redundant and introduced a saturation bottleneck.
With linear activation, the model learns unconstrained directions and
normalize_unit handles the normalization.

---

## Run 5 — 2026-07-18 (post-fix retraining: all 7 configs)

**Retrained:** All 7 configs (poc_a, poc_b, TCN, ResNet1D, CNN Baseline,
CNN Attention, InceptionTime) with `activation="linear"` on PERIODIC heads.

**Analysis:** 2026-07-20. See `tanh_to_linear_postmortem.md` for full details.

### Result: Tanh saturation fixed; new normalize_unit pathology found

The tanh→linear fix successfully removed the saturation bottleneck. Evidence:

- **Pre-fix:** std_ratio_coa_phase = **1.414** (exactly √2) — tanh output pinned
  at (±1, ±1) corners. Frozen across all 80 epochs.
- **Post-fix:** std_ratio_coa_phase = **103.7** at epoch 79 — raw predictions
  have varied magnitudes, confirming linear activation produces non-saturated
  outputs.

**BUT: the PERIODIC heads still do not learn.** Final validation MAE:

| Head                | MAE range       | Null expectation   | Status             |
|---------------------|-----------------|--------------------|--------------------|
| coa_phase           | 1.57–1.58 rad   | π/2 = 1.571 rad    | Mode collapsed     |
| polarization_angle  | 0.78–0.79 rad   | π/4 = 0.785 rad    | Mode collapsed     |
| inclination         | 1.54–1.59 rad   | π/2 = 1.571 rad    | Mode collapsed     |

Pre-fix vs post-fix delta for coa_phase MAE: **≤ 0.017 rad** across all
architectures. Zero meaningful improvement.

Non-PERIODIC heads train normally (mchirp R² = 0.96, snr R² = 0.79 for TCN).
The split is **PERIODIC heads (use normalize_unit) → dead. Scalar heads
(don't use normalize_unit) → healthy.** This is about mechanism, not physics.

### New pathology: normalize_unit gradient attenuation

With tanh removed, a new problem emerged. The `normalize_unit` layer computes
`u = v/|v|`, and the gradient chain is:

```
dL/dv = dL/du · du/dv = dL/du · (I − uuᵀ) / |v|
```

The gradient magnitude is **∝ sin(Δθ) / |v|**. When |v| drifts away from 1,
training fails:

| Head          | |v| (std_ratio, epoch 79) | Gradient scaling      | Trend      |
|---------------|--------------------------|-----------------------|------------|
| coa_phase     | ~104× unit circle        | **÷104 → vanished**   | Growing    |
| pol_angle     | ~13× unit circle         | **÷13 → attenuated**  | Stable     |
| inclination   | ~0.07× unit circle       | **×14 → unstable**    | Shrinking   |

The isotropic loss `1−cosΔθ` is computed *after* `normalize_unit`, so it is
completely blind to |v|. Nothing in the loss gives the optimizer any reason
to keep |v| near 1. Once |v| drifts (ordinary weight-update noise, no
restoring force), normalize_unit's backward pass either crushes or explodes
the gradient, and the drift accelerates.

### How this bug was introduced

The original Huber-on-vector loss (`‖v_pred−v_true‖²`) was magnitude-sensitive
by construction — its minimum is exactly at |v| = 1. It implicitly regularized
|v|. Swapping to the isotropic `1−cosΔθ` loss was the correct fix for
directional anisotropy, but it silently discarded that magnitude-regularizing
side effect. The fix needs a partner: an explicit magnitude penalty to replace
the implicit one that was removed.

### Inclination failure — a separate problem

**Inclination also mode-collapses** (MAE = π/2), but it does NOT use the same
loss path as φc and ψ. Code trace (`inclination_loss_trace.md`):

- Inclination's loss is **Huber** (`losses.py:138-147`), computed on the raw
  `Dense(2, linear)` output — no `normalize_unit`, no circular loss.
- `_build_combo_vectors` never touches `y_pred["inclination"]`; it only reads
  `y_true["inclination"][:, 1]` to get `cos(ι_true)` for curriculum weighting.
- Huber is magnitude-sensitive (`‖v−v_true‖²` penalizes |v| ≠ 1), so
  inclination's failure cannot be explained by the `normalize_unit` |v| drift.

Inclination's failure is a **separate, unresolved issue** — possibly weak
gradient signal from strain → inclination, or the model converging to the
dataset-mean (sin ι, cos ι) as the optimal constant predictor under Huber
loss when the true mapping is unlearnable. It does NOT provide evidence for
or against the `normalize_unit` hypothesis. It is tracked here as an open
question, not as a diagnostic for the φc/ψ problem.

### Combo heads (poc_b) also dead — but this doesn't test the degeneracy

circular_loss_combo_A ≈ 0.994 and combo_B ≈ 0.992 at epoch 79, essentially
flat at the random baseline (1.0) for all 80 epochs.

**However:** the combo heads are built by normalizing z_φc and z_ψ *first*,
then applying `complex_mul`. The |v| pathology enters upstream — z_φc and
z_ψ already have crushed/exploded gradients from normalize_unit BEFORE the
combo transform ever sees them. A flat circular loss on the combos tells you
the gradient never reached the upstream heads. It tells you nothing about
whether the combinations are physically well-constrained.

**This result does NOT refute the degeneracy hypothesis.** It's fully explained
by the unresolved |v| bug.

### Cross-architecture consistency — the split is by loss path

All five architectures + poc_a/poc_b show the same pattern. The division is:

| Loss path                              | Heads                                     | Status       |
|----------------------------------------|-------------------------------------------|--------------|
| Circular (1−cosΔθ via normalize_unit)  | coa_phase, polarization_angle             | All dead     |
| Huber (standard regression)            | inclination, mchirp, merger_time, snr     | Mixed*       |
| vMF                                    | sky_position                              | Healthy      |

\* mchirp, merger_time, snr are healthy; inclination is dead — cause unknown.

This rules out architecture-specific explanations for φc/ψ failure. The
normalize_unit → circular loss path is the common mechanism for the two
heads we care about. Inclination's failure is a separate open question
(Huber loss, no normalize_unit) — see `inclination_loss_trace.md`.

### Hypothesis status (updated post-fix, revised after review)

| Hypothesis                          | Status        | Evidence                                              |
|-------------------------------------|---------------|-------------------------------------------------------|
| Data pipeline bug                   | Ruled out     | Check 1 ×4                                            |
| Loss wiring bug                     | Fixed         | Check 2 confirmed ×3                                  |
| Tanh saturation on PERIODIC heads   | **FIXED**     | std_ratio ≠ √2; linear activation verified            |
| PERIODIC encoding broken            | Ruled out     | Same encoding for all angular heads                   |
| Disconnected computation graph      | Ruled out     | Gradient reaches all heads post-fix                   |
| normalize_unit gradient attenuation | **ACTIVE**    | \|v\| diverges → gradient ∝ 1/\|v\|; inclination fails |
| φc-ψ degeneracy confirmed           | **UNTESTED**  | Cannot test until \|v\| is stabilized                 |
| Combo heads break degeneracy        | **UNTESTED**  | Combos inherit upstream \|v\| pathology               |
| ι-conditioning needed               | **PREMATURE** | Fix the mechanism before evaluating the physics       |

### Open bugs

1. **normalize_unit gradient attenuation:** |v| drifts without bound because
   the isotropic loss is blind to magnitude. Fix: add explicit magnitude
   penalty `λ·(|v|−1)²` alongside the angular loss. This is the standard
   companion to cosine-distance losses in metric learning literature.

2. **Check 4 weight-name filter:** Fixed in Run 3. ✅

3. **Check 3 stale:** Reads old training CSVs. Superseded. ✅

### Next steps (revised)

- [x] Root cause: tanh saturation at random init for PERIODIC heads
- [x] Fix: `activation="tanh"` → `"linear"` in heads_spec.py
- [x] Retrain all 7 configs (2026-07-18)
- [x] Re-run analysis (2026-07-20) → **mode collapse persists; new pathology found**
- [ ] **Implement magnitude penalty** `λ·(|v_raw|−1)²` in the loss pipeline
- [ ] **Retrain poc_a + poc_b on TCN only** (minimal cost to test the fix)
- [ ] **Track std_ratio** over full training — success = stabilizes near 1.0
- [ ] **Only then evaluate physics:** if std_ratio is healthy and PERIODIC heads
      still don't learn, that's clean evidence about the degeneracy
- [ ] Do NOT conclude the degeneracy is fundamental, pivot to ι-conditioning,
      or explore alternative representations until the |v| fix is tested

---

## Run 6 — 2026-07-20 (magnitude penalty implemented; inclination path traced; pre-flight checks)

### Magnitude penalty: `λ·(|v_raw|−1)²`

Implemented in `trainer.py` (`_magnitude_penalty` method, lines 305-332):

```python
def _magnitude_penalty(self, *raw_vectors) -> tf.Tensor:
    if self._mag_lambda == 0.0:
        return tf.constant(0.0)
    total = tf.constant(0.0)
    for v in raw_vectors:
        norm = tf.sqrt(tf.reduce_sum(tf.square(v), axis=-1) + 1e-8)
        total = total + tf.reduce_mean(tf.square(norm - 1.0))
    return self._mag_lambda * total
```

Wired into both `_baseline_total_loss` and `_poc_total_loss`. Captures
`y_pred["coa_phase"]` and `y_pred["polarization_angle"]` BEFORE
`_build_combo_vectors` normalizes them.

**Config:** `magnitude_penalty_lambda: 0.01` in all 4 target configs.
**Runner:** `run_magnitude_fix.py` — chains train→plot→eval for
poc_a, poc_b, TCN, CNN Attention only.

### Pre-flight checks — five ways this retrain could fail silently

Before committing GPU time, five specific failure modes were traced through
the code. Each is a way the magnitude fix could produce another null result
that looks like "degeneracy confirmed" but is actually another engineering
artifact — exactly the class of mistake that produced the premature conclusion
in the first version of the postmortem.

**Check 1: Which tensor does the penalty land on?**

The penalty must apply to the *raw* (pre-`normalize_unit`) predictions
`z_phic_raw` and `z_psi_raw`. If it accidentally landed on the combo vectors
`combo_A_pred`/`combo_B_pred`, it would be a silent no-op: `complex_mul` of
two unit-norm vectors is itself unit-modulus, so `(|combo| − 1)²` would be
identically zero regardless of the upstream |v| state. The penalty would
appear to be in the loss, the code would look correct, and the std_ratio
would keep drifting — a perfect silent failure.

Verified: `z_phic_raw = y_pred["coa_phase"]` and `z_psi_raw = y_pred["polarization_angle"]`
are captured on `trainer.py:397-398` (baseline) and `trainer.py:437-438`
(poc), BEFORE the `_build_combo_vectors` call that normalizes them. The
normalized vectors `z_phic_pred`/`z_psi_pred` go to the angular loss; the
raw vectors go to the magnitude penalty. Separate paths, correct placement.

**Check 2: Is the penalty outside the uncertainty-weighting wrapper?**

The angular loss is wrapped in the standard uncertainty-weighting pattern:
`exp(−log_var) · angular_loss + log_var`. If the magnitude penalty were
accidentally nested inside this wrapper — `exp(−log_var) · (angular_loss +
λ·(|v|−1)²) + log_var` — then a swing toward very negative `log_var` (high
certainty) would multiply the penalty by a large factor, coupling numerical
stability to the learned uncertainty parameter. Different flavor, same
instability class we've spent three rounds tracking down.

Verified: the penalty is added as a separate term after the
uncertainty-weighted angular loss block — `total = total +
self._magnitude_penalty(raw)` — not inside `exp(−s) * loss_mean + s`.
`λ` is fixed and never multiplied by any learned parameter.

**Check 3: Is the penalty scaled by the curriculum weight w(ι)?**

`w(ι)` exists to downweight the angular loss on the poorly-constrained combo
at face-on inclinations where the degeneracy is strongest — it encodes a
physics prior about where real degeneracy-breaking information exists. The
magnitude penalty has nothing to do with this prior; it's a numerical
stability term that should apply uniformly to every sample. If it were
scaled by `w(ι)`, face-on samples (w ≈ 0) would receive no magnitude
regularization — exactly the regime where |v| drift is most likely to
develop unchecked, since those samples already have suppressed angular
gradients. You'd be removing the guardrail at the most dangerous part of the
road.

Verified: in `_poc_total_loss`, the `w(ι)` weighting is applied exclusively
to `loss_A_per_sample` and `loss_B_per_sample` via the per-sample masks
`w_A`/`w_B`. The magnitude penalty line `total = total +
self._magnitude_penalty(z_phic_raw, z_psi_raw)` sits after the entire
curriculum-weighting block. It applies uniformly to every sample in the
batch, independent of inclination.

**Check 4: Is the sign-dependent combo label actually per-sample?**

Step 1.1 found that the well-constrained combo flips depending on
`sign(cos ι)`: combo_B is well-constrained for cos ι > 0, combo_A for
cos ι < 0. The trainer is supposed to handle this dynamically — assigning
the curriculum weight to the *poorly*-constrained combo on a per-sample
basis. If this logic were broken and a single combo were hardcoded as
"well-constrained" globally, roughly half the training population would be
training against the wrong target — the model would be penalized for
predicting the combo that is actually well-constrained at that inclination.
That would produce a flat circular loss regardless of whether the magnitude
fix works, and would look exactly like another null result.

Verified: `cos_iota = y_true["inclination"][:, 1]` is a per-sample (N,)
tensor extracted from the batch (`trainer.py:293`). `pos_mask = tf.cast(cos_iota
>= 0.0, tf.float32)` creates a per-sample boolean mask. Config has
`well_constrained_combo: combo_B`, so the `else` branch at line 468 runs:

```python
w_B = pos_mask + neg_mask * w_iota    # line 470
w_A = neg_mask + pos_mask * w_iota    # line 471
```

Walk-through against Step 1.1 findings:

- **cos ι ≥ 0** (pos_mask=1, neg_mask=0): w_B = 1.0 (full weight — combo_B is
  well-constrained ✓), w_A = w_iota (curriculum weight — combo_A is
  poorly-constrained ✓).
- **cos ι < 0** (pos_mask=0, neg_mask=1): w_B = w_iota (curriculum weight —
  combo_B is now poorly-constrained ✓), w_A = 1.0 (full weight — combo_A is
  now well-constrained ✓).

The per-sample masks correctly swap which combo receives the curriculum
weight based on `sign(cos ι)` for every sample in the batch. The
curriculum always lands on the poorly-constrained direction and the
well-constrained direction always receives full weight. Correct.

**Check 5: Is the sky_position anomaly a SumDiffTrainer-specific bug?**

An earlier assessment noted poc_a/poc_b had 4× worse sky_position angular
error than plain TCN (12.9° vs 3.2°), and hypothesized a `transforms.json`
save/load issue specific to the `SumDiffTrainer` pipeline. If real, this
could indicate a broader transform-handling bug that might silently affect
φc/ψ target scaling too — the kind of thing you want to know about before
launching a retrain, not after.

Verified: `ra_mae_deg ≈ 90°` for sky_position across ALL 12 runs examined
(plain MultiHeadTrainer and SumDiffTrainer, 4-head and 7-head configs, old
and new). Sky position is uniformly not being learned by any model — it's
a pre-existing issue unrelated to the φc/ψ work. The old 3.2° vs 12.9°
comparison was likely using component-space MAE rather than angular degrees.
No `transforms.json` bug, no SumDiffTrainer-specific pathology. Not blocking.

### Inclination code-path trace

Full trace at `inclination_loss_trace.md`. Key finding:

- Inclination output: `Dense(2, activation="linear")` (`heads.py:99-103`)
- Loss: `keras.losses.Huber(delta=1.0)` (`losses.py:138-147`)
- `_build_combo_vectors` reads `y_true["inclination"][:, 1]` (cos ι for
  curriculum weight) but never touches `y_pred["inclination"]`
- → Inclination does NOT go through `normalize_unit`. Huber is
  magnitude-sensitive (`‖v−v_true‖²` penalizes |v| ≠ 1), so inclination's
  failure (MAE = π/2) cannot be explained by the |v|-drift hypothesis.
  It is a **separate, unresolved issue** — tracked as an open question,
  not as evidence for or against the normalize_unit theory.

The earlier version of this log (and the first draft of the postmortem)
incorrectly used inclination's failure as evidence for a "shared mechanism
bug" — the argument that since inclination also fails, the root cause must
be the mechanism all PERIODIC heads share. This was wrong: they don't share
the same mechanism. Inclination uses Huber; φc and ψ use normalize_unit →
circular loss. The two failures may have different root causes. The argument
is retired.

### Hypothesis status (updated)

| Hypothesis                          | Status         | Evidence                                             |
|-------------------------------------|----------------|------------------------------------------------------|
| normalize_unit gradient attenuation | **FIX PENDING**| Magnitude penalty implemented, not yet retrained     |
| Inclination failure = normalize_unit| **REFUTED**    | Code trace: Huber loss, no normalize_unit in path    |
| Inclination failure — root cause    | **OPEN**       | Separate investigation needed                       |
| φc-ψ degeneracy                     | **UNTESTED**   | Cannot test until |v| stabilized + retrained          |
| Combo heads break degeneracy        | **UNTESTED**   | Same — blocked by |v| bug                            |

### Next steps (updated)

- [x] Root cause: tanh saturation at random init for PERIODIC heads
- [x] Fix: `activation="tanh"` → `"linear"` in heads_spec.py
- [x] Retrain all 7 configs (2026-07-18)
- [x] Re-run analysis (2026-07-20) → **mode collapse persists; normalize_unit pathology found**
- [x] Implement magnitude penalty `λ·(|v_raw|−1)²` in loss pipeline (2026-07-20)
- [x] Code-trace inclination loss path → **does not use normalize_unit; separate issue**
- [x] Pre-flight 1: penalty lands on pre-normalize vectors ✓
- [x] Pre-flight 2: penalty outside uncertainty wrapper ✓
- [x] Pre-flight 3: penalty NOT scaled by w(ι) ✓
- [x] Pre-flight 4: sign-dependent combo is per-sample (literal source verified) ✓
- [x] Pre-flight 5: sky_position anomaly not SumDiffTrainer-specific ✓
- [ ] **Retrain poc_a + poc_b on TCN only** with magnitude penalty
- [ ] **Track std_ratio every epoch** — success = stabilizes 0.5–2.0, not diverging
- [ ] **Track combo circular loss trajectory** — success = departing from ~1.0 random baseline
- [ ] **Rerun Check 4 (prediction-perturbation) at early epochs** — verify real gradient-driven
      weight movement resumes for coa_phase/pol_angle (mean|Δ| comparable to healthy heads)
- [ ] **Only then evaluate physics** — see interpretation guide below

### Retrain interpretation guide

Three tracking signals to watch *during* the run, not just at epoch 79:

1. **`std_ratio` for `coa_phase` and `polarization_angle`, every epoch.**
   Success = it stabilizes in roughly 0.5–2.0 and stays there — not diverging
   toward 100 or collapsing toward 0.01. If it's still trending toward either
   extreme by epoch 79, `λ` needs tuning: try an order of magnitude up and
   down before concluding the approach failed. A single `λ` choice is not a
   verdict.

2. **`combo_A` / `combo_B` circular loss trajectory.** Success = visibly
   departing from the ~1.0 random baseline and decreasing over training, not
   just a small dip that re-flattens.

3. **Check 4 (prediction-perturbation test) in early epochs.** The
   name-matching bug was fixed in Run 3 — use the layer-based lookup to
   directly verify that `coa_phase` and `polarization_angle` now show
   `mean|Δ|` in the same rough order of magnitude as healthy heads
   (`mchirp`, `snr`), not the `0.00` seen before the fix. This is the most
   direct confirmation that the gradient path is actually working, independent
   of what aggregate metrics eventually show.

How to read the outcome:

- **All three signals healthy, but final R²/MAE still poor:** this is now a
  genuinely clean test. Do NOT jump to "degeneracy confirmed." First, bin
  the validation set by SNR tercile — check whether the top tercile shows
  any real signal (free, data you already have). Then, if still null, run
  a non-degenerate control target before trusting a null result as physics.

- **Any of the three signals still unhealthy:** that's a `λ`-tuning problem,
  not a reason to abandon the approach or escalate to ι-conditioning.
  Sweep `λ` by an order of magnitude in each direction and re-evaluate.

- **sky_position:** parked as a separate, non-blocking item. It sits at the
  random-guess baseline across every config (same signature as everything
  else that turned out to be a bug in this investigation), but it doesn't
  touch anything about how to read the φc/ψ retrain. Worth a trace later.

---

*Last updated: 2026-07-20 (revised after review + code trace + pre-flight checks)*
*Summary: Tanh fix necessary but not sufficient. normalize_unit gradient
pathology (|v| drift → 1/|v| gradient scaling) blocks φc and ψ circular loss.
Magnitude penalty implemented; five silent-failure modes traced and ruled out.
Inclination fails through a separate mechanism (Huber loss, no normalize_unit)
— tracked as an independent open question. Degeneracy hypothesis remains
untested pending |v| stabilization. Ready for retrain.*
