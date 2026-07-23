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
- [x] Check 7: instrument early training steps to determine WHEN saturation
      occurs (init vs drift) — **completed in Run 4**
- [x] Fix: change `activation="tanh"` → `"linear"` for PERIODIC heads —
      **applied 2026-07-18**
- [x] Retrain all models — **completed in Run 5 (2026-07-18)**
- [x] Re-run full diagnostics to test degeneracy hypothesis —
      **completed 2026-07-20; result: normalize_unit pathology found**

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
- [x] Retrain all models (7 configs) — **completed in Run 5 (2026-07-18)**
- [x] Re-run full diagnostics to test degeneracy hypothesis —
      **completed 2026-07-20; result: normalize_unit pathology found**

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

| Hypothesis                          | Status        | Evidence                                               |
|-------------------------------------|---------------|--------------------------------------------------------|
| Data pipeline bug                   | Ruled out     | Check 1 ×4                                             |
| Loss wiring bug                     | Fixed         | Check 2 confirmed ×3                                   |
| Tanh saturation on PERIODIC heads   | **FIXED**     | std_ratio ≠ √2; linear activation verified             |
| PERIODIC encoding broken            | Ruled out     | Same encoding for all angular heads                    |
| Disconnected computation graph      | Ruled out     | Gradient reaches all heads post-fix                    |
| normalize_unit gradient attenuation | **ACTIVE**    | \|v\| diverges → gradient ∝ 1/\|v\|; inclination fails |
| φc-ψ degeneracy confirmed           | **UNTESTED**  | Cannot test until \|v\| is stabilized                  |
| Combo heads break degeneracy        | **UNTESTED**  | Combos inherit upstream \|v\| pathology                |
| ι-conditioning needed               | **PREMATURE** | Fix the mechanism before evaluating the physics        |

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
- [x] **Implement magnitude penalty** `λ·(|v_raw|−1)²` in the loss pipeline —
      **completed in Run 6 (2026-07-20)**
- [x] **Retrain poc_a + poc_b on TCN only** (minimal cost to test the fix) —
      done in Run 7 (2026-07-20), alongside plain TCN and CNN Attention
- [x] **Track std_ratio** over full training — success = stabilizes near 1.0 —
      done (Run 7, `std_ratio_trajectories.md`); 2/4 models fully healthy at
      λ=0.01, the other two carried forward and closed via the Run 9a/9b
      λ retune
- [x] **Only then evaluate physics:** if std_ratio is healthy and PERIODIC heads
      still don't learn, that's clean evidence about the degeneracy — done
      (Run 7 verification Sections A–E): the two models with clean |v|-space
      show zero learning
- [x] Do NOT conclude the degeneracy is fundamental, pivot to ι-conditioning,
      or explore alternative representations until the |v| fix is tested —
      **premature conclusion retired; see Run 6**

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

| Hypothesis                           | Status          | Evidence                                          |
|--------------------------------------|-----------------|---------------------------------------------------|
| normalize_unit gradient attenuation  | **FIX PENDING** | Magnitude penalty implemented, not yet retrained  |
| Inclination failure = normalize_unit | **REFUTED**     | Code trace: Huber loss, no normalize_unit in path |
| Inclination failure — root cause     | **OPEN**        | Separate investigation needed                     |
| φc-ψ degeneracy                      | **UNTESTED**    | Cannot test until                                 |v| stabilized + retrained          |
| Combo heads break degeneracy         | **UNTESTED**    | Same — blocked by                                 |v| bug                            |

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
- [x] **Retrain poc_a + poc_b on TCN only** with magnitude penalty — done in
      Run 7 (2026-07-20), plus plain TCN and CNN Attention for broader coverage
- [x] **Track std_ratio every epoch** — success = stabilizes 0.5–2.0, not diverging —
      done (Run 7, `std_ratio_trajectories.md`); 2/4 models fully healthy, the
      other two carried forward and closed via Run 9a/9b
- [x] **Track combo circular loss trajectory** — success = departing from ~1.0 random baseline —
      done (Run 7 Check 3); **outcome: did not depart** — flat at ~1.0 across
      all 80 epochs, every model (the central null-result finding)
- [x] **Rerun Check 4 (prediction-perturbation) at early epochs** — verify real gradient-driven
      weight movement resumes for coa_phase/pol_angle (mean|Δ| comparable to healthy heads) —
      done (Run 7 Check 4): confirmed, gradient reaches φc/ψ weights
- [x] **Only then evaluate physics** — see interpretation guide below — done
      (Run 7 verification Sections A–E, extended by Run 8/9)

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

## Run 7 — 2026-07-20 (magnitude penalty retrain: 4 models)

**Retrained:** poc_a, poc_b, TCN, CNN Attention with `magnitude_penalty_lambda: 0.01`
in all configs. Magnitude penalty `λ·(|v_raw|−1)²` active in both baseline and PoC
loss paths (verified by code trace in Run 6 pre-flight checks).

**Training completed:** 2026-07-20. Output: `runs/phic_psi_{poc_a,poc_b,cnn_attention,tcn}/20260720_*`

**Analysis:** `analyse_predictions.py` re-run 2026-07-20 23:43:04. Output:
`experiments/phic_psi_poc/analysis_output/analysis_report_20260720_234304.md`

**Deep diagnostics:** `diagnostic_checks.py` re-run 2026-07-21 00:03:31. Output:
`experiments/phic_psi_poc/diagnostic_output/diagnostic_checks_20260721_000331.log`

### Result: Magnitude penalty prevents |v| drift; periodic heads still don't learn

**The magnitude penalty successfully prevents |v| drift**, but the final std_ratio
state varies by model. Full trajectories extracted at
[`std_ratio_trajectories.md`](std_ratio_trajectories.md):

| Model         | coa_phase std_ratio (ep 79) |         Late-epoch trend          | pol_angle std_ratio (ep 79) |         Late-epoch trend          |
|---------------|:---------------------------:|:---------------------------------:|:---------------------------:|:---------------------------------:|
| poc_b         |            0.85             |       +0.0037/ep, stable ✅        |            0.67             |       +0.0006/ep, stable ✅        |
| cnn_attention |            0.64             |       −0.0002/ep, stable ✅        |            0.62             |       −0.0007/ep, stable ✅        |
| poc_a         |            0.69             |     −0.0018/ep, mostly stable     |            0.44             | +0.0001/ep, flat but below 0.5 ⚠️ |
| tcn           |            0.34             | **−0.0078/ep, still declining** ❌ |            0.62             | +0.0138/ep, recovered from 0.07 ✅ |

**Two of four models are fully healthy on both heads** (poc_b, cnn_attention).
poc_a pol_angle is stable at the wrong value (30/40 late epochs below 0.5) —
likely a λ-tuning issue for that specific head. tcn coa_phase has not converged
(32/40 late epochs below 0.5, still declining) — cannot be used as evidence until
λ is re-tuned. tcn pol_angle underwent a dramatic recovery (0.07 at epoch 40 →
0.62 by epoch 60), which is direct evidence the magnitude penalty mechanism works
— the endpoint-only summary completely obscured this.

**The PERIODIC heads still do not learn.** Final validation metrics from
`analyse_predictions.py` (2026-07-20):

| Head               | Run 7 circ_r range | Run 7 ang_MAE range | Null expectation |
|--------------------|--------------------|---------------------|------------------|
| coa_phase          | 0.43–0.99          | 1.54–1.61 rad       | π/2 = 1.571 rad  |
| polarization_angle | 0.17–0.99          | 0.78–0.80 rad       | π/4 = 0.785 rad  |
| inclination        | 0.37–0.80          | 1.53–1.59 rad       | π/2 = 1.571 rad  |

poc_b shows COLLAPSE (circ_r ≈ 0.99) on both coa_phase and pol_angle — more severe
mode collapse than the plain baseline poc_a. This is explained by the curriculum
design interacting with the degeneracy (see [`poc_b_config_diff.md`](poc_b_config_diff.md)):
the curriculum suppresses the poorly-constrained combo, funneling gradient through
a single underdetermined channel. When that channel carries no real angular
information, the model collapses to the single best constant — worse than poc_a's
two independent loss terms, which at least create competing pressures.

cnn_attention shows the lowest circ_r (0.43 coa_phase, 0.17 pol_angle) but this
is a feature-statistics artifact from learned attention pooling producing
higher-variance trunk features — not evidence of phase learning (see
[`cnn_attention_config_diff.md`](cnn_attention_config_diff.md)). The ang_MAE
remains at baseline (1.60 rad for coa_phase — slightly worse than poc_a's 1.57).
No hidden config difference exists to port to other models; the q_tokens branch
is present in the architecture but unused (no `per_head` config overrides).

**Scalar heads (mchirp, merger_time, snr) remain healthy** — mchirp R² ≈ 0.93–0.96,
merger_time R² ≈ 0.91–0.92, snr R² ≈ 0.76–0.79. The split is by loss path: heads
that use circular loss via normalize_unit (coa_phase, pol_angle) fail; heads that
use Huber or vMF (scalars, sky_position) train normally. Inclination uses Huber
and also fails — a separate unresolved issue (see `inclination_loss_trace.md`).

**Sky position is working** — true angular errors are 3.3–10.0° (cnn_attention at
3.3°, tcn at 4.5°). The kappa-derived proxy previously showing 77–87° was misleading.

### Diagnostic findings (Checks 1–7, 2026-07-21)

All seven checks from `diagnostic_checks.py` re-run on Run 7 checkpoints:

**Check 1 (true labels):** ✅ CLEAN. Labels well-spread, no data pipeline bug.

**Check 2 (loss wiring):** ✅ Correct. `head_loss` shows `huber_loss` for periodic
heads, but this is **vestigial** from MultiHeadTrainer.__init__ — the SumDiffTrainer
overrides `_total_loss` to use circular `1−cosΔθ` loss. Verified by code trace:
`train_step` (losses.py:273) → `self._total_loss()` → dispatched to
`_baseline_total_loss` or `_poc_total_loss` based on mode. The circular loss IS
the training objective.

**Check 3 (circular loss trajectories):** 🔴 **Circular loss NEVER decreases**
from random baseline (~1.0) across all 80 epochs for any model:

| Model | Loss metric                    | Epoch 0 | Epoch 79  | Delta  |
|-------|--------------------------------|---------|-----------|--------|
| poc_a | val_circular_loss_coa_phase    | 0.995   | **1.020** | +0.025 |
| poc_a | val_circular_loss_pol_angle    | 0.990   | **1.006** | +0.016 |
| poc_b | val_circular_loss_combo_A      | 0.999   | **0.999** | 0.000  |
| poc_b | val_circular_loss_combo_B      | 1.006   | **0.991** | -0.015 |
| tcn   | val_circular_loss_coa_phase    | 0.995   | **1.016** | +0.021 |
| tcn   | val_circular_loss_pol_angle    | 0.992   | **1.006** | +0.014 |

This is the most direct piece of evidence: the training objective literally does
not improve. The model's optimal strategy under `1−cosΔθ` loss when the input carries
no φc/ψ information is to output a constant prediction — expected loss ≈ 1.0 (random
alignment with any particular true angle), but lower variance than random outputs.

**Check 4 (gradient routing, poc_b):** ✅ CONFIRMED. Gradient reaches φc/ψ weights
via combo path. Prediction perturbation `mean|Δ|` for coa_phase = 1.52e-02 and
pol_angle = 1.37e-02 — same order of magnitude as healthy heads. The pre-fix
problem (0.00 gradient, no weight movement) is completely resolved.

**Check 5 (logit saturation):** ✅ Healthy. Post-training est_logit_mag ≈ 0.44–0.49,
well below saturation threshold. However, the diagnostic script labels this "Pre-tanh
logit saturation" — this label is misleading since PERIODIC heads have used
`activation="linear"` since Run 5. The check is measuring kernel norm magnitude, not
actual tanh saturation. The values are healthy regardless.

**Check 6 (gradient chain, poc_b):** ✅ Healthy throughout the full computation:
```
dL/d(combo_A_pred)                    0.425  OK
dL/d(z_phic_raw) [model output]      0.328  OK
dL/d(z_psi_raw)  [model output]      0.470  OK
dL/d(inclination_raw)                 0.704  OK
```
All gradient norms are healthy — no vanishing or exploding gradients.

**Check 7 (init-time raw output magnitude):** ⚠️ At init, |z_raw| ≈ 100–250. With
linear activation, this does NOT cause tanh saturation (there is no tanh). However,
`normalize_unit`'s backward pass scales gradients by `1/|z|` ≈ 0.004–0.01, slowing
early learning. The magnitude penalty gradually pulls |z| toward 1 (visible in
std_ratio trajectories), and by epoch ~40 the gradient scaling is healthy. The
diagnostic script's `SATURATED` label and `sin=+/-1 cos=+/-1` notation are misleading
with linear activation — these are just large numbers, not saturation.

**Key insight from diagnostics:** The circular loss stays flat at ~1.0 even during
epochs 40–79 when |z| ≈ 1 and gradient scaling is healthy (1.0×). The model has
~40 epochs with unattenuated gradients and still cannot reduce the loss. This is
NOT a gradient-scaling artifact — it's genuine inability to learn phase.

### Verification plan execution — Sections A–E (2026-07-21)

A verification plan ([`run7_verification_plan.md`](run7_verification_plan.md)) was
executed to systematically rule out confounds before accepting the degeneracy
conclusion. Five sections (A–E) were completed. Results are documented in detail at:

- [`std_ratio_trajectories.md`](std_ratio_trajectories.md) — A.2 full trajectories
- [`poc_b_config_diff.md`](poc_b_config_diff.md) — B: poc_b config diff and collapse mechanism
- [`cnn_attention_config_diff.md`](cnn_attention_config_diff.md) — C: cnn_attention outlier
- [`bootstrap_output/bootstrap_ang_mae_20260721_093533.md`](bootstrap_output/bootstrap_ang_mae_20260721_093533.md) — D: significance test
- [`snr_output/snr_stratification_20260721_094039.md`](snr_output/snr_stratification_20260721_094039.md) — E: SNR stratification

#### A — Gating checks

**A.1 (magnitude penalty applied?): ✅ CONFIRMED.** λ=0.01 in all four configs,
active in both `_baseline_total_loss` and `_poc_total_loss` via code trace
(trainer.py:306-333, 415, 496).

**A.2 (std_ratio trajectories): ⚠️ MIXED.** Full epoch-by-epoch trajectories
extracted for all four models. 2/4 models are fully healthy on both heads
(poc_b, cnn_attention — 0/40 late epochs below 0.5). poc_a pol_angle is stable
but systematically below 0.5 (0.44, 30/40 late epochs below threshold; flat
trend at +0.0001/ep). tcn coa_phase has not converged (0.34, 32/40 late epochs
below 0.5, still declining at −0.008/ep). The rebuttal's characterization
"3/4 models healthy" is incorrect — only 2/4 are clean on both heads.

One notable finding: tcn pol_angle underwent a dramatic recovery (0.07 at epoch
40 → 0.62 by epoch 60, trend +0.014/ep), which is direct evidence the magnitude
penalty works — the endpoint-only summary completely missed this.

**A.3 (prediction perturbation): ⚠️ QUALIFIED.** Check 4 confirms gradients reach
φc/ψ weights and predictions change. However, `rel_change` for coa_phase is
`1.61e-02` vs `1.80e-04` for mchirp — an **89× difference**, not "comparable"
as the initial rebuttal stated. A single-step perturbation snapshot cannot
distinguish directional learning from noisy oscillation. A multi-step trace
(consecutive gradient steps on the same batch) would be needed to discriminate.

#### B — poc_b config diff

**Finding: No config bug.** The only functional differences between poc_a and
poc_b configs are `loss.mode` (baseline → poc), `combo_log_var_clamp`,
`well_constrained_combo`, and `sign_dependent_combo` — all intended design
elements of the PoC. Everything else (data, model architecture, optimizer,
magnitude_penalty_lambda) is byte-identical.

poc_b's more severe collapse (circ_r ≈ 0.99 vs poc_a's 0.85/0.49) is a
**prediction of the degeneracy hypothesis**, not a config bug. The curriculum
`w(ι) = 1 − cos²(ι)` suppresses the poorly-constrained combo's loss, funneling
gradient through a single underdetermined channel. When neither φc nor ψ is
learnable, this produces worse collapse than poc_a's two independent loss terms,
which at least create competing pressures that spread predictions.

The ~5% mchirp MAE regression (0.977 → 1.024) is mild and R² is essentially
unchanged (0.9594 → 0.9569), consistent with mild dead-head noise in the shared
trunk rather than a separate bug.

#### C — cnn_attention config diff

**Finding: No hidden config difference to port.** The only difference vs tcn is
`trunk: cnn_attention` and a trivial `min_lr` difference (1e-7 vs 1e-6). All
loss, optim, and head settings are identical. The `q_tokens` extra-feature branch
exists in the architecture but is **unused** — no head has a `per_head.<name>.branch`
config override pointing to it.

cnn_attention's lower circ_r (0.43/0.17) is a feature-statistics artifact:
learned attention pooling produces higher-variance trunk features than TCN's fixed
GAP+GMP, causing more spread in predictions. But ang_MAE confirms the predictions
are not more correct — coa_phase MAE is 1.597 rad, slightly *worse* than poc_a's
1.570 rad. Spread ≠ signal. The val_loss scale difference (−1.53 vs −3.79) is a
log_var calibration artifact from different trunk feature statistics.

The higher-variance features slightly degrade scalar regression (mchirp R² 0.926
vs tcn's 0.963) while helping sky_position (3.3° vs 4.5°) — a mixed effect
consistent with a more expressive but noisier trunk readout.

#### D — Bootstrap significance test

N=10,000 bootstrap shuffles, N=5,000 validation samples. One-sided test: is
observed ang_MAE significantly *below* the null distribution?

| Model         |    coa_phase     |    pol_angle     |       inclination       |
|---------------|:----------------:|:----------------:|:-----------------------:|
| poc_a         | z=−0.20, p=0.579 | z=+0.46, p=0.324 |    z=+1.04, p=0.152     |
| poc_b         | z=−1.25, p=0.895 | z=−0.05, p=0.518 |    z=+0.07, p=0.464     |
| tcn           | z=−0.56, p=0.711 | z=−0.33, p=0.630 |    z=+1.21, p=0.114     |
| cnn_attention | z=−2.43, p=0.994 | z=+0.07, p=0.472 | **z=+3.17, p=0.0007 ★** |

**Result: 11/12 model×head combinations are indistinguishable from random.**
coa_phase and polarization_angle are universally non-significant across all four
models — no model learns either angle at a level distinguishable from guessing.

cnn_attention inclination is significant (z=+3.17, p=0.0007) with a small effect
(observed 1.534 vs null 1.572, Δ≈0.038 rad ≈ 2.2°). With 12 tests, a single
significant result at α=0.05 has ~46% probability under the global null
(Bonferroni-adjusted threshold: p < 0.0042). The effect does not survive
correction for multiple comparisons and warrants independent replication before
being interpreted as evidence of learning.

**Validation ordering check (2026-07-21):** A potential confound for any
bootstrap shuffle-null test is residual ordering in the validation set. If
injections were generated in parameter-grid blocks and the data not shuffled,
true inclination could correlate with row index. A model outputting predictions
that vary smoothly with row position (even without learning from strain) could
then beat the shuffle-null, since shuffling destroys the row-wise association
while the original evaluation preserves it. This was tested:

- **r(row_idx, ι)** = −0.032 (p≈0.02) — effectively zero. No parameter shows
  |r| > 0.035 with row index across all 10 parameters.
- **Window variance ratio** (window=100) = 0.991 — ratio of mean within-window
  variance to global variance. A value near 1.0 indicates i.i.d. samples; a
  value ≪ 1.0 would indicate parameter-grid blocking. The observed 0.991
  definitively rules out block structure.
- **Sanity check**: Even a model that perfectly predicted the mean ι of each
  100-row window (the maximum information extractable from ordering) would
  achieve ang_MAE ≈ 1.558 rad, giving Δ-from-null ≈ +0.013. cnn_attention's
  observed Δ of +0.038 is 3× larger — row structure alone cannot explain it.
- **Lag-1 |Δι|** = 2.12 rad (expected for i.i.d. U[0,π]: π/3 ≈ 1.05 rad).
  Consecutive rows are more different than random, not more similar — the
  opposite of blocking.

**The validation set is effectively i.i.d. with respect to row index.** The
bootstrap shuffle-null is valid. The cnn_attention inclination result is not
a row-ordering artifact. The remaining interpretation — population-level bias
vs genuine weak strain→ι mapping — is unchanged from the SNR-stratification
finding (improvement is uniform across SNR, not concentrated in loud events).

#### E — SNR stratification

Validation set split into terciles by SNR. If real angular information exists in
the strain, it should be most extractable at high SNR.

**coa_phase:** No model shows ang_MAE meaningfully below null in the high-SNR
tercile. tcn is the only monotonic improver (1.633→1.557→1.543 rad), but this is
the model with unresolved std_ratio decline (0.34, still trending down). Three
other models show no SNR trend or get *worse* at high SNR. Null expectation:
π/2 = 1.571 rad.

**polarization_angle:** All four models within ±0.006 rad of null at high SNR.
No model shows SNR-dependent improvement. Null: π/4 = 0.785 rad.

**inclination:** cnn_attention's significant bootstrap result shows no SNR
dependence — improvement is uniform across terciles (1.526, 1.548, 1.527),
not concentrated in loud events as a genuine physical signal would be. The
uniform improvement at all SNR levels is more consistent with learning a
population-level bias than extracting per-sample angular information.

**Overall:** No compelling SNR-dependent improvement for any head on any model.
The one model with monotonic improvement (tcn coa_phase) carries an unresolved
|v|-drift confound. This is consistent with the degeneracy hypothesis: if the
strain carries no φc/ψ information, the highest-SNR events should be no more
learnable than the lowest.

### Cross-architecture consistency

The pattern persists across all architectures: PERIODIC heads (coa_phase,
polarization_angle, inclination) fail; scalar heads + sky_position succeed. The
division is by loss path:

| Loss path                              | Heads                                     | Status       |
|----------------------------------------|-------------------------------------------|--------------|
| Circular (1−cosΔθ via normalize_unit)  | coa_phase, polarization_angle             | All dead     |
| Huber (standard regression)            | inclination, mchirp, merger_time, snr     | Mixed*       |
| vMF                                    | sky_position                              | Healthy      |

\* mchirp, merger_time, snr are healthy; inclination is dead — cause still unknown
(separate open question, see `inclination_loss_trace.md`).

### Key difference from Run 5: the fix works, the physics doesn't change

Run 5 failed because |v| drifted without bound (no magnitude penalty → normalize_unit
gradient crushed or exploded → PERIODIC heads dead). Run 7 fixes this for two of
four models (poc_b, cnn_attention have clean, stable std_ratios on both heads;
poc_a and tcn have one head each with residual |v| issues). The two clean models
still show flat circular loss at ~1.0 with healthy gradients. With the |v|
pathology resolved in these models, a flat circular loss is clean evidence that
φc/ψ cannot be learned from strain alone.

### Hypothesis status (2026-07-21, post-verification)

| Hypothesis                                | Status              | Evidence                                                                                                                                                                                                                     |
|-------------------------------------------|---------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Data pipeline bug                         | Ruled out           | Check 1 ×5                                                                                                                                                                                                                   |
| Loss wiring bug                           | Ruled out           | Check 2 confirmed ×4 + code trace                                                                                                                                                                                            |
| Tanh saturation on PERIODIC heads         | **FIXED (R5)**      | linear activation since Run 5                                                                                                                                                                                                |
| PERIODIC encoding broken                  | Ruled out           | Same encoding for all angular heads                                                                                                                                                                                          |
| Disconnected computation graph            | Ruled out           | Checks 4+6: gradients reach φc/ψ weights                                                                                                                                                                                     |
| normalize_unit gradient attenuation       | **FIXED (R7)**      | Magnitude penalty λ=0.01; std_ratios stabilized in 2/4 models                                                                                                                                                                |
| poc_b collapse = config bug               | Ruled out           | Config diff identical except intended PoC design elements; collapse is curriculum+degeneracy prediction                                                                                                                      |
| cnn_attention outlier = config difference | Ruled out           | Only difference is trunk architecture; q_tokens unused; lower circ_r is feature-variance artifact                                                                                                                            |
| ang_MAE distinguishable from random       | **REFUTED**         | Bootstrap: 11/12 model×head combinations non-significant (D)                                                                                                                                                                 |
| SNR-dependent learning                    | **Not observed**    | No model shows SNR-dependent ang_MAE improvement on either φc or ψ head (E)                                                                                                                                                  |
| φc-ψ degeneracy (no ι)                    | **STRONG EVIDENCE** | Circular loss flat at ~1.0 in models with clean                                                                                                                                                                              |v| + healthy gradients. Three independent tests (loss trajectory, bootstrap, SNR stratification) converge. |
| Combo heads break degeneracy              | **REFUTED**         | poc_b circular loss flat at ~1.0; COLLAPSE worse than baseline                                                                                                                                                               |
| Inclination failure                       | **OPEN**            | Separate mechanism (Huber, no normalize_unit). Bootstrap: cnn_attention significant (z=+3.17, p=0.0007) but SNR-independent — may be population-level bias, not per-sample learning. Does not survive Bonferroni correction. |

### Remaining open items

These were identified during verification and should be resolved before
ι-conditioning to close out remaining confounds:

1. **Systematic increase in circular loss** — The validation circular loss
   systematically drifts *upward* (e.g., poc_a coa_phase: 0.995→1.020, +0.025).
   "No signal" explains a flat trajectory but not a directional upward trend. A
   short ablation with λ=0 (magnitude penalty disabled) would isolate whether the
   penalty (or its interaction with log_var uncertainty weighting) is driving the
   creep. If the creep disappears at λ=0, this is a penalty-interaction artifact;
   if it persists, it's a genuinely different and more supportive data point.

2. **Multi-step perturbation trace** — Check 4's single-step snapshot shows
   coa_phase `rel_change = 1.61e-02` vs mchirp `1.80e-04` (89× larger). A
   multi-step trace (several consecutive gradient steps, same batch) would
   discriminate between coherent directional learning and noisy oscillation.
   Several steps of decreasing perturbation → real learning. Oscillation with
   no net drift → noise.

3. **tcn coa_phase λ retuning** — std_ratio still declining at −0.008/ep
   (32/40 late epochs below 0.5). Try λ=0.05–0.10 for this architecture before
   using tcn as supporting evidence.

4. **poc_a pol_angle λ check** — std_ratio stable but systematically below 0.5
   (0.44, 30/40 late epochs). λ=0.01 may be slightly too weak for this specific
   head. A small λ increase would test whether this is a tuning issue.

### Overall assessment

The degeneracy case is **stronger post-verification than before it**, but
qualifiers that the initial rebuttal elided remain:

- **Solid**: The circular loss never decreases from the random baseline, in any
  model, on any head, at any point in 80 epochs of training. This is robust to
  architecture choice, loss mode (baseline/poc), and |v|-space health. Three
  independent tests (loss trajectory, bootstrap significance, SNR stratification)
  converge on the same answer for coa_phase and polarization_angle. Two models
  (poc_b, cnn_attention) have fully healthy |v|-space and healthy gradients yet
  show zero learning — these are clean tests.

- **Qualified**: Two of four models have residual |v| issues (tcn coa_phase still
  declining; poc_a pol_angle systematically low). The systematic increase in
  circular loss (F.1/F.2) lacks an explanation. The single-step perturbation
  snapshot can't distinguish learning from noise. None of these undermine the
  core finding, but they prevent the verdict from being "airtight."

- **Not yet tested**: Whether tcn coa_phase, with a corrected λ, would also
  show flat circular loss (strengthening the case) or start to learn
  (weakening it). Whether the perturbation is directional or oscillatory
  (multi-step trace, now scheduled as part of Run 9 rather than standalone).

---

## Run 8 — 2026-07-21 (λ=0 ablation)

**Purpose:** isolate whether the systematic upward creep in validation
circular loss at λ=0.01 (item 1 above) was caused by the magnitude penalty
itself (or its interaction with log_var uncertainty weighting), rather than
being a genuinely different, more supportive data point for the degeneracy
conclusion.

**Method:** retrained poc_a and tcn (the same two TCN-trunk baseline configs
as the λ=0.01 Run 7 pair) with `magnitude_penalty_lambda: 0.0`, 80 epochs
each. Config: `config_lam0_ablation.yaml`, `config_lam0_ablation_tcn.yaml`.
Runner: `run_lam0_ablation.py`. Report: `lam0_ablation_output/lam0_ablation_report.md`.

**Result:**

| Model | Head               | λ=0 Δ (val circ loss) | λ=0.01 Δ | Verdict                              |
|-------|--------------------|-----------------------|----------|--------------------------------------|
| poc_a | coa_phase          | +0.0010               | +0.0249  | drift absent at λ=0                  |
| poc_a | polarization_angle | +0.0002               | +0.0163  | drift absent at λ=0                  |
| tcn   | coa_phase          | +0.0011               | +0.0204  | drift absent at λ=0                  |
| tcn   | polarization_angle | +0.0072               | +0.0136  | **persists — penalty NOT the cause** |

3 of 4 signals: the λ=0.01 creep vanishes at λ=0, confirming it was a
λ/log-var interaction artifact, not evidence of active anti-learning. 1 of 4
(tcn pol_angle) still drifts upward at λ=0, about half the λ=0.01 magnitude —
small, and flagged rather than folded into either the "resolved" or
"unresolved" bucket. std_ratio diverges hard without the penalty as expected
(21.8→83.0 for poc_a coa_phase), confirming the ablation is a genuine
off-state. Final val MAE at λ=0 lands at the same null values as every other
run (coa_phase≈1.579, pol_angle≈0.780–0.785 rad — both ≈ theoretical null).
Train circular loss decreases slightly at λ=0 while val stays flat — a small
train/val split (mild memorization of phase noise), not generalizable signal.

**Item F.1/F.2 closed.** Full writeup: `assessment_lam0_ablation_2026-07-22.md`.

### Hypothesis status update (2026-07-22)

| Hypothesis                                  | Status                              | Evidence                                                                                          |
|---------------------------------------------|-------------------------------------|---------------------------------------------------------------------------------------------------|
| Val-loss creep at λ=0.01 = penalty artifact | **CONFIRMED (3/4)**                 | Drift vanishes at λ=0 for poc_a coa_phase, poc_a pol_angle, tcn coa_phase                         |
| Val-loss creep = genuinely different signal | **NOT SUPPORTED (3/4), open (1/4)** | tcn pol_angle still drifts at λ=0 — small effect, unexplained, doesn't change headline conclusion |

---

## Run 9 — pre-registration + setup (2026-07-22), execution pending

**Purpose:** retune the magnitude penalty λ for the two remaining |v|-space
problems from Run 7 — tcn coa_phase (still declining, 0.34 at epoch 79,
−0.008/ep) and poc_a polarization_angle (stable but below 0.5, 0.44) — and
re-check whether the circular loss moves once |v|-space is clean.

**Before training anything**, a reviewer flagged that this investigation has
repeatedly had aggregate metrics look like one thing and mean another —
mode-collapse posing as R²=0.75 (Round 1), an endpoint-only std_ratio summary
that hid tcn pol_angle's mid-training crash-and-recovery (Run 7, A.2), and
now the λ=0.01 val-loss creep that turned out to be mostly a penalty artifact
(Run 8). The fix each time was procedural, not just "look more carefully" —
so the success/failure criterion for this retune was written down **before**
any λ=0.05/0.10 run exists, removing the option to reinterpret the result
after the fact in either direction. See `preregistration_lam_retune.md` for
the full reasoning; summary of the decision table:

1. **Step 0 (gate):** std_ratio healthy (<10% of last 40 epochs outside
   [0.5, 2.0], |trend| < 0.005/ep). Gate failure → **UNINTERPRETABLE**, not
   counted as null or counter-evidence — retune λ further instead.
2. **Step 1 (significance):** bootstrap shuffle-null test on val ang_MAE
   (same procedure as `bootstrap_ang_mae.py`), Bonferroni-corrected for the
   **2** pre-declared primary tests → significance threshold p < 0.025.
3. **Step 2 (effect size floor):** Δang_MAE ≥ 0.10 rad, chosen in advance to
   sit ~3× above the cnn_attention inclination effect (0.038 rad, judged not
   compelling in Run 7) and ~8× above the row-ordering artifact bound
   (0.013 rad) — big enough that clearing it can't be explained by either
   known failure mode.
4. **Step 3 (SNR-monotonicity):** improvement must be monotonic with SNR
   tercile (same logic as `snr_stratification.py`) **and** the high-SNR
   tercile's own effect must independently clear the 0.10 rad floor — guards
   against a population-level-bias signature (uniform-across-SNR
   improvement) being mistaken for genuine per-sample learning, exactly the
   pattern seen in the cnn_attention inclination result.

Only a result that clears gate + significance + effect size + SNR-monotonicity
counts as **COUNTER-EVIDENCE**. Anything else is a **NULL** result — a clean
additional data point for the degeneracy conclusion (with sub-flags for
"replicate independently" or "population-bias signature" as appropriate,
mirroring how the cnn_attention case was handled).

**Files created (training/execution pending on lab GPU machine):**

- Configs: `config_lam005_retune.yaml`, `config_lam005_retune_tcn.yaml`,
  `config_lam010_retune.yaml`, `config_lam010_retune_tcn.yaml` — same
  architecture/data/optimizer as Run 7/8, only `magnitude_penalty_lambda`
  changed (0.05, 0.10).
- Runners: `run_lam005_retune.py`, `run_lam010_retune.py` — each chains
  `train_poc.py` → `plot_poc.py` → `evaluate_poc.py` → its diagnostic script
  for both configs, then overlays a 3-point λ sweep (0, 0.01, retuned) on
  circular loss and std_ratio trajectories.
- Diagnostics: `diagnostic_lam005_retune.py`, `diagnostic_lam010_retune.py`
  — implement the 4-step decision procedure above mechanically (no manual
  threshold-picking), plus a 5-step prediction-perturbation trace (folding in
  item 2/A.3 from Run 7's remaining-open-items list) that checks whether
  consecutive gradient steps on the same batch move coa_phase/pol_angle
  predictions coherently (directional learning) or cancel out (noise) —
  compared against mchirp as a healthy control.

Run λ=0.05 first (`run_lam005_retune.py`); fall back to λ=0.10 only if the
Step 0 gate fails at 0.05.

### Run 9a — λ=0.05 result (2026-07-22)

**Both primary tests FAILED the Step 0 gate — mechanical verdict UNINTERPRETABLE
for both, per the pre-registered decision table.** Since neither cleared the
gate, Steps 1–3 (bootstrap, effect size, SNR check, perturbation trace) did
not run — this is by design, not a missing analysis.

| Model            | Head               | frac unhealthy (last 40 ep) | late trend/ep                | Gate     |
|------------------|--------------------|-----------------------------|------------------------------|----------|
| tcn              | coa_phase          | 0.05 (passes <0.10)         | −0.00638 (fails \|·\|<0.005) | **FAIL** |
| poc_a (baseline) | polarization_angle | 0.35 (fails <0.10)          | +0.00718 (fails)             | **FAIL** |

Full trace (`runs/phic_psi_lam005_retune{,_tcn}/20260722_*/history.csv`) shows
*why* each failed differently, and it's worth recording precisely because a
coarser read could misclassify both:

- **tcn coa_phase**: noisy early in the 40-epoch window (0.45–1.1), but the
  last ~15 epochs sit in a tight, stable band (0.58–0.62) — comfortably
  inside [0.5, 2.0]. The failing trend statistic is measuring the decline
  *into* that plateau, not ongoing instability at the end of training. Close
  to passing; a few more epochs at this λ might clear the trend criterion on
  its own.
- **poc_a polarization_angle**: spends epochs ~41–58 well below 0.5 (0.2–0.45,
  climbing), then settles cleanly into 0.53–0.56 for the last ~20 epochs. The
  35% unhealthy fraction is almost entirely from that early ramp, not from
  current instability. Also close to passing, same shape.

Both models: **val circular loss for the primary head stays flat at
1.004–1.008 across the same window** — no movement, same as every prior run.
This doesn't change the verdict (the gate failing means the significance test
never ran, so no "peek ahead" conclusion is being drawn here), but it means
even in the epochs where std_ratio looks closest to healthy, there's no
visible loss movement to explain either.

**Per the pre-committed plan, the next step is λ=0.10** (`run_lam010_retune.py`
/ `diagnostic_lam010_retune.py`), not yet executed. One thing worth deciding
before that run, given the settling-then-plateau shape seen here: whether the
40-epoch/0.005-trend window is the right lens for a `plateau` LR schedule that
takes ~15–20 epochs to fully settle after each LR drop, or whether it's overly
strict early in that settling period. Any change to the gate window should be
made as a documented revision to `preregistration_lam_retune.md` *before*
looking at the λ=0.10 results — not decided after seeing them, for the same
reason the criterion was pre-registered in the first place.

### Run 9b — λ=0.10 result (2026-07-22)

**Both primary tests FAILED the Step 0 gate again — mechanical verdict per
the pre-registration:** "gate fail at λ=0.10 too → report λ alone
insufficient, not counted either way." Steps 1–3 correctly did not run.
This closes the λ-sweep branch (0, 0.01, 0.05, 0.10) for both primary
targets.

| Model            | Head               | frac unhealthy (last 40 ep) | late trend/ep                 | Gate     | vs λ=0.05              |
|------------------|--------------------|-----------------------------|-------------------------------|----------|------------------------|
| tcn              | coa_phase          | 0.28 (fails <0.10)          | −0.00255 (passes \|·\|<0.005) | **FAIL** | worse (0.05→0.28)      |
| poc_a (baseline) | polarization_angle | 0.73 (fails)                | +0.00731 (fails)              | **FAIL** | much worse (0.35→0.73) |

Unlike Run 9a, this is not a near-miss on either primary target. Raw traces
(`runs/phic_psi_lam010_retune{,_tcn}/20260722_*/history.csv`) show two
distinct failure shapes:

- **poc_a polarization_angle**: crashes hard through epochs ~30–48 (down to
  0.18–0.4, well below the healthy band), then climbs steadily and
  monotonically — 0.33 at epoch 49 up to 0.51 by epoch 68, then hovers at
  0.50–0.51 for the last ~11 epochs. It looks like it may be converging, but
  the recovery only started late enough that the 40-epoch window is mostly
  still below 0.5 (29 of 40 epochs) — the gate correctly doesn't credit an
  in-progress climb. Trend is positive and consistent with genuine (if slow)
  convergence, not divergence.
- **tcn coa_phase**: oscillates in roughly [0.2, 0.95] across the entire
  last-40-epoch window with no discernible convergence in either direction —
  qualitatively different from the "settled late" pattern at λ=0.05. This
  reads as instability, not a transient.

Non-primary combos mostly regressed too (see
[`lam010_retune_output/lam010_retune_report.md`](lam010_retune_output/lam010_retune_report.md)):
poc_a coa_phase HEALTHY (λ=0.05) → STILL UNHEALTHY (λ=0.10, frac 0.60); tcn
pol_angle "improved, not stable" (λ=0.05) → STILL UNHEALTHY (λ=0.10, frac
0.90). λ=0.10 helped none of the four head/model combos and made three
worse — raising λ further is not indicated.

**Verdict, filed per the pre-registered table, not re-derived after the
fact:** λ alone is insufficient to stabilize std_ratio for tcn coa_phase or
poc_a polarization_angle. This is explicitly neither a null result nor
counter-evidence for the degeneracy hypothesis — the pre-registration
anticipated exactly this outcome and specified it be reported as such rather
than forced into either bucket. The λ sweep for these two heads (0, 0.01,
0.05, 0.10) is exhausted; the next lever, if pursued, is architecture-level
(each diagnostic script's own gate-fail message says the same), not a
further λ value. The open question flagged in Run 9a — whether the 40-epoch
gate window is well-calibrated for the `plateau` schedule's settling
behavior — remains open but is now moot for this specific λ-sweep decision,
since λ=0.10's poc_a failure mode (still-crashing-then-slowly-recovering) is
not simply "settled late," and tcn's failure mode at λ=0.10 (oscillatory) is
not a plateau-window artifact at all.

### Next steps

- [x] Root cause: tanh saturation at random init for PERIODIC heads (Run 3–4)
- [x] Fix: `activation="tanh"` → `"linear"` in heads_spec.py (Run 5)
- [x] Retrain all 7 configs → normalize_unit pathology found (Run 5)
- [x] Implement magnitude penalty `λ·(|v_raw|−1)²` (Run 6)
- [x] Code-trace inclination loss path (Run 6)
- [x] Five pre-flight silent-failure checks passed (Run 6)
- [x] Retrain 4 models with magnitude penalty λ=0.01 (Run 7)
- [x] Run `analyse_predictions.py` on Run 7 checkpoints (2026-07-20)
- [x] Run full diagnostic checks 1–7 on Run 7 checkpoints (2026-07-21)
- [x] Verification plan Sections A–E executed (2026-07-21)
- [x] std_ratio full trajectories extracted (A.2)
- [x] poc_b config diff + collapse mechanism explained (B)
- [x] cnn_attention config diff + outlier explained (C)
- [x] Bootstrap CI on ang_MAE — 11/12 non-significant (D)
- [x] SNR stratification — no SNR-dependent improvement (E)
- [x] Run λ=0 ablation to isolate increasing-loss trend (F.1/F.2) — **Run 8**,
      2026-07-21: drift absent at λ=0 in 3/4 signals (artifact, not real
      drift); tcn pol_angle persists unresolved (small, +0.0072)
- [x] Pre-register success/failure criterion for the λ retune, before running
      it — **Run 9 setup**, 2026-07-22, see below
- [x] Run multi-step perturbation trace to discriminate learning vs noise (A.3)
      — **will not run for these two primary targets**: gate-failed at both
      λ=0.05 and λ=0.10, so Steps 1–3 correctly never execute, by design
      (see Run 9a, Run 9b). Branch closed.
- [x] Re-tune λ=0.05 for tcn coa_phase — **Run 9a**, 2026-07-22: gate FAILED
      (close — last ~15 epochs stable at 0.58–0.62, but trend/ep from the
      earlier settling period fails the strict threshold). Verdict:
      UNINTERPRETABLE, not null or counter-evidence.
- [x] Re-tune λ=0.10 for tcn coa_phase — **Run 9b**, 2026-07-22: gate FAILED
      again, worse than λ=0.05 (frac 0.05→0.28) — oscillatory, not a
      settling transient. Verdict: **λ alone insufficient**; branch closed.
- [x] Re-tune λ=0.05 for poc_a pol_angle — **Run 9a**, 2026-07-22: gate FAILED
      (35% of last-40-epoch window unhealthy, almost entirely from the
      pre-plateau ramp; last ~20 epochs stable at 0.53–0.56). Verdict:
      UNINTERPRETABLE.
- [x] Re-tune λ=0.10 for poc_a pol_angle — **Run 9b**, 2026-07-22: gate FAILED
      again, much worse than λ=0.05 (frac 0.35→0.73) — crashes early,
      recovers late, crosses 0.5 only in the last ~11 of 80 epochs. Verdict:
      **λ alone insufficient**; branch closed.
- [x] λ sweep exhausted for both primary targets (0, 0.01, 0.05, 0.10) — next
      lever, if pursued, is architecture-level, not a further λ value.
- [x] Standalone perturbation trace (A.3, un-gated) — **executed 2026-07-23**
      (`perturbation_trace_standalone.py` on the lab GPU machine, Run 7
      λ=0.01 checkpoints; output `perturbation_trace_output/
      perturbation_trace_20260723_091229.{md,log}`). **A.3 PROVISIONALLY
      closed — movement without angular learning**, downgraded from CLOSED
      by same-day review (failed mchirp positive control); see the closure
      section and its review addendum below.
- [x] Perturbation-trace calibration run (`early` stage) — **executed
      2026-07-23** (`perturbation_trace_early_20260723_095357.{md,log}`).
      Calibration criterion FAILED (mchirp never read DIRECTIONAL early) →
      geometry classifier retired per the pre-stated tree; but the paired
      probe-loss channel PASSED its control in the same run (early mchirp
      t = −3.4 to −8.5, learning detected) and reads every periodic head
      as null at both stages. **A.3 CLOSED on the validated channel**; see
      the calibration adjudication below.
- [ ] After above items resolved: **proceed to ι-conditioning experiments** —
      not yet started

### Closing punch list resolution (2026-07-23)

The closing punch list (`phic_psi_closing_punch_list.md`) was worked through before treating the investigation as write-up-ready.
Dispositions:

- **A.3 perturbation trace un-gated.** The multi-step trace was decoupled from the retune scripts' Step 0 gate into `perturbation_trace_standalone.py` (see checklist above); it runs against the existing Run 7 λ=0.01 checkpoints with no training and adds a net-vs-sum displacement ratio against the 1/√N random-walk reference to separate directional drift from oscillation.
  Pending lab-machine execution; the thesis chapter cites it as prepared-not-run.
- **λ-sweep wording decided.** The Run 9a/9b pattern (near-miss at λ=0.05, regression everywhere at λ=0.10) is peaked, not monotonic; "exhausted" is now explicitly scoped to the pre-registered sweep and its stopping rule, and a finer freshly pre-registered mini-sweep over λ ∈ [0.02, 0.08] is filed as future work rather than the λ dimension being claimed ruled out.
  This is a documentation decision only — no new training, and the pre-registered verdict language is unchanged.
- **Known-but-unresolved items stated as such** (in the record and in the chapter's threats-to-validity list): inclination's separate Huber-path failure mechanism (traced, ruled out as a φc/ψ confound, not itself resolved); the `SumDiffTrainer`-specific sky_position degradation (flagged, never investigated, out of scope); the 40-epoch/±0.005 gate window vs the plateau-schedule settling time (raised at Run 9a, correctly not retroactively changed, open for future gates).
- **Deferred to future work by name:** the finer λ mini-sweep; the architecture-level std_ratio fix for tcn/coa_phase and poc_a/pol_angle; ι-conditioning as the start of a new investigation rather than a tail of this one.
- **Thesis chapter** drafted from this record at `thesis/chapter_phic_psi_degeneracy.{md,tex}` (2026-07-22, punch-list amendments 2026-07-23); scope-of-conclusion paragraph verified present (§8.1 conditional-claim list, conclusion scoped to the model class tested).

### Standalone perturbation trace executed — A.3 closed (2026-07-23)

`perturbation_trace_standalone.py` ran on the lab GPU machine against all four Run 7 λ=0.01 checkpoints (run dirs confirmed: 20260720_210936 / _213202 / _215403 / _221625).
Design: 25 consecutive gradient steps on one fixed 128-sample batch, predictions tracked on a disjoint 512-sample probe; per head, mean consecutive-step cosine similarity, net-vs-sum displacement ratio (random-walk reference 1/√25 = 0.20), and probe circular loss before/after.
Output: `perturbation_trace_output/perturbation_trace_20260723_091229.{md,log}`.

Result summary (cos-sim / net-sum / probe Δ circ loss):

| Model | coa_phase | polarization_angle | mchirp (control) |
|---|---|---|---|
| poc_a | +0.93 / 0.29 / −0.017 | +0.96 / 0.59 / +0.048 | +0.60 / 0.04 / — |
| poc_b | +0.96 / 0.39 / −0.013 | +0.80 / 0.23 / +0.004 | +0.54 / 0.04 / — |
| tcn | +0.94 / 0.93 / −0.010 | +0.87 / 0.39 / +0.051 | +0.59 / 0.03 / — |
| cnn_attention | +0.66 / 0.40 / −0.046 | +0.64 / 0.32 / +0.029 | +0.51 / 0.16 / — |

Reading:

1. The Run 7 A.3 asymmetry (periodic-head rel_change 89× the scalar control's) is real and explained — periodic raw outputs move coherently and far while the converged mchirp control barely moves net (its positive cos-sim is Adam momentum, which correlates consecutive steps for every head).
2. The coherent movement is dominantly radial, not angular: circular loss depends only on the predicted angle, and |Δloss| ≤ 0.051 over 25 steps while raw displacement is a large fraction of the output norm; the two most directional cases (tcn/coa_phase net/sum 0.93, poc_a/pol_angle 0.59) are exactly the two heads with the known std_ratio pathologies — this is the documented |v|-magnitude dynamics, not phase decoding.
3. No learning signature in the angular residue: coa_phase probe loss down slightly in all four models (−0.010 to −0.046), pol_angle UP in all four (+0.004 to +0.051); all an order of magnitude below the 0.10 rad-equivalent effect floor and within the ~0.03 sampling scale of a 512-sample probe, straddling the random baseline 1.0.

One nominal trigger of the report's own escalation rule (tcn/coa_phase: DIRECTIONAL + Δloss −0.010) is evaluated and dismissed with stated reasons, not silently: sub-noise magnitude, the one head with non-converged std_ratio (radial drift), and arithmetic incompatibility with its flat 80-epoch validation history (a real per-step improvement of this size would have moved the epoch-scale loss by orders of magnitude more than observed).

**Disposition: A.3 CLOSED, consistent with the null.** The last open item from the Run 7 verification battery is resolved; nothing from the battery remains open.
Thesis chapter updated (§6.6 + threats list + future work) in `thesis/chapter_phic_psi_degeneracy.{md,tex}`.

#### Review addendum (2026-07-23, same day): closure downgraded to PROVISIONAL

An internal review of the trace output raised two objections; both are accepted and the CLOSED status above is downgraded to PROVISIONALLY CLOSED.

1. **The positive control failed.** mchirp — the one head with an unambiguous strong signal (R² ≈ 0.96 in every run) — read AMBIGUOUS in all four models, net/sum 0.029–0.164 sitting at the 0.200 random-walk reference.
   An instrument that cannot distinguish a known-learned head from a dead one at these checkpoints cannot certify the periodic-head verdicts.
   Likely cause: epoch-79 checkpoints are converged — a head at its optimum takes small, locally noisy correction steps that mimic "never learned" on a 25-step probe.
   Testable: rerun near the start of training. Since per-epoch checkpoints were never saved (best/final only), the script's new `early` stage substitutes fresh init + ~1-epoch warmup (200 steps).
   Pass = mchirp DIRECTIONAL early, AMBIGUOUS late (convergence effect confirmed; final-stage table interpretable). Fail = mchirp AMBIGUOUS even early (trace methodology unsound; A.3 reverts to open, no verdict from it usable).
2. **Per-case discipline on the two DIRECTIONAL periodic cases** (per the λ=0-ablation standard of naming exceptions instead of averaging them away):
   - poc_a/pol_angle (net/sum 0.586): probe Δcirc = **+0.0478** — loss INCREASED; fails the two-part escalation rule (DIRECTIONAL + decreasing) outright, individually. Clean.
   - tcn/coa_phase (net/sum 0.925 — closest number to the 1.0 ceiling in the table): probe Δcirc = **−0.0098** — nominal trigger, individually. The arithmetic dismissal stands (25 steps ≈ 0.13 epochs; a real −0.0098 at that rate implies epoch-scale movement ~2 orders larger than the flat 80-epoch history). The "inside sampling noise" dismissal does NOT stand as stated: it compared against the probe's marginal SE (~0.03), but the correct comparator for a before/after change on the same fixed 512 samples is the paired SE, which is not recoverable from the recorded output. Per-sample paired statistics (mean ± SE, t, frac improved) are now built into the script for the rerun.

**Status: A.3 PROVISIONALLY CLOSED**, pending the `early` calibration run.
If calibration passes and tcn/coa_phase's Δcirc is paired-insignificant, A.3 closes for good.
If calibration passes and the effect is paired-significant, tcn/coa_phase escalates per the pre-stated rule — it would be the most interesting number in the nine-run investigation.
If calibration fails, A.3 reverts to open with the trace methodology retired.
Chapter updated to match (§6.6 provisional framing, threats-list bullet restored, calibration run added as future-work item 0).

##### Paired-statistics rerun (2026-07-23, `perturbation_trace_final_20260723_095054`)

The final-stage trace was rerun with the per-sample paired statistics added by the review addendum.
Review objection 2 is now settled with the correct statistic:

- **tcn/coa_phase** (the nominal escalation trigger): probe Δcirc = **−0.0097 ± 0.0491 (paired SE), t = −0.20** — decisively insignificant. The trigger dissolves.
- **poc_a/pol_angle**: +0.0426 ± 0.0519, t = +0.82 — insignificant, and an increase besides.
- All periodic-head probe deltas are paired-insignificant (|t| ≤ 1.72); directionality/net-sum values reproduce the first run (tcn/coa_phase net/sum 0.923).
- Side observation supporting the convergence explanation: the converged mchirp head's probe MSE got significantly *worse* under the 25 single-batch steps (t = +4.2 to +9.0 across models) — steps on one repeated batch pull a converged head away from its global optimum, exactly the regime in which the instrument reads a learned head as AMBIGUOUS.

**Status: A.3 remains PROVISIONALLY closed** — the only outstanding condition is the `early`-stage mchirp calibration run (must read DIRECTIONAL early / AMBIGUOUS late). The tcn/coa_phase escalation branch is closed under the paired statistic unless the early-stage run revives it.

##### Calibration run adjudication (2026-07-23, `perturbation_trace_early_20260723_095357`) — A.3 CLOSED

The `early` calibration stage (fresh init + 200 warmup steps ≈ 1 epoch, then the standard 25-step trace) executed on the lab GPU machine for all four configs.

**Mechanical verdict per the pre-stated tree: calibration FAILED.**
mchirp never read DIRECTIONAL early — AMBIGUOUS in three models (net/sum 0.091, 0.128, 0.096) and OSCILLATORY in cnn_attention (net/sum 0.286, cos-sim +0.089).
Per the fail branch, the displacement-geometry classifier (cos-sim/net-sum → DIRECTIONAL/AMBIGUOUS/OSCILLATORY) is **retired**; no verdict from that column, in either run, is usable.

**What the same run validated.**
While the classifier read the early mchirp head as ambiguous-to-noise-like, that head's paired probe MSE was collapsing: Δ = −0.158 (t = −5.2), −0.129 (t = −3.4), −0.113 (t = −4.1), −0.646 (t = −8.5).
The classifier labeled the fastest-learning head in the table "noise-like" — the definitive demonstration that displacement geometry does not measure learning (per-step deltas decorrelate as different samples' errors are corrected; learning is not rigid drift).
The paired probe-loss channel therefore has a **passed positive control** from this very run.

**On the validated channel, every periodic head is null at both stages:**
early |t| ≤ 1.56 (poc_a coa_phase, the largest), final |t| ≤ 1.72, signs mixed across models and heads.
The early stage is a within-run, stage-matched, positive-controlled contrast: over the same 25 steps, the same instrument watches mchirp learn steeply while coa_phase/pol_angle sit at the random baseline.

**Disposition, in the required order:** the pre-stated fail branch is honored (classifier retired, verdict columns void); the closure of A.3 is then re-founded — transparently post hoc — on the paired-statistic channel, which was added by the review (not pre-registered) but carries its own passed control from the same run.
The tcn/coa_phase escalation branch is extinguished (paired t = −0.20 at the final stage); the radial explanation of the 89× asymmetry stands on arithmetic independent of the retired classifier.
**A.3: CLOSED — radial movement without angular learning.**
The post-hoc channel choice is recorded as a residual caveat in the chapter's threats-to-validity list.
Nothing from the Run 7 verification battery remains open.
Chapter updated (§6.6 two-stage table + adjudication, threats bullet, future work back to three items) in both formats.
