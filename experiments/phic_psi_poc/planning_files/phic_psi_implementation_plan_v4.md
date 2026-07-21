# φc/ψ Degeneracy Handling — Implementation Plan (Proof-of-Concept)

Companion to `phi_c_psi_degeneracy_poc.md` and
`phic_psi_inclination_investigation.md`. This is the build plan for testing
the sum/difference-head + inclination-curriculum design in an isolated
branch, before any decision about merging into the main gwml pipeline.

**Goal of this PoC:** get an honest answer to "does the difference-head
learn any real, modest signal once ψ is reactivated and trained with
inclination-aware curriculum weighting, or does it stay at near-zero
R² (confirming the degeneracy is effectively exact for this population)."
Not aiming for a finished feature — aiming for a clean go/no-go signal.

---

## 0. Isolation setup

- New branch off current `gwml` main, e.g. `poc/phic-psi-degeneracy`.
- New subfolder, e.g. `experiments/phic_psi_poc/`, holding:
  - all new code (transform utils, loss function, config)
  - a standalone validation script (Sec. 3 below), runnable independent of
    the full training pipeline
  - a `NOTES.md` logging what's confirmed vs. still assumed, so this
    doesn't silently drift out of sync with the two design docs
- Do not modify `DEFAULT_HEADS`, `heads_spec.py`, or `MultiHeadTrainer`
  in place — subclass / extend from the experiment folder so main-branch
  training runs are unaffected while this is being tested.

## 1. Prerequisite check (do this before writing any loss code)

This is Sec. 6 from the original PoC doc, now with the added inclination-
error dependency from the investigation doc. Do NOT skip to Step 2 until
this is done — everything downstream depends on the answer.

1.1. **Confirm the sign/combination empirically.** Using existing
     (currently dormant) `polarization_angle` and active `coa_phase`
     columns on a validation slice, check which of `φc+2ψ`, `φc−2ψ`
     correlates more cleanly with available ground truth, conditioned on
     `sign(cos ι)`.

1.2. **Derive/confirm the curriculum weight function `w(ι)` properly**,
     rather than guessing between `cos²ι` and `(1−cosι)²/2`. Suggested
     approach: reuse the standalone analytical test harness from the
     investigation doc (`joint_invert.py`-style script) — sweep `ι` from
     0 to π/2, and at each value, quantify how much the joint two-detector
     `(R,δ)` constraint region shrinks (e.g., distance between the true
     and nearest spurious solution, or a numerically estimated Jacobian
     condition number of the (φc,ψ)→(R,δ) map). Fit `w(ι)` to that curve
     rather than to intuition. This can be done entirely inside
     `experiments/phic_psi_poc/`, no training required.
     **Do not derive this at a single fixed sky-location/antenna-pattern
     pair** (h/t DeepSeek review) — the toy `(a,b)` coefficients used in
     the investigation doc were one arbitrary point, and the condition
     number of the (φc,ψ)→(R,δ) map depends on the specific antenna
     mixing at that sky position. Average the Jacobian/condition-number
     computation over ~50 random sky-locations (or, better, sample from
     the actual training population's sky distribution) before fitting a
     single population-representative `w(ι)`. This doesn't contradict the
     investigation doc's finding that sky position isn't an independent
     degeneracy-breaking axis — it's a separate claim about how much the
     *rate* of degeneracy-breaking varies with sky position, which needs
     averaging out for a `w(ι)` that's valid across the whole population.

1.3. **Confirm the data representation preserves phase** (h/t DeepSeek
     review, likely already satisfied): confirm the backbone ingests raw
     whitened time-domain strain (which inherently encodes phase in the
     temporal structure) rather than magnitude-only Q-transform
     spectrograms. If it's the former (expected, given the existing
     dual-detector time-domain pipeline), this is a non-issue — just a
     one-line confirmation, not a redesign.

1.4. **Confirm true inclination is actually accessible at the batch level
     during training** (not just as an iota-head label deep in some other
     part of the loss) — trace this through the current data loader /
     training loop before assuming it's a free variable to condition on.

1.5. **Decide the curriculum mechanism**, since "curriculum" was used
     loosely earlier: pick between (a) static per-sample loss weight
     `w(ι_true)` throughout training, (b) a training-progress schedule that
     phases in the diff-head loss over epochs, or (c) both. **Decision:
     start with (a) alone** — simpler to interpret, and (b) adds a second
     confound (epoch vs. sample) to an already-new head. Slow/gentle start
     preferred for this PoC over an aggressive schedule.

1.6. **Histogram the true `cos ι` distribution** in the training split
     before committing to Run B (h/t Kimi review). If e.g. 95% of samples
     have `|cos ι| > 0.9`, `w(ι)` will suppress the diff-head loss on
     nearly everything and Run B won't produce a statistically meaningful
     signal regardless of whether the underlying mechanism works. If the
     population is too face-on-heavy, either temporarily augment with more
     edge-on injections for this PoC, or explicitly accept and document
     that the test will be statistically weak.

## 2. Code components (only after Step 1 is resolved)

### 2.1. Reactivate `polarization_angle`
- Add back to `DEFAULT_HEADS` (experiment-local config, not main config)
  as an ordinary small `PERIODIC` head, period π, same as before it was
  dropped.

### 2.2. Vector transform utility (new, small)
- `normalize_unit(z)`: project a predicted `(sin,cos)` pair to unit
  modulus. One-line, same normalization vMF's `mu_raw` already uses.
- `complex_mul(z1, z2)`, `complex_mul_conj(z1, z2)`: sum-vector and
  difference-vector via complex multiplication / multiplication by
  conjugate. Pure math, no learned parameters.
- Unit test this in isolation against hand-computed angle sums on a
  handful of known angle pairs before touching any model code.

### 2.3. Sum/diff loss function (new)
- **Remove the individual φc and ψ PERIODIC losses entirely** (h/t Kimi
  review). The two heads' raw output tensors stay (needed to compute the
  transform below), but no loss is applied to them directly — all
  training signal flows through the sum/diff losses. Keeping the
  individual losses alongside sum/diff would double-apply gradient signal
  on the same two parameters, with the individual losses continuing to
  enforce exact precision in the direction the whole design is trying to
  relax.
- Label the two derived vectors `combo_A` and `combo_B` internally, not
  `sum`/`diff`, until Step 1.1's empirical sign check resolves which one
  is actually well-constrained (h/t Kimi review) — don't bake the
  "sum = good" assumption into the loss function before it's confirmed.
  Once resolved, assign the curriculum weight to whichever combo is the
  poorly-constrained one.
- Given predicted and true `(φc_vec, ψ_vec)`:
  1. normalize both predicted vectors
  2. compute predicted and true `combo_A` and `combo_B` vectors via
     complex multiply / multiply-by-conjugate
  3. **Use an isotropic circular loss, not Huber** (h/t DeepSeek review):
     `L = 1 − dot(pred_norm, true_norm)`, exactly equal to `1 − cos(Δθ)`
     for unit vectors — provably a function of angular difference only,
     never absolute direction. This sidesteps a real risk in the
     originally-assumed "reuse existing Huber-on-vector" plan: if the
     current `PERIODIC` loss applies Huber independently to the sin- and
     cos-components (rather than to the joint vector norm), the summed
     per-component loss is *not* exactly rotation-invariant in the
     large-error regime — a risk this PoC specifically can't afford,
     since the whole point is measuring whether a *weak* signal exists in
     the poorly-constrained combo, and an anisotropic loss could mask or
     fake that signal. (Note: pure squared-Euclidean distance between
     unit vectors *is* already exactly isotropic — `‖z1−z2‖²=2−2cosΔθ` —
     so this concern applies specifically to Huber's piecewise L1/L2
     behavior per-component, not to Euclidean distance in general.)
     **Integration point:** if existing heads combine their base loss with
     a learned `log_var` (heteroscedastic weighting, matching the
     `log_var_clamp` mechanism), wrap this new circular loss the same way
     rather than leaving it bare — confirm the existing combination
     pattern before assuming a drop-in swap. **This also reinforces the
     Step 2.4 clamp-initialization decision** (h/t DeepSeek review):
     `1−cosΔθ` has a vanishing gradient near `Δθ=0` (behaves like
     `Δθ²/2`), so if `log_var` starts very tight (very negative),
     `exp(−log_var)` sharply amplifies that gradient while predictions
     are still far off early in training — a plausible source of loss
     spikes/instability. This is an additional, concrete reason (beyond
     avoiding double-suppression with `w(ι)`) to keep the combo clamps at
     the same moderate default as other `PERIODIC` heads rather than a
     tight `q`-like preset.
  4. multiply the poorly-constrained combo's loss term by `w(ι_true)`
     from Step 1.2 (batch-level, per-sample)
  5. return both loss terms separately (not pre-summed) so each can get
     its own `log_var_clamp` entry, same pattern as `q`

### 2.4. Trainer integration
- Register two new `log_var_clamp` entries (combo_A, combo_B) in
  the experiment-local `MultiHeadTrainer` config.
- **Initialize both clamps at the same default used for other ordinary
  PERIODIC heads — not a tight preset like `q`'s** (h/t Kimi review,
  matches preference for a slow/gentle start). `w(ι)` already handles
  face-on suppression at the loss-weighting level; stacking a tight clamp
  on top risks double-suppressing signal on exactly the edge-on samples
  where the poorly-constrained combo should be learnable. Let the clamp
  tighten from training dynamics only if the combo genuinely fails to
  learn even on edge-on samples.
- Confirm the trainer's existing per-head diagnostics (std_ratio, val R²,
  whatever's used for `q` right now) apply cleanly to these two derived
  heads without special-casing — if they don't, that's a sign the
  sum/diff heads need their own diagnostic wiring, worth flagging early
  rather than discovering it after a training run.

### 2.5. Early diagnostic (before waiting on full convergence)
- Within the first 10-20 epochs, plot the poorly-constrained combo-head's
  predicted `log_var` against true `|cos ι|` (h/t Kimi review). If the
  curriculum weighting is working, predicted precision should correlate
  with inclination — high log_var (low precision) near face-on, lower
  log_var (higher precision) toward edge-on. This is a much faster
  go/no-go signal than waiting for R² to converge, and catches a broken
  `w(ι)` wiring early.
- **Even faster check, within the first 2-3 epochs** (h/t Kimi review):
  plot the raw (pre-clamp) combo loss value itself against true `|cos ι|`.
  If `w(ι)` is wired correctly, face-on samples should show near-zero
  loss immediately (since their loss term is heavily downweighted),
  while edge-on samples show non-zero loss. This catches a broken weight
  function before `log_var` has had any time to adapt, and is cheaper to
  check than the log_var correlation above.

### 2.6. Inference-time reconstruction (for the validation script, Sec. 3)
- Even though this PoC doesn't test inference end-to-end, the round-trip
  transform belongs in the validation script as a correctness check.
  **Use multiplication, not vector addition** — a naive
  `z_φc ≈ normalize(z_sum + z_diff)` shortcut only recovers the correct
  angle when a coefficient (`cos((S−D)/2)`) is positive; when it's
  negative, `normalize()` silently returns the angle flipped by π. Avoid
  it — use the complex-multiplication method below instead.
- **The branch ambiguity is real, not just relabeling, and the two
  angles' branch choices are coupled — do not resolve them
  independently** (h/t DeepSeek review, confirmed numerically; this
  corrects an under-specification in the original version of this
  section). Squaring `z_combo_A · z_combo_B` to solve for `φc` and
  squaring `z_combo_A · conj(z_combo_B)` to solve for `ψ` each introduce
  their own branch ambiguity (2-fold for φc, 4-fold for ψ before
  accounting for ψ's own period-π redundancy) — but picking a branch for
  φc and a branch for ψ *independently* produces spurious combinations
  that do not reproduce the original combo_A/combo_B when re-encoded.
  Confirmed numerically: of 2×4=8 independently-chosen combinations, only
  4 satisfy joint consistency, and those 4 collapse to just 2 physically
  distinct answers (the true point and the known `φc+π, ψ+π/2` alias).
  - **Correct procedure:** rather than decoding raw scalar angles and
    doing arithmetic on them directly, **compute via the `complex_mul`
    utilities from Step 2.2** (h/t DeepSeek review) — `z_prod =
    complex_mul(z_combo_A, z_combo_B)` gives `e^{i2φc}` directly, and
    `z_ratio = complex_mul_conj(z_combo_A, z_combo_B)` gives `e^{i4ψ}`
    directly, with the wraparound handled automatically by how complex
    multiplication works (no risk of a raw `A−B` scalar subtraction
    crossing the `±π` branch cut incorrectly). Then take
    `φc = angle(z_prod)/2 + k·π` for `k∈{0,1}` and
    `ψ = angle(z_ratio)/4 + k·π/2` for `k∈{0,1,2,3}`. This is
    mathematically equivalent to decoding `A,B` and wrapping their
    sum/difference before dividing — the enumerate-and-check step below
    already made the original formulation robust to this — but computing
    it via the existing `complex_mul` utilities is simpler, reuses code
    already written in 2.2, and removes an entire class of
    easy-to-introduce off-by-2π implementation mistakes.
  - **Test every `(φc,ψ)` candidate pair by re-encoding it and checking it
    reproduces both `A` and `B`** (not by picking each branch in
    isolation). Keep only jointly-consistent pairs.
  - For the **validation script specifically** (Step 3, where ground
    truth is available by construction), it's fine to pick the
    jointly-consistent candidate closest to the known true `(φc,ψ)` as a
    convenience — but note this "nearest to reference" trick has no
    equivalent at real inference time, where there is no ground truth to
    compare against. Resolving which of the jointly-consistent branches
    is correct at actual inference time (as opposed to validating the
    code against synthetic data) is an open problem, explicitly out of
    scope for this PoC (see Sec. 6).
  - Validation script should confirm the joint-consistency procedure
    recovers the original injected `(φc,ψ)` (up to the known alias) on
    synthetic known-answer cases before any of this is trusted on real
    predictions.

## 3. Standalone validation script (build this in parallel with 2.2, not after)

Before running any real training, a small script (reusing the
`invert_test2.py` / `joint_invert.py` logic from the investigation doc)
that:
- takes a batch of *true* (φc, ψ, ι, sky-position-equivalent antenna
  coefficients),
- confirms the sum/diff transform utilities (2.2) reproduce the same
  sum/diff values the analytical toy model predicts,
- confirms `w(ι)` behaves as expected at the extremes (→0 at ι=0, →max at
  ι=π/2).

This catches implementation bugs (sign errors, period mismatches) using
known-answer synthetic cases, before they get buried inside a multi-hour
training run.

## 3.5. Minor implementation notes (h/t Kimi review)
- **Normalization epsilon:** reuse the same epsilon as the vMF head's
  `mu_raw` normalization, to avoid division-by-zero if a predicted
  `(sin,cos)` pair is near-zero in both components.
- **Verify the Huber-on-2D-vector loss form actually exists** in the
  current codebase before assuming it's a drop-in reuse. The existing
  `PERIODIC` loss may be per-component Huber on raw `tanh` outputs rather
  than a true vector loss on the unit circle — if so, this is a small
  write, but confirm rather than assume.
- **True ι at batch level (Step 1.4):** if the current dataloader drops
  true `ι` after constructing the iota-head's own target, it will need to
  be piped through as an additional batch field. This is a data-plumbing
  check, not a modeling one — confirm early, per Step 1.4.

## 4. Training run plan

- **Run A (baseline / control):** φc and ψ as independent heads, no
  sum/diff transform, no curriculum weight — i.e., the naive setup this
  whole investigation is trying to improve on. Needed as the comparison
  point; without it "the diff-head trained to R²=0.05" has no baseline to
  judge against.
- **Run B (PoC):** sum/diff heads, curriculum-weighted per Step 1.2/2.3.
- Use the same data split, same seed, same base architecture (whichever
  backbone currently performs best, e.g. TCN per the multi-architecture
  comparison) for both runs — this is a controlled comparison, not a
  from-scratch architecture search.
- **Run A must use the exact same loss function class as Run B** (h/t
  DeepSeek review), just applied to the raw φc/ψ vectors individually
  instead of to the combos. Document this explicitly in `NOTES.md` — if
  Run B's `UnitVectorLoss`/circular loss differs from whatever Run A
  uses, a difference in results could come from the loss reshaping alone,
  not from the sum/diff decomposition + curriculum weighting. The only
  variable changed between A and B should be the grouping of the
  gradients (per-parameter vs. per-combo), nothing else.
- Optional Run C (mentioned earlier, still pending): fixed `ι=0` dataset
  slice, as a guaranteed-null-signal sanity check that the diff-head
  correctly collapses to near-zero when the degeneracy is exact by
  construction — useful for confirming the loss/clamp code works
  correctly, independent of whether the real population shows any signal.

## 5. Evaluation checklist

- [ ] Sum-head trains to reasonable R² (should behave like a normal
      well-identified `PERIODIC` head — if not, something upstream is
      broken, unrelated to the degeneracy question)
- [ ] Diff-head R² in Run B compared against Run A's raw ψ-head R² (not
      just against zero) — the real question is whether decomposition +
      curriculum improves on the naive baseline, not just whether the
      diff-head is "good"
- [ ] Diff-head residuals binned jointly by (true `cos ι`, iota-head
      prediction error) per the investigation doc's Sec. 6 addition —
      confirms whether iota-head accuracy is gating recoverable signal
- [ ] Run C (if done) confirms diff-head collapses to near-zero under
      guaranteed-exact degeneracy — sanity check on the code, not the
      physics
- [ ] Document whatever the result is in `NOTES.md`, including negative
      results — a clean "difference-head carries no more signal than
      Run A's naive ψ-head" is still a useful, publishable-adjacent
      finding for the eventual paper's discussion-of-degeneracies section

## 6. Explicitly out of scope for this PoC

- No architecture changes beyond reactivating the existing `PERIODIC`
  head type for ψ.
- No epoch-based curriculum scheduling (Step 1.5 option b) unless Run B
  results suggest static weighting isn't enough.
- No real antenna-pattern / sky-position-based refinement of `w(ι)` —
  Sec. 3 of the investigation doc already ruled out sky position as a
  needed second axis; revisit only if Run B results contradict that.
- No merge to main gwml branch until Run B vs Run A comparison is
  reviewed and the go/no-go call is made explicit.

## 7. Appendix A — Reference formulas (use exactly as given, do not re-derive)

These were derived and numerically verified across the design discussion
that produced this plan. **Whoever implements this should use these
directly rather than re-deriving signs/conventions from scratch** — the
signs and branch structure here were confirmed against working numerical
tests, and re-deriving from the physics independently risks reintroducing
errors (a sign flip, a wrong branch-cut assumption) that were already
found and fixed once.

**Convention:** a `PERIODIC` head's raw vector is the tuple `z=(s,c)`
representing `s=sinθ`, `c=cosθ` for whatever angle `θ` it encodes — i.e.
`z ≡ cosθ + i·sinθ` is *not* the convention; here `c` is the real part and
`s` is the imaginary part (`z ≡ c + i·s`), matching `atan2(s,c)=θ`. Keep
this consistent throughout — swapping which component is "real" flips
signs silently in every formula below.

### A.1 — Complex-multiplication transform utilities (Step 2.2)

Given `z1=(s1,c1)` at angle `θ1` and `z2=(s2,c2)` at angle `θ2`:

```
complex_mul(z1, z2)       -> angle (θ1+θ2):
    s_out = s1*c2 + c1*s2
    c_out = c1*c2 - s1*s2

complex_mul_conj(z1, z2)  -> angle (θ1−θ2):
    s_out = s1*c2 - c1*s2
    c_out = c1*c2 + s1*s2

normalize_unit(z):
    r = sqrt(s*s + c*c + eps)
    return (s/r, c/r)
```

### A.2 — Sum/diff (combo) construction (Step 2.3)

Given the raw φc head vector `z_φc=(s_φc,c_φc)` and ψ head vector
`z_ψ=(s_ψ,c_ψ)` — note `z_ψ` already encodes angle `2ψ` internally, not
`ψ`, because `PERIODIC.transform_head` rescales by `2π/period` before
taking sin/cos, and ψ's period is π (see `phi_c_psi_degeneracy_poc.md`
Sec. 4). Do not rescale again.

```
combo_A = complex_mul(z_φc, z_ψ)        # angle = φc + 2ψ
combo_B = complex_mul_conj(z_φc, z_ψ)   # angle = φc − 2ψ
```

Step 1.1 determines empirically whether `combo_A` (φc+2ψ) or `combo_B`
(φc−2ψ) is the well-constrained one — the labels `combo_A`/`combo_B`
above are fixed by this formula regardless of that result; only the
*curriculum-weight assignment* (which one gets `w(ι)` applied) depends on
Step 1.1's outcome.

### A.3 — Isotropic circular loss (Step 2.3)

For normalized predicted `z_pred=(s_p,c_p)` and true `z_true=(s_t,c_t)`
(true vectors are already unit-norm by construction):

```
dot = s_p*s_t + c_p*c_t     # = cos(θ_pred − θ_true)
L   = 1 - dot
```

This is exact and provably isotropic — a function of angular difference
only. Do not substitute a per-component Huber or L1/L2 form without
re-verifying isotropy (see Sec. 2.3 discussion above).

### A.4 — Reconstruction with branch handling (Step 2.6)

Given predicted (or true, for validation) `combo_A=(sA,cA)` and
`combo_B=(sB,cB)`:

```
z_prod  = complex_mul(combo_A, combo_B)        # angle = 2*phi_c
z_ratio = complex_mul_conj(combo_A, combo_B)    # angle = 4*psi

phi_c_half = atan2(z_prod[0], z_prod[1])        # in (-pi, pi]
four_psi   = atan2(z_ratio[0], z_ratio[1])      # in (-pi, pi]

phi_c_candidates = [ phi_c_half/2 + k*pi         for k in (0, 1) ]
psi_candidates   = [ four_psi/4   + k*(pi/2)     for k in (0, 1, 2, 3) ]
```

**Joint-consistency filter (mandatory — do not select branches
independently):** for every `(phi_c_cand, psi_cand)` pair formed from the
two lists above, re-encode:

```
z_phic_cand = (sin(phi_c_cand), cos(phi_c_cand))
z_2psi_cand = (sin(2*psi_cand), cos(2*psi_cand))
recombo_A   = complex_mul(z_phic_cand, z_2psi_cand)
recombo_B   = complex_mul_conj(z_phic_cand, z_2psi_cand)
```

Keep the pair only if `recombo_A ≈ combo_A` **and** `recombo_B ≈ combo_B`
(compare via `1 - dot(...)  < tol` on each, using A.3's dot form, not raw
angle subtraction). Discard all other pairs — they are combinatorial
artifacts of taking each square root independently, not valid solutions.
Confirmed numerically: this reduces 2×4=8 raw candidates to exactly the
physically meaningful set (the true answer plus the known `φc+π, ψ+π/2`
alias — never more, never fewer, for a properly-formed `combo_A/combo_B`
pair).

### A.5 — Toy antenna-pattern / detector-response model (Step 1.2 harness)

For deriving `w(ι)`, reuse this exact model (matches
`degeneracy_check.py`/`joint_invert.py` from the investigation) — do not
let an implementer reconstruct the antenna-pattern algebra from memory,
since sign conventions here are easy to flip:

```
k1(iota) = (1 + cos(iota)**2) / 2      # h_plus coefficient
k2(iota) = cos(iota)                   # h_cross coefficient

h_plus(phi_c, iota, Phi)  = k1(iota) * cos(2*Phi + phi_c)
h_cross(phi_c, iota, Phi) = k2(iota) * sin(2*Phi + phi_c)

# per-detector antenna pattern, given sky-position-derived (a, b):
F_plus(psi, a, b)  = a*cos(2*psi) + b*sin(2*psi)
F_cross(psi, a, b) = b*cos(2*psi) - a*sin(2*psi)

detector_signal(psi, phi_c, iota, a, b, Phi) =
    F_plus(psi,a,b)*h_plus(phi_c,iota,Phi) + F_cross(psi,a,b)*h_cross(phi_c,iota,Phi)

# project the (2*Phi)-frequency component to recover (R, delta):
c_proj = sum(signal * cos(2*Phi))
s_proj = sum(signal * sin(2*Phi))
R      = hypot(c_proj, s_proj)
delta  = atan2(-s_proj, c_proj)
```

At `iota=0` this collapses to `R,delta` independent of ψ along any line
`phi_c + 2*psi = const` (fully degenerate) — this is a correctness check
on the harness itself: if sweeping ψ along such a line at `iota=0` does
*not* produce constant `R,delta` to numerical precision, the harness has
a bug, stop and fix it before trusting any derived `w(ι)`.
