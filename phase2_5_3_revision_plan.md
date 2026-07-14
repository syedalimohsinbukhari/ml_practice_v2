# Phase 2.5 / 3 — Revision Plan & Concerns

Context: Phase 2.5/3 changes are implemented but not yet run. Several fixes
are well-targeted; a few risk confounding attribution or reintroducing
problems already diagnosed. This is a pre-run checklist, not a post-mortem —
resolve what's cheap to check now, flag the rest for interpretation later.

---

## Concerns to resolve before the GPU runs

### 1. ResNet: `sigmoid_bias` value — measured or guessed?
- Confirm what value is actually set for ResNet's q head.
- If based on the epoch-0 logit distribution check we discussed (is q's
  sigmoid already saturated before training meaningfully starts), good —
  it's a targeted fix.
- If it's a guess, that's fine as a first pass, but flag it as such so a
  bad result isn't over-interpreted as "the collapse-fix approach failed"
  when it might just be "the guessed bias was off."
- **Action:** log/print the actual `sigmoid_bias` value used per run.

### 2. cnn_baseline: regularization confound with cnn_attention's earlier result
- Was "the same proven regularization from cnn_attention" (32 hidden units,
  dropout 0.3, L2 1e-4) already present in the `cnn_attention` run that went
  from val R2(q) ≈ -0.2 to +0.204? If yes, that improvement was
  **clamp + regularization combined** — we don't know the split.
- Applying regularization now to cnn_baseline is applying a fix designed for
  *overfitting* (cnn_attention's failure mode) to a model that was mildly
  *underfitting* q (cnn_baseline's failure mode). Different diseases.
- **Action:** clarify the timeline. If unclear, consider one ablation run of
  cnn_attention with clamp-only (no q-head regularization) to isolate which
  component did the work, before trusting attribution on the next batch.

### 3. Oversampling — leakage risk
- `q_high_mchirp_low: 2` oversampling duplicates rows "before
  `TargetTransforms.fit()`" — need to confirm this happens **after** the
  train/val split, not before.
- If duplication happens pre-split, duplicated hard-regime examples could
  land in both train and val, inflating apparent val performance on exactly
  the subset being targeted — a false positive that won't show up anywhere
  except as "suspiciously good" q_high_mchirp_low numbers.
- **Action:** trace `run_experiment()`'s call order — split, then
  oversample-train-only, then fit transforms on the (oversampled) train set.
  Confirm val set is never touched by the oversampling step.

### 4. cnn_attention dedicated q branch — pooling method on `q_tokens`
- The trunk now returns per-token features `(B, 128, 128)` instead of the
  final pooled vector. How does `attach_heads` reduce this to a fixed-size
  input for the q MLP head?
- If it's a plain GAP/flatten internally, this reintroduces the same
  information-loss bottleneck diagnosed in the original `cnn_baseline`
  critique — just moved one level later, and the "dedicated branch" gains
  nothing over the shared pooled features.
- If it's attention-pooled or otherwise structure-preserving, this is the
  right move and should be checked in for correctness, not changed.
- **Action:** review `attach_heads`'s branch-handling code before the run,
  not after — cheap to check, expensive to discover post-hoc.

### 5. InceptionTime's dead `merger_time` head — unrelated but unresolved
- R2(merger_time) ≈ 0.005 train / -0.002 val, despite full loss weight
  (weight_merger_time at e³ ceiling like mchirp, which fits fine).
- Not a q problem, not addressed by any of the above changes. Worth flagging
  to avoid it getting lost while attention is on q — a fully dead head in
  one of five architectures is a real bug, not noise.
- **Action:** separate diagnostic pass on InceptionTime specifically —
  epoch-0 logit/output check for merger_time, same as suggested for
  ResNet's q, since dead-from-the-start is the same signature.

### 6. Weight trajectory sanity check per run
- With warmup added to ResNet and the clamp scheme changed there, its
  `weight_q` curve will look different than any prior run. Same for
  cnn_baseline now carrying q-head regularization for the first time.
- **Action:** before reading final R2/MAE numbers, glance at each run's
  weight_* trajectory (5 seconds via the history.csv columns) to confirm it
  behaved as intended, rather than discovering a new anomaly only after
  80 epochs are already spent.

---

## Suggested run + read order

1. **ResNet first** — cleanest single-variable test (clamp reverted to flat
   3.0 + warmup + sigmoid_bias). Check: did `std_ratio_q` move off 0.0? Did
   `val_r2_q` stop being flat-frozen? This tells you if the collapse
   mechanism is understood.
2. **cnn_baseline second** — check whether q-head regularization helps,
   hurts, or does nothing to a model that was underfitting, not
   overfitting. This is the "does the cnn_attention fix generalize"
   question.
3. **cnn_attention third** — check whether the dedicated `q_tokens` branch
   improves on the already-decent 0.204 val R2(q), and whether the
   oversampling shows up as a real gain or a leakage artifact (cross-check
   q_high_mchirp_low numbers against a quick manual leakage audit if the
   jump looks too good).
4. Hold off drawing conclusions about which fix mattered most until #1-#3
   are back — several changes are stacked per config, so isolating
   attribution matters more than the raw final number this round.

---

## What NOT to conclude yet

- A big jump in cnn_baseline's or cnn_attention's val R2(q) does **not** by
  itself confirm the oversampling or regularization worked as intended —
  check concern #3 (leakage) and #2 (confound) first.
- A fixed ResNet does **not** validate the `sigmoid_bias` approach in
  general unless you know whether the value used was measured or guessed
  (concern #1) — a lucky guess and a principled fix look identical in the
  metrics.

---

## Resolution notes (2026-07-14, before GPU runs)

### Concern 1 — sigmoid_bias: defaults to 0.0, not used

Verified: resnet1d has no `per_head` overrides and no global `sigmoid_bias`
in its YAML. `heads.py` defaults `sigmoid_bias` to 0.0 (Keras default
`bias_initializer="zeros"` — sigmoid(0) = 0.5, the midpoint of [0, 1]).
The sigmoid_bias infrastructure exists in code but is **not active** for
this run. The actual fixes are: clamp revert (1.0 → 3.0) + warmup (0 → 5
epochs). Clean attribution if resnet1d revives.

**Recommendation:** If resnet1d stays dead after clamp+warmup, try
`sigmoid_bias: 0.5` (pushes initial sigmoid output toward ~0.62, away from
the 0-region where gradients vanish) as a follow-up experiment.

### Concern 2 — cnn_baseline is overfitting, not underfitting

cnn_baseline Phase 2 attempt #2: train_r2_q = 0.628, val_r2_q = 0.075 →
**gap = 0.55**. This IS overfitting — the model fits training data 8× better
than validation. It's just less severe than cnn_attention's pre-fix gap of
0.90. Regularization (dropout + L2 + smaller hidden) targets exactly this gap.
The different failure mode concern is noted but regularization is the correct
tool for a model that overfits (even mildly).

### Concern 3 — leakage: CONFIRMED SAFE

Traced `run_experiment()` line by line:
1. `load_arrays(..., "training")` → `train_strain`, `train_params`
2. `load_arrays(..., "validation")` → `val_strain`, `val_params`
3. Oversampling block: only touches `train_strain`, `train_params`
4. `make_dataset(val_strain, val_params, ...)` — validation arrays untouched

The HDF5 has separate `training/` and `validation/` groups. No cross-
contamination possible. Oversampling happens pre-transforms-fit, which is
correct (z-score stats computed on the effective training population).

### Concern 4 — GAP pooling is deliberate, not accidental

The q_tokens branch uses `GlobalAveragePooling1D` (uniform weights) vs the
main path's `AttentionPooling` (learned per-position scores). These ARE
different operations:
- **Attention pool:** learns which time positions matter, can overfit to
  specific positions in training
- **GAP:** uniform weighting, every time step contributes equally

For q (mass ratio), the hypothesis is that q-relevant information is
diffusely distributed across the waveform (subtle amplitude/timing
differences throughout), so uniform pooling may generalize better than
learned attention which latches onto training-set-specific positions.

The representation being pooled also differs: q_tokens are the raw
transformer output BEFORE the attention-pool compression step, so they
haven't been selected/compressed for the other heads' objectives yet.

### Concern 5 — InceptionTime merger_time: CONFIRMED DEAD

Verified from `runs/inception_time/20260714_121546/history.csv`:
- `r2_merger_time` (train) = 0.005 (zero — can't fit training data)
- `val_r2_merger_time` = −0.002 (zero — no generalization)
- Physical MAE = 0.049 s (25% of the 1.6–1.8 s range)

Compare cnn_attention: train_r2 = 0.962, val_r2 = 0.904 — merger_time is
well-learned. InceptionTime's merger_time is a genuinely dead head with full
loss weight (weight_merger_time at exp(3.0) ceiling). This needs separate
diagnosis (epoch-0 logit check, same protocol as resnet1d's q) but is
**not a q problem** and not addressed by any Phase 2.5/3 change. Flagged
for next diagnostic pass.

### Concern 6 — weight trajectory: add to post-run checklist

Agreed. Each run's `weight_*` columns should be checked first (5 seconds
via `history.csv`) before reading any R²/MAE numbers. The key signals:
- resnet1d: weight_q should rise past 2.72 (clamp ceiling lifted to 20.09)
- cnn_baseline: weight_q should stay at 2.72 (clamp unchanged) but the
  trajectory shape may differ with regularization
- cnn_attention: same clamp as before (1.0), but q_tokens branch may
  change the loss landscape → weight_q could move differently
