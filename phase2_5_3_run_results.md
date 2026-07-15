# Phase 2.5 / 3 — Run Results & Analysis

New runs (2026-07-14 16:18–16:48) vs Phase 2 attempt #2 baselines (2026-07-14 11:33–12:15).

## Run index

| Model | Phase 2 baseline | Phase 2.5/3 new | Changes applied |
|-------|-----------------|-----------------|-----------------|
| cnn_attention | `20260714_113745` | `20260714_161833` | oversample + q_tokens branch |
| cnn_baseline | `20260714_113347` | `20260714_162400` | oversample + q-head reg |
| resnet1d | `20260714_114721` | `20260714_164144` | oversample + clamp revert (1.0→3.0) + warmup (0→5) |
| inception_time | `20260714_121546` | `20260714_162758` | oversample only |
| tcn | `20260714_115404` | `20260714_164826` | oversample only |

---

## Headline: q metrics at epoch 79 — before vs after

| Model | train_r2_q (old→new) | val_r2_q (old→new) | val_std_ratio_q (old→new) | weight_q (old→new) | q MAE phys (old→new) |
|-------|---------------------|---------------------|--------------------------|--------------------|--------------------|
| cnn_attention | 0.399 → **0.300** | 0.204 → **0.180** | 0.602 → **0.543** | 2.72 → 2.72 | 0.153 → 0.159 |
| cnn_baseline | 0.628 → **0.417** | 0.075 → **+0.151** | 0.653 → **0.531** | 2.72 → 2.72 | 0.165 → 0.161 |
| resnet1d | −5.563 → **+0.999** | −5.738 → **+0.100** | 0.000 → **0.682** | 2.72 → **20.09** | 0.395 → **0.162** |
| inception_time | 0.395 → **0.342** | 0.219 → **0.197** | 0.572 → **0.530** | 2.72 → 2.72 | 0.153 → 0.153 |
| tcn | 0.266 → **0.192** | 0.260 → **0.252** | 0.557 → **0.480** | 2.72 → 2.72 | 0.150 → 0.151 |

---

## Per-model analysis

### 1. resnet1d — 🟢 DEAD SIGMOID REVIVED (massive win)

**The clamp revert + warmup resurrected q from complete death.**

| Metric | Old (dead sigmoid) | New | Delta |
|--------|-------------------|-----|-------|
| train_r2_q | −5.56 | **+0.999** | +6.56 |
| val_r2_q | −5.74 | **+0.100** | +5.84 |
| val_std_ratio_q | 0.000 | **0.682** | +0.682 |
| weight_q | 2.72 | **20.09** | +17.37 |
| q MAE (phys) | 0.395 | **0.162** | −0.233 (−59%) |

**Trajectory:** Immediately starts learning q at epoch 0 (val_r2 = −0.10 vs old
−3.50). Weight_q ramps from 1.02 to the exp(3)=20.09 ceiling by epoch 17 — the
model assigns q full gradient share now. Train R² hits 0.999 by epoch 50 and
stays there. val_r2 peaks at 0.170 (epoch 12) then declines to ~0.10 plateau by
epoch 30, where it holds steady through epoch 79. No crash — a stable slightly-
positive plateau.

**BUT:** The train/val gap is now 0.90 (0.999 vs 0.100) — resnet1d is now
overfitting q just as badly as the CNN trunks were pre-fix. The clamp fix
revived the sigmoid, but without q-head regularization, the model memorizes q
on the training set while barely generalizing. This is the **same failure mode
cnn_attention had in the original Phase 2 runs** (train=0.91, val=−0.18) before
q-head regularization + clamp fix were applied.

**Verdict:** Phase 2.5 (clamp revert + warmup) successfully revived the dead
sigmoid. But resnet1d now needs the same q-head regularization the CNN trunks
got — it's overfitting massively. Next step: add `per_head: {q: {hidden_units:
32, dropout: 0.3, l2: 1e-4}}` and re-clamp q to 1.0 (keeping the warmup).

### 2. cnn_baseline — 🟢 CLEAR WIN (val_r2 doubled)

**Q-head regularization + oversampling: val_r2 doubled, decline flattened.**

| Metric | Old (clamp only) | New (+reg +oversample) | Delta |
|--------|-----------------|----------------------|-------|
| train_r2_q | 0.628 | **0.417** | −0.211 |
| val_r2_q | 0.075 | **+0.151** | +0.076 (+101%) |
| val_std_ratio_q | 0.653 | 0.531 | −0.122 |
| q MAE (phys) | 0.165 | **0.161** | −0.004 |

**Trajectory:** The old run showed a slow steady decline from peak 0.193 (epoch
3) to 0.075 (epoch 79). The new run peaks at 0.196 (epoch 18) and then settles
to a flat plateau at ~0.151 from epoch 45 onward — **no decline**. The
regularization successfully flattened the overfitting-driven decline.

The train/val gap narrowed from 0.55 (0.628−0.075) to 0.27 (0.417−0.151).
Regularization cut the overfitting in half while improving validation.

**Verdict:** Phase 2.5 (q-head reg) + Phase 3.1 (oversample) worked as intended
for cnn_baseline. The regularization generalized from cnn_attention to
cnn_baseline — confirming this is a trunk-agnostic fix for q overfitting.

### 3. cnn_attention — 🟡 MODEST REGRESSION

**q_tokens branch + oversample: slight decline across all metrics.**

| Metric | Old (reg+clamp) | New (+branch +oversample) | Delta |
|--------|----------------|--------------------------|-------|
| train_r2_q | 0.399 | **0.300** | −0.099 |
| val_r2_q | 0.204 | **0.180** | −0.024 |
| val_std_ratio_q | 0.602 | 0.543 | −0.059 |
| peak val_r2 | 0.244 (ep 16) | 0.218 (ep 23) | −0.026 |

**Analysis:** Both train and val R² declined slightly. The q_tokens branch
doesn't appear to help — val_r2 is slightly worse, not better. The GAP-pooled
per-token features may not carry more q-relevant information than the
attention-pooled features. Or the oversampling may be interacting negatively —
it's possible that duplicating hard-regime examples makes the model work harder
to fit them (lower train R²) without improving generalization (lower val R²).

The branch infrastructure is correct and the oversampling is the same as other
models — this is likely a real null result for the q_tokens hypothesis on this
architecture.

**Verdict:** The q_tokens branch didn't help cnn_attention. The model was
already at a stable plateau (val_r2 ~0.20) from clamp+regularization alone.
The decline is small enough that it could be run-to-run noise, but the
directional signal (both train and val down) suggests the branch and/or
oversampling made things slightly harder, not easier.

### 4. inception_time — 🟡 SLIGHT REGRESSION

**Oversample only: small decline, similar to cnn_attention.**

| Metric | Old (clamp only) | New (+oversample) | Delta |
|--------|-----------------|-------------------|-------|
| train_r2_q | 0.395 | **0.342** | −0.053 |
| val_r2_q | 0.219 | **0.197** | −0.022 |
| q MAE (phys) | 0.153 | 0.153 | 0.000 |

Same pattern as cnn_attention — both train and val slightly down, physical MAE
unchanged. The oversampling alone (without any other change) doesn't improve q.

**⚠️ InceptionTime merger_time is still dead:**
- train_r2_merger_time = 0.005, val_r2 = −0.002
- Physical MAE = 0.049 s (25% of the 1.6–1.8 s range)
- weight_merger_time is at exp(3.0) ceiling — not a loss-weighting issue
- This is a separate architectural bug, not addressed by any Phase 2.5/3 change

### 5. tcn — 🟢 STABLE (still the best model)

**Oversample only: essentially unchanged, still best for q.**

| Metric | Old (clamp only) | New (+oversample) | Delta |
|--------|-----------------|-------------------|-------|
| train_r2_q | 0.266 | **0.192** | −0.074 |
| val_r2_q | 0.260 | **0.252** | −0.008 |
| train/val gap | 0.006 | **0.060** (inverted!) | — |
| q MAE (phys) | 0.150 | 0.151 | +0.001 |

Train R² dropped more than val R², which is interesting — the oversampling made
it harder to fit q on training data (more hard examples) without hurting
validation. The gap inverted (train < val), which is unusual and suggests the
oversampled hard examples are dragging down the training metrics without
affecting generalization.

TCN remains the best model overall — best mchirp MAE (1.00), best q MAE
(0.151), best snr MAE (0.83). weight_snr = 12.68 at epoch 79 (still rising,
hasn't hit the exp(3)=20.09 ceiling yet).

---

## q_high × mchirp_low subset — did oversampling help?

| Model | Old r2_q | New r2_q | Old mae_q | New mae_q | Change |
|-------|---------|---------|-----------|-----------|--------|
| cnn_attention | −22.06 | **−16.94** | 0.295 | **0.256** | ✅ −13% MAE |
| cnn_baseline | −22.31 | **−16.50** | 0.286 | **0.252** | ✅ −12% MAE |
| resnet1d | −127.0 | **−21.55** | 0.745 | **0.282** | ✅ −62% MAE |
| inception_time | −21.67 | **−17.42** | 0.295 | **0.260** | ✅ −12% MAE |
| tcn | −22.01 | **−15.21** | 0.299 | **0.246** | ✅ −18% MAE |

**The oversampling IMPROVED the hardest cell across ALL five models.** MAE
dropped 12–18% on the CNN/Inception/TCN trunks, and 62% on resnet1d (but
resnet1d's old number was catastrophic because the sigmoid was dead — the
improvement is mostly from the clamp fix, not the oversampling).

The R² numbers are still deeply negative (−15 to −22), but the MAE reductions
are real and consistent across all trunks. This confirms the data-scarcity
hypothesis: giving the model more examples in the hard regime helps it learn
that regime better.

**However**, the full-set val_r2_q didn't improve proportionally — the
oversampling helps the targeted cell but the overall metric is dominated by
the easier regimes where the model already performs adequately.

---

## Weight trajectories — sanity check

| Model | weight_q behavior | Expected? |
|-------|------------------|-----------|
| cnn_attention | Hits 2.72 at epoch 6, stays there | ✅ clamp=1.0 ceiling |
| cnn_baseline | Hits 2.72 at epoch ~7, stays there | ✅ clamp=1.0 ceiling |
| resnet1d | Ramps 1.02→20.09 by epoch 17, stays there | ✅ clamp=3.0 ceiling, no longer starved |
| inception_time | Hits 2.72 at epoch ~7, stays there | ✅ clamp=1.0 ceiling |
| tcn | Hits 2.72 at epoch ~7, stays there | ✅ clamp=1.0 ceiling |

All weight trajectories match expectations. resnet1d's weight_q now reaches the
full exp(3.0)=20.09 ceiling (previously locked at 2.72). TCN's weight_snr is
still converging (12.68 at epoch 79) — it may benefit from more epochs.

---

## What worked (clear wins)

1. **resnet1d dead sigmoid revived** — clamp revert (1.0→3.0) + warmup (5
   epochs) resurrected q from val_r2=−5.74 to +0.100. val_std_ratio_q went
   from 0.0 (constant predictions) to 0.68 (meaningful variance). q MAE in
   physical units dropped 59% (0.395→0.162).

2. **cnn_baseline regularization generalized** — q-head regularization
   (hidden=32, dropout=0.3, L2=1e-4) doubled val_r2 (0.075→0.151) and
   flattened the decline. Train/val gap halved (0.55→0.27). This confirms
   the regularization approach works across trunk architectures.

3. **Oversampling helps the hardest cell** — MAE in q_high×mchirp_low dropped
   12–18% across all models. The data-scarcity problem is real and responds to
   more examples in that regime.

## What didn't work (null results)

4. **q_tokens branch for cnn_attention** — GAP-pooled per-token transformer
   features didn't improve q over the attention-pooled features. Small decline
   in both train and val R². The hypothesis that mass-ratio information is
   washed out by learned attention pooling is not supported.

5. **Oversampling alone doesn't improve full-set q** — inception_time and tcn
   got only oversampling and showed slight declines in val_r2_q. The
   oversampling helps the targeted cell but not enough to move the overall
   metric.

## What's still broken

6. **resnet1d now overfits q massively** — train_r2=0.999, val_r2=0.100. Gap
   of 0.90. This is the pre-fix cnn_attention failure mode. Needs q-head
   regularization + re-clamp.

7. **InceptionTime merger_time is dead** — train_r2=0.005, val_r2=−0.002.
   Full loss weight (exp(3.0)=20.09). Not a q problem, not addressed by any
   fix. Needs separate diagnosis (epoch-0 logit check).

---

## Recommended next steps

1. **resnet1d: add q-head reg + re-clamp to 1.0 (keep warmup).** The dead
   sigmoid is fixed. Now apply the same regularization + clamp combination
   that worked for cnn_attention and cnn_baseline:
   ```yaml
   head_cfg:
     per_head:
       q:
         hidden_units: 32
         dropout: 0.3
         l2: 1.0e-4
   loss:
     log_var_clamp:
       default: 3.0
       q: 1.0
   ```
   Expected: train_r2 drops from 0.999→~0.4-0.6, val_r2 rises from 0.100→~0.15-0.20.

2. **Revert cnn_attention's q_tokens branch** — it didn't help. Keep the
   oversampling (helped the hard cell, didn't hurt overall). Optionally
   remove `branch: q_tokens` from the config (or just set it to null) to
   fall back to the global pooled features.

3. **Investigate InceptionTime merger_time** — epoch-0 logit histogram, same
   protocol as resnet1d's q. If dead sigmoid: try warmup. If not: check
   whether InceptionTime's parallel-kernel architecture systematically fails
   to encode merger-time information.

4. **Consider raising oversample factor to 3 or 4** — the 2× oversampling
   helped the hard cell but didn't move the overall val_r2. A stronger dose
   might push it over the threshold. Or try oversampling different cells
   (e.g., `snr_low: 2` to help low-SNR examples).

5. **TCN: consider more epochs** — weight_snr still converging at 12.68 (cap
   is 20.09). The model hasn't finished optimizing. Running to 120-150 epochs
   might squeeze out a bit more performance.
