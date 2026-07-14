# q-Head Diagnosis & Fix Plan

Context: `cnn_baseline` underfits q (train R2 ~0.7-0.85, val R2 ~0). `cnn_attention`
overfits q (train R2 ~0.95-1.0, val R2 declines to ~ -0.2). Loss weights for all
four heads saturate identically at ~20.09 regardless of per-head performance.
`heads_spec.py` review ruled out a transform/loss mismatch but surfaced a new,
q-specific saturation hypothesis.

Status legend: ✅ resolved/implemented · 🔲 still open · ⏳ implemented, awaiting a
lab-machine run to verify

---

## Phase 0 — Resolved by heads_spec.py review

- ✅ **Transform/loss binding is not the bug.** `q` uses `UNIT_AFFINE` (bounds
  0.2-1.0) + sigmoid + Huber — identical recipe to `merger_time`, which
  generalizes well in both runs. Rules out "wrong transform paired with wrong
  loss" as the root cause.
- ✅ **Periodicity/wraparound is not a factor.** q is not a periodic parameter,
  no sin/cos handling needed or missing.
- ✅ **Sigmoid saturation near q→1 confirmed as a real, testable mechanism**
  (see Phase 1 step 1) and fixed (Phase 2 step 10).

---

## Phase 1 — Diagnose precisely (no retraining, cheap)

1. ✅ **Verify UNIT_AFFINE bounds against real data.** Pulled min/max of the
   raw q column: training `[0.2027, 0.99998]`, validation `[0.2052, 0.99995]`
   — both comfortably inside the old `(0.2, 1.0)` bounds, so **no clipping
   bug**. But 7.6% of training samples (1903/25000) sit in true-q ∈
   `[0.95, 1.0)`, which maps to transformed-space `t ≈ 0.94-1.0` — right at
   the sigmoid's saturating asymptote where gradients vanish. Fixed in Phase
   2 step 10 by padding the bounds.
2. ✅ **q-magnitude subset split, cross-tabbed with mchirp.** Implemented in
   `DiagnosticSubsetsCallback._build_subsets` (`q_low`/`q_mid`/`q_high`
   terciles + `q_{low,high}_mchirp_{low,high}`), so every future run logs
   this per epoch. A one-off check on the training split already found a
   **real confound in the data**: q_high pairs with mchirp_high 2.1x more
   often than mchirp_low (5643 vs 2691), and q_low pairs with mchirp_low 3x
   more often than mchirp_high (6277 vs 2056). The "mchirp_low weakness" seen
   in `cnn_attention` and "q is hard near equal-mass" are not independent
   hypotheses in this dataset — they're entangled by construction/sampling.
3. ⏳ **Pre-sigmoid logits, train vs val, split by true-q tercile.**
   Implemented `sigmoid_logit_hist` in `evaluation/plots.py` — recovers the
   logit algebraically (`ln(p/(1-p))` on the raw sigmoid output, no
   architecture change) and wired it into `scripts/evaluate.py`
   (`logits_train_vs_val.png`). Needs a lab-machine run against a checkpoint
   to actually confirm the extreme-train/compressed-val signature.
4. ✅ **Loss-weight clamp — root cause confirmed and fixed.** From
   `runs/cnn_attention/20260714_093319/history.csv`: all four `weight_*`
   columns (train *and* val) converge to the *exact* same value,
   20.085529... = `exp(3.0)`, starting epoch ~17-19 — hitting
   `log_var_clamp: 3.0` identically. Mechanism: the uncertainty-weighting
   optimum is `s_h* = log(L_h)` on *training* loss, so as `cnn_attention`
   (much higher capacity than `cnn_baseline`) drives every head's train loss
   down through overfitting, every head's optimal log-var keeps decreasing
   past -3 and clamps identically — destroying the per-head differentiation
   the scheme exists to provide. Critically, this saturation point (~epoch
   17-20) lines up almost exactly with when `val_r2_q` peaks (~0.23 around
   epoch 9-12) and begins its collapse into negative territory, while
   `r2_q` (train) keeps climbing past 0.9. Fixed in Phase 2 step 8.
5. ✅ **Huber delta — confirmed shared/global, not the primary bug.** A
   single `keras.losses.Huber(delta=cfg["huber_delta"])` instance
   (`losses.py`) is reused for every head. For q (`UNIT_AFFINE`, range width
   0.8, now 0.9) delta=1.0 means Huber is essentially always in its
   quadratic regime — no robustness benefit — but `merger_time` uses the
   identical recipe and generalizes fine, so per Phase 0 this isn't the
   differentiator. Documented, not chased further.
6. ⏳ **Residuals vs mchirp/SNR, `cnn_attention`.** Generalized
   `residuals_vs_snr` → `residuals_vs_param` and wired an mchirp variant into
   `scripts/evaluate.py` (`residuals_mchirp_<split>.png`). Not yet run. One
   data point already in hand: `val_std_ratio_q` sits at ~0.87-0.88 at the
   end of the `cnn_attention` run — predictions are *not* collapsing to the
   mean — consistent with memorization/overfitting rather than a
   population-mean fallback.
7. ✅ **q training-set distribution.** Histogrammed: density rises
   smoothly from ~1.5%/bin near q=0.2 to ~7.6%/bin near q=1.0 — no severe
   internal gap. The real scarcity is the *joint* mchirp_low × q_high cell
   (2691/25000, the smallest of the four quadrants from step 2) — the
   physically hardest region is also the most data-starved one.

---

## Phase 2 — Fix regularization & loss/weighting (highest expected payoff)

8. ✅ **Loosened the weight clamp per-head.** `MultiHeadTrainer` now accepts
   `loss.log_var_clamp` as either the old scalar or a dict
   (`{default: 3.0, q: 1.0}`); `configs/cnn_attention.yaml` sets q's clamp to
   1.0 so its uncertainty weight can't ride the same ceiling as heads whose
   train loss shrinks fastest.
9. ✅ **Regularized the q head specifically.** `attach_heads` now supports
   `head_cfg.per_head.<name>.{hidden_units,dropout,l2}`; `cnn_attention.yaml`
   gives q a smaller head (32 vs the shared 64 hidden units), dropout 0.3,
   and L2 1e-4 — capacity cut relative to the other three heads sharing the
   same trunk features.
10. ✅ **Addressed sigmoid saturation directly (first option: widened
    bounds).** `q`'s `UNIT_AFFINE` bounds moved from `(0.2, 1.0)` to
    `(0.15, 1.05)` in `heads_spec.py`, so q=1.0 now maps to `t≈0.944`,
    comfortably off the sigmoid's saturating edge.
11. 🔲 **Input-level augmentation targeted at the hard regime** (jitter,
    noise, SNR variation on low-mchirp/high-q examples) — not attempted;
    deferred pending step 12's result.
12. 🔲 **Re-run `cnn_attention` with these changes; re-check train/val R2(q)
    convergence.** All Phase 2 code is implemented and config-wired but
    **not yet run** — this needs the lab GPU machine, not the local T530.
    Target: val R2 climbs into positive territory and tracks train more
    closely, without train performance regressing toward baseline levels.

---

## Phase 3 — Only if Phase 2 plateaus

13. 🔲 **Give q a dedicated intermediate branch** off an earlier
    attention-block output rather than only the final pooled features — tests
    whether q needs different features, not just less capacity to overfit.
14. 🔲 **Targeted data augmentation/oversampling** in the mchirp_low / high-q
    regime — step 2 above already confirms this is a real, small (2691
    sample) cell, so this is worth prioritizing if step 12 plateaus.
15. 🔲 **Consider an ensemble or auxiliary-task approach for q** (e.g. wider
    uncertainty bands, or an auxiliary near-equal-mass classifier) if the
    problem is confirmed physical rather than a training artifact.

---

## What's left to do

Everything code-side for Phase 1 and Phase 2 is implemented (bounds widened,
per-head log-var clamp, per-head head regularization, q-tercile/mchirp
diagnostics, logit + mchirp-residual plotting) and covered by new/updated
tests (`tests/test_losses.py`, `tests/test_heads_spec.py`,
`tests/test_callbacks.py`). None of it has been exercised yet — that
requires the lab GPU machine:

1. `pytest -m "not slow"` (or the full suite) to confirm nothing regressed.
2. `python scripts/train.py configs/cnn_attention.yaml` — the actual Phase 2
   step 12 re-run.
3. `python scripts/evaluate.py configs/cnn_attention.yaml` against the new
   run to get `logits_train_vs_val.png` and `residuals_mchirp_*.png` (Phase 1
   steps 3 and 6, finally exercised).
