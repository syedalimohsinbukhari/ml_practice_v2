# q-Head Diagnosis & Fix Plan

Context: `cnn_baseline` underfits q (train R2 ~0.7-0.85, val R2 ~0). `cnn_attention`
overfits q (train R2 ~0.95-1.0, val R2 declines to ~ -0.2). Loss weights for all
four heads saturate identically at ~20.09 regardless of per-head performance.
`heads_spec.py` review ruled out a transform/loss mismatch but surfaced a new,
q-specific saturation hypothesis.

**Correction after Phase 2 attempt #1 (2026-07-14 10:41/10:45 runs):** the
"`cnn_baseline` underfits q" premise above does not hold in the actual
history.csv data, old run or new. `cnn_baseline`'s original run
(`20260714_091740`) ends at `r2_q=0.992` (train) / `val_r2_q=-0.164` — it was
*already* overfitting q just as badly as `cnn_attention`, not underfitting it.
See "Phase 2 attempt #1 results" below — this reframes the whole diagnosis:
the weight-clamp saturation and q collapse are not `cnn_attention`-capacity-
specific, they show up almost identically in the much smaller `cnn_baseline`
trunk too.

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

Full numbers for everything below: `q_head_run_comparison.md`.

8. ⏳ **Loosen the weight clamp per-head — implemented, NOT actually tested
   yet.** `MultiHeadTrainer` accepts `loss.log_var_clamp` as either the old
   scalar or a dict (`{default: 3.0, q: 1.0}`), and this was set in
   `configs/cnn_attention.yaml` — but the config was reverted back to the
   flat `log_var_clamp: 3.0` before the run in
   `runs/cnn_attention/20260714_104523/`. Confirmed from that run's
   `history.csv`: `weight_q` still saturates at the identical `exp(3.0) =
   20.0855` ceiling as every other head, same as every prior run, including
   `cnn_baseline` (which never had any clamp change applied at all). **This
   is still the one untested lever from the original plan** — re-apply the
   per-head clamp (to both configs, see step 9 below) and re-run.
9. ✅ **Regularized the q head specifically — tested, real partial
   improvement.** `cnn_attention.yaml`'s q head (32 hidden units, dropout
   0.3, L2 1e-4) measurably reduced memorization: `r2_q` (train) at epoch 79
   dropped from 0.906 (old run) to 0.807 (new run), and the collapse in
   `val_r2_q` roughly halved (−0.179 → −0.090). Peak `val_r2_q` (~0.21-0.23
   around epoch 10-20) was unchanged — the fix slows the *decline after* the
   peak, it doesn't raise the peak itself. Still net negative at the end, so
   not sufficient alone. **`cnn_baseline` did not get this treatment and
   should, per the correction above** — it's overfitting q just as badly and
   is a legitimate second trunk to test the same regularization on.
10. ✅ **Addressed sigmoid saturation directly (first option: widened
    bounds) — tested, small improvement in every run.** `q`'s `UNIT_AFFINE`
    bounds moved from `(0.2, 1.0)` to `(0.15, 1.05)` in `heads_spec.py`
    (applies globally, so it affected both new runs). `cnn_baseline`, which
    got *no other change*, still improved slightly from this alone:
    `val_r2_q` at epoch 79 went from −0.164 (old) to −0.125 (new). Small but
    real, and free — keep it.
11. 🔲 **Input-level augmentation targeted at the hard regime** (jitter,
    noise, SNR variation on low-mchirp/high-q examples) — not attempted;
    now backed by a quantified target (see step 12).
12. ✅ **Re-ran both `cnn_attention` and `cnn_baseline`; checked train/val
    R2(q) convergence and the new subset diagnostics.** Results: neither run
    reaches positive `val_r2_q` — both still finish negative (−0.090 and
    −0.125). The q-tercile × mchirp cross-tab (new this round) nails down
    the worst regime precisely: `q_high × mchirp_low` (near-equal-mass, low
    chirp mass, 544/5000 = ~11% of validation, matching its ~11% share of
    training) has MAE ~60% above the full-set average and by far the worst
    R2 in both trunks (`cnn_attention` −26.7, `cnn_baseline` −23.0) — this is
    now a measured, not hypothesized, hardest-and-scarcest region. See
    `q_head_run_comparison.md` for the full breakdown.

**Net read on Phase 2 attempt #1:** widened bounds (step 10) help a little
everywhere; q-head regularization (step 9) helps more but isn't sufficient
alone; the highest-leverage lever from the original diagnosis — the per-head
clamp (step 8) — was never actually run. And the premise that this is a
`cnn_attention`-specific overfitting problem doesn't hold: `cnn_baseline`
overfits q just as badly with no regularization at all, which points more
toward the uncertainty-weighting/clamp mechanism itself (or the data-scarce
`q_high × mchirp_low` regime) than toward trunk capacity.

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

Phase 2 attempt #1 (bounds + q-head regularization on `cnn_attention` only)
gave real but insufficient improvement — both runs still finish with
negative `val_r2_q`. Before reaching for Phase 3:

1. ✅ **Per-head `log_var_clamp` (`default: 3.0, q: 1.0`) is now applied to
   all five trunk configs** (`cnn_attention`, `cnn_baseline`, `resnet1d`,
   `inception_time`, `tcn`) — not just `cnn_attention`, and this time it's
   actually in the configs to be run. Previously the fix was written once,
   reverted before the run, and never applied to any trunk besides
   `cnn_attention`.
2. 🔲 **New data point: `resnet1d` is reportedly collapsing q to ~the lower
   bound for ~all true values** — a qualitatively different symptom from the
   cnn trunks' train/val overfit gap (which is q tracking train targets too
   well, not failing to track them at all). The tighter clamp *raises* q's
   weight floor (exp(-1)≈37% vs exp(-3)≈5%), which should make it harder for
   a collapsing head to get starved of gradient pressure — worth testing
   whether that alone fixes it. If it doesn't, `logits_train_vs_val.png`
   (Phase 1 step 3, already wired into `scripts/evaluate.py`) should show a
   pre-sigmoid logit stuck at a large negative value on both splits — a
   saturated/dead sigmoid unit, which a loss-weighting fix can't repair
   (gradient through `sigmoid'(logit)≈0` stays ≈0 regardless of loss weight)
   and would need a different fix (LR warmup, better init, or `bounded:
   false` as a diagnostic A/B).
3. **`cnn_baseline` did not get the q-head regularization `cnn_attention`
   got** (smaller hidden/dropout/L2) — it overfits q just as badly and is
   the cleanest control for whether that fix generalizes across trunks.
   `resnet1d`/`inception_time`/`tcn` haven't been characterized enough yet to
   know if they need it too (resnet's issue in particular looks different in
   kind, see #2) — don't add it there blind.
4. **Re-run all five configs**, then diff against the existing runs the same
   way `q_head_run_comparison.md` did — `r2_q`/`val_r2_q` final + trajectory,
   `weight_q` saturation, and the `q_high × mchirp_low` subset cell.
5. `python scripts/evaluate.py configs/<name>.yaml` against the new
   checkpoints to finally exercise `logits_train_vs_val.png` (Phase 1 step 3)
   and `residuals_mchirp_*.png` (Phase 1 step 6) — neither has been run yet,
   and #2 above makes the logit plot the priority for `resnet1d` specifically.
6. If `val_r2_q` is still negative after 1-5 on the cnn trunks: the
   `q_high × mchirp_low` cell (Phase 2 step 12 finding) is now a measured,
   not hypothesized, target — move to Phase 3 step 14 (targeted
   oversampling/augmentation for that specific cell) next.
7. `pytest -m "not slow"` on the lab machine at some point to confirm the
   new code (per-head clamp, per-head regularization, subset diagnostics,
   logit/residual plots) hasn't regressed anything — still not run.
