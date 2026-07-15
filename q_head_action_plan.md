# q-Head Diagnosis & Fix Plan

> **2026-07-14: q shelved.** The default head set has been switched from
> `(mchirp, q, merger_time, snr)` to `(mchirp, merger_time, snr, ra,
> declination, coa_phase)`. q remains in the registry and can be re-added
> via config. The work below brought val_r2_q from −0.18 to +0.26 (TCN) and
> revived resnet1d's dead sigmoid — a ~6 R²-unit improvement. The remaining
> gap (q_high×mchirp_low cell data scarcity) requires dataset-level fixes
> beyond the current scope. See `phase2_5_3_run_results.md` for final numbers.

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
lab-machine run to verify · ❌ tried, made things worse → reversed strategy

---

## Phase 2 attempt #2 — Full five-model run with per-head clamp (2026-07-14)

All five configs trained with `log_var_clamp: {q: 1.0}` (default 3.0 for others).
`evaluate.py` was run against every checkpoint. The results reverse the original
diagnosis.

### Headline

**The per-head clamp fix WORKED for 4/5 models.** For the first time in any run,
`val_r2_q` stays **positive** through epoch 79 on `cnn_attention`, `cnn_baseline`,
`inception_time`, and `tcn`. The crash-to-negative pattern (old: val_r2_q → −0.09
to −0.18) is gone. The mechanism is the opposite of what was hypothesized: lowering
q's clamp *reduces* q's gradient share (weight 2.72 vs 20.09 for other heads),
which *reduces* train overfitting without starving q completely. The model still
learns q — just more slowly and without the overfit-then-crash trajectory.

**`resnet1d` is the sole failure** — complete collapse (val_std_ratio_q = 0.0),
never positive val_r2_q. This is a dead-sigmoid problem, not a loss-weighting one;
the tighter clamp makes it worse by further reducing gradient to an already-dead
head.

### Before/after table

| model | clamp fix? | q-head reg? | train_r2_q | val_r2_q (final) | val_r2_q (peak) | weight_q |
|-------|-----------|------------|-----------|-------------------|------------------|----------|
| cnn_attention OLD | no (3.0) | yes | 0.807 | **−0.090** | 0.237 (ep 19) | 20.09 |
| cnn_attention NEW | yes (1.0) | yes | 0.399 | **+0.204** | 0.244 (ep 16) | 2.72 |
| cnn_baseline OLD | no (3.0) | no | 0.959 | **−0.125** | 0.146 (ep 6) | 20.09 |
| cnn_baseline NEW | yes (1.0) | no | 0.628 | **+0.075** | 0.193 (ep 3) | 2.72 |
| resnet1d OLD | no (3.0) | no | −4.51 | −4.66 | never positive | 5.17 |
| resnet1d NEW | yes (1.0) | no | −5.56 | −5.74 | never positive | 2.72 |
| inception_time | yes (1.0) | no | 0.395 | **+0.219** | 0.238 (ep 28) | 2.72 |
| tcn | yes (1.0) | no | 0.266 | **+0.260** | 0.264 (ep 64) | 2.72 |

Physical-unit q MAE (from `evaluate.py`):

| model | q MAE | q RMSE |
|-------|-------|--------|
| cnn_attention | 0.153 | 0.183 |
| cnn_baseline | 0.165 | 0.203 |
| resnet1d | **0.395** | 0.447 |
| inception_time | 0.153 | 0.185 |
| tcn | **0.150** | 0.181 |

### Per-model trajectory notes

- **cnn_attention:** val_r2_q peaks at 0.244 (epoch 16), then slowly settles to a
  flat plateau ~0.20 from epoch 35 onward. No decline after the plateau — the
  plateau is the new steady state. Train/val gap: 0.40 vs 0.20 — a real gap, but
  stable. The q-head regularization (hidden_units=32 + dropout=0.3 + L2=1e-4) on
  top of the clamp helps maintain this stability; it's the only model with both
  fixes.
- **cnn_baseline:** val_r2_q peaks very early at 0.193 (epoch 3), then slowly
  declines to 0.075 by epoch 79. Still positive! But the slow decline suggests
  overfitting continues at a reduced rate. This model did NOT get q-head
  regularization — adding it would likely flatten the decline, same as it did for
  cnn_attention.
- **resnet1d:** immediately collapses at epoch 0 (val_r2_q = −3.5), drops further
  to −5.7 at epoch 20. val_std_ratio_q = 0.0 — predictions have zero variance.
  The old run without the clamp had weight_q = 5.17 (still rising — hadn't
  saturated), suggesting resnet1d was still *trying* to learn q. The new clamp
  locks q's weight at 2.72, starving it further. **resnet1d's problem is not
  overfitting — it's a dead sigmoid that can't learn q at all.** The clamp fix
  is counterproductive here.
- **inception_time:** val_r2_q peaks at 0.238 (epoch 28), then very gentle
  plateau to ~0.22. Nearly identical trajectory to cnn_attention but without
  q-head regularization — suggesting inception_time's architecture is naturally
  less prone to q overfitting than cnn_attention's.
- **tcn:** best model for q by every metric. Train/val gap = 0.006 — essentially
  zero overfitting. val_r2_q steadily rises and plateaus at ~0.26 from epoch 40
  onward, still improving slightly at epoch 79. Also best overall across all
  heads (mchirp MAE=0.97, snr MAE=0.82). The TCN's dilated convolutions seem
  naturally well-suited to this parameter-estimation task.

### What the clamp fix actually does (corrected mechanism)

The original hypothesis: "q overfits because its loss weight hits the same
exp(3.0) ceiling as every other head → cap q's ceiling lower → less overfitting."

What actually happens: lowering q's ceiling from exp(3.0)=20.09 to exp(1.0)=2.72
means q gets ~7.4× less gradient share in the trunk compared to other heads.
This **reduces** q's train R² (less capacity devoted to memorizing q), which
paradoxically **improves** val R² because the model can no longer overfit q
enough to crash generalization. The model still learns q — 2.72× the base loss
weight is still meaningful gradient signal — but it can't drive train loss on q
low enough to memorize spurious q-correlated features.

The key insight: **the original weight ceiling (exp(3.0)=20.09) was TOO HIGH,
allowing overfitting; the new ceiling (exp(1.0)=2.72) is in the right ballpark
for preventing overfitting while still allowing learning.** The floor also rises
(exp(−1)=0.368 vs exp(−3)=0.050), which helps prevent starvation of a
struggling head — except for resnet1d where even 37% weight isn't enough to
revive a dead sigmoid.

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
3. ✅ **Pre-sigmoid logits, train vs val, split by true-q tercile.**
   Implemented `sigmoid_logit_hist` in `evaluation/plots.py` and wired into
   `scripts/evaluate.py` (`logits_train_vs_val.png`). Produced for all five
   models in the 2026-07-14 11:33–12:15 runs. For resnet1d, this is the key
   diagnostic to confirm the dead-sigmoid hypothesis (logits stuck at large
   negative values on both splits).
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
6. ✅ **Residuals vs mchirp/SNR, all trunks.** Generalized
   `residuals_vs_snr` → `residuals_vs_param` and wired an mchirp variant into
   `scripts/evaluate.py` (`residuals_mchirp_<split>.png`). Produced for all
   five models.
7. ✅ **q training-set distribution.** Histogrammed: density rises
   smoothly from ~1.5%/bin near q=0.2 to ~7.6%/bin near q=1.0 — no severe
   internal gap. The real scarcity is the *joint* mchirp_low × q_high cell
   (2691/25000, the smallest of the four quadrants from step 2) — the
   physically hardest region is also the most data-starved one.

---

## Phase 2 — Fix regularization & loss/weighting (highest expected payoff)

Full numbers for everything below: `q_head_run_comparison.md`.

8. ✅ **Per-head log-var clamp — tested on all five models, working, but effect
   is the opposite of what was hypothesized.** Lowering q's clamp from 3.0→1.0
   *reduces* q's gradient share (weight ceiling drops from exp(3)=20.09 to
   exp(1)=2.72), which *reduces* train overfitting. This turned val_r2_q
   positive for 4/5 models (first time ever). The original hypothesis was
   backward: q wasn't overfitting *because* of its high weight — it was
   overfitting *enabled* by the high weight ceiling, and lowering it removed
   the capacity to memorize q-specific noise. See "Phase 2 attempt #2" above
   for the full before/after table.
   - **However:** the `default: 3.0` key is missing from all five config YAMLs
     (they only have `{q: 1.0}`). The code's fallback makes it work, but the
     configs should be explicit about the default for readability.
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
12. ✅ **Re-ran all five trunk configs; evaluated every checkpoint.** Results:
    For the first time, 4/5 models maintain positive `val_r2_q` through epoch
    79. TCN is the star (val_r2_q=0.260, train/val gap=0.006). Only resnet1d
    fails (complete collapse, dead sigmoid). See "Phase 2 attempt #2" above
    for full analysis. `evaluate.py` produced `logits_train_vs_val.png`,
    `residuals_mchirp_validation.png`, and `metrics_validation.csv` for all
    five models.

**Net read on Phase 2 attempt #1 (2026-07-14 10:41/10:45, NO clamp fix):**
widened bounds (step 10) help a little everywhere; q-head regularization
(step 9) helps more but isn't sufficient alone; the highest-leverage lever
from the original diagnosis — the per-head clamp (step 8) — was never
actually run in this attempt. And the premise that this is a
`cnn_attention`-specific overfitting problem doesn't hold: `cnn_baseline`
overfits q just as badly with no regularization at all, which points more
toward the uncertainty-weighting/clamp mechanism itself than toward trunk
capacity.

**Net read on Phase 2 attempt #2 (2026-07-14 11:33–12:15, WITH clamp fix):**
the per-head clamp (`q: 1.0`) works — 4/5 models maintain positive val_r2_q
for the first time. The mechanism is the opposite of what was hypothesized
(lowering q's ceiling *reduces* gradient share, preventing overfitting rather
than preventing differentiation loss). TCN is the best model by every metric.
resnet1d is the only failure and its problem is a dead sigmoid, not
overfitting. See the before/after table in "Phase 2 attempt #2" above.

---

## Phase 3 — Only if Phase 2 plateaus

13. ✅ **Dedicated q branch for cnn_attention (Phase 3.2).** `cnn_attention`'s
    trunk now returns per-token transformer features `(B, 128, 128)` as
    `extra_features={"q_tokens": ...}`. `attach_heads` accepts an
    `extra_features` dict; heads with `per_head.<name>.branch: q_tokens` in
    their config connect to the GAP-pooled token features instead of the
    global attention-pooled features. `cnn_attention.yaml` enables this for
    q. Other trunks are unaffected (still return `(inputs, features)`).
14. ✅ **Targeted oversampling (Phase 3.1).** `build_subset_masks()` extracted
    from `DiagnosticSubsetsCallback` into `loader.py` as a shared utility.
    `run_experiment` reads `data.augmentation.oversample` and duplicates
    target-subset rows before `TargetTransforms.fit()`. All five configs have
    `q_high_mchirp_low: 2`. Config-driven, zero-architecture-change.
15. 🔲 **Ensemble / auxiliary-task approach** — deferred. Highest complexity,
    only worth it if 13+14 don't close the gap.

---

## What's left to do

Phase 2 attempt #2 (per-head clamp on all five trunks) succeeded where attempt
#1 failed — 4/5 models now maintain positive `val_r2_q` throughout training.
The crash-to-negative pattern is eliminated for cnn_attention, cnn_baseline,
inception_time, and tcn. Only resnet1d remains broken, and its failure mode is
qualitatively different (dead sigmoid, not overfitting).

### Immediate next steps

1. ✅ **resnet1d dead sigmoid fixes (Phase 2.5).** `log_var_clamp` reverted
   to flat `3.0` (q gets equal gradient budget). `warmup_epochs: 5` added
   (linear LR ramp to avoid early sigmoid saturation). `heads.py` now
   supports configurable `sigmoid_bias` per-head for nudging initialization
   away from the asymptote. Needs lab-machine run to verify.
2. ✅ **cnn_baseline q-head reg (Phase 2.5).** `per_head: {q: {hidden_units:
   32, dropout: 0.3, l2: 1e-4}}` added — same as cnn_attention's proven
   regularization. Needs lab-machine run to verify the decline flattens.
3. ✅ **Targeted oversampling (Phase 3.1).** `build_subset_masks()` extracted
   to `loader.py`. `data.augmentation.oversample.q_high_mchirp_low: 2` on
   all five configs. `run_experiment` duplicates the hard-cell rows before
   fitting transforms.
4. ✅ **Dedicated q branch for cnn_attention (Phase 3.2).** Trunk returns
   per-token transformer features. `attach_heads` supports `extra_features`
   dict with per-head `branch` config. `cnn_attention.yaml` enables
   `branch: q_tokens` for q.
5. 🔲 `pytest -m "not slow"` on the lab machine — still not run.
6. 🔲 **Re-run all five configs on the lab GPU** to measure the combined
   effect of Phase 2.5 + 3.1 + 3.2. Focus metrics: val_r2_q trajectory,
   q_high×mchirp_low cell R² in diagnostics.csv, resnet1d val_std_ratio_q
   (must be > 0).
7. 🔲 **Clamp sweep** (optional): try q clamp values of 1.5/2.0 on tcn and
   cnn_baseline to find the optimal per-head gradient budget.
8. 🔲 **Phase 3.3 auxiliary classifier** (deferred): only if 3.1+3.2 plateau.

### Strategic correction

The original plan (Phase 2 steps 8-9) framed the problem as "q's loss weight
saturates at the same ceiling as other heads, destroying per-head
differentiation." The attempted fix was to lower q's ceiling. **This worked,
but for the opposite reason:** lowering the ceiling *reduces* q's gradient
share, which *prevents* overfitting. The model was never "failing to
differentiate q from other heads" — it was devoting too much capacity to
memorizing q-specific noise patterns in the training set.

Going forward, the uncertainty-weighting clamp should be thought of as a
**per-head gradient budget**, not a differentiation mechanism:
- Heads that overfit (q) → tighter clamp (lower ceiling) = smaller budget
- Heads that collapse (resnet1d's q) → looser clamp (higher ceiling) = larger
  budget, or no clamp at all
- The *floor* (exp(−clamp)) matters for collapse prevention; the *ceiling*
  (exp(+clamp)) matters for overfitting prevention

### TCN as a new baseline

TCN with just the clamp fix (no q-head reg) achieves:
- val_r2_q = 0.260, train/val gap = 0.006 (essentially zero overfitting)
- Best overall metrics: mchirp MAE=0.97, q MAE=0.150, snr MAE=0.82
- Still improving slightly at epoch 79 (weight_snr=11.93 hasn't saturated)
- The dilated-convolution architecture seems naturally resistant to the
  q-overfitting problem that plagues the CNN trunks

This makes TCN the new performance baseline for q — any fix targeting the
other trunks should aim to match or exceed TCN's train/val gap.
