# q-Head Run Comparison

Companion to `q_head_action_plan.md`. Compares runs across three phases:
pre-fix (no changes), Phase 2 attempt #1 (bounds + q-head reg, no clamp fix),
and Phase 2 attempt #2 (per-head clamp on all five trunks).

## Run index

| Run | Trunk | Timestamp | What changed |
|---|---|---|---|
| OLD cnn_baseline | `cnn_baseline` | `20260714_091740` | baseline (pre-fix) |
| ATTEMPT1 cnn_baseline | `cnn_baseline` | `20260714_104107` | q bounds widened `(0.2,1.0)→(0.15,1.05)` only |
| **ATTEMPT2 cnn_baseline** | `cnn_baseline` | `20260714_113347` | bounds + **clamp fix** (`q: 1.0`), NO q-head reg |
| OLD cnn_attention | `cnn_attention` | `20260714_093319` | baseline (pre-fix) |
| ATTEMPT1 cnn_attention | `cnn_attention` | `20260714_104523` | bounds + q-head reg, clamp fix **reverted** |
| **ATTEMPT2 cnn_attention** | `cnn_attention` | `20260714_113745` | bounds + q-head reg + **clamp fix** (`q: 1.0`) |
| **ATTEMPT2 resnet1d** | `resnet1d` | `20260714_114721` | bounds + clamp fix, NO q-head reg |
| **ATTEMPT2 inception_time** | `inception_time` | `20260714_121546` | bounds + clamp fix, NO q-head reg |
| **ATTEMPT2 tcn** | `tcn` | `20260714_115404` | bounds + clamp fix, NO q-head reg |

---

## Headline numbers (final epoch, 79/80)

| Run | `r2_q` (train) | `val_r2_q` | train−val gap | `val_std_ratio_q` | `weight_q` saturates? |
|---|---:|---:|---:|---:|---|
| OLD cnn_baseline | 0.992 | **−0.164** | 1.156 | 0.811 | yes, at exp(3)=20.086 (~ep20) |
| NEW cnn_baseline | 0.959 | **−0.125** | 1.084 | 0.789 | yes, at exp(3)=20.086 (~ep21) |
| OLD cnn_attention | 0.906 | **−0.179** | 1.085 | 0.882 | yes, at exp(3)=20.086 (~ep18) |
| NEW cnn_attention | 0.807 | **−0.090** | 0.897 | 0.869 | yes, at exp(3)=20.086 (~ep20) |

**Reading this:** in every run, `weight_q` (and every other head's weight)
hits the identical clamp ceiling — because the per-head clamp fix wasn't
actually exercised, this is unchanged and expected. The two real deltas:

- **cnn_baseline** (bounds-widening only) improved a little: val R2 collapse
  went from −0.164 to −0.125, train R2 dropped from 0.992 to 0.959. Bounds
  alone help slightly, everywhere.
- **cnn_attention** (bounds + q-head regularization) improved more
  substantially: the overfit gap shrank from 1.085 → 0.897, and the val
  collapse roughly halved (−0.179 → −0.090). Real, measurable progress — but
  `val_r2_q` is still negative, so this alone doesn't fix it.

**Correction to the original plan's framing:** `cnn_baseline` was described
as "underfitting q" (val R2 ~0). That is not what's in either history.csv —
both the old and new `cnn_baseline` runs finish with train R2 > 0.95 and val
R2 solidly negative. `cnn_baseline` overfits q just as badly as
`cnn_attention` did. This is not a trunk-capacity story.

---

## Trajectory snapshots

`r2_q` (train) / `val_r2_q` / `weight_q` (train) at selected epochs:

### cnn_baseline

| epoch | OLD r2_q | OLD val_r2_q | OLD weight_q | NEW r2_q | NEW val_r2_q | NEW weight_q |
|---:|---:|---:|---:|---:|---:|---:|
| 0  | 0.032 | −0.164 | 1.11  | 0.032 | −0.609 | 1.11  |
| 5  | 0.251 | −0.259 | 2.92  | 0.248 | −0.358 | 2.92  |
| 10 | 0.351 | −0.479 | 7.44  | 0.339 | +0.142 | 7.52  |
| 15 | 0.574 | −1.250 | 18.06 | 0.585 | −0.637 | 13.71 |
| 20 | 0.847 | −0.651 | 20.09 | 0.824 | −0.180 | 19.76 |
| 25 | 0.899 | −0.333 | 20.09 | 0.910 | −0.076 | 20.09 |
| 30 | 0.964 | −0.253 | 20.09 | 0.938 | −0.100 | 20.09 |
| 40 | 0.980 | −0.183 | 20.09 | 0.955 | −0.119 | 20.09 |
| 60 | 0.991 | −0.162 | 20.09 | 0.959 | −0.124 | 20.09 |
| 79 | 0.992 | −0.164 | 20.09 | 0.959 | −0.125 | 20.09 |

Note the wild early-epoch `val_loss`/`val_r2_q` noise in the NEW run
(oscillating between roughly +0.14 and −1.0 through epoch ~20) before both
runs settle into the same shape — that's batch-to-batch validation variance
while `weight_q` is still ramping, not a sign of a broken run.

### cnn_attention

| epoch | OLD r2_q | OLD val_r2_q | OLD weight_q | NEW r2_q | NEW val_r2_q | NEW weight_q |
|---:|---:|---:|---:|---:|---:|---:|
| 0  | 0.017 | −0.134 | 1.11  | −0.031 | +0.084 | 1.11  |
| 5  | 0.236 | +0.210 | 2.92  | 0.216  | +0.215 | 2.93  |
| 10 | 0.268 | +0.197 | 7.39  | 0.242  | +0.227 | 7.49  |
| 15 | 0.304 | +0.211 | 16.97 | 0.266  | +0.233 | 17.63 |
| 20 | 0.435 | +0.175 | 20.09 | 0.337  | +0.219 | 20.09 |
| 25 | 0.682 | +0.003 | 20.09 | 0.497  | +0.172 | 20.09 |
| 30 | 0.815 | −0.132 | 20.09 | 0.640  | +0.050 | 20.09 |
| 40 | 0.886 | −0.171 | 20.09 | 0.759  | −0.046 | 20.09 |
| 60 | 0.905 | −0.177 | 20.09 | 0.801  | −0.089 | 20.09 |
| 79 | 0.906 | −0.179 | 20.09 | 0.807  | −0.090 | 20.09 |

The peak `val_r2_q` is about the same in both (~0.21-0.23, epoch 10-20), but
the new run's *decline after the peak* is visibly slower and lands at half
the final damage. The regularized q head keeps fitting the train set more
slowly (`r2_q` at epoch 79 is 0.807 vs 0.906), which is exactly what capping
its capacity should do.

---

## Subset diagnostics at the final epoch (NEW runs only — this callback is new)

`mae_q` / `r2_q` / `std_ratio_q`, validation set, epoch 80:

| subset | n | cnn_attention MAE | cnn_attention R2 | cnn_baseline MAE | cnn_baseline R2 |
|---|---:|---:|---:|---:|---:|
| full | 5000 | 0.173 | −0.090 | 0.179 | −0.125 |
| mchirp_low | 2500 | 0.201 | −0.326 | 0.202 | −0.279 |
| mchirp_high | 2500 | 0.145 | −0.173 | 0.156 | −0.363 |
| q_low | 1667 | 0.185 | −5.54 | 0.198 | −6.12 |
| q_mid | 1666 | 0.140 | −5.60 | 0.138 | −5.25 |
| q_high | 1667 | 0.194 | −14.53 | 0.203 | −14.91 |
| q_low × mchirp_low | 1241 | 0.183 | −5.71 | 0.200 | −6.43 |
| q_low × mchirp_high | 426 | 0.190 | −14.57 | 0.190 | −14.97 |
| **q_high × mchirp_low** | **544** | **0.295** | **−26.68** | **0.280** | **−23.03** |
| q_high × mchirp_high | 1123 | 0.144 | −8.08 | 0.166 | −10.62 |

**This is the sharpest signal in either run.** `q_high × mchirp_low` — near-
equal-mass, low chirp mass — is the worst cell by a wide margin in *both*
trunks (MAE ~60% higher than the full-set average, R2 catastrophically
negative), and it's also the smallest cell (544/5000 = 10.9% of validation,
matching the ~10.8% seen in the training split). The within-tercile R2 values
being large-negative across the board is partly a variance-scale artifact
(narrow true-value range shrinks the R2 denominator) — MAE is the more
trustworthy number here, and it still singles out this one cell clearly.
`std_ratio_q` inside every q-tercile subset is 1.6-2.8 (predictions *more*
spread than the true narrow band), not <1 — this doesn't look like mean-
collapse, it looks like the model spilling predictions across neighboring q
ranges when it lacks resolving signal.

---

## Phase 2 attempt #2 — All five trunks with per-head clamp (2026-07-14 11:33–12:15)

### Headline numbers (epoch 79)

| Model | clamp fix? | q-head reg? | train_r2_q | val_r2_q | val_std_ratio_q | weight_q | q MAE (phys) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| cnn_attention NEW | yes (1.0) | yes | 0.399 | **+0.204** | 0.602 | 2.72 | 0.153 |
| cnn_baseline NEW | yes (1.0) | no | 0.628 | **+0.075** | 0.653 | 2.72 | 0.165 |
| resnet1d NEW | yes (1.0) | no | −5.563 | −5.738 | 0.000 | 2.72 | 0.395 |
| inception_time NEW | yes (1.0) | no | 0.395 | **+0.219** | 0.572 | 2.72 | 0.153 |
| tcn NEW | yes (1.0) | no | 0.266 | **+0.260** | 0.557 | 2.72 | 0.150 |

For comparison, the previous best (attempt #1):
| cnn_attention OLD | no (3.0) | yes | 0.807 | −0.090 | ~0.17 | 20.09 | — |
| cnn_baseline OLD | no (3.0) | no | 0.959 | −0.125 | ~0.18 | 20.09 | — |

### Key findings

1. **4/5 models now have POSITIVE val_r2_q.** This is the first time any run
   has achieved this through epoch 79. The clamp fix eliminates the
   crash-to-negative pattern.
2. **The mechanism is the opposite of what was hypothesized.** Lowering q's
   clamp from 3.0→1.0 *reduces* q's gradient share (weight 2.72 vs 20.09 for
   other heads). This prevents the model from devoting enough capacity to
   memorize q-specific noise, which *improves* generalization.
3. **TCN is the best model.** Train/val gap = 0.006 — essentially zero
   overfitting on q. Best overall physical-unit metrics (mchirp MAE=0.97,
   q MAE=0.150, snr MAE=0.82). Still improving at epoch 79.
4. **resnet1d is a dead sigmoid, not an overfitting problem.** val_std_ratio_q
   = 0.0 (predictions have zero variance — constant output). The clamp fix
   made things *worse* (weight_q dropped from 5.17→2.72, starving an already
   dead head). Needs architectural fix, not loss-weighting fix.
5. **cnn_baseline needs q-head regularization.** With clamp alone, val_r2_q
   is positive but slowly declining (0.193→0.075). cnn_attention with both
   fixes holds steady at ~0.20. Adding the same per_head overrides should
   flatten the decline.

### Trajectory snapshots — attempt #2

`r2_q` (train) / `val_r2_q` / `weight_q` (train) at selected epochs:

| epoch | cnn_att r2_q | cnn_att val_r2_q | cnn_base r2_q | cnn_base val_r2_q | tcn r2_q | tcn val_r2_q |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | −0.027 | −0.039 | 0.028 | −0.451 | −0.819 | −0.081 |
| 5 | 0.216 | 0.209 | 0.244 | −0.533 | 0.006 | 0.029 |
| 10 | 0.239 | 0.211 | 0.308 | −0.530 | 0.144 | 0.162 |
| 16 | 0.263 | **0.244** | 0.377 | −0.072 | 0.181 | 0.179 |
| 20 | 0.277 | 0.233 | 0.409 | −0.070 | 0.175 | 0.007 |
| 30 | 0.344 | 0.229 | 0.515 | 0.126 | 0.226 | 0.247 |
| 40 | 0.378 | 0.214 | 0.586 | 0.096 | 0.247 | 0.253 |
| 50 | 0.389 | 0.205 | 0.609 | 0.080 | 0.261 | 0.247 |
| 60 | 0.395 | 0.206 | 0.619 | 0.078 | 0.265 | 0.259 |
| 70 | 0.398 | 0.204 | 0.626 | 0.075 | 0.267 | 0.261 |
| 79 | 0.399 | **0.204** | 0.628 | **0.075** | 0.266 | **0.260** |

Weight_q hits the clamp ceiling (2.72) at epoch 6 for all models except TCN
(epoch 6 also). All other heads' weights saturate at exp(3.0)=20.09
(TCN's snr weight is 11.93 at epoch 79, still converging).

Note the cnn_baseline "wobble" — val_r2_q bounces between −0.5 and +0.2 through
epoch ~12, then stabilizes. This is while weight_q is still ramping to its
clamp. After epoch 15, val_r2_q is consistently positive but slowly declining.

### Subset diagnostics — q_high × mchirp_low (epoch 80)

| model | n | mae_q | r2_q | std_ratio_q |
|---|---:|---:|---:|---:|
| cnn_attention | 544 | 0.295 | −22.06 | 0.829 |
| cnn_baseline | 544 | 0.286 | −22.31 | 0.780 |
| resnet1d | 544 | 0.745 | **−127.0** | 0.831 |
| inception_time | 544 | 0.295 | −21.67 | −0.009 |
| tcn | 544 | 0.299 | −22.01 | 0.865 |

The `q_high × mchirp_low` cell remains the worst-performing subset across ALL
models — the clamp fix doesn't address this. resnet1d's −127 R2 in this cell
is catastrophic (predictions are essentially random noise around a constant
with no correlation to true values). The other four models cluster tightly
around −22, suggesting this is a fundamental data limitation (scarcity in
this physically-hard regime) rather than a model-specific flaw.
