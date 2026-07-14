# q-Head Run Comparison — Phase 2 Attempt #1

Companion to `q_head_action_plan.md`. Compares the pre-fix runs against the
first post-fix runs, epoch-by-epoch, using `history.csv` and `diagnostics.csv`
directly (no retraining involved in producing this file).

| Run | Trunk | Timestamp | What changed vs the pre-fix run |
|---|---|---|---|
| OLD cnn_baseline | `cnn_baseline` | `20260714_091740` | baseline (pre-fix) |
| NEW cnn_baseline | `cnn_baseline` | `20260714_104107` | q bounds widened `(0.2,1.0)→(0.15,1.05)` only — config file itself untouched |
| OLD cnn_attention | `cnn_attention` | `20260714_093319` | baseline (pre-fix) |
| NEW cnn_attention | `cnn_attention` | `20260714_104523` | q bounds widened **+** q head regularized (`hidden_units: 32, dropout: 0.3, l2: 1e-4`) — **but** the per-head `log_var_clamp` fix (`q: 1.0`) was reverted back to the flat `3.0` before this run, so it was **not** actually tested |

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

## Bottom line for what to try next

1. **The clamp fix (Phase 2 step 8) still hasn't actually been tested** — it
   was written into `cnn_attention.yaml` and then reverted before this run.
   Re-apply `log_var_clamp: {default: 3.0, q: 1.0}` (or tighter) and re-run
   — this is the one lever from the original plan that hasn't had a real
   trial yet, and it's the one most directly aimed at the identical-
   saturation mechanism.
2. **cnn_baseline needs the same regularization cnn_attention got.** It
   overfits q just as badly, so it's a legitimate second data point for
   whether the per-head regularization + clamp combination generalizes
   across trunks, or was a `cnn_attention`-specific band-aid.
3. **The mchirp_low × q_high cell is now a confirmed, quantified priority**
   for Phase 3 step 14 (targeted oversampling) regardless of how step 1-2
   above turn out — it's the worst-performing *and* most data-starved region
   in both trunks.
