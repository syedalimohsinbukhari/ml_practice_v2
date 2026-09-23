# q (mass ratio) investigation — history recap

**Compiled**: 2026-07-27, from `git log`/`git diff` on the `q_value` branch, commit range `8c2ca3738d14d34fa0f9144e4296a010c82eefc7` (exclusive) to `de847fe85d7abb7f60a3aa8ffff27cd40f511ea9` (inclusive), 25 commits spanning 2026-07-13 12:48 to 2026-07-15 15:38.
This is a dated historical snapshot, not a living doc — it documents work already closed out on prior commits.
Do not rewrite it in place; add dated addenda below if new facts about this range come to light.
Commit messages in this range are uninformative auto-labels (`[master-minor-N]`); every claim below traces to a diff or to `q_head_action_plan.md` / `q_head_run_comparison.md` / `phase2_5_3_revision_plan.md` / `phase2_5_3_run_results.md` as they existed at `de847fe8`.

Two separate efforts happened in this commit range: the `q` investigation (primary, 07-13/07-14) and a parallel `sky_position`/vMF head refactor (07-15).
Part A covers `q` in full; Part B covers the sky_position work briefly so the two aren't conflated.

---

## Part A — the q (mass ratio) investigation

### A.1 — The problem

The starting hypothesis was that `cnn_baseline` underfits `q` (train R² ~0.7–0.85, val R² ~0) while `cnn_attention` overfits it (train R² ~0.95–1.0, val R² declining to ~−0.2).
This was falsified almost immediately by the first real retraining run: `cnn_baseline`'s original run (`runs/cnn_baseline/20260714_091740`) ended at `train r2_q=0.992` / `val_r2_q=−0.164` — it was already overfitting just as badly as `cnn_attention`, not underfitting.
This reframed the diagnosis away from a trunk-capacity story and toward a shared mechanism.

Diagnosis found that all four heads' uncertainty-loss weights (`mchirp`, `q`, `merger_time`, `snr`) were saturating identically at `exp(3.0) = 20.09`, the `log_var_clamp` ceiling, starting around epoch 17–19 in `runs/cnn_attention/20260714_093319/history.csv`.
That saturation point lines up almost exactly with when `val_r2_q` peaked (~0.23, around epoch 9–12) and began collapsing, while train `r2_q` kept climbing past 0.9.

Two further real findings from Phase 1 diagnosis (`q_head_action_plan.md`):

- **Sigmoid saturation**: `q`'s raw range is `[0.2027, 0.99998]` (train) / `[0.2052, 0.99995]` (val), inside the old `UNIT_AFFINE` bounds `(0.2, 1.0)` — no clipping bug — but 7.6% of training samples (1903/25000) have true q ≥ 0.95, mapping into transformed-space `t ≈ 0.94–1.0`, right at the sigmoid's saturating asymptote where gradients vanish.
- **A real data confound**: `q_high` pairs with `mchirp_high` 2.1× more often than `mchirp_low` (5643 vs 2691); `q_low` pairs with `mchirp_low` 3× more often than `mchirp_high` (6277 vs 2056).
  "mchirp_low weakness" and "q is hard near equal-mass" are entangled by construction/sampling, not independent hypotheses.
  The `q_high × mchirp_low` cell (2691/25000, the smallest quadrant) became the persistent hard case through the rest of the investigation.

Huber delta (shared/global) was ruled out as a primary cause, since it's the same recipe used by `merger_time`, which generalizes fine.
q's training-set density rises smoothly (~1.5%/bin near q=0.2 to ~7.6%/bin near q=1.0) with no severe internal gap — the real scarcity is the joint `mchirp_low × q_high` cell, not the marginal q distribution.

### A.2 — Plan structure (`q_head_action_plan.md`)

Status legend used in the source doc: resolved/implemented, still open, implemented but awaiting a lab-machine run, or tried and made things worse.

- **Phase 0** — resolved by a `heads_spec.py` review: transform/loss binding was not the bug, periodicity was not a factor, sigmoid saturation was confirmed real.
- **Phase 1** (7 steps) — diagnose precisely, no retraining. All 7 steps done.
- **Phase 2** (5 items, steps 8–12) — fix regularization and loss/weighting, framed as "highest expected payoff":
  - Step 8, done — per-head log-var clamp.
  - Step 9, done — regularize the q head specifically.
  - Step 10, done — widen UNIT_AFFINE bounds.
  - Step 11, **not attempted** — input-level augmentation targeted at the hard regime.
  - Step 12, done — re-ran all five trunk configs.
- **Phase 3** (3 items, steps 13–15) — "only if Phase 2 plateaus":
  - Step 13, done — dedicated q branch for cnn_attention (Phase 3.2).
  - Step 14, done — targeted oversampling (Phase 3.1).
  - Step 15, **deferred** — ensemble / auxiliary-task approach.
- The doc's own "what's left to do" list at the end tracks 8 items; see §A.6 below for the still-open subset.

### A.3 — Code changes implemented

`src/gwml/heads_spec.py` — `q`'s `UNIT_AFFINE` bounds widened from `(0.2, 1.0)` to `(0.15, 1.05)`, with a comment citing the 7.6%-in-saturation-zone finding.
The head definition (with the widened bounds) remains in `HEAD_SPECS` at the tip commit even though `q` was later shelved out of `DEFAULT_HEADS` (§A.6).

`src/gwml/training/losses.py` — `log_var_clamp` now accepts a scalar (applies to all heads) or a dict, e.g. `{"default": 3.0, "q": 1.0}`, resolved per head via a new `_resolve_clamps()` and applied per head through `ClampConstraint`.

`src/gwml/models/heads.py` (`attach_heads`) — three new per-head mechanisms, all driven by `head_cfg.per_head.<name>`:

- `hidden_units`, `dropout`, `l2` overridable per head (used for q: `hidden_units=32, dropout=0.3, l2=1e-4`, first on `cnn_attention`, later also `cnn_baseline`).
- `branch: <name>` connects a head to a named `extra_features` tensor from the trunk instead of the shared pooled features (used for q's dedicated branch, Phase 3.2).
- `sigmoid_bias`, a configurable bias initializer meant to nudge a saturated sigmoid off its asymptote at init (added for resnet1d, but determined to default to 0.0/unused — see §A.5, revision-plan concern 1).

`src/gwml/models/trunks/cnn_attention.py` — the trunk now returns a third element, `{"q_tokens": tokens}`, the per-token transformer output before `AttentionPooling`, so q can branch off finer-grained features instead of the globally attention-pooled vector.

`src/gwml/training/callbacks.py` / `src/gwml/data/loader.py` — `DiagnosticSubsetsCallback._build_subsets` was refactored into a new shared `build_subset_masks()` in `loader.py`, extended with q-specific subsets: `q_low`/`q_mid`/`q_high` terciles and their cross-tab with `mchirp_low`/`mchirp_high` (`q_low_mchirp_low`, `q_low_mchirp_high`, `q_high_mchirp_low`, `q_high_mchirp_high`).
`run_experiment()` in `train.py` reuses `build_subset_masks()` to implement `data.augmentation.oversample.<subset>: <factor>` — duplicating rows of a named subset in the training set only, after the split and before `TargetTransforms.fit()`.
Leakage safety was explicitly traced and confirmed in `phase2_5_3_revision_plan.md` (§A.5, concern 3): oversampling touches only `train_strain`/`train_params` after the split; validation arrays are untouched (separate HDF5 groups).

`src/gwml/evaluation/plots.py` / `scripts/evaluate.py` — new `sigmoid_logit_hist()` diagnostic (pre-sigmoid logits, train vs val, split by true-value tercile, producing `logits_train_vs_val.png`) is the key instrument for confirming a dead sigmoid.
`residuals_vs_snr` was generalized into `residuals_vs_param(true, pred, param_values, param_label, ...)`; `evaluate.py` now also produces `residuals_mchirp_<split>.png` alongside the SNR variant, and runs prediction on both splits in one invocation.
`train.py` gained `create_run_dir()` / `latest_run_dir()` so every training run gets its own `runs/<model>/<timestamp>/` directory instead of overwriting a flat `runs/<model>/` directory; `evaluate.py` resolves the run directory via the new `latest_run_dir()` helper.

New scripts: `scripts/plot_run.py` (history.csv/diagnostics.csv → summary PNGs, includes q terciles in its diagnostics grid), `scripts/run_all.py` (train → plot_run → evaluate chain for one config), `scripts/run_full.py` (same chain looped over every non-smoke config).

Config files (`configs/*.yaml`) were edited in-place during the investigation window with `log_var_clamp: {q: 1.0}`, `per_head.q` regularization blocks, `data.augmentation.oversample.q_high_mchirp_low: 2`, and (for resnet1d) `warmup_epochs: 5`.
By the tip of the branch, however, all five configs' `heads:` lists were rewritten to drop `q` entirely (§A.6), so none of the checked-in YAMLs train q by default any more — the q-specific `per_head`/clamp/oversample blocks are vestigial unless `q` is re-added to a config's `heads:` list.

Tests added: `tests/test_losses.py::test_per_head_log_var_clamp` (dict-form clamp resolves per head), `tests/test_callbacks.py::test_q_terciles_and_mchirp_cross_tab_partition_correctly` (tercile/cross-tab subset partitioning), `tests/test_heads_spec.py::test_per_head_regularization_overrides` (per_head dropout/hidden_units/l2 wiring), `tests/test_run_dirs.py` (new, tests `create_run_dir`/`latest_run_dir`).

### A.4 — Run index

All timestamps 2026-07-14 unless noted.

| Run                     | Trunk          | Timestamp         | What changed                                                     |
|-------------------------|----------------|-------------------|------------------------------------------------------------------|
| OLD cnn_baseline        | cnn_baseline   | `20260714_091740` | baseline, pre-fix                                                |
| ATTEMPT1 cnn_baseline   | cnn_baseline   | `20260714_104107` | bounds widened only                                              |
| ATTEMPT2 cnn_baseline   | cnn_baseline   | `20260714_113347` | bounds + clamp fix (q:1.0), no q-head reg                        |
| OLD cnn_attention       | cnn_attention  | `20260714_093319` | baseline, pre-fix                                                |
| ATTEMPT1 cnn_attention  | cnn_attention  | `20260714_104523` | bounds + q-head reg, clamp fix reverted                          |
| ATTEMPT2 cnn_attention  | cnn_attention  | `20260714_113745` | bounds + q-head reg + clamp fix                                  |
| ATTEMPT2 resnet1d       | resnet1d       | `20260714_114721` | bounds + clamp fix, no reg                                       |
| ATTEMPT2 inception_time | inception_time | `20260714_121546` | bounds + clamp fix, no reg                                       |
| ATTEMPT2 tcn            | tcn            | `20260714_115404` | bounds + clamp fix, no reg                                       |
| Phase 2.5/3 re-runs     | all five       | `20260714_16xxxx` | resnet1d warmup+clamp-revert, q-branch, oversampling — see below |

### A.5 — Results

Phase 2 attempt #1 (bounds + q-head reg only, no clamp fix), final epoch 79:

| Run                              | train r2_q | val_r2_q | train−val gap |
|----------------------------------|-----------:|---------:|--------------:|
| OLD cnn_baseline                 |      0.992 |   −0.164 |         1.156 |
| NEW cnn_baseline (bounds only)   |      0.959 |   −0.125 |         1.084 |
| OLD cnn_attention                |      0.906 |   −0.179 |         1.085 |
| NEW cnn_attention (bounds + reg) |      0.807 |   −0.090 |         0.897 |

Bounds alone gave a small but real improvement everywhere; reg+bounds on cnn_attention gave a bigger improvement but still net-negative val R².

Phase 2 attempt #2 (per-head clamp `q:1.0` on all five trunks), final epoch 79 — headline result, 4/5 models achieved positive val_r2_q for the first time:

| Model                       | train_r2_q | val_r2_q | val_std_ratio_q | weight_q | q MAE (phys) |
|-----------------------------|-----------:|---------:|----------------:|---------:|-------------:|
| cnn_attention (clamp+reg)   |      0.399 |    0.204 |           0.602 |     2.72 |        0.153 |
| cnn_baseline (clamp only)   |      0.628 |    0.075 |           0.653 |     2.72 |        0.165 |
| resnet1d (clamp only)       |     −5.563 |   −5.738 |           0.000 |     2.72 |        0.395 |
| inception_time (clamp only) |      0.395 |    0.219 |           0.572 |     2.72 |        0.153 |
| tcn (clamp only)            |      0.266 |    0.260 |           0.557 |     2.72 |        0.150 |

TCN was best by every metric: train/val gap 0.006 (essentially zero overfitting), best overall mchirp MAE (0.97) and snr MAE (0.82).
resnet1d collapsed completely — `val_std_ratio_q = 0.000` (constant predictions) — confirmed as a dead sigmoid, not overfitting; the tighter clamp made it worse by further starving an already-dead head of gradient.

The mechanism was the opposite of what was originally hypothesized.
The premise going in was "q overfits because its weight hits the same ceiling as other heads, so cap it lower to restore differentiation."
What actually happened: lowering the ceiling from `exp(3.0)=20.09` to `exp(1.0)=2.72` reduces q's gradient share by roughly 7.4×, which reduces train-loss memorization capacity, which prevents overfitting — not "restoring differentiation" as originally framed.
This was reframed as a per-head gradient budget: overfitting heads (q) need a tighter clamp ceiling; collapsing heads (resnet1d) need a looser clamp, since the floor (`exp(-clamp)`) matters for collapse prevention while the ceiling matters for overfitting prevention.

The worst subset in every trunk was `q_high × mchirp_low` (near-equal-mass, low-chirp-mass, ~544/5000 = 10.9% of validation):

| Model          | mae_q |   r2_q |
|----------------|------:|-------:|
| cnn_attention  | 0.295 | −22.06 |
| cnn_baseline   | 0.286 | −22.31 |
| resnet1d       | 0.745 | −127.0 |
| inception_time | 0.295 | −21.67 |
| tcn            | 0.299 | −22.01 |

MAE here runs ~60% higher than the full-set average and is attributed to data scarcity (the smallest of the four q×mchirp quadrants); the four non-resnet1d models cluster tightly around R²≈−22, read in the source doc as "a fundamental data limitation... rather than a model-specific flaw."

`phase2_5_3_revision_plan.md` reviewed the Phase 2.5/3 changes before spending further GPU time and resolved six concerns on paper:

1. resnet1d's `sigmoid_bias` was confirmed unused/0.0 — the real fixes for resnet1d were the clamp revert (1.0→3.0) plus warmup (0→5 epochs), not the bias trick.
2. cnn_baseline was confirmed to be overfitting (gap=0.55, train=0.628 vs val=0.075), not underfitting, so regularization was the correct tool.
3. Oversampling leakage was confirmed safe by tracing `run_experiment()` line by line.
4. The q_tokens branch's use of GAP pooling (vs. attention pooling) was confirmed deliberate, on the hypothesis that q-relevant information is diffusely distributed rather than concentrated at training-set-specific positions.
5. InceptionTime's `merger_time` head was confirmed dead (train_r2=0.005, val_r2=−0.002, MAE=0.049s = 25% of range) — flagged explicitly as unrelated to q, a separate bug worth its own diagnostic pass, unresolved in this range.
6. A weight-trajectory sanity check was added as a standard post-run checklist item.

Phase 2.5/3 re-runs (07-14, 16:18–16:48) vs. the Phase 2 attempt #2 baseline, final epoch 79:

| Model          | train_r2_q (old→new) | val_r2_q (old→new) | q MAE phys (old→new) |
|----------------|----------------------|--------------------|----------------------|
| cnn_attention  | 0.399→0.300          | 0.204→0.180        | 0.153→0.159          |
| cnn_baseline   | 0.628→0.417          | 0.075→0.151        | 0.165→0.161          |
| resnet1d       | −5.563→0.999         | −5.738→0.100       | 0.395→0.162          |
| inception_time | 0.395→0.342          | 0.219→0.197        | 0.153→0.153          |
| tcn            | 0.266→0.192          | 0.260→0.252        | 0.150→0.151          |

resnet1d's dead sigmoid was revived — the biggest single win in this round: clamp revert (1.0→3.0) plus 5-epoch warmup took `val_r2_q` from −5.74 to +0.10, `val_std_ratio_q` from 0.000 (constant output) to 0.682, and cut q MAE 59% (0.395→0.162).
It now overfits massively (train/val gap=0.90), the same failure mode the CNN trunks had before their fix; the recommended next step — q-head regularization plus re-clamping to 1.0 while keeping warmup — was not executed in this range.

cnn_baseline was a clear win: val R² doubled (0.075→0.151), its decline flattened into a stable plateau, and the train/val gap halved (0.55→0.27), confirming the regularization approach generalizes across trunks.

cnn_attention regressed modestly: the q_tokens branch plus oversampling made both train and val R² slightly worse (0.399→0.300 / 0.204→0.180), read in the source doc as a likely real null result — "the hypothesis that mass-ratio information is washed out by learned attention pooling is not supported."

inception_time regressed slightly from oversampling alone (0.219→0.197).

tcn stayed stable and remained the best model overall; its train R² dropped more than its val R² (gap inverted, 0.006→0.060), interpreted as oversampled hard examples making training harder without hurting generalization.

The `q_high × mchirp_low` subset improved consistently under oversampling: MAE dropped 12–18% across cnn_attention/cnn_baseline/inception_time/tcn, and 62% for resnet1d (mostly attributable to the clamp fix rather than oversampling, since resnet1d's baseline there was catastrophic).
R² in that subset stayed deeply negative (−15 to −22) even as MAE improved, read in the source doc as confirming the data-scarcity hypothesis rather than resolving it.

### A.6 — Status at branch tip: q shelved (2026-07-15)

`q_head_action_plan.md`, as it reads at the tip commit, states explicitly: "2026-07-14: q shelved.
The default head set has been switched from `(mchirp, q, merger_time, snr)` to `(mchirp, merger_time, snr, ra, declination, coa_phase)`.
q remains in the registry and can be re-added via config.
The work below brought val_r2_q from −0.18 to +0.26 (TCN) and revived resnet1d's dead sigmoid — a ~6 R²-unit improvement.
The remaining gap (q_high×mchirp_low cell data scarcity) requires dataset-level fixes beyond the current scope."

This is corroborated by the code at the tip commit: `heads_spec.py`'s `DEFAULT_HEADS` reads `("mchirp", "merger_time", "snr", "sky_position", "coa_phase")` — no `q` — and all five model configs (`cnn_attention.yaml`, `cnn_baseline.yaml`, `resnet1d.yaml`, `inception_time.yaml`, `tcn.yaml`, `smoke.yaml`) have their `heads:` lists rewritten to match.
`q`'s `HeadSpec` (with the widened `(0.15, 1.05)` bounds) is still present in `HEAD_SPECS`, so it can be reactivated by adding `"q"` to a config's `heads:` list, but it is not exercised by default.
`PLAN.md` was updated to reflect "6 heads, 8 output dims" with `q` dropped from the target table (this table rewrite also reflects the sky_position merge, Part B).

### A.7 — What's still open

Everything below is explicitly unresolved per `q_head_action_plan.md`'s own tracking at the tip commit; none of it has been touched on this new `q_value` branch yet.

1. Input-level augmentation targeted at the hard regime (jitter/noise/SNR variation) — never attempted.
2. Ensemble / auxiliary-task approach — deferred pending 3.1+3.2 results, which are now in (§A.5) but the deferred item itself was not picked back up.
3. `pytest -m "not slow"` on the lab machine — never run in this commit range.
4. A combined re-run of all five configs measuring the Phase 2.5+3.1+3.2 changes together (rather than against the Phase 2 attempt #2 baseline only) — not done.
5. A clamp-value sweep (1.5/2.0) on tcn/cnn_baseline — optional, not done.
6. resnet1d needs q-head regularization plus a re-clamp to 1.0 while keeping the warmup — it currently overfits q as badly as cnn_attention originally did, after being revived from its dead-sigmoid state.
7. InceptionTime's dead `merger_time` head — confirmed real, explicitly out of scope for the q investigation, unresolved.
8. A Phase 3.3 auxiliary classifier — deferred, contingent on Phase 3.1+3.2 plateauing.

`q_head_action_plan.md`'s closing line sets the reference target if this work resumes: "This makes TCN the new performance baseline for q — any fix targeting the other trunks should aim to match or exceed TCN's train/val gap."
TCN's train/val gap was 0.006 in the Phase 2 attempt #2 clamp-only run, later 0.060 with oversampling added.

---

## Part B — the parallel sky-position / vMF head refactor

This is a distinct, parallel effort that happened on the same branch, mostly on 2026-07-15, about `ra`/`declination` (sky localization), not about `q`.
It is the proximate reason `q` was pulled from `DEFAULT_HEADS` on 07-15 — to make room in the default head set — not because it fixed or was blocked by the q work.
No shared diagnosis, data finding, or numerical result connects the two efforts; the only code-level overlap is incidental, since both needed to extend the same generic head-registration machinery (`HeadSpec`, `attach_heads()`, the per-head loss table).

The problem being solved: the old `heads_spec.py` treated `ra` (PERIODIC sin/cos) and `declination` (UNIT_AFFINE sigmoid) as two independent 1D heads, discarding the fact that sky position is a genuinely 2D correlated quantity on a sphere and losing the ability to represent gravitational-wave sky localization's classic ring/mirror degeneracy, which is a joint 2D shape not visible in separate marginals.

Source: `src/gwml/vmf_head_testing/` (new module), containing `sky_transform.py`, `vmf_head.py`, `sanity_check.py`, and three planning docs — `vmf_head_loss_plan.md`, `vmf_head_loss_plan_rev1.md`, `vmf_head_loss_plan_rev2.md` — read as an iterative design-review dialogue, later integrated into the main pipeline per the module's own `__init__.py` docstring.

- Plan v0 proposed merging ra+dec into one `sky_position` head outputting a 3D unit vector on S², trained with a closed-form von Mises-Fisher (vMF) negative log-likelihood, with κ as a calibrated uncertainty.
  A synthetic sanity check showed angular error dropping from ~82° (random) to ~24° after training, with a stable loss and no NaNs, but flagged two honest caveats: val loss overfit after ~epoch 50 in the toy setup, and κ didn't track per-example noise well (correlation 0.23, weakly positive, expected negative).
- Rev1, a code review before implementation, flagged a dataclass field-ordering bug (the frozen `column` field needed a default plus `__post_init__` validation), a correctness landmine (generic z-score post-processing could destroy the unit-norm constraint the vMF loss depends on if not bypassed for this head), and scope creep (the head needs two output tensors, `mu_raw`/`kappa_raw`, jointly consumed by one loss, rippling into `attach_heads`, the loss table's function signature, and the diagnostics stack, since R²/MAE aren't meaningful on constrained unit-vector components).
  It recommended a two-pass rollout: wire the math with a standalone eval script first, fold into the generic diagnostics pipeline only once trusted.
- Rev2 nailed down the `columns=(dec_col, ra_col)` ordering convention explicitly, to avoid a silent ra/dec swap bug, and flagged the config sweep needed once `ra`/`declination` are removed from `HEAD_SPECS`.

What actually landed matches the plan as reviewed:

- `heads_spec.py` — `HeadName.DECLINATION`/`HeadName.RA` removed, replaced by `HeadName.SKY_POSITION`; new `TransformKind.SPHERICAL_UNIT_VECTOR`; `HeadSpec` gained a `columns: tuple[int, ...] | None` field alongside the now-optional `column`, with `__post_init__` validation that exactly one of `column`/`columns` is set, matching Rev1's fix; new spec `HeadSpec("sky_position", transform=SPHERICAL_UNIT_VECTOR, dim=3, activation="linear", loss="vmf", columns=(PARAM_COLUMNS["declination"], PARAM_COLUMNS["ra"]))`.
- `losses.py` — new `vmf_nll_loss()` (closed-form vMF NLL, numerically stable `log(sinh(κ))`) and `_vmf_kappa_from_raw()` (softplus plus a 0.1 floor); `MultiHeadTrainer` special-cases vMF heads throughout, tracking a separate `head_kappa` metric instead of MAE/R²/std_ratio, and skipping the variance penalty for vMF heads.
- `heads.py` (`attach_heads`) — vMF heads get two output Dense layers (`{name}_mu_raw`, dim 3, linear; `{name}_kappa_raw`, dim 1, linear) instead of one.
- `DEFAULT_HEADS` at the tip: `("mchirp", "merger_time", "snr", "sky_position", "coa_phase")`.
- Evaluation — `metrics_validation.csv` for `sky_position` reports `dec_mae_deg`/`ra_mae_deg` (degrees) instead of the generic `mae`/`rmse` columns scalar heads use; confirmed directly in, e.g., `runs/tcn/20260715_150727/metrics_validation.csv`: `sky_position, mae=86.8, rmse=94.96, dec_mae_deg=65.13, ra_mae_deg=90.67`.
  This is a from-scratch head at this point in training, so these numbers are early/untrained and no comparable baseline is quoted anywhere in the source docs.
- Tests updated throughout (`test_heads_spec.py`, `test_transforms.py`) to use `sky_position` instead of `ra`/`declination`, verifying `mu_raw`/`kappa_raw` output keys and shapes.

Run evidence: the `runs/*/20260715_1[34]xxxx`-style directories (five trunks, 07-15 12:03–15:38) are sky_position/vMF runs, confirmed by their `metrics_validation.csv` containing a `sky_position` row with `dec_mae_deg`/`ra_mae_deg` columns and no `q` row, unlike every 07-14 run.