# φc/ψ Degeneracy PoC — Running Notes

Companion to `phic_psi_implementation_plan_v4.md` and
`phi_c_psi_degeneracy_poc.md`.  Detailed run-by-run results are in
[`results.md`](results.md).

## Setup

- **Branch:** `poc/phic-psi-degeneracy` (off `master`)
- **Created:** 2026-07-16
- **Goal:** go/no-go signal on sum/difference-head + inclination-curriculum design

## Confirmed vs. Assumed

### Confirmed
- [Step 1.3] Backbone ingests raw whitened time-domain strain (N, 4096, 2) with
  detector order [h1, l1] — inherently encodes phase in temporal structure.
  Confirmed from `loader.py:29-35`.

- [Step 1.4] True inclination (column 2) is accessible via adding `inclination`
  to the active heads list. `TargetTransforms` encodes it as PERIODIC with
  period=2π, giving `y_true["inclination"] = [sin(ι), cos(ι)]`. Therefore
  `y_true["inclination"][:, 1]` = cos(ι_true). No extra data plumbing needed.

- [Step 1.5] Curriculum mechanism: static per-sample loss weight `w(ι_true)`
  throughout training (option a). No epoch-based scheduling.

### Still to verify
- None remaining — all prerequisite gates passed.

### Resolved (2026-07-16, rev2)

- [Step 1.1] ✓ Sign/combination check — **rev2: split by sign(cos ι).**
  - **cos ι > 0 (ι=π/4):** well-constrained = **combo_B** (φc−2ψ), ratio = 1.2×.
  - **cos ι < 0 (ι=3π/4):** well-constrained = **combo_A** (φc+2ψ), ratio = 1.2×.
  - **Sign flip: YES** — the well-constrained label depends on sign(cos ι).
  - **Trainer change:** `sign_dependent_combo=true` enables dynamic per-sample
    assignment at batch level (combo_B is good when cos ι ≥ 0, combo_A when cos ι < 0).
  - Note: the 1.2× ratio in each regime is still weak. The rev2 review flagged this —
    it could mean the degeneracy is genuinely near-symmetric, or that the
    correlation-based test statistic has limited sensitivity. Splitting by sign
    eliminated the dilution artifact (was mixing both regimes before).
  - **Config updated:** `well_constrained_combo: combo_B`, `sign_dependent_combo: true`.

- [Step 1.2] ✓ Curriculum weight derivation — **rev2: replaced broken polynomial fit.**
  - Old polynomial fit had residual 1.32 (4× the function range) and inconsistent
    endpoint values. Root cause: fitting an unconstrained 3-param polynomial to a
    sigmoid-like curve in cos²ι.
  - **Fix:** replaced with numpy linear interpolation on the raw empirical curve,
    clamped to [0, 1]. Residual = 0 (interpolation is exact at grid points).
  - w(cos²ι=1) = 0.0000 (face-on) — correct.
  - w(cos²ι=0) = 0.1212 (edge-on) — lower than the 1−cos²ι default would give (1.0).
    Edge-on asymptote is limited by how well (R,δ) can constrain (φc,ψ) even at
    favourable inclination — the analytical Jacobian says the condition number is
    still moderate at edge-on.
  - Improved Jacobian computation: eps=1e-5 (was 1e-6), n_Phi=200 (was 100).
  - **Decision unchanged: use w_iota_default (1−cos²ι) for Run B.** The empirical
    curve's low edge-on asymptote (0.12) may be a real physical signal or a Jacobian
    artifact — either way, the default is safer for an initial go/no-go run.

### Resolved (2026-07-16, rev3 — deep grid sweep + bootstrap)

- [Step 1.1] ✓ **rev3: ι sweep confirms physics trend + bootstrap confirms significance.**
  - Deep grid: 200 sky positions, 50 ι points (25/sign), 5000 bootstrap samples.
  - The ratio GROWS toward face-on (~1.6× at ι≈0.1) and SHRINKS toward edge-on
    (~1.1× at ι≈π/2) — exactly the trend physics predicts.
  - 95% CI excludes 1.0 in both sign regimes → statistically significant.
  - See `results.md` for the full sweep table and `sweep_1_1_ratio_vs_iota.png`
    for the plot.
  - **Verdict: real, modest signal. Proceed with sign_dependent_combo=true.**

- [Step 1.2] ✓ **rev3: sky-averaging confirmed, intermediate shape reported.**
  - 200 sky positions, 200 ι points, 5-point intermediate shape printed.
  - w peaks at intermediate ι (~0.8 rad) then drops toward edge-on — unexpected
    but may reflect real physics (both polarisations at intermediate angles).
  - **Decision unchanged: use w=1−cos²ι default for Run B.**

- [Step 1.6] ✓ cos ι histogram (run on lab GPU machine, 2026-07-16).
  - Face-on fraction (|cos ι| > 0.9): 28.7%.
  - Edge-on fraction (|cos ι| < 0.5): 32.7%.
  - Population is well-mixed — NO statistical power concern. Proceed.

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Subclass MultiHeadTrainer | Reuses log_var, metrics, train_step machinery |
| inclination as trained head | Simplest way to get cos(ι_true) at batch level |
| Both numpy + TF transform utils | numpy for validation, TF for graph ops |
| Circular loss (1−dot), not Huber | Provably isotropic; avoids per-component Huber anisotropy |
| Combo log_var clamps at 3.0 (default) | Don't double-suppress with w(ι) |
| Separate train_poc.py script | Doesn't modify main-branch train.py |
| TCN backbone | Current best performer per multi-architecture comparison |
| Sign-dependent combo label | Step 1.1 found flip at cos ι=0; dynamic per-sample assignment avoids label dilution |
| w=1−cos²ι default (not empirical fit) | Empirical fit gives edge-on w≈0.12 vs. default 1.0; default is safer for first go/no-go run |

## Run Log

### Round 1 — all architectures (2026-07-17/18)

Seven runs completed across 5 architectures + poc_a/poc_b on TCN.
See `analysis_output/` for detailed CSVs, plots, and markdown reports.

**Key finding: total mode collapse on φc and ψ.**

All PERIODIC heads (coa_phase, polarization_angle, inclination) collapsed to
predicting a single constant value regardless of input. circ_r = 1.000 for
φc across all TCN variants, 1.000 for ψ across nearly all models.

This was NOT random noise — the models converged to outputting exactly one
angle for every sample. The gradient is identically zero for these heads.

**ψ R² = 0.75 was a mathematical artifact.** For a PERIODIC head with
constant predictions against a uniform true distribution, the expected
null-hypothesis R² = 1 − (π²/48)/(π²/12) = 0.75. The diagnostic looked
healthy but was actually the null.

**Non-degenerate heads are fine across TCN variants** (mchirp R² > 0.95,
snr R² > 0.78). CNN baseline struggles on mchirp (R²=0.63) and merger_time
(R²=0.24). Inception can't do merger_time (R²=−0.001).

**Sky position anomaly:** poc_a/poc_b show 4× worse angular MAE (12.9°)
than plain TCN (3.2°), despite all being TCN. Needs investigation — may be
a transforms.json save/load issue in the SumDiffTrainer pipeline.

### Diagnostic scripts

- `analyse_predictions.py` — loads all models, predicts on validation,
  reports per-head stats (MAE, R², circular concentration, mode-collapse
  detection), writes CSVs + markdown report + PNG plots.
- `diagnostic_checks.py` — five deep checks (Check 6 pending):
  1. True label distribution
  2. Loss function verification
  3. Log-var trajectory over training
  4. Gradient routing + prediction perturbation
  5. Pre-tanh logit saturation
  6. (pending) Gradient chain instrument through combo pipeline
- `diagnostic_log.md` — **thesis reference**: chronological log of every
  diagnostic run, hypothesis tested, outcome, and wrong turns. Tracks which
  hypotheses are active/ruled out and what evidence supports each.
### Diagnostic findings — Run 2 (2026-07-18 13:17)

See `diagnostic_output/diagnostic_checks_20260718_131719.log` for full output.

1. **True labels: CLEAN.** Confirmed again — no data pipeline bug.
2. **Loss wiring: FIXED CONFIRMED.** poc_b shows `head_loss` without
   coa_phase/pol_angle. `head_loss.pop()` fix works.
3. **Circular loss still frozen.** Identical to Run 1 — this reads old
   training-history CSVs, so no change expected. Needs fresh retrain.
4. **Prediction perturbation: SMOKING GUN.** After one gradient step:
   - mchirp: mean|Δ| = 0.81 (changes)
   - inclination: mean|Δ| = 0.46 (changes — proves PERIODIC encoding is fine)
   - **coa_phase: mean|Δ| = 0.00** (zero change)
   - **polarization_angle: mean|Δ| = 0.00** (zero change)
   - combo_A/B log_vars DO get gradient (0.097, 0.095)
   - Weight names are flat (`kernel`, `bias` — Keras strips prefixes), so
     name-matching filter has false negatives. Layer lookup (Check 5) works.
5. **Tanh saturation: RULED OUT.** est_logit_mag ≈ 0.5 for all heads, well
   below 3.0 threshold. Kernels change ~3% over training — weights move but
   far too slowly. My prior hypothesis was wrong.

> **Note:** These conclusions were overturned by Check 6 (Run 3 ~25 minutes later). Tanh saturation IS the root cause. See `diagnostic_log.md`.

### Revised diagnosis (Run 2)

Not tanh saturation. Not a disconnected graph (combo log_vars get gradient,
trunk-dependent heads change). The problem: **severely attenuated gradient
through the circular loss → combo transform → head output chain**, specific
to the coa_phase/pol_angle output layers. The gradient exists but is ~orders
of magnitude too weak to train the heads in 80 epochs.

Inclination (same PERIODIC encoding, same trunk, NOT in the combo path) gets
healthy gradient — this isolates the problem to the normalize_unit →
complex_mul → 1−cosΔθ pipeline, not the PERIODIC encoding itself.

### Diagnostic findings — Run 3 (2026-07-18 13:41) — ROOT CAUSE FOUND

See `diagnostic_output/diagnostic_checks_20260718_134134.log` and
`diagnostic_log.md` for full details.

**Check 6: Tanh saturation CONFIRMED.** Forward-pass dump shows every sample
produces `z_phic_raw = [-1.000000, +1.000000]` (norm=√2=1.4142) regardless of
input. The post-tanh output gradient is healthy (0.308) but `tanh'(sat) ≈ 0`
kills gradient to the weights. All contradictions resolved:
- Check 3's `std_ratio = 1.414` = √2 = norm of saturated (±1,±1)
- Check 4's "zero gradient but weight movement" = Adam momentum coasting
- Check 5's "est_logit ≈ 0.5 ok" = estimate used wrong input-scale assumption

**Fix:** `tanh → linear` for PERIODIC heads. `normalize_unit` already handles
unit-circle projection — tanh is redundant and creates saturation bottleneck.

### Next steps

- [x] Root cause identified: tanh saturation on PERIODIC heads
- [x] Check 7: saturation at step 0 — init variance confirmed
- [x] Applied fix: `activation="tanh"` → `"linear"` for all 3 PERIODIC heads
      (inclination, coa_phase, polarization_angle) in `src/gwml/heads_spec.py`
- [ ] Retrain all models (7 configs) with linear activation
- [ ] Re-run `analyse_predictions.py` to check φc/ψ learnability
- [ ] THEN implement ι-conditioning plan (`plan_iota_conditioning.md`)
- [ ] Investigate sky_position degradation in SumDiffTrainer

## Run Log (original)

### Run A (baseline) — 2026-07-16 (re-run 2026-07-17)
- **Config:** `config_baseline.yaml`
- **Branch:** `poc/phic-psi-degeneracy`
- **Machine:** lab GPU
- **Epochs:** 80
- **Mode:** baseline (circular loss on individual φc/ψ)
- **Status:** Training complete. Full analysis available.
- **Key result:** φc mode-collapsed to 315° constant. ψ mode-collapsed to 67.5° constant.

### Run B (PoC) — 2026-07-16 (re-run 2026-07-18)
- **Config:** `config_poc.yaml`
- **Branch:** `poc/phic-psi-degeneracy`
- **Machine:** lab GPU
- **Epochs:** 80
- **Mode:** poc (combo heads + curriculum weighting, sign-dependent)
- **Status:** Training complete. Metrics pending (run `run_full.py` or evaluate manually).
- **Preliminary diagnostics (epoch 80, full):**
  - coa_phase MAE=1.572, R²=−0.012 (identical to baseline — not directly trained in poc mode)
  - polarization_angle MAE=0.779, R²=0.754 (identical to baseline — not directly trained)
  - inclination MAE=1.576, R²=−0.007 (slightly worse than baseline)
  - mchirp MAE=1.035, R²=0.958 (+10.3% worse than baseline)
  - **Note:** Individual φc/ψ heads are NOT trained in poc mode — they only
    receive gradient through combo_A/combo_B. The dead coa_phase is expected.
    The real comparison needs combo-level circular-loss metrics from evaluation.

### Multi-architecture baselines — ran but invalidated

Configs created for all five architectures with the full 7-head list
(mchirp, merger_time, snr, sky_position, coa_phase, polarization_angle,
inclination), all using `mode: baseline` (circular loss on individual
φc/ψ — same loss function class as Run B, so the only variable is
architecture).

| Config | Trunk | Ran (invalidated by tanh — pending retrain) |
|--------|-------|--------|
| `config_tcn.yaml` | tcn | `train_poc.py config_tcn.yaml` |
| `config_cnn_baseline.yaml` | cnn_baseline | `train_poc.py config_cnn_baseline.yaml` |
| `config_cnn_attention.yaml` | cnn_attention | `train_poc.py config_cnn_attention.yaml` |
| `config_inception_time.yaml` | inception_time | `train_poc.py config_inception_time.yaml` |
| `config_resnet1d.yaml` | resnet1d | `train_poc.py config_resnet1d.yaml` |

### Batch run (all configs at once)

```bash
python experiments/phic_psi_poc/run_full.py
```

Chains `train_poc.py` → `plot_poc.py` → `evaluate_poc.py` for every
`config_*.yaml` in the experiments directory (skips smoke configs).  Each
step runs as an independent subprocess so GPU memory is freed between
steps — mirrors the `scripts/run_full.py` pattern.

The three scripts can also be run individually:

```bash
python experiments/phic_psi_poc/train_poc.py experiments/phic_psi_poc/config_poc.yaml
python experiments/phic_psi_poc/plot_poc.py experiments/phic_psi_poc/config_poc.yaml
python experiments/phic_psi_poc/evaluate_poc.py experiments/phic_psi_poc/config_poc.yaml --split validation
```

### Run C (ι=0 slice) — not yet configured

Optional sanity check per plan Sec. 4: fixed `ι=0` dataset slice to
confirm diff-head collapses to near-zero under guaranteed-exact degeneracy.
Not urgent — useful for debugging if Run B results are ambiguous.
