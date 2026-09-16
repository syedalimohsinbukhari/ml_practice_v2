# Run Commands — φc/ψ Degeneracy PoC Redo

Every command run (or ready to run) so far, with what it invokes and where it's CPU-safe vs. needs the lab GPU machine. Stages are numbered by **dependency, not by when they were prepared** — a stage number implies its inputs exist, not that every lower-numbered stage must finish first. All commands assume repo root as the working directory. Auto-generated reference — regenerate/extend rather than hand-edit around stale entries; new stages get appended in the same change that runs them.

**Dependency graph** (so nothing below gets run before its inputs exist):

```
Stage 0 (prereq checks)
Stage 1 (standalone validation)
Stage 2 (Round-1 7-arch sweep)
   ├── Stage 2b (redo-completeness checks) — depends ONLY on Stage 2's checkpoints.
   │                                          Independent of the λ-retune arc below;
   │                                          could have run any time after Stage 2.
   └── Stage 3 (Step 0 diagnostic on poc_redo_a/b) — FAILS
        └── Stage 4 (λ=0.05 retune)            — FAILS again
             └── Stage 5 (λ=0.10 retune)       — FAILS, line of attack closed
```

Stage 2b was prepared *after* Stage 5 chronologically (2026-09-16), which is why an earlier revision of this file numbered it "Stage 6" and placed it after Stage 5's table — that numbering implied a dependency on the λ-retune outcome that never existed. Corrected here: Stage 2b's `CONFIGS` dicts all point at `config_baseline.yaml`/`config_poc.yaml`/`config_tcn.yaml`/`config_cnn_attention.yaml` — Stage 2's own λ=0.01 checkpoints — never at `config_lam0{05,10}_retune_*.yaml`. Nothing in Stage 2b needed Stage 3, 4, or 5 to exist first, and nothing in Stage 2b's results bears on the λ-retune verdict either way.

## Stage 0 — Prerequisite checks (CPU-only, ✅ done 2026-09-14)

| Command | Runs / reads | Writes |
|---|---|---|
| `python experiments/phic_psi_poc-redo/prereq_checks.py` | `curriculum.py` (corrected `h_plus`/`h_cross`, `derive_w_iota`), Step 1.1 sign/combo sweep, Step 1.6 histogram (reads `combined_repackaged.hdf`) | `sweep_1_1_ratio_vs_iota.{csv,png,pdf}`, `prereq_checks_output/prereq_checks_<ts>.log` |

## Stage 1 — Standalone validation (CPU-only, ✅ done 2026-09-14)

| Command | Runs / reads | Writes |
|---|---|---|
| `python experiments/phic_psi_poc-redo/validation_script.py` | `transform_utils.py` (`double_angle`, `reconstruct_phic_psi`), `curriculum.py` — 29 synthetic-data checks, no model, no GPU | stdout only (exit code 0/1) |

## Stage 2 — Round-1 full 7-architecture sweep (GPU required, ✅ done 2026-09-15)

Each config was trained individually. `run_full.py` chains train→plot→evaluate for every `config_*.yaml` it discovers — **see the caveat near the bottom before using it again; its glob now matches every stage's configs, not just Stage 2's.**

| Command | Runs | Produces (under `runs/<name>/<timestamp>/`) |
|---|---|---|
| `python experiments/phic_psi_poc-redo/train_poc.py experiments/phic_psi_poc-redo/config_baseline.yaml` | `trainer.py` (`SumDiffTrainer`, mode=baseline), `src/gwml/training/train.py`'s `_build_callbacks` (incl. `PeriodicCheckpoint`) | `history.csv`, `diagnostics.csv`, `best/final/epoch_*.weights.h5`, `transforms.json` |
| `python experiments/phic_psi_poc-redo/train_poc.py experiments/phic_psi_poc-redo/config_poc.yaml` | same, mode=poc (corrected `2φc±2ψ` combo) | same |
| `python experiments/phic_psi_poc-redo/train_poc.py experiments/phic_psi_poc-redo/config_tcn.yaml` | same, mode=baseline, tcn | same |
| `python experiments/phic_psi_poc-redo/train_poc.py experiments/phic_psi_poc-redo/config_cnn_baseline.yaml` | same, cnn_baseline trunk | same |
| `python experiments/phic_psi_poc-redo/train_poc.py experiments/phic_psi_poc-redo/config_cnn_attention.yaml` | same, cnn_attention trunk | same |
| `python experiments/phic_psi_poc-redo/train_poc.py experiments/phic_psi_poc-redo/config_inception_time.yaml` | same, inception_time trunk | same |
| `python experiments/phic_psi_poc-redo/train_poc.py experiments/phic_psi_poc-redo/config_resnet1d.yaml` | same, resnet1d trunk | same |
| `python experiments/phic_psi_poc-redo/plot_poc.py experiments/phic_psi_poc-redo/config_<name>.yaml` | local `scripts/plot_run.py` shim (`plot_history`, `plot_diagnostics`) — CSV-only, no GPU | `history_summary.png`, `diagnostics_summary.png` in the run dir |
| `python experiments/phic_psi_poc-redo/evaluate_poc.py experiments/phic_psi_poc-redo/config_<name>.yaml --split validation` | loads trained weights, `trainer.predict` on real validation data — **GPU/eval work, not run locally** | `metrics_validation.csv`, scatter/residual/logit PNGs |
| *(not a literal command — see the caveat below before ever invoking `run_full.py`)* | `run_full.py` chains the three rows above for every discovered `config_*.yaml`. Its interface takes optional `--split`/`--configs` flags, shown above with bracket notation in earlier revisions of this table — **that bracket notation is not meant to be pasted as-is**: with the brackets stripped it becomes the bare, unscoped invocation the caveat below warns against. | — |

Results: [`NOTES.md`](NOTES.md)'s "Round 1" section — down-select confirmed, central `combo_A`/`combo_B` result flagged UNINTERPRETABLE (see Stage 3). **These four models' checkpoints (`poc_a`, `poc_b`, `tcn`, `cnn_attention`) are also Stage 2b's entire input — nothing downstream of Stage 2 touches them again, so Stage 2b's results and Stage 3's gate both depend on these specific checkpoints staying exactly as they are.**

## Stage 2b — redo-completeness checks against the certified 4-model set

**Depends only on Stage 2. Independent of Stages 3–5 (the λ-retune arc) — none of these scripts read `config_lam0{05,10}_retune_*.yaml` or any retune run directory.** Ported from `experiments/phic_psi_poc/` and repointed at this directory's own configs/checkpoints (poc_a, poc_b, tcn, cnn_attention), per explicit instruction: the redo owes the original a full accounting of every check it ran, not just the ones expected to change under the corrected formula. Prepared 2026-09-16, chronologically after Stage 5 finished, but that is an accident of scheduling, not a dependency — nothing here needed Stage 5 (or 3, or 4) to happen first.

### Stage 2b (i) — GPU/eval work (prepped 2026-09-16, ⏳ not yet run)

Each script CPU-wiring-verified (config resolution, `best.weights.h5`/`transforms.json` presence, pure-math sanity on synthetic values) — not run, since each loads real weights and calls `trainer.predict`, same GPU/eval category as `evaluate_poc.py`. No ordering dependency between these six — each reads Stage 2's checkpoints independently and writes to its own output directory — **except the one noted below.**

| Command | Runs | Status |
|---|---|---|
| `python experiments/phic_psi_poc-redo/inclination_stratification.py` | Loads each of the 4 certified models' `best.weights.h5`, predicts on 5,000 validation samples, bins `coa_phase`/`polarization_angle`/`inclination` ang_MAE by face-on/mixed/edge-on `\|cos ι\|` bands | wiring-verified, not yet run |
| `python experiments/phic_psi_poc-redo/inclination_control_stratification.py` | Same bands, applied to `mchirp` (known-good, non-angular) as a control on whether the banding itself injects spurious variance | wiring-verified, not yet run |
| `python experiments/phic_psi_poc-redo/snr_stratification.py` | Same 4 models, ang_MAE binned by SNR tercile instead of inclination (Section E of the original's verification plan) | wiring-verified, not yet run |
| `python experiments/phic_psi_poc-redo/analyse_predictions.py` | Comprehensive prediction-distribution analysis (Table 4's COLLAPSE grade is the systematic version of this session's scatter-plot mode-collapse check); `.pdf` saving added to all 3 plot functions | wiring-verified, not yet run |
| `python experiments/phic_psi_poc-redo/diagnostic_checks.py` | 5 of the original's 7 checks (Checks 5/7, tanh-saturation, intentionally omitted — inapplicable, this redo uses `activation="linear"` from day 1); Check 6's gradient chain corrected to include the `tf_double_angle` step the original's version lacked | wiring-verified (incl. a synthetic-tensor check of the Check-6 fix), not yet run |
| `python experiments/phic_psi_poc-redo/bootstrap_ang_mae.py` | Shuffle-null bootstrap significance test (10,000 iterations) on `coa_phase`/`polarization_angle`/`inclination` ang_MAE, all 4 certified models — ungated, broader than `preregistration_lam_retune.md`'s Steps 1–3 | wiring-verified (config resolution + statistical-logic sanity: perfect-fit case reads z≈24σ significant, independent-random case reads p=0.68 not significant), not yet run |

**⚠ `perturbation_trace_standalone.py` has a real internal ordering requirement — run `early` before trusting `final`, not either-or:**

| Order | Command | Purpose | Must pass before trusting the next step |
|---|---|---|---|
| 1 | `python experiments/phic_psi_poc-redo/perturbation_trace_standalone.py early` | Fresh init + ~1 epoch warmup, then the trace — **calibrates the instrument itself** | mchirp must read **DIRECTIONAL**. Per this repo's positive-control rule (CLAUDE.md: "a probe whose positive control fails at the stage under test is untrusted") and the script's own docstring: if mchirp reads AMBIGUOUS/OSCILLATORY even under fresh-init calibration, the trace methodology is unsound and step 2's table must not be used for anything. |
| 2 | `python experiments/phic_psi_poc-redo/perturbation_trace_standalone.py` (defaults to `final`) | Traces the converged Stage 2 checkpoints — the actual question (is coa_phase/polarization_angle directional-but-slow, or oscillating around a constant?) | Only interpretable once step 1's mchirp calibration has passed. Running `final` first and reading its table without `early` having passed would be exactly the premature-analysis mistake this ordering note exists to prevent — this is why the two are listed as sequential steps here, not as interchangeable `[final\|early]` options. |

### Stage 2b (ii) — CPU-only, done 2026-09-16

| Command | Runs | Status |
|---|---|---|
| `python experiments/phic_psi_poc-redo/plot_certified_memorization.py` | Reads `history.csv` for all 4 down-select-confirmed models (no GPU) — train-vs-val circular loss, adapted from the original's 2-model "certified" version since this redo hasn't certified a null on anything | **done** → [`diagnostic_output/redo_models_train_val_loss.{png,pdf}`](diagnostic_output/) — found `cnn_attention` train/val divergence (memorization), see `NOTES.md` |

Every original `.py` check now has a redo counterpart (`bootstrap_ang_mae.py` ported 2026-09-16, closing the last gap) except the per-λ training-launcher wrapper scripts, which the redo's own config-based `train_poc.py` workflow already covers directly.

## Stage 3 — Step 0 diagnostic (CPU-only, ✅ done 2026-09-15)

Depends on Stage 2's `poc_redo_a`/`poc_redo_b` checkpoints specifically (reads their `history.csv`, no model loading). Independent of Stage 2b.

| Command | Reads | Writes |
|---|---|---|
| `python experiments/phic_psi_poc-redo/diagnostic_logvar_gate.py` | `runs/phic_psi_poc_redo_{a,b}/<latest>/history.csv` only — no model loading | `diagnostic_output/diagnostic_logvar_gate_<ts>.{log,md}`, `diagnostic_output/logvar_gate_trajectories.{png,pdf}` |

Result: gate **FAILS** on both heads, both runs → UNINTERPRETABLE, not NULL. Full numbers in [`NOTES.md`](NOTES.md).

## Stage 4 — λ=0.05 retune (GPU required, ✅ done 2026-09-15)

Depends on Stage 3's FAIL verdict. Criterion fixed in [`preregistration_lam_retune.md`](preregistration_lam_retune.md) before these commands were ever run.

| Command | Runs | Status |
|---|---|---|
| `python experiments/phic_psi_poc-redo/train_poc.py experiments/phic_psi_poc-redo/config_lam005_retune_b.yaml` | `trainer.py`, mode=poc, `magnitude_penalty_lambda=0.05` — **primary test** | done → `runs/phic_psi_lam005_retune_b/20260915_154203` |
| `python experiments/phic_psi_poc-redo/train_poc.py experiments/phic_psi_poc-redo/config_lam005_retune_a.yaml` | same, mode=baseline — **required control** | done → `runs/phic_psi_lam005_retune_a/20260915_175801` |
| `python experiments/phic_psi_poc-redo/diagnostic_logvar_gate.py` (rerun) | `ROUNDS` dict includes `λ=0.05 (retune)` → `runs/phic_psi_lam005_retune_{a,b}/` alongside λ=0.01 for context; last entry in `ROUNDS` is the round OVERALL VERDICT is computed for | done — **FAIL, all four readings.** Still UNINTERPRETABLE. See `NOTES.md`. |

Gate failed again at λ=0.05 → `config_lam010_retune_{a,b}.yaml` (Stage 5, below) is the pre-registered fallback.

## Stage 5 — λ=0.10 retune (GPU required, ✅ done 2026-09-16 — last round, line of attack closed)

Depends on Stage 4's FAIL verdict. Last λ value `preregistration_lam_retune.md`'s fallback order commits to.

| Command | Runs | Status |
|---|---|---|
| `python experiments/phic_psi_poc-redo/train_poc.py experiments/phic_psi_poc-redo/config_lam010_retune_b.yaml` | `trainer.py`, mode=poc, `magnitude_penalty_lambda=0.10` — **primary test** | done → `runs/phic_psi_lam010_retune_b/20260915_181803` |
| `python experiments/phic_psi_poc-redo/train_poc.py experiments/phic_psi_poc-redo/config_lam010_retune_a.yaml` | same, mode=baseline — **required control** | done → `runs/phic_psi_lam010_retune_a/20260916_020049` |
| `python experiments/phic_psi_poc-redo/diagnostic_logvar_gate.py` (rerun, `ROUNDS` extended with `"λ=0.10 (retune)"`) | same Step 0 gate | done — **FAIL, 3 of 4 readings.** Verdict: λ alone insufficient, post-formula-fix. See `NOTES.md`. |

**Gate failed at λ=0.10 → per the preregistration's own Step 0 section: "λ alone insufficient for this architecture/head, post-formula-fix."** No further λ value is authorized. The one reading that did pass (`lam010_retune_b`/coa_phase) was checked against its training-time scatter plots (`runs/phic_psi_lam010_retune_b/20260915_181803/scatter/epoch_{0005,0080}.png`, already written by the training run, no extra GPU work) and found to be a std_ratio false-positive — a mode-collapsed constant output, not real per-sample learning. See `NOTES.md`'s 2026-09-16 correction for detail. This closes the λ-retune line of attack; Steps 1–3 never ran.

## ⚠ Caveat: `run_full.py`'s default glob now matches every stage's configs, not just Stage 2's

`run_full.py` with no `--configs` flag discovers every `config_*.yaml` in this directory — Stage 2's original seven, plus all four `config_lam0{05,10}_retune_{a,b}.yaml`. Running it bare again would retrain **everything**, and the blast radius is wider than just "wastes GPU time on already-done configs":

- Retraining Stage 2's `config_baseline.yaml`/`config_poc.yaml`/`config_tcn.yaml`/`config_cnn_attention.yaml` would create new, later run directories under the same `run_dir` base. Every Stage 2b script and Stage 3's `diagnostic_logvar_gate.py` resolve their target checkpoint via `latest_run_dir(cfg)` — a bare rerun would silently redirect all of them at the new checkpoint, invalidating every number already reported against the original 2026-09-15 Round-1 checkpoints (including the `cnn_attention` memorization finding, which is specific to that exact checkpoint) without any error or warning.
- Retraining the λ-retune configs would similarly invalidate Stage 3/4/5's recorded verdicts.

Use `--configs` to scope it explicitly, e.g.:

```bash
# Stage 5 only:
python experiments/phic_psi_poc-redo/run_full.py --configs "experiments/phic_psi_poc-redo/config_lam010_retune_*.yaml"
```

or invoke `train_poc.py` directly per-config (as tabulated above) rather than through `run_full.py`. Do not run it bare unless the explicit intent is to regenerate every checkpoint this document's stages depend on.
