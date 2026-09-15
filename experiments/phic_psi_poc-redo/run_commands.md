# Run Commands — φc/ψ Degeneracy PoC Redo

Every command run (or ready to run) so far, in order, with what it invokes and where it's CPU-safe vs. needs the lab GPU machine. All commands assume repo root as the working directory. Auto-generated reference — regenerate/extend rather than hand-edit around stale entries; new stages get appended in the same change that runs them.

## Stage 0 — Prerequisite checks (CPU-only, ✅ done 2026-09-14)

| Command                                                 | Runs / reads                                                                                                                                     | Writes                                                                                 |
|---------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| `python experiments/phic_psi_poc-redo/prereq_checks.py` | `curriculum.py` (corrected `h_plus`/`h_cross`, `derive_w_iota`), Step 1.1 sign/combo sweep, Step 1.6 histogram (reads `combined_repackaged.hdf`) | `sweep_1_1_ratio_vs_iota.{csv,png,pdf}`, `prereq_checks_output/prereq_checks_<ts>.log` |

## Stage 1 — Standalone validation (CPU-only, ✅ done 2026-09-14)

| Command                                                     | Runs / reads                                                                                                                | Writes                      |
|-------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|-----------------------------|
| `python experiments/phic_psi_poc-redo/validation_script.py` | `transform_utils.py` (`double_angle`, `reconstruct_phic_psi`), `curriculum.py` — 29 synthetic-data checks, no model, no GPU | stdout only (exit code 0/1) |

## Stage 2 — Round-1 full 7-architecture sweep (GPU required, ✅ done 2026-09-15)

Each config was trained individually. `run_full.py` chains train→plot→evaluate for every `config_*.yaml` it discovers — **see the caveat at the bottom before using it again now that the λ-retune configs (Stage 4) also match that glob.**

| Command                                                                                                                    | Runs                                                                                                                           | Produces (under `runs/<name>/<timestamp>/`)                                          |
|----------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| `python experiments/phic_psi_poc-redo/train_poc.py experiments/phic_psi_poc-redo/config_baseline.yaml`                     | `trainer.py` (`SumDiffTrainer`, mode=baseline), `src/gwml/training/train.py`'s `_build_callbacks` (incl. `PeriodicCheckpoint`) | `history.csv`, `diagnostics.csv`, `best/final/epoch_*.weights.h5`, `transforms.json` |
| `python experiments/phic_psi_poc-redo/train_poc.py experiments/phic_psi_poc-redo/config_poc.yaml`                          | same, mode=poc (corrected `2φc±2ψ` combo)                                                                                      | same                                                                                 |
| `python experiments/phic_psi_poc-redo/train_poc.py experiments/phic_psi_poc-redo/config_tcn.yaml`                          | same, mode=baseline, tcn                                                                                                       | same                                                                                 |
| `python experiments/phic_psi_poc-redo/train_poc.py experiments/phic_psi_poc-redo/config_cnn_baseline.yaml`                 | same, cnn_baseline trunk                                                                                                       | same                                                                                 |
| `python experiments/phic_psi_poc-redo/train_poc.py experiments/phic_psi_poc-redo/config_cnn_attention.yaml`                | same, cnn_attention trunk                                                                                                      | same                                                                                 |
| `python experiments/phic_psi_poc-redo/train_poc.py experiments/phic_psi_poc-redo/config_inception_time.yaml`               | same, inception_time trunk                                                                                                     | same                                                                                 |
| `python experiments/phic_psi_poc-redo/train_poc.py experiments/phic_psi_poc-redo/config_resnet1d.yaml`                     | same, resnet1d trunk                                                                                                           | same                                                                                 |
| `python experiments/phic_psi_poc-redo/plot_poc.py experiments/phic_psi_poc-redo/config_<name>.yaml`                        | local `scripts/plot_run.py` shim (`plot_history`, `plot_diagnostics`) — CSV-only, no GPU                                       | `history_summary.png`, `diagnostics_summary.png` in the run dir                      |
| `python experiments/phic_psi_poc-redo/evaluate_poc.py experiments/phic_psi_poc-redo/config_<name>.yaml --split validation` | loads trained weights, `trainer.predict` on real validation data — **GPU/eval work, not run locally**                          | `metrics_validation.csv`, scatter/residual/logit PNGs                                |
| `python experiments/phic_psi_poc-redo/run_full.py [--split validation] [--configs <glob>]`                                 | chains the three above for every discovered `config_*.yaml` (default glob: this directory, skips names containing `"smoke"`)   | one full train/plot/evaluate cycle per matched config                                |

Results: [`NOTES.md`](NOTES.md)'s "Round 1" section — down-select confirmed, central `combo_A`/`combo_B` result flagged UNINTERPRETABLE (see Stage 3).

## Stage 3 — Step 0 diagnostic (CPU-only, ✅ done 2026-09-15)

| Command                                                          | Reads                                                                       | Writes                                                                                                           |
|------------------------------------------------------------------|-----------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| `python experiments/phic_psi_poc-redo/diagnostic_logvar_gate.py` | `runs/phic_psi_poc_redo_{a,b}/<latest>/history.csv` only — no model loading | `diagnostic_output/diagnostic_logvar_gate_<ts>.{log,md}`, `diagnostic_output/logvar_gate_trajectories.{png,pdf}` |

Result: gate **FAILS** on both heads, both runs → UNINTERPRETABLE, not NULL. Full numbers in [`NOTES.md`](NOTES.md).

## Stage 4 — λ=0.05 retune (GPU required, ✅ done 2026-09-15)

Criterion fixed in [`preregistration_lam_retune.md`](preregistration_lam_retune.md) before these commands were ever run.

| Command                                                                                                       | Runs                                                                                                                                                                              | Status                                                                     |
|---------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| `python experiments/phic_psi_poc-redo/train_poc.py experiments/phic_psi_poc-redo/config_lam005_retune_b.yaml` | `trainer.py`, mode=poc, `magnitude_penalty_lambda=0.05` — **primary test**                                                                                                        | done → `runs/phic_psi_lam005_retune_b/20260915_154203`                     |
| `python experiments/phic_psi_poc-redo/train_poc.py experiments/phic_psi_poc-redo/config_lam005_retune_a.yaml` | same, mode=baseline — **required control**                                                                                                                                        | done → `runs/phic_psi_lam005_retune_a/20260915_175801`                     |
| `python experiments/phic_psi_poc-redo/diagnostic_logvar_gate.py` (rerun)                                      | `ROUNDS` dict includes `λ=0.05 (retune)` → `runs/phic_psi_lam005_retune_{a,b}/` alongside λ=0.01 for context; last entry in `ROUNDS` is the round OVERALL VERDICT is computed for | done — **FAIL, all four readings.** Still UNINTERPRETABLE. See `NOTES.md`. |

Gate failed again at λ=0.05 → `config_lam010_retune_{a,b}.yaml` (Stage 5, below) is the pre-registered fallback.

## Stage 5 — λ=0.10 retune (GPU required, configs ready, ⏳ not yet run)

Last λ value `preregistration_lam_retune.md`'s fallback order commits to.

| Command                                                                                                                                                                          | Runs                                                                       | Status                                                    |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|-----------------------------------------------------------|
| `python experiments/phic_psi_poc-redo/train_poc.py experiments/phic_psi_poc-redo/config_lam010_retune_b.yaml`                                                                    | `trainer.py`, mode=poc, `magnitude_penalty_lambda=0.10` — **primary test** | not yet run                                               |
| `python experiments/phic_psi_poc-redo/train_poc.py experiments/phic_psi_poc-redo/config_lam010_retune_a.yaml`                                                                    | same, mode=baseline — **required control**, must be trained alongside      | not yet run                                               |
| `python experiments/phic_psi_poc-redo/diagnostic_logvar_gate.py` (rerun, with a `"λ=0.10 (retune)"` entry appended to `ROUNDS` pointing at `runs/phic_psi_lam010_retune_{a,b}/`) | same Step 0 gate                                                           | ready once the entry is appended and the runs above exist |

If the gate still fails at λ=0.10: per the preregistration's own Step 0 section, report "λ alone insufficient for this architecture/head, post-formula-fix" — not a new, unplanned λ value.

## ⚠ Caveat: `run_full.py`'s default glob now also matches Stage 4/5 configs

`run_full.py` with no `--configs` flag discovers every `config_*.yaml` in this directory — which now includes all four `config_lam0{05,10}_retune_{a,b}.yaml` alongside the Stage 2 set. Running it bare again would retrain everything, including configs already trained (Stage 2, Stage 4) and configs not yet meant to run standalone via `evaluate_poc.py` before their Step 0 gate is checked (Stage 5). Use `--configs` to scope it explicitly, e.g.:

```bash
# Stage 5 only:
python experiments/phic_psi_poc-redo/run_full.py --configs "experiments/phic_psi_poc-redo/config_lam010_retune_*.yaml"
```

or invoke `train_poc.py` directly per-config (as tabulated above) rather than through `run_full.py`, until the retune's Step 0 gate has been checked.
