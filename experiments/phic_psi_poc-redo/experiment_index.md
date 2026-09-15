# φc/ψ Degeneracy PoC — Redo — Experiment Index

**Branch**: `poc/phic-psi-degeneracy-redo`
**Last updated**: 2026-09-14

---

## Core narrative

| File | Description |
|---|---|
| [`NOTES.md`](NOTES.md) | Running narrative log — Step 1 prerequisite-check results, design decisions, next steps. Read this first. |
| [`redo_procedure.md`](redo_procedure.md) | Step-by-step build plan for the redo, mirroring the original `planning_files/phic_psi_implementation_plan_v4.md`'s structure, with every step marked `[UNCHANGED]`/`[CORRECTED]`/`[NEW]` against the corrected `2φc±2ψ` combo formula. |
| [`formulae_reference.md`](formulae_reference.md) | Exact physics/loss formulae used at each step (vector algebra, combo construction, isotropic circular loss, magnitude penalty, toy antenna-pattern model, curriculum weight derivation, reconstruction branch-handling). Model architectures given as a bare outline only, no per-layer math. |
| [`comments.md`](comments.md) | External review of the redo package (2026-09-14) — confirms the core `2φc±2ψ` fix is internally consistent throughout, hand-verifies the A.5 self-check, and proposed the A.8 branch-ambiguity re-derivation. The branch-*count* part held up; the closed-form parity-filter part did not (see below). |

## Code

| File | Description |
|---|---|
| [`curriculum.py`](curriculum.py) | Step 1.2 toy antenna-pattern model + Jacobian-condition-number sweep + `derive_w_iota`, copied from `experiments/phic_psi_poc/curriculum.py` with `h_plus`/`h_cross` corrected to `2*phi_c` (A.5). |
| [`transform_utils.py`](transform_utils.py) | Vector algebra primitives, copied from `experiments/phic_psi_poc/transform_utils.py` with `double_angle`/`tf_double_angle` added (A.1.5) and `reconstruct_phic_psi` corrected for the `2φc±2ψ` combo (A.8) — brute-force only, no parity shortcut (see below). |
| [`trainer.py`](trainer.py) | `SumDiffTrainer`, copied from the original with one change: `_build_combo_vectors` doubles `z_phic` before combining, per A.2. Verified against synthetic TF tensors. |
| [`validation_script.py`](validation_script.py) | Standalone (no GPU/model) test suite for the transform/reconstruction math, §3. 29/29 checks pass. |
| [`prereq_checks.py`](prereq_checks.py) | Steps 1.1/1.2/1.6 empirical checks, corrected combo↔(φc,ψ) conversion and sweep ranges. Results in `NOTES.md` and `prereq_checks_full_run.log`. |
| [`train_poc.py`](train_poc.py), [`plot_poc.py`](plot_poc.py), [`evaluate_poc.py`](evaluate_poc.py), [`run_full.py`](run_full.py) | Training/plotting/evaluation orchestration, copied verbatim in logic from the original — no formula-specific code in any of them. |
| `config_baseline.yaml`, `config_poc.yaml`, `config_tcn.yaml`, `config_cnn_baseline.yaml`, `config_cnn_attention.yaml`, `config_inception_time.yaml`, `config_resnet1d.yaml` | Full 7-architecture Round-1-equivalent sweep configs. `checkpoint_every_n: 10` and `magnitude_penalty_lambda: 0.01` present in all seven from day 1 (three of the original's configs never had the latter). `config_poc.yaml`'s `well_constrained_combo`/`sign_dependent_combo` set from this redo's own Step 1.1 rerun, not copied from the original. |

## Analysis outputs

| File | Description |
|---|---|
| [`prereq_checks_full_run.log`](prereq_checks_full_run.log) | Full console output of the Step 1.1/1.2/1.6 run (2026-09-14, full resolution: n_sky=200, n_iota_sweep=50/regime, n_boot=5000). |
| [`sweep_1_1_ratio_vs_iota.csv`](sweep_1_1_ratio_vs_iota.csv), [`sweep_1_1_ratio_vs_iota.png`](sweep_1_1_ratio_vs_iota.png), [`sweep_1_1_ratio_vs_iota.pdf`](sweep_1_1_ratio_vs_iota.pdf) | Step 1.1's ι-sweep data and plot, corrected-formula version. |

## What's still open?

**Step 1 (prerequisite checks) is complete** (2026-09-14, see `NOTES.md`) — harness self-check passes, sign-flip confirmed and now *statistically significant* in both regimes (unlike the original's borderline result), `w(ι)` re-derived, population histogram reconfirmed unchanged. Decision: `well_constrained_combo: combo_B`, `sign_dependent_combo: true`.

**Code implementation (§2/§3) is complete**: `transform_utils.py`, `curriculum.py`, `trainer.py` all built, all numerically verified (5,000–20,000-trial checks where relevant); `validation_script.py` 29/29 pass. All 7 config YAMLs built and verified end-to-end on CPU (forward pass, loss, gradient step — no `None` grads) for both `mode: baseline` and `mode: poc`.

**Round 1 (full 7-architecture sweep) trained 2026-09-15 — see `NOTES.md`.** Infrastructure fully validated (positive controls healthy in every run, periodic checkpoints present, plot/eval pipeline confirmed working end-to-end). Down-select re-validated on fresh evidence: original 4-model certified set (poc_a, poc_b, tcn, cnn_attention) still holds.

**⚠ Central question not yet answered.** `config_poc.yaml`'s combo_A/combo_B circular loss stays flat (~0.98–1.01) across all 80 epochs — nearly identical to the original's wrong-formula result (0.9989/0.9913 vs. this run's 0.9999/0.9895) — so the corrected formula alone did not produce visible learning. But `std_ratio` for the raw φc/ψ vectors is settling low (≈0.64/0.25) and the uncertainty weights are climbing while loss stays flat — the same log_var-runaway/std_ratio-gate pattern the original needed a dedicated λ-retune investigation to resolve before trusting its own null result. **Verdict: UNINTERPRETABLE, not NULL, pending that same diagnostic discipline here.**

**Resolved by the 2026-09-14 review** (`comments.md`): A.8's reconstruction branch-handling, previously flagged "re-derive before coding," now has confirmed branch counts — ψ is 2-fold ambiguous, φc is 4-fold ambiguous, exactly 4 of the naive 8 candidate pairs are always jointly consistent. Also noted, not a bug: A.5's toy `F_plus`/`F_cross` uses the opposite handedness convention (`e^{+2iψ}`) from some literature conventions — washes out, self-check passes regardless.

**Same-day self-correction (2026-09-14, after implementing `transform_utils.py`):** the review's proposed closed-form parity rule for picking the 4 consistent candidates without brute force (`k≡j mod 2`) was implemented, tested, and immediately falsified — a 20,000-trial numerical sweep showed the parity pattern is ~50/50 and data-dependent (depends on an integer `arctan2`'s mod-2π reduction destroys), not fixed. Retracted in `formulae_reference.md` §A.8 and `redo_procedure.md` §2.6, kept on the record rather than silently erased, per this repo's frozen-vs-living-docs convention. The reconstruction code brute-forces all 8 candidates every time — verified against 5,000 random trials, 100% recovery.

Next steps, in order:
1. **Blocking**: build and run the log_var-trajectory diagnostic (std_ratio + weight_combo_A/B gate) against `phic_psi_poc_redo_b`'s checkpoints before treating the flat combo_A/combo_B loss as a real null result.
2. If the gate fails (the climbing-weight/low-std_ratio pattern already visible suggests it will), a λ-retune pass mirroring the original's Runs 8–9b — pre-register the criterion first, per that investigation's own lesson about not eyeballing std_ratio after the fact.
3. Only once combo_A/combo_B reads as a clean, gate-passed null (or shows real learning) does the central degeneracy question get an actual answer.

The original (wrong-formula) investigation is untouched at `experiments/phic_psi_poc/`, and fully preserved as a static snapshot on the `archive/phic-psi-poc-v1` branch.
