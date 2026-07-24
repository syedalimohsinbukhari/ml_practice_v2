# φc/ψ Degeneracy Investigation — Full Experiment Summary

**Branch:** `poc/phic-psi-degeneracy`
**As of:** 2026-07-22 (through Run 9b)
**Purpose of this file:** a single self-contained reference for the whole
investigation, for dropping into a new conversation or onboarding review
without re-reading the full run-by-run history in `NOTES.md` /
`diagnostic_log.md`. Those two remain the authoritative, most detailed
record — this is a compressed synthesis of them.

---

## 1. The question

Can `coa_phase` (φc, orbital phase at coalescence) and `polarization_angle`
(ψ) be recovered from gravitational-wave strain alone by a neural network,
or is there a fundamental degeneracy — plausibly a sin/cos combination
ambiguity between the two — that makes them jointly unrecoverable from
strain-only input regardless of architecture or loss engineering?

Both are periodic quantities, trained with a circular loss
(`1 − cosΔθ`, via `normalize_unit`) rather than plain regression.

## 2. Experimental design

Two head-parameterization **modes**, crossed with several trunk
**architectures**:

- **baseline mode**: individual circular-loss heads directly on φc and ψ.
- **poc mode**: `SumDiffTrainer` — combo_A/combo_B (sum/diff) reparameterized
  heads plus inclination-dependent curriculum weighting `w(ι)`, on the
  theory that the sum/diff combination might be better-conditioned than
  the raw angles even if φc/ψ individually are degenerate.

Trunks used: `tcn`, `cnn_baseline`, `cnn_attention`, `inception_time`,
`resnet1d`. Named per-model configs in the run logs:
- `poc_a` = baseline mode, tcn trunk
- `poc_b` = poc mode, tcn trunk
- `tcn`, `cnn_attention`, `cnn_baseline`, `inception_time`, `resnet1d` =
  baseline mode, that trunk

Other heads present for context / health-checking (not the focus, but
useful negative controls): `mchirp`, `merger_time`, `snr`, `sky_position`,
`inclination`.

## 3. Chronological run log

### Runs 1–4 (2026-07-16 – 2026-07-18) — dead on arrival, wrong root cause chased first

φc/ψ heads mode-collapsed to a constant (φc → 315°, ψ → 67.5° in the
earliest baseline run). Diagnostic checks 1–4 traced this to `tanh`
activation on PERIODIC heads saturating at random initialization — the
heads were born dead before training started (Check 6/7 confirmed
saturation at step 0).

### Run 5 (2026-07-18) — tanh→linear fix; new pathology found

Fixed `activation="tanh"` → `"linear"` in `heads_spec.py` for periodic
heads. Retrained all 7 configs. Mode collapse persisted anyway — the tanh
fix was necessary but not sufficient. Traced to a second, independent bug:
**`normalize_unit` gradient attenuation** — the raw pre-normalization
vector magnitude `|v_raw|` was drifting far from 1, which suppresses the
gradient that flows back through the unit-normalize step used to convert
raw (x, y) outputs into an angle.

Initial (later superseded) read: inclination showed the identical failure
signature, which briefly looked like evidence the whole degeneracy was
"fundamental." That was walked back — inclination's failure was traced
to a *different* mechanism entirely (Huber loss, no `normalize_unit`
involved at all — see `inclination_loss_trace.md`), so it isn't evidence
either way for the φc/ψ question.

### Run 6 (2026-07-20) — magnitude penalty implemented

Added an explicit penalty term `λ·(|v_raw| − 1)²` to pull the raw vector
magnitude back toward 1 and stop the gradient-attenuation problem. Also:
code-traced the inclination loss path (confirmed separate mechanism, not
a confound for φc/ψ) and ran five pre-flight checks to rule out other
silent-failure modes before committing to a full retrain.

### Run 7 (2026-07-20) — magnitude penalty retrain, λ=0.01, 4 models

Retrained `poc_a`, `poc_b`, `tcn`, `cnn_attention` with
`magnitude_penalty_lambda: 0.01`.

**Result:** the magnitude penalty does what it's supposed to — `|v|` drift
is controlled for most models. But **periodic heads still don't learn**:
circular loss for φc/ψ sits flat at the random-baseline value (~1.0)
across all 80 epochs, for every model. The combo heads (`poc_b`) don't
learn either — combo-level circular loss also stays at random baseline.

Diagnostic checks 1–7 re-run on the new checkpoints; a five-section
verification plan (`run7_verification_plan.md`) was then executed to
systematically rule out confounds before trusting the null result:

| Section | What it checked | Finding |
|---|---|---|
| A | Gating checks (A.1–A.3) | A.1 pass. A.2: std_ratio healthy for only 2/4 models — the two problems this investigation's λ-retune work (Run 9) was aimed at. A.3: a 89× asymmetry in prediction rel_change vs. a perturbed `mchirp` input — flagged as needing the multi-step perturbation trace to resolve (still open, see §5). |
| B | `poc_b` config diff | No config bug. `poc_b`'s *worse* collapse than `poc_a` is explained by the curriculum weighting funneling gradient through one underdetermined combo channel — a curriculum+degeneracy interaction, not evidence of a bug or of learning. |
| C | `cnn_attention` config diff | No hidden config difference. Its lower `circ_r` is a feature-variance artifact of learned attention pooling, not phase learning — `ang_MAE` sits at the same null baseline as everything else. |
| D | Bootstrap significance (`bootstrap_ang_mae.py`, N=10,000 shuffles) | **11 of 12 model×head combinations non-significant.** For coa_phase and pol_angle specifically: **0 of 4 models significant.** |
| — | Validation row-ordering check | Data confirmed i.i.d. (window variance ratio ≈ 0.99, no row-index correlation) — the bootstrap shuffle-null test is valid, not confounded by ordering. |
| E | SNR stratification (`snr_stratification.py`, tercile analysis) | **No model shows SNR-dependent improvement on any periodic head.** Rules out "the aggregate metric looks null but improvement is hiding in high-SNR events" as an explanation. |

One loose thread survived verification: validation circular loss for
coa_phase/pol_angle **increased slightly** over 80 epochs in several
models (e.g. poc_a coa_phase 0.995→1.020) instead of staying perfectly
flat — this needed an explanation before being waved off (a *rising* loss
could mean the penalty was actively pushing predictions the wrong way,
which would be an engineering artifact worth fixing, not more null
evidence). This became Run 8.

### Run 8 (2026-07-21) — λ=0 ablation

Reran `poc_a`- and `tcn`-equivalent configs with
`magnitude_penalty_lambda: 0.0` to isolate whether the Run 7 val-loss creep
was caused by the penalty itself (or its interaction with the shared
log-var uncertainty weighting), rather than being a real anti-learning
signal.

| Model | Head | λ=0 Δ (val circ loss) | λ=0.01 Δ | Verdict |
|---|---|---|---|---|
| poc_a | coa_phase | +0.0010 | +0.0249 | drift absent at λ=0 |
| poc_a | polarization_angle | +0.0002 | +0.0163 | drift absent at λ=0 |
| tcn | coa_phase | +0.0011 | +0.0204 | drift absent at λ=0 |
| tcn | polarization_angle | +0.0072 | +0.0136 | **persists, ~half magnitude** |

**3 of 4 signals resolved as artifact**: with the penalty off, val circular
loss is flat within noise — the λ=0.01 creep was a penalty/log-var
interaction quirk, not real anti-learning. **1 of 4 (tcn pol_angle)**
doesn't resolve cleanly — still drifts upward at λ=0, about half the
λ=0.01 magnitude. Small (0.007 on a loss near 1.0) and could be noise, but
explicitly not rounded away into "resolved."

`std_ratio` diverged hard without the penalty as expected, confirming the
ablation was a genuine off-state rather than a no-op. Final validation MAE
at λ=0 landed at the same null values as every other run (coa_phase MAE ≈
1.579 ≈ theoretical null π/2 = 1.571; pol_angle ≈ 0.780–0.785 ≈ null
π/4 = 0.785).

This closed verification item F.1/F.2. It did **not** address the two
remaining std_ratio problems (tcn coa_phase, poc_a pol_angle) — λ=0 makes
`std_ratio` worse by construction, so it can't inform whether λ=0.05–0.10
would fix them. That became Run 9.

Full write-up: `assessment_lam0_ablation_2026-07-22.md` (**note:** as of
this summary's date, that file's §4/§5 still describe the λ retune as
open/hypothetical — it needs updating to reflect Run 9a/9b below; see
`doc_update_sweep_handoff_2026-07-22.md`).

### Run 9 pre-registration (2026-07-22) — locking the decision criterion before looking

Before running the λ=0.05/0.10 retune, a reviewer raised a structural
concern: this investigation had repeatedly had aggregate metrics look
better or worse than they actually were (mode collapse posing as
R²=0.75 early on; an endpoint-only std_ratio summary that missed a
mid-training crash-and-recovery; the Run 8 val-loss creep that turned out
to be a λ-interaction artifact rather than real signal). Rather than
eyeball the retune result after the fact, the success/failure criterion
was written down **in advance** — see `preregistration_lam_retune.md`.

Scope: exactly **two** primary pre-declared tests — `tcn`/coa_phase and
`poc_a` (baseline)/polarization_angle. Everything else is exploratory
only, not counted toward the verdict.

Decision procedure (mechanical, computed by `diagnostic_lam005_retune.py`
/ `diagnostic_lam010_retune.py`, not eyeballed):

- **Step 0 (interpretability gate):** std_ratio counts as healthy if
  <10% of the last 40 epochs fall outside [0.5, 2.0] AND the linear trend
  over those 40 epochs is within ±0.005/epoch. Gate fail at λ=0.05 → try
  λ=0.10 before any conclusion. Gate fail at λ=0.10 too → report "λ alone
  insufficient," **not counted either way** (neither null nor
  counter-evidence).
- **Step 1 (significance):** bootstrap shuffle-null (N=10,000, same method
  as Run 7 Section D), Bonferroni-corrected for 2 tests → p < 0.025.
- **Step 2 (effect-size floor):** Δang_MAE (null theory − observed) ≥ 0.10
  rad (~5.7°) — set ~3× above the cnn_attention inclination effect judged
  non-compelling in Run 7, and ~8× above the row-ordering artifact bound.
- **Step 3 (SNR-monotonicity):** improvement must be monotonic
  non-decreasing across SNR terciles, and the high-SNR tercile's own effect
  must independently clear the 0.10 rad floor.
- Only a **gate-pass + significant + big-effect + SNR-monotonic** result
  counts as counter-evidence to escalate; everything else short of that
  stays filed as null or uninterpretable, per an explicit decision table in
  the pre-registration document.

Also built in this step: `config_lam005_retune{,_tcn}.yaml`,
`config_lam010_retune{,_tcn}.yaml`, `run_lam005_retune.py`,
`run_lam010_retune.py` (each chains `train_poc.py` → `plot_poc.py` →
`evaluate_poc.py` → its diagnostic script, then overlays a 3-point λ sweep
against λ=0 and λ=0.01), `diagnostic_lam005_retune.py`,
`diagnostic_lam010_retune.py`.

### Run 9a (2026-07-22) — λ=0.05 result

**Both primary tests failed the Step 0 gate.**

| Model | Head | frac unhealthy (last 40 ep) | trend/ep | Gate |
|---|---|---|---|---|
| tcn | coa_phase | 0.05 | −0.00638 | FAIL |
| poc_a (baseline) | polarization_angle | 0.35 | +0.00718 | FAIL |

Mechanical verdict for both: **UNINTERPRETABLE** — Steps 1–3 correctly
never ran. Both failures are close, and share a shape: each settles into a
stable, healthy std_ratio plateau only in the back portion of training
(tcn coa_phase: 0.58–0.62 for the last ~15 epochs; poc_a pol_angle:
0.53–0.56 for the last ~20), but the 40-epoch gate window also captures an
earlier transient while std_ratio was still climbing into the band — that
transient alone is enough to fail the strict frac/trend thresholds even
though the *current* state looks healthy. Circular loss for both stayed
flat at 1.004–1.008 throughout — no movement, same as every prior run.

Per the pre-committed plan, the next step was λ=0.10.

### Run 9b (2026-07-22) — λ=0.10 result

**Both primary tests failed the gate again — and worse than at λ=0.05.**

| Model | Head | frac unhealthy | trend/ep | Gate | vs λ=0.05 |
|---|---|---|---|---|---|
| tcn | coa_phase | 0.28 | −0.00255 | FAIL | worse (0.05→0.28) |
| poc_a (baseline) | polarization_angle | 0.73 | +0.00731 | FAIL | much worse (0.35→0.73) |

Unlike Run 9a, this was not a near-miss, and the two failure shapes
diverged from each other:

- **poc_a polarization_angle:** crashes hard through epochs ~30–48 (down
  to 0.18–0.4), then climbs steadily and monotonically, crossing 0.5 only
  in the last ~11 of 80 epochs (0.501→0.513). Plausibly still converging,
  but the gate correctly doesn't credit an in-progress recovery.
- **tcn coa_phase:** oscillates in roughly [0.2, 0.95] across the entire
  window with no discernible convergence in either direction — genuinely
  unstable, not a one-time transient.

Non-primary (exploratory) combos mostly regressed too: `poc_a` coa_phase
went HEALTHY (λ=0.05) → STILL UNHEALTHY (λ=0.10); `tcn` pol_angle went
"improved, not stable" → STILL UNHEALTHY. λ=0.10 helped none of the four
head/model combinations and made three worse.

**Verdict, per the pre-registered table, not re-derived after seeing the
result:** λ alone is insufficient to stabilize std_ratio for either
primary target. Filed as **neither null nor counter-evidence** — the
pre-registration anticipated exactly this outcome and specified it be
reported this way rather than forced into either bucket. The λ-sweep
(0, 0.01, 0.05, 0.10) is now exhausted for these two heads; the next
lever, if pursued, is architecture-level, not a further λ value.

## 4. Current hypothesis status (as of Run 9b)

The degeneracy hypothesis — **φc/ψ carry no strain-only recoverable signal
in this architecture/loss combination** — remains the best-supported
reading of the evidence:

- Circular loss for both heads sits flat at the random-baseline value
  (~1.0) in every run, every model, every λ value tested (0, 0.01, 0.05,
  0.10) — 9 runs deep, no exception.
- 11/12 model×head combinations non-significant by bootstrap
  (Run 7 Section D); coa_phase/pol_angle specifically: 0/4 significant.
- No SNR-dependent improvement on any model/head (Run 7 Section E) — rules
  out the "improvement is hiding in high-SNR events" population-bias
  explanation.
- No case has cleared even Step 0 of the pre-registered bar, let alone
  Steps 1–3 — there has been no counter-evidence at any point.

The std_ratio/λ-tuning work (Runs 8, 9, 9a, 9b) was pursued to clear an
**interpretability confound** (an unhealthy `|v|` magnitude makes the
circular-loss/MAE numbers themselves untrustworthy), not because fixing it
was expected to unlock learning. Its failure to fully stabilize is a
separate, secondary engineering result — it does not itself weaken or
strengthen the degeneracy conclusion, it just means two of the many
model/head combinations remain formally uninterpretable rather than
cleanly null.

## 5. What's resolved vs. still open

**Resolved:**
- Root cause chain: tanh saturation (Run 3–4) → fixed (Run 5) →
  normalize_unit `|v|` drift found (Run 5) → magnitude penalty implemented
  (Run 6) → penalty controls drift but heads still don't learn (Run 7).
- Verification Sections A (partial), B, C, D, E — confounds ruled out
  (config bugs, curriculum interaction, attention-pooling artifact,
  significance, SNR bias).
- Run 7's val-loss creep — 3/4 signals resolved as a λ/log-var interaction
  artifact (Run 8); tcn pol_angle's small residual drift (+0.0072 at λ=0)
  noted as unexplained-but-small, not forced into the majority pattern.
- λ retune for tcn coa_phase / poc_a pol_angle — **closed** (Run 9a/9b):
  λ alone insufficient at either 0.05 or 0.10; filed as neither null nor
  counter-evidence, per pre-registration.

**Still open:**
- **A.3 multi-step prediction-perturbation trace** — meant to resolve the
  89× rel_change asymmetry vs. a perturbed `mchirp` input flagged in Run 7.
  Implemented inside `diagnostic_lam005_retune.py` /
  `diagnostic_lam010_retune.py`, but gated behind each config's Step 0
  passing — since neither primary target ever passed the gate, this trace
  has never run. Blocked by design, not forgotten; whether it's still
  worth running via a standalone script (decoupled from the λ-retune gate)
  is an open question, not yet decided.
  **Update (2026-07-23): CLOSED**, after a three-round sequence worth
  recording: (1) trace decoupled into `perturbation_trace_standalone.py`
  and executed; (2) review caught a failed mchirp positive control at the
  converged checkpoints and a wrong (marginal, not paired) noise
  comparator; (3) the instrument gained per-sample paired statistics and
  an `early` calibration stage (fresh init + ~1-epoch warmup) — which
  FAILED its pre-stated criterion: the displacement-geometry classifier
  labeled the fastest-learning head (early mchirp, paired t = −3.4 to
  −8.5) "noise-like" and was retired. On the paired probe-loss channel,
  which passed its control in that same run, every periodic head is null
  at both training stages (early |t| ≤ 1.6, final |t| ≤ 1.7); the
  tcn/coa_phase escalation trigger dissolves (paired t = −0.20). Verdict:
  the 89× asymmetry is radial movement without angular learning. Nothing
  from the Run 7 verification battery remains open. See
  `diagnostic_log.md`'s calibration adjudication and
  `perturbation_trace_output/`.
- Whether the pre-registered 40-epoch/0.005-trend gate window is
  well-calibrated for the `plateau` LR schedule's ~15–20 epoch settling
  behavior — flagged during Run 9a as a real question, but explicitly
  *not* acted on before or after Run 9b, since any change to that
  criterion must be a dated, pre-registered revision made before seeing
  further results, not a retroactive fix. Also, Run 9b's failure modes
  (still-crashing / oscillatory) aren't simply "settled late" the way
  Run 9a's were, so this question may be moot for this specific decision
  even if worth revisiting for future gates.
- Architecture-level fix for tcn coa_phase / poc_a pol_angle std_ratio —
  the next lever named by both diagnostic scripts' own gate-fail message,
  not yet scoped or started.
- ι-conditioning experiments (test whether providing true inclination as
  an auxiliary input enables ψ/φc recovery) — planned, not started.
- Inclination failure investigation (separate mechanism, Huber loss, no
  `normalize_unit`) — traced but not resolved; documented as not a
  confound for φc/ψ, but not itself closed out.
- Sky_position degradation in `SumDiffTrainer` — mentioned as a possible
  open issue in `NOTES.md`, not yet investigated.
- **Documentation sweep** — several supporting `.md` files still describe
  the λ retune as open/hypothetical (written before Run 9a/9b landed).
  See `doc_update_sweep_handoff_2026-07-22.md` for the specific list and
  how to bring them into line with `NOTES.md`/`diagnostic_log.md`.

## 6. Key files (pointers, not duplicated here — see `experiment_index.md` for the full index)

- `NOTES.md`, `diagnostic_log.md` — primary running record, most detailed
  and most current.
- `preregistration_lam_retune.md` — locked decision criteria for Run 9a/9b
  (do not edit retroactively).
- `assessment_lam0_ablation_2026-07-22.md` — Run 8 write-up (stale re: λ
  retune status, pending the doc sweep).
- `tanh_to_linear_postmortem.md`, `inclination_loss_trace.md` — Run 1–6
  postmortems.
- `poc_b_config_diff.md`, `cnn_attention_config_diff.md`,
  `std_ratio_trajectories.md` — Run 7 verification Section B/C/A.2
  artifacts.
- `bootstrap_output/`, `snr_output/` — Run 7 verification Section D/E raw
  outputs.
- `lam005_retune_output/`, `lam010_retune_output/` — Run 9a/9b outputs
  (auto-generated reports + diagnostic logs; do not hand-edit).
- `experiment_index.md` — full file/script/config index (itself due for a
  refresh — see the doc-sweep handoff file).

## 7. Recommended next steps, in priority order

1. **Documentation sweep** (cheap, no compute) — bring the remaining
   stale `.md` files in line with the Run 9a/9b outcome. See
   `doc_update_sweep_handoff_2026-07-22.md`.
2. **Decide on the perturbation trace** — either scope a standalone run
   decoupled from the λ-retune gate, or explicitly close A.3 as
   "unresolved, not pursued further" with a stated reason.
3. **Scope an architecture-level fix** for tcn coa_phase / poc_a pol_angle
   std_ratio, if that thread is worth continuing — or explicitly decide
   it isn't, and move on.
4. **ι-conditioning experiments** — the original next major phase, gated
   on the above being either resolved or explicitly deprioritized.
