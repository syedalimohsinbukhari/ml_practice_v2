# Assessment — λ=0 Ablation (Run 8) and Overall φc/ψ Degeneracy Status

**Date:** 2026-07-22
**Inputs reviewed:** `NOTES.md`, `diagnostic_log.md` (Runs 1–7 + verification A–E),
`lam0_ablation_output/lam0_ablation_report.md`, commits `0f189f7`, `0a92813`, `0b99d92`.

## 1. What this run was for

Run 7's verification pass (2026-07-21) left one loose thread: validation circular
loss for coa_phase/pol_angle *increased* over 80 epochs (e.g. poc_a coa_phase
0.995 → 1.020, Δ=+0.025) instead of staying flat. A flat loss is consistent with
"no signal, constant prediction." A *rising* loss needed an explanation before it
could be waved off as more of the same null result — it could mean the magnitude
penalty (or its interaction with the log-var uncertainty weighting) was actively
pushing predictions in the wrong direction, which would be a tunable engineering
artifact, not evidence about the physics.

Run 8 reruns the two TCN-trunk baseline configs (`poc_a`-equivalent and
`tcn`-equivalent) with `magnitude_penalty_lambda: 0.0` and compares the val-loss
drift against the λ=0.01 numbers from Run 7.

## 2. Result

| Model | Head | λ=0 Δ (val circ loss) | λ=0.01 Δ | Verdict |
|---|---|---|---|---|
| poc_a | coa_phase | +0.0010 | +0.0249 | drift absent at λ=0 |
| poc_a | polarization_angle | +0.0002 | +0.0163 | drift absent at λ=0 |
| tcn | coa_phase | +0.0011 | +0.0204 | drift absent at λ=0 |
| tcn | polarization_angle | +0.0072 | +0.0136 | **drift persists** |

**3 of 4 signals: the upward creep is a λ-interaction artifact, not new
evidence.** With the penalty off, val circular loss is flat within noise
(Δ ≤ 0.001) for poc_a/coa_phase, poc_a/pol_angle, and tcn/coa_phase — the same
"stuck at random baseline" signature as everything else in this investigation.
The penalty term (or its coupling through the shared log-var uncertainty weight)
was responsible for the small upward trend seen with λ=0.01 active.

**1 of 4 signals doesn't resolve cleanly.** tcn/polarization_angle still drifts
upward at λ=0 (+0.0072), about half the λ=0.01 magnitude (+0.0136) rather than
vanishing. This is not "drift absent," and the report is right to flag it as
"persists — penalty not the cause" rather than force it into the majority
pattern. It's a small effect (0.007 on a loss that sits at ~1.0) and could still
be noise, but it's the one place in this ablation where the tidy explanation
doesn't fully apply.

Two secondary observations, neither surprising but worth recording:

- **std_ratio diverges hard without the penalty** (e.g. poc_a coa_phase:
  21.8 → 83.0 over the run) — this is the expected, previously-diagnosed
  `|v|`-drift behavior and confirms the ablation is doing what it says (penalty
  genuinely off, not a no-op).
- **Train circular loss decreases slightly at λ=0** while val loss stays flat
  (e.g. poc_a coa_phase train: 0.997 → 0.994; pol_angle train: 1.000 → 0.982).
  This is a small train/val split — mild memorization of training-set phase
  noise, not generalizable signal. Consistent with, not contradictory to, the
  no-learning conclusion; flagged here so it isn't later mistaken for evidence
  of learning if someone only reads train curves.
- **Final validation MAE at λ=0** (`metrics_validation.csv` for both runs)
  lands at the same null values seen throughout: coa_phase MAE ≈ 1.579 rad
  (null π/2 = 1.571), polarization_angle ≈ 0.780–0.785 rad (null π/4 = 0.785),
  inclination ≈ 1.54–1.58 rad. Removing the penalty doesn't unlock learning —
  it just removes the |v|-drift guardrail, as expected.

## 3. What this changes in the overall picture

This closes open item **F.1/F.2** from the Run 7 verification (`diagnostic_log.md`,
"Remaining open items" #1). The systematic upward creep that looked like it might
complicate the degeneracy conclusion turns out to be mostly a λ/log-var interaction
quirk, not a sign the models are being pushed toward worse predictions by real
data. That actually *strengthens* the case slightly by removing an unexplained
wrinkle — three of the four val-loss trajectories now have a clean, understood
explanation for their (small) drift.

The one exception (tcn/pol_angle, +0.0072 at λ=0) is minor in magnitude and
doesn't change the headline conclusion, but it should be named explicitly rather
than folded into "resolved" — the honest state is 3/4 explained, 1/4 unexplained-
but-small.

**This does not, on its own, move the tcn/coa_phase or poc_a/pol_angle std_ratio
problems forward.** Those are separate open items (λ retuning) and this ablation
wasn't designed to address them — λ=0 makes std_ratio *worse* by construction, so
it can't be used to judge whether λ=0.05–0.10 would fix tcn's still-declining
coa_phase std_ratio.

## 4. Updated status vs. the Run 7 verification checklist

| Item | Status before | Status now |
|---|---|---|
| F.1/F.2 — isolate increasing-loss trend | Open | **Resolved (3/4 clean; 1/4 small residual, noted)** |
| A.3 — multi-step perturbation trace | Open | Still open — blocked by design (gated behind Step 0 passing; see below) |
| tcn coa_phase λ retune (try 0.05–0.10) | Open | **Resolved — λ alone insufficient (Run 9a/9b)** |
| poc_a pol_angle λ check | Open | **Resolved — λ alone insufficient (Run 9a/9b)** |

**Update (2026-07-22, post Run 9a/9b):** the λ retune was carried out under a
pre-registered decision criterion (`preregistration_lam_retune.md`, locked
before either run) at λ=0.05 (Run 9a) and λ=0.10 (Run 9b). Both primary
targets failed the Step 0 interpretability gate at both values, and got worse
at λ=0.10, not better — tcn coa_phase went from 5% of late epochs unhealthy at
λ=0.05 to 28% at λ=0.10 with no discernible convergence; poc_a pol_angle went
from 35% to 73%, with a hard mid-training crash. Per the pre-registered
decision table, this outcome is filed as **neither null nor counter-evidence**
for the degeneracy hypothesis — it means λ alone cannot stabilize `std_ratio`
for either combination, not that the underlying phase signal was probed and
found absent. The λ-sweep (0, 0.01, 0.05, 0.10) is exhausted for these two
heads; the next lever, if pursued, is architecture-level. Full results:
`NOTES.md` and `diagnostic_log.md`, Run 9a/9b sections.

Only the perturbation trace (A.3) remains from this punch list, and it is
blocked by design rather than merely unscheduled: it's implemented inside
`diagnostic_lam005_retune.py` / `diagnostic_lam010_retune.py` but gated behind
each config's Step 0 passing, which never happened for either primary target.

## 5. Recommendation

The degeneracy conclusion (φc/ψ carry no strain-only recoverable signal, at
least not in a form this architecture/loss combination extracts) remains the
best-supported reading of the evidence, and this ablation removes one of the
loose threads rather than adding a new one.

**Update (2026-07-22):** items 1 and 2 below have since run to completion —
see the Section 4 update above. Neither started to learn; both failed the
std_ratio interpretability gate at λ=0.05 and λ=0.10, so neither result
counts toward the degeneracy verdict either way. Item 3 remains open, blocked
by design (see Section 4). Original recommendation text, for record:

1. ~~**tcn coa_phase λ retune** (0.05, 0.10) — the highest-value remaining item.
   If a higher λ stabilizes std_ratio and the circular loss still doesn't move,
   that's a fifth clean-|v| null result strengthening the case. If it starts
   to learn, that's the first real counter-evidence in eight runs and would be
   worth chasing hard.~~ Done (Run 9a/9b) — gate failed at both values.
2. ~~**poc_a pol_angle λ check** — cheaper, same logic, lower stakes (already
   stable, just at the wrong magnitude).~~ Done (Run 9a/9b) — gate failed at
   both values.
3. **Multi-step perturbation trace** — blocked behind Step 0 gate passing
   (see Section 4); not run at the time of this assessment.
   **Update (2026-07-23):** decoupled into a standalone un-gated script
   (`perturbation_trace_standalone.py`) and executed against the Run 7
   λ=0.01 checkpoints. **Provisionally closed** — the 89× asymmetry reads
   as coherent but dominantly radial raw-output drift (movement without
   angular learning), but same-day review found the mchirp positive
   control failed at the converged checkpoints, so the verdict awaits the
   script's `early`-stage calibration run; see `diagnostic_log.md`'s A.3
   closure section + review addendum and `perturbation_trace_output/`.

With the λ-sweep branch closed, the next decision is whether to scope an
architecture-level fix for these two combinations or deprioritize that thread
and move on to ι-conditioning regardless — see
`experiment_summary_2026-07-22.md` §7 for the current priority order.
