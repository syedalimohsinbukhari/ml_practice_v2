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
| A.3 — multi-step perturbation trace | Open | Still open |
| tcn coa_phase λ retune (try 0.05–0.10) | Open | Still open |
| poc_a pol_angle λ check | Open | Still open |

Three of the four pre-ι-conditioning punch-list items from `diagnostic_log.md`
Run 7 remain outstanding. Only the loss-drift question has been closed.

## 5. Recommendation

The degeneracy conclusion (φc/ψ carry no strain-only recoverable signal, at
least not in a form this architecture/loss combination extracts) remains the
best-supported reading of the evidence, and this ablation removes one of the
loose threads rather than adding a new one. But per the interpretation guide
already in `diagnostic_log.md`, don't advance to ι-conditioning yet — the
remaining three items are cheap relative to the seven training runs already
done, and finishing them keeps the eventual go/no-go call airtight rather than
"probably fine":

1. **tcn coa_phase λ retune** (0.05, 0.10) — the highest-value remaining item.
   If a higher λ stabilizes std_ratio and the circular loss still doesn't move,
   that's a fifth clean-|v| null result strengthening the case. If it starts
   to learn, that's the first real counter-evidence in eight runs and would be
   worth chasing hard.
2. **poc_a pol_angle λ check** — cheaper, same logic, lower stakes (already
   stable, just at the wrong magnitude).
3. **Multi-step perturbation trace** — no retraining needed, just a script
   against existing checkpoints; resolves whether the 89× perturbation
   asymmetry (Run 7, Check 4) is directional learning or noise.

After those three, the ι-conditioning gate is clean to open.
