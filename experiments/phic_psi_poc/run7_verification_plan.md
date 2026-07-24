# Run 7 / Magnitude-Penalty Results — Consolidated Verification Plan

Purpose: nothing in Run 7 or the `analysis_report_20260720_234304.md` output
should be treated as a conclusion about the φc/ψ degeneracy — or about
whether the magnitude-penalty fix worked — until the items below are
resolved. This combines every open thread from the last two review rounds
into one ordered checklist. Do not proceed to ι-conditioning or any other
architectural pivot until Section A is fully cleared.

---

## A. Gating checks — must resolve before anything else is interpretable

### A.1 — Confirm the magnitude penalty was actually applied to these checkpoints

The single fact that determines whether `analysis_report_20260720_234304.md`
means anything. The timestamp suggests this is a distribution-script rerun
on Run 7's existing checkpoints, not a fresh post-fix retrain — if so, the
magnitude-penalty question from the prior round is still open, not answered.

- [ ] Confirm directly whether `poc_a`, `poc_b`, `tcn`, `cnn_attention` in
      this specific report were trained with `λ·(|v_raw|−1)²` active.
- [ ] If not: this report tells us nothing new about the fix — it needs to
      be regenerated from checkpoints that actually had it.
- [ ] If yes: proceed to A.2.

### A.2 — Pull the `std_ratio` trajectory for `coa_phase`/`polarization_angle`, full training run, all four models

Not just the final epoch. Success = stabilizes in a sane range (~0.5–2.0)
and stays there. Still diverging by the final epoch = `λ` needs tuning
before any conclusion is drawn, regardless of what the other metrics show.

### A.3 — Rerun the (name-matching-fixed) Check 4 prediction-perturbation test on these checkpoints

Confirm real gradient-driven weight movement resumed for `coa_phase`/
`polarization_angle` (`mean|Δ|` in the same order of magnitude as `mchirp`/
`snr`, not the `0.00` seen pre-fix). This is the most direct confirmation
the gradient path is actually healthy, independent of what final R²/MAE
show.

---

## B. `poc_b` — investigate as the priority anomaly, not the "best" result

`poc_b` is flagged `COLLAPSE` on both `coa_phase` and `polarization_angle`
in the health check — `circ_r ≈ 0.99` with a single peak carrying 42–44%
of predictions, i.e. **more severe mode collapse than the plain baseline
`poc_a`** (`circ_r` 0.85 / 0.49). Its marginally-better raw `ang_MAE` is a
product of that collapse landing near the true distribution by chance —
the same mechanism that made the earlier R²=0.754 look like real signal.
**Do not read `poc_b`'s MAE as evidence the sum/diff+curriculum design is
working — the concentration statistic says the opposite.**

- [ ] Diff `poc_b`'s config against `poc_a` (SumDiffTrainer-specific
      settings, curriculum weight application, combo-loss wiring) to find
      what's driving the more severe collapse.
- [ ] Check whether this shares a cause with the ~8% `mchirp` regression
      flagged in the Run 7 report (`0.962→1.036` MAE) — same run, same
      model, two anomalies; worth checking if they're one bug.
- [ ] Do not advance `poc_b` as the PoC's headline result until this is
      resolved.

---

## C. `cnn_attention` — diff against the other three, don't attribute to "better features" without evidence

Across both the Run 7 comparison and this analysis report, `cnn_attention`
is the consistent outlier — lowest `val_circ_loss` (0.601/0.525 vs ~0.98
for others in Run 7), best normalized SNR MAE, different `val_loss` scale
entirely (−1.53 vs −3.16 to −3.79), and now also the *lowest* `circ_r` in
this report (most spread-out, least collapsed, on both `coa_phase` (0.43)
and `polarization_angle` (0.17)). One architecture behaving categorically
differently from the other three, on every metric simultaneously, points
at a config difference, not emergent feature quality.

- [ ] Line-by-line config diff: `cnn_attention` vs `poc_a`/`poc_b`/`tcn` —
      learning rate, weight init scale, whether the magnitude penalty is
      even applied to this config, `log_var` initialization/calibration
      (the `val_loss` scale difference suggests this may differ).
- [ ] If a specific difference explains its healthier (less-collapsed)
      behavior, that's actionable — apply it to the other three configs
      and re-test, rather than treating `cnn_attention`'s numbers as a
      standalone curiosity.

---

## D. Statistical significance check on `ang_MAE` — needed before "learning" vs "noise" can be called either way

Every `ang_MAE` value across every model and every head in the latest
report sits within ~2% of its random-guessing baseline (`π/2` for
`coa_phase`/`inclination`, `π/4` for `polarization_angle`) — some *above*
baseline. Reduced mode collapse (`circ_r` moving off 1.0) is real and
separate from this; it does not by itself mean the small `ang_MAE`
deviations from baseline are real signal rather than sampling noise
around an unlearned mean.

- [ ] Bootstrap CI (or permutation test shuffling true labels) on
      `ang_MAE` for each model/head combination.
- [ ] Only treat a model/head as "learning something" if its `ang_MAE` is
      significantly better than its own random-baseline null distribution
      — not just numerically below `π/2` or `π/4` by a small margin.

---

## E. Extend the SNR-stratification check (Table 6 precedent) — final epoch, all models, both combo-relevant heads

The one partial run of this test so far (Run 7, `cnn_attention` only,
epoch 5, `coa_phase` only) showed flat-to-worse MAE across SNR terciles
even for the best-performing model — mildly informative, not decisive.

- [ ] Rerun at the **final epoch**, not epoch 5.
- [ ] All four models (`poc_a`, `poc_b`, `tcn`, `cnn_attention`), not just
      `cnn_attention`.
- [ ] Both `coa_phase` and `polarization_angle` — not just `coa_phase`.
- [ ] If even the highest-SNR tercile shows no improvement over the
      population-wide baseline, for every model, that's the first piece
      of evidence in this entire investigation that would actually speak
      to the physics rather than the engineering.

---

## Ordering

A (gating) → B and C (parallel, both are config-diff tasks) → D → E.

Nothing in Sections B–E should be read as evidence about the degeneracy
itself until Section A is fully resolved. If A.1 turns out negative (fix
wasn't actually applied to these checkpoints), stop here, apply it, and
regenerate this entire report before going further.
