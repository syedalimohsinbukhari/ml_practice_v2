# Pre-registered Decision Criterion — λ Retune (poc_redo_b combo_A / combo_B)

**Written:** 2026-09-15, **before** any λ=0.05/0.10 training run exists for this redo.

**Purpose:** lock down, in advance, exactly what result counts as "still uninterpretable," "null," or "counter-evidence" for the redo's λ-retune, before seeing any retuned result. This mirrors `experiments/phic_psi_poc/preregistration_lam_retune.md` directly — same discipline, same reason: that document was written after this investigation (in its original form) had already been burned more than once by reading aggregate metrics after the fact and having them mean the opposite of what they first looked like. The redo has already reproduced one instance of exactly that risk in miniature — an earlier same-day NOTES.md entry read `weight_combo_A`/`weight_combo_B` as "still climbing" by eye, and a mechanical check (`diagnostic_logvar_gate.py`) showed that reading didn't hold up. Deciding this criterion now, before the retune exists, removes the option to reinterpret the result after the fact in either direction.

## Scope

**Two primary, pre-declared tests — the redo's actual central question, not a proxy for it:**

1. **`poc_redo_b` / `combo_A`** — the combo angle `2φc+2ψ`.
2. **`poc_redo_b` / `combo_B`** — the combo angle `2φc−2ψ`.

These are the two quantities the corrected formula is supposed to make learnable. Both are tested (not just the sign-dependent "well-constrained" one from `NOTES.md`'s Step 1.1, `combo_B` for cos ι>0) because `well_constrained_combo`/`sign_dependent_combo` is a *training-time* weighting choice, not a claim that the other combo carries zero signal — Step 1.1's own ratios (1.17×/1.19×) were modest, not "combo_A is pure noise." Testing only one would bias the read toward whichever the trainer already favors.

**`poc_redo_a` (baseline mode, same TCN trunk) is retrained in parallel at each trial λ, as a required control, not as a third primary test.** It never builds `combo_A`/`combo_B` — its role is to stay a valid, apples-to-apples comparison point (same architecture, same λ, only the gradient grouping differs, per `config_baseline.yaml`'s own header comment) so that any real signal found in `poc_redo_b` can be checked against whether `poc_redo_a`'s individual heads learn anything on their own. `poc_redo_a`'s own Step 0 gate must also pass — not because it's under test, but because an unhealthy control isn't a valid baseline to compare against. If `poc_redo_a`'s gate fails while `poc_redo_b`'s passes (or vice versa), that asymmetry is reported, not silently dropped.

Any other reading from these runs (e.g. `poc_redo_a`'s own coa_phase/polarization_angle ang_MAE, individual-head diagnostics) is exploratory only — computed and reported if convenient, but not part of the pre-registered verdict, and not to be promoted to "primary" if it happens to look interesting.

## Step 0 — Interpretability gate (std_ratio)

Unchanged from the original — same thresholds, same window, reused verbatim (`diagnostic_logvar_gate.py`'s `std_ratio_gate`, itself copied from `diagnostic_lam005_retune.py`):

**Healthy** = fewer than 10% of the last 40 epochs with `val_std_ratio` outside `[0.5, 2.0]`, **and** the linear trend over those 40 epochs is within ±0.005/epoch. Applied to **all four** readings — `poc_redo_b`'s `coa_phase` and `polarization_angle` (both feed both combos, both must be healthy for the combo construction to be trustworthy), and `poc_redo_a`'s `coa_phase` and `polarization_angle` (control validity).

- **All four must pass** for Step 0 to clear. A single failing head/run keeps the whole round UNINTERPRETABLE — the combo depends on both source vectors jointly (A.2), so one unhealthy vector taints both `combo_A` and `combo_B` regardless of the other's health.
- **Gate fails at λ=0.05** → try λ=0.10 before drawing any conclusion (same fallback order the original used, and the same two trial values — no reason to deviate, and reusing them keeps this round comparable to the original's own λ=0.05/0.10 attempts).
- **Gate fails at λ=0.10 too** → report as "λ alone insufficient for this architecture/head, post-formula-fix," same as the original's own outcome for its two primary targets. Not counted toward a null tally or as counter-evidence — a negative result about the retune, not about the physics or the formula.

## Step 1 — Statistical significance (only if Step 0 passes)

**New for this redo, not a copy of the original's procedure — stated explicitly because it is a genuine methodological choice, not an assumption.** The original's primary targets (`tcn`/coa_phase, `poc_a`/polarization_angle) were always baseline-mode heads with direct per-sample supervision, so `bootstrap_ang_mae.py`'s shuffle-null test on individual-head angular MAE was the right tool. `poc_redo_b`'s `coa_phase`/`polarization_angle` heads receive **no individual gradient** in poc mode (`trainer.py`'s `_poc_total_loss` — only `combo_A`/`combo_B` are trained directly); reconstructing individual φc/ψ from them requires the branch-ambiguity machinery in `formulae_reference.md` A.8, which needs ground truth to disambiguate and is documented as a validation-script-only technique, not something to build a significance test on.

Instead: **`combo_A` and `combo_B` are each, on their own, a single well-defined periodic-2π quantity** — exactly what `circular_loss_combo_A`/`circular_loss_combo_B` already measure in training. Run the same bootstrap shuffle-null procedure `bootstrap_ang_mae.py` uses, applied directly to each combo's own angle (`period = 2π`, `angular_mae` computed on `atan2` of the predicted vs. true combo vector, same wrap-aware residual as the original's `angular_mae` helper) instead of an individual head's angle.

**Because exactly 2 primary tests are pre-declared** (combo_A, combo_B), the significance threshold is Bonferroni-corrected for 2 comparisons: **p < 0.025** — same correction, same count, as the original's own retune round (which also had exactly 2 primary tests, coincidentally).

## Step 2 — Effect size floor

**Pre-registered floor: Δang_MAE (null theory − observed) ≥ 0.10 rad (≈5.7°), reused directly from the original's own floor**, not independently recalibrated for the redo's fresh checkpoints. This is an explicit, flagged assumption: the original derived 0.10 rad from artifact scales specific to its own checkpoints (≈3× the cnn_attention/inclination population-bias effect, ≈8× its row-ordering-confound bound). Those exact numbers have not been reproduced against the redo's checkpoints. Reusing the floor is defensible because it's the same physical system, same population, same general training pipeline (not a different experiment) — but if this round's effect size lands close to 0.10 rad rather than clearly above or below it, that proximity is itself a reason to rerun the original's artifact-scale checks (row-ordering, population-bias) against the redo's own checkpoints before trusting the floor, rather than treating 0.10 rad as untouchable.

Effects that are statistically significant but below 0.10 rad are **not** promoted to counter-evidence — flagged for independent replication, counted as null for this round's verdict, same treatment the original gave its own sub-floor significant result (cnn_attention/inclination).

## Step 3 — SNR-stratification consistency

Run `snr_stratification.py`'s tercile logic (same procedure), applied to each combo's own angular error rather than an individual head's.

**Required for counter-evidence:** improvement must be monotonic non-decreasing with SNR tercile (`monotonic_improves=True`), **and** the high-SNR tercile's own Δ-from-null must independently clear the 0.10 rad floor (not just the pooled effect) — same requirement, same rationale as the original: a real per-sample strain→angle mapping should be easiest to recover in the loudest events, while a population-level bias improves uniformly regardless of signal strength.

## Final decision table (computed mechanically, not eyeballed)

Applied independently to `combo_A` and `combo_B` (2 verdicts, one per combo — not collapsed into one):

| Step 0 (gate, all 4 readings) | Step 1 (p) | Step 2 (effect ≥0.10 rad) | Step 3 (SNR-monotonic, high-SNR ≥0.10 rad) | Verdict |
|---|---|---|---|---|
| fail | — | — | — | **UNINTERPRETABLE** — retune λ further (0.05→0.10), or report "λ alone insufficient" if 0.10 also fails |
| pass | p ≥ 0.025 | — | — | **NULL** — clean data point: the corrected formula, with |v|-space healthy, still shows no learning |
| pass | p < 0.025 | Δ < 0.10 rad | — | **NULL** (flag for independent replication, per the original's cnn_attention precedent) |
| pass | p < 0.025 | Δ ≥ 0.10 rad | flat / non-monotonic | **NULL** (flag — population-bias signature, per the original's cnn_attention precedent) |
| pass | p < 0.025 | Δ ≥ 0.10 rad | monotonic, high-SNR ≥0.10 rad | **COUNTER-EVIDENCE** — the corrected formula recovers real signal; escalate immediately, this would be the headline result of the whole redo |

If `combo_A` and `combo_B` land on different rows of this table, both verdicts are reported as-is — no averaging, no picking the more interesting one, same "exceptions are reported per-case, never averaged away" rule this repo's `CLAUDE.md` states for living docs generally.

## Commitment

This table will be implemented directly in a diagnostic script (mirroring `diagnostic_lam005_retune.py`/`diagnostic_lam010_retune.py`) as an automated final verdict, computed from the retuned checkpoints with no manual threshold-picking after the fact. If a future re-read of the results disagrees with the mechanical verdict, that disagreement itself gets written down as a dated note on this document, not resolved by quietly picking a different threshold — same commitment the original made, and the same one this redo already had to honor once today (the `weight_combo_A/B` correction in `NOTES.md`).

## Next actions (as written 2026-09-15, before any retune existed)

- [ ] Build `config_lam005_retune.yaml`/`config_lam005_retune_tcn.yaml`-equivalent configs for `poc_redo_b` and `poc_redo_a` at λ=0.05 (copy from `experiments/phic_psi_poc/config_lam005_retune*.yaml`, same local-import/naming conventions already established for this redo).
- [ ] Hand the λ=0.05 retrain to the lab GPU machine — this machine is CPU-only.
- [ ] Rerun `diagnostic_logvar_gate.py`'s Step 0 gate on the results before touching Steps 1–3.
- [ ] Only if Step 0 passes at λ=0.05 (or, failing that, at λ=0.10): implement and run Steps 1–3 per this document.

---

## Dated addendum — 2026-09-15, λ=0.05 result

Per this document's own commitment ("frozen as of 2026-09-15... dated addenda only" — `experiment_index.md`'s description of this file), the checklist and criteria above are left exactly as written; this addendum records what happened against them, not a rewrite of them.

The λ=0.05 configs were built, CPU-verified, and trained on the lab GPU machine (`runs/phic_psi_lam005_retune_b/20260915_154203`, `runs/phic_psi_lam005_retune_a/20260915_175801`). `diagnostic_logvar_gate.py`'s Step 0 gate was rerun against them per this document's own Step 0 section: **all four readings FAIL** (`poc_redo_b`'s coa_phase and polarization_angle, `poc_redo_a`'s coa_phase and polarization_angle — see `NOTES.md`'s 2026-09-15 "λ=0.05 retune" entry for the full numbers). Per this document's own decision table ("fail → next fallback lambda... FAIL → UNINTERPRETABLE" / "Gate fails at λ=0.05 → try λ=0.10 before drawing any conclusion"), the verdict is **UNINTERPRETABLE**, and the pre-committed next action is the λ=0.10 fallback — not a revision of this criterion. Steps 1–3 were not run; they remain gated on Step 0 passing, exactly as this document specifies.

`config_lam010_retune_b.yaml`/`config_lam010_retune_a.yaml` (λ=0.10) have been built and CPU-verified, per the pre-committed fallback order, and are queued for the lab GPU machine. This is the last λ value this preregistration's fallback order commits to; if it also fails, per this document's own Step 0 section, the correct report is "λ alone insufficient for this architecture/head, post-formula-fix" — not a new, unplanned λ value chosen after seeing results.

---

## Dated addendum — 2026-09-16, λ=0.10 result (final round under this preregistration)

Again, the checklist and criteria above are left exactly as written; this addendum records the outcome against them.

The λ=0.10 configs were trained on the lab GPU machine (`runs/phic_psi_lam010_retune_b/20260915_181803`, `runs/phic_psi_lam010_retune_a/20260916_020049`) and the Step 0 gate rerun. Result: **3 of 4 readings FAIL** (`poc_redo_a`'s coa_phase and polarization_angle, `poc_redo_b`'s polarization_angle); **1 of 4 PASSES for the first time** (`poc_redo_b`'s coa_phase — frac_unhealthy=0.000, trend=+0.00011, final std_ratio=0.930). Full numbers in `NOTES.md`'s 2026-09-16 "λ=0.10 retune" entry.

Per this document's own Scope section ("All four must pass for Step 0 to clear... A single failing head/run keeps the whole round UNINTERPRETABLE"), one passing reading does not clear the round. Per this document's own Step 0 section's explicit instruction for exactly this case ("Gate fails at λ=0.10 too → report as 'λ alone insufficient for this architecture/head, post-formula-fix'... Not counted toward a null tally or as counter-evidence"), **that is the verdict of record: λ alone (0.01/0.05/0.10) is insufficient to stabilize `|v|`-space for TCN's coa_phase/polarization_angle heads under this formula, in either poc or baseline mode.**

This preregistration's fallback order commits to no λ value beyond 0.10. Steps 1–3 were never run, because Step 0 never cleared at any tried λ — exactly the gating this document specifies. No further λ trial is authorized under this document; a new λ value chosen now, after seeing this result, would be precisely the after-the-fact threshold-picking this document exists to prevent. This closes out this preregistration's scope. Whether to attempt a different stabilization mechanism, or treat this as the terminal state for the redo's `combo_A`/`combo_B` question on this architecture, is outside this document's remit and not decided here.

**Postscript, same date:** the one PASSing reading above (`poc_redo_b`'s coa_phase) was subsequently checked against its training-time scatter plots and found to be a std_ratio false-positive — a mode-collapsed constant output, not real per-sample learning (full detail in `NOTES.md`'s 2026-09-16 correction). This doesn't change the verdict recorded above (the round already failed mechanically on the other 3 readings regardless), but it means even the appearance of a near-miss here was weaker than the raw numbers suggested — nothing in this round came close to real signal.
