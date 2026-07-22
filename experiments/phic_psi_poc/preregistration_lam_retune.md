# Pre-registered Decision Criterion — λ Retune (tcn coa_phase / poc_a pol_angle)

**Written:** 2026-07-22, **before** any λ=0.05/0.10 training run exists.
**Purpose:** lock down, in advance, exactly what result counts as "still null"
vs. "started to learn" for the two targeted retune combinations. This
investigation has repeatedly been burned by reading aggregate metrics after
the fact and having them mean the opposite of what they looked like —
mode-collapse masquerading as R²=0.75 (`NOTES.md`, Round 1), an endpoint-only
std_ratio summary that missed a mid-training crash-and-recovery in tcn
pol_angle (`diagnostic_log.md` Run 7, A.2), a rising validation loss that
turned out to be a λ-interaction artifact rather than active anti-learning
(`lam0_ablation_output/lam0_ablation_report.md`). Deciding the criterion now,
before seeing λ=0.05/0.10 results, removes the option to reinterpret the
result after the fact in either direction.

## Scope

Two primary, pre-declared tests — one per combination this retune round
exists to fix:

1. **tcn / coa_phase** — std_ratio still declining at λ=0.01 (0.34, −0.008/ep).
2. **poc_a / polarization_angle** — std_ratio stable but below 0.5 at λ=0.01
   (0.44).

Any other model×head combination produced by these runs (e.g. poc_a
coa_phase, tcn polarization_angle) is exploratory only — computed and
reported, but not part of the pre-registered verdict, and not to be quietly
promoted to "primary" if it happens to look interesting.

## Step 0 — Interpretability gate (std_ratio)

Before anything else, `|v|`-space must be healthy or the result is
uninterpretable — neither a null point nor counter-evidence, just evidence
the penalty still needs retuning.

**Healthy** = fewer than 10% of the last 40 epochs with `val_std_ratio`
outside `[0.5, 2.0]`, **and** the linear trend over those 40 epochs is within
±0.005/epoch (already implemented in `run_lam005_retune.py` /
`run_lam010_retune.py`, `compare_trajectories()`).

- **Gate fails at λ=0.05** → try λ=0.10 before drawing any conclusion.
- **Gate fails at λ=0.10 too** → report as "λ alone insufficient for this
  architecture/head"; do not count toward the null tally or as
  counter-evidence. This would be a new, useful negative result about the
  fix, not about the physics.

## Step 1 — Statistical significance (only if Step 0 passes)

Run `bootstrap_ang_mae.py`'s exact procedure (N=10,000 shuffle-null
bootstrap, one-sided test — is observed ang_MAE significantly *below* the
null distribution) on the retuned checkpoint for the specific head.

**Because exactly 2 primary tests are pre-declared**, the significance
threshold is Bonferroni-corrected for 2 comparisons: **p < 0.025** (not the
uncorrected 0.05, and not the 12-test 0.0042 from Run 7's broader sweep,
which doesn't apply here since this round only re-tests 2 combinations).

## Step 2 — Effect size floor

A p-value alone is not sufficient — Run 7 already showed a Bonferroni-losing
but nominally-significant result (cnn_attention inclination, z=+3.17,
p=0.0007, Δ=0.038 rad) that turned out to be a population-level bias, not
per-sample learning. And the row-ordering check bounded the maximum artifact
size an ordering confound could produce at ≈0.013 rad.

**Pre-registered floor: Δang_MAE (null theory − observed) ≥ 0.10 rad
(≈5.7°).** This is chosen now, before seeing results, to sit clearly above
both known-artifact scales (≈3× the cnn_attention effect, ≈8× the
row-ordering bound) — big enough that clearing it can't be explained by
either failure mode already catalogued in this investigation.

Effects that are statistically significant but below 0.10 rad are **not**
promoted to counter-evidence — they get the same treatment as the
cnn_attention case: flagged for independent replication, counted as null for
this round's verdict.

## Step 3 — SNR-stratification consistency

Run `snr_stratification.py`'s tercile logic on the retuned checkpoint.

**Required for counter-evidence:** improvement must be monotonic
non-decreasing with SNR tercile (`monotonic_improves=True`), **and** the
high-SNR tercile's own Δ-from-null must independently clear the 0.10 rad
floor (not just the pooled effect).

**Rationale:** a real per-sample strain→angle mapping should be easiest to
recover in the loudest events. A population-level bias (the cnn_attention
inclination pattern) improves uniformly across SNR terciles regardless of
signal strength — that shape is the signature of an artifact, not physics,
and is explicitly excluded from counting as counter-evidence here.

## Final decision table (computed mechanically, not eyeballed)

| Step 0 (gate) | Step 1 (p) | Step 2 (effect ≥0.10 rad) | Step 3 (SNR-monotonic, high-SNR ≥0.10 rad) | Verdict |
|---|---|---|---|---|
| fail | — | — | — | **UNINTERPRETABLE** — retune λ further |
| pass | p ≥ 0.025 | — | — | **NULL** — clean data point for the degeneracy conclusion |
| pass | p < 0.025 | Δ < 0.10 rad | — | **NULL** (flag for independent replication, per cnn_attention precedent) |
| pass | p < 0.025 | Δ ≥ 0.10 rad | flat / non-monotonic | **NULL** (flag — population-bias signature, per cnn_attention precedent) |
| pass | p < 0.025 | Δ ≥ 0.10 rad | monotonic, high-SNR ≥0.10 rad | **COUNTER-EVIDENCE** — escalate immediately, do not fold into null tally |

## Commitment

This table is implemented directly in `diagnostic_lam005_retune.py` and
`diagnostic_lam010_retune.py` as an automated final verdict, computed from
the trained checkpoint with no manual threshold-picking after the fact. The
scripts print the verdict from this table; if a future re-read of the
results disagrees with the mechanical verdict, that disagreement itself
should be written down as a note on this document, not resolved by quietly
picking a different threshold.
