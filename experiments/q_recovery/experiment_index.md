# q Recovery — Experiment Index

**Branch**: `q_value`
**Last updated**: 2026-07-27

---

## Core narrative

| File                                                                             | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
|----------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`q_investigation_history_2026-07-27.md`](q_investigation_history_2026-07-27.md) | Dated snapshot recapping the prior `q`-head investigation from `git log`/`git diff` on commits `8c2ca373`..`de847fe8` (2026-07-13 to 2026-07-15): diagnosed cause (per-head loss-weight clamp saturation + sigmoid saturation + a real q×mchirp data confound), the fixes implemented (per-head `log_var_clamp`, q-head regularization, widened `UNIT_AFFINE` bounds, a dedicated q branch, targeted oversampling), concrete before/after run numbers, and the still-open items at the point `q` was shelved out of `DEFAULT_HEADS` on 2026-07-15. Also recaps the parallel, unrelated sky_position/vMF head refactor that displaced `q` from the default head set. |

---

## What's still open?

Carried forward from `q_investigation_history_2026-07-27.md` §A.7 — none of this has been acted on yet on this branch:

1. Input-level augmentation targeted at the hard `q_high × mchirp_low` regime — never attempted.
2. Ensemble / auxiliary-task approach for q — deferred.
3. `pytest -m "not slow"` on the lab machine — never run in the source commit range.
4. A combined re-run of all five trunk configs measuring the Phase 2.5+3.1+3.2 changes together — not done.
5. A clamp-value sweep (1.5/2.0) on tcn/cnn_baseline — optional, not done.
6. `resnet1d` needs q-head regularization plus a re-clamp to 1.0 (with warmup kept) — it currently overfits q again after being revived from its dead-sigmoid state.
7. InceptionTime's dead `merger_time` head — confirmed real, out of scope for q, unresolved.
8. A Phase 3.3 auxiliary classifier — deferred, contingent on items 1–2 plateauing.

Reference target if this resumes: TCN's train/val R² gap on `q` (0.006 clamp-only, 0.060 with oversampling) is the baseline any other trunk's fix should match or beat.