# SNR-Stratified Angular Separation — sky_position (Redo)

**Generated**: 20260918_153412
**N terciles**: 3
**Validation samples**: 5000
**Null expectation (theory)**: 90.00 deg (mean separation, two independent uniform points on S²)
**Materiality floor**: 5.73 deg (0.10 rad, reused from `preregistration_lam_retune.md`'s Step 2)

## Key question

Follow-up to `bootstrap_sky_separation.py` (2026-09-18): `cnn_attention` showed a floor-clearing, highly significant sky-localization effect (Δ=12.58°, z=+23.23σ), and a direct check ruled out the simplest population-bias explanation (always guessing the mean direction only scores ~89.4°, not 77.5°). This test is the discriminator: real per-sample recovery (e.g. via cross-detector H1/L1 amplitude-ratio information, which this 2-detector setup can partially provide independent of timing triangulation) should improve monotonically with SNR; a memorization-style shortcut has no reason to track signal quality.

| Model | ALL (deg) | Low SNR | Mid SNR | High SNR | Monotonic? | High-SNR Δ from null | Clears 0.10 rad floor? | Verdict |
|-------|-----------|---------|---------|----------|-------------|----------------------|-------------------------|--------|
| poc_a (baseline) | 86.69 | 86.56 | 86.52 | 86.99 | NO | +3.01 | no | NULL |
| poc_b (PoC) | 87.02 | 86.76 | 87.20 | 87.10 | NO | +2.90 | no | NULL |
| tcn | 87.11 | 86.12 | 87.74 | 87.47 | NO | +2.53 | no | NULL |
| cnn_attention | 77.52 | 80.15 | 77.95 | 74.47 | YES | +15.53 | YES | REAL-RECOVERY-CONSISTENT |

## Verdict logic (mirrors `preregistration_lam_retune.md`'s Step 3)

- **REAL-RECOVERY-CONSISTENT**: monotonic improvement with SNR *and* the high-SNR tercile's own Δ independently clears the 0.10 rad floor — the signature of a genuine per-sample strain→direction mapping, easiest to recover in the loudest events.
- **POPULATION-BIAS-SIGNATURE**: high-SNR Δ clears the floor but improvement is flat or non-monotonic across terciles — a population-level effect that doesn't require using the signal's own information content, consistent with a memorization-driven shortcut rather than real recovery.
- **NULL**: high-SNR tercile doesn't clear the floor either — same category as `poc_a`/`poc_b`/`tcn`'s pooled bootstrap result.
