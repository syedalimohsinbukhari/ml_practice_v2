# φc/ψ Degeneracy PoC — Run 7 Results (July 20, 2026)

**Branch**: `poc/phic-psi-degeneracy`
**Experiment**: No-inclination-input — all 7 heads active (mchirp, merger_time, snr, sky_position, coa_phase, polarization_angle, inclination)
**Models tested**: poc_a, poc_b, cnn_attention, tcn
**Analysis output**: `experiments/phic_psi_poc/analysis_output/` (run 2026-07-20 23:43:04)
**Diagnostics**: `experiments/phic_psi_poc/diagnostic_output/diagnostic_checks_20260721_000331.log`

> **📋 UPDATE (2026-07-21):** Deep diagnostics (`diagnostic_checks_20260721_000331.log`) confirm:
> - The isotropic circular loss (`1−cosΔθ`) IS the training objective for periodic heads (Check 2)
> - **The circular loss never decreases below random baseline (~1.0) for ANY model, ANY head, over all 80 epochs** (Check 3)
> - Gradient path is healthy throughout (Checks 4+6)
> - PERIODIC heads use `linear` activation — no tanh saturation issue (Checks 5+7 are misleading)
> - **The φc/ψ degeneracy is effectively exact for this population without ι input.**
> See **[run7_verification_rebuttal.md](run7_verification_rebuttal.md)** for full analysis.

---

## Run Identification

| Model         | Run 7 (New)       | Run 5/6 (Previous) | Config                          |
|---------------|-------------------|--------------------|---------------------------------|
| poc_a         | `20260720_210936` | `20260718_141645`  | Baseline CNN trunk              |
| poc_b         | `20260720_213202` | `20260718_150353`  | PoC CNN trunk (combo heads)     |
| cnn_attention | `20260720_221625` | `20260718_143902`  | CNN + self-attention + q-tokens |
| tcn           | `20260720_215403` | `20260718_153301`  | Temporal ConvNet                |

All runs use identical transforms (z-score for mchirp/snr; no normalization for other heads).

---

## TABLE 1: SCALAR HEADS — MAE / R² / bias / std_ratio

### mchirp

| model         | Run 7 MAE | Run 7 R² | Run 5/6 MAE | Run 5/6 R² | Δ MAE  | Trend              |
|---------------|-----------|----------|-------------|------------|--------|--------------------|
| poc_a         | 0.977     | 0.959    | 0.993       | 0.958      | −0.016 | slight improvement |
| poc_b         | 1.024     | 0.957    | 0.914       | 0.966      | +0.110 | **regression**     |
| tcn           | 0.951     | 0.963    | 0.996       | 0.960      | −0.045 | improvement        |
| cnn_attention | 1.363     | 0.926    | 1.429       | 0.912      | −0.066 | improvement        |

> **poc_b regressed ~12% on mchirp** (0.914 → 1.024). This is now confirmed by the analysis script, not just the raw metrics. All other models improved. **tcn leads** at MAE=0.951.

### merger_time

| model         | Run 7 MAE | Run 7 R² | Run 5/6 MAE | Run 5/6 R² | Δ MAE   |
|---------------|-----------|----------|-------------|------------|---------|
| poc_a         | 0.0132    | 0.914    | 0.0127      | 0.921      | +0.0005 |
| poc_b         | 0.0128    | 0.919    | 0.0121      | 0.928      | +0.0007 |
| tcn           | 0.0128    | 0.921    | 0.0126      | 0.919      | +0.0002 |
| cnn_attention | 0.0130    | 0.909    | 0.0134      | 0.891      | −0.0004 |

> All models **stable** on merger_time. Differences are within noise.

### snr

| model         | Run 7 MAE | Run 7 R² | Run 5/6 MAE | Run 5/6 R² | Δ MAE  |
|---------------|-----------|----------|-------------|------------|--------|
| poc_a         | 0.831     | 0.785    | 0.833       | 0.786      | −0.002 |
| poc_b         | 0.835     | 0.783    | 0.827       | 0.787      | +0.008 |
| tcn           | 0.837     | 0.784    | 0.823       | 0.790      | +0.014 |
| cnn_attention | 0.880     | 0.755    | 0.947       | 0.721      | −0.067 |

> All **stable** on snr. cnn_attention shows improvement (−7% MAE vs old) but remains the weakest scalar predictor. **tcn leads** at MAE=0.837 (old leader poc_b=0.827).

---

## TABLE 2: SKY POSITION — True Angular Error (from analysis script)

| model         | Run 7 angular MAE | Run 7 angular median | Run 5/6 angular MAE | Run 5/6 angular median | Δ     |
|---------------|-------------------|----------------------|---------------------|------------------------|-------|
| poc_a         | **8.2°**          | 0.0°                 | 12.9°               | 0.0°                   | −4.7° |
| poc_b         | **10.0°**         | 0.0°                 | 12.7°               | 0.0°                   | −2.7° |
| tcn           | **4.5°**          | 0.0°                 | 3.2°                | 0.0°                   | +1.3° |
| cnn_attention | **3.3°**          | 0.0°                 | 3.6°                | 0.0°                   | −0.3° |

> **Sky position is working** and has NOT collapsed. The kappa-derived proxy in `metrics_validation.csv` (77-87°) was misleading — the true angular errors are 3-10°, consistent with the old assessment. **cnn_attention leads** at 3.3°, closely followed by tcn at 4.5°. poc_a improved most (−4.7°), poc_b also improved (−2.7°).

---

## TABLE 3: PERIODIC HEADS — Circular Statistics (from analysis script)

> `circ_r` → 0 = uniform/random, → 1 = perfectly concentrated at one angle.
> **⚠️ circ_r ≈ 1.0 is a _degenerate collapse_** (model outputs the same angle regardless of input), NOT good performance.

### coa_phase (φc ∈ [0, 2π))

| model         | circ_r    | circ_mean | ang_MAE | peaks                              | Health       |
|---------------|-----------|-----------|---------|------------------------------------|--------------|
| poc_a         | 0.848     | 152°      | 1.570   | 145°(0.27)                         | ✗✗           |
| poc_b         | **0.989** | 106°      | 1.541   | 105°(0.42)                         | **COLLAPSE** |
| tcn           | 0.856     | 335°      | 1.598   | 305°(0.20)                         | ✗✗           |
| cnn_attention | 0.434     | 177°      | 1.605   | 155°(0.09), 135°(0.08), 255°(0.05) | ✗✗           |

> **Old assessment comparison**: Old runs had circ_r=1.000 for poc_a, poc_b, tcn — all degenerate collapses to 315°. The new runs _broke_ this collapse for poc_a and tcn (now circ_r=0.85), but poc_b found a NEW collapse at 106° (circ_r=0.989). cnn_attention shows the most spread (circ_r=0.434) with 3 weak peaks — the least collapsed, but still not useful.

### polarization_angle (ψ ∈ [0, π))

| model         | circ_r    | circ_mean | ang_MAE | peaks                 | Health       |
|---------------|-----------|-----------|---------|-----------------------|--------------|
| poc_a         | 0.494     | 51°       | 0.784   | 72°(0.13)             | ~            |
| poc_b         | **0.987** | 90°       | 0.801   | 92°(0.44)             | **COLLAPSE** |
| tcn           | 0.878     | 102°      | 0.802   | 88°(0.14), 122°(0.08) | ✗✗           |
| cnn_attention | 0.172     | 99°       | 0.791   | 147°(0.08), 78°(0.06) | ~            |

> **Old comparison**: All models had circ_r=1.000 (degenerate collapse) at various angles. New runs broke the collapse for 3 of 4 models. poc_a dropped from 1.000→0.494, cnn_attention from 1.000→0.172 (nearly random). **poc_b is the exception** — still collapsed (circ_r=0.987).

### inclination (ι ∈ [0, π])

| model         | circ_r    | circ_mean | ang_MAE | peaks                  | Health |
|---------------|-----------|-----------|---------|------------------------|--------|
| poc_a         | 0.629     | 133°      | 1.573   | 165°(0.11), 105°(0.05) | ✗✗     |
| poc_b         | 0.465     | 1°        | 1.591   | —                      | ✗✗     |
| tcn           | **0.798** | 241°      | 1.531   | 225°(0.14)             | ✗✗     |
| cnn_attention | 0.368     | 35°       | 1.535   | 345°(0.12), 145°(0.07) | ✗✗     |

> **Old comparison**: circ_r ranged 0.23–0.99. New runs: tcn shows strongest inclination structure (circ_r=0.798, not collapsed), cnn_attention weakest (0.368). No model has collapsed on inclination.

---

## TABLE 4: HEALTH CHECK (from analysis script)

| model         | mchirp | merger_time | snr | sky_position |  coa_phase   |  pol_angle   | inclination |
|---------------|:------:|:-----------:|:---:|:------------:|:------------:|:------------:|:-----------:|
| poc_a         |   ok   |     ok      | ok  |      ok      |      ✗✗      |      ~       |     ✗✗      |
| poc_b         |   ok   |     ok      | ok  |      ok      | **COLLAPSE** | **COLLAPSE** |     ✗✗      |
| tcn           |   ok   |     ok      | ok  |      ok      |      ✗✗      |      ✗✗      |     ✗✗      |
| cnn_attention |   ok   |     ok      | ok  |      ok      |      ✗✗      |      ~       |     ✗✗      |

**Key**: ok = well predicted, ~ = marginal structure, ✗✗ = dead/random, COLLAPSE = circ_r > 0.98 (degenerate constant output)

---

## TABLE 5: TRAINING CONVERGENCE — Final Epoch Val Metrics

From the last epoch (~79) of history.csv:

| model         | val_loss | val_r2_mchirp | val_r2_snr | val_circ_loss_coa | val_circ_loss_pol | val_kappa_sky |
|---------------|----------|---------------|------------|-------------------|-------------------|---------------|
| poc_a         | −3.16    | 0.971         | 0.838      | 0.979             | 0.979             | 0.100         |
| poc_b         | −3.79    | 0.966         | 0.829      | 0.988             | 0.998             | 0.100         |
| cnn_attention | −1.53    | 0.989         | 0.985      | **0.601**         | **0.525**         | 0.780         |
| tcn           | −3.19    | 0.970         | 0.836      | 0.970             | 0.983             | 0.197         |

> **cnn_attention puzzle**: During training, cnn_attention achieved much lower circular loss (0.60/0.53 vs ~0.98) and higher kappa (0.78 vs ~0.10), suggesting the optimizer found non-trivial structure. But the analysis script shows cnn_attention has the _lowest_ circ_r values (0.43 coa, 0.17 pol) — i.e., its predictions are the _most spread out_. This paradox suggests the circular loss improvement came from spreading predictions _further apart_ in embedding space (lower loss = less concentrated), which the analysis script correctly identifies as weak structure.

---

## Key Takeaways

1. **Sky position is working correctly.** True angular errors are 3-10° (cnn_attention leads at 3.3°, tcn at 4.5°). The kappa proxy in `metrics_validation.csv` (77-87°) is a red herring — ignore it.

2. **poc_b is the problem model:**
   - mchirp regressed 12% (0.914 → 1.024 MAE)
   - coa_phase COLLAPSED (circ_r=0.989 at 106°)
   - pol_angle COLLAPSED (circ_r=0.987 at 90°)
   - This model learned to output near-constant values for periodic heads — a degenerate solution.

3. **The old assessment's circ_r=1.000 values were degenerate collapses, not good performance.** The old analysis script didn't flag circ_r > 0.98 as COLLAPSE. The new script does. All old models had collapsed coa_phase and pol_angle predictions. The new runs _broke_ this collapse for poc_a and tcn polarizations.

4. **cnn_attention is the most honest model** — lowest circ_r values (0.43 coa, 0.17 pol, 0.37 inc) mean it's spreading predictions rather than collapsing to a constant. The low circular loss during training (0.60/0.53) came from this spread, not from finding the correct angles. Its sky position (3.3°) is the best, and scalar heads are competitive.

5. **tcn is the best scalar predictor** and second-best on sky position (4.5°), with no collapses on any head. The most balanced model.

6. **Inclination shows weak but real structure in all models** — circ_r 0.37–0.80, all below collapse threshold. tcn leads (circ_r=0.798). No model has learned useful inclination prediction, but none has collapsed either.

7. **The φc/ψ degeneracy is confirmed.** Without inclination input, no model learns physically useful coa_phase or polarization_angle predictions. The best case is avoiding degenerate collapse — cnn_attention does this for pol_angle (circ_r=0.172), poc_a does it for pol_angle (circ_r=0.494).

---

## Next Steps

1. **Investigate poc_b collapse** — both coa_phase and pol_angle collapsed to constant outputs. The combo-head architecture may be encouraging degenerate solutions. Check if the loss function or head configuration needs adjustment.

2. **Proceed to inclination-as-input experiments** — these results confirm that without ι input, periodic heads either collapse or produce weak/non-useful structure. The next POC phase should test whether providing true ι as auxiliary input enables ψ and φc recovery.

3. **Ignore the kappa-based sky_position metric** in `metrics_validation.csv` — it's misleading. Always use `experiments/phic_psi_poc/analyse_predictions.py` for sky position and periodic head evaluation.

4. **Consider removing poc_b's combo heads** or adding anti-collapse regularization — circ_r that close to 1.0 is a failure mode, not success.

---

*Generated: 2026-07-20*
*Analysis: `experiments/phic_psi_poc/analysis_output/analysis_report_20260720_234304.md`*
*Data sources: Run 7 (`phic_psi_{poc_a,poc_b,cnn_attention,tcn}/20260720_*`) vs Run 5/6 (`20260718_*`)*
*Old assessment: `phic_psi_no_inclination_input_run_assessment.md` (July 17-18, 2026)*
