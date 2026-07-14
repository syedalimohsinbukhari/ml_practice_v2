# Evaluation pipeline

Everything a human reads is in **physical units** (M☉, seconds, radians) and
**wrap-aware** for periodic heads — the `signed_error`/`abs_error` helpers in
`gwml/data/transforms.py` are the only sanctioned way to compute residuals.

## `scripts/evaluate.py`

```bash
python scripts/evaluate.py configs/resnet1d.yaml --split validation
# options: --split training|validation, --weights <file>
```

Steps: load config → resolve the latest run directory under `run_dir` (legacy
flat runs are still accepted) → rebuild the model (`build_trainer`) → build
variables on a dummy batch → `load_weights` (default
`<latest_run_dir>/best.weights.h5`) → restore `transforms.json` (never re-fit)
→ write:

- `metrics_<split>.csv` — MAE and RMSE per active head
  (`gwml/evaluation/metrics.py::evaluate_model`)
- `scatter_<split>.png` — pred-vs-true grid, y=x diagonal, MAE annotated
- `residuals_snr_<split>.png` — mean |residual| binned by true SNR
- `residuals_mchirp_<split>.png` — mean |residual| binned by true mchirp
  (same `residuals_vs_param` helper as the SNR plot, different column)
- `logits_train_vs_val.png` — pre-sigmoid logit histograms (`ln(p/(1-p))`
  recovered from the raw sigmoid output, no architecture change needed),
  train vs val, split by true-value tercile, for every sigmoid/`UNIT_AFFINE`
  head (e.g. `q`, `merger_time`). Needs both splits loaded regardless of
  `--split`, so this always runs. See "sigmoid saturation" below.

## Physics sanity checks to apply to every run

1. **Errors must shrink with SNR** (`residuals_snr_*.png`, and the SNR
   terciles in `diagnostics.csv`). If low- and high-SNR errors are equal, the
   model isn't using the signal — suspect collapse or a data bug.
2. **No mass-range or time-localization bias** — compare `mchirp_low` vs
   `mchirp_high` and `merger_early` vs `merger_late` rows in
   `diagnostics.csv`. For any head with a `UNIT_AFFINE` + sigmoid activation,
   also check its own `<head>_low`/`<head>_mid`/`<head>_high` terciles and
   their cross-tab with `mchirp_low`/`mchirp_high` (e.g. `q_high_mchirp_low`)
   — a head can look fine in aggregate while one narrow cross-tab cell is far
   worse than the rest.
3. **std_ratio ≈ 1, r2 well above 0** per head. `std_ratio → 0` is mean
   collapse; `std_ratio > 1` with poor r2 means the head is guessing with
   overconfident spread.
4. **`weight_<head>` history** (uncertainty weighting): a weight pinned at
   its clamp floor — `exp(-log_var_clamp)`, ~5% for the default clamp of 3.0,
   higher for any head with a tighter per-head override — flags a head the
   network gave up on. A weight pinned at the *ceiling* instead
   (`exp(+log_var_clamp)`) — especially if every head hits the exact same
   ceiling value at the same epoch — means the uncertainty-weighting scheme
   has lost its per-head signal entirely (it's tracking train loss, not
   generalization); check train vs val R² divergence before trusting that
   head's predictions.
5. **Sigmoid saturation** (`logits_train_vs_val.png`): if train logits go
   extreme (large |value|) for one true-value tercile while val logits for
   the same tercile stay compressed near 0, that's the saturation-driven
   overfitting signature — the head has memorized rather than learned in
   that regime, and gradients through the sigmoid there are vanishing.

## Comparing trunks

Same config apart from `model.trunk`/`trunk_cfg`, same seed, then compare
`metrics_validation.csv` across `runs/*/`. Per-head MAE in physical units is
the headline number; check the SNR-binned curves before declaring a winner —
two trunks with equal overall MAE can differ meaningfully in the low-SNR
regime.

## Caveats

- R² on a periodic quantity (diagnostics CSV) is computed on wrapped
  residuals against the linear variance of the angles — fine for spotting
  collapse and trends, but not a rigorous circular-statistics R²; treat the
  MAE as primary for angle heads.
- The training-log metrics (`mae_*`, `r2_*`, `std_ratio_*` in `history.csv`)
  are in **normalized** space (fast, per-batch accumulation). The same
  quantities in `diagnostics.csv` are physical. Don't compare across the two.
- Scatter plots of periodic heads show points near the wrap seam far off the
  diagonal — the annotated MAE is wrap-aware even though the dots look bad.
