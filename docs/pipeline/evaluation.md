# Evaluation pipeline

Everything a human reads is in **physical units** (M☉, seconds, radians) and
**wrap-aware** for periodic heads — the `signed_error`/`abs_error` helpers in
`gwml/data/transforms.py` are the only sanctioned way to compute residuals.

## `scripts/evaluate.py`

```bash
python scripts/evaluate.py configs/resnet1d.yaml --split validation
# options: --split training|validation, --weights <file>
```

Steps: load config → rebuild the model (`build_trainer`) → build variables on
a dummy batch → `load_weights` (default `<run_dir>/best.weights.h5`) →
restore `transforms.json` (never re-fit) → write:

- `metrics_<split>.csv` — MAE and RMSE per active head
  (`gwml/evaluation/metrics.py::evaluate_model`)
- `scatter_<split>.png` — pred-vs-true grid, y=x diagonal, MAE annotated
- `residuals_snr_<split>.png` — mean |residual| binned by true SNR

## Physics sanity checks to apply to every run

1. **Errors must shrink with SNR** (`residuals_snr_*.png`, and the SNR
   terciles in `diagnostics.csv`). If low- and high-SNR errors are equal, the
   model isn't using the signal — suspect collapse or a data bug.
2. **No mass-range or time-localization bias** — compare `mchirp_low` vs
   `mchirp_high` and `merger_early` vs `merger_late` rows in
   `diagnostics.csv`.
3. **std_ratio ≈ 1, r2 well above 0** per head. `std_ratio → 0` is mean
   collapse; `std_ratio > 1` with poor r2 means the head is guessing with
   overconfident spread.
4. **`weight_<head>` history** (uncertainty weighting): a weight pinned at
   the clamp floor (~0.05) flags a head the network gave up on — decide
   whether that's physics (label not recoverable) or a bug before believing
   it.

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
