# Testing pipeline

Written in-repo, **executed on the lab GPU machine** — the local box (ThinkPad
T530, CPU-only) runs nothing heavier than `python -m py_compile`.

```bash
pytest -m "not slow"     # quick suite: transforms, registry, losses, data, specs
pytest                   # + the slow overfit-one-batch tests
```

Config lives in `pyproject.toml` (`[tool.pytest.ini_options]`); `tests/conftest.py`
puts `src/` on `sys.path` (no editable install needed) and defines:

- `synthetic_params` — 200 fake rows within documented ranges, so most tests
  never need the real HDF5;
- `TINY_TRUNK_CFGS` — miniature configs for all five trunks;
- `requires_data` — auto-skip marker when `combined_repackaged.hdf` is absent.

## What each file covers

| file | guards against |
|------|----------------|
| `test_transforms.py` | broken round-trips (`inverse(transform(y)) ≈ y`), targets outside [0,1] for bounded heads, wrong shapes/dtypes, non-standardized z-scores, JSON persistence drift, silent use before `fit()` |
| `test_heads_spec.py` | registry accepting junk (`injection_time`, duplicates, unknowns), periodic encode/decode errors (incl. ψ's π-periodicity: ψ and ψ+π must encode identically), non-wrap-aware errors, wrong output dims for dynamic head sets |
| `test_registry.py` | a trunk failing to build, missing the `input_bn` contract, unnamed/mis-shaped outputs, sigmoid bounds not holding under extreme inputs |
| `test_losses.py` | missing/extra log-vars, unknown weighting keys, non-finite losses, uncertainty-at-init ≠ unit fixed weights, collapse metrics absent, clamp not applied, variance penalty inert, SNR weights non-monotonic, warmup touching the LR after handoff |
| `test_data_integrity.py` (needs HDF5) | shape drift vs `structure.md`, NaN/Inf in strain or params, params outside documented ranges, detector channel order swapped |
| `test_overfit.py` (slow, needs HDF5) | the deep check: every trunk must drive loss to <10% of its initial value on 32 real samples in 200 epochs. Failure means the *pipeline* (not the architecture) is broken — normalization, target routing, loss wiring |

## Workflow rules

1. **Before any long GPU run:** `pytest -m "not slow"` then
   `python scripts/train.py configs/smoke.yaml` — the smoke config (256
   samples, 2 epochs, `diagnostics_every_n: 1`) proves every callback
   actually writes its artifact.
2. **After touching losses.py or transforms.py:** run the full `pytest` —
   the overfit tests are the only ones that catch subtle sign/scale bugs
   that leave losses finite but unlearnable.
3. **New trunk:** add it to `TINY_TRUNK_CFGS` in `conftest.py`; the registry
   and overfit tests parametrize over that dict automatically. A trunk also
   needs its `docs/models/<name>.md` before it counts as merged.
4. **New head:** add the `HeadSpec`, then extend `test_heads_spec.py` with
   its round-trip; if it introduces a new transform or loss kind, the spec
   test must cover the new math.
