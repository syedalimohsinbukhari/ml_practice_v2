# GW multi-head parameter estimation

Deep-learning regression of compact-binary parameters from dual-detector whitened strain in `combined_repackaged.hdf`, using a shared trunk with configurable output heads and a pluggable model zoo.
The default heads are chirp mass, merger time, SNR, sky position, and coalescence phase; mass ratio `q`, inclination, and polarization angle are registered and can be added in the config.

## Current status

Where the work stands, what is decided, and what is next: [`HANDOFF.md`](HANDOFF.md) (read first when resuming), with the defect and observation log in [`BUGS.md`](BUGS.md) and the pending checks in [`VERIFY.md`](VERIFY.md).
A regeneration of the dataset with a seeded, provenance-recording generator is planned and not yet built; until then the data is `combined_repackaged.hdf` as described below.

## Documentation map

| doc | what's in it |
|-----|--------------|
| [`HANDOFF.md`](HANDOFF.md), [`BUGS.md`](BUGS.md), [`VERIFY.md`](VERIFY.md) | working state: where we stand and the decisions, the defect and observation log, the pending checks |
| [`CLAUDE.md`](CLAUDE.md) | the working rules, the machines table, the related repositories, the commit-message format |
| [`GENERATOR_OVERHAUL.md`](GENERATOR_OVERHAUL.md) | what the data generator lacks, the planned output file, and the decisions for the regeneration |
| [`SEEDING.md`](SEEDING.md) | the seeding scheme for training and tests (the generator has its own in `ml-gw-search`) |
| [`PHIC_PSI_CARRYOVER.md`](PHIC_PSI_CARRYOVER.md) | frozen snapshot of what the φc/ψ experiment established for this branch |
| [`PLAN.md`](PLAN.md) | the design document: objectives, decisions and their rationale, milestones |
| [`docs/pipeline/overview.md`](docs/pipeline/overview.md) | end-to-end data flow, module map, design invariants |
| [`docs/pipeline/data.md`](docs/pipeline/data.md) | HDF5 loading, target transforms, dataset construction |
| [`docs/pipeline/training.md`](docs/pipeline/training.md) | head registry, losses, collapse safeties, callbacks, **full YAML config reference**, run-directory contents |
| [`docs/pipeline/evaluation.md`](docs/pipeline/evaluation.md) | evaluate script, physics sanity checks, trunk comparison, caveats |
| [`docs/pipeline/testing.md`](docs/pipeline/testing.md) | test suite map and workflow rules |
| [`docs/models/*.md`](docs/models/) | one per trunk: architecture, rationale, provenance, config knobs |
| [`structure.md`](structure.md) | HDF5 file layout (file not present in the repo; `GENERATOR_OVERHAUL.md` section 4 drafts its replacement) |
| [`q_head_action_plan.md`](q_head_action_plan.md) | q-head diagnosis and fix plan (q shelved from the default heads, 2026-07-14/15) |
| [`q_head_run_comparison.md`](q_head_run_comparison.md) | run-by-run q metrics for Phase 2 attempts 1 and 2 |
| [`phase2_5_3_revision_plan.md`](phase2_5_3_revision_plan.md) | pre-run concerns and resolutions for Phase 2.5/3 |
| [`phase2_5_3_run_results.md`](phase2_5_3_run_results.md) | Phase 2.5/3 run results and next steps |

## Quick start (lab GPU machine)

```bash
uv sync                                        # deps from pyproject.toml
pytest -m "not slow"                           # quick test suite
python scripts/train.py configs/smoke.yaml     # pre-flight end-to-end check
python scripts/run_all.py configs/resnet1d.yaml  # train + plot_run + evaluate
```

`scripts/run_full.py` does the same for every non-smoke config in `configs/`.

`scripts/run_all.py` chains `train.py` → `plot_run.py` → `evaluate.py` for one
config so none of the three gets forgotten; each also runs standalone if you
only need one step (e.g. re-plotting or re-evaluating an existing run).

Outputs land in `runs/<name>/` — checkpoints, per-epoch history, live
scatter PNGs, diagnostics CSV, TensorBoard logs, history/diagnostics summary
plots (`plot_run.py`), and evaluation metrics/plots (`evaluate.py`).

> Local development note: code is written/checked on a CPU-only machine;
> tests and training run on the lab GPU machine only.
