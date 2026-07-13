# GW multi-head parameter estimation

Deep-learning regression of compact-binary parameters (chirp mass, mass
ratio, merger time, SNR — optionally sky/orientation angles) from
dual-detector whitened strain in `combined_repackaged.hdf`, using a shared
trunk with configurable output heads and a pluggable model zoo.

## Documentation map

| doc | what's in it |
|-----|--------------|
| [`PLAN.md`](PLAN.md) | the design document: objectives, decisions and their rationale, milestones |
| [`docs/pipeline/overview.md`](docs/pipeline/overview.md) | end-to-end data flow, module map, design invariants |
| [`docs/pipeline/data.md`](docs/pipeline/data.md) | HDF5 loading, target transforms, dataset construction |
| [`docs/pipeline/training.md`](docs/pipeline/training.md) | head registry, losses, collapse safeties, callbacks, **full YAML config reference**, run-directory contents |
| [`docs/pipeline/evaluation.md`](docs/pipeline/evaluation.md) | evaluate script, physics sanity checks, trunk comparison, caveats |
| [`docs/pipeline/testing.md`](docs/pipeline/testing.md) | test suite map and workflow rules |
| [`docs/models/*.md`](docs/models/) | one per trunk: architecture, rationale, provenance, config knobs |
| [`structure.md`](structure.md) | HDF5 file layout |

## Quick start (lab GPU machine)

```bash
uv sync                                        # deps from pyproject.toml
pytest -m "not slow"                           # quick test suite
python scripts/train.py configs/smoke.yaml     # pre-flight end-to-end check
python scripts/train.py configs/resnet1d.yaml  # real run
python scripts/evaluate.py configs/resnet1d.yaml --split validation
```

Outputs land in `runs/<name>/` — checkpoints, per-epoch history, live
scatter PNGs, diagnostics CSV, TensorBoard logs.

> Local development note: code is written/checked on a CPU-only machine;
> tests and training run on the lab GPU machine only.
