# Seeding scheme for training and tests (ml_practice_v2)

Written 2026-10-19T20:55Z as a combined scheme, revised 2026-10-19T21:52Z, and split on 2026-10-19T22:06Z when each repository got its own master seed.
Status: design agreed with the user; no code implements it yet (OBS-023, OBS-024 in `BUGS.md`).
This file covers the randomness of this repository: training runs and tests.
Data generation lives in `ml-gw-search`, which has its own `SEEDING.md` and its own, unrelated master seed (user's decision, 2026-10-19T22:06Z); neither repository reads the other's.
Adapted from a `SEEDING.md` that came from another repository (a GRB paper codebase); this file is the project's own record and replaced that copy.
Rules come from `CLAUDE.md` §14 (determinism) and §6 of `WORKING_CONVENTIONS.md` (a generated file carries its provenance).

## Why

Seed literals are scattered in this repository:

- `train.seed: 42` in the six `configs/*.yaml`, and `42` as the default in `src/gwml/training/train.py:147`.
- `seed: int = 0` as the default of `make_dataset` in `src/gwml/data/loader.py:87`.
- `0` to `3` in the tests (11 places).

`train.py` uses the one number twice, for weight initialisation and dropout (`keras.utils.set_random_seed`) and for the shuffle order of `make_dataset`, and a run does not record the seed it used.

## `MASTER_SEED`

```
MASTER_SEED = 1153279169
```

Provenance, reproduced verbatim:

```
$ python3 -c "import secrets, datetime; print(secrets.randbits(32)); print(datetime.datetime.now(datetime.timezone.utc).isoformat())"
1153279169
2026-10-19T22:04:48.392291+00:00
```

Drawn on host `thinkpad-t530` from the OS CSPRNG, not from NumPy, so it depends on no generator state.
It is this repository's own value: never reuse the master seed of another repository, including the generator's.
The constant is to be defined once, in a `seeding` module of `src/gwml`, when the scheme is implemented; this file is the record of its value and origin.

## Derivation: one key per domain and purpose

Every seed is a `numpy.random.SeedSequence` keyed by `MASTER_SEED` and an explicit integer tuple.
The tuple is the key, so a seed never depends on call order, process order, machine, or checkout path.
The first element, the domain, keeps training and tests from sharing a key space.

```python
def seed_seq(*key: int) -> np.random.SeedSequence:
    return np.random.SeedSequence(MASTER_SEED, spawn_key=key)
```

| `domain` | Use | Key after the domain |
|----------|-----|----------------------|
| 0 | training | `(seed_index, purpose)` |
| 1 | tests | `(name_key, purpose)` |

Checked on 2026-10-19T22:05Z (numpy 2.2.6, with the master above):

- The 28 keys tried (20 training keys for `seed_index` 0 to 9, and 8 test keys) give 28 distinct states, so the domains do not collide.
- A rebuilt key reproduces its state, and `seed_index` 3 and 4 differ.
- Every one of those keys differs from the same key under the generator's master, so the two repositories' streams are unrelated.

## Training seeds (domain 0)

The `train.seed: 42` literal in the six configs is replaced by `seed_index`, the replicate number (decided 2026-10-19T21:56Z), and the seeds come from the master.
Each use gets its own stream:

| `purpose` | Use |
|-----------|-----|
| 0 | weight initialisation and dropout: `keras.utils.set_random_seed(derived)` |
| 1 | data order: the `seed` argument of `make_dataset`, which passes it to `tf.data` `shuffle` |

```python
init_seed    = int(seed_seq(0, seed_index, 0).generate_state(1)[0])
shuffle_seed = int(seed_seq(0, seed_index, 1).generate_state(1)[0])
```

- `seed_index` runs 0 to 9: M = 10 seeds per trunk (OBS-008, decided 2026-10-19T22:06Z).
  For `seed_index` 0 the derived values are `init_seed = 4009658500` and `shuffle_seed = 1158819920`.
- The key has no dataset id and no trunk name: the same `seed_index` on another dataset or another trunk reuses the same stream, which is wanted for paired comparisons (an assumption to confirm).
- `make_dataset(seed=...)` accepts the derived value: it is a Python `int` below 2³², which `tf.data` `shuffle` takes as its seed.
  Checked (2026-10-19T21:50Z, tensorflow in this repository's `.venv`): the same derived seed reproduces the order of both epochs of a 12-element shuffle with `reshuffle_each_iteration=True`, the two epochs differ from each other, and another derived seed gives a different order.
- The `seed: int = 0` default of `make_dataset` is another scattered literal and should be removed (the validation dataset does not shuffle, so it needs no seed).
- A run should write its `seed_index`, the derived seeds, and the master seed next to `transforms.json`, because nothing records the seed today.

## Test seeds (domain 1)

The tests are this project's own files, so they follow the same scheme; the current literals are `0`, `1`, `2`, and `3` in `conftest.py`, `test_registry.py`, `test_losses.py`, `test_heads_spec.py`, and `test_overfit.py` (11 places, verified by grep).
Each test draws from a key built from its own name:

```python
def name_key(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")

rng = np.random.default_rng(seed_seq(1, name_key(request.node.nodeid), 0))              # data a test generates
init_seed = int(seed_seq(1, name_key(request.node.nodeid), 1).generate_state(1)[0])     # keras seed in a test
```

- The pytest node id is relative to the repository root, so it is stable across machines and checkout paths; a fixture in `conftest.py` would hand each test its `rng`, and helpers such as `_fake_batch` would take an `rng` instead of a `seed=0` default.
- Module-level data in `conftest.py` is keyed by a fixed descriptive name instead of a node id.
- Checked (2026-10-19T21:50Z and 22:05Z): the 64-bit name key is accepted by `SeedSequence`, the same name reproduces its state, and different names differ.
- Changing a test's seed changes its data, and some tests (the overfit-one-batch tests, tests with numeric tolerances) may be sensitive to it.
  If a test fails after reseeding, that is a decision point for the user: never try seeds until a test passes.
  The suite must be run on the lab machine after the change (`VERIFY.md`).

## Determinism check (not yet run)

On the lab machine, with the user's go-ahead, after the scheme is implemented:

1. Two runs of the smoke configuration with the same `seed_index` give identical first-epoch batch orders and identical initial weights; a different `seed_index` gives different ones.
2. The suite passes after the reseeding, and each test's draws reproduce from its node id.

Falsified by: any difference in step 1 under the same `seed_index`, equal batch orders or weights under a different `seed_index`, or a test failure in step 2.
Compare arrays, not file bytes.
Differences only at float-rounding level (for example from GPU non-determinism, since `deterministic_ops` is `false` in the configs) are a decision point for the user, not something to resolve quietly.

## Open items

- Whether a trunk belongs in the training key (proposed: no, for paired comparisons).
- This file's `MASTER_SEED` has no code constant yet.
- The head-inclusion rule that uses these seeds (M = 10, N = 8, K = all five trunks, per-head floors) is recorded in `BUGS.md` OBS-008.
