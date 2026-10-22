"""Output contract of the regenerated dataset file, as a check.

The layout comes from GENERATOR_OVERHAUL.md section 4 and the decisions D-01 to D-12; the column names come from
``heads_spec.PARAM_COLUMNS``. ``check_file`` reads a merged file or a repackaged shard written by
``ml-gw-search`` and returns one ``Result`` per check, so a file that lacks anything fails by name.
It imports only ``gwml.heads_spec``, so it runs without TensorFlow.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from gwml.heads_spec import PARAM_COLUMNS

COLUMNS = tuple(sorted(PARAM_COLUMNS, key=PARAM_COLUMNS.get))

SCHEMA_VERSION = 1
SAMPLE_LENGTH = 4096
SAMPLE_RATE = 2048
DETECTORS = ["H1", "L1"]
APPROXIMANT = "IMRPhenomD"
SNR_RANGE = [7.0, 15.0]
FFT_MODULE = "pycbc.fft.npfft"
BACKEND_KEYS = {"gen.py", "apply.whiten", "apply.regularize_psd"}
CONVENTION = "strain = noise + snr * unit_waveform"
SPLITS = ("training", "validation", "test")

INJECTION_DATASETS = {
    "h1": ("float32", SAMPLE_LENGTH),
    "l1": ("float32", SAMPLE_LENGTH),
    "h1_unit_waveform": ("float32", SAMPLE_LENGTH),
    "l1_unit_waveform": ("float32", SAMPLE_LENGTH),
    "params": ("float64", len(COLUMNS)),
}
NOISE_DATASETS = {"noise_h1": ("float32", SAMPLE_LENGTH), "noise_l1": ("float32", SAMPLE_LENGTH)}

# D-03: (injections, noise-only) per split, and the number of 10,000-sample shards.
FULL_COUNTS = {"training": (50_000, 50_000), "validation": (10_000, 10_000), "test": (5_000, 5_000)}
FULL_SHARDS = {"training": 10, "validation": 2, "test": 1}

# Bounds of the frozen generator's priors (GENERATOR_OVERHAUL.md G-13 to G-15); injection_time is float32 (128 s).
_TWO_PI = 2.0 * math.pi
PARAM_BOUNDS = {
    "mchirp": (8.7, 43.5),
    "q": (0.0, 1.0),
    "inclination": (0.0, _TWO_PI),
    "coa_phase": (0.0, _TWO_PI),
    "polarization_angle": (0.0, _TWO_PI),
    "declination": (-math.pi / 2, math.pi / 2),
    "ra": (0.0, _TWO_PI),
    "injection_time": (1238166018 - 128, 1253977218 + 128),
    "merger_time": (1.6, 1.8),
    "snr": tuple(SNR_RANGE),
}

REQUIRED_ATTRS = (
    "kind", "complete", "schema_version", "sample_length", "sample_rate", "detectors", "approximant", "noise_source",
    "low_frequency_cutoff", "segment_duration", "max_filter_length", "whitened", "fft_backend", "master_seed",
    "dataset_id", "columns", "params_columns", "backend_hashes", "snr_range", "convention", "python", "numpy", "pycbc",
    "h5py", "torch",
)
REQUIRED_ATTRS_BY_KIND = {
    "merged": REQUIRED_ATTRS + ("splits", "shards", "merge_order", "merge_git_sha", "merge_utc", "merge_host"),
    "repackaged_shard": REQUIRED_ATTRS + (
        "split", "shard", "n_injections", "n_noise_only", "seed_keys", "snr_seed_key", "source_sha256", "git_sha",
        "generated_utc", "host", "repackage_git_sha", "repackage_utc", "repackage_host"),
}


@dataclass(frozen=True)
class Result:
    name: str
    ok: bool
    detail: str = ""


def _plain(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_attr(attrs, name):
    return json.loads(_plain(attrs[name]))


class _Checker:
    def __init__(self):
        self.results: list[Result] = []

    def add(self, name, ok, detail=""):
        self.results.append(Result(name, bool(ok), "" if ok else str(detail)))

    def run(self, name, fn):
        """Run a check that returns (ok, detail); an exception is a failure, not a crash."""
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001 - a malformed file must fail by name
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        self.add(name, ok, detail)


def _check_attrs(c: _Checker, attrs) -> str | None:
    kind = _plain(attrs["kind"]) if "kind" in attrs else None
    c.add("attrs: kind is 'merged' or 'repackaged_shard'", kind in REQUIRED_ATTRS_BY_KIND, f"kind={kind!r}")
    required = REQUIRED_ATTRS_BY_KIND.get(kind, REQUIRED_ATTRS)
    missing = [a for a in required if a not in attrs]
    c.add("attrs: all required attributes present", not missing, f"missing {missing}")
    c.run("attrs: complete is true", lambda: (bool(_plain(attrs["complete"])) is True, f"complete={attrs.get('complete')}"))

    def eq(name, expected, key=None):
        c.run(f"attrs: {name}", lambda: (
            _plain(attrs[key or name]) == expected, f"{_plain(attrs[key or name])!r} != {expected!r}"))

    eq("schema_version", SCHEMA_VERSION)
    eq("sample_length", SAMPLE_LENGTH)
    eq("sample_rate", SAMPLE_RATE)
    eq("approximant", APPROXIMANT)
    eq("whitened", True)
    eq("fft_backend", FFT_MODULE)
    eq("low_frequency_cutoff", 20.0)
    c.run("attrs: detectors are H1, L1 in that order", lambda: (
        _json_attr(attrs, "detectors") == DETECTORS, f"{_plain(attrs['detectors'])!r}"))
    c.run("attrs: noise source is Gaussian", lambda: (
        _plain(attrs["noise_source"]).startswith("gaussian"), f"{_plain(attrs['noise_source'])!r}"))
    c.run("attrs: snr_range is (7, 15)", lambda: (
        [float(v) for v in _plain(attrs["snr_range"])] == SNR_RANGE, f"{_plain(attrs['snr_range'])!r}"))
    c.run("attrs: convention names strain = noise + snr * unit_waveform", lambda: (
        CONVENTION in _plain(attrs["convention"]), f"{_plain(attrs['convention'])!r}"))
    c.run("attrs: params_columns equals heads_spec.PARAM_COLUMNS in index order", lambda: (
        _json_attr(attrs, "params_columns") == list(COLUMNS), f"{_plain(attrs['params_columns'])!r}"))
    c.run("attrs: columns equals the first nine heads_spec names", lambda: (
        _json_attr(attrs, "columns") == list(COLUMNS[:-1]), f"{_plain(attrs['columns'])!r}"))

    def hashes():
        h = _json_attr(attrs, "backend_hashes")
        good = set(h) == BACKEND_KEYS and all(re.fullmatch(r"[0-9a-f]{64}", v) for v in h.values())
        return good, f"{h!r}"

    c.run("attrs: backend_hashes has the three sha256 pins", hashes)
    return kind if kind in REQUIRED_ATTRS_BY_KIND else None


def _check_group(c: _Checker, f: h5py.File, split: str, n_inj: int, n_noise: int, max_rows: int | None):
    g = f[split]
    expected = set(INJECTION_DATASETS) | set(NOISE_DATASETS)
    c.add(f"{split}: datasets are exactly the contract's seven", set(g.keys()) == expected,
          f"extra {sorted(set(g.keys()) - expected)}, missing {sorted(expected - set(g.keys()))}")
    present = expected & set(g.keys())
    for name, (dtype, width) in {**INJECTION_DATASETS, **NOISE_DATASETS}.items():
        if name not in present:
            continue
        rows = n_inj if name in INJECTION_DATASETS else n_noise
        ds = g[name]
        c.add(f"{split}/{name}: shape ({rows}, {width}) and dtype {dtype}",
              ds.shape == (rows, width) and ds.dtype == np.dtype(dtype), f"{ds.shape} {ds.dtype}")
    if "params" in present:
        c.run(f"{split}/params: columns attribute equals heads_spec names", lambda: (
            [_plain(x) for x in g["params"].attrs["columns"]] == list(COLUMNS),
            f"{[_plain(x) for x in g['params'].attrs['columns']]!r}"))
        params = g["params"][: max_rows]
        if params.shape[1:] == (len(COLUMNS),) and len(params):
            bad = []
            for name, (lo, hi) in PARAM_BOUNDS.items():
                col = params[:, PARAM_COLUMNS[name]]
                if not np.all(np.isfinite(col)) or col.min() < lo or col.max() > hi:
                    bad.append(f"{name} [{col.min():.6g}, {col.max():.6g}] not in [{lo:.6g}, {hi:.6g}]")
            c.add(f"{split}/params: every column finite and within its prior bounds", not bad, "; ".join(bad))
            c.add(f"{split}/params: q is above zero", bool(np.all(params[:, PARAM_COLUMNS["q"]] > 0)), "q <= 0 present")
    for name in sorted(present):
        ds = g[name]
        if not len(ds) or ds.ndim != 2:
            continue
        n = len(ds) if max_rows is None else min(len(ds), max_rows)
        finite = all(np.all(np.isfinite(ds[i:min(i + 1000, n)])) for i in range(0, n, 1000))
        c.add(f"{split}/{name}: no NaN or Inf in the {n} rows read", finite, "non-finite values present")
    # The remaining content checks read a few rows; a malformed shape must fail them, not crash the check.
    if {"h1", "l1"} <= present and n_inj:
        n = min(n_inj, 200)
        c.run(f"{split}: H1 and L1 differ in every one of the first {n} injection rows", lambda: (
            not np.any(np.all(g["h1"][:n] == g["l1"][:n], axis=1)), "a row has identical H1 and L1"))
    if {"h1_unit_waveform", "l1_unit_waveform"} <= present and n_inj:
        n = min(n_inj, 200)
        c.run(f"{split}: unit waveforms are not all zero in the first {n} rows", lambda: (
            bool(np.all(np.abs(g["h1_unit_waveform"][:n]).sum(axis=1)
                        + np.abs(g["l1_unit_waveform"][:n]).sum(axis=1) > 0)), "zero waveform"))
    if {"noise_h1", "noise_l1"} <= present and n_noise > 1:
        c.run(f"{split}: noise-only rows are not constant", lambda: (
            bool(np.all(g["noise_h1"][:200].std(axis=1) > 0)), "constant noise row"))


def check_file(path: str | Path, full: bool = False, max_rows: int | None = None) -> list[Result]:
    """Check a merged file or a repackaged shard against the contract.

    ``full`` also requires the D-03 counts (50,000 + 50,000, 10,000 + 10,000, 5,000 + 5,000) and, for a merged file,
    its 10 + 2 + 1 shards. ``max_rows`` limits the rows read per dataset for the content checks.
    """
    c = _Checker()
    try:
        f = h5py.File(path, "r")
    except OSError as exc:
        return [Result("file: opens as HDF5", False, str(exc))]
    with f:
        attrs = f.attrs
        kind = _check_attrs(c, attrs)
        groups = sorted(f.keys())
        c.add("file: top-level groups are splits", bool(groups) and set(groups) <= set(SPLITS), f"{groups}")
        if kind == "merged":
            c.run("merged: splits attribute lists exactly the groups", lambda: (
                sorted(_json_attr(attrs, "splits")) == groups, f"{_plain(attrs['splits'])!r} vs {groups}"))
            records = _json_attr(attrs, "shards") if "shards" in attrs else []
            c.add("merged: shards attribute is a non-empty list with a sha256 per shard",
                  bool(records) and all(re.fullmatch(r"[0-9a-f]{64}", r.get("file_sha256", "")) for r in records),
                  f"{len(records)} records")
        for split in groups:
            if split not in SPLITS:
                continue
            g = f[split]
            if kind == "repackaged_shard":
                n_inj, n_noise = int(_plain(attrs.get("n_injections", -1))), int(_plain(attrs.get("n_noise_only", -1)))
            else:
                n_inj, n_noise = int(_plain(g.attrs.get("n_injections", -1))), int(_plain(g.attrs.get("n_noise_only", -1)))
            c.add(f"{split}: row counts are recorded as attributes", n_inj >= 0 and n_noise >= 0,
                  f"n_injections={n_inj}, n_noise_only={n_noise}")
            if kind == "merged":
                recs = [r for r in records if r.get("split") == split]
                c.add(f"{split}: shard records sum to the stored row counts",
                      sum(r.get("n_injections", 0) for r in recs) == n_inj
                      and sum(r.get("n_noise_only", 0) for r in recs) == n_noise, f"{len(recs)} records")
                c.add(f"{split}: shard numbers are 0 to {len(recs) - 1} in order",
                      [r.get("shard") for r in recs] == list(range(len(recs))), f"{[r.get('shard') for r in recs]}")
            if full:
                c.add(f"{split}: D-03 counts {FULL_COUNTS.get(split)}", (n_inj, n_noise) == FULL_COUNTS.get(split),
                      f"({n_inj}, {n_noise})")
                if kind == "merged":
                    c.add(f"{split}: {FULL_SHARDS[split]} shards", len(recs) == FULL_SHARDS[split], f"{len(recs)}")
            _check_group(c, f, split, n_inj, n_noise, max_rows)
        if full:
            c.add("file: has training, validation, and test", groups == sorted(SPLITS), f"{groups}")
    return c.results
