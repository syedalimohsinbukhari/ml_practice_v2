"""The output contract check (gwml.contract): a valid file passes, and each broken rule fails by name.

Builds tiny files in tmp_path, so it needs no generated dataset and no TensorFlow.
"""

import json

import h5py
import numpy as np
import pytest

from conftest import DATA_PATH, requires_data
from gwml import contract
from gwml.contract import COLUMNS, check_file
from gwml.heads_spec import PARAM_COLUMNS

N_INJ, N_NOISE = 6, 4
ATTRS = {
    "schema_version": 1, "sample_length": 4096, "sample_rate": 2048, "detectors": json.dumps(["H1", "L1"]),
    "approximant": "IMRPhenomD", "noise_source": "gaussian, aLIGOZeroDetHighPower", "low_frequency_cutoff": 20.0,
    "segment_duration": 0.5, "max_filter_length": 512, "whitened": True, "fft_backend": "pycbc.fft.npfft",
    "master_seed": 3021456217, "dataset_id": 0, "columns": json.dumps(list(COLUMNS[:-1])),
    "params_columns": json.dumps(list(COLUMNS)),
    "backend_hashes": json.dumps({k: "a" * 64 for k in ("gen.py", "apply.whiten", "apply.regularize_psd")}),
    "snr_range": np.array([7.0, 15.0]),
    "convention": "strain = noise + snr * unit_waveform, both whitened; snr is the network SNR",
    "python": "3.10.12", "numpy": "2.2.6", "pycbc": "2.9.0", "h5py": "3.16.0", "torch": "2.0.0",
}
SHARD_ATTRS = {
    "split": "training", "shard": 0, "n_injections": N_INJ, "n_noise_only": N_NOISE, "seed_keys": "{}",
    "snr_seed_key": "[]", "source_sha256": "b" * 64, "git_sha": "c" * 40, "generated_utc": "t", "host": "h",
    "repackage_git_sha": "c" * 40, "repackage_utc": "t", "repackage_host": "h",
}
MERGED_ATTRS = {"merge_order": "o", "merge_git_sha": "c" * 40, "merge_utc": "t", "merge_host": "h"}


def _params(rng):
    lo_hi = contract.PARAM_BOUNDS
    p = np.empty((N_INJ, len(COLUMNS)))
    for name, idx in PARAM_COLUMNS.items():
        lo, hi = lo_hi[name]
        p[:, idx] = rng.uniform(lo + 1e-3 * (hi - lo), hi - 1e-3 * (hi - lo), N_INJ)
    return p


def _write_groups(f, split):
    rng = np.random.default_rng(0)
    g = f.create_group(split)
    for name in ("h1", "l1", "h1_unit_waveform", "l1_unit_waveform"):
        g.create_dataset(name, data=rng.standard_normal((N_INJ, 4096)).astype(np.float32))
    for name in ("noise_h1", "noise_l1"):
        g.create_dataset(name, data=rng.standard_normal((N_NOISE, 4096)).astype(np.float32))
    g.create_dataset("params", data=_params(rng))
    g["params"].attrs["columns"] = list(COLUMNS)
    return g


def make_file(path, kind="repackaged_shard", split="training"):
    with h5py.File(path, "w") as f:
        for k, v in ATTRS.items():
            f.attrs[k] = v
        f.attrs["kind"] = kind
        g = _write_groups(f, split)
        if kind == "repackaged_shard":
            for k, v in SHARD_ATTRS.items():
                f.attrs[k] = v
            f.attrs["split"] = split
        else:
            for k, v in MERGED_ATTRS.items():
                f.attrs[k] = v
            f.attrs["splits"] = json.dumps([split])
            g.attrs["n_injections"], g.attrs["n_noise_only"] = N_INJ, N_NOISE
            f.attrs["shards"] = json.dumps([{"split": split, "shard": 0, "file_sha256": "d" * 64,
                                             "n_injections": N_INJ, "n_noise_only": N_NOISE}])
        f.attrs["complete"] = True
    return path


def failed(path, **kwargs):
    return [r.name for r in check_file(path, **kwargs) if not r.ok]


@pytest.mark.parametrize("kind", ["repackaged_shard", "merged"])
def test_valid_file_passes(tmp_path, kind):
    assert failed(make_file(tmp_path / "ok.hdf", kind)) == []


def _del_attr(name):
    return lambda f: f.attrs.__delitem__(name)


def _set_attr(name, value):
    return lambda f: f.attrs.__setitem__(name, value)


def _swap_columns(f):
    cols = list(COLUMNS)
    cols[0], cols[1] = cols[1], cols[0]
    f.attrs["params_columns"] = json.dumps(cols)


def _wrong_group_columns(f):
    f["training/params"].attrs["columns"] = list(COLUMNS[::-1])


def _recreate(f, name, data):
    del f["training"][name]
    f["training"].create_dataset(name, data=data)


def _float64_h1(f):
    _recreate(f, "h1", f["training/h1"][()].astype(np.float64))


def _short_h1(f):
    _recreate(f, "h1", f["training/h1"][:-1])


def _nan_noise(f):
    f["training/noise_h1"][0, 5] = np.nan


def _snr_high(f):
    f["training/params"][0, PARAM_COLUMNS["snr"]] = 20.0


def _q_above_one(f):
    f["training/params"][0, PARAM_COLUMNS["q"]] = 1.5


def _extra_dataset(f):
    f["training"].create_dataset("h1_waveform", data=np.zeros((N_INJ, 4096), np.float32))


def _drop_noise(f):
    del f["training"]["noise_l1"]


def _identical_detectors(f):
    _recreate(f, "l1", f["training/h1"][()])


def _zero_waveform(f):
    f["training/h1_unit_waveform"][0] = 0.0
    f["training/l1_unit_waveform"][0] = 0.0


MUTATIONS = [
    ("missing complete", _del_attr("complete"), "required attributes present"),
    ("complete false", _set_attr("complete", False), "complete is true"),
    ("fftw backend", _set_attr("fft_backend", "pycbc.fft.fftw"), "fft_backend"),
    ("wrong approximant", _set_attr("approximant", "IMRPhenomXPHM"), "approximant"),
    ("not whitened", _set_attr("whitened", False), "whitened"),
    ("wrong snr range", _set_attr("snr_range", np.array([5.0, 15.0])), "snr_range"),
    ("swapped column names", _swap_columns, "params_columns equals heads_spec"),
    ("reversed params columns attr", _wrong_group_columns, "columns attribute equals heads_spec"),
    ("h1 as float64", _float64_h1, "training/h1: shape"),
    ("h1 with a missing row", _short_h1, "training/h1: shape"),
    ("NaN in noise-only rows", _nan_noise, "no NaN or Inf"),
    ("snr above the range", _snr_high, "prior bounds"),
    ("q above one", _q_above_one, "prior bounds"),
    ("extra dataset", _extra_dataset, "exactly the contract's seven"),
    ("missing noise dataset", _drop_noise, "exactly the contract's seven"),
    ("identical H1 and L1", _identical_detectors, "H1 and L1 differ"),
    ("zero unit waveform", _zero_waveform, "not all zero"),
]


@pytest.mark.parametrize("label,mutate,expected", MUTATIONS, ids=[m[0] for m in MUTATIONS])
@pytest.mark.parametrize("kind", ["repackaged_shard", "merged"])
def test_each_broken_rule_fails_by_name(tmp_path, kind, label, mutate, expected):
    path = make_file(tmp_path / "bad.hdf", kind)
    with h5py.File(path, "a") as f:
        mutate(f)
    names = failed(path)
    assert any(expected in n for n in names), f"{label}: expected a failure containing {expected!r}, got {names}"


def test_merged_shard_records_must_sum_to_row_counts(tmp_path):
    path = make_file(tmp_path / "m.hdf", "merged")
    with h5py.File(path, "a") as f:
        rec = json.loads(f.attrs["shards"])
        rec[0]["n_injections"] += 1
        f.attrs["shards"] = json.dumps(rec)
    assert any("sum to the stored row counts" in n for n in failed(path))


def test_full_requires_the_d03_counts(tmp_path):
    path = make_file(tmp_path / "ok.hdf")
    assert failed(path) == []
    assert any("D-03 counts" in n for n in failed(path, full=True))


def test_max_rows_limits_content_reads(tmp_path):
    path = make_file(tmp_path / "nan_late.hdf")
    with h5py.File(path, "a") as f:
        f["training/noise_h1"][N_NOISE - 1, 0] = np.nan
    assert any("no NaN or Inf" in n for n in failed(path))
    assert not any("no NaN or Inf" in n for n in failed(path, max_rows=N_NOISE - 1))


def test_not_an_hdf5_file_fails_without_crashing(tmp_path):
    bad = tmp_path / "x.hdf"
    bad.write_bytes(b"not hdf5")
    assert failed(bad) == ["file: opens as HDF5"]


def test_columns_follow_heads_spec_index_order():
    assert [PARAM_COLUMNS[c] for c in COLUMNS] == list(range(len(COLUMNS)))
    assert COLUMNS[-1] == "snr"


@requires_data
def test_legacy_file_without_attributes_fails_the_contract():
    """Known-bad control: combined_repackaged.hdf has no attributes and none of the new datasets."""
    names = [r.name for r in check_file(DATA_PATH, max_rows=50) if not r.ok]
    assert any("all required attributes present" in n for n in names)
    assert any("exactly the contract's seven" in n for n in names)
