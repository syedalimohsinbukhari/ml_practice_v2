"""HDF5 integrity: shapes per structure.md, no NaN/Inf, params in range.

Skipped automatically when combined_repackaged.hdf is absent.
"""

import h5py
import numpy as np
import pytest

from conftest import DATA_PATH, requires_data
from gwml.data.loader import load_arrays
from gwml.data.transforms import PARAM_COLUMNS

EXPECTED = {"training": 25000, "validation": 5000}
WINDOW_LEN = 4096
RANGES = {
    "mchirp": (5.0, 60.0),
    "q": (0.0, 1.0),
    "merger_time": (1.6, 1.8),
    "snr": (7.0, 15.0),
}

pytestmark = requires_data


@pytest.mark.parametrize("split", sorted(EXPECTED))
def test_shapes_match_structure_md(split):
    with h5py.File(DATA_PATH, "r") as f:
        n = EXPECTED[split]
        for key in ("h1", "l1", "h1_waveform", "l1_waveform"):
            assert f[split][key].shape == (n, WINDOW_LEN)
        assert f[split]["params"].shape == (n, 10)


@pytest.mark.parametrize("split", sorted(EXPECTED))
def test_no_nan_or_inf_in_sample(split):
    strain, params = load_arrays(DATA_PATH, split, max_samples=500)
    assert np.all(np.isfinite(strain))
    assert np.all(np.isfinite(params))


@pytest.mark.parametrize("split", sorted(EXPECTED))
def test_target_params_within_documented_ranges(split):
    with h5py.File(DATA_PATH, "r") as f:
        params = f[split]["params"][:]
    for head, (lo, hi) in RANGES.items():
        col = params[:, PARAM_COLUMNS[head]]
        assert col.min() >= lo, f"{split}/{head} below {lo}"
        assert col.max() <= hi, f"{split}/{head} above {hi}"


def test_loader_stacks_detectors_in_order():
    strain, _ = load_arrays(DATA_PATH, "training", max_samples=10)
    assert strain.shape == (10, WINDOW_LEN, 2)
    assert strain.dtype == np.float32
    with h5py.File(DATA_PATH, "r") as f:
        np.testing.assert_array_equal(strain[..., 0], f["training/h1"][:10])
        np.testing.assert_array_equal(strain[..., 1], f["training/l1"][:10])
