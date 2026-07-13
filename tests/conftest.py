"""Shared fixtures. Run this suite on the lab machine, not the local T530.

Quick suite:  pytest -m "not slow"
Full suite:   pytest            (includes overfit-one-batch tests)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

DATA_PATH = REPO_ROOT / "combined_repackaged.hdf"

requires_data = pytest.mark.skipif(
    not DATA_PATH.exists(), reason="combined_repackaged.hdf not present"
)

# Tiny trunk configs so build/overfit tests stay fast.
TINY_TRUNK_CFGS = {
    "cnn_baseline": {"filters": [8, 16]},
    "resnet1d": {
        "stem_filters": 8,
        "stage_filters": [8, 16],
        "stage_dilations": [1, 2],
        "blocks_per_stage": 1,
    },
    "inception_time": {"depth": 3, "filters": 8, "bottleneck": 8, "stem_stride": 8},
    # Dilations must keep the miniature's receptive field ~the full window
    # (2*(k-1)*sum(d) stride-4 steps); [1..128] covers ~4080 of 4096 samples.
    "tcn": {"filters": 8, "dilations": [1, 2, 4, 8, 16, 32, 64, 128],
            "stem_stride": 4},
    "cnn_attention": {
        "conv_filters": [8, 16, 16, 16, 16],
        "model_dim": 16,
        "num_blocks": 1,
        "num_heads": 2,
        "ff_dim": 32,
    },
}


@pytest.fixture(scope="session")
def synthetic_params():
    """(N, 10) params drawn uniformly within the documented ranges."""
    rng = np.random.default_rng(0)
    n = 200
    cols = [
        rng.uniform(8.85, 43.4, n),        # mchirp
        rng.uniform(0.21, 1.0, n),         # q
        rng.uniform(0, 2 * np.pi, n),      # inclination
        rng.uniform(0, 2 * np.pi, n),      # coa_phase
        rng.uniform(0, 2 * np.pi, n),      # polarization_angle
        rng.uniform(-1.5, 1.5, n),         # declination
        rng.uniform(0, 2 * np.pi, n),      # ra
        rng.uniform(1.238e9, 1.254e9, n),  # injection_time
        rng.uniform(1.6, 1.8, n),          # merger_time_in_windows
        rng.uniform(7.0, 15.0, n),         # snr
    ]
    return np.stack(cols, axis=1)


@pytest.fixture(scope="session")
def fitted_transforms(synthetic_params):
    from gwml.data.transforms import TargetTransforms

    return TargetTransforms().fit(synthetic_params)
