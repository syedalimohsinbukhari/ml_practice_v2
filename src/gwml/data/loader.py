"""HDF5 loading and tf.data pipeline construction.

Both splits fit comfortably in RAM (~800 MB for training strain), so arrays are
loaded eagerly and datasets built with from_tensor_slices. Strain is fed to the
model raw — normalization is a BatchNormalization layer inside every trunk.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf

from gwml.data.transforms import PARAM_COLUMNS, TargetTransforms


def load_arrays(
    path: str | Path,
    split: str,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (strain, params): strain (N, 4096, 2) float32, params (N, 10) float64.

    Detector order in the channel axis is [h1, l1]. ``max_samples`` truncates the
    split (used by the smoke config and the overfit tests).
    """
    with h5py.File(path, "r") as f:
        grp = f[split]
        sl = slice(None, max_samples)
        h1 = grp["h1"][sl]
        l1 = grp["l1"][sl]
        params = grp["params"][sl]
    strain = np.stack([h1, l1], axis=-1).astype(np.float32)
    return strain, params


def snr_sample_weights(params: np.ndarray, alpha: float) -> np.ndarray:
    """Optional per-sample loss weights (SNR/10)^alpha; see PLAN.md."""
    snr = params[:, PARAM_COLUMNS["snr"]]
    return ((snr / 10.0) ** alpha).astype(np.float32)


def make_dataset(
    strain: np.ndarray,
    params: np.ndarray,
    transforms: TargetTransforms,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 0,
    snr_weight_alpha: float | None = None,
) -> tf.data.Dataset:
    """Build a batched (x, y_dict[, sample_weight]) dataset for fit/evaluate."""
    targets = transforms.transform(params)
    if snr_weight_alpha is not None:
        tensors = (strain, targets, snr_sample_weights(params, snr_weight_alpha))
    else:
        tensors = (strain, targets)
    ds = tf.data.Dataset.from_tensor_slices(tensors)
    if shuffle:
        ds = ds.shuffle(len(strain), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
