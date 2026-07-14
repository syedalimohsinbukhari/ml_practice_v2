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


def build_subset_masks(params: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean masks for diagnostic subsets and targeted oversampling.

    Same logic used by ``DiagnosticSubsetsCallback`` — extracted here so
    ``run_experiment`` can reuse it for data augmentation without depending
    on the callback module.
    """
    snr = params[:, PARAM_COLUMNS["snr"]]
    mchirp = params[:, PARAM_COLUMNS["mchirp"]]
    mt = params[:, PARAM_COLUMNS["merger_time"]]
    q = params[:, PARAM_COLUMNS["q"]]
    s1, s2 = np.quantile(snr, [1 / 3, 2 / 3])
    q1, q2 = np.quantile(q, [1 / 3, 2 / 3])
    mchirp_low = mchirp < np.median(mchirp)
    mchirp_high = ~mchirp_low
    q_low = q < q1
    q_high = q >= q2
    return {
        "full": np.ones(len(params), dtype=bool),
        "snr_low": snr < s1,
        "snr_mid": (snr >= s1) & (snr < s2),
        "snr_high": snr >= s2,
        "mchirp_low": mchirp_low,
        "mchirp_high": mchirp_high,
        "merger_early": mt < np.median(mt),
        "merger_late": mt >= np.median(mt),
        "q_low": q_low,
        "q_mid": (q >= q1) & (q < q2),
        "q_high": q_high,
        "q_low_mchirp_low": q_low & mchirp_low,
        "q_low_mchirp_high": q_low & mchirp_high,
        "q_high_mchirp_low": q_high & mchirp_low,
        "q_high_mchirp_high": q_high & mchirp_high,
    }


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
