"""Temporal Convolutional Network trunk — see docs/models/tcn.md.

Non-causal variant: our 2 s windows are offline, and causal padding combined
with global pooling starves most pooled positions of merger information (the
merger sits at 80-90% of the window) — a structural push toward mean
collapse. 'same' padding gives every position symmetric context, making
GAP + GMP a sound readout.
"""

from __future__ import annotations

import keras
from keras import layers

from gwml.models.registry import register


def _tcn_block(x, filters, kernel, dilation, dropout, name):
    shortcut = x
    y = layers.Conv1D(
        filters, kernel, padding="same", dilation_rate=dilation, name=f"{name}_conv1"
    )(x)
    y = layers.BatchNormalization(name=f"{name}_bn1")(y)
    y = layers.Activation("relu", name=f"{name}_relu1")(y)
    y = layers.SpatialDropout1D(dropout, name=f"{name}_drop1")(y)
    y = layers.Conv1D(
        filters, kernel, padding="same", dilation_rate=dilation, name=f"{name}_conv2"
    )(y)
    y = layers.BatchNormalization(name=f"{name}_bn2")(y)
    y = layers.Activation("relu", name=f"{name}_relu2")(y)
    y = layers.SpatialDropout1D(dropout, name=f"{name}_drop2")(y)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding="same", name=f"{name}_proj")(shortcut)
    y = layers.Add(name=f"{name}_add")([y, shortcut])
    return layers.Activation("relu", name=f"{name}_out")(y)


@register("tcn")
def build(cfg: dict):
    window_len = cfg.get("window_len", 4096)
    filters = cfg.get("filters", 64)
    kernel = cfg.get("kernel_size", 3)
    # Dilations 1..512 give a receptive field of ~4092 samples — the full window.
    dilations = cfg.get("dilations", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    dropout = cfg.get("dropout", 0.1)
    stem_stride = cfg.get("stem_stride", 2)

    inputs = keras.Input(shape=(window_len, 2), name="strain")
    x = layers.BatchNormalization(name="input_bn")(inputs)
    if stem_stride > 1:
        x = layers.Conv1D(filters, 7, strides=stem_stride, padding="same", name="stem")(x)

    for i, d in enumerate(dilations):
        x = _tcn_block(x, filters, kernel, d, dropout, name=f"tcn_{i}")

    gap = layers.GlobalAveragePooling1D(name="gap")(x)
    gmp = layers.GlobalMaxPooling1D(name="gmp")(x)
    features = layers.Concatenate(name="features")([gap, gmp])
    return inputs, features
