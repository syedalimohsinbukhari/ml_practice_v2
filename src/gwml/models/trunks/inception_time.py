"""InceptionTime-style multi-scale trunk — see docs/models/inception_time.md."""

from __future__ import annotations

import keras
from keras import layers

from gwml.models.registry import register


def _inception_module(x, filters, kernel_sizes, bottleneck, name):
    if bottleneck and x.shape[-1] > 1:
        inp = layers.Conv1D(
            bottleneck, 1, padding="same", use_bias=False, name=f"{name}_bottleneck"
        )(x)
    else:
        inp = x
    branches = [
        layers.Conv1D(
            filters, k, padding="same", use_bias=False, name=f"{name}_conv_k{k}"
        )(inp)
        for k in kernel_sizes
    ]
    pooled = layers.MaxPooling1D(3, strides=1, padding="same", name=f"{name}_pool")(x)
    branches.append(
        layers.Conv1D(
            filters, 1, padding="same", use_bias=False, name=f"{name}_pool_conv"
        )(pooled)
    )
    y = layers.Concatenate(name=f"{name}_concat")(branches)
    y = layers.BatchNormalization(name=f"{name}_bn")(y)
    return layers.Activation("relu", name=f"{name}_relu")(y)


@register("inception_time")
def build(cfg: dict):
    window_len = cfg.get("window_len", 4096)
    depth = cfg.get("depth", 6)
    filters = cfg.get("filters", 32)
    kernel_sizes = cfg.get("kernel_sizes", [9, 19, 39])
    bottleneck = cfg.get("bottleneck", 32)
    # Stride-4 stem keeps the multi-scale blocks affordable on a 4096 window.
    stem_stride = cfg.get("stem_stride", 4)

    inputs = keras.Input(shape=(window_len, 2), name="strain")
    x = layers.BatchNormalization(name="input_bn")(inputs)
    if stem_stride > 1:
        x = layers.Conv1D(
            filters, 7, strides=stem_stride, padding="same", name="stem"
        )(x)

    residual = x
    for d in range(depth):
        x = _inception_module(
            x, filters, kernel_sizes, bottleneck, name=f"inception_{d}"
        )
        if d % 3 == 2:  # shortcut every 3 modules, as in the InceptionTime paper
            residual = layers.Conv1D(
                x.shape[-1], 1, padding="same", name=f"shortcut_{d}"
            )(residual)
            residual = layers.BatchNormalization(name=f"shortcut_bn_{d}")(residual)
            x = layers.Add(name=f"residual_add_{d}")([x, residual])
            x = layers.Activation("relu", name=f"residual_relu_{d}")(x)
            residual = x

    features = layers.GlobalAveragePooling1D(name="gap")(x)
    return inputs, features
