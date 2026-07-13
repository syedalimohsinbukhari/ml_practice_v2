"""1D ResNet, the expected workhorse — see docs/models/resnet1d.md."""

from __future__ import annotations

import keras
from keras import layers

from gwml.models.registry import register


def _residual_block(x, filters, kernel, stride, dilation, name):
    shortcut = x
    y = layers.Conv1D(
        filters, kernel, strides=stride, padding="same", name=f"{name}_conv1"
    )(x)
    y = layers.BatchNormalization(name=f"{name}_bn1")(y)
    y = layers.Activation("relu", name=f"{name}_relu1")(y)
    y = layers.Conv1D(
        filters, kernel, padding="same", dilation_rate=dilation, name=f"{name}_conv2"
    )(y)
    y = layers.BatchNormalization(name=f"{name}_bn2")(y)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(
            filters, 1, strides=stride, padding="same", name=f"{name}_proj"
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)
    y = layers.Add(name=f"{name}_add")([y, shortcut])
    return layers.Activation("relu", name=f"{name}_out")(y)


@register("resnet1d")
def build(cfg: dict):
    window_len = cfg.get("window_len", 4096)
    stem_filters = cfg.get("stem_filters", 64)
    stem_kernel = cfg.get("stem_kernel", 15)
    stage_filters = cfg.get("stage_filters", [64, 128, 256])
    blocks_per_stage = cfg.get("blocks_per_stage", 2)
    kernel = cfg.get("kernel_size", 7)
    # Later stages keep stride-1 second blocks but dilate them to widen the
    # receptive field without further downsampling.
    stage_dilations = cfg.get("stage_dilations", [1, 2, 4])

    inputs = keras.Input(shape=(window_len, 2), name="strain")
    x = layers.BatchNormalization(name="input_bn")(inputs)
    x = layers.Conv1D(stem_filters, stem_kernel, strides=2, padding="same", name="stem")(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("relu", name="stem_relu")(x)
    x = layers.MaxPooling1D(2, name="stem_pool")(x)

    for s, (filters, dilation) in enumerate(zip(stage_filters, stage_dilations)):
        for b in range(blocks_per_stage):
            stride = 2 if b == 0 else 1
            x = _residual_block(
                x, filters, kernel, stride, dilation, name=f"stage{s}_block{b}"
            )

    gap = layers.GlobalAveragePooling1D(name="gap")(x)
    gmp = layers.GlobalMaxPooling1D(name="gmp")(x)
    features = layers.Concatenate(name="features")([gap, gmp])
    return inputs, features
