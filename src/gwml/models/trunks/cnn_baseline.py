"""Small plain CNN. Exists to prove the pipeline — see docs/models/cnn_baseline.md."""

from __future__ import annotations

import keras
from keras import layers

from gwml.models.registry import register


@register("cnn_baseline")
def build(cfg: dict):
    window_len = cfg.get("window_len", 4096)
    filters = cfg.get("filters", [32, 64, 128, 128])
    kernel = cfg.get("kernel_size", 16)
    pool = cfg.get("pool_size", 4)

    inputs = keras.Input(shape=(window_len, 2), name="strain")
    x = layers.BatchNormalization(name="input_bn")(inputs)
    for i, f in enumerate(filters):
        x = layers.Conv1D(f, kernel, padding="same", name=f"conv_{i}")(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Activation("relu", name=f"relu_{i}")(x)
        x = layers.MaxPooling1D(pool, name=f"pool_{i}")(x)
    features = layers.GlobalAveragePooling1D(name="gap")(x)
    return inputs, features
