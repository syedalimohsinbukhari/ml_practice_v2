"""Conv front-end + transformer encoder + attention pooling.

See docs/models/cnn_attention.md. Heaviest trunk; GPU-targeted.
"""

from __future__ import annotations

import keras
from keras import layers

from gwml.models.registry import register


class PositionalEmbedding(layers.Layer):
    """Learned additive positional embedding over a fixed sequence length."""

    def build(self, input_shape):
        self.pos = self.add_weight(
            name="pos",
            shape=(input_shape[1], input_shape[2]),
            initializer="random_normal",
        )

    def call(self, x):
        return x + self.pos


class AttentionPooling(layers.Layer):
    """Learned-score softmax pooling over the time axis: (B, T, D) -> (B, D)."""

    def __init__(self, hidden=64, **kwargs):
        super().__init__(**kwargs)
        self.hidden = hidden
        self.score_hidden = layers.Dense(hidden, activation="tanh")
        self.score = layers.Dense(1)
        self.softmax = layers.Softmax(axis=1)

    def call(self, x):
        weights = self.softmax(self.score(self.score_hidden(x)))
        return keras.ops.sum(x * weights, axis=1)

    def get_config(self):
        return {**super().get_config(), "hidden": self.hidden}


def _encoder_block(x, dim, num_heads, ff_dim, dropout, name):
    attn = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=dim // num_heads, dropout=dropout,
        name=f"{name}_mha",
    )(x, x)
    x = layers.LayerNormalization(name=f"{name}_ln1")(layers.Add()([x, attn]))
    ff = layers.Dense(ff_dim, activation="relu", name=f"{name}_ff1")(x)
    ff = layers.Dense(dim, name=f"{name}_ff2")(ff)
    ff = layers.Dropout(dropout, name=f"{name}_ff_drop")(ff)
    return layers.LayerNormalization(name=f"{name}_ln2")(layers.Add()([x, ff]))


@register("cnn_attention")
def build(cfg: dict):
    window_len = cfg.get("window_len", 4096)
    conv_filters = cfg.get("conv_filters", [32, 64, 128, 128, 128])  # /2 each -> T/32
    kernel = cfg.get("kernel_size", 7)
    dim = cfg.get("model_dim", 128)
    num_blocks = cfg.get("num_blocks", 2)
    num_heads = cfg.get("num_heads", 4)
    ff_dim = cfg.get("ff_dim", 256)
    dropout = cfg.get("dropout", 0.1)

    inputs = keras.Input(shape=(window_len, 2), name="strain")
    x = layers.BatchNormalization(name="input_bn")(inputs)
    for i, f in enumerate(conv_filters):
        x = layers.Conv1D(f, kernel, strides=2, padding="same", name=f"conv_{i}")(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Activation("relu", name=f"relu_{i}")(x)

    x = layers.Dense(dim, name="proj")(x)
    x = PositionalEmbedding(name="pos_embed")(x)
    for b in range(num_blocks):
        x = _encoder_block(x, dim, num_heads, ff_dim, dropout, name=f"encoder_{b}")

    # Per-token transformer output before attention pooling — available as a
    # branch point for heads that benefit from finer-grained features (e.g. q,
    # whose mass-ratio information may live in subtle relative timing/amplitude
    # differences that global attention pooling averages away). See
    # q_head_action_plan.md Phase 3 step 13 / Phase 3.2.
    tokens = x  # (B, T, dim)
    features = AttentionPooling(name="attn_pool")(x)
    return inputs, features, {"q_tokens": tokens}
