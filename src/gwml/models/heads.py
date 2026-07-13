"""Multi-head output block shared by every trunk.

Which heads exist, their output width, and their activation all come from the
head registry (gwml.heads_spec) — the experiment config only names the active
heads. Each head is a small MLP (Dense hidden -> Dense dim) named after its
target so tf.data can feed targets by dict.

Bounded activations (sigmoid for [0,1]-mapped targets, tanh for sin/cos pairs)
guarantee physical predictions; ``head_cfg.bounded: false`` downgrades all of
them to linear for A/B experiments.
"""

from __future__ import annotations

import keras
from keras import layers

from gwml.heads_spec import DEFAULT_HEADS, resolve_heads


def attach_heads(inputs, features, heads=None, cfg: dict | None = None) -> keras.Model:
    cfg = cfg or {}
    hidden = cfg.get("hidden_units", 64)
    bounded = cfg.get("bounded", True)

    outputs = {}
    for spec in resolve_heads(heads or DEFAULT_HEADS):
        x = layers.Dense(hidden, activation="relu",
                         name=f"{spec.name}_hidden")(features)
        activation = spec.activation if bounded else "linear"
        outputs[spec.name] = layers.Dense(
            spec.dim, activation=activation, name=spec.name
        )(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="gw_multihead")
