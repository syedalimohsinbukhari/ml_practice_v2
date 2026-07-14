"""Multi-head output block shared by every trunk.

Which heads exist, their output width, and their activation all come from the
head registry (gwml.heads_spec) — the experiment config only names the active
heads. Each head is a small MLP (Dense hidden -> Dense dim) named after its
target so tf.data can feed targets by dict.

Bounded activations (sigmoid for [0,1]-mapped targets, tanh for sin/cos pairs)
guarantee physical predictions; ``head_cfg.bounded: false`` downgrades all of
them to linear for A/B experiments.

Regularization (``dropout``, ``l2``, ``hidden_units``) defaults apply to every
head and can be overridden per head via ``head_cfg.per_head.<name>`` — e.g. to
shrink or dropout-regularize a head that overfits faster than the others
sharing the same trunk (see q_head_action_plan.md Phase 2 step 9). Defaults of
0.0/global hidden_units reproduce the original unregularized graph exactly.
"""

from __future__ import annotations

import keras
from keras import layers

from gwml.heads_spec import DEFAULT_HEADS, resolve_heads


def attach_heads(inputs, features, heads=None, cfg: dict | None = None) -> keras.Model:
    cfg = cfg or {}
    default_hidden = cfg.get("hidden_units", 64)
    default_dropout = float(cfg.get("dropout", 0.0) or 0.0)
    default_l2 = float(cfg.get("l2", 0.0) or 0.0)
    bounded = cfg.get("bounded", True)
    per_head = cfg.get("per_head", {})

    outputs = {}
    for spec in resolve_heads(heads or DEFAULT_HEADS):
        overrides = per_head.get(spec.name, {})
        hidden = overrides.get("hidden_units", default_hidden)
        dropout = float(overrides.get("dropout", default_dropout) or 0.0)
        l2 = float(overrides.get("l2", default_l2) or 0.0)
        regularizer = keras.regularizers.l2(l2) if l2 > 0.0 else None

        x = layers.Dense(hidden, activation="relu",
                         kernel_regularizer=regularizer,
                         name=f"{spec.name}_hidden")(features)
        if dropout > 0.0:
            x = layers.Dropout(dropout, name=f"{spec.name}_dropout")(x)
        activation = spec.activation if bounded else "linear"
        outputs[spec.name] = layers.Dense(
            spec.dim, activation=activation,
            kernel_regularizer=regularizer,
            name=spec.name
        )(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="gw_multihead")
