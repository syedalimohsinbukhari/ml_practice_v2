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

from gwml.heads_spec import DEFAULT_HEADS, TransformKind, resolve_heads


def attach_heads(inputs, features, heads=None, cfg: dict | None = None,
                 extra_features: dict | None = None) -> keras.Model:
    """Attach output heads to a trunk's pooled features.

    Parameters
    ----------
    extra_features : dict[str, keras.KerasTensor] or None
        Optional per-head feature tensors keyed by head name. When a head's
        ``per_head.<name>.branch`` config references a key here, that head
        connects to this tensor instead of the global ``features``. 3-D
        tensors (B, T, D) are GAP-pooled to (B, D) first. Used by
        ``cnn_attention`` to give q its own branch from the per-token
        transformer output before ``AttentionPooling``.
    """
    cfg = cfg or {}
    extra_features = extra_features or {}
    default_hidden = cfg.get("hidden_units", 64)
    default_dropout = float(cfg.get("dropout", 0.0) or 0.0)
    default_l2 = float(cfg.get("l2", 0.0) or 0.0)
    bounded = cfg.get("bounded", True)
    default_sigmoid_bias = float(cfg.get("sigmoid_bias", 0.0) or 0.0)
    per_head = cfg.get("per_head", {})

    outputs = {}
    for spec in resolve_heads(heads or DEFAULT_HEADS):
        overrides = per_head.get(spec.name, {})
        hidden = overrides.get("hidden_units", default_hidden)
        dropout = float(overrides.get("dropout", default_dropout) or 0.0)
        l2 = float(overrides.get("l2", default_l2) or 0.0)
        regularizer = keras.regularizers.l2(l2) if l2 > 0.0 else None
        sigmoid_bias = float(overrides.get("sigmoid_bias", default_sigmoid_bias) or 0.0)

        # Per-head feature branching: if this head has a "branch" key in its
        # per_head overrides and the trunk provided that feature tensor, use
        # it instead of the global pooled features. 3-D tensors (time series)
        # are GAP-pooled first.
        head_features = features
        branch_name = overrides.get("branch")
        if branch_name and branch_name in extra_features:
            ef = extra_features[branch_name]
            if len(ef.shape) == 3:
                ef = layers.GlobalAveragePooling1D(
                    name=f"{spec.name}_branch_gap")(ef)
            head_features = ef

        x = layers.Dense(hidden, activation="relu",
                         kernel_regularizer=regularizer,
                         name=f"{spec.name}_hidden")(head_features)
        if dropout > 0.0:
            x = layers.Dropout(dropout, name=f"{spec.name}_dropout")(x)
        activation = spec.activation if bounded else "linear"
        # Sigmoid heads use a configurable bias initializer so a dead
        # (saturated) sigmoid can be nudged away from the asymptote at
        # init time (see q_head_action_plan.md Phase 2.5 / resnet1d).
        bias_init = "zeros"
        if sigmoid_bias != 0.0 and activation == "sigmoid":
            import tensorflow as tf
            bias_init = tf.constant_initializer(sigmoid_bias)
        if spec.transform is TransformKind.SPHERICAL_UNIT_VECTOR:
            # vMF head: two output tensors jointly consumed by one loss.
            # Use _ separator (not /) so the key matches the Keras output name.
            outputs[f"{spec.name}_mu_raw"] = layers.Dense(
                3, activation="linear",
                kernel_regularizer=regularizer,
                name=f"{spec.name}_mu_raw"
            )(x)
            outputs[f"{spec.name}_kappa_raw"] = layers.Dense(
                1, activation="linear",
                kernel_regularizer=regularizer,
                name=f"{spec.name}_kappa_raw"
            )(x)
        else:
            outputs[spec.name] = layers.Dense(
                spec.dim, activation=activation,
                kernel_regularizer=regularizer,
                bias_initializer=bias_init,
                name=spec.name
            )(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="gw_multihead")
