"""Multi-head loss machinery.

Per-head loss is Huber (default) or closed-form von Mises-Fisher NLL for
the sky_position head (``loss: "vmf"`` in the head spec). Head balancing is either:

  - "uncertainty" (default): Kendall, Gal & Cipolla (2018). Each head gets a
    learnable log-variance s_h; total = sum_h exp(-s_h) * L_h + s_h. The
    exp(-s_h) weights are exposed as metrics (``weight_<head>``) so training
    logs show which parameters the data constrains.
  - "fixed": hand-set weights from the config.

Mean-collapse safeties (a GW-regression failure mode where every head predicts
the target mean with ~0 variance):

  - Per-head R^2 (``r2_<head>``) and std(pred)/std(true) (``std_ratio_<head>``)
    are tracked as epoch metrics — collapse reads as r2 -> 0, std_ratio -> 0.
  - s_h is clamped to +/- log_var_clamp (default 3.0) via a variable
    constraint, so uncertainty weighting can never drive a head's effective
    weight below exp(-3) ~ 5% and entrench collapse. log_var_clamp accepts a
    per-head dict ({"default": 3.0, "<head>": tighter_value}) — useful when
    one head's train loss shrinks fast enough (e.g. via overfitting) to drag
    every head into the same clamp ceiling, erasing the weighting scheme's
    per-head signal (see q_head_action_plan.md).
  - Optional variance-matching penalty (``variance_penalty`` > 0): adds
    lambda * (std(pred) - std(true))^2 per head, computed per batch. Off by
    default — large lambda can distort calibration.

Optional SNR sample weighting arrives via the dataset's sample_weight element
(see gwml.data.loader.snr_sample_weights) and applies to every head equally.
"""

from __future__ import annotations

import keras
import tensorflow as tf

from gwml.heads_spec import DEFAULT_HEADS, HEAD_SPECS, resolve_heads


def _vmf_kappa_from_raw(kappa_raw):
    """Unconstrained -> kappa > 0.1 (floor prevents kappa->0 log-singularity)."""
    return keras.ops.softplus(kappa_raw) + 0.1


def vmf_nll_loss(y_true_unitvec, mu_raw, kappa_raw):
    """Closed-form vMF negative log-likelihood on S^2.

    y_true_unitvec: (B, 3), already unit-norm target vectors.
    mu_raw:         (B, 3), unnormalized network output.
    kappa_raw:      (B, 1), unconstrained network output.
    """
    mu = mu_raw / keras.ops.maximum(
        keras.ops.sqrt(keras.ops.sum(keras.ops.square(mu_raw), axis=-1,
                                     keepdims=True)),
        1e-8,
    )
    kappa = _vmf_kappa_from_raw(kappa_raw)[..., 0]  # (B,)

    cos_sep = keras.ops.sum(mu * y_true_unitvec, axis=-1)  # (B,)

    # stable log(sinh(kappa)) = kappa + log(1 - exp(-2*kappa)) - log(2)
    log_sinh_kappa = (
        kappa
        + keras.ops.log(
            keras.ops.maximum(1.0 - keras.ops.exp(-2.0 * kappa), 1e-12)
        )
        - keras.ops.log(2.0)
    )

    nll = (
        -keras.ops.log(kappa)
        + keras.ops.log(4.0 * 3.141592653589793)
        + log_sinh_kappa
        - kappa * cos_sep
    )
    return keras.ops.mean(nll)


class ClampConstraint(keras.constraints.Constraint):
    """Keep a scalar variable within [-limit, limit]."""

    def __init__(self, limit: float):
        self.limit = float(limit)

    def __call__(self, w):
        return keras.ops.clip(w, -self.limit, self.limit)

    def get_config(self):
        return {"limit": self.limit}


class StdRatio(keras.metrics.Metric):
    """std(pred) / std(true), accumulated exactly over the epoch.

    ~1 for a healthy head; -> 0 when the head collapses to the mean.
    """

    def __init__(self, name="std_ratio", **kwargs):
        super().__init__(name=name, **kwargs)
        self.count = self.add_weight(name="count", shape=(), initializer="zeros")
        self.sum_p = self.add_weight(name="sum_p", shape=(), initializer="zeros")
        self.sumsq_p = self.add_weight(name="sumsq_p", shape=(), initializer="zeros")
        self.sum_t = self.add_weight(name="sum_t", shape=(), initializer="zeros")
        self.sumsq_t = self.add_weight(name="sumsq_t", shape=(), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        t = tf.cast(tf.reshape(y_true, [-1]), self.dtype)
        p = tf.cast(tf.reshape(y_pred, [-1]), self.dtype)
        self.count.assign_add(tf.cast(tf.size(t), self.dtype))
        self.sum_p.assign_add(tf.reduce_sum(p))
        self.sumsq_p.assign_add(tf.reduce_sum(p * p))
        self.sum_t.assign_add(tf.reduce_sum(t))
        self.sumsq_t.assign_add(tf.reduce_sum(t * t))

    def result(self):
        n = tf.maximum(self.count, 1.0)
        var_p = tf.maximum(self.sumsq_p / n - (self.sum_p / n) ** 2, 0.0)
        var_t = tf.maximum(self.sumsq_t / n - (self.sum_t / n) ** 2, 0.0)
        return tf.sqrt(var_p) / (tf.sqrt(var_t) + 1e-12)


class MultiHeadTrainer(keras.Model):
    """Wraps the functional multi-head model with a custom multi-task loss.

    Weights are saved/loaded via save_weights/load_weights; the model is
    rebuilt from its YAML config, so no custom serialization is needed.
    """

    def __init__(self, base: keras.Model, loss_cfg: dict | None = None,
                 heads=None, **kwargs):
        super().__init__(**kwargs)
        cfg = loss_cfg or {}
        self.base = base
        self.head_names = [s.name for s in resolve_heads(heads or DEFAULT_HEADS)]
        self.huber = keras.losses.Huber(delta=cfg.get("huber_delta", 1.0))
        # Head -> loss binding comes from the spec registry, never the YAML.
        # Periodic heads are (sin, cos) pairs, so Huber on the pair is correct.
        loss_table = {"huber": self.huber, "vmf": vmf_nll_loss}
        self.head_loss = {}
        for h in self.head_names:
            kind = HEAD_SPECS[h].loss
            if kind not in loss_table:
                raise ValueError(
                    f"head {h!r} declares loss {kind!r}, which MultiHeadTrainer "
                    f"does not implement; known: {sorted(loss_table)}"
                )
            self.head_loss[h] = loss_table[kind]
        self.weighting = cfg.get("weighting", "uncertainty")
        self.variance_penalty = float(cfg.get("variance_penalty", 0.0) or 0.0)
        fixed = cfg.get("fixed_weights", {})
        self.fixed_weights = {h: float(fixed.get(h, 1.0)) for h in self.head_names}

        if self.weighting == "uncertainty":
            clamps = self._resolve_clamps(cfg.get("log_var_clamp", 3.0))
            self.log_vars = {
                h: self.add_weight(
                    name=f"log_var_{h}",
                    shape=(),
                    initializer="zeros",
                    trainable=True,
                    constraint=ClampConstraint(clamps[h]),
                )
                for h in self.head_names
            }
        elif self.weighting != "fixed":
            raise ValueError(f"unknown weighting {self.weighting!r}")

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.head_mae = {
            h: keras.metrics.Mean(name=f"mae_{h}")
            for h in self.head_names
            if HEAD_SPECS[h].loss != "vmf"
        }
        self.head_r2 = {
            h: keras.metrics.R2Score(name=f"r2_{h}")
            for h in self.head_names
            if HEAD_SPECS[h].loss != "vmf"
        }
        self.head_std_ratio = {
            h: StdRatio(name=f"std_ratio_{h}")
            for h in self.head_names
            if HEAD_SPECS[h].loss != "vmf"
        }
        self.head_kappa = {
            h: keras.metrics.Mean(name=f"kappa_{h}")
            for h in self.head_names
            if HEAD_SPECS[h].loss == "vmf"
        }
        if self.weighting == "uncertainty":
            self.weight_trackers = {
                h: keras.metrics.Mean(name=f"weight_{h}") for h in self.head_names
            }
        else:
            self.weight_trackers = {}

    def _resolve_clamps(self, clamp_cfg) -> dict:
        """log_var_clamp accepts a scalar (all heads) or a dict of overrides.

        Dict form: {"default": 3.0, "q": 1.0} — heads absent from the dict
        fall back to "default" (itself defaulting to 3.0). A head with a
        tighter clamp can't have its uncertainty weight run away to the same
        ceiling as heads whose train loss shrinks fastest; see
        q_head_action_plan.md Phase 2 step 8.
        """
        if isinstance(clamp_cfg, dict):
            default = float(clamp_cfg.get("default", 3.0))
            return {h: float(clamp_cfg.get(h, default)) for h in self.head_names}
        return {h: float(clamp_cfg) for h in self.head_names}

    def call(self, x, training=False):
        return self.base(x, training=training)

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            *self.head_mae.values(),
            *self.head_r2.values(),
            *self.head_std_ratio.values(),
            *self.head_kappa.values(),
            *self.weight_trackers.values(),
        ]

    def _total_loss(self, y_true, y_pred, sample_weight=None):
        total = 0.0
        for h in self.head_names:
            if HEAD_SPECS[h].loss == "vmf":
                head_loss = self.head_loss[h](
                    y_true[h],
                    y_pred[f"{h}_mu_raw"],
                    y_pred[f"{h}_kappa_raw"],
                )
            else:
                head_loss = self.head_loss[h](
                    y_true[h], y_pred[h], sample_weight=sample_weight
                )
            if self.weighting == "uncertainty":
                s = self.log_vars[h]
                total = total + tf.exp(-s) * head_loss + s
            else:
                total = total + self.fixed_weights[h] * head_loss
            if self.variance_penalty > 0.0:
                # Skip vMF heads: variance penalty on unit-vector components is
                # not physically meaningful.
                if HEAD_SPECS[h].loss != "vmf":
                    spread_gap = tf.math.reduce_std(y_pred[h]) - tf.math.reduce_std(
                        y_true[h]
                    )
                    total = total + self.variance_penalty * tf.square(spread_gap)
        return total

    def _update_metrics(self, loss, y_true, y_pred):
        self.loss_tracker.update_state(loss)
        for h in self.head_names:
            if HEAD_SPECS[h].loss == "vmf":
                # vMF heads: track kappa; per-axis MAE/R²/std_ratio on 3D
                # unit-vector components are not physically meaningful.
                kappa = _vmf_kappa_from_raw(y_pred[f"{h}_kappa_raw"])
                self.head_kappa[h].update_state(tf.reduce_mean(kappa))
            else:
                self.head_mae[h].update_state(
                    tf.reduce_mean(tf.abs(y_true[h] - y_pred[h]))
                )
                self.head_r2[h].update_state(y_true[h], y_pred[h])
                self.head_std_ratio[h].update_state(y_true[h], y_pred[h])
            if self.weight_trackers:
                self.weight_trackers[h].update_state(tf.exp(-self.log_vars[h]))

    def train_step(self, data):
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self._total_loss(y, y_pred, sample_weight)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self._update_metrics(loss, y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        y_pred = self(x, training=False)
        loss = self._total_loss(y, y_pred, sample_weight)
        self._update_metrics(loss, y, y_pred)
        return {m.name: m.result() for m in self.metrics}
