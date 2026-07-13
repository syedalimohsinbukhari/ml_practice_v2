"""Multi-head loss machinery.

Per-head loss is Huber. Head balancing is either:

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
    weight below exp(-3) ~ 5% and entrench collapse.
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
        loss_table = {"huber": self.huber}
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
            clamp = float(cfg.get("log_var_clamp", 3.0))
            self.log_vars = {
                h: self.add_weight(
                    name=f"log_var_{h}",
                    shape=(),
                    initializer="zeros",
                    trainable=True,
                    constraint=ClampConstraint(clamp),
                )
                for h in self.head_names
            }
        elif self.weighting != "fixed":
            raise ValueError(f"unknown weighting {self.weighting!r}")

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.head_mae = {
            h: keras.metrics.Mean(name=f"mae_{h}") for h in self.head_names
        }
        self.head_r2 = {
            h: keras.metrics.R2Score(name=f"r2_{h}") for h in self.head_names
        }
        self.head_std_ratio = {
            h: StdRatio(name=f"std_ratio_{h}") for h in self.head_names
        }
        if self.weighting == "uncertainty":
            self.weight_trackers = {
                h: keras.metrics.Mean(name=f"weight_{h}") for h in self.head_names
            }
        else:
            self.weight_trackers = {}

    def call(self, x, training=False):
        return self.base(x, training=training)

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            *self.head_mae.values(),
            *self.head_r2.values(),
            *self.head_std_ratio.values(),
            *self.weight_trackers.values(),
        ]

    def _total_loss(self, y_true, y_pred, sample_weight=None):
        total = 0.0
        for h in self.head_names:
            head_loss = self.head_loss[h](
                y_true[h], y_pred[h], sample_weight=sample_weight
            )
            if self.weighting == "uncertainty":
                s = self.log_vars[h]
                total = total + tf.exp(-s) * head_loss + s
            else:
                total = total + self.fixed_weights[h] * head_loss
            if self.variance_penalty > 0.0:
                spread_gap = tf.math.reduce_std(y_pred[h]) - tf.math.reduce_std(y_true[h])
                total = total + self.variance_penalty * tf.square(spread_gap)
        return total

    def _update_metrics(self, loss, y_true, y_pred):
        self.loss_tracker.update_state(loss)
        for h in self.head_names:
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
