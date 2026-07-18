"""SumDiffTrainer — φc/ψ degeneracy PoC custom trainer.

Extends ``MultiHeadTrainer`` to replace individual φc and ψ PERIODIC losses
with an isotropic circular loss on derived sum/difference combo vectors,
optionally weighted by an inclination-based curriculum weight ``w(ι)``.

Two modes (controlled by the ``mode`` init parameter):

  ``"baseline"`` (Run A)
      Circular loss on individual φc and ψ vectors, no curriculum weighting.
      Uses the *same loss function class* as Run B so the only variable
      changed between A and B is the gradient grouping.

  ``"poc"`` (Run B)
      Circular loss on combo_A (φc + 2ψ) and combo_B (φc − 2ψ), with
      ``w(ι)`` applied to whichever combo Step 1.1 determines is poorly
      constrained.  No individual φc or ψ loss is applied.

Integration: this file imports from the parent ``MultiHeadTrainer`` and
from the local ``transform_utils`` / ``curriculum`` modules.  It does NOT
modify any main-branch source files.
"""

from __future__ import annotations

import keras
import tensorflow as tf

from curriculum import tf_w_iota
from gwml.heads_spec import HEAD_SPECS
from gwml.training.losses import (
    ClampConstraint,
    MultiHeadTrainer,
    _vmf_kappa_from_raw,
)
from transform_utils import (
    tf_complex_mul,
    tf_complex_mul_conj,
    tf_normalize_unit,
)


# ---------------------------------------------------------------------------
# Combo-specific metrics
# ---------------------------------------------------------------------------


class CircularMetric(keras.metrics.Metric):
    """Per-sample ``1 − cos(Δθ)`` tracked as an epoch-level mean.

    Replaces the standard MAE for combo heads where losses are circular
    rather than Huber.  Inputs are (N, 2) unit vectors.
    """

    def __init__(self, name: str = "circular_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", shape=(), initializer="zeros")
        self.count = self.add_weight(name="count", shape=(), initializer="zeros")

    def update_state(self, y_true_vec, y_pred_vec, sample_weight=None):
        dot = (
            y_pred_vec[..., 0] * y_true_vec[..., 0]
            + y_pred_vec[..., 1] * y_true_vec[..., 1]
        )
        loss = 1.0 - dot
        self.total.assign_add(tf.reduce_sum(loss))
        self.count.assign_add(tf.cast(tf.size(loss), self.dtype))

    def result(self):
        return self.total / tf.maximum(self.count, 1.0)


# ---------------------------------------------------------------------------
# SumDiffTrainer
# ---------------------------------------------------------------------------


class SumDiffTrainer(MultiHeadTrainer):
    """Trainer that replaces individual φc/ψ losses with circular combo losses.

    The model outputs ``coa_phase``, ``polarization_angle``, and ``inclination``
    as standard PERIODIC heads (built by ``attach_heads()``).  This trainer
    intercepts those outputs in ``_total_loss()``, applying the isotropic
    circular loss (Appendix A.3) rather than per-component Huber.

    Parameters
    ----------
    base : keras.Model
        The functional multi-head model.
    loss_cfg : dict or None
        Loss configuration.  Must include ``"weighting": "uncertainty"``.
        Additional PoC-specific keys:

        - ``mode``: ``"baseline"`` or ``"poc"`` (default: ``"poc"``).
        - ``well_constrained_combo``: ``"combo_A"`` or ``"combo_B"``
          (determined by Step 1.1).
        - ``combo_log_var_clamp``: scalar or dict override for combo log-var
          clamps (falls back to ``log_var_clamp``).
    heads : list of str or None
        Active head names.  Must include ``"coa_phase"``,
        ``"polarization_angle"``, and ``"inclination"``.
    mode : str
        ``"baseline"`` or ``"poc"``.
    w_iota_fn : callable or None
        ``w(cos_iota) -> weight in [0, 1]``.  Defaults to ``tf_w_iota``
        (the ``1 − cos²ι`` fallback).
    well_constrained_combo : str
        ``"combo_A"`` or ``"combo_B"`` — which combo Step 1.1 found to be
        well-constrained (for cos ι > 0, if sign-dependent).  The *other*
        combo receives the curriculum weight.
    sign_dependent_combo : bool
        If True (Step 1.1 found a sign flip), the well-constrained combo
        is selected dynamically at batch level based on ``sign(cos_iota)``
        rather than using a single fixed label.  When cos_iota > 0 the
        ``well_constrained_combo`` is used as-is; when cos_iota < 0 the
        opposite combo is treated as well-constrained.
    """

    # Heads whose individual losses are removed in "poc" mode.
    _SUMDIFF_SOURCE_HEADS = ("coa_phase", "polarization_angle")

    def __init__(
        self,
        base: keras.Model,
        loss_cfg: dict | None = None,
        heads=None,
        mode: str = "poc",
        w_iota_fn: callable | None = None,
        well_constrained_combo: str = "combo_A",
        sign_dependent_combo: bool = False,
        **kwargs,
    ):
        self._poc_mode = mode
        self._w_iota_fn = w_iota_fn or tf_w_iota
        self._well_constrained = well_constrained_combo
        self._sign_dependent = sign_dependent_combo
        self._poor_combo = (
            "combo_B" if well_constrained_combo == "combo_A" else "combo_A"
        )

        cfg = loss_cfg or {}

        # ---- Call parent (creates log_vars for all head_names) ----
        super().__init__(base, cfg, heads, **kwargs)

        if self.weighting != "uncertainty":
            raise ValueError(
                "SumDiffTrainer requires weighting='uncertainty'; "
                f"got {self.weighting!r}"
            )

        # ---- Patch log_vars for "poc" mode ----
        if self._poc_mode == "poc":
            self._patch_log_vars(cfg)

        # ---- Build combo-specific metrics ----
        self._combo_metrics = self._build_combo_metrics()

        # ---- Remove stale parent metrics for replaced heads ----
        if self._poc_mode == "poc":
            for h in self._SUMDIFF_SOURCE_HEADS:
                self.head_mae.pop(h, None)
                self.head_r2.pop(h, None)
                self.head_std_ratio.pop(h, None)
                self.weight_trackers.pop(h, None)

    # ------------------------------------------------------------------
    # Constructor helpers
    # ------------------------------------------------------------------

    def _patch_log_vars(self, cfg: dict) -> None:
        """Remove φc/ψ log_vars; add combo_A / combo_B log_vars."""
        for h in self._SUMDIFF_SOURCE_HEADS:
            self.log_vars.pop(h, None)
            self.head_loss.pop(h, None)  # also remove stale huber registrations

        combo_clamp_cfg = cfg.get("combo_log_var_clamp", None)
        if combo_clamp_cfg is None:
            combo_clamp_cfg = cfg.get("log_var_clamp", 3.0)

        clamps = self._resolve_combo_clamps(combo_clamp_cfg)
        for name in ("combo_A", "combo_B"):
            limit = clamps.get(name, clamps.get("default", 3.0))
            self.log_vars[name] = self.add_weight(
                name=f"log_var_{name}",
                shape=(),
                initializer="zeros",
                trainable=True,
                constraint=ClampConstraint(limit),
            )

    def _resolve_combo_clamps(self, clamp_cfg) -> dict[str, float]:
        """Same pattern as ``_resolve_clamps`` but for combo_A, combo_B."""
        if isinstance(clamp_cfg, dict):
            default = float(clamp_cfg.get("default", 3.0))
            return {
                "combo_A": float(clamp_cfg.get("combo_A", default)),
                "combo_B": float(clamp_cfg.get("combo_B", default)),
                "default": default,
            }
        v = float(clamp_cfg)
        return {"combo_A": v, "combo_B": v, "default": v}

    def _build_combo_metrics(self) -> dict:
        """Create per-combo metrics depending on mode."""
        metrics = {}
        if self._poc_mode == "poc":
            for name in ("combo_A", "combo_B"):
                metrics[f"circular_loss_{name}"] = CircularMetric(
                    name=f"circular_loss_{name}"
                )
                metrics[f"weight_{name}"] = keras.metrics.Mean(
                    name=f"weight_{name}"
                )
        else:
            # Baseline: circular loss metric on individual φc and ψ
            for h in self._SUMDIFF_SOURCE_HEADS:
                metrics[f"circular_loss_{h}"] = CircularMetric(
                    name=f"circular_loss_{h}"
                )
        return metrics

    # ------------------------------------------------------------------
    # metrics property — includes combo metrics
    # ------------------------------------------------------------------

    @property
    def metrics(self):
        base = [
            self.loss_tracker,
            *self.head_mae.values(),
            *self.head_r2.values(),
            *self.head_std_ratio.values(),
            *self.head_kappa.values(),
            *self.weight_trackers.values(),
            *self._combo_metrics.values(),
        ]
        return base

    # ------------------------------------------------------------------
    # Combo vector construction (shared between loss & metrics)
    # ------------------------------------------------------------------

    def _build_combo_vectors(self, y_true, y_pred):
        """Extract φc/ψ predicted & true vectors, build combo_A / combo_B.

        Returns a flat tuple; callers unpack by position.

        Returns
        -------
        z_phic_true, z_phic_pred  : (N, 2) unit vectors
        z_psi_true, z_psi_pred    : (N, 2) unit vectors
        combo_A_true, combo_A_pred : (N, 2) unit vectors at φc+2ψ
        combo_B_true, combo_B_pred : (N, 2) unit vectors at φc−2ψ
        cos_iota                  : (N,) cos(ι_true)
        """
        z_phic_true = y_true["coa_phase"]
        z_phic_pred = y_pred["coa_phase"]
        z_psi_true = y_true["polarization_angle"]
        z_psi_pred = y_pred["polarization_angle"]

        # Normalise predicted vectors; true vectors are unit by construction
        # but normalise for safety.
        z_phic_pred = tf_normalize_unit(z_phic_pred)
        z_psi_pred = tf_normalize_unit(z_psi_pred)
        z_phic_true = tf_normalize_unit(z_phic_true)
        z_psi_true = tf_normalize_unit(z_psi_true)

        # Build combos (A.2)
        # z_ψ already encodes 2ψ internally (PERIODIC, period=π)
        combo_A_true = tf_complex_mul(z_phic_true, z_psi_true)
        combo_A_pred = tf_complex_mul(z_phic_pred, z_psi_pred)
        combo_B_true = tf_complex_mul_conj(z_phic_true, z_psi_true)
        combo_B_pred = tf_complex_mul_conj(z_phic_pred, z_psi_pred)

        # cos(ι_true) from inclination head (PERIODIC, period=2π)
        # y_true["inclination"] = [sin(ι), cos(ι)] → column 1 = cos(ι)
        cos_iota = y_true["inclination"][:, 1]

        return (
            z_phic_true, z_phic_pred,
            z_psi_true, z_psi_pred,
            combo_A_true, combo_A_pred,
            combo_B_true, combo_B_pred,
            cos_iota,
        )

    # ------------------------------------------------------------------
    # _total_loss overrides
    # ------------------------------------------------------------------

    def _total_loss(self, y_true, y_pred, sample_weight=None):
        if self._poc_mode == "baseline":
            return self._baseline_total_loss(y_true, y_pred, sample_weight)
        return self._poc_total_loss(y_true, y_pred, sample_weight)

    # ------------------------------------------------------------------
    # Shared helper: loss for all heads EXCEPT the sumdiff source heads
    # ------------------------------------------------------------------

    def _other_heads_loss(self, y_true, y_pred, sample_weight) -> tf.Tensor:
        """Compute uncertainty-weighted loss for every head that is NOT
        a sumdiff source head (coa_phase, polarization_angle).

        Replicates the parent ``_total_loss`` logic exactly so that the
        non-φc/ψ heads are treated identically to a standard training run.
        """
        total = tf.constant(0.0)
        for h in self.head_names:
            if h in self._SUMDIFF_SOURCE_HEADS:
                continue
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
            if self.variance_penalty > 0.0 and HEAD_SPECS[h].loss != "vmf":
                spread_gap = tf.math.reduce_std(y_pred[h]) - tf.math.reduce_std(
                    y_true[h]
                )
                total = total + self.variance_penalty * tf.square(spread_gap)
        return total

    # ------------------------------------------------------------------
    # Run A — baseline (circular loss on individual φc / ψ)
    # ------------------------------------------------------------------

    def _baseline_total_loss(self, y_true, y_pred, sample_weight=None):
        """Circular loss on coa_phase and polarization_angle individually.

        Uses the SAME ``1 − dot(pred, true)`` circular loss as Run B —
        only the gradient grouping differs.  No curriculum weighting.
        """
        total = tf.constant(0.0)

        z_phic_true, z_phic_pred, z_psi_true, z_psi_pred, _, _, _, _, _ = (
            self._build_combo_vectors(y_true, y_pred)
        )

        # Circular loss on φc
        loss_phic = 1.0 - tf.reduce_sum(z_phic_true * z_phic_pred, axis=-1)
        s_phic = self.log_vars["coa_phase"]
        total = total + tf.exp(-s_phic) * tf.reduce_mean(loss_phic) + s_phic

        # Circular loss on ψ
        loss_psi = 1.0 - tf.reduce_sum(z_psi_true * z_psi_pred, axis=-1)
        s_psi = self.log_vars["polarization_angle"]
        total = total + tf.exp(-s_psi) * tf.reduce_mean(loss_psi) + s_psi

        # All other heads (standard losses)
        total = total + self._other_heads_loss(y_true, y_pred, sample_weight)

        return total

    # ------------------------------------------------------------------
    # Run B — PoC (circular loss on combos, curriculum-weighted)
    # ------------------------------------------------------------------

    def _poc_total_loss(self, y_true, y_pred, sample_weight=None):
        """Circular loss on combo_A & combo_B, with w(ι) on the
        poorly-constrained combo.

        All φc/ψ training signal flows through the combo losses.  No
        individual φc or ψ loss is applied.
        """
        total = tf.constant(0.0)

        (_, _, _, _,
         combo_A_true, combo_A_pred,
         combo_B_true, combo_B_pred,
         cos_iota) = self._build_combo_vectors(y_true, y_pred)

        # Isotropic circular loss on each combo (A.3)
        loss_A_per_sample = 1.0 - tf.reduce_sum(
            combo_A_true * combo_A_pred, axis=-1
        )
        loss_B_per_sample = 1.0 - tf.reduce_sum(
            combo_B_true * combo_B_pred, axis=-1
        )

        # Build per-sample weight for each combo.  This encodes BOTH:
        #   (a) curriculum weighting w(ι) on the poorly-constrained combo
        #   (b) sign-dependent good/poor assignment when cos ι flips sign
        w_iota = self._w_iota_fn(cos_iota)  # (N,) ∈ [0, 1]

        if self._sign_dependent:
            # cos ι ≥ 0 → well_constrained_combo is good (weight=1),
            #              the other combo is poor (weight=w_iota)
            # cos ι < 0 → roles swap
            pos_mask = tf.cast(cos_iota >= 0.0, tf.float32)  # 1 where cos≥0
            neg_mask = 1.0 - pos_mask  # 1 where cos<0

            if self._well_constrained == "combo_A":
                # combo_A good when cos≥0, poor when cos<0
                w_A = pos_mask + neg_mask * w_iota
                w_B = neg_mask + pos_mask * w_iota
            else:
                # combo_B good when cos≥0, poor when cos<0
                w_B = pos_mask + neg_mask * w_iota
                w_A = neg_mask + pos_mask * w_iota
        else:
            # Fixed assignment (no sign flip)
            if self._well_constrained == "combo_A":
                w_A = tf.ones_like(w_iota)  # good, no suppression
                w_B = w_iota  # poor, curriculum-weighted
            else:
                w_B = tf.ones_like(w_iota)
                w_A = w_iota

        # Per-combo weighted mean loss
        loss_A_mean = tf.reduce_mean(w_A * loss_A_per_sample)
        loss_B_mean = tf.reduce_mean(w_B * loss_B_per_sample)

        s_A = self.log_vars["combo_A"]
        s_B = self.log_vars["combo_B"]

        total = total + tf.exp(-s_A) * loss_A_mean + s_A
        total = total + tf.exp(-s_B) * loss_B_mean + s_B

        # All other heads (standard losses)
        total = total + self._other_heads_loss(y_true, y_pred, sample_weight)

        return total

    # ------------------------------------------------------------------
    # Metrics update
    # ------------------------------------------------------------------

    def _update_metrics(self, loss, y_true, y_pred):
        """Override: update base metrics (skipping sumdiff source heads),
        then add combo-specific circular-loss metrics."""
        # Replicate parent's _update_metrics but skip the sumdiff source heads
        # (their individual losses are removed; we track combos instead).
        self.loss_tracker.update_state(loss)
        for h in self.head_names:
            if h in self._SUMDIFF_SOURCE_HEADS and self._poc_mode == "poc":
                continue  # combo metrics handled below
            if HEAD_SPECS[h].loss == "vmf":
                kappa = _vmf_kappa_from_raw(y_pred[f"{h}_kappa_raw"])
                self.head_kappa[h].update_state(tf.reduce_mean(kappa))
            else:
                self.head_mae[h].update_state(
                    tf.reduce_mean(tf.abs(y_true[h] - y_pred[h]))
                )
                self.head_r2[h].update_state(y_true[h], y_pred[h])
                self.head_std_ratio[h].update_state(y_true[h], y_pred[h])
            if self.weight_trackers and h in self.weight_trackers:
                self.weight_trackers[h].update_state(
                    tf.exp(-self.log_vars[h])
                )

        if self._poc_mode == "poc":
            (_, _, _, _,
             combo_A_true, combo_A_pred,
             combo_B_true, combo_B_pred,
             _) = self._build_combo_vectors(y_true, y_pred)

            self._combo_metrics["circular_loss_combo_A"].update_state(
                combo_A_true, combo_A_pred
            )
            self._combo_metrics["circular_loss_combo_B"].update_state(
                combo_B_true, combo_B_pred
            )

            # Track effective uncertainty weights
            self._combo_metrics["weight_combo_A"].update_state(
                tf.exp(-self.log_vars["combo_A"])
            )
            self._combo_metrics["weight_combo_B"].update_state(
                tf.exp(-self.log_vars["combo_B"])
            )

        else:
            # Baseline: track circular loss on individual heads
            z_phic_true, z_phic_pred, z_psi_true, z_psi_pred, _, _, _, _, _ = (
                self._build_combo_vectors(y_true, y_pred)
            )
            self._combo_metrics["circular_loss_coa_phase"].update_state(
                z_phic_true, z_phic_pred
            )
            self._combo_metrics["circular_loss_polarization_angle"].update_state(
                z_psi_true, z_psi_pred
            )
