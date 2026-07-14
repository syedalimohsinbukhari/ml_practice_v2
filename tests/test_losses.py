"""Loss machinery: finite losses, learnable log-vars, SNR weight behavior."""

import numpy as np
import pytest

from conftest import TINY_TRUNK_CFGS
from gwml.data.loader import snr_sample_weights
from gwml.data.transforms import HEAD_ORDER, PARAM_COLUMNS
from gwml.models import build_model
from gwml.training.losses import MultiHeadTrainer

WINDOW_LEN = 4096


def _tiny_trainer(loss_cfg):
    base = build_model("cnn_baseline", TINY_TRUNK_CFGS["cnn_baseline"],
                       {"hidden_units": 8})
    return MultiHeadTrainer(base, loss_cfg)


def _fake_batch(n=8, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, WINDOW_LEN, 2)).astype(np.float32)
    y = {h: rng.uniform(0, 1, size=(n, 1)).astype(np.float32) for h in HEAD_ORDER}
    return x, y


def test_warmup_lr_ramps_linearly_then_stops_touching():
    import keras

    from gwml.training.callbacks import WarmupLR

    trainer = _tiny_trainer({"weighting": "fixed"})
    trainer.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))
    cb = WarmupLR(base_lr=1e-3, warmup_epochs=4)
    cb.set_model(trainer)

    cb.on_epoch_begin(0)
    assert float(trainer.optimizer.learning_rate) == pytest.approx(0.25e-3)
    cb.on_epoch_begin(3)
    assert float(trainer.optimizer.learning_rate) == pytest.approx(1e-3)

    # After warmup the callback must not touch the LR (plateau owns it).
    trainer.optimizer.learning_rate.assign(5e-4)
    cb.on_epoch_begin(4)
    assert float(trainer.optimizer.learning_rate) == pytest.approx(5e-4)


def test_uncertainty_trainer_has_learnable_log_vars():
    trainer = _tiny_trainer({"weighting": "uncertainty"})
    names = [v.name for v in trainer.trainable_variables]
    for head in HEAD_ORDER:
        assert any(f"log_var_{head}" in n for n in names)


def test_fixed_weighting_has_no_log_vars():
    trainer = _tiny_trainer({"weighting": "fixed",
                             "fixed_weights": {h: 1.0 for h in HEAD_ORDER}})
    assert not any("log_var" in v.name for v in trainer.trainable_variables)


def test_unknown_weighting_raises():
    with pytest.raises(ValueError):
        _tiny_trainer({"weighting": "banana"})


@pytest.mark.parametrize("weighting", ["uncertainty", "fixed"])
def test_total_loss_is_finite_scalar(weighting):
    trainer = _tiny_trainer({"weighting": weighting})
    x, y = _fake_batch()
    y_pred = trainer(x, training=False)
    loss = float(trainer._total_loss(y, y_pred))
    assert np.isfinite(loss)


def test_uncertainty_loss_at_zero_logvar_matches_unit_fixed_weights():
    x, y = _fake_batch()
    unc = _tiny_trainer({"weighting": "uncertainty"})
    y_pred = unc(x, training=False)
    # At initialization s_h = 0, so exp(-s)*L + s == 1.0 * L for every head.
    fixed = MultiHeadTrainer(unc.base, {"weighting": "fixed"})
    np.testing.assert_allclose(
        float(unc._total_loss(y, y_pred)),
        float(fixed._total_loss(y, y_pred)),
        rtol=1e-6,
    )


def test_snr_sample_weights_monotonic(synthetic_params):
    w = snr_sample_weights(synthetic_params, alpha=1.0)
    snr = synthetic_params[:, PARAM_COLUMNS["snr"]]
    order = np.argsort(snr)
    assert np.all(np.diff(w[order]) >= 0)
    np.testing.assert_allclose(w, (snr / 10.0) ** 1.0, rtol=1e-6)


def test_collapse_metrics_present_per_head():
    trainer = _tiny_trainer({"weighting": "uncertainty"})
    names = {m.name for m in trainer.metrics}
    for head in HEAD_ORDER:
        assert f"r2_{head}" in names
        assert f"std_ratio_{head}" in names


def test_std_ratio_metric_matches_numpy():
    from gwml.training.losses import StdRatio

    rng = np.random.default_rng(1)
    t = rng.normal(0.0, 2.0, size=(64, 1)).astype(np.float32)
    p = rng.normal(0.0, 0.5, size=(64, 1)).astype(np.float32)
    metric = StdRatio()
    metric.update_state(t[:32], p[:32])  # two batches: accumulation must be exact
    metric.update_state(t[32:], p[32:])
    expected = p.std() / (t.std() + 1e-12)
    np.testing.assert_allclose(float(metric.result()), expected, rtol=1e-4)


def test_std_ratio_detects_collapse():
    from gwml.training.losses import StdRatio

    t = np.random.default_rng(2).normal(size=(64, 1)).astype(np.float32)
    p = np.full_like(t, t.mean())  # collapsed predictor
    metric = StdRatio()
    metric.update_state(t, p)
    assert float(metric.result()) < 1e-6


def test_log_vars_are_clamped():
    trainer = _tiny_trainer({"weighting": "uncertainty", "log_var_clamp": 3.0})
    for head in HEAD_ORDER:
        constraint = trainer.log_vars[head].constraint
        assert constraint is not None
        assert float(constraint(np.float32(10.0))) == pytest.approx(3.0)
        assert float(constraint(np.float32(-10.0))) == pytest.approx(-3.0)


def test_per_head_log_var_clamp():
    trainer = _tiny_trainer({
        "weighting": "uncertainty",
        "log_var_clamp": {"default": 3.0, "q": 1.0},
    })
    for head in HEAD_ORDER:
        limit = 1.0 if head == "q" else 3.0
        constraint = trainer.log_vars[head].constraint
        assert float(constraint(np.float32(10.0))) == pytest.approx(limit)
        assert float(constraint(np.float32(-10.0))) == pytest.approx(-limit)


def test_variance_penalty_changes_loss():
    x, y = _fake_batch()
    base = build_model("cnn_baseline", TINY_TRUNK_CFGS["cnn_baseline"],
                       {"hidden_units": 8})
    plain = MultiHeadTrainer(base, {"weighting": "fixed"})
    penalized = MultiHeadTrainer(base, {"weighting": "fixed",
                                        "variance_penalty": 10.0})
    y_pred = plain(x, training=False)
    l_plain = float(plain._total_loss(y, y_pred))
    l_pen = float(penalized._total_loss(y, y_pred))
    # An untrained model's prediction spread differs from the target spread,
    # so the penalty must strictly increase the loss.
    assert l_pen > l_plain


def test_snr_weighting_changes_loss():
    trainer = _tiny_trainer({"weighting": "fixed"})
    x, y = _fake_batch()
    y_pred = trainer(x, training=False)
    unweighted = float(trainer._total_loss(y, y_pred))
    sw = np.linspace(0.1, 2.0, len(x)).astype(np.float32)
    weighted = float(trainer._total_loss(y, y_pred, sample_weight=sw))
    assert not np.isclose(unweighted, weighted)
