"""Head registry: validation, periodic transforms, dynamic model outputs."""

import numpy as np
import pytest

from conftest import TINY_TRUNK_CFGS
from gwml.data.transforms import TargetTransforms, abs_error
from gwml.heads_spec import (
    DEFAULT_HEADS,
    HEAD_SPECS,
    HeadName,
    TransformKind,
    resolve_heads,
)
from gwml.models import build_model
from gwml.training.losses import MultiHeadTrainer

WINDOW_LEN = 4096


def test_resolve_rejects_unknown_and_injection_time():
    with pytest.raises(ValueError, match="unknown heads"):
        resolve_heads(["mchirp", "banana"])
    with pytest.raises(ValueError, match="unknown heads"):
        resolve_heads(["injection_time"])  # deliberately not a valid head


def test_resolve_rejects_duplicates_and_empty():
    with pytest.raises(ValueError):
        resolve_heads(["q", "q"])
    with pytest.raises(ValueError):
        resolve_heads([])


def test_resolve_accepts_enum_members():
    specs = resolve_heads([HeadName.MCHIRP, HeadName.SKY_POSITION])
    assert [s.name for s in specs] == ["mchirp", "sky_position"]


def test_every_spec_column_matches_structure_md():
    from gwml.heads_spec import PARAM_COLUMNS

    for name, spec in HEAD_SPECS.items():
        # Multi-column heads (e.g. sky_position) use `columns`, not `column`.
        if spec.column is None:
            continue
        assert spec.column == PARAM_COLUMNS[name]


@pytest.mark.parametrize("head", ["coa_phase", "inclination"])
def test_periodic_round_trip_two_pi(head, synthetic_params):
    tr = TargetTransforms(heads=[head]).fit(synthetic_params)
    raw = np.random.default_rng(3).uniform(0, 2 * np.pi, 100)
    encoded = tr.transform_head(head, raw)
    assert encoded.shape == (100, 2)
    assert np.all(np.abs(encoded) <= 1.0)
    recovered = tr.inverse_head(head, encoded)
    # (sin, cos) targets are float32, so angles recover to ~1e-7 rad.
    np.testing.assert_allclose(recovered, raw % (2 * np.pi), atol=1e-6)


def test_polarization_angle_has_pi_period(synthetic_params):
    tr = TargetTransforms(heads=["polarization_angle"]).fit(synthetic_params)
    raw = np.array([0.3, 0.3 + np.pi, 0.3 + 2 * np.pi])
    encoded = tr.transform_head("polarization_angle", raw)
    # psi and psi + pi produce identical strain, so identical encodings
    # (to float32 precision, since targets are cast for training).
    np.testing.assert_allclose(encoded[0], encoded[1], atol=1e-6)
    np.testing.assert_allclose(encoded[0], encoded[2], atol=1e-6)
    recovered = tr.inverse_head("polarization_angle", encoded)
    np.testing.assert_allclose(recovered, raw % np.pi, atol=1e-6)


def test_abs_error_is_wrap_aware():
    err = abs_error("coa_phase", np.array([0.05]), np.array([2 * np.pi - 0.05]))
    np.testing.assert_allclose(err, [0.1], atol=1e-10)
    # Non-periodic heads keep the plain difference.
    np.testing.assert_allclose(abs_error("snr", np.array([9.0]), np.array([12.0])),
                               [3.0])


def test_model_builds_with_custom_heads_and_dims():
    heads = ["mchirp", "sky_position"]
    model = build_model("cnn_baseline", TINY_TRUNK_CFGS["cnn_baseline"],
                        {"hidden_units": 8}, heads=heads)
    x = np.random.default_rng(0).normal(size=(4, WINDOW_LEN, 2)).astype(np.float32)
    out = model(x, training=False)
    # sky_position is a vMF head: it contributes two flat output keys
    # ({head}_mu_raw, {head}_kappa_raw), not a single key matching its name.
    assert "mchirp" in out
    assert "sky_position_mu_raw" in out
    assert "sky_position_kappa_raw" in out
    assert tuple(out["mchirp"].shape) == (4, 1)
    assert tuple(out["sky_position_mu_raw"].shape) == (4, 3)
    assert tuple(out["sky_position_kappa_raw"].shape) == (4, 1)


def test_trainer_with_custom_heads_computes_finite_loss(synthetic_params):
    heads = ["mchirp", "sky_position"]
    base = build_model("cnn_baseline", TINY_TRUNK_CFGS["cnn_baseline"],
                       {"hidden_units": 8}, heads=heads)
    trainer = MultiHeadTrainer(base, {"weighting": "uncertainty"}, heads=heads)
    tr = TargetTransforms(heads=heads).fit(synthetic_params)
    y = tr.transform(synthetic_params[:8])
    x = np.random.default_rng(1).normal(size=(8, WINDOW_LEN, 2)).astype(np.float32)
    loss = float(trainer._total_loss(y, trainer(x, training=False)))
    assert np.isfinite(loss)
    assert set(trainer.log_vars) == set(heads)


def test_per_head_regularization_overrides():
    heads = ["mchirp", "q"]
    head_cfg = {
        "hidden_units": 64,
        "per_head": {"q": {"hidden_units": 8, "dropout": 0.3, "l2": 1e-4}},
    }
    model = build_model("cnn_baseline", TINY_TRUNK_CFGS["cnn_baseline"],
                        head_cfg, heads=heads)
    layer_names = {layer.name for layer in model.layers}
    assert "q_dropout" in layer_names
    assert "mchirp_dropout" not in layer_names
    assert model.get_layer("q_hidden").units == 8
    assert model.get_layer("mchirp_hidden").units == 64

    x = np.random.default_rng(0).normal(size=(4, WINDOW_LEN, 2)).astype(np.float32)
    out = model(x, training=False)
    assert set(out.keys()) == set(heads)


def test_default_heads_are_the_core_four():
    assert tuple(DEFAULT_HEADS) == (
        "mchirp", "merger_time", "snr", "sky_position", "coa_phase"
    )
    # sky_position is SPHERICAL_UNIT_VECTOR (not PERIODIC), and coa_phase IS
    # PERIODIC — verify that at least mchirp, merger_time, and snr are
    # non-periodic (scalar regression heads).
    non_periodic = {"mchirp", "merger_time", "snr", "sky_position"}
    for name in non_periodic:
        assert HEAD_SPECS[name].transform is not TransformKind.PERIODIC
    assert HEAD_SPECS["coa_phase"].transform is TransformKind.PERIODIC
