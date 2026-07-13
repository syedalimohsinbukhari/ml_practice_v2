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
    specs = resolve_heads([HeadName.MCHIRP, HeadName.RA])
    assert [s.name for s in specs] == ["mchirp", "ra"]


def test_every_spec_column_matches_structure_md():
    from gwml.heads_spec import PARAM_COLUMNS

    for name, spec in HEAD_SPECS.items():
        assert spec.column == PARAM_COLUMNS[name]


@pytest.mark.parametrize("head", ["ra", "coa_phase", "inclination"])
def test_periodic_round_trip_two_pi(head, synthetic_params):
    tr = TargetTransforms(heads=[head]).fit(synthetic_params)
    raw = np.random.default_rng(3).uniform(0, 2 * np.pi, 100)
    encoded = tr.transform_head(head, raw)
    assert encoded.shape == (100, 2)
    assert np.all(np.abs(encoded) <= 1.0)
    recovered = tr.inverse_head(head, encoded)
    np.testing.assert_allclose(recovered, raw % (2 * np.pi), atol=1e-10)


def test_polarization_angle_has_pi_period(synthetic_params):
    tr = TargetTransforms(heads=["polarization_angle"]).fit(synthetic_params)
    raw = np.array([0.3, 0.3 + np.pi, 0.3 + 2 * np.pi])
    encoded = tr.transform_head("polarization_angle", raw)
    # psi and psi + pi produce identical strain, so identical encodings.
    np.testing.assert_allclose(encoded[0], encoded[1], atol=1e-12)
    np.testing.assert_allclose(encoded[0], encoded[2], atol=1e-12)
    recovered = tr.inverse_head("polarization_angle", encoded)
    np.testing.assert_allclose(recovered, raw % np.pi, atol=1e-10)


def test_abs_error_is_wrap_aware():
    err = abs_error("ra", np.array([0.05]), np.array([2 * np.pi - 0.05]))
    np.testing.assert_allclose(err, [0.1], atol=1e-10)
    # Non-periodic heads keep the plain difference.
    np.testing.assert_allclose(abs_error("snr", np.array([9.0]), np.array([12.0])),
                               [3.0])


def test_model_builds_with_custom_heads_and_dims():
    heads = ["mchirp", "ra", "declination"]
    model = build_model("cnn_baseline", TINY_TRUNK_CFGS["cnn_baseline"],
                        {"hidden_units": 8}, heads=heads)
    x = np.random.default_rng(0).normal(size=(4, WINDOW_LEN, 2)).astype(np.float32)
    out = model(x, training=False)
    assert set(out.keys()) == set(heads)
    assert tuple(out["mchirp"].shape) == (4, 1)
    assert tuple(out["ra"].shape) == (4, 2)        # sin/cos pair
    assert tuple(out["declination"].shape) == (4, 1)


def test_trainer_with_custom_heads_computes_finite_loss(synthetic_params):
    heads = ["mchirp", "ra", "declination"]
    base = build_model("cnn_baseline", TINY_TRUNK_CFGS["cnn_baseline"],
                       {"hidden_units": 8}, heads=heads)
    trainer = MultiHeadTrainer(base, {"weighting": "uncertainty"}, heads=heads)
    tr = TargetTransforms(heads=heads).fit(synthetic_params)
    y = tr.transform(synthetic_params[:8])
    x = np.random.default_rng(1).normal(size=(8, WINDOW_LEN, 2)).astype(np.float32)
    loss = float(trainer._total_loss(y, trainer(x, training=False)))
    assert np.isfinite(loss)
    assert set(trainer.log_vars) == set(heads)


def test_default_heads_are_the_core_four():
    assert tuple(DEFAULT_HEADS) == ("mchirp", "q", "merger_time", "snr")
    for name in DEFAULT_HEADS:
        assert HEAD_SPECS[name].transform is not TransformKind.PERIODIC
