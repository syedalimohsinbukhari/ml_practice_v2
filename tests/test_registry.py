"""Every registered trunk builds and honors the model contract."""

import numpy as np
import pytest

from conftest import TINY_TRUNK_CFGS
from gwml.data.transforms import HEAD_ORDER
from gwml.models import available, build_model, build_trunk

WINDOW_LEN = 4096


def test_all_expected_trunks_registered():
    assert available() == sorted(TINY_TRUNK_CFGS)


def test_unknown_trunk_raises():
    with pytest.raises(KeyError):
        build_trunk("nope")


@pytest.mark.parametrize("name", sorted(TINY_TRUNK_CFGS))
def test_trunk_builds_and_outputs_named_heads(name):
    model = build_model(name, TINY_TRUNK_CFGS[name], {"hidden_units": 8})
    x = np.random.default_rng(0).normal(size=(4, WINDOW_LEN, 2)).astype(np.float32)
    out = model(x, training=False)
    assert set(out.keys()) == set(HEAD_ORDER)
    for head in HEAD_ORDER:
        assert tuple(out[head].shape) == (4, 1)


@pytest.mark.parametrize("name", sorted(TINY_TRUNK_CFGS))
def test_trunk_starts_with_input_batchnorm(name):
    model = build_model(name, TINY_TRUNK_CFGS[name], {"hidden_units": 8})
    assert model.get_layer("input_bn") is not None


def test_bounded_heads_respect_sigmoid_range():
    model = build_model("cnn_baseline", TINY_TRUNK_CFGS["cnn_baseline"],
                        {"hidden_units": 8, "bounded": True})
    # Extreme inputs must still give (0, 1) predictions on the bounded heads.
    x = np.full((4, WINDOW_LEN, 2), 1e3, dtype=np.float32)
    out = model(x, training=False)
    for head in ("q", "merger_time"):
        vals = np.asarray(out[head])
        assert vals.min() >= 0.0 and vals.max() <= 1.0
