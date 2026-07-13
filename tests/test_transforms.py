"""Transform round-trips and range guarantees."""

import json

import numpy as np
import pytest

from gwml.data.transforms import HEAD_ORDER, PARAM_COLUMNS, TargetTransforms


@pytest.mark.parametrize("head", HEAD_ORDER)
def test_round_trip(head, synthetic_params, fitted_transforms):
    raw = synthetic_params[:, PARAM_COLUMNS[head]]
    normalized = fitted_transforms.transform_head(head, raw)
    recovered = fitted_transforms.inverse_head(head, normalized)
    # transform_head outputs float32 (training targets), so round trips are
    # exact only to single precision.
    np.testing.assert_allclose(recovered, raw, rtol=1e-6)


def test_bounded_heads_land_in_unit_interval(synthetic_params, fitted_transforms):
    targets = fitted_transforms.transform(synthetic_params)
    for head in ("q", "merger_time"):
        assert targets[head].min() >= 0.0
        assert targets[head].max() <= 1.0


def test_transform_output_shapes_and_dtype(synthetic_params, fitted_transforms):
    targets = fitted_transforms.transform(synthetic_params)
    n = len(synthetic_params)
    for head in HEAD_ORDER:
        assert targets[head].shape == (n, 1)
        assert targets[head].dtype == np.float32


def test_zscore_heads_are_standardized(synthetic_params, fitted_transforms):
    targets = fitted_transforms.transform(synthetic_params)
    for head in ("mchirp", "snr"):
        assert abs(float(targets[head].mean())) < 0.05
        assert abs(float(targets[head].std()) - 1.0) < 0.05


def test_json_round_trip(tmp_path, synthetic_params, fitted_transforms):
    path = tmp_path / "transforms.json"
    fitted_transforms.to_json(path)
    reloaded = TargetTransforms.from_json(path)
    payload = json.loads(path.read_text())
    assert reloaded.stats == payload["stats"]
    assert reloaded.heads == payload["heads"] == fitted_transforms.heads
    raw = synthetic_params[:, PARAM_COLUMNS["mchirp"]]
    np.testing.assert_allclose(
        reloaded.transform_head("mchirp", raw),
        fitted_transforms.transform_head("mchirp", raw),
    )


def test_unfitted_transforms_raise(synthetic_params):
    with pytest.raises(RuntimeError):
        TargetTransforms().transform(synthetic_params)
