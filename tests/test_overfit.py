"""Overfit-one-batch: every trunk must be able to memorize 32 samples.

If a trunk can't drive its loss well below the constant-mean baseline on 32
fixed samples, the bug is in the pipeline/loss/normalization, not the
architecture. Marked slow — run with plain `pytest` on the lab machine.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from conftest import DATA_PATH, TINY_TRUNK_CFGS, requires_data
from gwml.data.loader import load_arrays
from gwml.data.transforms import TargetTransforms
from gwml.models import build_model
from gwml.training.losses import MultiHeadTrainer

pytestmark = [pytest.mark.slow, requires_data]

N_SAMPLES = 32
EPOCHS = 200


@pytest.fixture(scope="module")
def tiny_batch():
    strain, params = load_arrays(DATA_PATH, "training", max_samples=N_SAMPLES)
    transforms = TargetTransforms().fit(params)
    return strain, transforms.transform(params)


@pytest.mark.parametrize("name", sorted(TINY_TRUNK_CFGS))
def test_trunk_overfits_one_batch(name, tiny_batch):
    strain, targets = tiny_batch
    keras.utils.set_random_seed(0)
    base = build_model(name, TINY_TRUNK_CFGS[name], {"hidden_units": 32})
    # Fixed weighting keeps the loss comparable across epochs (no moving s_h).
    trainer = MultiHeadTrainer(base, {"weighting": "fixed"})
    trainer.compile(optimizer=keras.optimizers.Adam(1e-3))

    ds = tf.data.Dataset.from_tensor_slices((strain, targets)).batch(N_SAMPLES)
    history = trainer.fit(ds, epochs=EPOCHS, verbose=0)
    losses = history.history["loss"]

    assert np.isfinite(losses[-1])
    assert losses[-1] < 0.1 * losses[0], (
        f"{name}: loss only went {losses[0]:.4f} -> {losses[-1]:.4f} "
        f"in {EPOCHS} epochs on {N_SAMPLES} samples"
    )
