"""Standalone sanity check for the vMF sky head.

Fits toy "features -> sky position" mapping: 3 noisy input features that
linearly determine a true unit vector, mixed with 5 pure-noise features (to
mimic a trunk producing a mostly-irrelevant feature vector). Confirms:
  1. The loss decreases and trains stably (no NaNs from log(sinh(kappa))).
  2. Predicted mu tracks true direction (mean angular error shrinks).
  3. kappa is higher (more confident) for easier examples, lower for the
     noisier ones -- i.e. it behaves like a real uncertainty, not a constant.
"""
import numpy as np
import keras
from keras import layers

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from sky_transform import radec_to_unit_vector, unit_vector_to_radec, angular_separation
from vmf_head import build_vmf_head, vmf_nll_loss, vmf_predict

rng = np.random.default_rng(0)
N = 4000

# synthetic true sky positions
ra_true = rng.uniform(0, 2 * np.pi, N)
dec_true = np.arcsin(rng.uniform(-1, 1, N))  # uniform on sphere
target_vec = radec_to_unit_vector(ra_true, dec_true).astype("float32")

# "features": true position information plus noise, at varying difficulty
# per-example noise level so kappa should track it
noise_level = rng.uniform(0.05, 0.6, size=(N, 1)).astype("float32")
feat_signal = target_vec + rng.normal(0, 1, size=(N, 3)).astype("float32") * noise_level
feat_junk = rng.normal(0, 1, size=(N, 5)).astype("float32")
features_np = np.concatenate([feat_signal, feat_junk], axis=-1).astype("float32")

split = int(N * 0.8)
X_train, X_val = features_np[:split], features_np[split:]
y_train, y_val = target_vec[:split], target_vec[split:]
noise_val = noise_level[split:, 0]

inputs = keras.Input(shape=(8,))
h = layers.Dense(64, activation="relu")(inputs)
h = layers.Dense(64, activation="relu")(h)
head = build_vmf_head(h, hidden_units=32)
model = keras.Model(inputs, [head["mu_raw"], head["kappa_raw"]])

optimizer = keras.optimizers.Adam(5e-3)

X_train_t = keras.ops.convert_to_tensor(X_train)
y_train_t = keras.ops.convert_to_tensor(y_train)
X_val_t = keras.ops.convert_to_tensor(X_val)
y_val_t = keras.ops.convert_to_tensor(y_val)


import tensorflow as tf

for epoch in range(250):
    with tf.GradientTape() as tape:
        mu_raw, kappa_raw = model(X_train_t, training=True)
        loss = vmf_nll_loss(y_train_t, mu_raw, kappa_raw)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if epoch % 25 == 0 or epoch == 249:
        mu_raw_v, kappa_raw_v = model(X_val_t, training=False)
        val_loss = vmf_nll_loss(y_val_t, mu_raw_v, kappa_raw_v).numpy()
        mu_pred, kappa_pred = vmf_predict(mu_raw_v.numpy(), kappa_raw_v.numpy())
        ang_err = angular_separation(y_val, mu_pred)
        print(
            f"epoch {epoch:3d}  train_loss={loss.numpy():7.3f}  "
            f"val_loss={val_loss:7.3f}  "
            f"mean_ang_err_deg={np.degrees(ang_err.mean()):6.2f}  "
            f"mean_kappa={kappa_pred.mean():6.2f}"
        )

# final check: does kappa track per-example noise level? (should be negative corr)
mu_raw_v, kappa_raw_v = model(X_val_t, training=False)
_, kappa_pred = vmf_predict(mu_raw_v.numpy(), kappa_raw_v.numpy())
corr = np.corrcoef(kappa_pred, noise_val)[0, 1]
print(f"\ncorr(kappa, noise_level) = {corr:.3f}  (expect clearly negative)")
