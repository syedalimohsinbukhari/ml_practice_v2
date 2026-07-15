"""Von Mises-Fisher head for sky position (RA/Dec) on S^2.

On the 2-sphere (ambient dim n=3), the vMF normalizer has a CLOSED FORM --
no modified Bessel function machinery needed, because I_{1/2}(kappa) reduces
to sinh(kappa). This is the one case where vMF is cheap and numerically easy.

    p(x; mu, kappa) = C_3(kappa) * exp(kappa * mu . x)
    C_3(kappa)      = kappa / (4*pi*sinh(kappa))

    NLL = -log(kappa) + log(4*pi) + log(sinh(kappa)) - kappa * (mu . x)

`log(sinh(kappa))` is computed via the stable form
`kappa + log(1 - exp(-2*kappa)) - log(2)` to avoid overflow for large kappa
and underflow issues near kappa=0.
"""
from __future__ import annotations

import keras
from keras import layers, ops


def build_vmf_head(features, name: str = "sky_position", hidden_units: int = 64):
    """Attach a vMF sky-position head to a pooled feature vector.

    Outputs a dict with 'mu_raw' (unnormalized 3-vector, normalize at loss/
    inference time) and 'kappa_raw' (unconstrained; softplus at use time).
    Keeping mu unnormalized in the graph output avoids a non-differentiable
    normalize-then-renormalize round trip; normalization happens once in the
    loss function and once in the inference postprocessing.
    """
    h = layers.Dense(hidden_units, activation="relu", name=f"{name}_hidden")(features)
    h = layers.Dropout(0.2, name=f"{name}_dropout")(h)

    mu_raw = layers.Dense(3, activation="linear", name=f"{name}_mu_raw")(h)
    # softplus + small floor keeps kappa > 0 and away from the kappa->0
    # singularity in log(sinh(kappa)) during early training.
    kappa_raw = layers.Dense(1, activation="linear", name=f"{name}_kappa_raw")(h)

    return {"mu_raw": mu_raw, "kappa_raw": kappa_raw}


def vmf_kappa_from_raw(kappa_raw):
    """Unconstrained -> kappa > 0.1 (floor prevents the kappa->0 log-singularity)."""
    return ops.softplus(kappa_raw) + 0.1


def vmf_nll_loss(y_true_unitvec, mu_raw, kappa_raw):
    """Closed-form vMF negative log-likelihood on S^2.

    y_true_unitvec: (B, 3), already unit-norm target vectors.
    mu_raw:         (B, 3), unnormalized network output.
    kappa_raw:      (B, 1), unconstrained network output.
    """
    mu = mu_raw / ops.maximum(
        ops.sqrt(ops.sum(ops.square(mu_raw), axis=-1, keepdims=True)), 1e-8
    )
    kappa = vmf_kappa_from_raw(kappa_raw)[..., 0]  # (B,)

    cos_sep = ops.sum(mu * y_true_unitvec, axis=-1)  # (B,) = mu . x_true

    # stable log(sinh(kappa)) = kappa + log(1 - exp(-2*kappa)) - log(2)
    log_sinh_kappa = kappa + ops.log(
        ops.maximum(1.0 - ops.exp(-2.0 * kappa), 1e-12)
    ) - ops.log(2.0)

    nll = -ops.log(kappa) + ops.log(4.0 * 3.141592653589793) + log_sinh_kappa \
        - kappa * cos_sep
    return ops.mean(nll)


def vmf_predict(mu_raw, kappa_raw):
    """Postprocess raw head outputs -> (mu unit vector, kappa) for inference.

    kappa is directly interpretable: larger kappa = tighter/more confident
    prediction. Rough angular std (radians, valid for kappa >> 1):
        sigma_approx = 1 / sqrt(kappa)
    """
    import numpy as np

    mu = mu_raw / np.clip(
        np.linalg.norm(mu_raw, axis=-1, keepdims=True), 1e-8, None
    )
    kappa = np.log1p(np.exp(kappa_raw[..., 0])) + 0.1  # numpy softplus + floor
    return mu, kappa
