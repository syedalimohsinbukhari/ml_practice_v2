"""RA/Dec <-> unit-vector conversion for the joint sky-position head.

Replaces the two independent `ra` (PERIODIC) and `declination` (UNIT_AFFINE)
heads with a single `sky_position` head that outputs a 3D unit vector on S^2.
This is necessary because sky localization is a genuinely 2D, correlated
quantity -- fitting RA and Dec as independent 1D targets throws away the
ring/mirror degeneracy structure that only exists jointly.
"""
from __future__ import annotations

import numpy as np


def radec_to_unit_vector(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """(ra, dec) in radians -> (N, 3) unit vectors.

    ra in [0, 2*pi), dec in [-pi/2, pi/2].
    """
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.stack([x, y, z], axis=-1)


def unit_vector_to_radec(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N, 3) unit vectors -> (ra, dec) in radians. Normalizes defensively."""
    v = v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    dec = np.arcsin(np.clip(z, -1.0, 1.0))
    ra = np.mod(np.arctan2(y, x), 2 * np.pi)
    return ra, dec


def angular_separation(v_true: np.ndarray, v_pred: np.ndarray) -> np.ndarray:
    """Great-circle angular error in radians between two sets of unit vectors.

    This is the metric to log/plot for the sky head -- NOT separate MAE(ra)
    and MAE(dec). A joint angular error is the physically meaningful
    quantity; per-axis errors near the poles are misleading (RA is degenerate
    at dec=+/-90deg) and don't reflect the actual ring-shaped uncertainty.
    """
    v_true = v_true / np.clip(np.linalg.norm(v_true, axis=-1, keepdims=True), 1e-12, None)
    v_pred = v_pred / np.clip(np.linalg.norm(v_pred, axis=-1, keepdims=True), 1e-12, None)
    cos_sep = np.clip(np.sum(v_true * v_pred, axis=-1), -1.0, 1.0)
    return np.arccos(cos_sep)
