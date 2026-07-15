"""Per-head evaluation metrics in physical units (wrap-aware for angles).

``evaluate_model`` is the single entry-point: it inverts raw model predictions
via ``TargetTransforms.inverse`` (which handles vMF raw-output postprocessing
for the ``sky_position`` head), extracts physical ground-truth columns, and
computes MAE / RMSE per head.

Scalar and periodic heads use ``signed_error`` for wrap-aware per-element
differences.  The ``sky_position`` head (SPHERICAL_UNIT_VECTOR) is handled
specially: instead of naively flattening its (N, 2) = (dec, ra) output, which
would mix two physically different quantities into one meaningless metric, the
function recomputes unit vectors and reports mean / RMS great-circle angular
separation (degrees) in the ``mae`` / ``rmse`` columns, plus per-component
``dec_mae_deg`` / ``ra_mae_deg`` columns for debugging.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gwml.data.sky_transform import angular_separation, radec_to_unit_vector
from gwml.data.transforms import TargetTransforms, signed_error
from gwml.heads_spec import HEAD_SPECS, TransformKind


def evaluate_model(
    model,
    strain: np.ndarray,
    params: np.ndarray,
    transforms: TargetTransforms,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Return a DataFrame with MAE and RMSE per active head, physical units.

    sky_position is a multi-column head (dec, ra) -- naively flattening its
    (N, 2) true/pred arrays via signed_error/np.ravel interleaves two
    physically different quantities (dec in [-pi/2, pi/2], ra in [0, 2*pi))
    into one series, producing a meaningless mixed MAE. It's reported
    instead as mean/median great-circle angular separation in degrees,
    under the ``mae``/`rmse`` columns for a consistent DataFrame shape
    (mae = mean angular separation, rmse = sqrt(mean(angular_sep**2))), plus
    dedicated ``dec_mae_deg``/``ra_mae_deg`` columns for component-level
    debugging.
    """
    pred = transforms.inverse(model.predict(strain, batch_size=batch_size, verbose=0))
    true = transforms.physical_targets(params)
    rows = []
    for head in transforms.heads:
        spec = HEAD_SPECS[head]
        if spec.transform is TransformKind.SPHERICAL_UNIT_VECTOR:
            dec_true, ra_true = true[head][:, 0], true[head][:, 1]
            dec_pred, ra_pred = pred[head][:, 0], pred[head][:, 1]
            v_true = radec_to_unit_vector(ra_true, dec_true)
            v_pred = radec_to_unit_vector(ra_pred, dec_pred)
            ang_sep_deg = np.degrees(angular_separation(v_true, v_pred))
            dec_err_deg = np.degrees(np.abs(dec_true - dec_pred))
            ra_err_deg = np.degrees(
                np.abs((ra_true - ra_pred + np.pi) % (2 * np.pi) - np.pi)
            )
            rows.append({
                "head": head,
                "mae": float(np.mean(ang_sep_deg)),
                "rmse": float(np.sqrt(np.mean(ang_sep_deg ** 2))),
                "dec_mae_deg": float(np.mean(dec_err_deg)),
                "ra_mae_deg": float(np.mean(ra_err_deg)),
            })
            continue
        res = signed_error(head, true[head], pred[head])
        rows.append({
            "head": head,
            "mae": float(np.mean(np.abs(res))),
            "rmse": float(np.sqrt(np.mean(res**2))),
        })
    return pd.DataFrame(rows).set_index("head")