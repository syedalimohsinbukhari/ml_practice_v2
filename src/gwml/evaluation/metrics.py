"""Per-head evaluation metrics in physical units (wrap-aware for angles)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gwml.data.transforms import TargetTransforms, signed_error


def evaluate_model(
    model,
    strain: np.ndarray,
    params: np.ndarray,
    transforms: TargetTransforms,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Return a DataFrame with MAE and RMSE per active head, physical units."""
    pred = transforms.inverse(model.predict(strain, batch_size=batch_size, verbose=0))
    true = transforms.physical_targets(params)
    rows = []
    for head in transforms.heads:
        res = signed_error(head, true[head], pred[head])
        rows.append({
            "head": head,
            "mae": float(np.mean(np.abs(res))),
            "rmse": float(np.sqrt(np.mean(res**2))),
        })
    return pd.DataFrame(rows).set_index("head")
