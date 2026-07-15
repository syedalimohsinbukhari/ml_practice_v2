"""Target transforms: physical parameters <-> normalized training space.

Which transform each head uses is defined once in gwml.heads_spec (the head
registry); this module executes it for whichever heads a run activates:

  LOG_ZSCORE            log then z-score (stats fitted on the training split)
  ZSCORE                z-score (stats fitted on the training split)
  UNIT_AFFINE           fixed documented bounds -> [0, 1]
  PERIODIC              angle -> (sin, cos) pair; inverse via atan2, result in [0, period)
  SPHERICAL_UNIT_VECTOR (ra, dec) -> 3D unit vector on S²; deterministic (no fit needed)

Fitted statistics and the active head list are persisted to JSON next to each
run so evaluation and callbacks can always invert predictions into physical
units.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from gwml.heads_spec import (  # noqa: F401  (PARAM_COLUMNS re-exported)
    DEFAULT_HEADS,
    HEAD_SPECS,
    PARAM_COLUMNS,
    TransformKind,
    resolve_heads,
)
from gwml.data.sky_transform import radec_to_unit_vector, unit_vector_to_radec

# Backward-compatible default head tuple (the core four).
HEAD_ORDER = tuple(DEFAULT_HEADS)

# Legacy head names that were replaced by sky_position (see heads_spec.py).
_LEGACY_SKY_HEADS = {"ra", "declination"}


def _migrate_heads(heads: list[str]) -> list[str]:
    """Replace legacy ``ra`` + ``declination`` with ``sky_position``.

    Old ``transforms.json`` files may contain the pre-migration separate-head
    names.  The z-score statistics (mean / std) for other heads are still
    valid; only the head list needs patching.
    """
    legacy_found = _LEGACY_SKY_HEADS & set(heads)
    if not legacy_found:
        return heads
    warnings.warn(
        f"Replacing legacy heads {sorted(legacy_found)} with 'sky_position' "
        f"in the loaded head list.  Old model checkpoints that were trained "
        f"with separate ra/declination heads are incompatible with the "
        f"current sky_position architecture and must be retrained."
    )
    # Keep the position of the first legacy head, drop both.
    first_idx = min(heads.index(h) for h in legacy_found)
    out = [h for h in heads if h not in _LEGACY_SKY_HEADS]
    out.insert(first_idx, "sky_position")
    return out


def signed_error(head: str, true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """true - pred in physical units, wrap-aware for periodic heads."""
    spec = HEAD_SPECS[head]
    d = np.ravel(true) - np.ravel(pred)
    if spec.transform is TransformKind.PERIODIC:
        p = spec.period
        d = (d + p / 2) % p - p / 2
    return d


def abs_error(head: str, true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.abs(signed_error(head, true, pred))


class TargetTransforms:
    """Fit on training params, then map targets to/from normalized space."""

    def __init__(self, heads=None, stats: dict | None = None):
        self.heads = [s.name for s in resolve_heads(heads or DEFAULT_HEADS)]
        self.stats = stats or {}

    def _needs_stats(self):
        return [
            h for h in self.heads
            if HEAD_SPECS[h].transform
            in (TransformKind.LOG_ZSCORE, TransformKind.ZSCORE)
        ]

    def fit(self, params: np.ndarray) -> "TargetTransforms":
        """Compute z-score statistics from a (N, 10) training params array."""
        for h in self._needs_stats():
            spec = HEAD_SPECS[h]
            col = params[:, spec.column]
            if spec.transform is TransformKind.LOG_ZSCORE:
                col = np.log(col)
            self.stats[h] = {"mean": float(col.mean()), "std": float(col.std())}
        return self

    def _require_fit(self):
        missing = [h for h in self._needs_stats() if h not in self.stats]
        if missing:
            raise RuntimeError(
                f"TargetTransforms used before fit()/from_json(); missing stats "
                f"for {missing}"
            )

    def transform(self, params: np.ndarray) -> dict[str, np.ndarray]:
        """(N, 10) physical params -> dict of (N, dim) float32 targets."""
        out = {}
        for h in self.heads:
            spec = HEAD_SPECS[h]
            if spec.columns is not None:
                raw = params[:, list(spec.columns)]
            else:
                raw = params[:, spec.column]
            out[h] = self.transform_head(h, raw)
        return out

    def transform_head(self, head: str, values: np.ndarray) -> np.ndarray:
        self._require_fit()
        spec = HEAD_SPECS[head]
        v = np.asarray(values, dtype=np.float64)
        if spec.transform is TransformKind.LOG_ZSCORE:
            s = self.stats[head]
            out = (np.log(v) - s["mean"]) / s["std"]
        elif spec.transform is TransformKind.ZSCORE:
            s = self.stats[head]
            out = (v - s["mean"]) / s["std"]
        elif spec.transform is TransformKind.UNIT_AFFINE:
            lo, hi = spec.bounds
            out = (v - lo) / (hi - lo)
        elif spec.transform is TransformKind.PERIODIC:
            theta = v * (2.0 * np.pi / spec.period)
            out = np.stack([np.sin(theta), np.cos(theta)], axis=-1)
        elif spec.transform is TransformKind.SPHERICAL_UNIT_VECTOR:
            # values is (N, 2) ordered as spec.columns = (dec_col, ra_col)
            # radec_to_unit_vector expects (ra, dec) in that order.
            ra = values[:, 1]
            dec = values[:, 0]
            out = radec_to_unit_vector(ra, dec)  # (N, 3)
        else:  # pragma: no cover
            raise KeyError(f"unhandled transform {spec.transform}")
        return out.reshape(len(v), spec.dim).astype(np.float32)

    def inverse_head(self, head: str, values: np.ndarray) -> np.ndarray:
        """Normalized predictions (N, dim) -> physical units (N,)."""
        self._require_fit()
        spec = HEAD_SPECS[head]
        v = np.asarray(values, dtype=np.float64).reshape(-1, spec.dim)
        if spec.transform is TransformKind.LOG_ZSCORE:
            s = self.stats[head]
            return np.exp(v[:, 0] * s["std"] + s["mean"])
        if spec.transform is TransformKind.ZSCORE:
            s = self.stats[head]
            return v[:, 0] * s["std"] + s["mean"]
        if spec.transform is TransformKind.UNIT_AFFINE:
            lo, hi = spec.bounds
            return v[:, 0] * (hi - lo) + lo
        if spec.transform is TransformKind.PERIODIC:
            theta = np.arctan2(v[:, 0], v[:, 1])
            return (theta * spec.period / (2.0 * np.pi)) % spec.period
        if spec.transform is TransformKind.SPHERICAL_UNIT_VECTOR:
            ra, dec = unit_vector_to_radec(v)  # (N, 3) -> (N,), (N,)
            # Return (dec, ra) to match spec.columns = (dec_col, ra_col) order.
            return np.stack([dec, ra], axis=-1)  # (N, 2) in (dec, ra) order
        raise KeyError(f"unhandled transform {spec.transform}")  # pragma: no cover

    def inverse(self, predictions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        # vMF heads (SPHERICAL_UNIT_VECTOR) output two raw tensors
        # ({head}_mu_raw, {head}_kappa_raw) as flat keys. Postprocess them
        # into unit vectors before calling inverse_head.
        out = {}
        for h in self.heads:
            spec = HEAD_SPECS[h]
            if spec.loss == "vmf":
                mu_key = f"{h}_mu_raw"
                if mu_key in predictions:
                    mu_raw = np.asarray(predictions[mu_key], dtype=np.float64)
                    mu = mu_raw / np.clip(
                        np.linalg.norm(mu_raw, axis=-1, keepdims=True), 1e-8, None
                    )
                    out[h] = self.inverse_head(h, mu)
                    continue
            if h in predictions:
                out[h] = self.inverse_head(h, predictions[h])
        return out

    def physical_targets(self, params: np.ndarray) -> dict[str, np.ndarray]:
        """Extract the active heads' raw target columns without any transform."""
        out = {}
        for h in self.heads:
            spec = HEAD_SPECS[h]
            if spec.columns is not None:
                out[h] = params[:, list(spec.columns)].copy()
            else:
                out[h] = params[:, spec.column].copy()
        return out

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"heads": self.heads, "stats": self.stats}, indent=2)
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "TargetTransforms":
        payload = json.loads(Path(path).read_text())
        heads = _migrate_heads(payload["heads"])
        return cls(heads=heads, stats=payload["stats"])
