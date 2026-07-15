"""Single source of truth for every possible output head.

Each head binds a physical parameter to its params-column(s), target transform,
output dimension, activation, and loss. Experiment YAMLs choose *which* heads
are active (``model.heads``); they cannot override how a head is transformed
or which loss it uses — that binding lives here precisely so a periodic
parameter can never be silently paired with a plain loss on raw angles.

Periodic parameters are represented as (sin, cos) pairs of the angle scaled to
the head's period, so the standard Huber loss on the pair is correct — no
wraparound discontinuity ever reaches the loss. polarization_angle uses period
pi because the strain is invariant under psi -> psi + pi.

Sky position (ra, declination) is handled jointly via
``TransformKind.SPHERICAL_UNIT_VECTOR``: the two angles are converted to a
3D unit vector on S² and trained with a closed-form von Mises-Fisher NLL.
This replaces the old per-angle independent heads (``ra`` PERIODIC,
``declination`` UNIT_AFFINE) which threw away the 2D correlation structure
on the sphere.

injection_time is deliberately absent: absolute GPS time is not learnable from
a 2 s whitened window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class HeadName(str, Enum):
    MCHIRP = "mchirp"
    Q = "q"
    INCLINATION = "inclination"
    COA_PHASE = "coa_phase"
    POLARIZATION_ANGLE = "polarization_angle"
    SKY_POSITION = "sky_position"
    MERGER_TIME = "merger_time"
    SNR = "snr"


class TransformKind(str, Enum):
    LOG_ZSCORE = "log_zscore"      # log then z-score; stats fitted on training
    ZSCORE = "zscore"              # z-score; stats fitted on training
    UNIT_AFFINE = "unit_affine"    # fixed bounds -> [0, 1]
    PERIODIC = "periodic"          # angle -> (sin, cos) of 2*pi*value/period
    SPHERICAL_UNIT_VECTOR = "spherical_unit_vector"  # (ra, dec) -> unit vec on S²


@dataclass(frozen=True)
class HeadSpec:
    name: str
    column: int | None = None          # index into the (N, 10) params array
    transform: TransformKind | None = None
    label: str | None = None            # plot axis label
    loss: str = "huber"                 # key into MultiHeadTrainer's loss table
    dim: int = 1                        # output width (2 for sin/cos pairs)
    activation: str = "linear"          # downgraded to linear when head_cfg.bounded=False
    bounds: tuple[float, float] | None = None   # UNIT_AFFINE only
    period: float | None = None                  # PERIODIC only
    columns: tuple[int, ...] | None = None  # multi-column heads (e.g. sky_position)

    def __post_init__(self):
        if (self.column is None) == (self.columns is None):
            raise ValueError(
                f"HeadSpec {self.name!r}: exactly one of 'column' or 'columns' "
                f"must be set, got column={self.column!r}, columns={self.columns!r}"
            )


# Column indices into the (N, 10) params dataset, per structure.md.
PARAM_COLUMNS = {
    "mchirp": 0,
    "q": 1,
    "inclination": 2,
    "coa_phase": 3,
    "polarization_angle": 4,
    "declination": 5,
    "ra": 6,
    "injection_time": 7,
    "merger_time": 8,
    "snr": 9,
}

_TWO_PI = 2.0 * math.pi

HEAD_SPECS: dict[str, HeadSpec] = {
    s.name: s
    for s in (
        HeadSpec("mchirp", 0, TransformKind.LOG_ZSCORE,
                 label=r"$\mathcal{M}_c\ [M_\odot]$"),
        # Bounds padded beyond the true data range (0.2027-0.99998, see
        # q_head_action_plan.md Phase 1 step 1) so q's top decile — ~7.6% of
        # training samples sit in true-q>=0.95 — doesn't map to t=1.0, the
        # sigmoid's exact saturating asymptote where gradients vanish.
        HeadSpec("q", 1, TransformKind.UNIT_AFFINE, label=r"$q$",
                 activation="sigmoid", bounds=(0.15, 1.05)),
        HeadSpec("inclination", 2, TransformKind.PERIODIC, label=r"$\iota$ [rad]",
                 dim=2, activation="tanh", period=_TWO_PI),
        HeadSpec("coa_phase", 3, TransformKind.PERIODIC, label=r"$\phi_c$ [rad]",
                 dim=2, activation="tanh", period=_TWO_PI),
        HeadSpec("polarization_angle", 4, TransformKind.PERIODIC,
                 label=r"$\psi$ [rad]", dim=2, activation="tanh",
                 period=math.pi),
        HeadSpec("sky_position",
                 transform=TransformKind.SPHERICAL_UNIT_VECTOR,
                 label="Sky position", dim=3, activation="linear",
                 loss="vmf",
                 columns=(PARAM_COLUMNS["declination"], PARAM_COLUMNS["ra"])),
        HeadSpec("merger_time", 8, TransformKind.UNIT_AFFINE,
                 label=r"$t_{merger}$ [s]", activation="sigmoid",
                 bounds=(1.6, 1.8)),
        HeadSpec("snr", 9, TransformKind.ZSCORE, label="SNR"),
    )
}

DEFAULT_HEADS = ("mchirp", "merger_time", "snr", "sky_position", "coa_phase")


def resolve_heads(names) -> list[HeadSpec]:
    """Validate head names (str or HeadName) and return their specs, in order."""
    names = [str(getattr(n, "value", n)) for n in names]
    if not names:
        raise ValueError("at least one output head is required")
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate heads in {names}")
    unknown = [n for n in names if n not in HEAD_SPECS]
    if unknown:
        raise ValueError(
            f"unknown heads {unknown}; valid heads: {sorted(HEAD_SPECS)}"
        )
    return [HEAD_SPECS[n] for n in names]
