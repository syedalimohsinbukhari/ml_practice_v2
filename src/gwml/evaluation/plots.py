"""Plotting helpers shared by callbacks and the evaluate script.

Head labels come from the head registry; every function takes the active head
list so plots adapt to whatever heads a run trains.

Panel layout
------------
Most heads produce a single scatter panel.  ``sky_position``
(SPHERICAL_UNIT_VECTOR) is a multi-column head stored as (N, 2) = (dec, ra);
``_panel_specs`` expands it into two panels (one per angle component) so that
dec values in [-pi/2, pi/2] and ra values in [0, 2*pi) aren't interleaved
into a single misleading bimodal scatter.  The sky_position panels are
annotated with the physically meaningful mean great-circle angular separation
(in degrees) rather than a naive per-component MAE.

Periodic heads (coa_phase, inclination, polarization_angle, and
sky_position's ra component) have their predicted values shifted by a
multiple of the period so wrap-boundary points render near the diagonal
instead of at opposite corners.  That shift is purely cosmetic — same value
modulo the period — and does not change the wrap-aware error.

Residual-vs-param binning
--------------------------
``residuals_vs_param`` bins absolute residuals by an external parameter
(usually SNR or mchirp).  For multi-column heads (sky_position), per-sample
L2-norm reduction is applied before binning so that the (N, 2) error maps to
an (N,) array matching the (N,) parameter axis; otherwise ``np.ravel`` would
blow up the error array to 2N and break boolean-index alignment.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gwml.data.transforms import abs_error
from gwml.data.sky_transform import angular_separation, radec_to_unit_vector
from gwml.heads_spec import HEAD_SPECS, TransformKind


def _grid(n: int):
    cols = 2 if n <= 4 else 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.0 * rows),
                             squeeze=False)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def _wrap_residual(t: np.ndarray, p: np.ndarray, period: float) -> np.ndarray:
    """true - pred wrapped into [-period/2, period/2]. Same formula as
    ``gwml.data.transforms.signed_error``'s PERIODIC branch, but usable for
    a sub-component (e.g. sky_position's ra column) whose *head* isn't
    itself PERIODIC-typed."""
    d = t - p
    return (d + period / 2.0) % period - period / 2.0


def _panel_specs(heads: list[str]):
    """Expand the head list into individual scatter panels.

    Most heads produce one panel. ``sky_position`` (SPHERICAL_UNIT_VECTOR,
    stored as (N, 2) = (dec, ra) per transforms.inverse_head) produces two
    panels, one per angle -- np.ravel-ing a (N, 2) array of two physically
    different quantities into one flat series is the bug this replaces
    (dec values in [-pi/2, pi/2] and ra values in [0, 2*pi) interleaved on
    one axis look like bimodal clustering at both extremes, when really
    it's just two different unrelated quantities overlaid).
    """
    specs = []
    for head in heads:
        spec = HEAD_SPECS[head]
        if spec.transform is TransformKind.SPHERICAL_UNIT_VECTOR:
            specs.append((head, "dec"))
            specs.append((head, "ra"))
        else:
            specs.append((head, None))
    return specs


def scatter_grid(
    true: dict[str, np.ndarray],
    pred: dict[str, np.ndarray],
    heads: list[str],
    path: str | Path,
    title: str | None = None,
) -> None:
    """Pred-vs-true scatter per head (physical units), y=x diagonal, MAE.

    Multi-column heads (currently only sky_position) get one panel per
    component instead of being flattened together, and are annotated with
    the physically meaningful mean angular separation (degrees) rather than
    a naive per-component MAE. Periodic heads (coa_phase, inclination,
    polarization_angle, and sky_position's ra component) have their
    predicted values shifted by a multiple of the period so wrap-boundary
    points (true near 0, pred near period, or vice versa) render near the
    diagonal instead of at opposite corners of the plot -- that shift is
    purely cosmetic (same value modulo the period) and does not change the
    wrap-aware MAE already used elsewhere.
    """
    specs = _panel_specs(heads)
    fig, axes = _grid(len(specs))
    for ax, (head, sub) in zip(axes, specs):
        spec = HEAD_SPECS[head]
        t_full, p_full = true[head], pred[head]

        if sub is not None:
            # sky_position: t_full/p_full are (N, 2) = (dec, ra)
            dec_true, ra_true = t_full[:, 0], t_full[:, 1]
            dec_pred, ra_pred = p_full[:, 0], p_full[:, 1]
            v_true = radec_to_unit_vector(ra_true, dec_true)
            v_pred = radec_to_unit_vector(ra_pred, dec_pred)
            ang_sep_deg = np.degrees(angular_separation(v_true, v_pred))
            if sub == "dec":
                t, p, period, axis_label = dec_true, dec_pred, None, "Dec [rad]"
            else:
                t, p, period, axis_label = ra_true, ra_pred, 2.0 * np.pi, "RA [rad]"
            metric_str = f"mean ang. sep = {ang_sep_deg.mean():.2f} deg"
        else:
            t, p = np.ravel(t_full), np.ravel(p_full)
            axis_label = spec.label
            period = spec.period if spec.transform is TransformKind.PERIODIC else None
            mae = float(np.mean(abs_error(head, t, p)))
            metric_str = f"MAE = {mae:.4g}"

        if period is not None:
            # Shift p into the branch nearest t so wrap-boundary points sit
            # near the diagonal visually, instead of at opposite corners.
            p_display = t - _wrap_residual(t, p, period)
        else:
            p_display = p

        lo = float(min(t.min(), p_display.min()))
        hi = float(max(t.max(), p_display.max()))
        ax.plot([lo, hi], [lo, hi], color="gray", lw=1, zorder=1)
        ax.scatter(t, p_display, s=4, alpha=0.35, zorder=2)
        title_prefix = f"{spec.label} ({sub})" if sub is not None else axis_label
        ax.set_title(f"{title_prefix}   {metric_str}")
        ax.set_xlabel("true")
        ax.set_ylabel("predicted" + (" (unwrapped)" if period is not None else ""))
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def residuals_vs_param(
    true: dict[str, np.ndarray],
    pred: dict[str, np.ndarray],
    param_values: np.ndarray,
    param_label: str,
    heads: list[str],
    path: str | Path,
    n_bins: int = 8,
) -> None:
    """|residual| binned by an arbitrary true param per head.

    Used for SNR (errors should shrink with SNR) and, e.g., mchirp — a
    systematic bias toward the population mean in a low-signal regime reads
    differently from uncorrelated noise (memorization); see
    q_head_action_plan.md Phase 1 step 6.
    """
    fig, axes = _grid(len(heads))
    edges = np.linspace(param_values.min(), param_values.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for ax, head in zip(axes, heads):
        t, p = true[head], pred[head]
        # Multi-column heads (e.g. sky_position) need per-sample error
        # reduction before binning, otherwise np.ravel blows up the
        # array length to k*N while param_values is only N.
        if t.ndim == 2 and t.shape[1] > 1:
            abs_res = np.linalg.norm(t - p, axis=1)  # (N,)
        else:
            abs_res = abs_error(head, t, p)
        binned = [
            abs_res[(param_values >= edges[i]) & (param_values < edges[i + 1])].mean()
            if np.any((param_values >= edges[i]) & (param_values < edges[i + 1]))
            else np.nan
            for i in range(n_bins)
        ]
        ax.plot(centers, binned, marker="o")
        ax.set_title(HEAD_SPECS[head].label)
        ax.set_xlabel(f"true {param_label}")
        ax.set_ylabel("mean |residual|")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def residuals_vs_snr(
    true: dict[str, np.ndarray],
    pred: dict[str, np.ndarray],
    snr: np.ndarray,
    heads: list[str],
    path: str | Path,
    n_bins: int = 8,
) -> None:
    """|residual| binned by true SNR per head — errors should shrink with SNR."""
    residuals_vs_param(true, pred, snr, "SNR", heads, path, n_bins=n_bins)


def sigmoid_logit_hist(
    train_raw_pred: dict[str, np.ndarray],
    train_true_physical: dict[str, np.ndarray],
    val_raw_pred: dict[str, np.ndarray],
    val_true_physical: dict[str, np.ndarray],
    heads: list[str],
    path: str | Path,
    n_bins: int = 40,
) -> None:
    """Pre-sigmoid logit histograms, train vs val, split by true-value tercile.

    Recovers the logit algebraically from the sigmoid output —
    logit = ln(p/(1-p)) on the model's raw (pre-inverse-transform) [0,1]
    prediction — no architecture change needed. Train logits going extreme
    while val logits for the same true-value tercile stay compressed is the
    saturation-driven-overfitting signature from q_head_action_plan.md
    Phase 1 step 3. Only applies to UNIT_AFFINE heads with a sigmoid
    activation (e.g. q, merger_time); heads is filtered down to those.
    """
    sig_heads = [
        h for h in heads
        if HEAD_SPECS[h].transform is TransformKind.UNIT_AFFINE
        and HEAD_SPECS[h].activation == "sigmoid"
    ]
    if not sig_heads:
        return

    def _logit(raw: np.ndarray) -> np.ndarray:
        p = np.clip(np.ravel(raw), 1e-6, 1.0 - 1e-6)
        return np.log(p / (1.0 - p))

    fig, axes = plt.subplots(len(sig_heads), 2,
                             figsize=(9.0, 4.0 * len(sig_heads)), squeeze=False)
    for row, head in enumerate(sig_heads):
        terciles = np.quantile(np.ravel(train_true_physical[head]), [1 / 3, 2 / 3])
        splits = [
            ("train", train_raw_pred, train_true_physical),
            ("val", val_raw_pred, val_true_physical),
        ]
        for col, (name, raw_pred, true_physical) in enumerate(splits):
            ax = axes[row][col]
            logit = _logit(raw_pred[head])
            tercile_idx = np.digitize(np.ravel(true_physical[head]), terciles)
            for i, label in enumerate(["low", "mid", "high"]):
                vals = logit[tercile_idx == i]
                if len(vals):
                    ax.hist(vals, bins=n_bins, alpha=0.5, label=f"true {label}")
            ax.set_title(f"{HEAD_SPECS[head].label} — {name}")
            ax.set_xlabel("logit = ln(p/(1-p))")
            if col == 0:
                ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)