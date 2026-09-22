"""Shared name -> physical-quantity label convention for combination-space plots.

Single source of truth for how an internal combo name (``"combo_A"``, ``"combo_B"``, ...)
is rendered as a LaTeX label in figures and log messages across this experiment directory.
Import this rather than re-declaring the mapping locally, so a renamed/added combo only
needs to change in one place.

Usage::

    from combo_labels import COMBO_LABELS
    ax.set_title(COMBO_LABELS["combo_A"])
"""

from __future__ import annotations

COMBO_LABELS: dict[str, str] = {
    "combo_A": r"$2\phi_c+2\psi$",
    "combo_B": r"$2\phi_c-2\psi$",
}
