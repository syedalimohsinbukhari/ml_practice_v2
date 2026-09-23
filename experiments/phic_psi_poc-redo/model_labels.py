r"""Shared name -> display label convention for model/trunk identifiers across plots and logs.

Single source of truth for how an internal model name (``"poc_a"``, ``"poc_b"``, ``"tcn"``, ...) is rendered as a
figure title/legend label across this experiment directory.
Import this rather than re-declaring the mapping locally -- it used to be hand-duplicated in
``plot_certified_memorization.py`` and ``diagnostic_checks.py``, which is exactly the drift this module exists to
prevent (mirrors ``combo_labels.py`` for the combination-space names).
The same canonical keys (``poc_a``, ``poc_b``, ``tcn``, ``cnn_attention``, ``cnn_baseline``, ``inception_time``,
``resnet1d``) are what the paper's LaTeX macros (``\pocA``, ``\pocB``, ``\Tcn``, ``\cnnAttn``, ``\cnnBaseline``,
``\inceptionTime``, ``\resnetOneD`` in ``paper/000_preamble.tex``) spell out, so a model referenced in prose and a
model plotted in a figure always name the same thing the same way -- notably ``cnn_attention``, never the
``cnn_attn`` abbreviation that had crept into some table cells.

Usage::

    from model_labels import MODEL_LABELS
    ax.set_title(MODEL_LABELS[model])
"""

from __future__ import annotations

MODEL_LABELS: dict[str, str] = {
    "poc_a": "POC_A",
    "poc_b": "POC_B",
    "tcn": "TCN",
    "cnn_baseline": r"CNN$_\text{baseline}$",
    "cnn_attention": r"CNN$_\text{attention}$",
    "inception_time": "InceptionTime",
    "resnet1d": "ResNet1D",
}
