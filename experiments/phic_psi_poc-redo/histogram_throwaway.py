"""THROWAWAY: re-plot the per-model prediction histograms (default: psi) with display names, typeset labels, no title.

The histograms need the 5,000 raw validation predictions per model, which ``analyse_predictions.py`` never saved
(only summary CSVs), so they cannot be rebuilt from ``analysis_output/`` alone.
Two stages, split so that only the first needs the GPU machine:

  dump  (GPU machine) -- reuses ``analyse_predictions.analyse_run`` to run inference once and writes
        ``histogram_throwaway_output/predictions_<head>.npz`` (true/pred arrays plus circular r, angular MAE, peaks).
  plot  (CPU, any machine) -- reads that .npz and draws the figure; restyle freely without touching the GPU again.

With no stage given, plots if the .npz exists, otherwise dumps then plots.
Delete this file once the figure is folded back into ``analyse_predictions.py``.

Usage:
    python experiments/phic_psi_poc-redo/histogram_throwaway.py [dump|plot] [head]
    head: polarization_angle (default) | coa_phase | inclination
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = HERE / "histogram_throwaway_output"

# Display names, matching model_labels.py (POC_A, POC_B, TCN, CNN_attention).
MODEL_NAMES = {
    "poc_a": r"POC$_\mathrm{A}$",
    "poc_b": r"POC$_\mathrm{B}$",
    "tcn": "TCN",
    "cnn_attention": r"CNN$_\mathrm{attention}$",
}
HEAD_SYMBOLS = {
    "coa_phase": (r"$\phi_c$", 2 * np.pi),
    "polarization_angle": (r"$\psi$", np.pi),
    "inclination": (r"$\iota$", 2 * np.pi),
}


def model_key(label: str) -> str:
    return label.split(" (")[0]


def dump(head: str) -> Path:
    from analyse_predictions import CONFIGS, analyse_run

    arrays: dict[str, np.ndarray] = {}
    for label, cfg in CONFIGS.items():
        r = analyse_run(label, cfg)
        if not r or head not in r.get("_pred", {}):
            continue
        k = model_key(label)
        s = r[head]
        arrays[f"{k}__pred"] = np.ravel(r["_pred"][head])
        arrays[f"{k}__true"] = np.ravel(r["_true"][head])
        arrays[f"{k}__r"] = np.array(s["circular_r"])
        arrays[f"{k}__mae"] = np.array(s["angular_mae"])
        arrays[f"{k}__peaks_deg"] = np.array([np.degrees(p[0]) for p in s.get("peaks", [])[:2]])
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"predictions_{head}.npz"
    np.savez(path, **arrays)
    print(f"Dumped: {path}")
    return path


def plot(head: str) -> None:
    from experiments.plot_style import update_style

    update_style()
    symbol, period = HEAD_SYMBOLS[head]
    z = np.load(OUT_DIR / f"predictions_{head}.npz")
    models = [m for m in MODEL_NAMES if f"{m}__pred" in z]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    fig, axes = plt.subplots(2, 2, figsize=(14, 6), sharex=True, squeeze=False)
    for idx, m in enumerate(models):
        ax = axes[idx // 2][idx % 2]
        ax.hist(z[f"{m}__pred"], bins=40, range=(0, period), alpha=0.7, color=colors[idx],
                label="Predicted", edgecolor="white", linewidth=0.3)
        ax.hist(z[f"{m}__true"], bins=40, range=(0, period), alpha=0.3, color="grey",
                label="True", edgecolor="white", linewidth=0.3)
        r, mae = float(z[f"{m}__r"]), float(z[f"{m}__mae"])
        status = "collapse" if r > 0.9 and mae > 0.5 else "good" if mae < 0.3 else "marginal" if mae < 0.8 else "dead"
        peaks = ", ".join(f"{p:.0f}°" for p in z[f"{m}__peaks_deg"])
        text = rf"$r$ = {r:.3f},  MAE = {mae:.2f} rad  ({status})"
        if peaks:
            text += f"\npeaks at {peaks}"
        ax.text(0.99, 0.95, text, transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
        ax.set_title(MODEL_NAMES[m], fontsize=11)
        ax.set_ylabel("Count")
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.legend(fontsize=8, loc="upper left")
    for j in range(len(models), 4):
        axes[j // 2][j % 2].set_visible(False)
    fig.supxlabel(rf"{symbol} [rad]", fontsize=11)
    fig.tight_layout()

    stem = OUT_DIR / f"histogram_throwaway_{head}"
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=200)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Plot: {stem}.png / .pdf")


def main() -> None:
    args = [a for a in sys.argv[1:] if a not in ("dump", "plot")]
    stage = next((a for a in sys.argv[1:] if a in ("dump", "plot")), None)
    head = args[0] if args else "polarization_angle"
    npz = OUT_DIR / f"predictions_{head}.npz"
    if stage == "dump" or (stage is None and not npz.exists()):
        dump(head)
    if stage != "dump":
        plot(head)


if __name__ == "__main__":
    main()
