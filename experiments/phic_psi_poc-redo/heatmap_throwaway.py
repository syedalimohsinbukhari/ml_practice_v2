"""THROWAWAY: standalone re-plot of the health-check heatmap from saved CSVs (CPU only, no model inference).

Reads ``scalar_heads_<ts>.csv``, ``sky_position_<ts>.csv`` and ``periodic_heads_<ts>.csv`` from ``analysis_output/``
and redraws ``_plot_health_heatmap`` from ``analyse_predictions.py`` with display model names, typeset head labels
and no figure title.
Grading thresholds and colours are copied from that function; delete this file once the figure is folded back in.

Usage:
    python experiments/phic_psi_poc-redo/heatmap_throwaway.py [timestamp]   # default: 20261019_125442
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
IN_DIR = HERE / "analysis_output"
OUT_DIR = HERE / "heatmap_throwaway_output"
TS = sys.argv[1] if len(sys.argv) > 1 else "20261019_125442"

# Display names, matching model_labels.py (POC_A, POC_B, TCN, CNN_attention).
MODEL_NAMES = {
    "poc_a": r"POC$_\mathrm{A}$",
    "poc_b": r"POC$_\mathrm{B}$",
    "tcn": "TCN",
    "cnn_attention": r"CNN$_\mathrm{attention}$",
}
HEADS = ["mchirp", "merger_time", "snr", "sky_position", "coa_phase", "polarization_angle", "inclination"]
HEAD_NAMES = {
    "mchirp": "Chirp mass",
    "merger_time": "Merger time",
    "snr": "SNR",
    "sky_position": "Sky position",
    "coa_phase": r"$\phi_c$",
    "polarization_angle": r"$\psi$",
    "inclination": r"$\iota$",
}


def read(name: str) -> list[dict]:
    with open(IN_DIR / f"{name}_{TS}.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def model_key(label: str) -> str:
    return label.split(" (")[0]


def main() -> None:
    stats: dict[str, dict[str, dict]] = {}
    for r in read("scalar_heads"):
        stats.setdefault(model_key(r["model"]), {})[r["head"]] = {"r2": float(r["r2"])}
    for r in read("sky_position"):
        stats.setdefault(model_key(r["model"]), {})["sky_position"] = {"mae_deg": float(r["angular_mae_deg"])}
    for r in read("periodic_heads"):
        stats.setdefault(model_key(r["model"]), {})[r["head"]] = {
            "r": float(r["circular_r"]), "mae": float(r["angular_mae"])}

    models = [m for m in MODEL_NAMES if m in stats]
    data = np.zeros((len(models), len(HEADS)))
    annot = [[""] * len(HEADS) for _ in models]
    for i, m in enumerate(models):
        for j, h in enumerate(HEADS):
            s = stats[m][h]
            if h in ("mchirp", "merger_time", "snr"):
                annot[i][j] = rf"$R^2$ = {s['r2']:.2f}"
                data[i, j] = max(0, s["r2"])
            elif h == "sky_position":
                annot[i][j] = f"{s['mae_deg']:.0f}°"
                data[i, j] = max(0, 1 - s["mae_deg"] / 180)
            elif s["r"] > 0.9 and s["mae"] > 0.5:
                annot[i][j] = f"collapse\n$r$ = {s['r']:.2f}"
                data[i, j] = -0.2
            else:
                annot[i][j] = f"MAE = {s['mae']:.2f} rad"
                data[i, j] = 1.0 if s["mae"] < 0.3 else 0.5 if s["mae"] < 0.8 else 0.1

    cmap = LinearSegmentedColormap.from_list(
        "health", [(0.8, 0.1, 0.1), (0.95, 0.95, 0.95), (1.0, 0.9, 0.5), (0.3, 0.8, 0.3)], N=256)

    fig, ax = plt.subplots(figsize=(2 + 1.8 * len(HEADS), 1.2 + 0.6 * len(models)))
    ax.imshow(data, aspect="auto", cmap=cmap, vmin=-0.3, vmax=1.2)
    for i in range(len(models)):
        for j in range(len(HEADS)):
            v = data[i, j]
            ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=9,
                    color="white" if v < 0 else "black",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white" if v >= 0 else "none",
                              alpha=0.6 if v >= 0 else 0, edgecolor="none"))
    ax.set_xticks(range(len(HEADS)))
    ax.set_xticklabels([HEAD_NAMES[h] for h in HEADS], fontsize=10)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_NAMES[m] for m in models], fontsize=10)
    ax.xaxis.tick_top()
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()

    OUT_DIR.mkdir(exist_ok=True)
    stem = OUT_DIR / f"heatmap_throwaway_{TS}"
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=200)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Plot: {stem}.png / .pdf")


if __name__ == "__main__":
    main()
