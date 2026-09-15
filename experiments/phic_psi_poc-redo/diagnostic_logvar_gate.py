#!/usr/bin/env python
"""Step-0 std_ratio/log_var gate for the redo's first combo-mode run.

Answers exactly one question, mechanically: is `phic_psi_poc_redo_b`'s
flat combo_A/combo_B circular loss (NOTES.md, 2026-09-15) interpretable as
a real null result, or is it UNINTERPRETABLE because |v|-space (the raw
coa_phase/polarization_angle vectors feeding the combo) hasn't stabilized?

This is Step 0 only, from `experiments/phic_psi_poc/preregistration_lam_
retune.md` — the std_ratio gate, reused verbatim (same thresholds,
same window, same pass/fail logic, copied from
`experiments/phic_psi_poc/diagnostic_lam005_retune.py`'s `std_ratio_gate`
function). Steps 1-3 of that preregistration (bootstrap significance,
effect size, SNR stratification) all require loading the trained model and
running prediction on real validation data — GPU/eval work this machine
(CPU-only T530, per CLAUDE.md) does not run locally. This script only
reads existing `history.csv` files; no model loading, no GPU.

Unlike the original's retune round, this run's config isn't yet backed by
a pre-registered decision document (this is the *first* combo-mode run
under the corrected 2phic+/-2psi formula, not a targeted retune of a
known-bad head/model pair) — so only the std_ratio gate is applied as a
hard mechanical pass/fail, using the original's own calibrated thresholds.
The weight_combo_A/weight_combo_B (=exp(-log_var)) trajectory is reported
descriptively alongside it, not as a second hard gate: unlike std_ratio's
[0.5, 2.0]/10%/0.005-per-epoch thresholds (calibrated in the original
investigation against known artifact scales), no equivalent calibration
exists yet for a "weight still rising" threshold, and inventing one now,
after already having seen this run's numbers, would violate exactly the
discipline preregistration_lam_retune.md itself was written to enforce.

Also runs the same gate on `phic_psi_poc_redo_a` (baseline mode, no combo
transform) for comparison context: if the same std_ratio pathology shows
up there too, that points at something upstream of the combo construction
(the shared normalize_unit/magnitude-penalty pathway) rather than
something specific to the corrected formula.

Usage (safe to run locally, CPU-only, reads history.csv only)::

    python experiments/phic_psi_poc-redo/diagnostic_logvar_gate.py
"""

from __future__ import annotations

import sys
from datetime import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))

from experiments.plot_style import update_style  # noqa: E402


# ---------------------------------------------------------------------------
# Logging (console-tee, matches diagnostic_lam005_retune.py's _Tee pattern)
# ---------------------------------------------------------------------------


class _Tee:
    def __init__(self, file_path):
        self.stdout = sys.stdout
        self.file = open(file_path, "w", buffering=1)

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


OUT_DIR = _EXPERIMENT_DIR / "diagnostic_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Step 0 gate — thresholds copied verbatim from
# experiments/phic_psi_poc/diagnostic_lam005_retune.py (calibrated there
# against this investigation's own known artifact scales; not re-derived
# here, per that file's own "use as given" discipline).
# ---------------------------------------------------------------------------

STD_RATIO_LOW, STD_RATIO_HIGH = 0.5, 2.0
STD_RATIO_FRAC_THRESHOLD = 0.10
STD_RATIO_TREND_THRESHOLD = 0.005
GATE_WINDOW = 40

RUNS = {
    "poc_redo_b (poc, tcn)": _REPO_ROOT / "runs" / "phic_psi_poc_redo_b",
    "poc_redo_a (baseline, tcn)": _REPO_ROOT / "runs" / "phic_psi_poc_redo_a",
}
HEADS = ("coa_phase", "polarization_angle")


def _latest_run_subdir(base_dir: Path) -> Path | None:
    if not base_dir.exists():
        return None
    children = sorted(p for p in base_dir.iterdir() if p.is_dir())
    return children[-1] if children else None


def std_ratio_gate(history_csv: Path, head: str) -> dict:
    """Step 0 gate, copied from diagnostic_lam005_retune.py's std_ratio_gate."""
    col = f"val_std_ratio_{head}"
    if not history_csv.exists():
        return {"passed": False, "reason": f"no history.csv at {history_csv}"}
    df = pd.read_csv(history_csv)
    if col not in df.columns:
        return {"passed": False, "reason": f"{col} not in history.csv"}
    late = df[col].iloc[-GATE_WINDOW:] if len(df) >= GATE_WINDOW else df[col]
    frac_unhealthy = float(((late < STD_RATIO_LOW) | (late > STD_RATIO_HIGH)).mean())
    trend = float(np.polyfit(np.arange(len(late)), late.values, 1)[0])
    passed = (
        frac_unhealthy < STD_RATIO_FRAC_THRESHOLD
        and abs(trend) < STD_RATIO_TREND_THRESHOLD
    )
    return {
        "passed": passed,
        "frac_unhealthy": frac_unhealthy,
        "trend": trend,
        "final_value": float(df[col].iloc[-1]),
    }


def weight_trajectory_summary(history_csv: Path, col: str) -> dict | None:
    """Descriptive only (not a hard gate) — see module docstring."""
    if not history_csv.exists():
        return None
    df = pd.read_csv(history_csv)
    if col not in df.columns:
        return None
    s = df[col]
    late = s.iloc[-GATE_WINDOW:] if len(s) >= GATE_WINDOW else s
    late_trend = float(np.polyfit(np.arange(len(late)), late.values, 1)[0])
    return {
        "first": float(s.iloc[0]),
        "last": float(s.iloc[-1]),
        "late_trend": late_trend,
        "still_rising_in_window": abs(late_trend) >= STD_RATIO_TREND_THRESHOLD,
    }


def _plot_trajectories(run_histories: dict[str, tuple[Path, pd.DataFrame]]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(11, 12), sharex=True)
    for col_idx, (label, (_, df)) in enumerate(run_histories.items()):
        epochs = df["epoch"] + 1
        ax = axes[0][col_idx]
        for head in HEADS:
            c = f"val_std_ratio_{head}"
            if c in df.columns:
                ax.plot(epochs, df[c], label=head)
        ax.axhspan(STD_RATIO_LOW, STD_RATIO_HIGH, color="green", alpha=0.08, label="healthy band")
        ax.set_title(f"{label}\nstd_ratio (val)")
        ax.set_ylabel("std_ratio")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.25)

        ax2 = axes[1][col_idx]
        loss_cols = [c for c in df.columns if c.startswith("val_circular_loss_")]
        for c in loss_cols:
            ax2.plot(epochs, df[c], label=c.replace("val_circular_loss_", ""))
        ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="null baseline")
        ax2.set_ylabel("circular loss (val)")
        ax2.legend(fontsize="small")
        ax2.grid(True, alpha=0.25)

        ax3 = axes[2][col_idx]
        weight_cols = [c for c in df.columns if c.startswith("val_weight_") and
                       ("combo" in c or any(h in c for h in HEADS))]
        for c in weight_cols:
            ax3.plot(epochs, df[c], label=c.replace("val_weight_", ""))
        ax3.set_ylabel("weight = exp(-log_var) (val)")
        ax3.set_xlabel("epoch")
        ax3.legend(fontsize="small")
        ax3.grid(True, alpha=0.25)

    fig.suptitle("Step 0 diagnostic: std_ratio gate / circular loss / uncertainty weight")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"logvar_gate_trajectories.{ext}", dpi=200 if ext == "png" else None)
    plt.close(fig)


def main() -> None:
    ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    log_path = OUT_DIR / f"diagnostic_logvar_gate_{ts}.log"
    tee = _Tee(str(log_path))
    sys.stdout = tee
    try:
        update_style()
        print("=" * 100)
        print("Step 0 diagnostic — std_ratio gate (phic_psi_poc_redo_b combo-mode run)")
        print("Thresholds copied from experiments/phic_psi_poc/diagnostic_lam005_retune.py")
        print("=" * 100)

        run_histories = {}
        summary_rows = []

        for label, base_dir in RUNS.items():
            run_dir = _latest_run_subdir(base_dir)
            print(f"\n{'-' * 100}\n  {label}  ->  {run_dir}\n{'-' * 100}")
            if run_dir is None:
                print("  no run directory found")
                continue
            history_csv = run_dir / "history.csv"
            if not history_csv.exists():
                print(f"  no history.csv at {run_dir}")
                continue
            df = pd.read_csv(history_csv)
            run_histories[label] = (run_dir, df)

            for head in HEADS:
                gate = std_ratio_gate(history_csv, head)
                verdict = "PASS" if gate.get("passed") else "FAIL"
                print(f"\n  Step 0 gate — {head}:")
                print(f"    frac unhealthy (last {GATE_WINDOW} ep) = {gate.get('frac_unhealthy', float('nan')):.3f}")
                print(f"    late-epoch trend/ep                    = {gate.get('trend', float('nan')):+.5f}")
                print(f"    final val_std_ratio                    = {gate.get('final_value', float('nan')):.3f}")
                print(f"    GATE {verdict}")
                summary_rows.append((label, head, gate, verdict))

            # weight/log_var descriptive trajectory (only meaningful for combo runs)
            for wcol in ("val_weight_combo_A", "val_weight_combo_B"):
                summ = weight_trajectory_summary(history_csv, wcol)
                if summ is None:
                    continue
                print(f"\n  {wcol} (descriptive, not a hard gate):")
                print(f"    first={summ['first']:.4f}  last={summ['last']:.4f}  "
                      f"late trend/ep={summ['late_trend']:+.5f}  "
                      f"still rising in last {GATE_WINDOW} ep? {summ['still_rising_in_window']}")

        if run_histories:
            _plot_trajectories(run_histories)
            print(f"\nPlots written: {OUT_DIR / 'logvar_gate_trajectories.png'} / .pdf")

        # ---- Overall verdict for the central question ----
        b_heads = [r for r in summary_rows if r[0] == "poc_redo_b (poc, tcn)"]
        overall_pass = bool(b_heads) and all(r[3] == "PASS" for r in b_heads)
        print("\n" + "=" * 100)
        print("OVERALL VERDICT (poc_redo_b, both heads must pass):",
              "PASS -> combo_A/combo_B flat loss is interpretable as a real null result, proceed to Steps 1-3 on the lab GPU machine"
              if overall_pass else
              "FAIL -> UNINTERPRETABLE. Per preregistration_lam_retune.md's own decision table, do not read the flat "
              "circular loss as evidence either way. Next step: lambda-retune, same pattern as the original's Runs 8-9b.")
        print("=" * 100)

        # ---- Summary .md ----
        md_path = OUT_DIR / f"diagnostic_logvar_gate_{ts}.md"
        with open(md_path, "w") as f:
            f.write("# Step 0 Diagnostic — std_ratio Gate (phic_psi_poc_redo_b)\n\n")
            f.write(
                "Answers whether the flat `circular_loss_combo_A`/`combo_B` result "
                "(`NOTES.md`, 2026-09-15) is interpretable, using the std_ratio gate "
                "from [`preregistration_lam_retune.md`]"
                "(../phic_psi_poc/preregistration_lam_retune.md) "
                "(thresholds copied verbatim, not re-derived).\n\n"
            )
            f.write("| Run | Head | frac unhealthy (last 40 ep) | trend/ep | final std_ratio | Gate |\n")
            f.write("|---|---|---|---|---|---|\n")
            for label, head, gate, verdict in summary_rows:
                f.write(
                    f"| {label} | {head} | {gate.get('frac_unhealthy', float('nan')):.3f} "
                    f"| {gate.get('trend', float('nan')):+.5f} "
                    f"| {gate.get('final_value', float('nan')):.3f} | {verdict} |\n"
                )
            f.write(
                f"\n**Overall verdict (poc_redo_b, both heads must pass): "
                f"{'PASS' if overall_pass else 'FAIL — UNINTERPRETABLE'}**\n\n"
            )
            f.write("See log for the descriptive weight_combo_A/B trajectory and the "
                     "baseline-mode (poc_redo_a) comparison.\n")
        print(f"\nSummary written: {md_path}")

    finally:
        sys.stdout = tee.stdout
        tee.close()


if __name__ == "__main__":
    main()
