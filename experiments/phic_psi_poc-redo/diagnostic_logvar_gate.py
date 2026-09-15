#!/usr/bin/env python
"""Step-0 std_ratio/log_var gate for the redo's combo-mode runs, across λ rounds.

Answers exactly one question, mechanically, per λ round: is `poc_redo_b`'s
(or its retune's) combo_A/combo_B circular loss interpretable as a real
null result, or is it UNINTERPRETABLE because |v|-space (the raw
coa_phase/polarization_angle vectors feeding the combo) hasn't stabilized?

This is Step 0 only, from `preregistration_lam_retune.md` — the std_ratio
gate, reused verbatim (same thresholds, same window, same pass/fail logic,
copied from `experiments/phic_psi_poc/diagnostic_lam005_retune.py`'s
`std_ratio_gate` function). Steps 1-3 of that preregistration (bootstrap
significance, effect size, SNR stratification) all require loading the
trained model and running prediction on real validation data — GPU/eval
work this machine (CPU-only T530, per CLAUDE.md) does not run locally.
This script only reads existing `history.csv` files; no model loading, no
GPU.

`ROUNDS` (below) lists every λ tried so far, each with its `"b"` (poc mode,
the primary target) and `"a"` (baseline mode, the required control) run
directories — see `preregistration_lam_retune.md`'s Scope: "All four must
pass for Step 0 to clear" for a given round, since combo_A/combo_B depend
on *both* poc_redo_b's raw vectors, and an unhealthy control isn't a valid
comparison point. Earlier rounds (e.g. λ=0.01, already known to fail) stay
in the dict for side-by-side context every time this reruns; the **last**
entry in `ROUNDS` is treated as the round under test — its per-round
verdict is the "OVERALL VERDICT" this script reports and the one that
decides whether to try the next λ or move on to Steps 1-3. Extending to a
λ=0.10 fallback (per the preregistration's own fallback order) is just
appending one more entry.

The weight_combo_A/weight_combo_B (=exp(-log_var)) trajectory is reported
descriptively alongside the gate, not as a second hard gate: unlike
std_ratio's [0.5, 2.0]/10%/0.005-per-epoch thresholds (calibrated in the
original investigation against known artifact scales), no equivalent
calibration exists for a "weight still rising" threshold, and inventing
one now would violate exactly the discipline `preregistration_lam_retune.md`
itself was written to enforce (this script already corrected one instance
of eyeballing this exact trajectory wrongly — see NOTES.md, 2026-09-15).

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

# Every λ round tried so far, in order. Each entry: "b" = poc_redo_b
# equivalent (primary, poc mode), "a" = poc_redo_a equivalent (required
# control, baseline mode). The LAST entry is the round under test — see
# module docstring. Add the λ=0.10 fallback here (same pattern) if λ=0.05
# also fails.
ROUNDS = {
    "λ=0.01 (Round 1)": {
        "b": _REPO_ROOT / "runs" / "phic_psi_poc_redo_b",
        "a": _REPO_ROOT / "runs" / "phic_psi_poc_redo_a",
    },
    "λ=0.05 (retune)": {
        "b": _REPO_ROOT / "runs" / "phic_psi_lam005_retune_b",
        "a": _REPO_ROOT / "runs" / "phic_psi_lam005_retune_a",
    },
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

    n_cols = len(run_histories)
    fig, axes = plt.subplots(3, n_cols, figsize=(5.5 * n_cols, 12), sharex=True, squeeze=False)
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
        print("Step 0 diagnostic — std_ratio gate across λ rounds (poc_redo_b / poc_redo_a)")
        print("Thresholds copied from experiments/phic_psi_poc/diagnostic_lam005_retune.py")
        print("=" * 100)

        run_histories = {}
        summary_rows = []  # (round_label, variant, label, head, gate, verdict)
        round_verdicts = {}  # round_label -> (all_pass, [ (variant, head, verdict), ... ])

        for round_label, variants in ROUNDS.items():
            round_rows = []
            for variant, base_dir in variants.items():  # variant: "b" (primary) or "a" (control)
                label = f"{round_label} [{variant}]"
                run_dir = _latest_run_subdir(base_dir)
                print(f"\n{'-' * 100}\n  {label}  ->  {run_dir}\n{'-' * 100}")
                if run_dir is None:
                    print("  no run directory found")
                    round_rows.append((variant, None, "NO RUN"))
                    continue
                history_csv = run_dir / "history.csv"
                if not history_csv.exists():
                    print(f"  no history.csv at {run_dir}")
                    round_rows.append((variant, None, "NO RUN"))
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
                    summary_rows.append((round_label, variant, label, head, gate, verdict))
                    round_rows.append((variant, head, verdict))

                # weight/log_var descriptive trajectory (only meaningful for combo/poc runs)
                for wcol in ("val_weight_combo_A", "val_weight_combo_B"):
                    summ = weight_trajectory_summary(history_csv, wcol)
                    if summ is None:
                        continue
                    print(f"\n  {wcol} (descriptive, not a hard gate):")
                    print(f"    first={summ['first']:.4f}  last={summ['last']:.4f}  "
                          f"late trend/ep={summ['late_trend']:+.5f}  "
                          f"still rising in last {GATE_WINDOW} ep? {summ['still_rising_in_window']}")

            round_verdicts[round_label] = round_rows

        if run_histories:
            _plot_trajectories(run_histories)
            print(f"\nPlots written: {OUT_DIR / 'logvar_gate_trajectories.png'} / .pdf")

        # ---- Overall verdict: the LAST round in ROUNDS is the one under test.
        # Per preregistration_lam_retune.md's Scope: all four readings (both
        # variants x both heads) must pass for that round to clear Step 0.
        current_round_label = list(ROUNDS.keys())[-1]
        current_rows = round_verdicts.get(current_round_label, [])
        have_all_four = len(current_rows) == 4 and all(v not in (None, "NO RUN") for _, _, v in current_rows)
        overall_pass = have_all_four and all(v == "PASS" for _, _, v in current_rows)
        print("\n" + "=" * 100)
        print(f"OVERALL VERDICT — round under test: {current_round_label} (all 4 readings must pass):")
        if not current_rows or not have_all_four:
            print("  NOT YET RUN — no data for this round yet.")
        else:
            print("  PASS -> combo_A/combo_B flat loss is interpretable as a real null result, "
                  "proceed to Steps 1-3 on the lab GPU machine"
                  if overall_pass else
                  "  FAIL -> UNINTERPRETABLE. Per preregistration_lam_retune.md's own decision table, do not read "
                  "the flat circular loss as evidence either way. Next step: next fallback lambda "
                  "(0.10), same pattern as the original's Runs 8-9b.")
        print("=" * 100)

        # ---- Summary .md ----
        md_path = OUT_DIR / f"diagnostic_logvar_gate_{ts}.md"
        with open(md_path, "w") as f:
            f.write("# Step 0 Diagnostic — std_ratio Gate, across λ rounds\n\n")
            f.write(
                "Answers whether `poc_redo_b`'s `circular_loss_combo_A`/`combo_B` result "
                "is interpretable at each λ tried, using the std_ratio gate from "
                "[`preregistration_lam_retune.md`](preregistration_lam_retune.md) "
                "(thresholds copied verbatim, not re-derived).\n\n"
            )
            f.write("| λ round | Variant | Head | frac unhealthy (last 40 ep) | trend/ep | final std_ratio | Gate |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for round_label, variant, label, head, gate, verdict in summary_rows:
                f.write(
                    f"| {round_label} | {variant} | {head} | {gate.get('frac_unhealthy', float('nan')):.3f} "
                    f"| {gate.get('trend', float('nan')):+.5f} "
                    f"| {gate.get('final_value', float('nan')):.3f} | {verdict} |\n"
                )
            f.write(
                f"\n**Overall verdict — round under test ({current_round_label}), all 4 readings must pass: "
                f"{'PASS' if overall_pass else ('NOT YET RUN' if not current_rows or not have_all_four else 'FAIL — UNINTERPRETABLE')}**\n\n"
            )
            f.write("Earlier rounds are shown for context, not re-adjudicated. "
                     "See log for the descriptive weight_combo_A/B trajectory.\n")
        print(f"\nSummary written: {md_path}")

    finally:
        sys.stdout = tee.stdout
        tee.close()


if __name__ == "__main__":
    main()
