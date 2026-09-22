# Understanding — working notes on the $\varphi_c/\psi$ redo

This is a growing reference doc, not a narrative log: a running collection of the intricacies, code mechanics, and "wait, why is it built that way" questions that come up while walking through the paper and thesis, written down once so they don't get re-derived from scratch each time.
It has two tracks, each with its own numbering so neither has to be renumbered when the other grows:

- **Active** topics, numbered `N.M` — things currently live in the paper/thesis/code as written.
- **Deferred** topics, numbered `P.Q` — discussions that did happen, but whose subject is no longer current (a plot element that got removed, an approach that got superseded, etc.). Kept rather than deleted, so the earlier discussion isn't silently lost, but clearly marked as not describing present-day artifacts.

A topic moves from Active to Deferred in place when its subject stops being current — see §1.6/§P.1 below for the first example.
Tag any such in-place change note with a `date: time` stamp (`YYYY-MM-DD HH:MM`, e.g. `**Change (2026-09-22 11:20):**`) rather than date alone, so multiple same-day changes stay distinguishable.

## Topics so far

**Active**
1. Step 1.1 prerequisite-check math and the $\iota$-sweep plot (`prereq_checks.py`, `curriculum.py`)

**Deferred**
- P.1. $\pi/4$ / $3\pi/4$ reference-point plot annotations (removed from the figure 2026-09-22 11:20)
- P.2. $y=1.0$ "no preference" line, legend, $y$-axis floor, and face-on/edge-on subtitle arrows (removed from the figure 2026-09-22 11:42, so it can go straight into the paper without clutter)

---

# Active

## 1. Step 1.1 prerequisite-check math and the $\iota$-sweep plot

### 1.1 $\iota$'s domain vs. $\psi$'s domain — two unrelated facts that both involve $\pi$

$\iota$ (inclination) is the angle between the binary's orbital angular momentum and the line of sight.
It is a genuine geometric angle between two vectors, so it ranges over $[0,\pi]$ with no periodicity: $\cos\iota$ sweeps its full range $[-1,1]$ exactly once.
$\iota=0$ and $\iota=\pi$ are both "face-on" in the sense of zero inclination, but they are not the same physical configuration — they are mirror images (prograde vs. retrograde orbital sense as seen by the detector).

$\psi$ (polarization angle) is a different kind of quantity: it enters the waveform doubled ($F_+$/$F_\times$ depend on $2\psi$, see §1.3 below), so $\psi$ and $\psi+\pi$ produce an identical detector response.
Its true period is $\pi$, which is why the circular-regression head for $\psi$ encodes $2\psi$ and its domain is written $[0,\pi)$ (`002_problem-formulation.tex`, "the scientific targets are ... $\psi \in [0,\pi)$, period $\pi$").

These two $\pi$s are coincidental, not causally related.
$\iota$'s range is $[0,\pi]$ because that is what "angle between two vectors" means.
$\psi$'s period is $\pi$ because of how it enters the waveform.
Neither fact derives from the other.

### 1.2 Why the sweep and the plot split at $\iota=\pi/2$

The split point is a third, independent fact, and it is the one with real physical content: $\cos\iota$ changes sign at $\iota=\pi/2$, and that sign change directly flips which sum/difference combination is better-constrained.

From `curriculum.py:29-36`:

```python
def _k1(iota):  # h_plus amplitude coefficient
    return (1.0 + np.cos(iota) ** 2) / 2.0

def _k2(iota):  # h_cross amplitude coefficient
    return np.cos(iota)
```

$h_+$'s coefficient depends on $\cos^2\iota$ — always positive, never flips sign as $\iota$ crosses $\pi/2$.
$h_\times$'s coefficient is $\cos\iota$ itself — it changes sign exactly at $\iota=\pi/2$.
Since the detector strain is $F_+ h_+ + F_\times h_\times$ (`curriculum.py:64-88`, `detector_signal`), that sign change in $h_\times$'s amplitude is what causes the well-constrained combination to flip between $\text{combo}_B$ (for $\cos\iota>0$) and $\text{combo}_A$ (for $\cos\iota<0$) — the "clean sign flip" reported in `003_prereq_study.tex`.

`prereq_checks.py:193-194` builds the sweep as two separate grids matching this physical boundary, not an arbitrary bisection:

```python
sweep_iotas_pos = np.linspace(0.05, np.pi / 2 - 0.02, n_iota_sweep)  # cos ι > 0
sweep_iotas_neg = np.linspace(np.pi / 2 + 0.02, np.pi - 0.05, n_iota_sweep)  # cos ι < 0
```

### 1.3 The waveform and antenna pattern, in full

For reference, the four functions that everything above derives from (`curriculum.py:39-61`):

$$
\begin{aligned}
h_+ &= k_1(\iota)\cdot\cos(2\Phi+2\varphi_c), \quad k_1(\iota) = \tfrac{1+\cos^2\iota}{2} \\
h_\times &= k_2(\iota)\cdot\sin(2\Phi+2\varphi_c), \quad k_2(\iota) = \cos\iota \\
F_+ &= a\cos(2\psi) + b\sin(2\psi) \\
F_\times &= b\cos(2\psi) - a\sin(2\psi)
\end{aligned}
$$

$(a,b)$ are sky-position-derived antenna coefficients, held fixed per sky draw.
$\varphi_c$ and $\psi$ both enter doubled, which is the entire reason a single combination ($2\varphi_c \pm 2\psi$) — not $\varphi_c$ and $\psi$ separately — is the physically meaningful, potentially-recoverable quantity in the face-on limit.

### 1.4 $\text{combo}_A$ / $\text{combo}_B$ — definitions and the shared label convention

$$
\text{combo}_A = 2\varphi_c+2\psi, \qquad \text{combo}_B = 2\varphi_c-2\psi
$$

Confirmed directly in `prereq_checks.py:145-147`'s own comment, and used consistently across `trainer.py`, `transform_utils.py`, `plot_certified_memorization.py`.
Because $\text{combo}_A+\text{combo}_B=4\varphi_c$ and $\text{combo}_A-\text{combo}_B=4\psi$, converting a swept combo grid value back to physical $(\varphi_c,\psi)$ divides by 4, not 2 (`prereq_checks.py:154-155`).

The name → LaTeX-label mapping used for plots/log messages is centralized in `combo_labels.py`:

```python
COMBO_LABELS: dict[str, str] = {
    "combo_A": r"$2\phi_c+2\psi$",
    "combo_B": r"$2\phi_c-2\psi$",
}
```

Import it with a flat `from combo_labels import COMBO_LABELS` — `phic_psi_poc-redo` has a hyphen in its name and cannot be a real Python package, so every script in this directory already relies on flat same-directory imports (e.g. `trainer.py`'s `from curriculum import tf_w_iota`), and any script physically in this directory gets it auto-added to `sys.path`.
Add new combo names here if they're ever introduced; consuming dicts should reference `COMBO_LABELS["combo_A"]` etc. explicitly rather than `**`-unpacking it into a larger unrelated dict (e.g. `plot_certified_memorization.py`'s `log_var_targets`, which also has non-combo head names like `coa_phase`, `mchirp`).

### 1.5 The correlation-ratio sweep: what $\text{mean}_A$/$\text{mean}_B$ actually measure

`_correlations_at_iota` (`prereq_checks.py:139-185`) is the function both the sweep and the reference points in §1.6 call.
At a given $\iota$, for each of 200 sky-position draws:

- Hold $\text{combo}_B$ fixed at a random value, sweep $\text{combo}_A$ across a grid, decode each swept value back to $(\varphi_c,\psi)$, build the detector signal, and project it onto $(R,\delta)$ — the amplitude and phase of the signal's $2\Phi$-frequency component (`curriculum.py:91-108`, `project_to_R_delta`). Correlate the swept $\text{combo}_A$ grid against the resulting $R$ and $\delta$ sequences → `cA_corr_R`, `cA_corr_delta`.
- Do the mirror-image test for $\text{combo}_B$: hold $\text{combo}_A$ fixed, sweep $\text{combo}_B$, correlate against $R$/$\delta$ → `cB_corr_R`, `cB_corr_delta`.

Both combos get the identical treatment — this is not "$\text{combo}_A$ tested against a slew of $\text{combo}_B$ values"; each combo is independently swept while the other is held fixed, and each combo's own effect on the observable is measured on its own terms.

$R$ and $\delta$ are two different observable channels (amplitude and phase).
A combo could constrain the signal through either one, so `prereq_checks.py:181-182` sums both correlations rather than picking one:

```python
mean_A = np.mean(cA_corr_R) + np.mean(cA_corr_delta)
mean_B = np.mean(cB_corr_R) + np.mean(cB_corr_delta)
```

Whichever mean is larger, that combo "wins" (is better-constrained) at that $\iota$.
The ratio computed at every $\iota$ (sweep points and reference points alike) is $\max(\text{mean}_A,\text{mean}_B)/\min(\text{mean}_A,\text{mean}_B)$ — always $\geq 1$ by construction, with $1.0$ meaning "no preference between the two combos."

### 1.6 The $\pi/4$ / $3\pi/4$ reference points

`prereq_checks.py:216-247` picks exactly two fixed inclinations, $\iota=\pi/4$ (representing $\cos\iota>0$) and $\iota=3\pi/4$ (representing $\cos\iota<0$), and calls `_correlations_at_iota` on them a second time, independently of the sweep loop.
This is not the sweep curve's value at those points — it is a fresh calculation, with its own 200-sky-position bootstrap (`n_boot` resamples) to attach a 95% CI to the ratio.

The reason two extra calls exist at all: the sweep gives the trend, but a single citable number-with-error-bar needs one fixed point to bootstrap around, so $\pi/4$/$3\pi/4$ were chosen as roughly-representative midpoints of each half-range.
These are still the numbers `003_prereq_study.tex` quotes as "population-bootstrapped mean ratios": $1.17\times$ (95% CI $[1.11,1.24]$) at $\pi/4$, $1.19\times$ (95% CI $[1.12,1.26]$) at $3\pi/4$ — this computation and its use in the paper are still current.

**Change (2026-09-22 11:20 and 11:42):** these numbers, and the $y=1.0$/legend/subtitle scaffolding described in the original version of this section, used to be drawn directly on `sweep_1_1_ratio_vs_iota.png`.
Both rounds of that plot-level presentation were removed from `_plot_ratio_vs_iota`, deliberately, so the figure is clean enough to go straight into the paper without clutter — see §P.1 and §P.2 for what each round removed and why.
The reference-point numbers themselves are still computed exactly as described above and still get printed to the console (`prereq_checks.py:260-264`) and appended as trailer rows in `sweep_1_1_ratio_vs_iota.csv` (`_export_sweep_csv`, `prereq_checks.py:328-332`) — only the figure-level presentation is gone, not the computation or its role in the paper text.

### 1.7 Reading `sweep_1_1_ratio_vs_iota.png` (current script version)

As of §P.1/§P.2, the figure is deliberately minimal: a bare two-panel scatter, nothing else drawn on it.

- **Scatter points:** the sweep from §1.2/§1.5, one point per $\iota$, colored by which combo won at that $\iota$ (orange/`C0` = $\text{combo}_A$, blue/`C1` = $\text{combo}_B$ — see `_plot_ratio_vs_iota`'s `colors_pos`/`colors_neg` list comprehensions). This is the only data series on the plot.
- **Panel titles ("$\cos\iota>0$" / "$\cos\iota<0$"):** just the regime each panel covers ($\iota\in[0,\pi/2)$ and $\iota\in(\pi/2,\pi]$ respectively — §1.2); no face-on/edge-on reading guide is drawn anymore (§P.2), so pair this with §1.1/§1.2 above if the direction along the x-axis isn't obvious from the axis label alone.
- **Axis labels:** `$\iota$ [rad]` (x, both panels) and `Correlation ratio` (y, left panel only — `sharey=True` makes the right panel's axis redundant).
- **No $y=1.0$ line, no legend, no $y$-axis floor:** removed in §P.2. The ratio is still, structurally, $\max(\text{mean}_A,\text{mean}_B)/\min(\text{mean}_A,\text{mean}_B) \geq 1$ (§1.5) — that mathematical/physical floor still exists, it's just not drawn on the figure; see §P.2 for the reasoning that was previously attached to that line.
- **The empty box in each panel's top-left corner:** an unstyled artifact — a legend frame with no handles/labels rendered inside it (no `.legend()` call remains in `_plot_ratio_vs_iota`, so this is most likely a leftover default from `update_style()`'s global rcParams, not something the function draws on purpose). Cosmetic only; doesn't carry information.
- **If you want the $\pi/4$/$3\pi/4$ numbers, or the removed annotations/line, alongside the figure:** the numbers are in the console log and the CSV's trailer rows (§1.6); what the annotations/line used to look like is in §P.1/§P.2.

---

# Deferred

## P.1 $\pi/4$ / $3\pi/4$ reference-point plot annotations (removed 2026-09-22 11:20)

This describes a plot element that existed when this doc was first written and no longer does — kept for the record, not as a description of the current figure.
See Active §1.6 for what's still true (the underlying bootstrap computation and its numbers, which are unaffected by this).

`_plot_ratio_vs_iota` used to also take `results_by_sign` as a second argument and draw, on each panel, a dotted vertical line at the panel's reference $\iota$ plus a `matplotlib` `annotate()` call with a gray arrow, reading e.g. "$\iota=\pi/4$: $1.17\times$ ($\text{combo}_B$)" on the left panel and "$\iota=3\pi/4$: $1.19\times$ ($\text{combo}_A$)" on the right.
The dotted line itself carried no statistical meaning — it was purely a visual anchor pointing at the x-position the annotation text referred to, unlike the $y=1.0$ dashed line (which is the actual null reference).
The annotated ratio was not the sweep curve's own value at that $x$; it was the separately-bootstrapped reference-point number from §1.6, placed at that $x$-coordinate for the reader's convenience.

This was removed by editing `_plot_ratio_vs_iota` down to a single argument (`sweep_results`) and deleting the `axvline`/`annotate` calls on both panels.
If the annotation is ever wanted back on the figure, the reference numbers to draw it from are still computed and available via `results_by_sign` inside `_check_sign_combination` (§1.6) — nothing needs to be recomputed, only re-plotted.

## P.2 $y=1.0$ "no preference" line, legend, $y$-axis floor, and subtitle arrows (removed 2026-09-22 11:42)

Same status as §P.1: describes a plot presentation that existed earlier in this doc's own life and no longer does, kept for the record.
This was the second, separate round of simplification (after §P.1's), done deliberately so the figure needs no further editing before going into the paper — see Active §1.7 for the resulting bare figure.

Before this round, `_plot_ratio_vs_iota` also had, on each panel:

- `ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.75, label="no preference (1.0)")` — the dashed horizontal floor line (§1.5's mathematical/physical null, see the original explanation preserved here: dividing $\max(\text{mean}_A,\text{mean}_B)$ by $\min(\text{mean}_A,\text{mean}_B)$ can never go below 1, and $\text{mean}_A=\text{mean}_B$ is also the actual physical expectation at exact edge-on).
- `ax.legend(loc=...)` — a legend box, whose only entry was that line's `"no preference (1.0)"` label.
- `ax.set_ylim(0.9, None)` — a fixed lower $y$-limit, so the floor line always had headroom below it.
- Two-line titles with a reading-direction subtitle, e.g. `"cos ι > 0 \n(face-on ← ι=0,  ι=π/2 → edge-on)"`, instead of the current bare `r"$\cos\iota > 0$"`.
- Plain-text axis labels (`"ι [rad]"`, `"correlation ratio (well / poorly)"`) instead of the current LaTeX (`r"$\iota$ [rad]"`, `"Correlation ratio"`).

All of this was cut in the same pass, along with switching the function's `matplotlib` import from a function-local `matplotlib.use("Agg")` block to a module-level `import matplotlib.pyplot as plt`, and switching `save_fig`'s DPI from a hardcoded `300` to `SAVE_DPI` imported from `gwml.evaluation.plot_style` (`prereq_checks.py:39`) — a move toward the shared project plotting-style convention rather than a per-script constant.
The floor/null value itself is unchanged (§1.5); only its visual presentation on this specific figure is gone.
