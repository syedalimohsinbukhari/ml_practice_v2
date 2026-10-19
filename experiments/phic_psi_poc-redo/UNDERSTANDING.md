# Understanding — working notes on the $\varphi_c/\psi$ redo

This is a growing reference doc, not a narrative log: a running collection of the intricacies, code mechanics, and "wait, why is it built that way" questions that come up while walking through the paper and thesis, written down once so they don't get re-derived from scratch each time.
Its prime concept is the paper's *understanding* — what's currently true about a mechanism and why — not the investigative trail of getting there: open questions, hypotheses tried, dated status updates, and "fix applied" narration belong in `NOTES.md` (or another appropriately-scoped file) as they're worked through, and only the settled result gets written here, once, in reference-document voice.
When a topic's own investigation is worth preserving in detail, cross-reference it from here rather than reproducing it (see §4's closing pointer to `NOTES.md` for the pattern).
It has three tracks, each with its own numbering so none has to be renumbered when the others grow:

- **Active** topics, numbered `N.M` — things currently live in the paper/thesis/code as written. `N` matches the paper's own section-file number in the paper repo (`../gw_research_paper/`) (e.g. §2 ↔ `002_problem-formulation.tex`, §3 ↔ `003_prereq_study.tex`), so the doc section a topic sits under tells you which paper file it documents; `M` is just that topic's own sub-numbering.
- **Deferred** topics, numbered `P.Q` — discussions that did happen, but whose subject is no longer current (a plot element that got removed, an approach that got superseded, etc.). Kept rather than deleted, so the earlier discussion isn't silently lost, but clearly marked as not describing present-day artifacts. Deliberately *not* tied to a paper section number, since by definition they no longer describe what the paper currently says.
- **Tooling** topics, numbered `T.Q` — lessons about the writing/build infrastructure itself (LaTeX macro conventions, `\ensuremath` pitfalls, the paper/Python notation-consistency setup), not about the paper's scientific content. Also not tied to a paper section number, for the opposite reason from Deferred: these are evergreen craft notes that apply across every section rather than documenting one of them.

A topic moves from Active to Deferred in place when its subject stops being current — see §3.6/§P.1 below for the first example.
Tag any such in-place change note with a `date: time` stamp (`YYYY-MM-DD HH:MM`, e.g. `**Change (2026-09-22 11:20):**`) rather than date alone, so multiple same-day changes stay distinguishable.

## Topics so far

**Active**
2. The circular (cosine) loss and unit-circle normalization (`transform_utils.py`, `trainer.py`, `002_problem-formulation.tex`)
3. Step 1.1 prerequisite-check math and the $\iota$-sweep plot (`prereq_checks.py`, `curriculum.py`, `003_prereq_study.tex`)
4. The dataset's inclination prior: $\iota$'s true $[0,2\pi)$ support, the face-on/edge-on over-representation direction, and the inclination control head's partial mode collapse (`004_methods.tex`, `003_prereq_study.tex`, `thesis/chapter_phic_psi_degeneracy.{md,tex}`, `inclination_stratification.py`, `ml-gw-search/mlgwsc-1/gen.py`)

**Deferred**
- P.1. $\pi/4$ / $3\pi/4$ reference-point plot annotations (removed from the figure 2026-09-22 11:20)
- P.2. $y=1.0$ "no preference" line, legend, $y$-axis floor, and face-on/edge-on subtitle arrows (removed from the figure 2026-09-22 11:42, so it can go straight into the paper without clutter)

**Tooling**
- T.1. The paper's notation-macro convention (`../gw_research_paper/000_preamble.tex`), and two `\ensuremath` pitfalls found while extending it

---

# Active

## 2. The circular (cosine) loss and unit-circle normalization

### 2.1 The unit-circle encoding, and three different normalization implementations

Every periodic head (`002_problem-formulation.tex` §1.5) outputs a raw two-vector $\mathbf{v}\in\mathbb{R}^2$, which the paper says is projected onto the unit circle by $\hat{\mathbf{u}} = \mathbf{v}/\max(\|\mathbf{v}\|,\varepsilon)$, $\varepsilon=10^{-8}$.
In code, this normalization exists in **three** slightly different forms, and only one of them literally matches that formula.

The one actually used in training (`transform_utils.py:131-136`, `tf_normalize_unit`):

```python
sq = tf.reduce_sum(tf.square(z), axis=-1, keepdims=True)
norm = tf.sqrt(tf.maximum(sq, eps))        # max on ‖v‖², not ‖v‖
return z / norm
```

This clamps $\|v\|^2$, not $\|v\|$: the effective floor becomes $\|v\| \to \sqrt{\max(\|v\|^2,\varepsilon)}$, i.e. with $\varepsilon=10^{-8}$ the real floor on $\|v\|$ is $\sqrt{10^{-8}}=10^{-4}$ — four orders of magnitude looser than $\max(\|v\|,10^{-8})$ as literally written in the paper.

A second, NumPy-only variant (`transform_utils.py:28-44`, `normalize_unit`, used by validation scripts, not training) uses an *additive* epsilon instead of a clamp at all:

```python
norm = np.sqrt(z[..., 0] ** 2 + z[..., 1] ** 2) + eps
```

Its docstring claims it "reuses the same epsilon convention as the vMF head's `mu_raw` normalisation" (`src/gwml/training/losses.py:52-55`) — and that vMF normalization genuinely does implement the paper's literal $\max(\|v\|,\varepsilon)$ form, unclamped-square, unlike either periodic-head variant above.
So the periodic-head normalization (the one $\varphi_c$/$\psi$/$\text{combo}_A$/$\text{combo}_B$ actually train through) is the odd one out relative to both the paper text and the vMF head it claims consistency with.

This is harmless in practice: it only changes behavior when $\|v\|$ is within $10^{-8}$–$10^{-4}$ of zero, which isn't the collapse mode actually observed anywhere in this study — collapsed heads concentrate at a fixed *angle* with a well-behaved $\|v\|$ (that's the entire premise of the $\sigma_\text{head,target}$/std-ratio diagnostic, §3.5's sibling concept for the raw-magnitude axis rather than the sweep-ratio axis), not at the origin.

### 2.2 The loss formula, and where it's actually computed

$$
L_{\text{circ}} = 1 - \hat{\mathbf{u}}_{\text{pred}}\cdot\mathbf{u}_{\text{true}} = 1 - \cos\Delta\theta
$$

Since both vectors are unit-normalized, their dot product is exactly $\cos(\Delta\theta)$, the cosine of the angular difference — so $L_{\text{circ}}\in[0,2]$, zero when the prediction is exact, smoothly increasing (no discontinuity) as the angular error grows.
This is what makes it a *circular* loss: an angle near $0$ and an angle near $2\pi$ are treated as close, unlike a naive MSE on the raw angle value, which would see them as maximally far apart despite being adjacent on the circle.

Confirmed literally as `1.0 - tf.reduce_sum(pred * true, axis=-1)`, computed **inline**, independently, at three separate call sites in `trainer.py`:

- Baseline mode ($\varphi_c$, $\psi$ individually): lines 420, 425.
- PoC mode ($\text{combo}_A$, $\text{combo}_B$, built via `tf_complex_mul`/`tf_complex_mul_conj`/`tf_double_angle` in `_build_combo_vectors`, lines 268-316): lines 462-467.
- `CircularMetric.update_state` (lines 58-80) — a parallel, metrics-only reimplementation of the identical formula, used only for logging `circular_loss_<head>`, not for gradients.

A standalone `circular_loss()` function also exists (`transform_utils.py:98-117`, same one-line formula, docstring: `"Isotropic circular loss: L = 1 − dot(pred, true) = 1 − cos(Δθ)"`) but is **not** actually in the training gradient path — it's called elsewhere in `transform_utils.py` only for an unrelated self-consistency/roundtrip check (lines 253-254).
So there are effectively three independent inline copies of the same formula rather than one shared function every head routes through — a minor DRY gap, not a semantic inconsistency, since all three are algebraically identical.

### 2.3 Order of operations: curriculum weight → mean → uncertainty weighting

The per-sample circular loss (shape `(N,)`) is reduced to a scalar in stages, not applied directly:

1. Per-sample loss computed (§2.2).
2. Reduced to a mean — in PoC mode, curriculum-weighted by $w(\iota)$ first (`trainer.py:499-500`; $w(\iota)$ itself is §3's `curriculum.py` derivation, the same function whose face-on/edge-on endpoints are discussed in §3.2/§3.3).
3. Kendall-style homoscedastic uncertainty weighting applied to that mean: `total += exp(-s_h) * loss_mean + s_h` (`trainer.py:422,427,505-506`), matching the parent `MultiHeadTrainer._total_loss` pattern (`src/gwml/training/losses.py:239`).

One real subtlety in PoC mode: the curriculum weight is baked into the mean *before* the uncertainty weight $s_h$ ever sees it.
So the learned per-head uncertainty $s_A$/$s_B$ cannot distinguish "this combo is intrinsically hard to learn" from "this combo was already down-weighted by the curriculum at the inclinations in this batch" — both show up to the uncertainty-weighting stage as the same thing: a smaller loss mean. This doesn't invalidate either mechanism on its own terms, but it means the two weighting schemes aren't fully independent axes of adjustment the way their separate motivations (curriculum: which combo is *informative* at this $\iota$; uncertainty: which head is *harder* overall) might suggest.

## 3. Step 1.1 prerequisite-check math and the $\iota$-sweep plot

### 3.1 $\iota$'s domain vs. $\psi$'s domain — two unrelated facts that both involve $\pi$

$\iota$ (inclination) is the angle between the binary's orbital angular momentum and the line of sight.
It is a genuine geometric angle between two vectors, so it ranges over $[0,\pi]$ with no periodicity: $\cos\iota$ sweeps its full range $[-1,1]$ exactly once.
$\iota=0$ and $\iota=\pi$ are both "face-on" in the sense of zero inclination, but they are not the same physical configuration — they are mirror images (prograde vs. retrograde orbital sense as seen by the detector).

$\psi$ (polarization angle) is a different kind of quantity: it enters the waveform doubled ($F_+$/$F_\times$ depend on $2\psi$, see §3.3 below), so $\psi$ and $\psi+\pi$ produce an identical detector response.
Its true period is $\pi$, which is why the circular-regression head for $\psi$ encodes $2\psi$ and its domain is written $[0,\pi)$ (`002_problem-formulation.tex`, "the scientific targets are ... $\psi \in [0,\pi)$, period $\pi$").

These two $\pi$s are coincidental, not causally related.
$\iota$'s range is $[0,\pi]$ because that is what "angle between two vectors" means.
$\psi$'s period is $\pi$ because of how it enters the waveform.
Neither fact derives from the other.

### 3.2 Why the sweep and the plot split at $\iota=\pi/2$

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

### 3.3 The waveform and antenna pattern, in full

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

### 3.4 $\text{combo}_A$ / $\text{combo}_B$ — definitions and the shared label convention

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

### 3.5 The correlation-ratio sweep: what $\text{mean}_A$/$\text{mean}_B$ actually measure

`_correlations_at_iota` (`prereq_checks.py:139-185`) is the function both the sweep and the reference points in §3.6 call.
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

### 3.6 The $\pi/4$ / $3\pi/4$ reference points

`prereq_checks.py:216-247` picks exactly two fixed inclinations, $\iota=\pi/4$ (representing $\cos\iota>0$) and $\iota=3\pi/4$ (representing $\cos\iota<0$), and calls `_correlations_at_iota` on them a second time, independently of the sweep loop.
This is not the sweep curve's value at those points — it is a fresh calculation, with its own 200-sky-position bootstrap (`n_boot` resamples) to attach a 95% CI to the ratio.

The reason two extra calls exist at all: the sweep gives the trend, but a single citable number-with-error-bar needs one fixed point to bootstrap around, so $\pi/4$/$3\pi/4$ were chosen as roughly-representative midpoints of each half-range.
These are still the numbers `003_prereq_study.tex` quotes as "population-bootstrapped mean ratios": $1.17\times$ (95% CI $[1.11,1.24]$) at $\pi/4$, $1.19\times$ (95% CI $[1.12,1.26]$) at $3\pi/4$ — this computation and its use in the paper are still current.

**Change (2026-09-22 11:20 and 11:42):** these numbers, and the $y=1.0$/legend/subtitle scaffolding described in the original version of this section, used to be drawn directly on `sweep_1_1_ratio_vs_iota.png`.
Both rounds of that plot-level presentation were removed from `_plot_ratio_vs_iota`, deliberately, so the figure is clean enough to go straight into the paper without clutter — see §P.1 and §P.2 for what each round removed and why.
The reference-point numbers themselves are still computed exactly as described above and still get printed to the console (`prereq_checks.py:260-264`) and appended as trailer rows in `sweep_1_1_ratio_vs_iota.csv` (`_export_sweep_csv`, `prereq_checks.py:328-332`) — only the figure-level presentation is gone, not the computation or its role in the paper text.

### 3.7 Reading `sweep_1_1_ratio_vs_iota.png` (current script version)

As of §P.1/§P.2, the figure is deliberately minimal: a bare two-panel scatter, nothing else drawn on it.

- **Scatter points:** the sweep from §3.2/§3.5, one point per $\iota$, colored by which combo won at that $\iota$ (orange/`C0` = $\text{combo}_A$, blue/`C1` = $\text{combo}_B$ — see `_plot_ratio_vs_iota`'s `colors_pos`/`colors_neg` list comprehensions). This is the only data series on the plot.
- **Panel titles ("$\cos\iota>0$" / "$\cos\iota<0$"):** just the regime each panel covers ($\iota\in[0,\pi/2)$ and $\iota\in(\pi/2,\pi]$ respectively — §3.2); no face-on/edge-on reading guide is drawn anymore (§P.2), so pair this with §3.1/§3.2 above if the direction along the x-axis isn't obvious from the axis label alone.
- **Axis labels:** `$\iota$ [rad]` (x, both panels) and `Correlation ratio` (y, left panel only — `sharey=True` makes the right panel's axis redundant).
- **No $y=1.0$ line, no legend, no $y$-axis floor:** removed in §P.2. The ratio is still, structurally, $\max(\text{mean}_A,\text{mean}_B)/\min(\text{mean}_A,\text{mean}_B) \geq 1$ (§3.5) — that mathematical/physical floor still exists, it's just not drawn on the figure; see §P.2 for the reasoning that was previously attached to that line.
- **The empty box in each panel's top-left corner:** an unstyled artifact — a legend frame with no handles/labels rendered inside it (no `.legend()` call remains in `_plot_ratio_vs_iota`, so this is most likely a leftover default from `update_style()`'s global rcParams, not something the function draws on purpose). Cosmetic only; doesn't carry information.
- **If you want the $\pi/4$/$3\pi/4$ numbers, or the removed annotations/line, alongside the figure:** the numbers are in the console log and the CSV's trailer rows (§3.6); what the annotations/line used to look like is in §P.1/§P.2.

### 3.8 The curriculum weight $w(\iota)$: a measured Jacobian-condition-number curve exists, but training uses the simpler analytic fallback

`curriculum.py`'s `derive_w_iota` computes $w(\iota)$ empirically, not by guessing.
At a fixed reference $(\varphi_c,\psi)$, for 50 random sky-position $(a,b)$ draws, it sweeps $\iota$ across $[0,\pi/2]$ and at each point computes the **Jacobian condition number** of the map $(\varphi_c,\psi)\to(R,\delta)$ (`_jacobian_condition_number`: central finite differences, then the ratio of largest to smallest singular value of the resulting $2\times2$ Jacobian).
A high condition number means that map is nearly singular — two input directions are collapsing onto barely-distinguishable observables, i.e. genuinely degenerate at that $\iota$.
Weight is set $\propto1/\text{cond}(J)$, averaged across the 50 sky draws, normalized to max $1$, then smoothed by linear interpolation on $\cos^2\iota$ — chosen specifically because an earlier unconstrained polynomial fit produced negative weights and residuals larger than the function's own range, an already-fixed, unrelated problem.
Result: $w(\text{face-on})=0.0000$, rising to a peak $\approx0.9956$ near $\iota\approx0.789$, then *falling back down* to only $w(\text{edge-on})=0.1416$ — an intermediate-peak shape, not a monotonic one.

Training instead uses `w_iota_default`, the simple closed-form $w(\iota)=1-\cos^2\iota=\sin^2\iota$: zero at face-on (matching the measured curve there), but monotonically increasing all the way to $1$ at edge-on, unlike the measured curve's dip.
Reason for the choice: the measured curve's edge-on dip is ambiguous — it could be real physics (this quasi-linear map genuinely re-degrading near edge-on, for a reason the simple face-on/edge-on story doesn't capture), or a harness artifact (the fixed reference $(\varphi_c,\psi)$ point, the specific sky-position sampling, or the interpolation smoothing itself), and nobody had independently verified which.
Baking an unverified dip directly into the training signal risks telling the network to under-weight edge-on samples — exactly the regime §3.5's own analytic sweep says is *least* degenerate — for no confirmed reason.
The fallback is the conservative choice for a first training pass: it encodes only the one certain fact (exact degeneracy at face-on) and adds no unverified assumption beyond that.
`build_w_iota_from_fit` exists to swap in the measured curve later, if its edge-on behavior is ever independently confirmed as real rather than artifact — not done as of this writing.

## 4. The dataset's inclination prior, and the inclination control head's behavior

### 4.1 $\iota$ is drawn uniform over $[0,2\pi)$; only $\cos\iota$ ever matters physically

$\iota$ is drawn $\mathrm{Uniform}(0,2\pi)$ in this dataset (`004_methods.tex`'s population table; generator `ml-gw-search/mlgwsc-1/gen.py:201`, `angles = np_gen.uniform(0., 2*np.pi, 3)` for `coa_phase`/`inclination`/`polarization_angle` together, matching `heads_spec.py:99-100`'s `PERIODIC, period=2π` declaration).
Every piece of machinery that trains on $\iota$ — the curriculum weight $w(\iota)=\sin^2\iota$, the analytic $k_1(\iota)=(1+\cos^2\iota)/2$/$k_2(\iota)=\cos\iota$ coefficients (§3.2/§3.3), the prereq sweep, the combo_A/combo_B construction — depends on $\iota$ only through $\cos\iota$, never on $\iota$ itself: $\cos(\cdot)$ is $2\pi$-periodic and even, so $\mathrm{Uniform}(0,2\pi)$ and $\mathrm{Uniform}(0,\pi)$ push forward to the identical $\cos\iota$ distribution.

Confirmed directly against the real generator, not the toy model: `pycbc.waveform.get_td_waveform(approximant='IMRPhenomD', ...)` gives bit-identical $h_+$/$h_\times$ at $\iota$ and $2\pi-\iota$ (`ml-gw-search`'s own `pycbc==2.9.0` venv, max difference $\approx10^{-34}$ against a signal scale of $\approx10^{-19}$) — the exact $\iota\leftrightarrow2\pi-\iota$ waveform degeneracy, universal for a non-precessing, dominant-$(2,2)$-mode-only signal (this dataset's generation, `004_methods.tex:5`), not a toy-model artifact.

### 4.2 This population over-represents face-on, not edge-on, relative to a truly isotropic one

For a truly isotropic 3D orbital orientation, $\cos\iota$ is uniform on $[-1,1]$, giving $\iota$'s own density $p(\iota)=\tfrac{1}{2}\sin\iota$ — zero at the poles (face-on), peaked at $\iota=\pi/2$ (edge-on).
This dataset's flat draw gives the opposite skew: `003_prereq_study.tex:27`'s population-balance numbers (28.7% face-on at $|\cos\iota|>0.9$, 32.7% edge-on at $|\cos\iota|<0.5$) compare to 10%/50% under true isotropy — roughly 3× the isotropic face-on fraction, under two-thirds the isotropic edge-on fraction.
`004_methods.tex` states this directly.

Face-on is where the $\varphi_c/\psi$ degeneracy is strongest (§3.5's preference ratio peaks $\approx1.6\times$ there); edge-on is where it's weakest.
So this skew could in principle nudge a *pooled* statistic — one averaged across the whole population, e.g. `006_results.tex`'s headline circular-loss numbers ($\comboA=0.9999$, $\comboB=0.9895$) — mildly toward looking like a null, by weighting more hard-to-recover cases into the average.
It cannot explain the null result itself: the inclination-band stratification (`\S\ref{sec:phicpsi:inclination}`) measures recovery *within* each band separately, so the population mix has no channel through which to act on it, and the edge-on band shows no recovery in isolation either.
This is a real limit on astrophysical realism (external validity), not evidence the null is a sampling artifact.

### 4.3 The inclination control head shows partial mode collapse, not a small real signal or a clean noise floor

Inclination is used as a same-model noise-floor calibration for judging $\varphi_c/\psi$'s own band-to-band deviations.
It is neither a clean noise floor nor a genuinely recoverable signal.
A predicted-vs-true scatter (`inclination_output/inclination_scatter_20260923_132228.{png,pdf}`) shows no diagonal trend toward either the true value or the exact $\iota\leftrightarrow2\pi-\iota$ mirror value (§4.1) in any of the four models.
Instead, predictions cluster into a handful of preferred angles largely independent of the true value — partial mode collapse, quantified by this head's own circular resultant ($\text{circ}_r=0.32$–$0.60$ across models: `poc_a` 0.603, `poc_b` 0.534, `tcn` 0.432, `cnn_attention` 0.317), well short of the $\text{circ}_r>0.9$ full-collapse grade used for $\varphi_c$/$\psi$ elsewhere, but far above the $\approx0$ that either clean noise or genuine recovery would give.
The ranking tracks each model's own $\varphi_c$/$\psi$ collapse severity exactly: `poc_a`/`poc_b` (COLLAPSE-graded on $\varphi_c$/$\psi$) show the highest inclination $\text{circ}_r$, `tcn`/`cnn_attention` (not collapsed there) the lowest — consistent with a shared training-dynamics cause across a model's periodic heads, not head-specific recoverable physics.

This does not threaten the paper's headline $\varphi_c$/$\psi$ null.
That result rests on four methods measured directly on $\varphi_c$/$\psi$'s own predictions — collapse/histogram inspection, the gradient-chain trace, the label-permutation bootstrap, and the SNR/inclination-band stratification — none of which consume inclination's own recoverability.
Inclination's role is always secondary/interpretive, and `008_discussion.tex`'s threats-to-validity list already reads §6.7 as descriptive rather than a formal calibration, independent of this finding.

The investigative trail behind §4.1–4.3 (the backwards over-representation claim, the wrong-support table, the pycbc verification steps, the Monte Carlo hypothesis that didn't survive contact with the actual scatter, and the discovery that the paper's prior "verified against scatter plots" claim had no artifact behind it) is recorded in `NOTES.md`'s 2026-09-23 "Paper-review correction pass" entry, not here — this section states what's currently true and why, not how it was found.

---

# Deferred

## P.1 $\pi/4$ / $3\pi/4$ reference-point plot annotations (removed 2026-09-22 11:20)

This describes a plot element that existed when this doc was first written and no longer does — kept for the record, not as a description of the current figure.
See Active §3.6 for what's still true (the underlying bootstrap computation and its numbers, which are unaffected by this).

`_plot_ratio_vs_iota` used to also take `results_by_sign` as a second argument and draw, on each panel, a dotted vertical line at the panel's reference $\iota$ plus a `matplotlib` `annotate()` call with a gray arrow, reading e.g. "$\iota=\pi/4$: $1.17\times$ ($\text{combo}_B$)" on the left panel and "$\iota=3\pi/4$: $1.19\times$ ($\text{combo}_A$)" on the right.
The dotted line itself carried no statistical meaning — it was purely a visual anchor pointing at the x-position the annotation text referred to, unlike the $y=1.0$ dashed line (which is the actual null reference).
The annotated ratio was not the sweep curve's own value at that $x$; it was the separately-bootstrapped reference-point number from §3.6, placed at that $x$-coordinate for the reader's convenience.

This was removed by editing `_plot_ratio_vs_iota` down to a single argument (`sweep_results`) and deleting the `axvline`/`annotate` calls on both panels.
If the annotation is ever wanted back on the figure, the reference numbers to draw it from are still computed and available via `results_by_sign` inside `_check_sign_combination` (§3.6) — nothing needs to be recomputed, only re-plotted.

## P.2 $y=1.0$ "no preference" line, legend, $y$-axis floor, and subtitle arrows (removed 2026-09-22 11:42)

Same status as §P.1: describes a plot presentation that existed earlier in this doc's own life and no longer does, kept for the record.
This was the second, separate round of simplification (after §P.1's), done deliberately so the figure needs no further editing before going into the paper — see Active §3.7 for the resulting bare figure.

Before this round, `_plot_ratio_vs_iota` also had, on each panel:

- `ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.75, label="no preference (1.0)")` — the dashed horizontal floor line (§3.5's mathematical/physical null, see the original explanation preserved here: dividing $\max(\text{mean}_A,\text{mean}_B)$ by $\min(\text{mean}_A,\text{mean}_B)$ can never go below 1, and $\text{mean}_A=\text{mean}_B$ is also the actual physical expectation at exact edge-on).
- `ax.legend(loc=...)` — a legend box, whose only entry was that line's `"no preference (1.0)"` label.
- `ax.set_ylim(0.9, None)` — a fixed lower $y$-limit, so the floor line always had headroom below it.
- Two-line titles with a reading-direction subtitle, e.g. `"cos ι > 0 \n(face-on ← ι=0,  ι=π/2 → edge-on)"`, instead of the current bare `r"$\cos\iota > 0$"`.
- Plain-text axis labels (`"ι [rad]"`, `"correlation ratio (well / poorly)"`) instead of the current LaTeX (`r"$\iota$ [rad]"`, `"Correlation ratio"`).

All of this was cut in the same pass, along with switching the function's `matplotlib` import from a function-local `matplotlib.use("Agg")` block to a module-level `import matplotlib.pyplot as plt`, and switching `save_fig`'s DPI from a hardcoded `300` to `SAVE_DPI` imported from `gwml.evaluation.plot_style` (`prereq_checks.py:39`) — a move toward the shared project plotting-style convention rather than a per-script constant.
The floor/null value itself is unchanged (§3.5); only its visual presentation on this specific figure is gone.

---

# Tooling

## T.1 The paper's notation-macro convention, and two `\ensuremath` pitfalls found while extending it

`../gw_research_paper/000_preamble.tex` centralizes every repeated math symbol, diagnostic-statistic name, and model/trunk identifier into one `\newcommand`, precisely so the paper can't drift into two spellings of the same thing across files the way it already had once: `combo \textbf{A}` in `003_prereq_study.tex` prose vs. `$\text{combo}_A$` in `004`/`005`/`006`'s math mode; `MAE_angular` (formally defined) vs. `ang_MAE` (used in `006`'s table); `cnn_attention` (prose) vs. `cnn_attn` (bare table cells).
Current inventory: `\comboA`/`\comboB`/`\comboAFull`/`\comboBFull` (via a shared `\combo{}` helper), `\circr`, `\stdratio`, `\MAEang`, `\Lcirc`, `\Lmag`, and `\pocA`/`\pocB`/`\Tcn`/`\cnnAttn`/`\cnnBaseline`/`\inceptionTime`/`\resnetOneD`.
The model-identifier macros mirror `../model_labels.py`'s `MODEL_LABELS` dict on the Python side — itself created to deduplicate a `model_label_dict` that had been hand-copied (and had already started drifting in trailing-comma style) between `diagnostic_checks.py` and `plot_certified_memorization.py` — so the same canonical name and casing is used whether a model is named in paper prose, a table cell, or a figure title.

Two of these decisions are worth recording because they reversed mid-stream, not because the final answer was obvious:

- **`std_ratio` vs. `\sigma_\text{head,target}`:** `002_problem-formulation.tex` formally defines `\sigma_\text{head,target}`, but `006`/`007`/`008`/`010` (6 occurrences) call the same diagnostic `std_ratio` and never use the `\sigma` symbol. Canonicalized on `std_ratio`, since that's the literal diagnostic-script/CSV column name (traceability to the artifact wins over the formal symbol); `002` keeps its `\sigma_\text{head,target}` definition but now notes it's "called `std_ratio` from §prereg onward."
- **Model-identifier rendering:** `\pocA`/`\Tcn`/etc. were first defined as `\texttt{poc\_a}` etc. — the literal lowercase code identifier, for direct grep-traceability to the actual config name. Later switched, on request, to render `model_labels.py`'s pretty display string instead (`POC_A`, `CNN$_\text{baseline}$`, `InceptionTime`, ...), so prose/tables read identically to figure titles/legends — at the direct cost of that `\texttt{poc\_a}` traceability. Both were reasonable; the paper currently uses the second.

### T.1.1 Two `\ensuremath` rules

1. **`\ensuremath` must be the only thing that ever inserts a `$`.** Hand-writing a literal `$...$` inside an `\ensuremath{...}` argument breaks math-mode tracking — the hand-written `$` toggles whatever mode `\ensuremath` already opened rather than adding "more math," which can silently corrupt otherwise-correct LaTeX into a cascading `! Missing $ inserted.` failure several tokens later. Wrap the whole expression in one `\ensuremath` call instead of nesting a raw `$...$` inside it.
2. **A bare multi-letter word inside math mode renders as italicized, implicitly-multiplied single letters, not the word** — `combo` without `\text{}` becomes five separate math-italic variables, *c·o·m·b·o*, silently, with no compile error to flag it. Same root cause as the original combo-A/B typography inconsistency that motivated this whole macro effort (§3.4) — exactly why `\circr`, `\stdratio`, `\Lcirc`, `\Lmag` all wrap their word parts in `\text{}`.

Minor, not a bug: nesting `\ensuremath` inside another macro that already calls `\ensuremath` is harmless (it checks `\ifmmode` and no-ops when already in math mode) but pointless, since that's the one thing `\ensuremath` already exists to decide.

The two bugs that taught these rules (the exact broken macro definitions, the token-by-token trace of why each one failed, and the actual `pdflatex` error output) are recorded in `NOTES.md`'s 2026-09-23 tooling entry, not here.
