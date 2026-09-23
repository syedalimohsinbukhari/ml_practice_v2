# Understanding — working notes on the $\varphi_c/\psi$ redo

This is a growing reference doc, not a narrative log: a running collection of the intricacies, code mechanics, and "wait, why is it built that way" questions that come up while walking through the paper and thesis, written down once so they don't get re-derived from scratch each time.
It has three tracks, each with its own numbering so none has to be renumbered when the others grow:

- **Active** topics, numbered `N.M` — things currently live in the paper/thesis/code as written. `N` matches the paper's own section-file number in `paper/` (e.g. §2 ↔ `002_problem-formulation.tex`, §3 ↔ `003_prereq_study.tex`), so the doc section a topic sits under tells you which paper file it documents; `M` is just that topic's own sub-numbering.
- **Deferred** topics, numbered `P.Q` — discussions that did happen, but whose subject is no longer current (a plot element that got removed, an approach that got superseded, etc.). Kept rather than deleted, so the earlier discussion isn't silently lost, but clearly marked as not describing present-day artifacts. Deliberately *not* tied to a paper section number, since by definition they no longer describe what the paper currently says.
- **Tooling** topics, numbered `T.Q` — lessons about the writing/build infrastructure itself (LaTeX macro conventions, `\ensuremath` pitfalls, the paper/Python notation-consistency setup), not about the paper's scientific content. Also not tied to a paper section number, for the opposite reason from Deferred: these are evergreen craft notes that apply across every section rather than documenting one of them.

A topic moves from Active to Deferred in place when its subject stops being current — see §3.6/§P.1 below for the first example.
Tag any such in-place change note with a `date: time` stamp (`YYYY-MM-DD HH:MM`, e.g. `**Change (2026-09-22 11:20):**`) rather than date alone, so multiple same-day changes stay distinguishable.

## Topics so far

**Active**
2. The circular (cosine) loss and unit-circle normalization (`transform_utils.py`, `trainer.py`, `002_problem-formulation.tex`)
3. Step 1.1 prerequisite-check math and the $\iota$-sweep plot (`prereq_checks.py`, `curriculum.py`, `003_prereq_study.tex`)
4. The dataset's inclination prior: face-on/edge-on over-representation direction, and $\iota$'s true $[0,2\pi)$ support plus the resulting inclination-head degeneracy (`004_methods.tex`, `003_prereq_study.tex`, `thesis/chapter_phic_psi_degeneracy.{md,tex}`, `inclination_stratification.py`, `ml-gw-search/mlgwsc-1/gen.py`)

**Deferred**
- P.1. $\pi/4$ / $3\pi/4$ reference-point plot annotations (removed from the figure 2026-09-22 11:20)
- P.2. $y=1.0$ "no preference" line, legend, $y$-axis floor, and face-on/edge-on subtitle arrows (removed from the figure 2026-09-22 11:42, so it can go straight into the paper without clutter)

**Tooling**
- T.1. The paper's notation-macro convention (`paper/000_preamble.tex`), and two `\ensuremath` pitfalls found while extending it

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

## 4. The dataset's inclination prior and the face-on/edge-on over-representation direction

### 4.1 $\iota\sim\mathrm{Uniform}[0,\pi]$ over-represents face-on, not edge-on, relative to an isotropic population

**Status: resolved 2026-09-23.**
Flagged to the user, then fixed the same day in `004_methods.tex:37` and mirrored into `thesis/chapter_phic_psi_degeneracy.md` and `.tex` (§8.3's "Threats to validity" bullet, which stated the identical backwards claim independently of §4/§8.3 in the paper).

For a truly isotropic 3D orbital orientation, $\cos\iota$ is uniform on $[-1,1]$, so $\iota$'s own density is $p(\iota)=\tfrac{1}{2}\sin\iota$.
That density is zero at the poles ($\iota=0,\pi$, face-on) and peaks at $\iota=\pi/2$ (edge-on).
This dataset instead draws $\iota\sim\mathrm{Uniform}[0,\pi]$ directly (`004_methods.tex`'s population table, "Uniform in $\iota$") — flat, with no $\sin\iota$ weighting at all.

Checking the direction against numbers already stated elsewhere in this same paper (`003_prereq_study.tex:27`, "Population balance: 28.7% ... near-face-on ($|\cos\iota|>0.9$), 32.7% near-edge-on ($|\cos\iota|<0.5$)") against what a true isotropic population gives in the identical bins (both computed directly from $\cos\iota$ being uniform on $[-1,1]$):

| Bin | This dataset | True isotropic |
|---|---|---|
| Face-on, $\|\cos\iota\|>0.9$ | 28.7% | 10% |
| Edge-on, $\|\cos\iota\|<0.5$ | 32.7% | 50% |

The dataset has roughly 3× the isotropic-predicted face-on fraction and well under two-thirds of the isotropic-predicted edge-on fraction.
Flat-$\iota$ over-represents face-on and under-represents edge-on, relative to isotropic truth — the direction is unambiguous from the paper's own already-published numbers, no external reference population needed.

`004_methods.tex:37` currently states the opposite: "over-representing edge-on systems relative to a real population."

Why the direction matters, not just the label: face-on is where the $\varphi_c/\psi$ degeneracy is strongest (§3.5's preference ratio peaks $\approx1.6\times$ there); edge-on is where it's weakest (ratio $\to1.0\times$).
If the dataset over-represents face-on — the hardest-to-recover regime — as the numbers above show, that pads the population with genuinely-degenerate cases, which makes a null (non-recovery) result *easier* to obtain, not harder.
That is the opposite of `004_methods.tex:37`'s own stated conclusion, that this "does not bias the null toward a self-serving conclusion."
As currently worded, the sentence's own stated direction would argue the reverse of what it concludes.

One live caveat, unresolved: this compares against the pure geometric isotropic prior.
If "a real population" was meant to reference a detection-selected (SNR-weighted) population instead — face-on binaries are intrinsically louder, so *detected* populations skew more face-on than the geometric prior even before this dataset's own flat-$\iota$ choice is layered on top — the comparison could shift, but that would be a different, so-far-unstated claim needing its own justification and citation, not something derivable from what's currently in the paper.

**Does the corrected direction mean the dataset is biased in a way that threatens the null?**
No — this is a real limit on astrophysical realism (external validity: this population doesn't match how inclinations are actually distributed in nature), not evidence the null itself is an artifact of a skewed sample.

**"Mildly favors a null result" and "the stratification shows this isn't the case" are not in tension — they answer two different questions, about two different kinds of statistic, not the same one.**
"Mildly favors" is a claim about a *pooled* statistic: one averaged across the whole (skewed) population, like `006_results.tex`'s headline circular-loss numbers ($\comboA=0.9999$, $\comboB=0.9895$).
For a pooled average, sample composition genuinely can matter in principle — mix in more hard-to-recover (face-on) cases and the average tilts a little toward "looks unrecoverable," purely as a compositional artifact, independent of whether recovery is actually possible.
That's a real, honest concession about the pooled number, not something to wave away.

The inclination-band stratification (`\S\ref{sec:phicpsi:inclination}`) is a structurally different check: it does not pool.
It measures recovery *within* each inclination band separately (face-on against face-on, edge-on against edge-on), so the population *mix* — how many face-on vs. edge-on samples exist overall — has no channel through which to act on a within-band result.
Finding no recovery in the edge-on band, examined in isolation, isn't "proof the skew doesn't exist"; the skew is exactly as described.
It's proof that whatever the skew's compositional effect might be doing to the *pooled* headline number, the actual finding doesn't depend on it, since the same null appears even in the one subpopulation where that compositional effect had zero opportunity to contribute.
Belt-and-suspenders: a theoretical concession about what a pooled metric *could* be sensitive to, plus the empirical check that shows it isn't actually the explanation — not a contradiction between "biased" and "unbiased."

**Fix applied, 2026-09-23 (two passes).**
First pass corrected the direction: `004_methods.tex:37` moved from the backwards "over-representing edge-on... does not bias the null toward a self-serving conclusion" to "over-representing face-on... this skew mildly favors a null result."
Second pass made the pooled-vs-stratified distinction explicit, since the first pass's "mildly favors... however, the null itself does not depend on this" reads as a flat contradiction without it: `004_methods.tex` now names the headline numbers of `\S\ref{sec:phicpsi:results}` explicitly as the *pooled* statistic the skew could in principle nudge, and states that population mix "has no opportunity to act" on the *band-separated* stratification check, rather than asserting the two facts side by side with no mechanism connecting them.
The same two-pass fix was mirrored into `thesis/chapter_phic_psi_degeneracy.md` and `.tex` §8.3, which had independently stated the identical backwards claim (not copied from the paper — restated separately, so it needed its own fix).
Repo-wide sweep for the same claim (`grep -rl "over-representing edge-on"`) found two more hits, both correctly left alone: `review_inheritance_map.md:86` only reports the raw 28.7%/32.7% numbers without asserting a direction, so it was never wrong; `experiments/phic_psi_poc/thesis/reviews/kimi_ai_adversarial_review_v2.md:69` is a frozen adversarial-review record quoting the *original* (pre-redo) chapter's identical claim back at it — a dated historical snapshot per this repo's frozen-vs-living-docs convention, never rewritten after the fact regardless of whether the claim it quotes was right.

### 4.2 The population table's $\iota$ support was also wrong ($[0,\pi]$ vs. the actual $[0,2\pi)$) — and the deeper question it raised

**Status: table fixed 2026-09-23; structural finding confirmed, does not change any reported number.**

While building the §4.1 illustration, the histogram of the real `inclination` column (`combined_repackaged.hdf`, `params[:, 2]`) turned out to span the full $[0,2\pi)$ — min $\approx0$, max $\approx2\pi$, exactly half the 30,000 samples $>\pi$ — not $[0,\pi]$ as `004_methods.tex`'s population table stated and as §3.1 above assumed ("$\iota$'s range is $[0,\pi]$ because that is what 'angle between two vectors' means").
Traced to the actual generator: `ml-gw-search/mlgwsc-1/gen.py:201`, `angles = np_gen.uniform(0., 2*np.pi, 3)`, draws `coa_phase`, `inclination`, and `polarization_angle` identically, all as period-$2\pi$ quantities — matching `src/gwml/heads_spec.py:99-100`'s `TransformKind.PERIODIC, period=_TWO_PI` for the inclination head.
**Fixed:** `004_methods.tex`'s table now reads `$\iota$ | $[0,2\pi)$ | Uniform in $\iota$`.

**Does this affect the φ_c/ψ null, or anything else load-bearing?**
No, and this is provable, not just plausible.
Every piece of machinery that trains on $\iota$ — the curriculum weight $w(\iota)=\sin^2\iota$, the analytic $k_1(\iota)=(1+\cos^2\iota)/2$/$k_2(\iota)=\cos\iota$ coefficients (§3.2/§3.3), the prereq sweep, the combo_A/combo_B construction — depends on $\iota$ *only* through $\cos\iota$, never on $\iota$ itself.
$\cos(\cdot)$ is $2\pi$-periodic and even, so $\mathrm{Uniform}(0,2\pi)$ and $\mathrm{Uniform}(0,\pi)$ push forward to the *identical* distribution of $\cos\iota$ — not approximately, exactly — which is exactly why §4.1's 28.7%/32.7% cross-check matched the paper's own numbers to the decimal regardless of which range was assumed.

**A real hypothesis this raised, investigated and only partly confirmed: does the inclination *control head* have its own structural degeneracy?**
The inclination head regresses the raw $(\sin\iota,\cos\iota)$ two-vector from the true stored $\iota$ (`heads_spec.py`'s `PERIODIC` transform is literally "angle $\to$ (sin, cos) of $2\pi\cdot\text{value}/\text{period}$", no folding).
`inclination_stratification.py:99-114`'s `angular_mae` confirms the evaluation side doesn't special-case this either — `("inclination", 2*np.pi, np.pi/2)`, full-period wrap-aware error, same treatment as `coa_phase`.
So if the strain only ever carries information about $\cos\iota$, the sign of $\sin\iota$ (i.e. which of the two $\iota\leftrightarrow2\pi-\iota$ labels was recorded) should be exactly as unrecoverable as $\varphi_c$/$\psi$ are — a second, independent instance of the same class of problem, hiding in what the paper treats as a clean positive control.

This is standard, well-established GW physics for a non-precessing, dominant-$(2,2)$-mode-only signal (exactly this dataset's generation, `004_methods.tex:5`) — the $(2,\pm2)$ spherical-harmonic angular dependence gives $h_+\propto(1+\cos^2\iota)/2$, $h_\times\propto\cos\iota$ and nothing else, universally, not a toy-model approximation.
But rather than trust that from memory, it was checked directly against the *real* generator: `ml-gw-search`'s own `.venv` (confirmed present, `pycbc==2.9.0`) was used to call `pycbc.waveform.get_td_waveform` with `approximant='IMRPhenomD'`, identical `mass1/mass2/coa_phase/delta_t/f_lower` to `gen.py`'s own call, once at `inclination=1.0` and once at `inclination=2π-1.0`.
Result: `max|h_plus_1 - h_plus_2| ≈ 1.4e-34`, `max|h_cross_1 - h_cross_2| ≈ 1.9e-34`, against a signal scale of `≈4.2e-19` — floating-point-identical, 15 orders of magnitude below the signal itself, on the actual production waveform generator, not the toy model.
The structural degeneracy is real and exact, confirmed empirically, not merely argued analytically.

**But it does not explain the inclination effect size already reported in `006_results.tex`/§6.7 — checked quantitatively, and it doesn't fit.**
A Monte Carlo idealization (true $\iota\sim\mathrm{Uniform}(0,2\pi)$, a predictor that recovers $\cos\iota$ perfectly but guesses $\sin\iota$'s sign at random) gives an expected wrap-aware angular MAE of $\approx0.785\,\text{rad}$ ($=\pi/4$ exactly, i.e. half the $\pi/2$ null) — a large, easily-detected improvement over the null.
The paper's own reported inclination effect is $\approx0.02$–$0.04\,\text{rad}$ *above* the null, roughly 20–40$\times$ smaller than what "good $\cos\iota$ recovery, random sign" would produce.
So whatever mechanism drives §6.7's small real signal, it is not "the network learned $\cos\iota$ well and is merely capped by the unrecoverable sign" — that story predicts an effect far larger than what's observed.
The structural degeneracy is a genuine, previously-undocumented ceiling on how good the inclination head could ever score (nothing better than $\approx0.785\,\text{rad}$ MAE is achievable even in the idealized best case), and it reinforces *why* §6.7 already hedges inclination as "not a clean noise floor" rather than a fully clean positive control — but it is not, by itself, the explanation for the specific small magnitude already reported there.
That magnitude remains whatever §6.7 already says it is: a small, real, sub-materiality-floor signal of separately-undetermined origin.

**Not yet done, flagged for a decision:** none of this has been added to the paper itself (no changes beyond the table's support column) — whether the structural-degeneracy finding belongs as an additional caveat in `004_methods.tex` (near the inclination sentence) or `006_results.tex` §6.7 (near "not a clean noise floor") is an open call, not made unilaterally here.

**Follow-up, 2026-09-23: prepared, not yet run, since this needs the GPU machine.**
The quantitative Monte Carlo argument above is still an aggregate-statistic argument, and this repo's own house rule (`008_discussion.tex`: *"the corrective... is mechanism inspection — a scatter plot, not a summary statistic"*) applies here just as much as it did to `std_ratio`.
`inclination_stratification.py` was extended to also save a per-model predicted-vs-true inclination scatter, `inclination_output/inclination_scatter_<timestamp>.{png,pdf}`, colored by face-on/mixed/edge-on band, with two reference lines drawn: $y=x$ (perfect recovery) and $y=2\pi-x$ (the exact waveform-degeneracy line confirmed above via direct `pycbc.waveform.get_td_waveform` comparison).
If the small ang_MAE gap already on record is real partial $\cos\iota$ recovery, points should show *some* visible pull toward one or both lines rather than filling the square uniformly; if it's noise that happens to average out slightly below null, the scatter will look uniform regardless of what the aggregate number says.
Not run locally (needs the trained checkpoints + GPU, per this repo's CPU-only rule) — queued for the lab GPU machine.

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

`paper/000_preamble.tex` centralizes every repeated math symbol, diagnostic-statistic name, and model/trunk identifier into one `\newcommand`, precisely so the paper can't drift into two spellings of the same thing across files the way it already had once: `combo \textbf{A}` in `003_prereq_study.tex` prose vs. `$\text{combo}_A$` in `004`/`005`/`006`'s math mode; `MAE_angular` (formally defined) vs. `ang_MAE` (used in `006`'s table); `cnn_attention` (prose) vs. `cnn_attn` (bare table cells).
Current inventory: `\comboA`/`\comboB`/`\comboAFull`/`\comboBFull` (via a shared `\combo{}` helper), `\circr`, `\stdratio`, `\MAEang`, `\Lcirc`, `\Lmag`, and `\pocA`/`\pocB`/`\Tcn`/`\cnnAttn`/`\cnnBaseline`/`\inceptionTime`/`\resnetOneD`.
The model-identifier macros mirror `../model_labels.py`'s `MODEL_LABELS` dict on the Python side — itself created to deduplicate a `model_label_dict` that had been hand-copied (and had already started drifting in trailing-comma style) between `diagnostic_checks.py` and `plot_certified_memorization.py` — so the same canonical name and casing is used whether a model is named in paper prose, a table cell, or a figure title.

Two of these decisions are worth recording because they reversed mid-stream, not because the final answer was obvious:

- **`std_ratio` vs. `\sigma_\text{head,target}`:** `002_problem-formulation.tex` formally defines `\sigma_\text{head,target}`, but `006`/`007`/`008`/`010` (6 occurrences) call the same diagnostic `std_ratio` and never use the `\sigma` symbol. Canonicalized on `std_ratio`, since that's the literal diagnostic-script/CSV column name (traceability to the artifact wins over the formal symbol); `002` keeps its `\sigma_\text{head,target}` definition but now notes it's "called `std_ratio` from §prereg onward."
- **Model-identifier rendering:** `\pocA`/`\Tcn`/etc. were first defined as `\texttt{poc\_a}` etc. — the literal lowercase code identifier, for direct grep-traceability to the actual config name. Later switched, on request, to render `model_labels.py`'s pretty display string instead (`POC_A`, `CNN$_\text{baseline}$`, `InceptionTime`, ...), so prose/tables read identically to figure titles/legends — at the direct cost of that `\texttt{poc\_a}` traceability. Both were reasonable; the paper currently uses the second.

### T.1.1 The two `\ensuremath` bugs

Found while hand-editing `\combo`/`\comboA`/`\comboB`/`\comboAFull`/`\comboBFull` (`000_preamble.tex:22-26`) to learn `\ensuremath`, both in the same edit:

1. **A hand-written `$...$` inside an `\ensuremath{...}` argument breaks math-mode tracking.**
   Broken: `\newcommand{\comboAFull}{\ensuremath{\comboA\ $(2\varphi_c+2\psi)$}}`.
   `\ensuremath{#1}` checks `\ifmmode`: if not already in math mode, it wraps `#1` in `$...$` itself.
   If `#1` also contains a hand-written `$`, that `$` doesn't add "more math" — it toggles whatever mode is currently active.
   Trace: `\ensuremath` opens math with its own `$`; `\comboA` is processed fine (already in math mode, no new `$` needed); the body's own `$` then **closes** that math, since a `$` while already in math mode closes rather than opens; `(2\varphi_c+2\psi)` is then typeset in plain text mode, where `\varphi_c` isn't a valid command; the body's second `$` reopens math; `\ensuremath`'s own closing `$` closes it again.
   Actual compile result: `! Missing $ inserted.` at every `\comboAFull`/`\comboBFull` call site, cascading errors through the rest of the document (verified — `pdflatex` exits 1).
   Rule: `\ensuremath` must be the *only* thing that ever inserts a `$`; never hand-write `$` inside its argument — wrap the whole expression in one `\ensuremath` instead.
   Fixed: `\newcommand{\comboAFull}{\ensuremath{\comboA\ (2\varphi_c+2\psi)}}`.

2. **A bare multi-letter word inside math mode renders as italicized, implicitly-multiplied single letters, not the word.**
   Broken: `\newcommand{\combo}[1]{\ensuremath{combo_\text{#1}}}` — `\text{}` wraps `#1` (e.g. "A") but not `combo` itself.
   A bare `combo` inside math mode is parsed as five separate one-letter variables (c, o, m, b, o), each set in math italic and implicitly multiplied — renders as *c o m b o*, not the upright word "combo."
   Silent, not a compile error, which makes it easy to miss.
   This is the same root cause as the original combo-A/B typography inconsistency that motivated this whole macro effort in the first place (§3.4) — exactly why `\circr`, `\stdratio`, `\Lcirc`, `\Lmag` all wrap their word parts in `\text{}`.
   Fixed: `\newcommand{\combo}[1]{\ensuremath{\text{combo}_\text{#1}}}`.

Minor, not a bug: `\comboA` was originally `\ensuremath{\combo{A}}`, double-wrapping since `\combo` already calls `\ensuremath` itself.
Harmless — `\ensuremath` checks `\ifmmode` and no-ops (just emits its argument) when already inside math mode, so the inner call passes through — but pointless: `\ensuremath`'s entire purpose is to be the one place that decides whether a `$` is needed, so anything that already routes through it doesn't need wrapping again.
Simplified to `\newcommand{\comboA}{\combo{A}}`.
