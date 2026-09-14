# φc/ψ Degeneracy PoC — Redo — Formulae Reference

Companion to [`redo_procedure.md`](redo_procedure.md). Use these formulae exactly as given — do not re-derive signs/conventions ad hoc mid-implementation, same discipline the original plan's Appendix A required (`planning_files/phic_psi_implementation_plan_v4.md` §7). Physics/loss math is given in full; model architectures (§B) are bare outlines only, by design — no per-layer derivations here.

## A. Physics and loss formulae

### A.1 Vector algebra primitives — unchanged from the original

A `PERIODIC` head's raw vector is `z = (s, c)` where `s = sin(θ)`, `c = cos(θ)` — equivalently a unit complex number `z = c + i·s`.

```
complex_mul(z1, z2)       -> angle (θ1+θ2):
    s_out = s1*c2 + c1*s2
    c_out = c1*c2 - s1*s2

complex_mul_conj(z1, z2)  -> angle (θ1−θ2):
    s_out = s1*c2 - c1*s2
    c_out = c1*c2 + s1*s2

normalize_unit(z):
    r = sqrt(s*s + c*c + eps)      # eps = 1e-8
    return (s/r, c/r)
```

### A.1.5 Angle doubling — new for this redo

```
double_angle(z) = complex_mul(z, z)   # angle 2*theta
```

Self-multiplying a unit vector by itself doubles its angle: `complex_mul((s,c), (s,c))` gives `s_out = 2sc = sin(2θ)`, `c_out = c²−s² = cos(2θ)`. Used to turn `z_φc` (angle `φc`, from the unchanged period-2π `coa_phase` head) into a `2φc`-angle vector before combining with `z_ψ`.

### A.2 Combo construction — corrected

```
z_2phic  = double_angle(z_phic)              # angle = 2*phic
combo_A  = complex_mul(z_2phic, z_psi)       # angle = 2*phic + 2*psi
combo_B  = complex_mul_conj(z_2phic, z_psi)  # angle = 2*phic - 2*psi
```

`z_ψ` already encodes `2ψ` internally — `polarization_angle` is declared `PERIODIC` with `period=π` in `src/gwml/heads_spec.py`, unchanged, and `PERIODIC.transform_head` rescales by `2π/period` before taking sin/cos, so the stored vector is `(sin(2ψ), cos(2ψ))` by construction. Do not rescale again.

Which of `combo_A`/`combo_B` is well-constrained is determined empirically (`redo_procedure.md` §1.1) — do not assume it matches the original's `combo_B`-well-constrained finding, which was for the wrong pair.

### A.3 Isotropic circular loss — unchanged

For normalized predicted `z_pred=(s_p,c_p)` and true `z_true=(s_t,c_t)` (true vectors are unit-norm by construction):

```
dot = s_p*s_t + c_p*c_t     # = cos(theta_pred - theta_true)
L   = 1 - dot
```

Exact, provably isotropic — a function of angular difference only. This formula was never implicated in the combo-formula bug; it operates correctly regardless of which vectors it's given.

### A.4 Magnitude penalty — unchanged

```
magnitude_penalty = lambda * (|v_raw| - 1)^2
```

applied to each *raw* (pre-`normalize_unit`) `coa_phase`/`polarization_angle` prediction, summed, scaled by `lambda` (0.01 from day 1 for this redo — see `redo_procedure.md` §2.5 for why this isn't rediscovered from λ=0). Prevents `normalize_unit`'s `÷|v|` backward pass from crushing/exploding the angular gradient when `|v|` drifts away from 1.

### A.5 Toy antenna-pattern / detector-response model — corrected

```
k1(iota) = (1 + cos(iota)^2) / 2                       # h_plus coefficient
k2(iota) = cos(iota)                                    # h_cross coefficient

h_plus(phi_c, iota, Phi)  = k1(iota) * cos(2*Phi + 2*phi_c)
h_cross(phi_c, iota, Phi) = k2(iota) * sin(2*Phi + 2*phi_c)

F_plus(psi, a, b)  = a*cos(2*psi) + b*sin(2*psi)         # unchanged
F_cross(psi, a, b) = b*cos(2*psi) - a*sin(2*psi)         # unchanged

detector_signal(psi, phi_c, iota, a, b, Phi)
    = F_plus(psi,a,b) * h_plus(phi_c,iota,Phi)
    + F_cross(psi,a,b) * h_cross(phi_c,iota,Phi)
```

`Phi` is orbital phase; `2*Phi` is GW phase for the dominant (2,2) mode. `h_plus`/`h_cross` previously added `phi_c` singly to `2*Phi` — inconsistent with the fact that `Phi` is explicitly orbital-frame while `phi_c` (as actually injected, see A.6) is also orbital-frame and must be doubled to enter the (2,2)-mode phase, same as the `2*Phi` term itself.

`R, δ` are recovered by projecting `detector_signal` onto its `2*Phi`-frequency component (mechanically unchanged from the original):

```
c_proj = sum(detector_signal * cos(2*Phi))
s_proj = sum(detector_signal * sin(2*Phi))
R      = hypot(c_proj, s_proj)
delta  = atan2(-s_proj, c_proj)
```

**Required self-check** before trusting anything downstream (same discipline as the original harness check): sweep `psi` while holding `2*phi_c + 2*psi = const`, confirm `(R, δ)` stays exactly constant at `ι=0`. If it doesn't, the formula fix has a bug — stop and fix before running `derive_w_iota`.

**Verified by hand (2026-09-14 review):** at ι=0, `k1=k2=1`, so `detector_signal = a·cos(Θ+2ψ) + b·sin(Θ+2ψ)` where `Θ=2Φ+2φc` — the signal depends on `(φc,ψ)` only through `Δ ≡ 2φc+2ψ`, exactly the invariance the self-check demands. Confirmed passing.

**Aside, not a bug:** this toy `F_plus`/`F_cross` pair transforms as `e^{+2iψ}` rather than `e^{-2iψ}` — opposite handedness from some conventions in the literature — but `k1/k2 = (1+cos²ι)/(2cosι)` still matches the true physical amplitude ratio exactly, and the self-check above passes regardless of handedness. Nothing to fix; noted for anyone cross-checking against a different sign convention later.

### A.6 Injection-convention note — new, the evidence for A.5's correction

`coa_phase` is injected via `pycbc.waveform.get_td_waveform(coa_phase=...)` (confirmed in `src/gwml/gen_py_data_pipeline.md`'s documented `gen.py`, and structurally identical in the sibling generator scripts `ml-gw-search/mlgwsc-1/gen.py:202` and `ml-gw-search/extended_mass/gen.py:179`, both of which set `waveform_kwargs['coa_phase'] = angles[0]` and pass it straight to `get_td_waveform`). PyCBC's `coa_phase` is LALSimulation's `phiRef` — the **orbital** reference phase, range `[0, 2π]`. The approximant is IMRPhenomD, dominant (2,2) mode only (confirmed, `experiments/phic_psi_poc/NOTES.md` v2 adversarial-review section). For a pure (2,2)-mode signal, the strain depends on the orbital reference phase through `e^{i·2·coa_phase}` — the standard "waveform invariant under `coa_phase → coa_phase + π`" result — so `coa_phase`, as actually injected in this dataset, enters the waveform doubled, exactly like `ψ`. Full chain: `experiments/phic_psi_poc/diagnostic_log.md`, dated entry 2026-09-09.

### A.7 Curriculum weight w(ι) — corrected

Same recipe as the original, re-run against the corrected A.5:

1. Sweep `ι` over `[0.001, π/2−0.001]`.
2. At each `ι`, average the condition number of the Jacobian of `(φc,ψ) → (R,δ)` (A.5's map) over `n_sky_samples` random sky-position coefficient pairs `(a,b)`.
3. `w_raw = (1/condition_number) / max(1/condition_number)`.
4. Fit `w_raw` via linear interpolation on `cos²ι` (not a polynomial — the original explicitly rejected a polynomial fit for a bad residual, that decision is unrelated to the combo-formula bug and still holds).

**Do not carry forward the original's `w = 1 − cos²ι` default or its empirical-fit numbers** — both were derived from the uncorrected A.5 antenna-pattern model and must be re-derived from the corrected one before use.

### A.8 Reconstruction branch-handling — derived and verified (2026-09-14)

Let `U = 2φc` (φc's own physical period is 2π, so U ranges over [0,4π) as φc completes one cycle) and `V = 2ψ` (ψ's own physical period is π, so V ranges over exactly [0,2π) as ψ completes one cycle — V alone is already a *bijection* onto ψ's full physical range, no ambiguity from the doubling itself). `combo_A ≡ U+V` and `combo_B ≡ U−V` (both mod 2π, since combo_A/B are unit-vector angles).

**Recovering ψ (2-fold):**
```
2V ≡ combo_A − combo_B  (mod 2π)   ⇒   V ≡ ½(combo_A − combo_B)  (mod π)
```
Knowing `2V mod 2π` only pins `V` down mod π — a 2-fold ambiguity in V's own [0,2π) range (`V₀`, `V₀+π`). Since `V=2ψ` is already bijective onto ψ's range, this maps 1:1 to a **2-fold** ambiguity in ψ: candidates `ψ₀`, `ψ₀+π/2`.

**Recovering φc (4-fold):**
```
2U ≡ combo_A + combo_B  (mod 2π)   ⇒   U ≡ ½(combo_A + combo_B)  (mod π)
```
Same logic gives `U` mod π, i.e. `2φc` mod π, i.e. `φc` mod π/2 — a **4-fold** ambiguity in φc's [0,2π) range: candidates spaced π/2 apart, `φc₀ + k·π/2` for `k∈{0,1,2,3}`.

**The two ambiguities are not independently combinable — this is the part that matters for the reconstruction code.** Parametrize `φc = φc₀+k·π/2` (k=0..3), `ψ = ψ₀+j·π/2` (j=0,1). Re-encoding via A.2:
```
combo_A_candidate = combo_A_true + (k+j)π   (mod 2π)
combo_B_candidate = combo_B_true + (k−j)π   (mod 2π)
```
Both vanish (reproduce the true pair) iff `(k+j)` is even — and `(k+j)` and `(k−j)` always share parity (they differ by `2j`), so **one** parity check (`k≡j mod 2`) gates both conditions simultaneously. This keeps exactly 4 of the naive `4×2=8` `(k,j)` combinations — not an independent 4-fold×2-fold product.

**Implication for the reconstruction code:** write the parity rule (`k≡j mod 2`) directly into candidate generation as the fast path — it's a closed-form filter, not something to brute-force. Still keep the original's re-encode-and-check-against-both-combos filter as a validation cross-check alongside it (cheap, and catches any implementation bug in the parity shortcut itself). As in the original, this is a validation-script technique requiring ground truth to disambiguate the surviving 4 candidates — not a real-inference reconstruction method.

## B. Model architecture — bare outline only

No per-layer formulae here by design — see `src/gwml/models.py` for actual layer definitions.

### B.1 Trunk options

| Trunk | One-line description |
|---|---|
| `tcn` | Temporal convolutional network — current best performer per the original multi-architecture comparison. |
| `cnn_baseline` | Plain 1D CNN stack, no attention. |
| `cnn_attention` | 1D CNN with a learned attention-pooling head. |
| `inception_time` | InceptionTime-style multi-scale 1D conv blocks. |
| `resnet1d` | 1D ResNet-style residual conv stack. |

### B.2 Head layer

Small dense head per `HeadSpec` (`src/gwml/heads_spec.py`), `hidden_units=64`, `bounded=true` where applicable (per the original configs' `head_cfg`). Transform-kind table (name / column / transform / period / dim), unchanged from the original — this redo touches none of it:

| Head | Column | Transform | Period | Dim |
|---|---|---|---|---|
| `mchirp` | 0 | LOG_ZSCORE | — | 1 |
| `q` | 1 | UNIT_AFFINE | — | 1 |
| `inclination` | 2 | PERIODIC | 2π | 2 |
| `coa_phase` | 3 | PERIODIC | 2π | 2 |
| `polarization_angle` | 4 | PERIODIC | π | 2 |
| `sky_position` | (5,6) | SPHERICAL_UNIT_VECTOR | — | 3 |
| `merger_time` | 8 | UNIT_AFFINE | — | 1 |
| `snr` | 9 | ZSCORE | — | 1 |

### B.3 `SumDiffTrainer` (redo version)

Wraps the parent `MultiHeadTrainer` (shared, unmodified) and swaps in the combo-level circular loss (A.3) on `combo_A`/`combo_B` from A.2, plus the magnitude penalty (A.4) on the raw `coa_phase`/`polarization_angle` outputs, plus the curriculum weight (A.7) applied to whichever combo Step 1.1 finds poorly-constrained. See `redo_procedure.md` §2.3–2.5 for the construction order and rationale — this section intentionally doesn't repeat that math, only names what the trainer wraps.
