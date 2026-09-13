# φc / ψ Degeneracy — Story So Far & Vector-Based Proof-of-Concept

Kept separate from `q_head_action_plan.md` and the `sky_position` vMF work —
this is its own head-design question, currently at the "does the math check
out" stage, not yet implemented.

Status: **theoretical / proof-of-concept only.** `polarization_angle` (ψ) is
not currently in `DEFAULT_HEADS` — it isn't being trained at all right now.
`coa_phase` (φc) is active as a standalone `PERIODIC` head.

---

## 1. The physical story

For a compact binary waveform dominated by the quadrupole (ℓ=2, m=±2) mode —
the standard assumption unless higher-order modes are explicitly included —
the coalescence phase φc and polarization angle ψ do not enter the strain
independently. For a near-face-on (or face-off) system, the waveform depends
on them only through the combination:

```
2ψ + φc          (up to an overall sign convention, see Sec. 4)
```

The orthogonal combination is weakly constrained or fully degenerate for
face-on/off systems, and only becomes partially recoverable via subdominant
effects at more edge-on inclinations. This is a well-established result in
GW parameter estimation (see prior literature search: near-face-on binaries
have ψ and φc individually unmeasurable, only their combination is).

**Consequence for training:** predicting φc and ψ as two independent heads
(the "obvious" setup) forces each head's loss to fight over both the
well-identified combination *and* the poorly-identified one simultaneously —
neither head can be cleanly up- or down-weighted, because both mix
information from both directions.

## 2. What we're trying to do about it

Rotate the target representation so the well-identified combination and the
poorly-identified combination become **two separate heads**, each with its
own loss weight/clamp — mirroring the fix already applied to `q` (Phase 2,
`log_var_clamp` per-head override). If the poorly-identified head is
confirmed to carry near-zero real signal, it can be clamped low (like `q`
was) so it doesn't drag gradient away from the head that matters, or dropped
from training entirely once confirmed uninformative.

## 3. Why this must be done in vector space, not raw-angle arithmetic

Two problems with naive `φc + ψ` / `φc - ψ` on raw angles:

1. **Period mismatch.** `coa_phase` has period 2π; `polarization_angle` has
   period π (per the existing `heads_spec.py` docstring: "the strain is
   invariant under ψ → ψ + π"). Raw addition of two angles with different
   periods doesn't live cleanly on a single well-defined circle.
2. **Wraparound discontinuity.** Any raw-angle arithmetic reintroduces the
   exact wrap-boundary problem the `PERIODIC` transform (sin/cos encoding)
   was built to avoid in the first place.

**The fix: represent each angle as a unit vector (already partially true —
see below) and use complex multiplication instead of angle addition.**

- `PERIODIC` already encodes each angle as `(sin θ, cos θ)`, i.e. a 2D point
  on the unit circle — equivalently, a unit complex number
  `z = cos θ + i sin θ`.
- **Complex multiplication of two unit-modulus numbers *is* angle addition:**
  `z_φc · z_ψ` = the vector at angle `(θ_φc + θ_ψ)`, automatically
  unit-modulus, no modular arithmetic needed.
- **Multiplying by the conjugate is angle subtraction:**
  `z_φc · conj(z_ψ)` = the vector at angle `(θ_φc − θ_ψ)`.

This sidesteps both problems above simultaneously — the arithmetic never
leaves vector space, so there's no discontinuity to wrap around, and no
explicit period-matching step is needed.

## 4. A convenient coincidence: the existing PERIODIC rescaling already gives the right combination

`PERIODIC.transform_head` rescales each angle before taking sin/cos:
```python
theta = v * (2*pi / spec.period)
```
For `polarization_angle` (period = π), this means the stored vector encodes
`2ψ`, not `ψ`, by construction (unrelated to this degeneracy work — it's
just how the existing sin/cos encoding handles a π-periodic quantity).

The physically relevant combination from Sec. 1 is `2ψ + φc`. Since the
`ψ` vector already represents `2ψ` internally, **`z_φc · z_ψ` (computed
directly from the existing per-head unit vectors) already equals the
degenerate combination we want** — no extra rescaling needs to be
engineered; it falls out of infrastructure that already exists for
unrelated reasons.

**Caveat — confirm before relying on this:** the exact sign and whether it's
`2ψ + φc` or `2ψ − φc` (or a convention-dependent variant) depends on the
waveform/polarization sign convention in use (IMRPhenomXPHM's specific
convention, and the sign of `cos ι`). Section 6 covers how to check this
empirically rather than assuming it.

## 5. Sketch of the two-head design

Keep `φc` and `ψ` as ordinary small `PERIODIC` heads (reactivating
`polarization_angle`, currently dormant). Then, **as a loss-level
transformation on top of both heads' raw outputs** — not a new output
architecture:

1. Normalize each predicted `(sin, cos)` pair to unit modulus before use
   (same one-line normalization the vMF head already applies to `mu_raw`;
   `PERIODIC` heads use `tanh`, which is not itself unit-norm-constrained).
2. Compute the sum-vector `z_φc · z_ψ` and difference-vector
   `z_φc · conj(z_ψ)` for both predicted and true angle-vectors.
3. Apply two independent losses (Huber on the resulting 2D vectors, or a
   von-Mises-style angular loss) — one for the sum-vector, one for the
   difference-vector.
4. Give each its own `log_var_clamp` entry in `MultiHeadTrainer`, exactly
   like `q`'s override — expect (pending confirmation) the sum-head to
   train like a normal well-identified target, and the difference-head to
   plausibly warrant a tight clamp if it turns out to carry near-zero
   signal, same treatment `q` got before its clamp fix.

This requires **no new output tensor shapes** beyond what two ordinary
`PERIODIC` heads already produce — the sum/difference computation and
clamp-per-head logic sit in the loss function, similar in spirit to how
`vmf_nll_loss` sits on top of the `sky_position` head's two raw tensors.

## 6. Validation steps before implementing (do these first)

1. **Confirm the sign/combination empirically**, don't hardcode from the
   literature statement alone. Using the existing (currently unused for
   training) `polarization_angle` column and the active `coa_phase` column
   on a validation-style data slice: check which of `φc+2ψ`, `φc−2ψ`
   correlates more cleanly with whatever ground truth is available,
   conditioned on the sign of `cos(inclination)` — the degeneracy and its
   sign are inclination-dependent.
2. **Check whether the difference-head carries any real signal at all**,
   rather than assuming it's fully uninformative. If it trains to some
   nontrivial (even if modest) R²/angular accuracy, that's informative in
   itself — it would mean the injected population isn't as face-on-dominated
   as the idealized theory assumes, and the clamp strategy should be less
   aggressive than for a truly degenerate direction.
3. **Decide whether the difference-head needs to exist as a trained target
   at all**, or whether it's only useful as a diagnostic ("canary") to
   confirm the degeneracy model is correct — if no downstream use needs
   ψ and φc individually recovered, the sum-head alone may be the only
   head worth carrying forward, same reasoning as why `sky_position`
   doesn't need a dedicated "unconstrained sky direction" head.

## 7. Explicitly not yet decided / not yet done

- `polarization_angle` is not reactivated in any config.
- No loss function for the sum/difference vectors has been written.
- No empirical check of the sign convention (Sec. 6, step 1) has been run.
- This entire document describes a proof-of-concept design, not a plan
  ready for implementation — Sec. 6 is the prerequisite work before writing
  any code.
