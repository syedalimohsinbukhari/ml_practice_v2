Good news: for RA/Dec specifically you're on the 2-sphere (S²), which is the *one* case where the von Mises-Fisher distribution has a clean closed form — no Bessel-function approximation needed, unlike the general case. That makes it very practical to implement directly in Keras.

## The structural issue first, before any code

Your current `heads_spec.py` treats `ra` and `declination` as **two independent heads** — `ra` is `PERIODIC` (sin/cos), `declination` is `UNIT_AFFINE` (sigmoid). That means the model fits RA and Dec as if they were unrelated 1D problems, throwing away the fact that sky position is a genuinely 2D, correlated quantity on a sphere — and the ring/mirror degeneracy is a *joint* 2D shape, not something that shows up separately in RA's and Dec's individual marginals. This is worth fixing structurally: **merge RA+Dec into one head** that outputs a 3D unit vector, and use a spherical loss on that vector directly.

## Plan

1. **Represent target as a unit vector**, not two angles: `(ra, dec) → (x,y,z)` on the unit sphere.
2. **Network outputs an unnormalized 3-vector + a concentration parameter κ** (like predicting mean + inverse-variance, but on the sphere).
3. **Loss = negative log-likelihood of the vMF distribution on S²**, which has this closed form (no Bessel functions needed, because on the 2-sphere the order-1/2 Bessel function reduces to `sinh`):

```
NLL(x; μ, κ) = -log(κ) + log(4π) + log(sinh(κ)) - κ·(μ·x)
```

4. **At inference**, `μ` (normalized) gives you the predicted sky position, and `κ` gives you a natural, calibrated angular uncertainty — bonus over your current point-estimate heads.
5. **If a single vMF still looks like it's "compromising" between two modes** (the classic ring-degeneracy symptom), extend to a small mixture of vMFs (2-3 components) — genuinely bimodal-capable, at modest extra complexity.
6. **Cheaper alternative to try first/in parallel:** sector classification — bin the sphere into coarse pixels, treat as multi-class softmax. Much simpler to bolt on, naturally handles multimodality (softmax can put mass on two sectors), and the literature gets >90% accuracy at coarse resolution with this. Good sanity-check baseline before committing to the vMF machinery.

## Code

**1. Target conversion (data/transforms.py):****2. The vMF head + loss (models/heads.py addition + training/losses.py addition):****3. A minimal runnable sanity check** (synthetic sphere data, confirms the loss trains and recovers known directions with sensible κ):It's training stably (no NaNs, monotonic loss decrease, angular error shrinking) but hasn't converged yet at 60 epochs — let me give it more room to confirm the κ-uncertainty behavior clearly before showing you the final numbers.The mechanics work — loss trains smoothly with no NaNs from the `log(sinh(κ))` term, and angular error drops from ~82° (random guess on a sphere) to ~24° as the model learns the true direction from the noisy synthetic features. Two honest notes from this quick check, not glossed over:

1. **Val loss overfits after ~epoch 50** (0.772 → 0.883) even though angular error stays roughly flat — that's the toy setup being too small/noisy relative to model capacity, not a flaw in the loss itself. Expected for a 30-second sanity check, not a real training run.
2. **κ didn't track per-example noise level well here** (corr = 0.23, weakly positive, when I'd want to see it clearly negative — more noise should mean lower confidence). This is worth taking as an open question rather than assuming it'll just work on real data: the κ-head may need more capacity, more epochs, or a separate learning rate from the μ-head to actually learn calibrated uncertainty. Worth checking directly once you plug this into your real pipeline rather than trusting it blindly.

## What to check when you plug this into your actual pipeline

1. **Merge `ra`+`declination` into one `sky_position` head** in `heads_spec.py` — add a new `TransformKind.SPHERICAL_UNIT_VECTOR` that converts `(ra, dec) → (x,y,z)` at data-load time (using `sky_transform.py`), with `dim=3`, no activation (linear, since normalization happens in the loss).
2. **Wire the vMF loss into `losses.py`'s per-head loss table**, keyed to this new head — it replaces Huber for this one head only, everything else stays as-is.
3. **New diagnostic metric needed:** don't reuse per-axis MAE(ra)/MAE(dec) for this head — plot **mean angular separation in degrees** (`angular_separation()` in the code above) as the headline metric, and track κ's distribution/correlation with SNR as a secondary "is uncertainty calibrated" check.
4. **If single-vMF plateaus** (μ looking like it's splitting the difference between two real modes — check by eye on a few examples with known ring/mirror ambiguity), that's your trigger to extend to a 2-3 component mixture — same NLL idea, just softmax-weighted sum of vMF densities instead of one.

Want me to also sketch the sector-classification alternative code (the cheaper baseline), so you can run both in parallel and see which one actually earns its complexity on your data?
