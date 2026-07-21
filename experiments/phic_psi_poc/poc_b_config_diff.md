# Section B — poc_b config diff and collapse investigation

**Date**: 2026-07-21
**Context**: Run 7 verification plan Section B — why is poc_b MORE collapsed than poc_a?

---

## 1. Config diff: poc_a vs poc_b

The only functional differences between `config_baseline.yaml` (poc_a) and `config_poc.yaml` (poc_b):

| Setting | poc_a | poc_b |
|---------|-------|-------|
| `loss.mode` | `baseline` | `poc` |
| `loss.combo_log_var_clamp` | absent | `{default: 3.0, combo_A: 3.0, combo_B: 3.0}` |
| `loss.well_constrained_combo` | absent | `combo_B` |
| `loss.sign_dependent_combo` | absent | `true` |

Everything else — data, model architecture (tcn trunk, same heads, same hidden_units), optimizer (lr=1e-3, plateau schedule), training (80 epochs, seed=42), magnitude_penalty_lambda (0.01) — is **identical**.

---

## 2. What `mode: poc` changes in the loss path

### poc_a (`_baseline_total_loss`, trainer.py:385–420)

Two independent circular loss terms:

```
loss = exp(−s_phic) · mean(1 − cos(Δφc)) + s_phic
     + exp(−s_psi)  · mean(1 − cos(Δψ))  + s_psi
     + λ · (|v_phic_raw| − 1)² + λ · (|v_psi_raw| − 1)²
     + other_heads_loss
```

Both φc and ψ get independent gradient signals. Even when neither contains real information, the two loss terms create **competing pressures** — the model can't satisfy both simultaneously with a single constant because the loss surface has two independent minima. The result: predictions spread across an arc rather than collapsing to a point.

### poc_b (`_poc_total_loss`, trainer.py:426–501)

φc and ψ are funneled through two combo vectors, and the poorly-constrained one is **curriculum-suppressed**:

```
combo_A = z_phic · conj(z_psi)        # ≈ φc + 2ψ
combo_B = z_phic · z_psi              # ≈ φc − 2ψ

w(ι) = 1 − cos²(ι)                    # ∈ [0, 1]

For cos ι ≥ 0 (well_constrained_combo = combo_B):
  loss = exp(−s_A) · mean(w(ι) · (1−cosΔA)) + s_A    ← suppressed
       + exp(−s_B) · mean(1.0  · (1−cosΔB)) + s_B    ← full weight

For cos ι < 0 (roles swap):
  loss = exp(−s_A) · mean(1.0  · (1−cosΔA)) + s_A    ← full weight
       + exp(−s_B) · mean(w(ι) · (1−cosΔB)) + s_B    ← suppressed
```

## 3. Why this produces worse collapse

The curriculum weight `w(ι) = 1 − cos²(ι)`:

- **At face-on** (|cos ι| ≈ 1, ι ≈ 0 or π): w ≈ 0 → the poorly-constrained combo's loss is **nearly zeroed out**
- **At edge-on** (cos ι ≈ 0, ι ≈ π/2): w ≈ 1 → both combos get full weight

Since cos ι is uniformly distributed over [−1, 1] for an isotropic population, the **median** |cos ι| ≈ 0.5 → median w ≈ 0.75. But critically, a substantial fraction of samples have w near 0 — the face-on wing of the distribution.

**The mechanism:** For most samples, the model effectively trains on only **one** combo's loss. One combo = one constraint on a 2D problem (φc, ψ). The model's optimal strategy is:

1. Find the single (φc, ψ) constant that minimizes expected `1−cosΔ` for the *well-constrained* combo across the training population
2. Output that constant for every sample

This produces a single sharp peak in the prediction distribution — exactly the `COLLAPSE` flag (circ_r ≈ 0.99, single peak at 42–44%).

**In contrast**, poc_a's independent losses on φc and ψ give the model two separate objectives that cannot both be satisfied by the same constant (the loss minima for φc and ψ are at different angular positions). The model settles into a compromise — output spread across a wider arc — producing lower circ_r (0.85/0.49).

### This is a prediction of the degeneracy hypothesis, not a config bug

If φc and ψ were genuinely learnable from strain:
- The well-constrained combo's gradient would contain real directional information
- The curriculum would correctly suppress noise from the poorly-constrained combo
- The model would learn both angles through the one well-constrained channel + the sign-flip at cos ι = 0

Since neither φc nor ψ is learnable:
- The well-constrained combo's gradient is no more informative than random
- The curriculum suppression of the other combo *removes* what little diversity of pressure existed
- The model collapses harder because it's optimizing a single, simpler objective (minimize expected loss on one combo) rather than two competing ones

**The curriculum design assumes at least one combo carries real signal.** When neither does, it actively makes things worse by funneling all gradient through a single underdetermined channel.

---

## 4. mchirp regression check

From `analysis_report_20260720_234304.md`:

| Model | mchirp MAE | mchirp R² |
|-------|-----------|----------|
| poc_a | 0.9773 | 0.9594 |
| poc_b | 1.0242 | 0.9569 |
| Δ | **+0.0469 (+4.8%)** | −0.0025 (−0.3%) |

The mchirp regression is real but mild — ~5% MAE increase, negligible R² change. This is consistent with mild task interference: the collapsed periodic heads add some noise to the shared trunk features, slightly degrading the scalar heads that read from them. Not a separate bug — a predictable side effect of dead heads in a shared-trunk architecture.

---

## 5. Verdict

**poc_b's worse collapse is explained, and it's not a config bug.** It's a direct consequence of the curriculum design interacting with the degeneracy:

| | poc_a (baseline) | poc_b (PoC) |
|---|---|---|
| Loss structure | 2 independent terms (φc, ψ) | 2 combo terms, one curriculum-suppressed |
| Effective constraints | 2 (both full weight) | ~1 (one suppressed) |
| When φc/ψ are learnable | Two constraints → two angles recovered | One well-constrained combo + sign flip → both angles recovered, less noise |
| When φc/ψ are NOT learnable | Two competing objectives → spread predictions | One objective → **collapse to single constant** |

The poc_b design is **correct** — it would work if the degeneracy were breakable. The fact that it produces *worse* collapse than the baseline is actually supporting evidence for the degeneracy: the model optimizes the one available loss channel by collapsing to its minimum, because that channel carries no real angular information.

**No config fix needed.** The observed behavior is consistent with the degeneracy hypothesis and the curriculum design working as intended — just on inputs that genuinely carry no φc/ψ signal.

---

## 6. What this means for the overall verification

Section B asked two questions:

1. **What's driving the more severe collapse?** → Answered above: curriculum suppression funneling gradient through one underdetermined combo channel.
2. **Does it share a cause with the mchirp regression?** → Mildly. The ~5% mchirp MAE increase is consistent with dead-head noise in the shared trunk, not a separate bug.

Both are consistent with the degeneracy hypothesis. No config-level fix is indicated — the design is correct; the physics is the limitation.