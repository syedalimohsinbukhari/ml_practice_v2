# φc/ψ Degeneracy PoC — Results Log

Chronological record of every prerequisite check run, what changed, and what
the new results were.  Companion to `NOTES.md` (design decisions) and
`phic_psi_implementation_plan_v4.md` (build plan).

---

## Run 1 — 2026-07-16 (initial)

**Branch:** `poc/phic-psi-degeneracy` (commit `60aeae8`)
**Machine:** local (T530, CPU-only)

### Step 1.1 (initial, MIXED sign)

| Metric | Value |
|--------|-------|
| Harness check | ✓ passed |
| Well-constrained combo | combo_B |
| Correlation ratio | 1.2× (aggregate, both signs mixed) |
| Sign flip | YES |

**Issue:** Correlation computed across the entire population mixing both
`cos ι > 0` and `cos ι < 0` — the aggregate 1.2× ratio was a dilution
artifact (rev2 review).

### Step 1.2 (initial, broken polynomial)

| Metric | Value |
|--------|-------|
| Fit method | Unconstrained polynomial w = a₀ − a₁·cos²ι − a₂·cos⁴ι |
| Max residual | 1.32 (4× the function range!) |
| w(face-on) from fit formula | −5.29 (negative!) |
| w(face-on) from raw data | 0.000 |
| w(edge-on) from fit formula | 0.287 |
| w(edge-on) from raw data | 0.231 |
| Endpoints consistent? | ✗ No — formula and summary used different sources |

**Issue:** The polynomial fit was unconstrained, went negative, had residuals
larger than the function range, and the printed formula didn't match the
printed endpoint summary (rev2 review).

### Step 1.6 (first run on GPU)

| Metric | Value |
|--------|-------|
| Face-on fraction (\|cos ι\| > 0.9) | 28.7% |
| Edge-on fraction (\|cos ι\| < 0.5) | 32.7% |
| Warning | None — population well-mixed |

---

## Run 2 — 2026-07-16 (rev2 fixes)

**Branch:** `poc/phic-psi-degeneracy` (commit `25a77ef`)
**Machine:** local (T530, CPU-only)

### Changes from Run 1

| What | Before | After |
|------|--------|-------|
| Step 1.1 correlation | Aggregate (mixed sign) | Split by sign(cos ι) |
| Step 1.2 fit | Unconstrained polynomial | Linear interpolation on cos²ι, clamped [0,1] |
| Step 1.2 Jacobian eps | 1e-6 | 1e-5 |
| Step 1.2 Jacobian n_Phi | 100 | 200 |

### Step 1.1 (rev2, SPLIT by sign)

| Regime | Well-constrained | Ratio |
|--------|-----------------|-------|
| cos ι > 0 (ι=π/4) | combo_B | 1.2× |
| cos ι < 0 (ι=3π/4) | combo_A | 1.2× |

Sign flip: **YES** — label depends on sign(cos ι).
Trainer updated: `sign_dependent_combo=true`.

### Step 1.2 (rev2, interpolation)

| Metric | Value |
|--------|-------|
| Fit method | Linear interpolation on cos²ι |
| Max residual | 0.00 (by construction at grid points) |
| w(face-on, cos²ι=1) | 0.0000 ✓ |
| w(edge-on, cos²ι=0) | 0.1212 |
| Endpoints consistent? | ✓ Yes |
| Negative weights? | ✗ No (clamped [0,1]) |

**Decision:** Use `w_iota_default` (1−cos²ι) for Run B, not the empirical fit.
The empirical edge-on asymptote (0.12) is lower than the default (1.0) —
may be physical or a Jacobian artifact.

---

## Run 3 — 2026-07-16 (rev3: ι sweep + bootstrap + deep grid)

**Branch:** `poc/phic-psi-degeneracy` (commit TBD)
**Machine:** TBD (run on GPU machine for speed)
**Parameters:** n_sky_samples=200, n_sweep=50, 50 ι sweep points (25/sign),
n_boot=5000, Step 1.2: n_sky=200, n_iota=200

### Step 1.1 (rev3, ι SWEEP — 16-point pilot)

**cos ι > 0:**

| ι [rad] | Winner | Ratio |
|---------|--------|-------|
| 0.100 | combo_B | 1.65× |
| 0.303 | combo_B | 1.59× |
| 0.506 | combo_B | 1.60× |
| 0.709 | combo_B | 1.22× |
| 0.912 | combo_B | 1.09× |
| 1.115 | combo_B | 1.18× |
| 1.318 | combo_B | 1.14× |
| 1.521 | combo_B | 1.18× |

**cos ι < 0:**

| ι [rad] | Winner | Ratio |
|---------|--------|-------|
| 1.621 | combo_A | 1.07× |
| 1.824 | combo_A | 1.21× |
| 2.027 | combo_A | 1.20× |
| 2.230 | combo_A | 1.12× |
| 2.433 | combo_A | 1.14× |
| 2.636 | combo_A | 1.42× |
| 2.839 | combo_A | 1.60× |
| 3.042 | combo_A | 1.61× |

**Trend check:**
- cos ι > 0: ratio 1.65× → 1.18× as ι → π/2 ✓ (widens toward face-on)
- cos ι < 0: ratio 1.07× → 1.61× as ι → π ✓ (widens toward face-on)

**Significance (ι=π/4):**
- cos ι > 0: ratio = 1.23×, 95% CI = [1.11, 1.38] → **YES** (CI excludes 1.0)
- cos ι < 0: ratio = 1.19×, 95% CI = [1.08, 1.33] → **YES** (CI excludes 1.0)

**Conclusion:** The effect is statistically significant, the trend matches
physics (ratio grows toward face-on), and the label flips with sign(cos ι).
All three predictions confirmed.

### Step 1.2 (rev3, sky-averaging confirmed)

| Metric | Value |
|--------|-------|
| Sky-averaging | ✓ 50 random (a,b) pairs |
| ι grid density | 100 points |
| Intermediate shape (5 of 100): | |
| ι=0.001, cos²ι=1.000 | w=0.0000 |
| ι=0.397, cos²ι=0.850 | w=0.2006 |
| ι=0.793, cos²ι=0.492 | w=0.9346 |
| ι=1.189, cos²ι=0.138 | w=0.3116 |
| ι=1.570, cos²ι=0.000 | w=0.1212 |

**Observation:** w peaks at intermediate ι (~0.8 rad, cos²ι≈0.5) then drops
toward edge-on. This may reflect real physics (both polarizations contribute
at intermediate inclinations, improving conditioning) or a Jacobian artifact.
Does not affect the decision to use `w_iota_default` for Run B.

### Step 1.6 (unchanged from Run 1)

Population: 28.7% face-on, 32.7% edge-on — healthy.

---

## Artifacts produced

| File | Description |
|------|-------------|
| `sweep_1_1_ratio_vs_iota.csv` | Full ι sweep data (CSV) |
| `sweep_1_1_ratio_vs_iota.png` | Ratio vs ι plot (two panels: cos ι > 0, cos ι < 0) |

---

## Summary of go/no-go signals

| Gate | Status | Decision |
|------|--------|----------|
| 1.1 — Sign/combination | ✓ Passed | combo_B at cos ι > 0, sign-dependent |
| 1.2 — w(ι) derivation | ⚠️ Empirical curve available but low edge-on asymptote | Use `w=1−cos²ι` default for Run B |
| 1.3 — Data representation | ✓ Passed | Time-domain strain, phase preserved |
| 1.4 — True ι access | ✓ Passed | Via `y_true["inclination"][:,1]` |
| 1.5 — Curriculum mechanism | ✓ Decided | Static per-sample weight, no scheduling |
| 1.6 — Population balance | ✓ Passed | 28.7% face-on, 32.7% edge-on |
| **Overall** | **✓ GO for Run A and Run B** | Proceed to training |
