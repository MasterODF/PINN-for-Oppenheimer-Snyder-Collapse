# Curriculum Training in the OS Collapse GA-PINN

## Why curriculum training?

A PINN has to satisfy several constraints simultaneously:

1. **Initial conditions** — ρ, uₓ, C, Φ must match the t=0 profile
2. **Constraint equations** — dm/dr and dΦ/dr (elliptic, must hold at every time)
3. **Evolution equations** — ∂D/∂t, ∂Mₓ/∂t, ∂E/∂t (hyperbolic PDEs)

The problem is that at random initialisation the neural network outputs are noisy,
and the evolution PDE residuals involve **products of multiple noisy outputs** (e.g.
`D = X·ρ·W`, `fuente_Mx = (Mx·ux - E - D)·α·X·m/r²`). These residuals are enormous.

If you turn on all three loss terms from epoch 0, the gradient from the evolution
equations overwhelms the IC gradient. The optimizer finds that **setting everything
to a constant kills the evolution residual to zero** (all derivatives vanish), and
it gets stuck there — the IC loss alone is too weak to pull it out.

**Curriculum training** solves this by teaching the network in stages, from easiest
to hardest.

---

## The three phases

```
Epoch:  0 ──────── 5000 ──────── 15000 ────────────── 50000
         │  Phase 1  │  Phase 2  │      Phase 3        │
         │  IC only  │  ramp up  │    full loss         │
```

### Phase 1: 0 → `curriculum_phase1_end` (default 5000)

**Active losses:**
- ✅ Initial conditions (ρ, uₓ, C, Φ at t=0) — with log-cosh + high IC weight
- ✅ Constraint equations (dm/dr, dΦ/dr) — elliptic, hold at all t
- ✅ Symmetry at r=0 (uₓ=0, C=0)
- ✅ Sparse analytical supervision (C, Φ at t=0.25, 0.5, 0.75)
- ❌ Evolution equations — **weight = 0**

**What happens:** The network learns the *shape* of the initial data and the metric
structure. By epoch ~3000–5000, ℒ_ic is typically down by 2–3 orders of magnitude
from its initial value. This gives the network a physically meaningful starting
point before it has to deal with time evolution.

In code (`utils.py`):
```python
if epoch < phase1_end:
    w_R_eff = 0.0   # evolution equations completely off
```

---

### Phase 2: `curriculum_phase1_end` → `curriculum_phase2_end` (default 5000–15000)

**Active losses:** Everything from Phase 1, plus:
- ✅ Evolution equations — **linearly ramped** from 0 → w_R

The ramp factor is:
```python
ramp = (epoch - phase1_end) / (phase2_end - phase1_end)   # 0.0 → 1.0
w_R_eff = w_R * ramp
```

At epoch 5000: evolution weight = 0  
At epoch 10000: evolution weight = 0.5 × w_R  
At epoch 15000: evolution weight = w_R (full)

**What happens:** The evolution equations enter gradually. Because the network
already knows the initial profile, it has a good "anchor" — instead of starting
from noise it propagates a physically reasonable IC forward in time. The causal
weighting (ε_t) further ensures it learns early times before late times.

This is where `AUC` starts rising from 0: causality becomes active once the
evolution equations turn on (w_R_eff > 0).

---

### Phase 3: `curriculum_phase2_end` → end (default 15000–50000)

**Active losses:** All terms at full weight.

```python
else:
    w_R_eff = w_R   # 1.0
```

**What happens:** The network fine-tunes the full space-time solution. The causal
weighting (the ε_t exponential) takes over and drives learning forward in time,
slice by slice. The `AUC` metric (area under the causal weight curve w_t) tracks
how far "into" the time domain the network has converged — AUC=1 means perfect,
AUC≈0 means only t=0 has converged.

---

## Interaction with causal weighting (ε_t)

Causal weighting and curriculum are complementary:

| Tool | Controls |
|------|----------|
| **Curriculum** | *Which* loss terms are active (macro scale) |
| **Causal weighting (ε_t)** | *Which time slices* are emphasised (micro scale) |

Causal weighting computes a per-time-slice weight:
```
w_t[i] = exp(-ε_t · cumsum(ℒ_evolution[:i]))
```
Time slices where the earlier loss is still high get exponentially downweighted.
This forces the network to "solve" early times before advancing to later ones.

It is applied **only to the evolution equations**, not to the constraints (which
are elliptic and must hold everywhere simultaneously regardless of time order).

---

## Interaction with IC weight scheduling

During Phase 1, the IC loss is boosted by `ic_factor`:

```python
ic_factor = max(1.0, 100.0 * exp(-epoch / ic_decay_tau))
```

At epoch 0: ic_factor = 100 → IC weights are 100× their base value  
At epoch 5000 (τ=5000): ic_factor ≈ 1 → back to base value  

This ensures the IC dominates while the evolution equations are off, and
fades away as the curriculum advances — so by the time evolution turns on,
the IC is strongly satisfied but no longer artificially inflated.

---

## Tuning the curriculum (config.json)

```json
"curriculum_phase1_end": 5000,    // end of IC-only phase
"curriculum_phase2_end": 15000,   // end of ramp phase
"ic_decay_tau": 5000.0            // IC boost decay timescale (epochs)
```

**If Phase 1 is too short:** IC won't be well-learned before evolution turns on → 
the network may still collapse to a constant.

**If Phase 1 is too long:** Wasted compute. The ℒ_ic metric tells you when Phase 1
is "done" — once it stops decreasing significantly, Phase 2 can begin.

**Rule of thumb:** Set `phase1_end` to when `ℒ_ic` plateaus. Set `ic_decay_tau ≈ phase1_end`.

---

## Reading the training output

```
6734/50000 [14:55] ℒ_ic=0.00334, ℒ=2.69, l2=1.65, AUC=0.00196
```

- **ℒ_ic = 0.00334** — IC almost perfectly satisfied (was ~0.6 at epoch 0). Phase 1 succeeded.
- **ℒ = 2.69** — total loss, dominated by evolution + constraint terms now active.
- **l2 = 1.65** — L2 error vs analytical solution; will drop as Phase 2/3 propagate
  the learned IC forward in time.
- **AUC = 0.00196** — causal frontier is near t=0. Will rise toward 1 as training
  advances through Phase 3.
