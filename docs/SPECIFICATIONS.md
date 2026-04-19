# Formal Specifications

This document records the logic formulas that are actually used in the
repository, how continuous fields are reduced to monitored signals, and which
numbers in the repo come from which committed artifacts.

The main rule for this file is simple: **if a formula or threshold appears in a
plot, README table, or paper draft, it should be written here explicitly.**

---

## 1. STL background used by the repository

For a scalar sampled signal `s[k]` with time step `dt`, the repository uses the
standard quantitative sign convention:

- `ρ > 0` means satisfaction with margin,
- `ρ < 0` means violation with margin `|ρ|`.

For atomic predicates:

```text
ρ(s <= c, k) = c - s[k]
ρ(s >= c, k) = s[k] - c
```

For the bounded temporal operators used here:

```text
ρ(G_[a,b] φ, 0) = min_k  ρ(φ, k)      over k with t_k in [a, b]
ρ(F_[a,b] φ, 0) = max_k  ρ(φ, k)      over k with t_k in [a, b]
```

During training, hard `min` and `max` can be replaced by smooth log-sum-exp
approximations. The audit tables in this repository, however, are computed from
the committed sampled fields with exact sampled `min` and `max`.

---

## 2. Case study A: 1D diffusion

### PDE and domain

```text
u_t = α u_xx
α = 0.1
x in [0, 1]
t in [0, 1]
u(0, t) = u(1, t) = 0
u(x, 0) = sin(πx)
```

The dense committed audit grid has shape `(128, 64)` and
time step `dt = 0.015873017`.

### Field-to-signal reduction

The audited scalar trace is the sampled spatial maximum:

```text
s[k] = max_j u(x_j, t_k)
```

This corresponds to a discretized version of `max_x u(x,t)`.

For training-time penalties, the code also supports other reductions such as
softmax, mean, and point sampling. The committed audit numbers in
`results/diffusion1d_main_summary.json` use the hard sampled maximum.

### Audited STL formulas

The repository's main audit formulas are:

```text
φ_safe      := G_[0,1]   (max_x u(x,t) <= 1.0)
φ_cool^0.3  := F_[0.8,1] (max_x u(x,t) <= 0.3)
φ_cool^0.4  := F_[0.8,1] (max_x u(x,t) <= 0.4)
```

Interpretation:

- `φ_safe` asks whether the peak value stays under `1.0` throughout the run.
- `φ_cool^0.3` asks whether the field gets below `0.3` somewhere in the last
  fifth of the horizon.
- `φ_cool^0.4` is a looser version of the same cooling requirement.

### Main committed checkpoint comparison

These values are derived from the committed diffusion field sidecars
`results/diffusion1d_baseline_field.pt` and `results/diffusion1d_stl_field.pt`
(with checkpoint metadata used for the training hyperparameters).

| Case | Epochs | `λ` | PDE MSE | Peak value | `ρ(φ_safe)` | `ρ(φ_cool^0.3)` | `ρ(φ_cool^0.4)` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 400 | 0.0 | 0.000105 | 1.003969 | -0.003969 | -0.074980 | +0.025020 |
| STL-regularized | 800 | 5.0 | 0.024337 | 0.656221 | +0.343779 | +0.049668 | +0.149668 |

Two points are worth keeping straight:

1. the baseline model is closer to the diffusion PDE;
2. the STL-regularized model has substantially larger safety and cooling
   margins.

### Training-time objective

For the diffusion training script, the intended objective is

```text
L_total = L_PDE + L_BC/IC + λ · L_STL
```

where:

- `L_PDE` is the diffusion residual loss,
- `L_BC/IC` groups boundary and initial-condition losses,
- `L_STL` is obtained by mapping robustness to a penalty, and
- `λ` is `stl.weight` in the YAML config.

The current committed default STL config uses `λ = 5.0` in
`configs/diffusion1d_stl.yaml`.

### Separate lambda ablation sweep

The file `results/diffusion1d_ablation_summary.json` comes from a separate sweep
of checkpoints:

```text
λ in {0, 2, 4, 6, 8, 10}
```

Those checkpoints are distinct from the main baseline/STL pair above. The sweep
should be read as an internal tradeoff study, not as the source of the main
checkpoint table.

| `λ` | PDE MSE | Peak value | `ρ(φ_safe)` | `ρ(φ_cool^0.3)` |
|---:|---:|---:|---:|---:|
| 0 | 0.000746 | 0.960324 | +0.039676 | -0.062167 |
| 2 | 0.008022 | 0.838382 | +0.161618 | +0.005826 |
| 4 | 0.014918 | 0.778372 | +0.221628 | +0.039070 |
| 6 | 0.020745 | 0.737922 | +0.262078 | +0.061299 |
| 8 | 0.025974 | 0.707548 | +0.292452 | +0.079427 |
| 10 | 0.030322 | 0.683520 | +0.316480 | +0.092548 |


According to the recomputed sweep:

- the tight cooling property first turns positive at `λ = 2`,
- `λ = 4` is the best
  compromise among the committed ablation runs.

---

## 3. Case study B: 2D heat and STREL-style containment

### Committed rollout

The repository commits a heat-field rollout with shape
`(32, 32, 50)` and `dt = 0.05`. The case study uses a
`32 x 32` spatial grid and monitors thresholded hot regions over time.

### Spatial predicate

For a chosen threshold `θ`, define:

```text
hot(x, y, t) := [ u(x, y, t) >= θ ]
```

### Spatial-temporal property

The MoonLight file expresses a containment/quench pattern that can be read as:

```text
contain := F ( G ( nowhere_hot ) )
```

On the committed connected grid, this reduces to:

```text
there exists a time t* such that for all later times t >= t*,
max_(x,y) u(x,y,t) < θ
```

So the practical audit can be performed directly from the sampled spatial
maximum trace.

### Threshold analysis on the committed rollout

| Threshold rule | `θ` | Satisfied? | First quench time | Final robustness |
|---|---:|:---:|---:|---:|
| mean+0.5*std | 0.0894 | NO | -- | -0.5815 |
| quantile_0.99 | 0.6376 | NO | -- | -0.0333 |
| quantile_0.995 | 0.7255 | YES | 1.85 | +0.0546 |
| quantile_0.999 | 0.8604 | YES | 0.70 | +0.1895 |


The important limitation is that this case study is presently a **monitoring**
example. The repository does not yet claim a full STREL-constrained training
loop for the 2D heat problem.

---

## 4. Mapping from formulas to code and artifacts

| Item | Primary code path | Artifact(s) |
|---|---|---|
| Diffusion training loop | `scripts/train_diffusion_stl.py` | `results/diffusion1d_*.pt`, `results/diffusion1d_*.csv` |
| Diffusion summary refresh / semantic validation | `scripts/refresh_committed_summaries.py` | `results/diffusion1d_main_summary.json` |
| Offline diffusion audit | `scripts/eval_diffusion_rtamt.py` | `results/diffusion1d_*_rtamt.json` |
| Heat spatial monitoring | `scripts/eval_heat2d_moonlight.py` | `results/heat2d_strel_monitoring.json` |
| Figure generation | `scripts/generate_all_figures.py` | `figs/*.png`, selected `assets/*.png` |

---

## 5. What this file does not claim

- It does not claim formal verification of the learned models.
- It does not claim that every framework probe in the repository has a complete
  benchmark attached to it.
- It does not claim that the 2D heat case study is already a full spatially
  constrained training benchmark.
