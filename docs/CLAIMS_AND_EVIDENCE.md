# Claims and Evidence Matrix

This document exists for one reason:

> if a sentence appears in a README, slide deck, or manuscript, there should be
> an obvious artifact that backs it.

That is especially important for this repository because several reviewers and
readers benefit from tighter claim discipline.

---

## 1. The strongest safe claims

| Safe claim | Evidence | Notes on wording |
|---|---|---|
| Adding the committed STL regularizer improves safety and cooling robustness on the 1D diffusion case study. | `results/diffusion1d_main_summary.json`, `results/diffusion1d_*_rtamt.json` | Say **on the committed diffusion benchmark**, not in general. |
| The robustness gain comes with a PDE-fidelity cost. | `results/diffusion1d_main_summary.json`, `figs/diffusion1d_training_dynamics.png` | Use the actual MSE / robustness numbers. |
| The repository contains a separate lambda sweep that exposes a robustness / fidelity tradeoff. | `results/diffusion1d_ablation_summary.json`, `figs/diffusion1d_lambda_ablation.png` | Keep the sweep distinct from the main baseline/STL pair. |
| Spatial monitoring matters once the model output is a field rather than a scalar trace. | `results/heat2d_strel_monitoring.json`, `figs/heat2d_field_evolution.png`, `figs/heat2d_stl_traces.png` | Phrase this as a case-study observation, not a theorem. |
| The 2D heat example is presently a monitoring result, not a full spatially constrained training benchmark. | `docs/EXAMPLE_STATUS.md`, `docs/SPECIFICATIONS.md` | This distinction should appear everywhere the 2D example is described. |
| The repository is reproducible from committed artifacts first and from retraining second. | `scripts/refresh_committed_summaries.py`, `scripts/generate_all_figures.py`, `docs/REPRODUCIBILITY.md` | This is a workflow claim, not a guarantee of byte-identical plots across machines. |
| Neuromancer, TorchPhysics, and PhysicsNeMo are represented by integration paths of different maturity. | `docs/FRAMEWORK_SURVEY.md`, `scripts/train_neuromancer_stl.py`, `scripts/train_burgers_torchphysics.py`, `src/neural_pde_stl_strel/frameworks/physicsnemo_hello.py` | Use the words **demo**, **draft**, or **probe** as appropriate. |
| The default benchmark-cost numbers in the repo are CPU-first snapshot numbers, not universal performance claims. | `results/benchmark_training.csv`, `results/env_report.json` | Explicitly say these are committed snapshot numbers. |

---

## 2. Numbers that are safe to quote directly

### Main diffusion comparison

Source of truth: `results/diffusion1d_main_summary.json`

- Baseline PDE MSE: `0.000105`
- Baseline peak value: `1.0040`
- Baseline `ρ(G_[0,1](max_x u <= 1.0))`: `-0.0040`
- STL-regularized PDE MSE: `0.024337`
- STL-regularized peak value: `0.6562`
- STL-regularized `ρ(G_[0,1](max_x u <= 1.0))`: `+0.3438`
- STL config weight: `λ = 5.0`

### Diffusion lambda sweep

Source of truth: `results/diffusion1d_ablation_summary.json`

- first positive tight-cooling margin appears at `λ = 2`
- recommended reference lambda among the committed ablations: `λ = 4`

### 2D heat monitoring

Source of truth: `results/heat2d_strel_monitoring.json`

- field shape: `(32, 32, 50)`
- time step: `0.05`
- thresholds `quantile_0.995` and `quantile_0.999` are satisfied on the
  committed rollout
- threshold `mean+0.5*std` is not satisfied on the committed rollout

### Runtime snapshot

Source of truth: `results/benchmark_training.csv`

- diffusion baseline: `45.2 s`, `512 MB`
- diffusion STL default: `108.6 s`, `624 MB`
- heat rollout: `0.3 s`, `128 MB`

---

## 3. Claims that require careful qualifiers

| Risky topic | Safe qualifier |
|---|---|
| Differentiable STL regularization | "a simple smooth STL regularizer used in this repository" |
| Spatial logic for the 2D case | "STREL-style monitoring on the committed rollout" |
| Framework integration | "documented integration path" or "probe" |
| Runtime comparison | "CPU-first committed snapshot" |
| Benchmark recommendation | "recommended next step" rather than "best benchmark overall" |
| Aerospace relevance | "plausible aerospace-flavored extension" |

---

## 4. Claims that should be avoided unless new evidence is added

### Avoid these formulations

- "the first STL/STREL framework for neural PDEs"
- "formal verification of the PINN"
- "framework-agnostic benchmark comparison"
- "2D STREL-constrained training result"
- "best lambda"
- "state-of-the-art"

### Better replacements

- "reproducible scaffold"
- "monitoring plus training-time regularization"
- "case study"
- "committed benchmark"
- "reference lambda among the committed ablations"
- "integration surface"

---

## 5. Sentence templates that are safe to reuse

### Contribution language

- "We study how explicit temporal or spatial specifications can be attached to a
  compact neural-PDE / physics-ML workflow in a reproducible way."
- "On the committed 1D diffusion benchmark, adding the STL penalty improves
  robustness margins while increasing PDE error."
- "The 2D heat example is used as a spatial monitoring case study rather than as
  a full training-time STREL benchmark."

### Limitation language

- "The reported spatial semantics are discretized on the sampled grid."
- "The current repository demonstrates monitoring and regularization, not formal
  proof of correctness."
- "The framework probes are included to expose attachment points, not to claim
  full benchmark coverage."

---

## 6. Where to look before writing any paper paragraph

Recommended order:

1. `docs/SPECIFICATIONS.md`
2. `docs/EXAMPLE_STATUS.md`
3. `docs/IMPLEMENTATION_MAP.md`
4. `results/diffusion1d_main_summary.json`
5. `results/diffusion1d_ablation_summary.json`
6. `results/heat2d_strel_monitoring.json`
7. `results/benchmark_training.csv`

If a paragraph cannot be grounded in those files, it probably needs either a
better qualifier or a new experiment.
