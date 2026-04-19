# Implementation Map

This note answers a practical question that came up repeatedly during review:

> Which files are the real evidence for the repository's claims, and which files
> are supporting utilities, demos, or integration probes?

The goal is to make the repository easy to audit without guessing.

---

## 1. Core reported case studies

### 1.1 1D diffusion with training-time STL regularization

**What this path demonstrates**

- a compact PINN training loop,
- a differentiable STL penalty added to the loss,
- post-hoc auditing on committed field sidecars / checkpoints,
- fidelity versus robustness tradeoffs.

**Primary files**

| Purpose | File(s) |
|---|---|
| Main training script | `scripts/train_diffusion_stl.py` |
| Core experiment implementation | `src/neural_pde_stl_strel/experiments/diffusion1d.py` |
| Smooth STL operators and penalty | `src/neural_pde_stl_strel/monitoring/stl_soft.py` |
| Exact post-hoc audit | `scripts/eval_diffusion_rtamt.py` |
| Main configs | `configs/diffusion1d_baseline.yaml`, `configs/diffusion1d_stl.yaml` |
| Ablation sweep | `scripts/run_ablations_diffusion.py` |
| Summary refresh / semantic check | `scripts/refresh_committed_summaries.py` |
| Main outputs | `results/diffusion1d_main_summary.json`, `results/diffusion1d_ablation_summary.json` |
| Main figures | `figs/diffusion1d_comparison.png`, `figs/diffusion1d_training_dynamics.png`, `figs/diffusion1d_lambda_ablation.png` |

### 1.2 2D heat with spatial/STREL-style monitoring

**What this path demonstrates**

- a spatially meaningful field output,
- threshold-based spatial predicates,
- MoonLight-oriented monitoring on a committed rollout,
- a monitoring example that is explicitly *not* yet claimed as a full
  STREL-constrained training benchmark.

**Primary files**

| Purpose | File(s) |
|---|---|
| Heat training / export example | `scripts/train_heat2d_strel.py` |
| Core experiment implementation | `src/neural_pde_stl_strel/experiments/heat2d.py` |
| MoonLight audit path | `scripts/eval_heat2d_moonlight.py` |
| STREL spec file | `scripts/specs/contain_hotspot.mls` |
| Monitoring config | `configs/heat2d_baseline.yaml` |
| Main outputs | `results/heat2d_strel_monitoring.json`, `results/heat2d_threshold_analysis.csv` |
| Main figures | `figs/heat2d_field_evolution.png`, `figs/heat2d_stl_traces.png` |

---

## 2. Additional executable paths that are useful, but not the main paper evidence

### 2.1 Heat2D scalar-STL training configs

These files show how the same 2D heat implementation can be exercised with a
scalar STL loss before moving to a full spatial-STREL-in-the-loop benchmark:

- `configs/heat2d_stl_safe.yaml`
- `configs/heat2d_stl_eventually.yaml`

They are useful stepping stones, but the current committed publication-facing figures
still center on the diffusion training benchmark and the 2D monitoring example.

### 2.2 Framework probes and demos

| Framework | File(s) | Intended interpretation |
|---|---|---|
| Neuromancer | `src/neural_pde_stl_strel/frameworks/neuromancer_stl_demo.py`, `scripts/train_neuromancer_stl.py` | Toy end-to-end demonstration |
| TorchPhysics | `src/neural_pde_stl_strel/frameworks/torchphysics_hello.py`, `scripts/train_burgers_torchphysics.py` | Import/probe plus draft benchmark script |
| PhysicsNeMo | `src/neural_pde_stl_strel/frameworks/physicsnemo_hello.py` | Import/integration probe only |
| RTAMT | `src/neural_pde_stl_strel/monitoring/rtamt_monitor.py` | Actual audit backend for diffusion |
| MoonLight | `src/neural_pde_stl_strel/monitoring/moonlight_helper.py` | Actual audit backend for the 2D heat example |

These are intentionally included so the logic layer is not repo-native only,
but they should not be over-claimed as full benchmark results unless figures and
summary tables are tied to them.

---

## 3. Where the core diagrams come from

| Need | Artifact |
|---|---|
| High-level block diagram | `figs/architecture_diagram.png` |
| Example-level data flow | `figs/training_pipeline.png` |
| Explicit formulas/specs | `docs/SPECIFICATIONS.md` |
| Runtime/cost snapshot | `results/benchmark_training.csv`, `figs/benchmark_cost.png` |
| Clear example inventory | `docs/EXAMPLE_STATUS.md`, `scripts/example_status.py` |

---

## 4. Which files should back tables in a paper draft

| Paper content | File(s) that should be treated as the source of truth |
|---|---|
| Main diffusion result table | `results/diffusion1d_main_summary.json` |
| Diffusion ablation table | `results/diffusion1d_ablation_summary.json`, `results/diffusion1d_ablation_summary.csv` |
| 2D monitoring threshold table | `results/heat2d_strel_monitoring.json`, `results/heat2d_threshold_analysis.csv` |
| Runtime/cost table | `results/benchmark_training.csv` |
| Spec table | `docs/SPECIFICATIONS.md` |

If prose, slides, or a paper draft disagree with those files, the result files
should win until the experiments are rerun and the summaries are regenerated.

---

## 5. Fast audit path for a fresh reader

If someone needs to understand the repo quickly, this is the recommended order:

1. `README.md`
2. `docs/EXAMPLE_STATUS.md`
3. `docs/SPECIFICATIONS.md`
4. `docs/CLAIMS_AND_EVIDENCE.md`
5. `figs/architecture_diagram.png`
6. `figs/training_pipeline.png`
7. `results/diffusion1d_main_summary.json`
8. `results/heat2d_strel_monitoring.json`
9. `scripts/example_status.py`

That path avoids getting lost in every helper script before the main story is
clear.
