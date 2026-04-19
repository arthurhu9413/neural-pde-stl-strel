# Framework Survey

This note compares the external physics-ML and logic tools that matter for this
repository and, just as importantly, records **what is actually evidenced here**
versus what is only an integration path.

The repository's main experiments come from the repo-native PyTorch PINN code.
That is intentional. The native path makes the PDE residual, the field
sampling, the STL/STREL reductions, and the loss terms easy to inspect. The
external frameworks are included to show where the same logic layer could
attach, not to pretend that every framework already backs a paper-quality
benchmark.

---

## 1. What problem the framework has to solve

A good framework for this project has to do more than train a PDE surrogate. It
has to support four operations cleanly:

1. produce a field-valued prediction ``ûθ(x, t[, y])``,
2. expose that field on a grid that can be monitored,
3. allow a logic-derived penalty or constraint to enter the objective, and
4. preserve an artifact trail that can be audited after training.

A framework that is excellent for PDEs but awkward for custom logic losses is
not automatically the best fit for this repository.

---

## 2. Tool snapshot

| Tool / framework | Official project emphasis | Natural logic-attachment point | Evidence in this repo | Status in reported results |
|---|---|---|---|---|
| **Repo-native PyTorch PINN** | Small, explicit training loops and direct control of autograd | Directly inside the loss computation | `scripts/train_diffusion_stl.py`, `src/neural_pde_stl_strel/experiments/heat2d.py` | **Primary experimental path** |
| **Neuromancer** | Differentiable programming for constrained optimization, physics-informed system identification, and control | Loss / constraint objects over model outputs | `src/neural_pde_stl_strel/frameworks/neuromancer_stl_demo.py`, `scripts/train_neuromancer_stl.py` | **Toy end-to-end demo** |
| **TorchPhysics** | PyTorch-based deep learning methods for differential equations, inverse problems, and operator learning | Extra condition / penalty term on sampled fields | `src/neural_pde_stl_strel/frameworks/torchphysics_hello.py`, `scripts/train_burgers_torchphysics.py` | **Probe plus draft benchmark** |
| **PhysicsNeMo** | NVIDIA Physics AI framework for building, training, fine-tuning, and inferring SciML models | Custom loss components, validators, callbacks | `src/neural_pde_stl_strel/frameworks/physicsnemo_hello.py` | **Import probe only** |
| **RTAMT** | Online and offline STL monitoring for dense-time and discrete-time signals | Post-training scalar-trace auditing | `scripts/eval_diffusion_rtamt.py` | **Used in diffusion auditing** |
| **MoonLight** | Monitoring of temporal, spatial, and spatio-temporal properties of distributed systems | Post-training spatial / STREL-style auditing | `scripts/eval_heat2d_moonlight.py` | **Used in heat monitoring** |

---

## 3. Why the repo-native PyTorch path is still the main one

The current paper story depends on being able to answer very concrete questions:

- what the PDE is,
- what signal is extracted from the field,
- which STL/STREL formula is monitored,
- how the logic term enters the loss, and
- what fidelity / runtime cost comes from that choice.

The repo-native path is the cleanest way to answer those questions because the
field tensor, the reduction, and the robustness term are all visible in a small
amount of code.

That is why the committed diffusion figures and the main tables come from the
repo-native path rather than from a heavier framework wrapper.

---

## 4. What each external framework contributes

### Neuromancer

Why it remains attractive:

- it is explicitly built around composable differentiable objectives and
  constraints,
- it already fits the language of "physics loss plus logic penalty", and
- the toy demo in this repo shows the minimum viable attachment point.

What is still missing:

- a PDE benchmark in Neuromancer that mirrors the committed diffusion case
  study closely enough to compare figures and metrics side by side.

### TorchPhysics

Why it is the most natural next external benchmark:

- it is PDE-native,
- it exposes conditions and samplers in a way that maps reasonably well onto
  logic-aware penalties, and
- Burgers or diffusion-style examples are already close to the repo's current
  narrative.

What is still missing:

- a completed end-to-end benchmark in this repository with figures, monitored
  specs, and a summary table that can be cited directly.

### PhysicsNeMo

Why it matters:

- it is a serious Physics AI framework with a broader engineering focus,
- it is relevant if the project later scales to larger surrogate workflows.

Why it is not the main reported path here:

- the current repository only includes an import / integration probe,
- the present case studies are CPU-first and intentionally lightweight.

That is a scope decision, not a criticism of PhysicsNeMo.

---

## 5. Logic-layer comparison

| Logic tool | Role here | Strength | Limitation in this repo |
|---|---|---|---|
| **Repo smooth STL** | Training-time penalty and quick audits | Fully differentiable, easy to wire into PyTorch, transparent code path | Heuristic regularizer; not a proof method |
| **RTAMT** | Offline audit for scalar temporal traces | Mature STL monitoring workflow and direct formula-based auditing | Requires field-to-scalar reduction first |
| **MoonLight** | Spatial-temporal audit for the 2D heat example | Natural fit for spatial predicates and STREL-style properties | Current repo example is monitoring-only |

---

## 6. How to describe the framework story honestly

Safe wording:

- "repo-native main benchmark",
- "Neuromancer toy demo",
- "TorchPhysics draft integration path",
- "PhysicsNeMo import probe",
- "RTAMT / MoonLight audit backends".

Unsafe wording unless new evidence is added:

- "all frameworks were benchmarked",
- "framework-agnostic performance comparison",
- "complete external integrations",
- "formal verification across all toolchains".

---

## 7. Recommendation for the next strong benchmark

If only one external framework benchmark is added next, the best choice is
usually **TorchPhysics**.

Reason:

- it stays close to the PDE-centric story,
- it is easier to explain than a much larger engineering stack, and
- it complements rather than duplicates the Neuromancer toy demo.

Neuromancer should still stay in the repo because it demonstrates the most
obvious "logic as differentiable constraint" pathway.

---

## 8. Primary project links

- Neuromancer: https://github.com/pnnl/neuromancer
- Neuromancer docs: https://pnnl.github.io/neuromancer/
- PhysicsNeMo: https://github.com/NVIDIA/physicsnemo
- PhysicsNeMo docs: https://docs.nvidia.com/physicsnemo/latest/index.html
- TorchPhysics: https://github.com/boschresearch/torchphysics
- TorchPhysics docs: https://boschresearch.github.io/torchphysics/
- RTAMT: https://github.com/nickovic/rtamt
- MoonLight: https://github.com/MoonLightSuite/moonlight
- DeepXDE: https://github.com/lululxvi/deepxde
