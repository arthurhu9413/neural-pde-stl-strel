# Example Status

This file is a blunt inventory of what is implemented in the repository right
now.

It exists so readers do not have to infer from scattered scripts whether an
example is a smoke test, a complete benchmark, or just an idea.

---

## Status legend

- **Reported result**: appears in the README, figures, or committed summary
  tables.
- **Runnable demo**: executable and useful for illustration, but not a primary
  paper result.
- **Probe**: confirms that a framework or dependency path can be imported or
  lightly exercised.
- **Planned**: discussed in docs or scripts, but not yet a committed benchmark.

---

## Current inventory

| Example / path | Logic | Training-time constraint? | Monitoring backend | Status | Main evidence |
|---|---|---:|---|---|---|
| 1D diffusion baseline | None | No | Optional RTAMT audit | **Reported result** | `results/diffusion1d_main_summary.json` |
| 1D diffusion + STL regularization | STL | Yes | RTAMT + exact sampled audit | **Reported result** | `results/diffusion1d_main_summary.json`, diffusion figures |
| 1D diffusion lambda ablation | STL | Yes | Exact sampled audit | **Reported result** | `results/diffusion1d_ablation_summary.json` |
| 2D heat committed rollout + STREL-style monitoring | STREL-style containment/quench | No | MoonLight-oriented audit | **Reported result** | `results/heat2d_strel_monitoring.json`, heat figures |
| 2D heat + scalar STL safety | STL | Yes | repo smooth STL | **Runnable demo / stepping stone** | `configs/heat2d_stl_safe.yaml` |
| 2D heat + scalar STL eventually-cool | STL | Yes | repo smooth STL | **Runnable demo / stepping stone** | `configs/heat2d_stl_eventually.yaml` |
| Neuromancer toy signal example | STL | Yes | repo smooth STL | **Runnable demo** | `scripts/train_neuromancer_stl.py` |
| TorchPhysics Burgers draft | STL planned | Partial | none yet | **Planned / draft** | `scripts/train_burgers_torchphysics.py` |
| PhysicsNeMo hello probe | None | No | none | **Probe** | `src/neural_pde_stl_strel/frameworks/physicsnemo_hello.py` |
| SpaTiaL / spatial-spec hello | spatial relations | No | spatial-spec | **Probe** | `src/neural_pde_stl_strel/frameworks/spatial_spec_hello.py` |

---

## What is safe to show in a talk or report today

The strongest evidence-backed story is:

1. **A training-time result** on 1D diffusion showing the STL fidelity/
   robustness tradeoff.
2. **A spatial monitoring result** on 2D heat showing where STREL-style logic
   matters for field-valued outputs.
3. **Framework flexibility evidence** via Neuromancer/TorchPhysics/
   PhysicsNeMo probes, with careful wording that they are not all full
   benchmarks yet.

That story is honest and already supported by committed artifacts.

---

## What still needs work before it should be described as a stronger paper result

The biggest remaining gap is a second **2D benchmark with training-time spatial
logic in the loop**.

Until that exists, the repository should keep saying:

- the 2D heat case is a monitoring result,
- scalar STL on heat is an intermediate step,
- external framework demos are integration evidence, not the headline results.

---

## Quick command

To print a machine-readable summary for a fresh reader:

```bash
python scripts/example_status.py
```
