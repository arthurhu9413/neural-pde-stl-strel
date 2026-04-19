# Publication Notes and Revision Priorities

This note keeps manuscript work aligned with the repository's **actual
artifacts** rather than with aspirational claims.

The strongest framing for the project is:

> a reproducible logic-aware neural-PDE scaffold with one end-to-end temporal
> training case study and one spatial monitoring case study.

That framing is narrower than a sweeping "first" or "best" claim, but it is
much easier to defend from the committed code, result files, and figures.

---

## 1. Defensible manuscript claim

A strong, honest manuscript-level claim is:

> We show that temporal and spatial logic can be attached to compact
> physics-informed neural-PDE workflows in a reproducible way, and that doing so
> exposes a measurable tradeoff between physical fidelity and specification
> robustness.

That statement is directly supported by the repository's current evidence.

### What the repo can support today

1. **A reproducible logic-aware workflow** with explicit artifact regeneration.
2. **A 1D diffusion benchmark** showing the tradeoff between PDE fidelity and
   specification robustness under a differentiable STL penalty.
3. **A 2D heat monitoring case study** showing why a spatial view matters for
   field-valued outputs.
4. **Documented integration surfaces** for Neuromancer, TorchPhysics, and
   PhysicsNeMo.

### What the repo should not overclaim

- that the 2D heat case is already a full STREL-constrained training result,
- that every external framework probe is already a benchmark-quality
  integration,
- that the repository proves formal correctness of the learned surrogates,
- that the approach is broadly the first or best of its kind.

---

## 2. Current evidence inventory

| Artifact | What it supports |
|---|---|
| `results/diffusion1d_main_summary.json` | main diffusion result table |
| `results/diffusion1d_ablation_summary.json` | lambda-tradeoff table / plot source |
| `figs/diffusion1d_comparison.png` | qualitative field comparison |
| `figs/diffusion1d_training_dynamics.png` | fidelity / robustness dynamics |
| `results/heat2d_strel_monitoring.json` | 2D spatial monitoring table |
| `figs/heat2d_field_evolution.png` | spatial qualitative evidence |
| `figs/heat2d_stl_traces.png` | threshold / quench traces |
| `results/benchmark_training.csv` | runtime / cost snapshot |
| `figs/architecture_diagram.png` | high-level system overview |
| `figs/training_pipeline.png` | example-level data flow |

These files should anchor any manuscript tables, captions, and claims.

---

## 3. Minimum figure and table set

### Figures worth keeping

1. high-level system architecture,
2. detailed training data-flow diagram,
3. diffusion field comparison,
4. diffusion training dynamics,
5. lambda-ablation tradeoff plot,
6. 2D heat field evolution,
7. 2D threshold / quench traces.

### Tables worth keeping

1. explicit monitored formulas,
2. main diffusion results,
3. 2D threshold / quench analysis,
4. runtime / cost snapshot.

If a figure or table does not answer a concrete technical question, it should
probably stay in the repository and out of the main manuscript.

---

## 4. Writing rules

### Do

- write the exact STL / STREL formulas used in the experiments,
- define `λ` the first time it appears,
- keep notation stable once introduced,
- say explicitly when a result is **monitoring-only**,
- make the tradeoff between robustness, fidelity, and runtime visible,
- verify every citation and every table entry against the source artifacts.

### Do not

- imply formal verification when the repo performs monitoring or regularization,
- bury the cost of the logic term behind a single accuracy number,
- describe the framework probes as full benchmark integrations,
- let prose outrun what the committed artifacts actually show.

---

## 5. Highest-value next extensions

If there is time for one substantial addition, the best next result is still:

- a second **2D** benchmark with a genuine spatial training-time penalty.

Other strong additions would be:

- clearer runtime comparisons across lambda values,
- one cleaner external-framework benchmark,
- a stronger ablation around the robustness-to-loss mapping.

---

## 6. Release checklist

Before tagging a release or syncing manuscript prose to the repo:

- refresh committed summaries,
- regenerate figures,
- run tests,
- inspect every equation against the implementation,
- inspect every caption against the plotted quantity,
- verify every citation by hand,
- check that README prose still matches the committed artifacts.

The safest sentence in a manuscript is the one that can be traced directly back
into the repository.
