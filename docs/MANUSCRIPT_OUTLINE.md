# Manuscript Outline and Writing Notes

This document is a working manuscript plan for the STL / STREL + neural-PDE
project. It is built around the repository's **actual evidence**, not around an
ideal future paper.

A strong manuscript has to answer three questions clearly and early:

1. **What technical problem is being solved?**
2. **What do we gain by combining logic monitoring with neural-PDE / physics-ML models?**
3. **Which results are already implemented, and which are still future work?**

If those points stay muddy, the draft reads like a tutorial instead of a
research paper.

---

## 1. Candidate manuscript claim

A restrained claim that matches the repository today is:

> We present a reproducible scaffold for attaching STL/STREL monitoring and a
> simple differentiable STL penalty to physics-informed neural-PDE models, and
> we evaluate the scaffold on one training-time diffusion case study plus one
> spatial monitoring case study.

That is narrower than "first" or "best", but it is defensible.

---

## 2. Working hypothesis

The empirical hypothesis should be explicit:

- adding an STL penalty to a PINN can move the learned field toward desired
  temporal properties,
- doing so introduces a measurable fidelity and runtime tradeoff, and
- spatial monitoring becomes important as soon as the model output is a field
  rather than a scalar trace.

That hypothesis gives the results section a clear target.

---

## 3. Core contributions the manuscript can honestly claim today

1. **A reproducible logic-aware training / audit scaffold** for compact
   physics-informed neural-PDE models.
2. **A 1D diffusion case study** showing how an STL penalty changes robustness,
   PDE error, and runtime.
3. **A 2D heat monitoring case study** showing how a spatial / STREL-style view
   changes what is visible in a field-valued output.
4. **A documented integration map** for Neuromancer, TorchPhysics, and
   PhysicsNeMo.

---

## 4. Recommended structure

### 4.1 Introduction

The introduction should do three jobs in about one page:

- motivate why temporal / spatial requirements matter for learned physical
  surrogates,
- explain why field outputs require more than scalar error metrics,
- state exactly what the repository contributes.

The final paragraph of the introduction should answer:

- what the method takes as input,
- what it returns,
- what case studies are included,
- what the main empirical takeaway is.

### 4.2 Problem setup

Write the setup with equations, not just prose:

- define the PDE,
- define the neural surrogate,
- define the field-to-signal reduction,
- write the exact STL / STREL formulas,
- define the training objective.

Keep the notation for `λ` fixed once introduced.

### 4.3 Method

Break the method into pieces:

1. field sampling,
2. scalar reduction or spatial predicate construction,
3. robustness evaluation,
4. robustness-to-loss mapping,
5. integration into the training loop.

Use the block diagram and the example-level data-flow figure already in the
repository.

### 4.4 Case studies

Separate the cases cleanly:

- **Case study 1:** 1D diffusion with training-time STL regularization.
- **Case study 2:** 2D heat with spatial monitoring.

Do not blur the second into a training-time STREL claim unless that experiment
actually exists.

### 4.5 Results

Each figure should answer a concrete question:

- Did robustness improve?
- What fidelity was lost?
- What runtime cost appeared?
- What does the field actually look like?
- Where does spatial reasoning matter?

### 4.6 Related work

Organize the section by topic, not by a sequence of one-line summaries:

- STL and runtime monitoring,
- spatial / spatio-temporal logic,
- differentiable STL,
- PINNs and physics-ML frameworks,
- benchmarks and evaluation culture.

### 4.7 Discussion and limitations

Say clearly:

- monitoring is not formal verification,
- spatial quantification is discretized on a sampled grid,
- the 2D case study is currently monitoring-only,
- framework probes are not benchmark-quality results.

---

## 5. Minimum figure set

A strong manuscript should have at least:

1. high-level system architecture,
2. detailed training data-flow diagram,
3. diffusion field comparison,
4. diffusion training dynamics,
5. lambda ablation tradeoff plot,
6. 2D heat evolution snapshots,
7. 2D spatial-max / threshold traces,
8. runtime / cost table or plot.

If a figure does not answer a manuscript question, cut it.

---

## 6. Table checklist

At minimum:

- one table of the audited formulas,
- one table of the main diffusion results,
- one table of the threshold / quench analysis for the 2D heat rollout,
- one runtime / cost table.

Every number in a table should be traceable to a committed artifact or a script.

---

## 7. Writing rules

### Do

- verify every citation by hand,
- write the actual formulas used,
- keep notation consistent,
- say when a result is monitoring-only,
- use consistent American English,
- make the contribution / hypothesis explicit.

### Do not

- imply "first", "best", or "verified" unless the evidence really supports it,
- fill the background with high-level tutorial prose instead of technical detail,
- cite papers that were not checked directly,
- hand-enter final result tables without a regeneration path.

---

## 8. Additional experiments that would strengthen the manuscript

The highest-value additions are:

1. one more 2D benchmark with a real spatial training-time penalty,
2. stronger runtime comparisons across lambda values and monitors,
3. one cleaner external-framework benchmark, probably TorchPhysics,
4. an aerospace-flavored temporal benchmark only if the data path is stable.

---

## 9. Current repo-to-manuscript mapping

| Repo artifact | Manuscript role |
|---|---|
| `figs/architecture_diagram.png` | system overview |
| `figs/training_pipeline.png` | method / data flow |
| `results/diffusion1d_main_summary.json` | main diffusion table |
| `results/diffusion1d_ablation_summary.json` | lambda ablation table / plot source |
| `figs/diffusion1d_comparison.png` | visual diffusion result |
| `figs/diffusion1d_training_dynamics.png` | training tradeoff plot |
| `results/heat2d_strel_monitoring.json` | 2D spatial monitoring table |
| `figs/heat2d_field_evolution.png` | 2D qualitative figure |
| `results/benchmark_training.csv` | cost table |

---

## 10. Final checklist before syncing prose back to the repo

Before updating manuscript-facing documentation:

- refresh summaries,
- regenerate figures,
- run tests,
- inspect every caption,
- inspect every reference,
- inspect every equation against the code,
- read the draft once as a standalone document instead of as a repo insider.

That last step catches the places where the manuscript quietly assumes too much
repository knowledge.
