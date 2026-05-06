# Paper Positioning Notes

This note is not a manuscript draft. It is a guardrail against vague or
inflated claims.

The repository is strongest when it is framed as:

> a reproducible logic-aware neural-PDE scaffold with one training-time case
> study and one spatial monitoring case study.

That framing is narrower than a grand "first/best" claim, but it is much easier
to defend from the committed artifacts. It also lines up better with a focused
case-study-style manuscript than with a sweeping original-method claim; see
`docs/PUBLICATION_NOTES.md`.

---

## 1. What problem the paper is solving

The paper is not about adding formal-methods vocabulary to a PINN for style.

It is about a concrete gap:

- a field-valued neural surrogate may fit the PDE residual reasonably well,
- but the same surrogate may still violate a temporal or spatial requirement
  that is natural in the application.

The method matters if it helps answer either of these questions:

1. **audit question**: does the learned field satisfy the desired property?
2. **training question**: can we bias training toward larger robustness margins?

That is the real problem statement. Everything else should serve it.

---

## 2. A defensible claim for the current repository

A strong, honest claim is:

> We show that temporal and spatial logic can be attached to compact
> physics-informed neural PDE workflows in a reproducible way, and that doing so
> exposes a measurable tradeoff between physical fidelity and specification
> robustness.

That claim is directly supported by:

- the diffusion STL-regularization experiment,
- the lambda ablation,
- the 2D heat monitoring example,
- the runtime snapshot,
- the architecture and data-flow diagrams.

---

## 3. What the contribution is not

Do **not** imply any of the following unless new experiments are added:

- that the 2D heat example is already a full STREL-constrained training result,
- that all external framework probes are benchmark-quality integrations,
- that the method proves formal correctness of the learned surrogate,
- that the work is broadly the first or best.

When in doubt, the safe verbs are:

- **monitor**,
- **audit**,
- **regularize**,
- **bias training**,
- **case study**,
- **reproducible scaffold**.

---

## 4. What we gain by combining logic with neural-PDE / physics-ML models

This answer needs to be explicit in the paper.

### Benefit 1: the requirement is written at the right abstraction

Temporal or spatial logic expresses "always safe", "eventually cool", or
"contain the hotspot" more cleanly than a hand-built scalar penalty.

### Benefit 2: the same formula can be used twice

The logic specification can be used:

- softly during training through a differentiable robustness surrogate, and
- exactly or more faithfully during post-hoc auditing.

### Benefit 3: the tradeoff becomes inspectable

Once the logic term is explicit, the paper can report:

- robustness gain,
- PDE-fidelity loss,
- runtime cost.

That is a more informative story than accuracy alone.

---

## 5. Reviewer questions the paper should answer early

A reviewer or reader will naturally ask:

### What is the input?

For the main diffusion case:

- a PDE and IC/BCs,
- a neural field architecture,
- a field-to-signal reduction,
- an STL formula,
- a penalty weight ``λ``.

### What is the output?

- a trained surrogate,
- a sampled field,
- monitored robustness values,
- plots and summary artifacts.

### Why is spatial logic needed at all?

Because once the output is a field, the question is not only whether something
happened, but **where** it happened and whether it remained contained.

### What is the hypothesis?

That a logic penalty can move the model toward desired temporal behavior, but
that this comes with a measurable fidelity and runtime cost that should be
reported rather than hidden.

---

## 6. Best structure for a talk or paper

1. motivation and problem statement,
2. method inputs and outputs,
3. architecture and data-flow diagrams,
4. diffusion case study with equations and exact formulas,
5. diffusion tradeoff results and lambda ablation,
6. 2D spatial monitoring case study,
7. framework flexibility and current limits,
8. limitations and next steps.

That sequence matches the repository's strongest evidence and avoids sounding
like a tutorial.

---

## 7. Highest-value next extension

If there is time for one substantial addition, the best next result is:

- a second **2D** benchmark with a genuine spatial training-time penalty.

That would rebalance the paper from "1D training + 2D monitoring" toward a
stronger spatial story.

---

## 8. Writing discipline

Helpful reminders:

- write the actual formulas used in the experiments,
- keep notation stable once introduced,
- make the paper answer "why this matters" before listing tools,
- keep related work organized by idea, not by one-sentence paper summaries,
- do not let prose outrun experiments.

The more technical the paper becomes, the less it reads like generated filler.
