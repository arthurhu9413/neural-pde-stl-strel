# Dataset and Benchmark Recommendations

This note separates **what is already implemented** from **what is worth adding
next**.

That distinction matters because the repository should not imply that every
interesting benchmark already has results behind it.

---

## 1. Implemented in the current repository

### 1D diffusion

- Type: synthetic PDE benchmark
- Logic: STL
- Status: training-time logic penalty plus offline auditing
- Why it matters: it is the clearest place to show the fidelity / robustness
  tradeoff and to explain the method end to end.

### 2D heat

- Type: synthetic PDE rollout
- Logic: STREL-style monitoring
- Status: monitoring case study
- Why it matters: it introduces a genuinely spatial field and produces figures
  where spatial logic is easy to interpret.

---

## 2. What a good next benchmark should do

A candidate benchmark is a strong fit if it satisfies most of these:

1. the output is naturally temporal or spatial-temporal,
2. a useful STL/STREL property can be written down without sounding contrived,
3. the result can be visualized clearly in one or two figures,
4. the run is repeatable on modest hardware, and
5. it strengthens the repository's main story instead of diluting it.

This project gains more from one well-chosen, clearly plotted 2D case study than
from a long list of unrelated benchmark names.

---

## 3. Best next benchmark: a second 2D PDE with a real spatial property

The strongest next step is still another visually clear **2D PDE** with a
training-time spatial property.

Best-fit examples:

- **2D diffusion-reaction** with region-of-interest cooling or containment,
- **2D shallow water** with "eventually no flooding in region R" style
  predicates,
- **2D Burgers / convection-diffusion** if the training path stays manageable,
- **2D advection-diffusion heat transport** if the paper wants a cleaner bridge
  to aerospace or thermal-engineering language.

Why this class of benchmark fits well:

- it keeps the paper centered on field-valued physical surrogates,
- it gives a natural reason to use spatial logic rather than scalar traces, and
- it complements the current 2D monitoring example.

---

## 4. Benchmark suites worth mining

### PDEBench

Why it is useful:

- it is a broad benchmark suite for Scientific ML,
- it includes time-dependent PDE tasks and code / data infrastructure,
- its published dataset spans examples such as 1D advection, 1D Burgers, 1D and
  2D diffusion-reaction, compressible Navier-Stokes, Darcy flow, and shallow
  water.

Why it helps this repo:

- it supplies PDEs with clear dynamics and benchmark culture,
- it provides realistic candidates for temporal or spatial safety properties.

Project: https://github.com/pdebench/PDEBench  
Dataset summary: https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi%3A10.18419%2Fdarus-2986

### PINNacle

Why it is useful:

- it is a broad benchmark for Physics-Informed Neural Networks,
- it spans more than 20 PDEs across heat conduction, fluid dynamics, biology,
  and electromagnetics,
- it is useful both for benchmark selection and for understanding what reviewers
  expect from a PINN evaluation section.

Project: https://github.com/i207M/PINNacle  
Paper / benchmark overview: https://proceedings.neurips.cc/paper_files/paper/2024/hash/8c63299fb2820ef41cb05e2ff11836f5-Abstract-Datasets_and_Benchmarks_Track.html

---

## 5. Aerospace-flavored options

### 5.1 NASA C-MAPSS is an **optional side branch**, but current access is mixed

C-MAPSS is relevant if the project needs an aerospace-oriented **temporal**
monitoring example such as:

- temperature or degradation envelopes that should stay within limits,
- "eventually enters degraded mode" style trend detection,
- rolling-window health constraints.

Current official pages are mixed rather than fully settled:

- the NASA Open Data Portal lists the dataset page and a `CMAPSSData.zip`
  resource, but
- the federal catalog entry for **C-MAPSS Aircraft Engine Simulator Data**
  still carries a note saying C-MAPSS and C-MAPSS40K are currently
  unavailable for download.

For repository planning, the safest conclusion is not "available" or
"unavailable", but **availability is unstable enough that the paper should not
depend on it**.

Why it still should not be the critical path:

- it moves the paper away from field-valued PDE surrogates toward degradation /
  remaining-useful-life time series,
- it introduces a very different evaluation culture from the repo's current PINN
  case studies,
- it is easier to explain as an auxiliary temporal benchmark than as the paper's
  central result.

Useful official links:

- https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
- https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data/resource/5224bcd1-ad61-490b-93b9-2817288accb8
- https://catalog.data.gov/dataset/c-mapss-aircraft-engine-simulator-data
- https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

### 5.2 Cleaner aerospace-PDE path: aerothermal / TPS-style heat transfer

If the paper needs an aerospace flavor **without** abandoning the PDE-centric
story, a thermal benchmark is usually cleaner than C-MAPSS.

Why:

- the repository already has a heat-transfer visual language,
- safety properties such as "surface temperature never exceeds limit" or
  "eventually cools below threshold in region R" map naturally to STL/STREL,
- thermal protection and aerothermal analyses are already part of NASA's public
  technical framing, so the motivation remains recognizable.

Useful public starting points:

- NASA overview of thermal protection systems:
  https://www.nasa.gov/reference/jsc-thermal-protection-systems/
- NASA Thermal Protection Materials Branch:
  https://www.nasa.gov/thermal-protection-materials-branch/
- NASA design / analysis note on thermal-response tools:
  https://www.nasa.gov/general/thermal-protection-materials-branch-design-and-analysis/

For this repository, a compact 2D advection-diffusion or heat equation with a
region-of-interest cooling / containment property is more coherent than jumping
straight to a large aerospace system benchmark.

---

## 6. Recommendation order

If time is limited, the most coherent order is:

1. add a second 2D PDE benchmark with spatial logic in the loop,
2. add one cleaner external-framework benchmark, most likely TorchPhysics,
3. use PDEBench / PINNacle to justify benchmark choice and reporting style,
4. treat C-MAPSS as optional venue-specific seasoning rather than the core plan.

---

## 7. What not to do

- Do not add a benchmark just because it is famous.
- Do not add a dataset if the logic specification is weak or artificial.
- Do not mix monitoring-only and training-time results in the same table without
  labeling them clearly.
- Do not make paper claims depend on data that the repository cannot currently
  access or reproduce.
- Do not say a benchmark is unavailable unless the official data page was
  checked recently.
