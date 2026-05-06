# Checked Reference Pool

This file is a safer alternative to assembling citations from memory under time
pressure.

It is split into two parts:

1. **checked references**: core papers, tools, and official docs that were
   re-opened against an official or primary page,
2. **candidate expansions**: likely useful additions that should still be
   reopened directly before they enter a manuscript.

That split is intentional. It keeps the repository helpful without pretending
that every possible reference has already been polished into final BibTeX.

Checked on: **2026-03-17**.

---

## A. Checked core references

### A1. STL and quantitative monitoring

1. **Oded Maler and Dejan Nickovic (2004)**  
   *Monitoring Temporal Properties of Continuous Signals*. FORMATS/FTRTFT 2004.  
   Why it matters: canonical early STL monitoring reference.  
   URL: https://link.springer.com/chapter/10.1007/978-3-540-30206-3_12

2. **Georgios E. Fainekos and George J. Pappas (2009)**  
   *Robustness of Temporal Logic Specifications for Continuous-Time Signals*. Theoretical Computer Science.  
   Why it matters: standard quantitative robustness semantics.  
   URL: https://doi.org/10.1016/j.tcs.2009.06.021

3. **Dejan Nickovic and Tomoyuki Yamaguchi (2020)**  
   *RTAMT: Online Robustness Monitors from STL*. ATVA 2020 / arXiv.  
   Why it matters: direct monitor backend used by this repo for diffusion auditing.  
   URL: https://arxiv.org/abs/2005.11827

4. **RTAMT project**  
   Why it matters: implementation reference for scalar STL auditing.  
   URL: https://github.com/nickovic/rtamt

### A2. Spatial and spatio-temporal logic

5. **Ezio Bartocci, Luca Bortolussi, Michele Loreti, Laura Nenzi, Simone Silvetti (2021)**  
   *MoonLight: A Lightweight Tool for Monitoring Spatio-Temporal Properties*.  
   Why it matters: tool paper for the MoonLight line used in the 2D case study.  
   URL: https://arxiv.org/abs/2104.14333

6. **Laura Nenzi, Ezio Bartocci, Luca Bortolussi, Michele Loreti (2022)**  
   *A Logic for Monitoring Dynamic Networks of Spatially-distributed Cyber-Physical Systems*. Logical Methods in Computer Science.  
   Why it matters: strong STREL-oriented semantics reference.  
   URL: https://lmcs.episciences.org/8936/pdf

7. **MoonLight project**  
   Why it matters: official codebase for the spatial-temporal monitoring workflow used here.  
   URL: https://github.com/MoonLightSuite/moonlight

8. **SpaTiaL project**  
   Why it matters: neighboring spatial-specification toolchain worth citing in related work.  
   URL: https://github.com/KTH-RPL-Planiacs/SpaTiaL

### A3. Differentiable STL and learning with logic

9. **Meiyi Ma et al. (2020)**  
   *STLnet: Signal Temporal Logic Enforced Multivariate Recurrent Neural Networks*. NeurIPS 2020.  
   Why it matters: direct motivation for using logic as a training signal.  
   URL: https://proceedings.neurips.cc/paper/2020/hash/a7da6ba0505a41b98bd85907244c4c30-Abstract.html

10. **STLnet code**  
    Why it matters: implementation reference for the original logic-enforced training idea.  
    URL: https://github.com/meiyima/STLnet

11. **P. Kapoor, K. Mizuta, E. Kang, K. Leung (2025)**  
    *STLCG++: A Masking Approach for Differentiable Signal Temporal Logic Specification*. IEEE RA-L / arXiv.  
    Why it matters: recent differentiable-STL point of comparison, especially on efficiency.  
    URL: https://arxiv.org/abs/2501.04194

12. **Mark Chevallier, Filip Smola, Richard Schmoetten, Jacques D. Fleuriot (2025)**  
    *GradSTL: Comprehensive Signal Temporal Logic for Neurosymbolic Reasoning and Learning*. TIME 2025.  
    Why it matters: recent correctness-oriented differentiable STL implementation.  
    URL: https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.TIME.2025.6

### A4. PINNs, neural operators, and physics-ML foundations

13. **Maziar Raissi, Paris Perdikaris, George Em Karniadakis (2017/2019)**  
    *Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations*.  
    Why it matters: preprint lineage behind the canonical PINN formulation.  
    URL: https://arxiv.org/abs/1711.10561

14. **George Em Karniadakis et al. (2021)**  
    *Physics-informed Machine Learning*. Nature Reviews Physics.  
    Why it matters: broad review for positioning physics-ML / PINN work.  
    URL: https://doi.org/10.1038/s42254-021-00314-5

15. **Lu Lu et al. (2019/2021)**  
    *DeepXDE: A Deep Learning Library for Solving Differential Equations*.  
    Why it matters: widely cited PINN library reference.  
    URL: https://epubs.siam.org/doi/10.1137/19M1274067

16. **Ricky T. Q. Chen et al. (2018)**  
    *Neural Ordinary Differential Equations*. NeurIPS 2018 / arXiv.  
    Why it matters: foundational neural ODE reference.  
    URL: https://arxiv.org/abs/1806.07366

17. **Lu Lu et al. (2019)**  
    *DeepONet: Learning Nonlinear Operators for Identifying Differential Equations Based on the Universal Approximation Theorem of Operators*.  
    Why it matters: standard operator-learning reference for PDE surrogates.  
    URL: https://arxiv.org/abs/1910.03193

18. **Zongyi Li et al. (2020/2021)**  
    *Fourier Neural Operator for Parametric Partial Differential Equations*.  
    Why it matters: core neural-operator baseline family for PDE learning.  
    URL: https://arxiv.org/abs/2010.08895

### A5. Benchmarks and evaluation culture

19. **Makoto Takamoto et al. (2022)**  
    *PDEBench: An Extensive Benchmark for Scientific Machine Learning*. NeurIPS Datasets and Benchmarks.  
    Why it matters: benchmark suite for time-dependent PDE tasks.  
    URL: https://proceedings.neurips.cc/paper_files/paper/2022/hash/0a9747136d411fb83f0cf81820d44afb-Abstract-Datasets_and_Benchmarks.html

20. **PDEBench project**  
    Why it matters: official code and dataset entry point.  
    URL: https://github.com/pdebench/PDEBench

21. **Hao Zhongkai et al. (2024)**  
    *PINNacle: A Comprehensive Benchmark of Physics-Informed Neural Networks for Solving PDEs*. NeurIPS Datasets and Benchmarks.  
    Why it matters: broad PINN benchmark and reporting model.  
    URL: https://proceedings.neurips.cc/paper_files/paper/2024/hash/8c63299fb2820ef41cb05e2ff11836f5-Abstract-Datasets_and_Benchmarks_Track.html

22. **PINNacle project**  
    Why it matters: codebase and benchmark definitions.  
    URL: https://github.com/i207M/PINNacle

### A6. Frameworks actually discussed in this repository

23. **Neuromancer documentation**  
    Why it matters: official docs for the differentiable-programming framework referenced in the repo.  
    URL: https://pnnl.github.io/neuromancer/

24. **Neuromancer repository**  
    Why it matters: official source tree and project description.  
    URL: https://github.com/pnnl/neuromancer

25. **PhysicsNeMo documentation**  
    Why it matters: current official docs for NVIDIA's Physics AI framework.  
    URL: https://docs.nvidia.com/physicsnemo/latest/index.html

26. **PhysicsNeMo examples catalog**  
    Why it matters: official example inventory, including PDE and thermal-flow tasks.  
    URL: https://docs.nvidia.com/physicsnemo/latest/examples_catalog.html

27. **PhysicsNeMo repository**  
    Why it matters: official source repository.  
    URL: https://github.com/NVIDIA/physicsnemo

28. **TorchPhysics documentation**  
    Why it matters: official docs for the PDE-focused PyTorch framework referenced in the repo.  
    URL: https://boschresearch.github.io/torchphysics/

29. **TorchPhysics repository**  
    Why it matters: official project source.  
    URL: https://github.com/boschresearch/torchphysics

### A7. Dataset and application pages that affect planning

30. **NASA CMAPSS dataset page**  
    Why it matters: official access point for the aerospace temporal side branch.  
    URL: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

31. **NASA CMAPSS zip resource page**  
    Why it matters: shows that the NASA Open Data Portal currently lists a direct `CMAPSSData.zip` resource.  
    URL: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data/resource/5224bcd1-ad61-490b-93b9-2817288accb8

32. **Catalog.data.gov C-MAPSS entry**  
    Why it matters: the federal catalog still carries a note that C-MAPSS and C-MAPSS40K are currently unavailable for download, which is why repo planning should treat availability as unstable rather than settled.  
    URL: https://catalog.data.gov/dataset/c-mapss-aircraft-engine-simulator-data

33. **NASA PCoE data repository page**  
    Why it matters: legacy context for prognostics datasets, including C-MAPSS mentions.  
    URL: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

### A8. Aerospace / thermal motivation references

34. **NASA Thermal Protection Systems overview**  
    Why it matters: accessible high-level justification for thermal safety / cooling properties.  
    URL: https://www.nasa.gov/reference/jsc-thermal-protection-systems/

35. **NASA Thermal Protection Materials Branch**  
    Why it matters: public description of thermal-protection materials and modeling work.  
    URL: https://www.nasa.gov/thermal-protection-materials-branch/

36. **NASA Thermal Protection Materials Branch – Design and Analysis**  
    Why it matters: directly mentions thermal-response tools such as FIAT.  
    URL: https://www.nasa.gov/general/thermal-protection-materials-branch-design-and-analysis/

37. **Kathryn E. Wurster, Christopher J. Riley, E. Vincent Zoby (1998)**  
    *Engineering Aerothermal Analysis for X-34 Thermal Protection Design*.  
    Why it matters: concrete aerothermal/TPS example if the paper wants an aerospace-results bridge.  
    URL: https://ntrs.nasa.gov/api/citations/19980025468/downloads/19980025468.pdf

38. **M. Mahzari et al. (2022)**  
    *Development and Sizing of the Mars 2020 Thermal Protection System*.  
    Why it matters: modern TPS application reference.  
    URL: https://ntrs.nasa.gov/api/citations/20220006688/downloads/Development%20and%20Sizing%20of%20the%20Mars%202020%20Thermal%20Protection%20System_AIAA%20Aviation%202022.pdf

39. **S. A. Tobin et al. (2024)**  
    *LOFTID Aeroshell Thermal Response Uncertainty Analysis at Aerocapture and Entry Conditions*.  
    Why it matters: recent thermal-response modeling reference for entry systems.  
    URL: https://ntrs.nasa.gov/api/citations/20230017339/downloads/LOFTID%20THERMAL%20UNCERTAINTY%20SciTech2024%20RevH.pdf

---

## B. Candidate expansion references

These look useful, but they should still be re-opened directly before they are
inserted into the manuscript or BibTeX.

1. **Alexandre Donzé** — *Efficient Robust Monitoring for STL*  
   Good if the paper wants a stronger robustness-computation lineage.

2. **Ezio Visconti et al. (2021)** — *Online Monitoring of Spatio-Temporal Properties for Imprecise Signals*  
   Useful if the spatial-monitoring section wants a second MoonLight-family tool paper.

3. **Deep Learning Methods for Partial Differential Equations and Hidden Physics Models**  
   Useful for broad PDE-learning related work beyond PINNs.

4. **When and Why PINNs Fail to Train: A Neural Tangent Kernel Perspective**  
   Useful if the discussion section talks about optimization difficulty.

5. **Is Physics-Informed Loss Always Suitable for Training Physics-Informed Neural Networks?**  
   Useful if the paper wants a deeper discussion of loss-design pathologies.

6. **NSFnets (Navier-Stokes Flow Nets)**  
   Useful if fluid examples become more central.

7. **Error Estimates for DeepONets**  
   Useful if operator-learning baselines become part of the discussion.

8. **Advanced Lightweight TUFROC Thermal Protection System**  
   Useful if the aerospace angle moves toward reusable TPS materials.

9. **Thermal Protection Materials and Systems: Past, Present, and Future**  
   Useful as a broader survey citation for the aerospace-thermal motivation.

10. **Overview of Thermal Protection System Facility (TPSF)**  
    Useful if the paper wants to cite current NASA TPS engineering capability.

11. **Reusable Thermal Protection Material Development (2025)**  
    Useful for a more recent NASA materials reference.

12. **Recent Advances in Physics-Informed Machine Learning (2024)**  
    Useful as a modern survey, especially if the paper needs extra context on SciML libraries.

13. **Safe Physics-informed Machine Learning for Dynamics and Control (2025 tutorial)**  
    Useful if the discussion explicitly connects to safe control.

14. **Additional benchmark-suite papers linked from PINNacle and PDEBench**  
    Useful if a benchmark section grows and needs more domain-specific task citations.

15. **Aerospace heat-transfer / re-entry analysis reports on NTRS**  
    Useful if the final 2D example is framed as an aerothermal toy benchmark.

16. **Recent differentiable-STL libraries beyond STLCG++ and GradSTL**  
    Useful if the method comparison section is expanded.

---

## C. Practical citation rules for this repo

Before moving any entry from this file into the actual paper:

- reopen the source directly,
- verify author list, title, venue, and year,
- make sure the paper is relevant to a sentence you actually need,
- prefer primary sources over tertiary summaries,
- do not cite a tool or paper that nobody on the author list has read.

A smaller, accurate bibliography is better than a longer one with one fake or
mis-stated citation.
