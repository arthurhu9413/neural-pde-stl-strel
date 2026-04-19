# Reading List

This is a **checked starter bibliography** for the topics that matter most to
this repository. It is narrower than a full paper bibliography, but it is meant
to be a safer foundation for manuscript writing than an ad hoc list of search
results. For a larger, manuscript-facing pool, see
[docs/VERIFIED_REFERENCE_POOL.md](VERIFIED_REFERENCE_POOL.md).

The rule for paper writing is still simple:

> every reference cited in a manuscript should be re-opened and verified again
> in context.

---

## 1. Signal Temporal Logic and monitoring

1. **O. Maler and D. Nickovic (2004)**  
   *Monitoring Temporal Properties of Continuous Signals*.  
   Classic STL monitoring reference.

2. **G. Fainekos and G. J. Pappas (2009)**  
   *Robustness of Temporal Logic Specifications for Continuous-Time Signals*.  
   Standard quantitative robustness reference.

3. **A. Donzé (2013)**  
   *Efficient Robust Monitoring for Signal Temporal Logic*.  
   Good background on offline robustness computation.

4. **D. Nickovic and T. Yamaguchi (2020)**  
   *RTAMT: Online Robustness Monitors from STL*.  
   Directly relevant because RTAMT is the scalar STL audit backend used here.  
   Paper: https://arxiv.org/abs/2005.11827  
   Project: https://github.com/nickovic/rtamt

---

## 2. Spatial and spatio-temporal logic

5. **E. Bartocci et al. (2021)**  
   *A Lightweight Tool for Monitoring Spatio-Temporal Properties*.  
   Tool paper for MoonLight and a good entry point for STREL-oriented
   monitoring.  
   Paper: https://arxiv.org/abs/2104.14333

6. **MoonLight project**  
   Official tool repository and examples.  
   Project: https://github.com/MoonLightSuite/moonlight

7. **SpaTiaL project**  
   Neighboring spatial-specification toolchain worth knowing, especially for
   comparison language in related work.  
   Project: https://github.com/KTH-RPL-Planiacs/SpaTiaL

---

## 3. Differentiable STL

8. **M. Ma et al. (2020)**  
   *STLnet: Signal Temporal Logic Enforced Multivariate Recurrent Neural Networks*.  
   Key motivation for "logic as a training signal".  
   Paper: https://proceedings.neurips.cc/paper/2020/hash/a7da6ba0505a41b98bd85907244c4c30-Abstract.html  
   Code: https://github.com/meiyima/STLnet

9. **P. Kapoor, K. Mizuta, E. Kang, and K. Leung (2025)**  
   *STLCG++: A Masking Approach for Differentiable Signal Temporal Logic Specification*.  
   Important recent point of comparison for differentiable STL efficiency.  
   Paper: https://arxiv.org/abs/2501.04194

10. **M. Chevallier et al. (2025)**  
    *GradSTL: Comprehensive Signal Temporal Logic for Neurosymbolic Reasoning and Learning*.  
    Relevant because it emphasizes comprehensive differentiable semantics and
    formal correctness of the implementation.  
    Paper: https://arxiv.org/abs/2508.04438  
    TIME 2025 version: https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.TIME.2025.6

---

## 4. Physics-informed ML and PINNs

11. **M. Raissi, P. Perdikaris, and G. E. Karniadakis (2019)**  
    *Physics-Informed Neural Networks: A Deep Learning Framework for Solving
    Forward and Inverse Problems Involving Nonlinear Partial Differential
    Equations*.  
    Core PINN reference.

12. **G. E. Karniadakis et al. (2021)**  
    *Physics-Informed Machine Learning*.  
    Broad review article.

13. **L. Lu et al. (2021)**  
    *DeepXDE: A Deep Learning Library for Solving Differential Equations*.  
    Useful framework reference even though this repo does not integrate it yet.  
    Project: https://github.com/lululxvi/deepxde

---

## 5. Benchmarks and evaluation culture

14. **M. Takamoto et al. (2022)**  
    *PDEBench: An Extensive Benchmark for Scientific Machine Learning*.  
    Important benchmark reference and source of candidate PDE tasks.  
    Paper: https://proceedings.neurips.cc/paper_files/paper/2022/hash/0a9747136d411fb83f0cf81820d44afb-Abstract-Datasets_and_Benchmarks.html  
    Project: https://github.com/pdebench/PDEBench

15. **Z. Hao et al. (2024)**  
    *PINNacle: A Comprehensive Benchmark of Physics-Informed Neural Networks for Solving PDEs*.  
    Strong source of benchmark tasks and reporting structure.  
    Paper: https://proceedings.neurips.cc/paper_files/paper/2024/hash/8c63299fb2820ef41cb05e2ff11836f5-Abstract-Datasets_and_Benchmarks_Track.html  
    Project: https://github.com/i207M/PINNacle

---

## 6. Frameworks that matter for this repository

16. **Neuromancer**  
    Official repo: https://github.com/pnnl/neuromancer  
    Docs: https://pnnl.github.io/neuromancer/

17. **PhysicsNeMo**  
    Official repo: https://github.com/NVIDIA/physicsnemo  
    Docs: https://docs.nvidia.com/physicsnemo/latest/index.html

18. **TorchPhysics**  
    Official repo: https://github.com/boschresearch/torchphysics  
    Docs: https://boschresearch.github.io/torchphysics/

---

## 7. Application / dataset notes

19. **NASA C-MAPSS portal**  
    Useful if an aerospace-flavored temporal monitoring benchmark is pursued.
    Current official pages are mixed: the NASA Open Data Portal lists the
    dataset and zip resource, while the federal catalog entry for C-MAPSS
    still says the download is unavailable. Treat it as an optional side
    branch rather than the core PDE story.  
    Portal: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data  
    Zip resource: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data/resource/5224bcd1-ad61-490b-93b9-2817288accb8  
    Catalog entry: https://catalog.data.gov/dataset/c-mapss-aircraft-engine-simulator-data  
    PCoE repository: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

---

## 8. Suggested reading order for a new contributor

1. Maler and Nickovic
2. Fainekos and Pappas
3. RTAMT
4. MoonLight
5. STLnet
6. Raissi et al.
7. PDEBench / PINNacle
8. the framework docs for whichever integration path you want to extend

That order gets you from logic semantics to tools to benchmarks without jumping
straight into implementation details.
