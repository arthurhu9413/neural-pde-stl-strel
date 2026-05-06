# Reproducibility Guide

This repository is meant to be reproducible from committed artifacts first and
from retraining second.

That distinction matters. The fastest reliable path is:

1. rebuild JSON / CSV summaries from the committed dense-field sidecars,
2. regenerate figures from those refreshed summaries,
3. run tests.

Only after that should you rerun training jobs.

---

## 1. Environment setup

### Base requirements

- Python 3.10 or newer
- NumPy
- PyYAML

### For training and figure regeneration

- PyTorch
- Matplotlib
- pandas

### Optional extras

- RTAMT for offline STL audits
- MoonLight plus a Java runtime for STREL monitoring
- Neuromancer / TorchPhysics for framework probes

### Install commands

```bash
python -m venv .venv
source .venv/bin/activate

# common working setup
pip install -e ".[torch,plot,dev]"
# Linux CI / CPU-only alternative:
# pip install --index-url https://download.pytorch.org/whl/cpu torch && pip install -e ".[plot,dev]"

# add MoonLight and framework probes
pip install -e ".[strel,frameworks]"

# add RTAMT in a Python 3.10 / 3.11 environment
python3.11 -m pip install -e ".[stl]"
```

See [docs/INSTALL_EXTRAS.md](INSTALL_EXTRAS.md) for package-by-package notes.

---

## 2. Sanity check the environment

```bash
python -m neural_pde_stl_strel --version
python -m neural_pde_stl_strel --about
python scripts/check_env.py
```

One committed environment-summary snapshot is stored in `results/env_report.json`.
It documents a validation environment, not a universal requirement.

### Snapshot highlights from `results/env_report.json`

| Field | Value |
|---|---|
| Python | 3.13.5 |
| PyTorch | 2.10.0+cpu |
| CUDA available | No |
| Logical CPU cores | 56 |
| Reported RAM | 4.0 GiB |
| Java runtime | OpenJDK 21 present |
| RTAMT on this interpreter | not applicable on Python 3.13; use Python 3.11 |

That table is useful for the paper's experimental-setup paragraph because it
lets the manuscript cite one concrete validation environment without implying
that every reproduction must use the same machine.

---

## 3. Refresh summaries from the committed field sidecars

This is the first command to run after cloning the repository:

```bash
python scripts/refresh_committed_summaries.py
```

What it does by default:

- reads the committed diffusion dense-field sidecars,
- recomputes dense-field audit metrics,
- refreshes `results/diffusion1d_main_summary.json` only if it has meaningfully drifted,
- refreshes the ablation summary tables only if they have meaningfully drifted, and
- refreshes the RTAMT-facing JSON sidecars.

This step exists so that README tables and figure scripts are tied to computed
values rather than to hand-entered numbers.

For CI, use the read-only validation path instead:

```bash
python scripts/refresh_committed_summaries.py --check
# or
make refresh-check
```

Only use `--prefer-checkpoints` (and especially
`--rewrite-main-fields-from-checkpoints`) when you intentionally want to rebuild
the dense-field sidecars from the raw checkpoints.

---

## 4. Regenerate figures

### Standard path: refresh the committed `figs/` and selected `assets/`

```bash
python scripts/generate_all_figures.py --dpi 180
```

### Non-destructive validation path: write to scratch space and verify outputs

```bash
python scripts/generate_all_figures.py \
  --dpi 180 \
  --output-root /tmp/neural-pde-stl-strel-figcheck \
  --manifest /tmp/neural-pde-stl-strel-figcheck/figure_manifest.json \
  --check
```

or, using the Makefile helper:

```bash
make figures-check
```

This second path is what CI should use. It validates that figure generation
still works **without** assuming that Matplotlib will reproduce byte-identical
PNG files across every environment.

Key outputs include:

- `figs/architecture_diagram.png`
- `figs/training_pipeline.png`
- `figs/diffusion1d_comparison.png`
- `figs/diffusion1d_training_dynamics.png`
- `figs/diffusion1d_lambda_ablation.png`
- `figs/heat2d_field_evolution.png`
- `figs/heat2d_stl_traces.png`
- `figs/summary_results.png`
- `figs/quality_dashboard.png`

---

## 5. Inspect the committed runtime snapshot

```bash
python scripts/show_benchmark_snapshot.py
# or
make benchmark
```

This command does **not** rerun training. It prints the committed
`results/benchmark_training.csv` table that backs the README and cost figure.

---

## 6. Run tests

### Full pytest suite

```bash
python -m pytest -q
```

### Standalone lightweight checks

```bash
PYTHONPATH=src python tests/run_tests.py
```

The standalone suite is useful on minimal installs because many tests avoid
optional dependencies.

---

## 7. Optional: rerun the diffusion experiments

If you want to regenerate the committed checkpoints from scratch rather than
just refresh summaries:

```bash
python scripts/run_experiment.py -c configs/diffusion1d_baseline.yaml
python scripts/run_experiment.py -c configs/diffusion1d_stl.yaml
python scripts/eval_diffusion_rtamt.py
python scripts/run_ablations_diffusion.py
```

The default committed configs correspond to:

- `configs/diffusion1d_baseline.yaml`
- `configs/diffusion1d_stl.yaml` with `stl.weight = 5.0`

The ablation sweep is stored separately from the main baseline / STL comparison.

---

## 8. Optional: rerun the 2D heat monitoring path

```bash
python scripts/gen_heat2d_frames.py
python scripts/eval_heat2d_moonlight.py
```

This path uses the committed heat rollout and threshold sweep. At present it is
a spatial monitoring example, not a full STREL-constrained training benchmark.

---

## 9. Expected result artifacts

| Kind | Location | Notes |
|---|---|---|
| Diffusion checkpoints | `results/diffusion1d_*.pt` | Main baseline / STL runs |
| Diffusion fields | `results/diffusion1d_*_field.pt` / `.npz` | Dense fields used for plotting and audits |
| Diffusion summaries | `results/diffusion1d_main_summary.json` | Recomputed from committed field sidecars |
| Ablation summary | `results/diffusion1d_ablation_summary.json` / `.csv` | Separate sweep from the main pair |
| Heat monitoring summary | `results/heat2d_strel_monitoring.json` | Threshold analysis on committed rollout |
| Benchmark cost table | `results/benchmark_training.csv` | CPU-first cost snapshot |
| Figures | `figs/*.png` | Generated from committed artifacts |

---

## 10. Notes on determinism

- Seeds are set in the training scripts.
- CPU runs are the intended default for comparable results.
- GPU and Apple MPS runs may change floating-point reductions slightly.
- MoonLight introduces a Java dependency; make sure the runtime is available
  before expecting STREL commands to work.
- Matplotlib output can differ slightly across versions and platforms, so CI
  validates generated artifacts in scratch space rather than diffing tracked PNG
  files byte for byte.
- Dense-field regeneration from checkpoints can also drift at the last few
  floating-point digits across environments, so CI validates summary semantics
  with `python scripts/refresh_committed_summaries.py --check` instead of
  rewriting tracked `results/` binaries.

---

## 11. Minimal paper-refresh checklist

When updating the repo for a paper draft, use this order:

1. rerun or refresh result summaries,
2. run `python scripts/refresh_committed_summaries.py --check`,
3. regenerate figures,
4. run tests,
5. update prose,
6. verify every citation and every table entry by hand.

That order keeps the documentation downstream of the data rather than the other
way around.
