# Installing Optional Dependencies

The core package is intentionally light. PyTorch, the monitor backends, and the
external frameworks are all optional.

Use only the extras you need.

---

## 1. Common install patterns

```bash
# core package only
pip install -e "."

# typical working setup for training + figures + tests
pip install -e ".[torch,plot,dev]"

# Linux CI / CPU-only machine: install the CPU torch wheel first, then the repo extras
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -e ".[plot,dev]"

# add STL monitoring (RTAMT currently works on Python 3.10/3.11)
python3.11 -m pip install -e ".[stl]"

# add STREL monitoring (plus Java runtime)
pip install -e ".[strel]"

# add the packaged framework probes bundled in pyproject.toml
pip install -e ".[frameworks]"

# everything listed in pyproject.toml
pip install -e ".[all]"
```

---

## 2. What each extra gives you

### `torch`

Installs PyTorch, which is required for:

- training the diffusion PINN,
- loading committed checkpoints and dense-field sidecars,
- running most of the main case-study scripts.

For Linux CI or a CPU-only workstation, PyTorch's official install selector also supports a CPU wheel index. In that case, install `torch` first via the CPU index and then install this repository's extras **without** requesting the `torch` extra a second time.

### `plot`

Installs Matplotlib **and pandas**, which are required for:

- `scripts/generate_all_figures.py`
- `scripts/refresh_committed_summaries.py`
- `make refresh-check`
- the plotting helpers used in the demo scripts

### `stl`

Installs RTAMT for offline STL audits.

At the moment the repository intentionally installs this extra only on
**Python < 3.12** because current RTAMT releases still import `typing.io`,
which was removed from newer Python versions. If you need RTAMT, create a
Python 3.10 or 3.11 environment first.

Relevant files:

- `scripts/eval_diffusion_rtamt.py`
- `src/neural_pde_stl_strel/monitoring/rtamt_monitor.py`

### `strel`

Installs the Python-side MoonLight package. You still need a Java runtime
available on your machine for the actual monitor invocation.

Relevant files:

- `scripts/eval_heat2d_moonlight.py`
- `src/neural_pde_stl_strel/monitoring/moonlight_helper.py`

### `frameworks`

Installs the packaged framework probes declared in `pyproject.toml`:

- PyTorch
- Neuromancer
- TorchPhysics

Note that **PhysicsNeMo and the SpaTiaL helpers are not included in this extra
right now**. If you want to experiment with those probes, install them
separately according to the notes below.

---

## 3. Manual installs for other projects

### PhysicsNeMo

```bash
pip install nvidia-physicsnemo
```

Used only for the import/probe helper in this repository.

### SpaTiaL Specifications (`spatial-spec`)

```bash
pip install spatial-spec
```

Used for `src/neural_pde_stl_strel/frameworks/spatial_spec_hello.py`.

### SpaTiaL spatial-lib (`spatial`)

```bash
pip install "spatial @ git+https://github.com/KTH-RPL-Planiacs/SpaTiaL.git#subdirectory=spatial-lib"
```

Used for `src/neural_pde_stl_strel/monitors/spatial_demo.py`.

### DeepXDE

```bash
pip install deepxde
```

Not currently wired into the repository's framework extra, but useful as an
external comparison point.

---

## 4. Java requirement for MoonLight

MoonLight-based commands expect a Java runtime. On common systems:

```bash
# Ubuntu
sudo apt install openjdk-21-jre

# macOS
brew install openjdk@21
```

After installation, verify:

```bash
java -version
```

---

## 5. Sanity checks

```bash
python -m neural_pde_stl_strel --about
python -m neural_pde_stl_strel doctor
python -m neural_pde_stl_strel doctor --require physics stl
python scripts/check_env.py
```

`python -m neural_pde_stl_strel doctor` now validates the core runtime group by
default (`numpy`, `pyyaml`), including the repository's minimum supported
versions. Add `--require physics stl` when you want the CLI to gate optional
framework or monitor stacks too.
