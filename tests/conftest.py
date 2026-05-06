"""tests/conftest.py -- hermetic, CPU-friendly pytest bootstrap.

Design goals (in priority order):
  1) Determinism & stability on laptops/CI.
  2) Sane performance (avoid BLAS/OpenMP thread storms).
  3) Auditability (print effective seed + key env/versions in the pytest header).

Notes:
- Environment variables are set *before* importing numerical stacks so they take
  effect during library initialization.
- We intentionally avoid importing heavyweight optional stacks (e.g., TensorFlow)
  just to seed them; doing so can slow tests and spam logs.
"""

from __future__ import annotations

import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable


# A. Environment defaults

def _setdefault_env(name: str, value: str) -> None:
    """Set an env var only if it is not already set by the user/CI."""

    os.environ.setdefault(name, value)


def _int_env(name: str, default: int) -> int:
    """Parse an int from the environment, falling back to *default*."""

    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


# Keep timestamps consistent in assertions/logs.
_setdefault_env("TZ", "UTC")
if hasattr(time, "tzset"):
    try:
        time.tzset()
    except Exception:
        # If tzset exists but fails (uncommon), keep going.
        pass

# Avoid thread explosions across common math stacks.
for _var in (
    "OMP_NUM_THREADS",  # OpenMP consumers (NumPy/SciPy, PyTorch, etc.)
    "OPENBLAS_NUM_THREADS",  # OpenBLAS
    "MKL_NUM_THREADS",  # Intel/oneMKL
    "NUMEXPR_NUM_THREADS",  # numexpr
    "VECLIB_MAXIMUM_THREADS",  # Apple Accelerate
    "BLIS_NUM_THREADS",  # BLIS / AOCL
    "RAYON_NUM_THREADS",  # Rust rayon (tokenizers/other deps)
):
    _setdefault_env(_var, "1")

# Disable dynamic thread adjustment for more reproducible behavior.
_setdefault_env("OMP_DYNAMIC", "FALSE")
_setdefault_env("MKL_DYNAMIC", "FALSE")

# Hint the OpenMP runtime not to busy-spin.
_setdefault_env("OMP_WAIT_POLICY", "PASSIVE")

# Quiet & de-parallelize HuggingFace tokenizers (prevents deadlocks after fork).
_setdefault_env("TOKENIZERS_PARALLELISM", "false")

# Encourage deterministic cuBLAS behavior when CUDA is present.
_setdefault_env("CUBLAS_WORKSPACE_CONFIG", ":16:8")

# Headless-safe plotting defaults.
_setdefault_env("MPLBACKEND", "Agg")

# Prefer a writable, repo-local Matplotlib config/cache dir when possible.
# (If the user already set MPLCONFIGDIR, respect it and do nothing.)
if os.environ.get("MPLCONFIGDIR") is None:
    _repo_root = Path(__file__).resolve().parents[1]
    _mpl_config_dir = _repo_root / ".pytest_cache" / "matplotlib"
    try:
        _mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(_mpl_config_dir)
    except Exception:
        # Fall back to Matplotlib's internal temp-dir behavior.
        pass

# NOTE on deterministic hashing:
# - PYTHONHASHSEED must be set *before* Python starts to affect this process.
# - Setting it here can still help any Python subprocesses spawned by tests.
#   Example:  PYTHONHASHSEED=0 pytest


# B. Seed everything we might use

DEFAULT_SEED = _int_env("TEST_SEED", 0)


def _seed_core(seed: int) -> None:
    """Seed Python's stdlib RNG and NumPy (if available)."""

    random.seed(seed)
    try:
        import numpy as np  # local import: respects env vars set above

        np.random.seed(seed)
    except Exception:
        # NumPy is expected for this repo, but keep the bootstrap robust.
        pass


def _seed_torch(seed: int) -> None:
    """Seed PyTorch (if available) and set conservative determinism knobs."""

    try:
        import torch

        torch.manual_seed(seed)
        try:
            torch.cuda.manual_seed_all(seed)  # no-op if CUDA unavailable
        except Exception:
            pass

        # Keep thread usage predictable, but respect user overrides via env.
        threads = max(1, _int_env("OMP_NUM_THREADS", 1))
        try:
            torch.set_num_threads(threads)
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        # Prefer deterministic behavior on GPUs.
        try:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            # Disable TF32 to reduce subtle numeric drift across GPUs.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass

        # Prefer deterministic algorithms where available.
        # Warn-only avoids brittle failures across different hardware/backends.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    except Exception:
        # Torch is optional for parts of the suite.
        pass


def _seed_optional_backends(seed: int) -> None:
    """Seed optional stacks *only if already imported*.

    Importing heavyweight libraries just to set seeds can slow tests
    significantly and can produce noisy import-time logs.
    """

    def _seed_if_loaded(module_name: str, seeder: Callable[[Any], None]) -> None:
        mod = sys.modules.get(module_name)
        if mod is None:
            return
        try:
            seeder(mod)
        except Exception:
            pass

    _seed_if_loaded("cupy", lambda cp: cp.random.seed(seed))
    _seed_if_loaded("tensorflow", lambda tf: tf.random.set_seed(seed))


# Initial seeding at import time (before tests import project modules).
_seed_core(DEFAULT_SEED)
_seed_torch(DEFAULT_SEED)
_seed_optional_backends(DEFAULT_SEED)


# C. Pytest integration

def pytest_addoption(parser: Any) -> None:
    parser.addoption(
        "--seed",
        action="store",
        type=int,
        default=DEFAULT_SEED,
        help="Override global RNG seed (default: env TEST_SEED or 0).",
    )


def pytest_configure(config: Any) -> None:
    """Apply CLI seed overrides (if any) and make the effective seed visible."""

    seed = config.getoption("seed")
    if not isinstance(seed, int):
        try:
            seed = int(seed)
        except Exception:
            seed = DEFAULT_SEED

    # Make the effective seed visible to the rest of the run (and to logs).
    os.environ["TEST_SEED"] = str(seed)

    # Re-seed if the CLI overrides the import-time seed.
    if seed != DEFAULT_SEED:
        _seed_core(seed)
        _seed_torch(seed)
        _seed_optional_backends(seed)


def _dist_version(dist_name: str) -> str | None:
    """Return installed distribution version without importing the package."""

    try:
        from importlib import metadata

        try:
            return metadata.version(dist_name)
        except metadata.PackageNotFoundError:
            return None
    except Exception:
        return None


def pytest_report_header(config: Any) -> list[str]:
    """Add a small, audit-friendly header to each pytest run."""

    seed = config.getoption("seed")

    # Key environment knobs affecting determinism/perf.
    keys = (
        "TEST_SEED",
        "PYTHONHASHSEED",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "OMP_DYNAMIC",
        "MKL_DYNAMIC",
        "OMP_WAIT_POLICY",
        "TOKENIZERS_PARALLELISM",
        "CUBLAS_WORKSPACE_CONFIG",
        "MPLBACKEND",
        "MPLCONFIGDIR",
        "TZ",
    )
    knobs = ", ".join(
        f"{k}={os.environ[k]}" for k in keys if k in os.environ and os.environ[k] != ""
    )

    # Basic runtime + dependency versions (use importlib.metadata, not imports).
    deps = {
        "pytest": _dist_version("pytest"),
        "numpy": _dist_version("numpy"),
        "torch": _dist_version("torch"),
        "matplotlib": _dist_version("matplotlib"),
    }
    deps_s = ", ".join(
        f"{name}={ver}" for name, ver in deps.items() if ver is not None
    )
    if not deps_s:
        deps_s = "(dependency versions unavailable)"

    # Hardware-ish context (cheap to query, avoids long /proc parsing).
    runtime = (
        f"python={platform.python_version()} | "
        f"os={platform.system()} {platform.release()} | "
        f"arch={platform.machine()} | "
        f"cpu_count={os.cpu_count()}"
    )

    return [
        f"neural-pde-stl-strel: seed={seed}",
        runtime,
        f"deps: {deps_s}",
        f"env: {knobs}",
    ]
