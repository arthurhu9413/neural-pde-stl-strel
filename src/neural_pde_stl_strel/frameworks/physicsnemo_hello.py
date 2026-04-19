"""neural_pde_stl_strel.frameworks.physicsnemo_hello

Small, zero-training helpers around **NVIDIA PhysicsNeMo**.

This module is intentionally *optional-dependency safe*: it does **not** import
``physicsnemo`` or ``torch`` at module import time. Instead, callers can:

- probe whether PhysicsNeMo is importable,
- report the installed version, and
- run a tiny CPU-only smoke test that mirrors the official "Hello World"
  example (a single forward pass through ``FullyConnected``).

Why this exists in this repository
PhysicsNeMo offers many realistic PDE examples and high-performance components,
but it also comes with a heavier dependency stack (often GPU-oriented). For this
project we primarily want a reproducible *probe* that confirms the core API can
run and that highlights where **Signal Temporal Logic (STL)** / spatial STL
monitors can intercept tensors.

Integration hook (conceptual)
In a typical PhysicsNeMo pipeline, you can insert monitors at the same place you
compute losses/metrics:

    input tensor x  ->  PhysicsNeMo model  ->  output tensor y  ->  monitor(y)

For PDE surrogates, ``y`` corresponds to field samples (e.g., ``u(x, t, ...)``)
on a grid/batch; reshape/aggregate as needed before monitoring.

Optional symbolic PDE module
PhysicsNeMo's symbolic PDE utilities live in a separate distribution
(``nvidia-physicsnemo.sym`` on PyPI). The helper ``physicsnemo_pde_summary`` is
best-effort and returns ``None`` when that optional module is unavailable.

References
- PhysicsNeMo installation guide and Hello World example:
  https://docs.nvidia.com/physicsnemo/latest/getting-started/installation.html
- FullyConnected API docs:
  https://docs.nvidia.com/physicsnemo/latest/physicsnemo/api/models/fully_connected.html
"""

from __future__ import annotations

from importlib import import_module, metadata as _metadata
from typing import Any

# Public constants for clarity in messages and for downstream tooling.
PHYSICSNEMO_DIST_NAME: str = "nvidia-physicsnemo"
PHYSICSNEMO_MODULE_NAME: str = "physicsnemo"

# Normalized package name for the optional symbolic/PDE utilities.
# (PEP 503 normalization means dots/underscores map to dashes in pip.)
PHYSICSNEMO_SYM_DIST_NAME: str = "nvidia-physicsnemo.sym"


# Lightweight import helpers
def _require_physicsnemo() -> Any:
    """Import and return the top-level :mod:`physicsnemo` module.

    This keeps module import time cheap and deterministic in environments where
    optional stacks are not installed.

    Raises
    ------
    ImportError
        If PhysicsNeMo cannot be imported.
    """
    try:
        return import_module(PHYSICSNEMO_MODULE_NAME)
    except Exception as e:  # pragma: no cover - error path exercised in tests
        # Keep the message short and actionable. PhysicsNeMo's docs recommend
        # installing PyTorch first, then PhysicsNeMo via pip.
        raise ImportError(
            "PhysicsNeMo is not installed (or could not be imported).\n"
            "Install PyTorch for your platform, then install PhysicsNeMo:\n"
            f"  pip install {PHYSICSNEMO_DIST_NAME}"
        ) from e


def _looks_like_real_package(mod: Any) -> bool:
    """Heuristic: distinguish real installed packages from test stubs.

    The test suite injects tiny stub modules into ``sys.modules`` which do not
    have a meaningful ``__file__``. When the real distribution is installed,
    this attribute should be present.
    """
    return bool(getattr(mod, "__file__", None))


# Public API
def physicsnemo_version() -> str:
    """Return the installed PhysicsNeMo version string.

    Preference order:
    1) ``physicsnemo.__version__`` if present and truthy
    2) ``importlib.metadata.version('nvidia-physicsnemo')`` when PhysicsNeMo
       appears to be a real installed package
    3) Fallback to ``"unknown"``

    Raises
    ------
    ImportError
        If PhysicsNeMo is not importable.
    """
    mod = _require_physicsnemo()

    ver = getattr(mod, "__version__", None)
    if isinstance(ver, str) and ver.strip():
        return ver

    # Avoid consulting global distribution metadata for stub modules used in
    # tests; it would make results depend on the ambient environment.
    if not _looks_like_real_package(mod):
        return "unknown"

    try:
        return _metadata.version(PHYSICSNEMO_DIST_NAME)
    except Exception:
        return "unknown"


def physicsnemo_available() -> bool:
    """Return ``True`` if :mod:`physicsnemo` can be imported, else ``False``."""
    try:
        import_module(PHYSICSNEMO_MODULE_NAME)
        return True
    except Exception:  # pragma: no cover - tiny negative path
        return False


def physicsnemo_smoke(
    batch: int = 128,
    in_features: int = 32,
    out_features: int = 64,
    seed: int = 0,
) -> dict[str, float | str | tuple[int, int]]:
    """Run a minimal, CPU-only PhysicsNeMo forward pass.

    This is a "hello world" probe (no training): build a small
    ``FullyConnected`` network and run a single forward pass under
    ``torch.no_grad()``.

    The output tensor ``y`` is the natural place to *export* values for STL/
    spatial-STL monitoring in PhysicsNeMo pipelines.

    Parameters
    batch:
        Batch size for the dummy input.
    in_features:
        Input dimensionality of the MLP.
    out_features:
        Output dimensionality of the MLP.
    seed:
        Torch seed for determinism.

    Returns
    -------
    dict
        A compact metrics dictionary suitable for CI assertions and simple
        dashboards.

        Keys:
        - ``version``: PhysicsNeMo version string
        - ``out_shape``: tuple ``(batch, out_features)``
        - ``out_batch``: ``float(out_shape[0])`` (JSON-friendly)
        - ``out_dim``: ``float(out_shape[1])`` (JSON-friendly)

    Raises
    ------
    ImportError
        If PhysicsNeMo/PyTorch are not importable.
    """
    _require_physicsnemo()

    # Lazy imports keep module import fast and optional-dep safe.
    try:
        import torch

        try:
            # Canonical import (used in NVIDIA docs and README).
            from physicsnemo.models.mlp.fully_connected import FullyConnected  # type: ignore
        except Exception:
            # Some docs also show a re-export from ``physicsnemo.models.mlp``.
            from physicsnemo.models.mlp import FullyConnected  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "PhysicsNeMo smoke test requires PyTorch and the PhysicsNeMo core "
            "model APIs.\n"
            "Install CPU-only PyTorch (example):\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            f"Then install PhysicsNeMo:\n  pip install {PHYSICSNEMO_DIST_NAME}"
        ) from e

    torch.manual_seed(int(seed))
    device = torch.device("cpu")

    # Prefer a tiny network when the signature supports it; fall back to the
    # minimal constructor for compatibility with older versions and test stubs.
    fc_kwargs: dict[str, int] = {
        "in_features": int(in_features),
        "out_features": int(out_features),
    }
    try:
        model = FullyConnected(**fc_kwargs, layer_size=32, num_layers=2).to(device)
    except TypeError:
        model = FullyConnected(**fc_kwargs).to(device)

    if hasattr(model, "eval"):
        model.eval()

    x = torch.randn(int(batch), int(in_features), device=device)
    with torch.no_grad():
        y = model(x)

    # Robustly coerce to a 2D (batch, dim) shape.
    shape = getattr(y, "shape", None)
    if shape is None or len(shape) < 2:
        raise RuntimeError(f"Unexpected PhysicsNeMo output shape: {shape!r}")

    out_shape: tuple[int, int] = (int(shape[0]), int(shape[1]))
    return {
        "version": physicsnemo_version(),
        "out_shape": out_shape,
        "out_batch": float(out_shape[0]),
        "out_dim": float(out_shape[1]),
    }


def physicsnemo_pde_summary() -> list[str] | None:
    """Return a short textual summary of a PhysicsNeMo symbolic PDE object.

    Best-effort helper for environments that have the optional symbolic
    distribution installed (``nvidia-physicsnemo.sym``). Returns ``None`` if the
    symbolic module is unavailable.

    The summary is captured from ``NavierStokes(...).pprint()`` (first 1-3
    non-empty lines).
    """
    try:  # pragma: no cover - exercised only when PhysicsNeMo Sym is installed
        from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes  # type: ignore
    except Exception:
        return None

    try:
        ns = NavierStokes(nu=0.01, rho=1, dim=2)
    except Exception:
        return None

    # Capture stdout from the public ``pprint`` helper.
    try:
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            pprint = getattr(ns, "pprint", None)
            if callable(pprint):
                pprint()
            else:
                print(repr(ns))

        lines = [ln.strip() for ln in buf.getvalue().splitlines() if ln.strip()]
        return lines[:3] if lines else [ns.__class__.__name__]
    except Exception:
        return [ns.__class__.__name__]


__all__ = [
    "PHYSICSNEMO_DIST_NAME",
    "PHYSICSNEMO_MODULE_NAME",
    "PHYSICSNEMO_SYM_DIST_NAME",
    "physicsnemo_version",
    "physicsnemo_available",
    "physicsnemo_smoke",
    "physicsnemo_pde_summary",
]
