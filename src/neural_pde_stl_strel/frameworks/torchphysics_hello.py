# ruff: noqa: I001
from __future__ import annotations

"""neural_pde_stl_strel.frameworks.torchphysics_hello

Import-safe helpers for Bosch TorchPhysics (optional dependency).

Goals
-----
TorchPhysics is one of the PDE-focused physics-ML frameworks evaluated in this
repo. This helper module:

* avoids importing TorchPhysics at *module* import time,
* provides version/availability checks with actionable error messages, and
* offers a tiny, deterministic, CPU-only smoke test that touches core APIs.

How TorchPhysics fits in this project
High-level dataflow (matching the report diagrams):

    PDE/ODE + IC/BC (+ optional data)   +   STL/STREL spec
                     |                        |
                     v                        |
      TorchPhysics: spaces/domains/samplers    |
                     |
                     v
             PyTorch model (PINN)
                     |
                     v
            signal u(x, t, ...)  ----->  monitor(spec, signal)
                     |                           |
                     |                           v
                     |                    robustness / violations
                     |
     (optional training) loss = physics_loss + (stl_weight) * stl_penalty

`stl_weight` is the scalar hyperparameter that scales the STL penalty when it is
added to the loss.

Repo pointers
* Full TorchPhysics + STL regularization example:
  - scripts/train_burgers_torchphysics.py
* Tiny toy ODE + STL monitoring + plot (CI/regression):
  - tests/test_torchphysics_hello.py

References
* Docs: https://torchphysics.ai/ (legacy: https://boschresearch.github.io/torchphysics/)
* Code: https://github.com/boschresearch/torchphysics
"""

from importlib import import_module, metadata as _metadata
from importlib.util import find_spec
from typing import Any

TORCHPHYSICS_DIST_NAME = "torchphysics"
TORCHPHYSICS_MODULE_NAME = "torchphysics"

SmokeMetrics = dict[str, float | str | tuple[int, ...]]


def _require_torchphysics() -> Any:
    """Import TorchPhysics or raise an actionable :class:`ImportError`.

    Notes
    -----
    We intentionally perform a real import (not just `find_spec`) so that we
    return a truthful signal even when the package exists but fails to import due
    to missing runtime dependencies.
    """
    try:
        return import_module(TORCHPHYSICS_MODULE_NAME)
    except Exception as e:  # pragma: no cover
        # Common pitfall: TorchPhysics is installed, but PyTorch is not.
        if isinstance(e, ModuleNotFoundError) and getattr(e, "name", None) == "torch":
            raise ImportError(
                "TorchPhysics requires PyTorch, but PyTorch could not be imported.\n"
                "Install PyTorch first (CPU example):\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                f"Then install TorchPhysics:\n"
                f"  pip install {TORCHPHYSICS_DIST_NAME}"
            ) from e

        raise ImportError(
            "TorchPhysics is not installed (or failed to import).\n"
            f"Install the PyPI distribution `{TORCHPHYSICS_DIST_NAME}` "
            f"(module `{TORCHPHYSICS_MODULE_NAME}`).\n"
            f"Example: pip install {TORCHPHYSICS_DIST_NAME}"
        ) from e


def torchphysics_version() -> str:
    """Return the installed TorchPhysics version string.

    Prefers `torchphysics.__version__` when available; otherwise falls back to
    `importlib.metadata.version('torchphysics')`.

    Raises
    ------
    ImportError
        If TorchPhysics is not installed.
    """
    mod = _require_torchphysics()
    ver = getattr(mod, "__version__", None)
    if isinstance(ver, str) and ver.strip():
        return ver.strip()

    # Some builds don't set __version__ (e.g., editable/dev installs). Metadata is a
    # good fallback.
    try:
        return _metadata.version(TORCHPHYSICS_DIST_NAME)
    except Exception:
        return "unknown"


def torchphysics_available() -> bool:
    """Return ``True`` iff TorchPhysics can be imported.

    This is a *runtime* check (not just metadata): if the import fails due to
    missing transitive dependencies, this returns ``False``.
    """
    try:
        if find_spec(TORCHPHYSICS_MODULE_NAME) is None:
            return False
    except Exception:
        # If the import system is in a strange state, fall back to an import
        # attempt (the import is inside try/except anyway).
        pass

    try:
        import_module(TORCHPHYSICS_MODULE_NAME)
        return True
    except Exception:
        return False


def torchphysics_smoke(
    n_points: int = 32,
    hidden: tuple[int, ...] = (8, 8),
    seed: int = 0,
) -> SmokeMetrics:
    """Run a tiny TorchPhysics micro-check (CPU-only, **no training**).

    The goal is to exercise a representative slice of TorchPhysics' API surface:
    spaces -> domain -> sampler -> model -> condition -> one forward pass.

    We intentionally keep this:
    * deterministic (seeded)
    * CPU-friendly
    * solver-free (no PyTorch Lightning)

    Parameters
    n_points:
        Number of randomly sampled points in the 1D interval.
    hidden:
        Hidden-layer widths for the FCN model.
    seed:
        RNG seed for repeatability.

    Returns
    -------
    dict
        At minimum contains:
        ``{"version": str, "loss": float, "points": int, "hidden": tuple[int, ...]}``.

    Notes
    -----
    This is **not** intended to validate numerical correctness, only that the
    library is importable and the core objects behave as expected.
    """
    if int(n_points) <= 0:
        raise ValueError(f"n_points must be positive, got {n_points}")

    hidden_tuple = tuple(int(h) for h in hidden)
    if not hidden_tuple or any(h <= 0 for h in hidden_tuple):
        raise ValueError(f"hidden must be a non-empty tuple of positive ints, got {hidden}")

    tp = _require_torchphysics()

    import torch

    torch.manual_seed(int(seed))
    device = torch.device("cpu")

    # Build a minimal 1D PINN condition

    X = tp.spaces.R1("x")
    U = tp.spaces.R1("u")

    # Domain (be generous about API/keyword differences across TorchPhysics versions).
    try:
        interval = tp.domains.Interval(X, 0.0, 1.0)
    except TypeError:  # pragma: no cover
        # Fallback for keyword signatures (seen in some older snippets).
        try:
            interval = tp.domains.Interval(space=X, lower_bound=0.0, upper_bound=1.0)
        except TypeError:
            interval = tp.domains.Interval(space=X, lower=0.0, upper=1.0)

    # Sampler
    try:
        sampler = tp.samplers.RandomUniformSampler(interval, n_points=int(n_points))
    except TypeError:  # pragma: no cover
        sampler = tp.samplers.RandomUniformSampler(domain=interval, n_points=int(n_points))

    # Model
    try:
        model = tp.models.FCN(input_space=X, output_space=U, hidden=hidden_tuple)
    except TypeError:  # pragma: no cover
        # Some versions use positional args.
        model = tp.models.FCN(X, U, hidden=hidden_tuple)

    # Differential operator helper (location differs across versions).
    grad_fn: Any | None = None
    try:
        from torchphysics.utils.differentialoperators import grad as _grad  # type: ignore

        grad_fn = _grad
    except Exception:
        grad_fn = getattr(getattr(tp, "utils", None), "grad", None)

    if not callable(grad_fn):  # pragma: no cover
        raise RuntimeError(
            "Unable to locate a TorchPhysics gradient operator. Expected either "
            "`torchphysics.utils.differentialoperators.grad` or `torchphysics.utils.grad`."
        )

    def residual_du_dx(u: Any, x: Any) -> Any:
        """Residual for a toy PDE: du/dx (drives the condition loss)."""
        g = grad_fn(u, x)
        # Some operator variants return a tuple/list; PINNCondition expects a
        # tensor-like residual.
        if isinstance(g, (tuple, list)):
            g = g[0]
        return g

    # PINNCondition class path moved across versions.
    PINNCondition: Any | None = None
    try:
        PINNCondition = tp.problem.conditions.PINNCondition
    except Exception:
        try:
            PINNCondition = tp.conditions.PINNCondition
        except Exception:
            try:
                from torchphysics.problem.conditions import PINNCondition as _PINNCondition  # type: ignore

                PINNCondition = _PINNCondition
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "Unable to locate TorchPhysics PINNCondition class. Tried "
                    "`tp.problem.conditions.PINNCondition`, `tp.conditions.PINNCondition`, "
                    "and `torchphysics.problem.conditions.PINNCondition`."
                ) from e

    condition = PINNCondition(module=model, sampler=sampler, residual_fn=residual_du_dx)

    with torch.enable_grad():
        # Some versions accept torch.device, others accept strings.
        try:
            loss_t = condition.forward(device=device)
        except TypeError:
            loss_t = condition.forward(device=str(device))

    if not torch.is_tensor(loss_t):  # pragma: no cover
        loss_t = torch.as_tensor(loss_t)

    # Return a scalar even if the condition returns per-point values.
    loss_cpu = loss_t.detach().cpu()
    if loss_cpu.numel() != 1:
        loss_cpu = loss_cpu.mean()

    return {
        "version": torchphysics_version(),
        "loss": float(loss_cpu.item()),
        "points": int(n_points),
        "hidden": hidden_tuple,
    }


__all__ = [
    "TORCHPHYSICS_DIST_NAME",
    "TORCHPHYSICS_MODULE_NAME",
    "torchphysics_version",
    "torchphysics_available",
    "torchphysics_smoke",
]
