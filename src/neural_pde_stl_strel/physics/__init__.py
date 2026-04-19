"""Physics helpers used by the STL/STREL neural-PDE demos.

This subpackage intentionally keeps import-time overhead low by lazily importing
PyTorch-heavy code only when you access a submodule or a forwarded helper.

Lazy submodules
- diffusion1d
- heat2d

Forwarded helpers (resolved on first access and cached in this namespace)
- 1-D diffusion: pde_residual, residual_loss, boundary_loss, Interval1D, sine_ic,
  sine_solution, make_dirichlet_mask_1d
- 2-D heat: residual_heat2d, bc_ic_heat2d, SquareDomain2D, gaussian_ic,
  make_dirichlet_mask

Power-user toggles
Set via environment variables *before* importing this package:

- NEURAL_PDE_STL_STREL_EAGER_IMPORTS=1
    Eagerly import submodules and (best-effort) resolve forwarded helpers.
    Useful for IDE indexing or interactive exploration.
- NEURAL_PDE_STL_STREL_STRICT_INIT=1
    Validate that all forwarded attributes exist (implies eager imports).
    Fails fast if an export mapping is incorrect.

By default, everything is lazy and incurs near-zero overhead at import time.
"""

from __future__ import annotations

import importlib
import os
from difflib import get_close_matches
from typing import Any, TYPE_CHECKING


_LAZY_MODULES: dict[str, str] = {
    "diffusion1d": "neural_pde_stl_strel.physics.diffusion1d",
    "heat2d": "neural_pde_stl_strel.physics.heat2d",
}

# Forwarded attributes: provide nice `from neural_pde_stl_strel.physics import pde_residual`.
# These are fetched from their submodules on first access and then cached here.
_FORWARD_ATTRS: dict[str, tuple[str, str]] = {
    # 1-D diffusion
    "pde_residual": ("diffusion1d", "pde_residual"),
    "residual_loss": ("diffusion1d", "residual_loss"),
    "boundary_loss": ("diffusion1d", "boundary_loss"),
    "Interval1D": ("diffusion1d", "Interval1D"),
    "sine_ic": ("diffusion1d", "sine_ic"),
    "sine_solution": ("diffusion1d", "sine_solution"),
    "make_dirichlet_mask_1d": ("diffusion1d", "make_dirichlet_mask_1d"),
    # 2-D heat
    "residual_heat2d": ("heat2d", "residual_heat2d"),
    "bc_ic_heat2d": ("heat2d", "bc_ic_heat2d"),
    "SquareDomain2D": ("heat2d", "SquareDomain2D"),
    "gaussian_ic": ("heat2d", "gaussian_ic"),
    "make_dirichlet_mask": ("heat2d", "make_dirichlet_mask"),
}

# What `from neural_pde_stl_strel.physics import *` exposes.
# Keep this ordered (submodules first, then forwarded helpers) for IDEs/docs.
__all__ = [*_LAZY_MODULES, *_FORWARD_ATTRS]

# Reuse in __dir__ without re-allocating a set each time.
_PUBLIC_NAMES = frozenset(__all__)


def _truthy_env(var: str) -> bool:  # pragma: no cover - trivial
    return os.getenv(var, "").strip().lower() in {"1", "true", "yes", "on"}


def _import_module(mod_path: str) -> Any:  # pragma: no cover - tiny wrapper
    """Import a submodule, raising a clearer error if PyTorch is missing/broken."""
    try:
        return importlib.import_module(mod_path)
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", "") or ""
        if missing == "torch" or missing.startswith("torch."):
            raise ImportError(
                "neural_pde_stl_strel.physics requires PyTorch (`torch`), but it was not found. "
                "Install it with `pip install torch` (or install the repo requirements)."
            ) from e
        raise
    except ImportError as e:
        # PyTorch can also fail with ImportError if native extensions/shared libs are missing.
        msg = str(e).lower()
        if "torch" in msg or "pytorch" in msg:
            raise ImportError(
                "PyTorch (`torch`) failed to import, which is required for neural_pde_stl_strel.physics. "
                "Try reinstalling/upgrading PyTorch for your platform."
            ) from e
        raise


def _load_submodule(name: str) -> Any:
    module = _import_module(_LAZY_MODULES[name])
    globals()[name] = module  # cache in module globals for subsequent lookups
    return module


def _load_forward(name: str) -> Any:
    mod_name, attr = _FORWARD_ATTRS[name]
    module = globals().get(mod_name)
    if module is None:
        module = _load_submodule(mod_name)
    value = getattr(module, attr)
    globals()[name] = value  # cache resolved attribute
    return value


def __getattr__(name: str) -> Any:  # pragma: no cover - exercised implicitly
    # Submodule?
    if name in _LAZY_MODULES:
        return _load_submodule(name)
    # Forwarded attribute?
    if name in _FORWARD_ATTRS:
        return _load_forward(name)

    # Friendly suggestions for typos.
    near = get_close_matches(name, __all__, n=3, cutoff=0.6)
    hint = f" Did you mean {', '.join(repr(c) for c in near)}?" if near else ""
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}.{hint}")


def __dir__() -> list[str]:  # pragma: no cover - trivial
    # Expose both already-bound globals and the lazy/public attributes.
    return sorted(set(globals()) | _PUBLIC_NAMES)


_STRICT_INIT = _truthy_env("NEURAL_PDE_STL_STREL_STRICT_INIT")
_EAGER_IMPORTS = _STRICT_INIT or _truthy_env("NEURAL_PDE_STL_STREL_EAGER_IMPORTS")

# Eager import if the user requested it (IDE indexing or strict CI sanity check).
if _EAGER_IMPORTS:  # pragma: no cover
    # Import submodules first.
    for _m in _LAZY_MODULES:
        try:
            _load_submodule(_m)
        except Exception:
            if _STRICT_INIT:
                raise

    if _STRICT_INIT:
        # Validate that forwarded names actually resolve (and show all failures at once).
        missing: list[str] = []
        for _name, (_mod, _attr) in _FORWARD_ATTRS.items():
            try:
                _load_forward(_name)
            except Exception as _e:
                missing.append(f"{_name}  <-  {_mod}.{_attr}  ({_e.__class__.__name__}: {_e})")
        if missing:
            bullet = "\n  - "
            raise ImportError(
                "neural_pde_stl_strel.physics: forward export validation failed:" + bullet + bullet.join(missing)
            )
    else:
        # Best-effort: warm forwarded helpers so IDEs can discover the public API.
        for _name in _FORWARD_ATTRS:
            try:
                _load_forward(_name)
            except Exception:
                pass


if TYPE_CHECKING:  # pragma: no cover
    from . import diffusion1d as diffusion1d, heat2d as heat2d  # noqa: F401
    from .diffusion1d import (  # noqa: F401
        boundary_loss,
        Interval1D,
        make_dirichlet_mask_1d,
        pde_residual,
        residual_loss,
        sine_ic,
        sine_solution,
    )
    from .heat2d import (  # noqa: F401
        bc_ic_heat2d,
        gaussian_ic,
        make_dirichlet_mask,
        residual_heat2d,
        SquareDomain2D,
    )
