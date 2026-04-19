"""Training utilities for the neural-PDE + STL/STREL scaffold.

This subpackage contains small helpers used by the PINN/PDE examples in this
repository. The primary public module is :mod:`neural_pde_stl_strel.training.grids`,
which provides:

- grid constructors for collocation/monitoring points (`grid1d`, `grid2d`, `grid3d`)
- spacing helpers (`spacing1d`, `spacing2d`, `spacing3d`)
- interior/boundary samplers (`sample_interior_*d`, `sample_boundary_*d`)
- simple domain dataclasses (`Box1D`, `Box2D`, `Box3D`)

Design goals
- **Fast imports:** importing :mod:`neural_pde_stl_strel.training` should be near
  zero-cost and should *not* import PyTorch unless these utilities are actually
  used.
- **Ergonomics:** common helpers are available directly from the package, e.g.::

      from neural_pde_stl_strel.training import grid1d, Box2D

- **Actionable failures:** if PyTorch is not installed, accessing these helpers
  raises an informative :class:`ImportError`.

Implementation notes
We implement lazy loading with module-level ``__getattr__`` / ``__dir__``
(:pep:`562`). For end-to-end training dataflow diagrams, see
``docs/REPRODUCIBILITY.md`` in the repository.
"""

from __future__ import annotations

import importlib
import os
from difflib import get_close_matches
from typing import TYPE_CHECKING, Any


_LAZY_MODULES: dict[str, str] = {
    "grids": "neural_pde_stl_strel.training.grids",
}

# Forwarded attributes: provide nice `from neural_pde_stl_strel.training import grid1d`.
# These are fetched from their submodule on first access and then cached here.
_FORWARD_ATTRS: dict[str, tuple[str, str]] = {
    # Domain dataclasses
    "Box1D": ("grids", "Box1D"),
    "Box2D": ("grids", "Box2D"),
    "Box3D": ("grids", "Box3D"),
    # Grid constructors
    "grid1d": ("grids", "grid1d"),
    "grid2d": ("grids", "grid2d"),
    "grid3d": ("grids", "grid3d"),
    # Spacing helpers
    "spacing1d": ("grids", "spacing1d"),
    "spacing2d": ("grids", "spacing2d"),
    "spacing3d": ("grids", "spacing3d"),
    # Sampling helpers
    "sample_interior_1d": ("grids", "sample_interior_1d"),
    "sample_boundary_1d": ("grids", "sample_boundary_1d"),
    "sample_interior_2d": ("grids", "sample_interior_2d"),
    "sample_boundary_2d": ("grids", "sample_boundary_2d"),
    "sample_interior_3d": ("grids", "sample_interior_3d"),
    "sample_boundary_3d": ("grids", "sample_boundary_3d"),
}

# Backwards-compatible / convenience aliases (kept out of __all__).
_ALIASES: dict[str, str] = {
    # CamelCase variants for grid constructors
    "Grid1D": "grid1d",
    "Grid2D": "grid2d",
    "Grid3D": "grid3d",
    # Underscore variants for grid constructors
    "grid_1d": "grid1d",
    "grid_2d": "grid2d",
    "grid_3d": "grid3d",
    # Lowercase domain aliases
    "Box1d": "Box1D",
    "Box2d": "Box2D",
    "Box3d": "Box3D",
    "box1d": "Box1D",
    "box2d": "Box2D",
    "box3d": "Box3D",
    "box_1d": "Box1D",
    "box_2d": "Box2D",
    "box_3d": "Box3D",
    # Spacing underscores
    "spacing_1d": "spacing1d",
    "spacing_2d": "spacing2d",
    "spacing_3d": "spacing3d",
    # Sampler underscores (common typo: drop underscore before dimension)
    "sample_interior1d": "sample_interior_1d",
    "sample_boundary1d": "sample_boundary_1d",
    "sample_interior2d": "sample_interior_2d",
    "sample_boundary2d": "sample_boundary_2d",
    "sample_interior3d": "sample_interior_3d",
    "sample_boundary3d": "sample_boundary_3d",
}

# What `from neural_pde_stl_strel.training import *` exposes.
__all__ = sorted({*_LAZY_MODULES.keys(), *_FORWARD_ATTRS.keys()})


_TORCH_IMPORT_ERROR = (
    "The training utilities require PyTorch (module 'torch' not found).\n"
    "-> Install via: pip install torch\n"
    "  (repo users: make install  OR  pip install -r requirements.txt)"
)


def _truthy_env(var: str) -> bool:  # pragma: no cover - trivial
    return os.getenv(var, "").strip().lower() in {"1", "true", "yes", "on"}


def _import_module(mod_path: str) -> Any:  # pragma: no cover - tiny wrapper
    """Import *mod_path* with a friendlier error if PyTorch is missing."""
    try:
        return importlib.import_module(mod_path)
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", "") or ""
        if missing == "torch" or missing.startswith("torch."):
            raise ImportError(_TORCH_IMPORT_ERROR) from e
        raise
    except ImportError as e:
        # Offer a friendlier nudge if the failure was due to missing torch.
        msg = str(e).lower()
        if "torch" in msg or "pytorch" in msg:
            raise ImportError(_TORCH_IMPORT_ERROR) from e
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

    # Alias?
    if name in _ALIASES:
        canonical = _ALIASES[name]
        # Resolve through the canonical path so caching works consistently.
        value = __getattr__(canonical)
        globals()[name] = value  # cache alias as well
        return value

    # Friendly suggestions for typos
    candidates = list(__all__) + list(_ALIASES.keys())
    near = get_close_matches(name, candidates, n=3, cutoff=0.6)
    msg = f"module {__name__!r} has no attribute {name!r}"
    if near:
        msg += f". Did you mean {', '.join(repr(c) for c in near)}?"
    raise AttributeError(msg)


def __dir__() -> list[str]:  # pragma: no cover - trivial
    # Expose already-bound globals plus all lazy attributes for IDE completion.
    return sorted({*globals().keys(), *__all__, *_ALIASES.keys()})


# Eager import if the user requested it (IDE indexing or strict CI sanity check).
if (
    _truthy_env("NEURAL_PDE_STL_STREL_STRICT_INIT")
    or _truthy_env("NEURAL_PDE_STL_STREL_EAGER_IMPORTS")
):  # pragma: no cover
    strict = _truthy_env("NEURAL_PDE_STL_STREL_STRICT_INIT")

    # Import submodules first.
    for _m in list(_LAZY_MODULES.keys()):
        try:
            _load_submodule(_m)
        except Exception:
            if strict:
                raise

    if strict:
        problems: list[str] = []

        # Validate alias targets.
        for _alias, _canonical in _ALIASES.items():
            if _canonical not in _LAZY_MODULES and _canonical not in _FORWARD_ATTRS:
                problems.append(f"{_alias}  ->  {_canonical} (unknown canonical)")

        # Validate forwarded names actually resolve.
        for _name, (_mod, _attr) in _FORWARD_ATTRS.items():
            try:
                _load_forward(_name)
            except Exception as _e:
                problems.append(f"{_name}  <-  {_mod}.{_attr}  ({_e.__class__.__name__}: {_e})")

        if problems:
            bullet = "\n  - "
            raise ImportError(
                "neural_pde_stl_strel.training: export validation failed:" + bullet + bullet.join(problems)
            )


if TYPE_CHECKING:  # pragma: no cover
    from . import grids as grids  # noqa: F401
    from .grids import (  # noqa: F401
        Box1D,
        Box2D,
        Box3D,
        grid1d,
        grid2d,
        grid3d,
        sample_boundary_1d,
        sample_boundary_2d,
        sample_boundary_3d,
        sample_interior_1d,
        sample_interior_2d,
        sample_interior_3d,
        spacing1d,
        spacing2d,
        spacing3d,
    )
