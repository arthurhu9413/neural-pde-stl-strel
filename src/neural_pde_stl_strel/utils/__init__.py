# ruff: noqa: I001
# isort: skip_file
"""Lazy-import shim for :mod:`neural_pde_stl_strel.utils`.

This subpackage intentionally keeps import time tiny: its public utilities are
loaded **on first access** via :pep:`562` (module-level ``__getattr__`` and
``__dir__``).

Why this exists
The wider *neural-pde-stl-strel* project has many optional integrations (e.g.,
PyTorch, Neuromancer, PhysicsNeMo, TorchPhysics, RTAMT, MoonLight). Keeping
``neural_pde_stl_strel.utils`` light avoids pulling any of those optional
dependencies just to import the top-level package.

Public API
The stable, supported names exported by this package are:

- ``seed`` and ``logger`` submodules (lazy)
- ``seed_everything``, ``seed_worker``, ``torch_generator``
- ``CSVLogger``

Implementation notes
- Uses :func:`importlib.import_module` with a **package anchor** (``__name__``)
  so relative imports are correct under all packaging layouts.
- Caches resolved objects in ``globals()`` so subsequent attribute access is
  effectively free.
- Uses a small re-entrant lock to avoid rare races when multiple threads access
  the same attribute concurrently for the first time.

References
- :pep:`562` -- Module ``__getattr__`` and ``__dir__``.
- Python docs: :func:`importlib.import_module`.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from threading import RLock
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

# Map public attribute name -> "relative.module[:qualified_object]".
# Submodules and re-exports are imported on first access and then cached.
_LAZY: Final[Mapping[str, str]] = MappingProxyType(
    {
        # Submodules (kept import-light)
        "seed": ".seed",
        "logger": ".logger",
        # Re-exports from the submodules
        "seed_everything": ".seed:seed_everything",
        "seed_worker": ".seed:seed_worker",
        "torch_generator": ".seed:torch_generator",
        "CSVLogger": ".logger:CSVLogger",
    }
)

# Export exactly what the package intends to expose, in insertion order.
# Using the mapping as the single source of truth avoids drift.
__all__: tuple[str, ...] = tuple(_LAZY)

# A tiny lock prevents rare races on first access under concurrency.
_LOCK: Final = RLock()


def _resolve_qualname(obj: Any, qualname: str) -> Any:
    """Resolve a dotted attribute path (e.g. ``\"a.b.c\"``) from *obj*."""
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def __getattr__(name: str) -> Any:  # pragma: no cover - tiny import shim
    """Lazily resolve attributes defined in :data:`_LAZY` on first access."""

    target = _LAZY.get(name)
    if target is None:
        # Avoid importing difflib on the hot path -- only on attribute errors.
        hint: str | None = None
        try:
            from difflib import get_close_matches  # local import: error path only
        except ImportError:  # pragma: no cover
            pass
        else:
            matches = get_close_matches(name, _LAZY.keys(), n=1, cutoff=0.8)
            if matches:
                hint = matches[0]

        msg = f"module {__name__!r} has no attribute {name!r}"
        if hint is not None:
            msg += f". Did you mean {hint!r}?"
        raise AttributeError(msg)

    g = globals()

    # Fast-path if another thread already cached the attribute.
    if name in g:
        return g[name]

    with _LOCK:
        # Re-check under the lock.
        if name in g:
            return g[name]

        mod_name, sep, qual = target.partition(":")
        module = import_module(mod_name, __name__)
        value = module if not sep else _resolve_qualname(module, qual)

        g[name] = value  # cache for subsequent lookups
        return value


def __dir__() -> list[str]:  # pragma: no cover - tiny shim
    """Return a minimal, stable list of attributes for tab-completion."""

    dunders = {k for k in globals() if k.startswith("__") and k.endswith("__")}
    return sorted(set(__all__) | dunders)


# Help IDEs and type checkers with concrete symbols without runtime cost.
if TYPE_CHECKING:  # pragma: no cover
    from . import logger, seed  # noqa: F401
    from .logger import CSVLogger  # noqa: F401
    from .seed import seed_everything, seed_worker, torch_generator  # noqa: F401
