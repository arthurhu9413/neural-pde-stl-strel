"""neural_pde_stl_strel.monitors

Import-safe *demo* monitors for STL / STREL tooling.

This subpackage contains small "hello world" examples that demonstrate how to
**monitor** temporal (STL) and spatio-temporal (STREL) specifications using
optional third-party libraries:

- **RTAMT** (STL monitoring)
- **MoonLight** (temporal + STREL monitoring via a Java engine with Python bindings)
- **SpaTiaL** (a small slice of the SpaTiaL project, via the ``spatial-lib`` codebase)

The demo modules are intentionally lightweight and safe to import in minimal
environments: they avoid importing heavy/optional dependencies (Java/MoonLight,
RTAMT, SpaTiaL) at module import time. Instead, each demo function imports its
backend lazily when executed.

If you are looking for the *reusable* monitoring utilities (e.g., differentiable
STL losses used by the PINN examples), see :mod:`neural_pde_stl_strel.monitoring`.

Public API
Lazy submodules (imported on first attribute access):

- :mod:`neural_pde_stl_strel.monitors.rtamt_hello`
- :mod:`neural_pde_stl_strel.monitors.moonlight_hello`
- :mod:`neural_pde_stl_strel.monitors.moonlight_strel_hello`
- :mod:`neural_pde_stl_strel.monitors.spatial_demo`

Helpers:

- :func:`available_backends` / :func:`probe_backend` for quick environment probes
- :func:`require_backend` for actionable, user-facing error messages

Design goals
1. **Import safety**: importing :mod:`neural_pde_stl_strel.monitors` must *never*
   require Java/RTAMT/SpaTiaL to be installed.
2. **Helpful failures**: if a demo is invoked without its backend installed,
   raise an error that clearly explains how to fix the environment.
3. **IDE friendliness**: expose a stable, type-checker-visible surface area
   without eager imports (PEP 562-style lazy exports).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module, metadata as importlib_metadata, util as importlib_util
import os
from typing import TYPE_CHECKING, Any, Final

from .._versioning import version_satisfies_minimum

# Lazy submodule exports (PEP 562)

_SUBMODULES: Final[Mapping[str, str]] = {
    "rtamt_hello": "neural_pde_stl_strel.monitors.rtamt_hello",
    "moonlight_hello": "neural_pde_stl_strel.monitors.moonlight_hello",
    "moonlight_strel_hello": "neural_pde_stl_strel.monitors.moonlight_strel_hello",
    "spatial_demo": "neural_pde_stl_strel.monitors.spatial_demo",
}

if TYPE_CHECKING:  # pragma: no cover
    # For static analyzers: make the public submodules discoverable without
    # forcing eager imports at runtime.
    from . import moonlight_hello, moonlight_strel_hello, rtamt_hello, spatial_demo  # noqa: F401

# Optional backend probing

# NOTE: ``spatial`` is a common package name on PyPI.  The SpaTiaL project's
# ``spatial-lib`` (used by this repository) contains the submodule
# ``spatial.logic``.  Probing that submodule helps avoid false positives.
_BACKEND_PROBES: Final[Mapping[str, str]] = {
    "rtamt": "rtamt",
    "moonlight": "moonlight",
    "spatial": "spatial.logic",
}

# Candidate *distribution* names (for ``importlib.metadata.version`` lookups).
# These do not affect availability; availability is determined by probing the
# import system for the module named in ``_BACKEND_PROBES``.
_DIST_CANDIDATES: Final[Mapping[str, tuple[str, ...]]] = {
    "rtamt": ("rtamt",),
    "moonlight": ("moonlight",),
    # When installed from SpaTiaL via PEP 508 direct reference, the distribution
    # name is often set to ``spatial`` (see requirements-extra.txt in this repo).
    "spatial": ("spatial",),
}

# Human-oriented installation hints tailored to *this repository*.
# (We intentionally do not claim that a PyPI extra exists unless it actually does.)
_INSTALL_HINTS: Final[Mapping[str, str]] = {
    "rtamt": "python -m pip install rtamt",
    "moonlight": "python -m pip install moonlight",
    "spatial": (
        "python -m pip install \"spatial @ "
        "git+https://github.com/KTH-RPL-Planiacs/SpaTiaL.git#subdirectory=spatial-lib\"\n"
        "  (or: python -m pip install -r requirements-extra.txt)"
    ),
}

_BACKEND_NOTES: Final[Mapping[str, str]] = {
    "moonlight": "MoonLight requires a Java runtime (Java 21+) available on PATH.",
    "spatial": (
        "This demo expects the SpaTiaL 'spatial-lib' codebase (import 'spatial.logic').\n"
        "The separate PyPI package 'spatial-spec' (import 'spatial_spec') is *not* sufficient for this demo."
    ),
}


@dataclass(frozen=True, slots=True)
class BackendInfo:
    """Result of probing a backend dependency."""

    name: str
    available: bool
    version: str | None = None
    dist: str | None = None
    error: str | None = None


def _try_find_spec(module_name: str) -> tuple[bool, str | None]:
    """Return (available, error) for ``module_name`` without importing it.

    ``importlib.util.find_spec()`` sometimes raises :class:`ModuleNotFoundError`
    for missing parent packages when probing dotted names (e.g., ``spatial.logic``).
    In that case we treat the backend as simply *unavailable* (no error).
    """
    try:
        return importlib_util.find_spec(module_name) is not None, None
    except ModuleNotFoundError:
        return False, None
    except Exception as e:  # pragma: no cover
        return False, str(e)


def _get_version(backend_name: str) -> tuple[str | None, str | None, str | None]:
    """Best-effort (version, dist, error) triple for an available backend."""
    for dist in _DIST_CANDIDATES.get(backend_name, (backend_name,)):
        try:
            return importlib_metadata.version(dist), dist, None
        except Exception:
            continue

    # Fall back to ``__version__`` if metadata is missing (e.g., editable installs).
    try:
        mod = import_module(backend_name)
        ver = getattr(mod, "__version__", None)
        return (str(ver) if ver is not None else None), None, None
    except Exception as e:  # pragma: no cover
        return None, None, f"version lookup failed: {e}"


@lru_cache(maxsize=None)
def probe_backend(name: str) -> BackendInfo:
    """Probe a single backend by name.

    Parameters
    name:
        One of ``{"rtamt", "moonlight", "spatial"}``.

    Returns
    -------
    BackendInfo
        Availability and (best-effort) version information.
    """
    probe_name = _BACKEND_PROBES.get(name, name)
    spec_ok, err = _try_find_spec(probe_name)
    if not spec_ok:
        return BackendInfo(name=name, available=False, error=err)

    version, dist, err = _get_version(name)
    return BackendInfo(name=name, available=True, version=version, dist=dist, error=err)


# Cache for the JSON-ish view returned by ``available_backends``.
_AVAILABLE_CACHE: dict[str, dict[str, bool | str | None]] | None = None


def _no_probe_enabled() -> bool:
    """Return True if probing should be skipped for ``available_backends()``.

    This is a convenience knob for very constrained environments and CI runs.
    It does *not* affect :func:`probe_backend` or :func:`require_backend`.

    Any of the following values enable the flag (case-insensitive):
    ``1``, ``true``, ``yes``, ``on``.
    """
    raw = os.getenv("NEURAL_PDE_STL_STREL_NO_PROBE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def clear_backend_cache() -> None:
    """Clear cached backend probe results."""
    global _AVAILABLE_CACHE
    _AVAILABLE_CACHE = None
    probe_backend.cache_clear()


def available_backends() -> dict[str, dict[str, bool | str | None]]:
    """Detect optional monitoring toolkits.

    Returns a mapping from backend name to a small, JSON-serializable dict with:
    ``available`` (bool), ``version`` (str|None), ``dist`` (str|None), and
    ``error`` (str|None).

    If the environment variable ``NEURAL_PDE_STL_STREL_NO_PROBE=1`` (or ``true/yes/on``) is
    set, this returns an empty mapping immediately.
    """
    if _no_probe_enabled():
        return {}

    global _AVAILABLE_CACHE
    if _AVAILABLE_CACHE is None:
        out: dict[str, dict[str, bool | str | None]] = {}
        for name in _BACKEND_PROBES:
            info = probe_backend(name)
            out[name] = {
                "available": info.available,
                "version": info.version,
                "dist": info.dist,
                "error": info.error,
            }
        _AVAILABLE_CACHE = out

    # Return a shallow copy to keep callers from mutating the cache.
    return dict(_AVAILABLE_CACHE)


def rtamt_available() -> bool:
    """Return True iff RTAMT is importable."""
    return probe_backend("rtamt").available


def moonlight_available() -> bool:
    """Return True iff MoonLight's Python bindings are importable."""
    return probe_backend("moonlight").available


def spatial_available() -> bool:
    """Return True iff SpaTiaL's ``spatial-lib`` (``spatial.logic``) is importable."""
    return probe_backend("spatial").available


def require_backend(name: str, *, min_version: str | None = None) -> None:
    """Ensure that a backend is available (and optionally meets a minimum version).

    Parameters
    name:
        Backend name (see :func:`probe_backend`).
    min_version:
        If provided, enforce ``backend_version >= min_version`` when a version
        can be determined.

    Raises
    ------
    ModuleNotFoundError
        If the backend cannot be imported.
    RuntimeError
        If a minimum version is requested but not satisfied.
    """
    info = probe_backend(name)
    if not info.available:
        hint = _INSTALL_HINTS.get(name, f"python -m pip install {name}")
        note = _BACKEND_NOTES.get(name)
        msg_lines = [f"Optional backend '{name}' is not available."]

        if info.error:
            msg_lines.append(str(info.error))

        msg_lines.append("\nHow to install:")
        msg_lines.append(f"  {hint}")

        if note:
            msg_lines.append("\nNotes:")
            msg_lines.append(f"  {note}")

        # Make it easy to find the canonical instructions when working in-repo.
        msg_lines.append("\nSee also: docs/INSTALL_EXTRAS.md (in this repository).")

        raise ModuleNotFoundError("\n".join(msg_lines))

    if min_version and info.version is not None and not version_satisfies_minimum(info.version, min_version):
        raise RuntimeError(
            f"Backend '{name}' version {info.version} is older than required {min_version}."
        )


def __getattr__(name: str) -> Any:  # noqa: D401 - required by PEP 562
    """Dynamically import demo submodules on first access."""
    mod_path = _SUBMODULES.get(name)
    if mod_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        mod = import_module(mod_path)
    except ModuleNotFoundError as e:
        # Surface a more actionable message when the missing module is an
        # optional backend dependency (as opposed to a project bug).
        missing = getattr(e, "name", None)
        if missing in _BACKEND_PROBES.values() or missing in {"rtamt", "moonlight", "spatial"}:
            raise ModuleNotFoundError(
                f"Failed to import demo module {mod_path!r} because optional dependency {missing!r} is missing.\n"
                f"Try installing the relevant backend (see available_backends()/require_backend())."
            ) from e
        raise

    globals()[name] = mod
    return mod


def __dir__() -> list[str]:
    # Provide predictable tab-completion / introspection.
    public = set(__all__)
    public.update(_SUBMODULES.keys())
    public.update(globals().keys())
    return sorted(public)


__all__ = [
    # Lazy submodules
    *sorted(_SUBMODULES.keys()),
    # Probe helpers
    "BackendInfo",
    "available_backends",
    "clear_backend_cache",
    "moonlight_available",
    "probe_backend",
    "require_backend",
    "rtamt_available",
    "spatial_available",
]
