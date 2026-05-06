from __future__ import annotations

"""Top-level package for :mod:`neural_pde_stl_strel`.

This project focuses on neural PDE, PINN, and broader physics-ML workflows
alongside monitoring and training with temporal and spatio-temporal logic
specifications (STL, STREL, ...).

This ``__init__`` is intentionally conservative:

- Keep ``import neural_pde_stl_strel`` fast and side-effect free.
- Lazily expose subpackages via :pep:`562` (``__getattr__`` / ``__dir__``).
- Re-export a couple of tiny utilities used throughout demos.
- Provide lightweight optional-dependency diagnostics (:func:`about`,
  :func:`optional_dependencies`) without importing heavyweight stacks.
"""

from collections.abc import Mapping
from importlib import import_module
from importlib import metadata as _metadata
from importlib import util as _import_util
from typing import TYPE_CHECKING, Any

from ._versioning import version_satisfies_minimum

__all__ = [
    "__version__",
    # Lazy subpackages/modules
    "datasets",
    "experiments",
    "frameworks",
    "models",
    "monitoring",
    "monitors",
    "physics",
    "training",
    "utils",
    "pde_example",
    # Small re-exports
    "seed_everything",
    "CSVLogger",
    # Helpers
    "about",
    "optional_dependencies",
    "require_optional",
]

# Single-source version string used by the CLI (e.g. ``python -m neural_pde_stl_strel about``)
# and by downstream tooling. Keep this as a simple literal for easy discovery.
__version__ = "0.7.9"

# Lazy access to subpackages (PEP 562)

# Map attribute name -> fully qualified module path.
_SUBMODULES: Mapping[str, str] = {
    "datasets": "neural_pde_stl_strel.datasets",
    "experiments": "neural_pde_stl_strel.experiments",
    # NOTE: ``frameworks/`` is a regular subpackage with a lightweight ``__init__``.
    "frameworks": "neural_pde_stl_strel.frameworks",
    "models": "neural_pde_stl_strel.models",
    "monitoring": "neural_pde_stl_strel.monitoring",
    "monitors": "neural_pde_stl_strel.monitors",
    "physics": "neural_pde_stl_strel.physics",
    "training": "neural_pde_stl_strel.training",
    "utils": "neural_pde_stl_strel.utils",
    "pde_example": "neural_pde_stl_strel.pde_example",
}

# Lightweight helpers (attribute -> "module:object").
_HELPERS: Mapping[str, str] = {
    "seed_everything": "neural_pde_stl_strel.utils.seed:seed_everything",
    "CSVLogger": "neural_pde_stl_strel.utils.logger:CSVLogger",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - tiny shim
    """Dynamically resolve lazily-exposed attributes."""

    if name in _SUBMODULES:
        module = import_module(_SUBMODULES[name])
        globals()[name] = module
        return module

    if name in _HELPERS:
        mod_name, obj_name = _HELPERS[name].split(":", 1)
        obj = getattr(import_module(mod_name), obj_name)
        globals()[name] = obj
        return obj

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - tiny shim
    """Return a helpful list of attributes for :func:`dir` and tab-completion."""

    public = set(__all__)
    public.update(n for n in globals() if n.startswith("__"))
    return sorted(public)


# Optional dependency inspection

# Optional dependency registry.
#
# Keys are **import names** used throughout this repository.
# Values are **distribution names** for :func:`importlib.metadata.version`.
#
# Notes:
#   * NVIDIA's Modulus project has been renamed to PhysicsNeMo; some ecosystems
#     still use ``modulus`` while newer code uses ``physicsnemo``.
#   * SpaTiaL Specifications is published on PyPI as ``spatial-spec`` but is
#     imported as ``spatial_spec``.
_OPT_DIST_NAMES: Mapping[str, str] = {
    # Core scientific stack
    "numpy": "numpy",
    "pyyaml": "PyYAML",
    "torch": "torch",
    # Physics-ML frameworks
    "neuromancer": "neuromancer",  # PNNL NeuroMANCER
    "physicsnemo": "nvidia-physicsnemo",
    "modulus": "nvidia-modulus",
    "torchphysics": "torchphysics",  # Bosch TorchPhysics
    # STL / spatio-temporal monitoring
    "rtamt": "rtamt",
    "moonlight": "moonlight",
    "spatial_spec": "spatial-spec",
    # SpaTiaL runtime library (optional; installed from source in requirements-extra)
    "spatial": "spatial",
}

# Some import names are ambiguous on PyPI. When probing for availability, we can
# optionally check for a deeper module path that is characteristic of the
# intended dependency.
_OPT_PROBE_TARGETS: Mapping[str, str] = {
    # PyYAML is imported as ``yaml`` but distributed as ``PyYAML``.
    "pyyaml": "yaml",
    # Avoid false positives from unrelated packages named ``spatial``.
    "spatial": "spatial.logic",
}

# Human-readable installation hints for nicer error messages.
_INSTALL_HINT: Mapping[str, str] = {
    "numpy": "pip install numpy",
    "pyyaml": "pip install pyyaml",
    "torch": "pip install torch",
    "neuromancer": "pip install neuromancer",
    "physicsnemo": "pip install nvidia-physicsnemo",
    "modulus": "pip install nvidia-modulus",
    "torchphysics": "pip install torchphysics",
    "rtamt": "create a Python 3.11 environment and run `pip install rtamt`",
    "moonlight": "pip install moonlight",
    "spatial_spec": "pip install spatial-spec",
    # IMPORTANT: Do *not* suggest "pip install spatial" (a different PyPI project).
    "spatial": (
        'pip install "spatial @ '
        'git+https://github.com/KTH-RPL-Planiacs/SpaTiaL.git#subdirectory=spatial-lib"'
    ),
}

# Minimum versions that the repository actively relies on when a dependency
# is present for CLI requirement checks. Keep this aligned with
# ``pyproject.toml`` and ``scripts/check_env.py``.
_MIN_REQUIRED_VERSION: Mapping[str, str] = {
    "numpy": "1.24",
    "pyyaml": "6.0",
    "torch": "2.0",
    "rtamt": "0.3",
    "moonlight": "0.3",
}

# Cache probe results to keep repeated environment reports O(1).
_OPT_CACHE: dict[str, tuple[bool, str | None]] = {}


def _unique_preserve_order(items: list[str]) -> list[str]:
    """Return a de-duplicated copy of *items* preserving first-seen order."""

    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _probe_module(mod_name: str) -> tuple[bool, str | None]:
    """Return ``(available, version)`` for *mod_name*.

    Availability is checked with :func:`importlib.util.find_spec` (no import side
    effects). Version resolution relies on distribution metadata and does not
    import the module.
    """

    cached = _OPT_CACHE.get(mod_name)
    if cached is not None:
        return cached

    probe_name = _OPT_PROBE_TARGETS.get(mod_name, mod_name)
    try:
        spec_found = _import_util.find_spec(probe_name) is not None
    except (ModuleNotFoundError, ValueError):
        # find_spec raises ModuleNotFoundError for dotted names (e.g.
        # "spatial.logic") when the parent package is not installed.
        # ValueError can occur with certain namespace package edge cases.
        spec_found = False
    if not spec_found:
        result = (False, None)
        _OPT_CACHE[mod_name] = result
        return result

    # Try the explicitly configured dist name first, then fall back to the
    # dynamic mapping from top-level packages to distributions.
    dist_candidates: list[str] = []
    explicit = _OPT_DIST_NAMES.get(mod_name)
    if explicit:
        dist_candidates.append(explicit)

    top_level_pkg = probe_name.split(".", 1)[0]
    try:
        packages_distributions = getattr(_metadata, "packages_distributions", None)
        if packages_distributions is not None:
            dist_candidates.extend(packages_distributions().get(top_level_pkg, []))
    except Exception:
        # Best-effort only; keep probing resilient across environments.
        pass

    version: str | None = None
    for dist_name in _unique_preserve_order(dist_candidates):
        try:
            version = _metadata.version(dist_name)
            break
        except _metadata.PackageNotFoundError:
            continue
        except Exception:
            continue

    result = (True, version)
    _OPT_CACHE[mod_name] = result
    return result


def optional_dependencies(
    refresh: bool = False,
    include_pip_hints: bool = True,
) -> dict[str, dict[str, str | bool | None]]:
    """Inspect the small dependency set reported by the CLI.

    This report intentionally includes the required NumPy / PyYAML base
    dependencies plus the repository's optional framework / monitor
    integrations so that a single command can summarize the environment
    without importing heavyweight stacks.

    Parameters
    refresh:
        If ``True``, clear cached probe results and re-scan the environment.
    include_pip_hints:
        If ``True`` (default), include a concise ``"pip"`` hint for packages
        that are not available.

    Returns
    -------
    dict
        Mapping from import name to a small record:

        ``{"available": bool, "version": Optional[str], "pip": Optional[str]}``

        The ``"pip"`` key is present only when ``include_pip_hints=True``.
    """

    if refresh:
        _OPT_CACHE.clear()

    report: dict[str, dict[str, str | bool | None]] = {}
    for mod in _OPT_DIST_NAMES.keys():
        ok, ver = _probe_module(mod)
        item: dict[str, str | bool | None] = {"available": ok, "version": ver}
        if include_pip_hints and not ok:
            item["pip"] = _INSTALL_HINT.get(mod)
        report[mod] = item
    return report


def require_optional(mod_name: str, min_version: str | None = None) -> None:
    """Assert that an optional dependency is available (and optionally recent).

    This helper keeps import sites clean and produces actionable errors.

    Parameters
    mod_name:
        The **import name** (e.g. ``"moonlight"`` or ``"spatial_spec"``).
    min_version:
        Optional minimum version string. If supplied and resolvable, an
        informative ``ImportError`` is raised when the installed version is
        older.

    Raises
    ------
    ImportError
        If the module is not discoverable or does not satisfy ``min_version``.
    """

    ok, found_version = _probe_module(mod_name)
    if not ok:
        dist = _OPT_DIST_NAMES.get(mod_name) or mod_name
        hint = _INSTALL_HINT.get(mod_name) or f"pip install {dist}"

        msg = (
            f"Optional dependency '{mod_name}' is required but not installed.\n"
            f"-> Install via: {hint}"
        )
        if mod_name in {"physicsnemo", "modulus"}:
            msg += (
                "\nNOTE: NVIDIA's 'Modulus' project has been renamed to 'PhysicsNeMo'. "
                "Some environments may have only one of these installed."
            )
        raise ImportError(msg)

    if min_version and found_version and not version_satisfies_minimum(found_version, min_version):
        raise ImportError(
            f"'{mod_name}' version >= {min_version} required; found {found_version}."
        )


def about() -> str:
    """Return a compact, human-readable environment summary."""

    lines = [f"neural_pde_stl_strel {__version__}", "Dependency probes:"]
    report = optional_dependencies(include_pip_hints=False)

    width = max((len(k) for k in report), default=0)
    for name in sorted(report):
        avail = report[name]["available"]
        ver = report[name]["version"]
        lines.append(f"  {name.ljust(width)}  {'yes' if avail else 'no ':<3} ({ver or '-'})")

    # Friendly one-liner about the Modulus -> PhysicsNeMo rename if relevant.
    phys_ok = bool(report.get("physicsnemo", {}).get("available"))
    mod_ok = bool(report.get("modulus", {}).get("available"))
    if mod_ok and not phys_ok:
        lines.append("  note: 'modulus' is installed; consider migrating to 'physicsnemo'.")

    return "\n".join(lines)


# Static imports for type checkers only

if TYPE_CHECKING:  # pragma: no cover
    # These imports help IDEs and type checkers without paying runtime cost.
    from . import (  # noqa: F401
        datasets,
        experiments,
        frameworks,
        models,
        monitoring,
        monitors,
        pde_example,
        physics,
        training,
        utils,
    )
    from .utils.logger import CSVLogger as CSVLogger  # noqa: F401
    from .utils.seed import seed_everything as seed_everything  # noqa: F401
