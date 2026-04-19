# ruff: noqa: I001
# isort: skip_file
from __future__ import annotations

"""neural_pde_stl_strel.experiments

Light-weight experiment registry and lazy imports.

This subpackage is intentionally dependency-light at *import time* (it should
not import PyTorch or any other heavy ML stack until you actually run an
experiment).

Public API
The following functions are considered stable and are safe to rely on from
scripts/notebooks:

- ``names()``: list known experiment keys.
- ``available()``: best-effort availability probe (no heavy imports).
- ``describe(name)``: human-readable details and module docstring.
- ``get_runner(name)``: resolve a runner callable.
- ``run(name, config, **kwargs)``: convenience wrapper around ``get_runner``.
- ``register(name, runner)``: register a custom experiment.
- ``about()``: formatted listing used by CLI helpers.

Lazy exports
Bundled experiment modules (``diffusion1d``, ``heat2d``) and common symbols
(e.g., ``run_diffusion1d``) are exposed lazily via PEP 562
(module ``__getattr__``/``__dir__``).

Plugins
-------
Third-party experiments may be advertised via the optional entry-point group
``neural_pde_stl_strel.experiments``. Each entry point name becomes an experiment
key, and its value should be of the form ``"pkg.mod:callable"``.

Discovery is best-effort and performed lazily on first registry access.
"""

from collections.abc import Callable, Mapping
import ast as _ast
from importlib import import_module as _import_module
from importlib import util as _import_util
import tokenize as _tokenize
from typing import Any, Protocol, TYPE_CHECKING

__all__ = [
    "about",
    "available",
    "describe",
    "get_runner",
    "names",
    "register",
    "run",
]

_ENTRYPOINT_GROUP = "neural_pde_stl_strel.experiments"
_TORCH_MODULE = "torch"


# Built-in experiments shipped with this repository/package.
_EXPERIMENTS: dict[str, str | Callable[..., Any]] = {
    "diffusion1d": "neural_pde_stl_strel.experiments.diffusion1d:run_diffusion1d",
    "heat2d": "neural_pde_stl_strel.experiments.heat2d:run_heat2d",
}

# Keep a snapshot of built-ins so we can treat them differently from plugins.
_BUILTIN_EXPERIMENTS: frozenset[str] = frozenset(_EXPERIMENTS.keys())

_DOCS: dict[str, str] = {
    "diffusion1d": "1-D diffusion (heat) PINN with optional STL penalty.",
    "heat2d": "2-D heat equation (toy) with optional STREL monitoring.",
}

_LAZY_MODULES: dict[str, str] = {
    "diffusion1d": "neural_pde_stl_strel.experiments.diffusion1d",
    "heat2d": "neural_pde_stl_strel.experiments.heat2d",
}

# Optional convenience exports so users can do:
#   from neural_pde_stl_strel.experiments import run_diffusion1d
# without importing torch unless they actually use it.
_FORWARD_ATTRS: dict[str, tuple[str, str]] = {
    "run_diffusion1d": ("diffusion1d", "run_diffusion1d"),
    "run_heat2d": ("heat2d", "run_heat2d"),
    "Diffusion1DConfig": ("diffusion1d", "Diffusion1DConfig"),
    "Heat2DConfig": ("heat2d", "Heat2DConfig"),
}

_HAS_TORCH: bool | None = None
_PLUGINS_DISCOVERED: bool = False


class Runner(Protocol):
    """Callable signature for experiment runners."""

    def __call__(
        self,
        config: Mapping[str, Any] | None = None,
        /,
        **kwargs: Any,
    ) -> Any: ...  # pragma: no cover


def _normalize_name(name: str) -> str:
    return name.strip().lower()


def _discover_plugins_once() -> None:
    """Populate the registry from entry points (best-effort).

    Kept lazy (called on first public registry access) to minimize import-time
    overhead.
    """

    global _PLUGINS_DISCOVERED
    if _PLUGINS_DISCOVERED:
        return
    _PLUGINS_DISCOVERED = True

    try:
        from importlib import metadata as _md  # type: ignore[attr-defined]
    except Exception:
        return

    try:
        eps = _md.entry_points()
        if hasattr(eps, "select"):
            candidates = eps.select(group=_ENTRYPOINT_GROUP)
        else:
            candidates = eps.get(_ENTRYPOINT_GROUP, [])

        for ep in candidates:
            key = _normalize_name(str(getattr(ep, "name", "")))
            if not key or key in _EXPERIMENTS:
                continue

            value = str(getattr(ep, "value", "")).strip()
            if not value:
                continue

            # Keep the raw value string; resolve lazily in get_runner().
            _EXPERIMENTS[key] = value
    except Exception:
        # Never fail import because plugin metadata is malformed.
        return


def _has_torch() -> bool:
    global _HAS_TORCH
    if _HAS_TORCH is None:
        _HAS_TORCH = _import_util.find_spec(_TORCH_MODULE) is not None
    return _HAS_TORCH


def _torch_error_message() -> str:
    if _has_torch():
        return (
            "PyTorch (torch) appears to be installed but could not be imported.\n"
            "This often indicates a broken/mismatched wheel (e.g., CUDA/ABI).\n\n"
            "Try reinstalling a compatible PyTorch build. For a CPU-only wheel:\n"
            "  python -m pip install --force-reinstall --index-url "
            "https://download.pytorch.org/whl/cpu torch\n"
        )

    return (
        "PyTorch (torch) is required for this experiment but is not installed.\n\n"
        "Install the project requirements (in the repository):\n"
        "  python -m pip install -r requirements.txt\n\n"
        "Or install PyTorch directly (CPU-only wheel):\n"
        "  python -m pip install --index-url https://download.pytorch.org/whl/cpu "
        "torch\n"
    )


def _import_with_friendly_error(mod_path: str) -> Any:
    """Import a module, adding a friendlier message for common optional deps."""

    try:
        return _import_module(mod_path)
    except ModuleNotFoundError as e:
        # Only special-case our built-in experiments; third-party plugins may
        # have arbitrary dependency patterns.
        if mod_path.startswith("neural_pde_stl_strel.experiments."):
            missing = getattr(e, "name", "") or ""
            if missing == _TORCH_MODULE or missing.startswith(f"{_TORCH_MODULE}."):
                raise ImportError(_torch_error_message()) from e
        raise
    except ImportError as e:
        if mod_path.startswith("neural_pde_stl_strel.experiments."):
            msg = str(e).lower()
            if "torch" in msg or "libtorch" in msg or "pytorch" in msg:
                raise ImportError(_torch_error_message()) from e
        raise


def _strip_entrypoint_extras(target: str) -> str:
    """Remove optional ``[extra]`` markers from an entry-point value string."""

    s = target.strip()
    bracket = s.find("[")
    if bracket != -1:
        s = s[:bracket].strip()
    return s


def _split_target(target: str) -> tuple[str, str]:
    """Split a ``module:callable`` target string.

    The value in entry points may optionally include extras at the end
    (``module:callable [extra]``); these are ignored.
    """

    base = _strip_entrypoint_extras(target)
    if ":" not in base:
        raise ValueError(f"Invalid experiment target {target!r} (expected 'module:callable')")
    mod, func = base.split(":", 1)
    mod = mod.strip()
    func = func.strip()
    if not mod or not func:
        raise ValueError(f"Invalid experiment target {target!r} (expected 'module:callable')")
    return mod, func


def _module_docstring(module_path: str) -> str | None:
    """Return a module docstring without importing the module (best-effort)."""

    try:
        spec = _import_util.find_spec(module_path)
    except Exception:
        return None

    origin = getattr(spec, "origin", None) if spec is not None else None
    if not origin or origin in {"built-in", "frozen"}:
        return None

    try:
        with _tokenize.open(origin) as f:
            source = f.read()
        tree = _ast.parse(source)
        doc = _ast.get_docstring(tree)
        return (doc or "").strip()
    except Exception:
        return None


def names() -> list[str]:
    """Return known experiment keys."""

    _discover_plugins_once()
    return sorted(_EXPERIMENTS)


def available() -> dict[str, bool]:
    """Return a best-effort map of experiment key -> availability.

    Notes
    -----
    - Built-in experiments shipped in this repository require PyTorch.
    - For plugin targets registered as callables, availability is assumed.
    - For plugin targets registered as ``"module:callable"`` strings, this
      checks whether the module *can be found* (it does not import it).
    """

    _discover_plugins_once()

    has_torch = _has_torch()
    out: dict[str, bool] = {}

    for name, target in _EXPERIMENTS.items():
        if callable(target):
            out[name] = True
            continue

        # Built-ins: treat availability as "PyTorch importable".
        if name in _BUILTIN_EXPERIMENTS:
            out[name] = has_torch
            continue

        # Plugins: treat availability as "module discoverable".
        try:
            mod_path, _ = _split_target(target)
        except ValueError:
            out[name] = False
            continue

        try:
            out[name] = _import_util.find_spec(mod_path) is not None
        except Exception:
            out[name] = False

    return out


def register(name: str, runner: Runner | str) -> None:
    """Register a custom experiment runner.

    Parameters
    name:
        The experiment key. Keys are normalized case-insensitively.
    runner:
        Either a callable (preferred) or a ``"module:callable"`` string.
    """

    key = _normalize_name(name)
    _EXPERIMENTS[key] = runner


def get_runner(name: str) -> Runner:
    """Resolve and return a runner callable for an experiment."""

    _discover_plugins_once()

    key = _normalize_name(name)
    if key not in _EXPERIMENTS:
        raise KeyError(f"Unknown experiment {name!r}. Available: {', '.join(names())}")

    target = _EXPERIMENTS[key]
    if callable(target):
        return target  # type: ignore[return-value]

    mod_path, func_name = _split_target(target)
    mod = _import_with_friendly_error(mod_path)
    runner = getattr(mod, func_name)

    if not callable(runner):
        raise TypeError(f"Experiment target {target!r} resolved to non-callable {runner!r}")

    # Memoize: replace the string target with the resolved callable.
    _EXPERIMENTS[key] = runner
    return runner  # type: ignore[return-value]


def run(name: str, config: Mapping[str, Any] | None = None, /, **kwargs: Any) -> Any:
    """Run an experiment by key."""

    runner = get_runner(name)
    return runner(config, **kwargs)


def about() -> str:
    """Return a short human-readable listing of experiments."""

    avail = available()
    lines: list[str] = ["experiments:"]

    for key in names():
        ok = avail.get(key, False)
        status = "yes" if ok else "no"

        line = f"  - {key:12s} available={status}"
        if not ok and key in _BUILTIN_EXPERIMENTS and not _has_torch():
            line += " (requires torch)"

        blurb = _DOCS.get(key, "").strip()
        if blurb:
            line += f" :: {blurb}"

        lines.append(line)

    return "\n".join(lines)


def describe(name: str) -> str:
    """Return a human-readable description of an experiment (best-effort)."""

    _discover_plugins_once()

    key = _normalize_name(name)
    if key not in _EXPERIMENTS:
        raise KeyError(f"Unknown experiment {name!r}. Available: {', '.join(names())}")

    target = _EXPERIMENTS[key]
    is_available = available().get(key, False)

    if callable(target):
        module = getattr(target, "__module__", "<unknown>")
        qualname = getattr(target, "__qualname__", getattr(target, "__name__", "<callable>"))
        target_str = f"{module}:{qualname}"
        doc = (getattr(target, "__doc__", "") or "").strip() or None
    else:
        target_str = target
        try:
            mod_path, _ = _split_target(target)
        except ValueError:
            doc = None
        else:
            doc = _module_docstring(mod_path)

    lines: list[str] = [
        f"experiment: {key}",
        f"available: {'yes' if is_available else 'no'}",
        f"target: {target_str}",
    ]

    blurb = _DOCS.get(key, "").strip()
    if blurb:
        lines.extend(["", blurb])

    if doc:
        lines.extend(["", doc])

    if not is_available and key in _BUILTIN_EXPERIMENTS:
        lines.extend(["", "Hint:", _torch_error_message().rstrip()])

    return "\n".join(lines).rstrip()


def __getattr__(name: str) -> Any:
    # Stable API helpers.
    if name in __all__:
        return globals()[name]

    # Lazy submodules.
    if name in _LAZY_MODULES:
        mod = _import_with_friendly_error(_LAZY_MODULES[name])
        globals()[name] = mod
        return mod

    # Forwarded symbols from lazy submodules.
    if name in _FORWARD_ATTRS:
        module_key, attr_name = _FORWARD_ATTRS[name]
        mod = __getattr__(module_key)
        attr = getattr(mod, attr_name)
        globals()[name] = attr
        return attr

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    # Include lazy submodules and forwarded names for tab completion.
    public = set(__all__) | set(_LAZY_MODULES) | set(_FORWARD_ATTRS)
    return sorted(public)


if TYPE_CHECKING:
    # For type checkers only: these imports are never evaluated at runtime.
    from . import diffusion1d as diffusion1d
    from . import heat2d as heat2d
    from .diffusion1d import Diffusion1DConfig, run_diffusion1d
    from .heat2d import Heat2DConfig, run_heat2d
