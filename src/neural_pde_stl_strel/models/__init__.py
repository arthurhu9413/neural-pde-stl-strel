# ruff: noqa: I001
# isort: skip_file
"""neural_pde_stl_strel.models

Lightweight model shims + a tiny registry.

This repository is designed to be *CPU-friendly* and easy to run in minimal
environments. In particular, the core package should remain importable even if
optional/heavy dependencies (notably **PyTorch**) are not installed.

Accordingly:

- Public classes (e.g., :class:`~neural_pde_stl_strel.models.mlp.MLP`) are **lazy
  re-exports** (PEP 562) and are imported only when accessed.
- ``build()`` / ``from_spec()`` provide a small, ergonomic factory for creating
  models from short names (e.g., ``"mlp"``) or explicit import targets
  (e.g., ``"pkg.mod:Class"``).

This is intentionally minimal--it's meant to support experiments in this repo,
not to be a full plugin framework.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from importlib import import_module
from importlib import util as _import_util
from threading import RLock
from typing import TYPE_CHECKING, Any, Final, TypeAlias


# Lazy re-exports (keeps imports fast; avoids importing torch at module import)

# Map public attribute name -> "relative.module[:qualified_object]".
_LAZY: dict[str, str] = {
    # Submodule
    "mlp": ".mlp",
    # Re-exports from the submodule
    "MLP": ".mlp:MLP",
    "Sine": ".mlp:Sine",
}


# Public API

ModelBuilder: TypeAlias = Callable[..., Any]
# Callable that constructs a model instance.

ModelSpec: TypeAlias = Mapping[str, Any]
# Mapping-based specification accepted by :func:`from_spec`.
#
# Common keys:
# - ``name`` / ``model`` / ``type``: registry name (e.g., ``"mlp"``)
# - ``target``: import target (e.g., ``"neural_pde_stl_strel.models.mlp:MLP"``)
# - ``args``: positional args (list/tuple)
# - ``kwargs``: keyword args (mapping)
#
# Any remaining keys are treated as keyword args.

__all__: tuple[str, ...] = (
    # Submodules / re-exports (lazy)
    "mlp",
    "MLP",
    "Sine",
    # Registry surface
    "ModelBuilder",
    "ModelSpec",
    "register",
    "register_model",
    "available",
    "get_builder",
    "build",
    "from_spec",
    "about",
)


# A small lock prevents rare races when resolving lazy attributes concurrently.
_LOCK: Final[RLock] = RLock()


def _torch_is_missing() -> bool:
    """Return True iff PyTorch cannot be imported (without importing it)."""

    return _import_util.find_spec("torch") is None


def _maybe_rewrite_import_error(err: ImportError) -> ImportError:
    """Add a friendlier hint when the missing dependency is PyTorch."""

    if not _torch_is_missing():
        return err

    # ModuleNotFoundError is a subclass of ImportError and carries a `.name`.
    missing_name = getattr(err, "name", "") or ""
    msg = str(err).lower()

    # Heuristic: only rewrite if the error is plausibly about torch.
    if "torch" in msg or "pytorch" in msg or missing_name.startswith("torch"):
        hint = (
            "PyTorch is required for `neural_pde_stl_strel.models` but is not installed.\n\n"
            "Install it via one of the following options:\n"
            "  - pip install \"neural-pde-stl-strel[torch]\"\n"
            "  - pip install -r requirements-extra.txt\n"
            "  - or install PyTorch directly from https://pytorch.org\n"
        )
        rewritten = ImportError(hint)
        # Preserve the original exception as the direct cause.
        rewritten.__cause__ = err
        rewritten.__suppress_context__ = True
        return rewritten

    return err


def __getattr__(name: str) -> Any:  # pragma: no cover - import shim
    """Resolve lazy re-exports declared in :data:`_LAZY`.

    The resolved value is cached in ``globals()`` for zero-cost subsequent
    lookups.
    """

    target = _LAZY.get(name)
    if target is None:
        # Keep difflib off the hot path (import only on error).
        try:
            from difflib import get_close_matches  # local import: error path only
        except Exception:
            get_close_matches = None  # type: ignore[assignment]

        hint = ""
        if get_close_matches is not None:
            matches = get_close_matches(name, __all__, n=1, cutoff=0.8)
            if matches:
                hint = f" Did you mean {matches[0]!r}?"

        raise AttributeError(f"module {__name__!r} has no attribute {name!r}.{hint}")

    # Fast path: another thread may have already cached the value.
    if name in globals():
        return globals()[name]

    with _LOCK:
        if name in globals():
            return globals()[name]

        try:
            if ":" in target:
                mod_name, qual = target.split(":", 1)
                value = getattr(import_module(mod_name, __name__), qual)
            else:
                value = import_module(target, __name__)  # submodule itself
        except ImportError as e:
            rewritten = _maybe_rewrite_import_error(e)
            if rewritten is e:
                raise
            raise rewritten

        globals()[name] = value
        return value


def __dir__() -> list[str]:  # pragma: no cover - import shim
    # Expose the public surface (stable ordering for IDEs / autocomplete).
    return sorted(set(__all__) | set(_LAZY.keys()))


if TYPE_CHECKING:  # pragma: no cover - imported only for static type checkers
    from . import mlp as mlp  # noqa: F401
    from .mlp import MLP, Sine  # noqa: F401


# Tiny model registry


def _canonical(key: str) -> str:
    """Normalize registry keys.

    The normalization is case-insensitive and ignores common separators
    (underscores, dashes, spaces, punctuation). For example, all of the
    following map to the same key:

    - ``"fully_connected"``
    - ``"Fully-Connected"``
    - ``"fully connected"``
    """

    if not isinstance(key, str):  # defensive: nicer errors for config typos
        raise TypeError(f"Model name must be a str, got {type(key).__name__}.")

    canon = "".join(ch for ch in key.lower() if ch.isalnum())
    if not canon:
        raise ValueError("Model name must contain at least one alphanumeric character.")
    return canon


def _resolve_target(target: str) -> Any:
    """Resolve an import target like ``"pkg.mod:Symbol"``.

    Supports either ``pkg.mod:Symbol`` or ``pkg.mod.Symbol``. Relative module
    names (e.g., ``".mlp:MLP"``) are anchored to this package.
    """

    t = target.strip()
    if not t:
        raise ValueError("Import target must be a non-empty string.")

    if ":" in t:
        mod_name, qual = t.split(":", 1)
    else:
        if "." not in t:
            raise ValueError(
                "Import target must contain ':' (preferred) or '.' to separate module and symbol. "
                f"Got: {target!r}"
            )
        mod_name, qual = t.rsplit(".", 1)

    mod = import_module(mod_name, __name__)
    try:
        return getattr(mod, qual)
    except AttributeError as e:
        raise AttributeError(
            f"Could not resolve target {target!r}: module {mod.__name__!r} has no attribute {qual!r}."
        ) from e


@dataclass(frozen=True, slots=True)
class _ModelInfo:
    """Metadata for a *primary* registered model."""

    name: str
    target: str | None
    summary: str
    aliases: tuple[str, ...] = ()
    homepage: str | None = None
    tags: tuple[str, ...] = ()


# Canonical-key registry for fast lookup; includes aliases.
_MODEL_REGISTRY: dict[str, ModelBuilder] = {}

# Metadata for primary model names only.
_MODEL_INFO: dict[str, _ModelInfo] = {}


def register(
    name: str,
    builder: ModelBuilder,
    *,
    aliases: Iterable[str] | None = None,
    summary: str | None = None,
) -> None:
    """Register a model builder under ``name`` (and optional ``aliases``)."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("Model name must be a non-empty string.")

    if not callable(builder):
        raise TypeError("builder must be callable.")

    key = _canonical(name)

    # Primary registration.
    existing = _MODEL_REGISTRY.get(key)
    if existing is not None and existing is not builder:
        raise KeyError(f"Model {name!r} is already registered under key {key!r}.")
    _MODEL_REGISTRY[key] = builder

    # Prepare/merge metadata for the primary key.
    display_name = name.strip()
    info = _MODEL_INFO.get(key)
    if info is None:
        info = _ModelInfo(
            name=display_name,
            target=None,
            summary="User-registered model.",
            aliases=(),
        )

    # Keep the first-registered display name stable.
    display_name = info.name

    # Only overwrite the summary if the caller provided one.
    new_summary = info.summary
    if isinstance(summary, str) and summary.strip():
        new_summary = summary.strip()

    # Merge aliases (dedupe by canonical form).
    merged_aliases: dict[str, str] = {}
    for a in info.aliases:
        merged_aliases[_canonical(a)] = a

    if aliases is not None:
        for a in aliases:
            if not isinstance(a, str) or not a.strip():
                continue
            a_disp = a.strip()
            a_key = _canonical(a_disp)
            if a_key == key:
                continue

            # Alias registration must not conflict with an existing different builder.
            ex = _MODEL_REGISTRY.get(a_key)
            if ex is not None and ex is not builder:
                raise KeyError(
                    f"Alias {a_disp!r} (key {a_key!r}) is already registered for a different model."
                )
            _MODEL_REGISTRY[a_key] = builder
            merged_aliases[a_key] = a_disp

    # Update primary metadata (stable ordering).
    _MODEL_INFO[key] = _ModelInfo(
        name=display_name,
        target=info.target,
        summary=new_summary,
        aliases=tuple(sorted(merged_aliases.values(), key=str.lower)),
        homepage=info.homepage,
        tags=info.tags,
    )


def register_model(
    name: str,
    target: str,
    *,
    summary: str | None = None,
    aliases: Iterable[str] | None = None,
    homepage: str | None = None,
    tags: Iterable[str] | None = None,
) -> None:
    """Convenience wrapper to register a model by import target.

    Parameters
    name:
        Registry name.
    target:
        Import target, e.g. ``"neural_pde_stl_strel.models.mlp:MLP"``.
    """

    if not isinstance(target, str) or not target.strip():
        raise ValueError("target must be a non-empty string.")

    def _builder(*args: Any, **kwargs: Any) -> Any:
        obj = _resolve_target(target)
        return obj(*args, **kwargs)

    register(name, _builder, aliases=aliases, summary=summary)

    key = _canonical(name)
    info = _MODEL_INFO.get(key)
    if info is None:  # pragma: no cover - register() above should have created this
        info = _ModelInfo(name=name.strip(), target=None, summary="", aliases=())

    new_homepage = info.homepage
    if isinstance(homepage, str) and homepage.strip():
        new_homepage = homepage.strip()

    merged_tags: set[str] = set(info.tags)
    if tags is not None:
        for t in tags:
            if isinstance(t, str) and t.strip():
                merged_tags.add(t.strip())

    new_summary = info.summary
    if isinstance(summary, str) and summary.strip():
        new_summary = summary.strip()

    _MODEL_INFO[key] = _ModelInfo(
        name=info.name,
        target=target.strip(),
        summary=new_summary,
        aliases=info.aliases,
        homepage=new_homepage,
        tags=tuple(sorted(merged_tags, key=str.lower)),
    )


def available() -> list[str]:
    """Return all registered names (including aliases), sorted."""

    names: set[str] = set()
    for info in _MODEL_INFO.values():
        names.add(info.name)
        names.update(info.aliases)

    # Fall back to canonical keys if metadata is missing for some reason.
    if not names:
        names.update(_MODEL_REGISTRY.keys())

    return sorted(names, key=str.lower)


def get_builder(name_or_target: str) -> ModelBuilder:
    """Return a builder for a registered name or an import target."""

    if not isinstance(name_or_target, str) or not name_or_target.strip():
        raise ValueError("name_or_target must be a non-empty string.")

    # If it looks like a Python import target, treat it as such.
    if ":" in name_or_target or "." in name_or_target:

        def _builder(*args: Any, **kwargs: Any) -> Any:
            obj = _resolve_target(name_or_target)
            return obj(*args, **kwargs)

        return _builder

    key = _canonical(name_or_target)
    try:
        return _MODEL_REGISTRY[key]
    except KeyError as e:
        raise KeyError(
            f"Unknown model {name_or_target!r}. Available: {', '.join(available())}."
        ) from e


def build(name_or_target: str, *args: Any, **kwargs: Any) -> Any:
    """Instantiate a model by registry name or import target."""

    try:
        builder = get_builder(name_or_target)
        return builder(*args, **kwargs)
    except ImportError as e:
        rewritten = _maybe_rewrite_import_error(e)
        if rewritten is e:
            raise
        raise rewritten


def from_spec(spec: str | ModelSpec, /, **kwargs: Any) -> Any:
    """Build a model from a string name/target or a config mapping."""

    if isinstance(spec, str):
        return build(spec, **kwargs)

    if not isinstance(spec, Mapping):
        raise TypeError(f"spec must be a str or a Mapping, got {type(spec).__name__}.")

    data = dict(spec)

    ident = (
        data.pop("name", None)
        or data.pop("model", None)
        or data.pop("type", None)
        or data.pop("target", None)
    )
    if not isinstance(ident, str) or not ident.strip():
        raise ValueError(
            "Model spec mapping must include one of: 'name', 'model', 'type', or 'target'."
        )
    ident = ident.strip()

    args = data.pop("args", ())
    if args is None:
        args = ()
    if not isinstance(args, (list, tuple)):
        raise TypeError("spec['args'] must be a list or tuple if provided.")

    init_kwargs = data.pop("kwargs", {})
    if init_kwargs is None:
        init_kwargs = {}
    if not isinstance(init_kwargs, Mapping):
        raise TypeError("spec['kwargs'] must be a mapping/dict if provided.")

    # Remaining keys are treated as keyword args (a convenient shorthand).
    extra_kwargs: dict[str, Any] = dict(init_kwargs)
    extra_kwargs.update(data)
    extra_kwargs.update(kwargs)

    return build(ident, *args, **extra_kwargs)


def about() -> str:
    """Return a human-readable summary of the available models."""

    if not _MODEL_INFO and not _MODEL_REGISTRY:
        return "No models registered."

    lines: list[str] = ["neural_pde_stl_strel.models registry:"]

    # Sort by primary display name for readability.
    infos = sorted(_MODEL_INFO.items(), key=lambda kv: kv[1].name.lower())
    for _, info in infos:
        alias_str = f" (aliases: {', '.join(info.aliases)})" if info.aliases else ""
        target_str = f" [target: {info.target}]" if info.target else ""
        lines.append(f"  - {info.name}{alias_str}: {info.summary}{target_str}")

    if _MODEL_REGISTRY and len(_MODEL_REGISTRY) > sum(1 + len(i.aliases) for _, i in infos):
        # Defensive: if some entries are missing metadata, still mention them.
        lines.append("  (Some additional internal registry keys are present without metadata.)")

    lines.append("")
    lines.append("Use build(name, **kwargs) or from_spec(mapping) to construct models.")
    return "\n".join(lines)


# Built-ins

# Keep the default registry tiny.  Add more models here as the project grows.
register_model(
    "mlp",
    ".mlp:MLP",
    summary="Small fully-connected MLP used for PINN baselines.",
    aliases=("dense", "fully_connected", "fc"),
    tags=("pytorch", "baseline", "pinn"),
)
