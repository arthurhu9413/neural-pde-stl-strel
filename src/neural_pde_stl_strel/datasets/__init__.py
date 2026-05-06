"""neural_pde_stl_strel.datasets

Lightweight dataset hub for neural-PDE + STL/STREL experiments.

Goals
-----
- **Fast imports**: keep the package importable even in minimal environments by
  lazily importing dataset implementations only when they are accessed.
- **Simple experiments**: provide a tiny name->class registry so scripts can
  refer to datasets by a short identifier.
- **Type-checker friendly**: expose concrete symbols to IDEs/type checkers
  without paying runtime import cost.

Optional import-time toggles
These are mostly useful for CI sanity checks or IDE indexing. Set them *before*
importing this package:

- ``NEURAL_PDE_STL_STREL_EAGER_IMPORTS=1``
    Eagerly import the lazily exposed submodules/attributes.
- ``NEURAL_PDE_STL_STREL_STRICT_INIT=1``
    Validate that all lazily exposed symbols resolve correctly (implies eager
    imports).

By default, everything stays lazy.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from difflib import get_close_matches
from importlib import import_module
import os
from types import ModuleType
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

__all__ = sorted(
    {
        # Primary re-exports (lazily loaded from submodules)
        "SyntheticSTLNetDataset",
        "BoundedAtomicSpec",
        # Submodule access
        "stlnet_synthetic",
        # Registry helpers
        "DatasetInfo",
        "available_datasets",
        "create_dataset",
        "get_dataset_cls",
        "register_dataset",
    }
)


# Lazily exposed submodules.
_LAZY_MODULES: dict[str, str] = {
    "stlnet_synthetic": f"{__name__}.stlnet_synthetic",
}

# Forwarded public attributes: provide nice `from neural_pde_stl_strel.datasets import SyntheticSTLNetDataset`.
_FORWARD_ATTRS: dict[str, tuple[str, str]] = {
    "SyntheticSTLNetDataset": ("stlnet_synthetic", "SyntheticSTLNetDataset"),
    "BoundedAtomicSpec": ("stlnet_synthetic", "BoundedAtomicSpec"),
}


def _truthy_env(var: str) -> bool:  # pragma: no cover - trivial helper
    return os.getenv(var, "").strip().lower() in {"1", "true", "yes", "on"}


def _load_submodule(name: str) -> ModuleType:
    module = import_module(_LAZY_MODULES[name])
    globals()[name] = module  # cache
    return module


def _load_forward(name: str) -> Any:
    mod_name, attr = _FORWARD_ATTRS[name]
    module = globals().get(mod_name)
    if module is None:
        module = _load_submodule(mod_name)
    value = getattr(module, attr)
    globals()[name] = value  # cache
    return value


def __getattr__(name: str) -> Any:  # pragma: no cover - exercised implicitly
    # Submodule?
    if name in _LAZY_MODULES:
        return _load_submodule(name)
    # Forwarded attribute?
    if name in _FORWARD_ATTRS:
        return _load_forward(name)

    # Friendly suggestions for typos.
    candidates = list(__all__)
    near = get_close_matches(name, candidates, n=3, cutoff=0.6)
    hint = f" Did you mean {', '.join(repr(c) for c in near)}?" if near else ""
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}.{hint}")


def __dir__() -> list[str]:  # pragma: no cover - trivial
    # Expose both already-bound globals and the public API.
    return sorted(list(globals().keys()) + list(__all__))


# Optional: eager validation / IDE-friendly imports.
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

    # Then validate forwarded names actually resolve (strict mode only).
    if strict:
        missing: list[str] = []
        for _name, (_mod, _attr) in _FORWARD_ATTRS.items():
            try:
                _load_forward(_name)
            except Exception as _e:
                missing.append(f"{_name}  <-  {_mod}.{_attr}  ({_e.__class__.__name__}: {_e})")
        if missing:
            bullet = "\n  - "
            raise ImportError(
                "neural_pde_stl_strel.datasets: forward export validation failed:" + bullet + bullet.join(missing)
            )


@runtime_checkable
class TimeSeriesDataset(Protocol):
    """A minimal protocol for simple time-series datasets used in this repo."""

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> tuple[float, float]: ...


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata for a dataset that can be resolved by name.

    Attributes
    name:
        A short, stable identifier (e.g., ``"stlnet_synth"``).
    target:
        Import target of the form ``"module[:qualname]"``. Relative targets are
        resolved relative to :mod:`neural_pde_stl_strel.datasets`.
    summary:
        One-line human-readable description.
    tags:
        Optional free-form tags.
    homepage:
        Optional URL for the dataset/paper/code.
    """

    name: str
    target: str
    summary: str
    tags: tuple[str, ...] = ()
    homepage: str | None = None


# Internal registry maps a canonical key -> DatasetInfo.
_REGISTRY: MutableMapping[str, DatasetInfo] = {}


def _canonical(key: str) -> str:
    """Normalize lookup keys (case/underscore/dash/space insensitive)."""

    return "".join(ch for ch in key.lower() if ch.isalnum())


_DEFAULT_DATASETS: tuple[DatasetInfo, ...] = (
    DatasetInfo(
        name="stlnet_synth",
        target=".stlnet_synthetic:SyntheticSTLNetDataset",
        summary=(
            "Clean sinusoid on [0,1] with optional Gaussian noise; "
            "handy for STL demos and windowed robustness."
        ),
        tags=("timeseries", "synthetic", "stl", "unit-tests"),
        homepage="https://proceedings.neurips.cc/paper/2020/hash/a7da6ba0505a41b98bd85907244c4c30-Abstract.html",
    ),
)


def _ensure_defaults() -> None:
    """Idempotently populate the registry with built-in datasets."""

    for info in _DEFAULT_DATASETS:
        _REGISTRY.setdefault(_canonical(info.name), info)


def register_dataset(info: DatasetInfo) -> None:
    """Register a dataset for name-based lookup.

    Parameters
    info:
        Dataset metadata. Re-registering the same canonical name replaces the
        existing entry.
    """

    if not isinstance(info, DatasetInfo):
        raise TypeError("info must be a DatasetInfo instance")
    if not isinstance(info.name, str) or not info.name.strip():
        raise ValueError("info.name must be a non-empty string")
    if not isinstance(info.target, str) or not info.target.strip():
        raise ValueError("info.target must be a non-empty string")

    _REGISTRY[_canonical(info.name)] = info


def available_datasets() -> Mapping[str, DatasetInfo]:
    """Return a read-only mapping of *display names* to :class:`DatasetInfo`.

    Notes
    -----
    Dataset lookup is tolerant to case, whitespace, underscores, and dashes.
    For example, ``"stlnet_synth"``, ``"STLNET-SYNTH"``, and ``"stlnetsynth"``
    all refer to the same built-in dataset.
    """

    _ensure_defaults()
    # Display names should be human-friendly (not canonicalized).
    infos = sorted(_REGISTRY.values(), key=lambda d: d.name)
    return {info.name: info for info in infos}


def _resolve_target(target: str) -> Any:
    """Import an object given an import target ``'module[:qualname]'``."""

    target = target.strip()
    if not target:
        raise ValueError("target must be a non-empty string")

    mod_name, sep, qual = target.partition(":")
    module = import_module(mod_name, __name__)
    return getattr(module, qual) if sep else module


def get_dataset_cls(name_or_target: str) -> type[TimeSeriesDataset]:
    """Resolve a dataset class from a registry name or import target.

    Examples
    --------
    >>> cls = get_dataset_cls("stlnet_synth")
    >>> ds = cls(length=128, noise=0.1)
    """

    if not isinstance(name_or_target, str) or not name_or_target.strip():
        raise ValueError("name_or_target must be a non-empty string")

    _ensure_defaults()

    key = _canonical(name_or_target)
    info = _REGISTRY.get(key)

    if info is None:
        # If it doesn't look like an import target, treat it as a dataset name
        # and raise a clearer error than an ImportError from importlib.
        s = name_or_target.strip()
        if ":" not in s and "." not in s:
            candidates = sorted(available_datasets().keys())
            near = get_close_matches(s, candidates, n=3, cutoff=0.6)
            hint = f" Did you mean {', '.join(repr(c) for c in near)}?" if near else ""
            raise KeyError(f"Unknown dataset name {s!r}. Available: {', '.join(candidates)}.{hint}")
        target = s
    else:
        target = info.target

    obj = _resolve_target(target)

    if isinstance(obj, ModuleType):
        raise TypeError(f"Target {target!r} resolved to a module; expected a dataset class.")
    if not isinstance(obj, type):
        raise TypeError(f"Target {target!r} resolved to {type(obj).__name__}; expected a class.")

    return obj  # type: ignore[return-value]


def create_dataset(name_or_target: str, /, *args: Any, **kwargs: Any) -> TimeSeriesDataset:
    """Instantiate a dataset via registry name or fully-qualified import target."""

    cls = get_dataset_cls(name_or_target)
    return cls(*args, **kwargs)  # type: ignore[misc]


if TYPE_CHECKING:  # pragma: no cover
    from . import stlnet_synthetic as stlnet_synthetic  # noqa: F401
    from .stlnet_synthetic import BoundedAtomicSpec as BoundedAtomicSpec  # noqa: F401
    from .stlnet_synthetic import SyntheticSTLNetDataset as SyntheticSTLNetDataset  # noqa: F401
