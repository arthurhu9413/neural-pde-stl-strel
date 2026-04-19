"""Monitoring backends and convenience wrappers.

This subpackage provides lightweight adapters for monitoring temporal and
spatio-temporal specifications over time series produced by physics-based ML
models (PINNs, neural ODE/PDE surrogates, differentiable MPC, etc.).

The public namespace is intentionally **import-cheap**: optional heavy
dependencies (PyTorch, RTAMT, MoonLight) are lazily imported (via :pep:`562`)
only when accessed.

Backends
--------
- ``soft``: differentiable, PyTorch-based "soft STL" operators (see
  :mod:`neural_pde_stl_strel.monitoring.stl_soft`).
- ``rtamt``: quantitative STL monitoring via RTAMT (dense or discrete time).
- ``moonlight``: STREL / spatial monitoring via MoonLight.

High-level data flow
At a conceptual level, most experiments in this repository follow the same
pipeline:

    signals / fields  ->  predicate margins  ->  STL/STREL spec  ->  monitor  ->  robustness

where robustness >= 0 indicates satisfaction (standard quantitative semantics).

This module also re-exports commonly used helpers from the backend adapters so
experiments can depend on a single import surface:

    from neural_pde_stl_strel.monitoring import stl_soft, rtamt_monitor, moonlight_helper

    from neural_pde_stl_strel.monitoring import (
        build_stl_spec, evaluate_multi, satisfied,        # RTAMT
        get_monitor, field_to_signal, monitor_graph_time_series,  # MoonLight
        STLPenalty, soft_implies, always_window, ...       # soft STL
    )
"""

from __future__ import annotations

import importlib
import importlib.metadata as _metadata
import importlib.util as _import_util
import math
from collections.abc import Iterable, Mapping
from functools import lru_cache
from typing import Any, Literal, TYPE_CHECKING

# Backend registry + dependency probing

BackendName = Literal["rtamt", "moonlight", "soft"]
TimeSeries = Iterable[float] | Iterable[tuple[float, float]]

# Internal adapter modules shipped with this repository.
_BACKENDS: dict[BackendName, str] = {
    "rtamt": "neural_pde_stl_strel.monitoring.rtamt_monitor",
    "moonlight": "neural_pde_stl_strel.monitoring.moonlight_helper",
    "soft": "neural_pde_stl_strel.monitoring.stl_soft",
}

# External dependencies required for each backend: (importable module name, dist name).
_BACKEND_DEPS: dict[BackendName, tuple[str, str]] = {
    "rtamt": ("rtamt", "rtamt"),
    "moonlight": ("moonlight", "moonlight"),
    "soft": ("torch", "torch"),
}


def _normalize_backend(name: str) -> BackendName:
    """
    Normalize a backend name (case/whitespace insensitive) and allow a few
    ergonomic aliases.

    Raises:
        KeyError: if the backend name is unknown.
    """
    key = name.strip().lower()

    # Common aliases.
    if key in {"torch", "pytorch", "stl_soft"}:
        key = "soft"

    if key not in _BACKENDS:
        raise KeyError(f"Unknown monitoring backend: {name!r}. Valid: {sorted(_BACKENDS)}")
    return key  # type: ignore[return-value]


@lru_cache(maxsize=None)
def _probe_module(module_name: str, dist_name: str) -> tuple[bool, str | None]:
    """
    Probe whether *module_name* is importable, and (if available) return a
    distribution version without importing heavy modules.

    This uses `importlib.util.find_spec` to avoid importing the dependency.
    """
    if _import_util.find_spec(module_name) is None:
        return False, None

    try:
        ver = _metadata.version(dist_name)
    except _metadata.PackageNotFoundError:
        ver = None
    return True, ver


@lru_cache(maxsize=1)
def available_backends() -> dict[str, dict[str, object]]:
    """
    Return a quick availability report for each supported backend.

    Returns:
        Mapping backend -> {"available": bool, "version": str|None}
    """
    out: dict[str, dict[str, object]] = {}
    for backend, (mod, dist) in _BACKEND_DEPS.items():
        ok, ver = _probe_module(mod, dist)
        out[backend] = {"available": ok, "version": ver}
    return out


def is_available(name: str) -> bool:
    """Return True iff the named backend is available in the current environment."""
    key = _normalize_backend(name)
    rep = available_backends()
    return bool(rep[key]["available"])


def ensure(*backends: str) -> None:
    """
    Raise an ImportError if any requested backend is unavailable.

    Notes:
        - For the ``soft`` backend, this checks for PyTorch.
        - For the ``rtamt`` backend, this checks for RTAMT.
        - For the ``moonlight`` backend, this checks for the MoonLight Python bindings.

    Args:
        backends: Backend names (case-insensitive). Supported: rtamt, moonlight, soft.

    Raises:
        KeyError: if an unknown backend name is provided.
        ImportError: if at least one requested backend is missing.
    """
    if not backends:
        return

    missing: list[BackendName] = []
    for b in backends:
        key = _normalize_backend(b)
        mod, dist = _BACKEND_DEPS[key]
        ok, _ver = _probe_module(mod, dist)
        if not ok:
            missing.append(key)

    if not missing:
        return

    missing_sorted = sorted(set(missing))

    hints = {
        "rtamt": [
            "pip install rtamt",
            "Docs: https://github.com/nickovic/rtamt",
        ],
        "moonlight": [
            "pip install moonlight",
            "Docs: https://github.com/MoonLightSuite/moonlight",
            "Note: MoonLight requires a Java runtime (see MoonLight docs).",
        ],
        "soft": [
            "pip install torch",
            "Docs: https://pytorch.org/get-started/locally/",
        ],
    }

    lines = [f"Missing monitoring backends: {', '.join(missing_sorted)}.", "Install via:"]
    for b in missing_sorted:
        for hint in hints.get(b, [f"pip install {b}"]):
            lines.append(f"  - {hint}")
    raise ImportError("\n".join(lines))


def get_backend(name: str) -> Any:
    """
    Import and return the backend adapter module for *name*.

    Args:
        name: 'rtamt', 'moonlight', or 'soft' (case-insensitive; 'torch' is alias for 'soft').

    Raises:
        KeyError: unknown backend.
        ImportError: backend dependency missing.
    """
    key = _normalize_backend(name)
    ensure(key)
    return importlib.import_module(_BACKENDS[key])


def prefer_backend(prefer: Iterable[str] = ("rtamt", "soft", "moonlight")) -> BackendName:
    """
    Pick the first available backend from a preference-ordered iterable.

    If none are available, this returns the first *known* backend in the `prefer`
    list; otherwise it falls back to 'soft'.
    """
    rep = available_backends()

    first_known: BackendName | None = None
    for cand in prefer:
        try:
            key = _normalize_backend(cand)
        except KeyError:
            continue
        if first_known is None:
            first_known = key
        if rep[key]["available"]:
            return key

    return first_known or "soft"


def about() -> str:
    """Human-readable backend availability summary."""
    rep = available_backends()

    def _fmt(key: BackendName) -> str:
        info = rep[key]
        ok = "yes" if info["available"] else "no"
        ver = info.get("version")
        if ver:
            return f"  - {key}: {ok} (v{ver})"
        return f"  - {key}: {ok}"

    lines = ["Available monitoring backends:", _fmt("soft"), _fmt("rtamt"), _fmt("moonlight")]
    return "\n".join(lines)


# Lazy imports (PEP 562) + graceful missing-backend behavior


class _MissingBackendProxy:
    """Proxy object that raises a helpful ImportError when used."""

    def __init__(self, backend: BackendName, attr: str) -> None:
        self.backend = backend
        self.attr = attr

    def _message(self) -> str:
        return (
            f"Optional monitoring backend '{self.backend}' is not available; "
            f"cannot access '{self.attr}'.\n"
            f"Hint: call neural_pde_stl_strel.monitoring.ensure('{self.backend}') for installation guidance."
        )

    def __getattr__(self, _name: str) -> Any:  # pragma: no cover
        raise ImportError(self._message())

    def __call__(self, *_args: Any, **_kwargs: Any) -> Any:  # pragma: no cover
        raise ImportError(self._message())

    def __repr__(self) -> str:  # pragma: no cover
        return f"<missing backend {self.backend!r} for {self.attr!r}>"


# Attribute -> (backend, module path)
_SUBMODULE_ATTRS: dict[str, tuple[BackendName, str]] = {
    "rtamt_monitor": ("rtamt", _BACKENDS["rtamt"]),
    "moonlight_helper": ("moonlight", _BACKENDS["moonlight"]),
    "stl_soft": ("soft", _BACKENDS["soft"]),
}

# Public re-exports: attr -> (backend, module path, object name)
_REEXPORTS: dict[str, tuple[BackendName, str, str]] = {
    # RTAMT
    "build_stl_spec": ("rtamt", _BACKENDS["rtamt"], "build_stl_spec"),
    "stl_always_upper_bound": ("rtamt", _BACKENDS["rtamt"], "stl_always_upper_bound"),
    "stl_response_within": ("rtamt", _BACKENDS["rtamt"], "stl_response_within"),
    "evaluate_series": ("rtamt", _BACKENDS["rtamt"], "evaluate_series"),
    "evaluate_multi": ("rtamt", _BACKENDS["rtamt"], "evaluate_multi"),
    "satisfied": ("rtamt", _BACKENDS["rtamt"], "satisfied"),
    # MoonLight helpers
    "load_script_from_text": ("moonlight", _BACKENDS["moonlight"], "load_script_from_text"),
    "load_script_from_file": ("moonlight", _BACKENDS["moonlight"], "load_script_from_file"),
    "set_domain": ("moonlight", _BACKENDS["moonlight"], "set_domain"),
    "get_monitor": ("moonlight", _BACKENDS["moonlight"], "get_monitor"),
    "build_grid_graph": ("moonlight", _BACKENDS["moonlight"], "build_grid_graph"),
    "field_to_signal": ("moonlight", _BACKENDS["moonlight"], "field_to_signal"),
    "as_graph_time_series": ("moonlight", _BACKENDS["moonlight"], "as_graph_time_series"),
    "monitor_graph_time_series": ("moonlight", _BACKENDS["moonlight"], "monitor_graph_time_series"),
    # Soft STL (PyTorch)
    "softmin": ("soft", _BACKENDS["soft"], "softmin"),
    "softmax": ("soft", _BACKENDS["soft"], "softmax"),
    "soft_and": ("soft", _BACKENDS["soft"], "soft_and"),
    "soft_or": ("soft", _BACKENDS["soft"], "soft_or"),
    "soft_not": ("soft", _BACKENDS["soft"], "soft_not"),
    "soft_implies": ("soft", _BACKENDS["soft"], "soft_implies"),
    "pred_leq": ("soft", _BACKENDS["soft"], "pred_leq"),
    "pred_geq": ("soft", _BACKENDS["soft"], "pred_geq"),
    "pred_abs_leq": ("soft", _BACKENDS["soft"], "pred_abs_leq"),
    "pred_linear_leq": ("soft", _BACKENDS["soft"], "pred_linear_leq"),
    "always": ("soft", _BACKENDS["soft"], "always"),
    "eventually": ("soft", _BACKENDS["soft"], "eventually"),
    "always_window": ("soft", _BACKENDS["soft"], "always_window"),
    "eventually_window": ("soft", _BACKENDS["soft"], "eventually_window"),
    "until_window": ("soft", _BACKENDS["soft"], "until_window"),
    "release_window": ("soft", _BACKENDS["soft"], "release_window"),
    "once_window": ("soft", _BACKENDS["soft"], "once_window"),
    "historically_window": ("soft", _BACKENDS["soft"], "historically_window"),
    "shift_left": ("soft", _BACKENDS["soft"], "shift_left"),
    "STLPenalty": ("soft", _BACKENDS["soft"], "STLPenalty"),
    "STLPenaltyConfig": ("soft", _BACKENDS["soft"], "STLPenaltyConfig"),
}


def __getattr__(name: str) -> Any:  # pragma: no cover
    # Lazily expose submodules
    if name in _SUBMODULE_ATTRS:
        backend, mod_path = _SUBMODULE_ATTRS[name]
        if is_available(backend):
            mod = importlib.import_module(mod_path)
            globals()[name] = mod
            return mod
        proxy = _MissingBackendProxy(backend, name)
        globals()[name] = proxy
        return proxy

    # Lazily re-export objects from backend modules
    if name in _REEXPORTS:
        backend, mod_path, obj_name = _REEXPORTS[name]
        if not is_available(backend):
            proxy = _MissingBackendProxy(backend, name)
            globals()[name] = proxy
            return proxy
        mod = importlib.import_module(mod_path)
        obj = getattr(mod, obj_name)
        globals()[name] = obj
        return obj

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(set(__all__))


# Type-checker friendliness: expose the names for static analysis without importing at runtime.
if TYPE_CHECKING:  # pragma: no cover
    from . import moonlight_helper, rtamt_monitor, stl_soft
    from .moonlight_helper import (
        as_graph_time_series,
        build_grid_graph,
        field_to_signal,
        get_monitor,
        load_script_from_file,
        load_script_from_text,
        monitor_graph_time_series,
        set_domain,
    )
    from .rtamt_monitor import (
        build_stl_spec,
        evaluate_multi,
        evaluate_series,
        satisfied,
        stl_always_upper_bound,
        stl_response_within,
    )
    from .stl_soft import (
        STLPenalty,
        STLPenaltyConfig,
        always,
        always_window,
        eventually,
        eventually_window,
        historically_window,
        once_window,
        pred_abs_leq,
        pred_geq,
        pred_leq,
        pred_linear_leq,
        release_window,
        shift_left,
        soft_and,
        soft_implies,
        soft_not,
        soft_or,
        softmax,
        softmin,
        until_window,
    )

# Convenience helpers (backend-agnostic wrappers)


def _values_from_series(series: TimeSeries) -> list[float]:
    """
    Convert a series of either values or (t, value) pairs to a list of floats.

    For the soft backend, timestamps are ignored; pass `dt` explicitly if needed.
    """
    it = iter(series)
    try:
        first = next(it)
    except StopIteration:
        return []

    # Timestamped series
    if isinstance(first, (tuple, list)) and len(first) == 2:
        out = [float(first[1])]
        for x in it:
            if not isinstance(x, (tuple, list)) or len(x) != 2:
                raise TypeError(
                    "Mixed series types: expected all elements to be (t, value) pairs."
                )
            out.append(float(x[1]))
        return out

    # Plain values
    if isinstance(first, (tuple, list)):
        raise TypeError("Expected a value series (float) or a (t, value) series.")
    out = [float(first)]
    for x in it:
        if isinstance(x, (tuple, list)):
            raise TypeError("Mixed series types: encountered (t, value) in a value series.")
        out.append(float(x))
    return out


def monitor_global_upper_bound(
    series: TimeSeries,
    *,
    var: str = "u",
    u_max: float = 1.0,
    dt: float = 1.0,
    backend: Literal["auto", "rtamt", "soft"] = "auto",
    time_semantics: Literal["dense", "discrete"] = "dense",
    temp: float = 0.1,
) -> float:
    """
    Robustness for the safety bound:

        G (u <= u_max)

    The result is a scalar robustness value (>=0 means satisfied).

    Args:
        series: Values u(t) or (t, u(t)) pairs.
        var: Variable name (RTAMT only).
        u_max: Upper bound.
        dt: Sampling period for non-timestamped series (RTAMT); used only if timestamps are not provided.
        backend: 'rtamt', 'soft', or 'auto' (prefer rtamt, fallback soft).
        time_semantics: RTAMT semantics ('dense' or 'discrete').
        temp: Softness parameter for the soft backend (smaller -> closer to min/max).

    Notes:
        The soft backend implements a smooth approximation using softmin.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")

    backend_norm = str(backend).strip().lower()
    if backend_norm == "auto":
        chosen: BackendName = prefer_backend(("rtamt", "soft"))
    else:
        chosen = _normalize_backend(backend_norm)
        if chosen not in {"rtamt", "soft"}:
            raise ValueError("backend must be one of: 'auto', 'rtamt', 'soft'")

    if chosen == "rtamt":
        rt = get_backend("rtamt")
        spec = rt.stl_always_upper_bound(var=var, u_max=u_max, time_semantics=time_semantics)
        return float(rt.evaluate_series(spec, var, series, dt=dt))

    # Soft semantics: robustness = min_t (u_max - u(t))
    ensure("soft")
    from . import stl_soft  # type: ignore  # noqa: PLC0415
    import torch  # noqa: PLC0415

    vals = _values_from_series(series)
    if not vals:
        return float("inf")

    u = torch.tensor(vals, dtype=torch.float32)
    margins = stl_soft.pred_leq(u, float(u_max))
    rob = stl_soft.always(margins, temp=float(temp))
    return float(rob.detach().cpu().item())


def monitor_response_within(
    req: TimeSeries,
    resp: TimeSeries,
    *,
    var_req: str = "req",
    var_resp: str = "resp",
    theta: float = 0.0,
    within: float = 1.0,
    dt: float = 1.0,
    backend: Literal["auto", "rtamt", "soft"] = "auto",
    time_semantics: Literal["dense", "discrete"] = "dense",
    temp: float = 0.1,
) -> float:
    r"""
    Robustness for the response property:

        G ( (req >= theta) -> F[0, within] (resp >= theta) )

    This is a common "request-response within a deadline" pattern.

    Conventions:
        - req and resp are numeric signals.
        - The predicate is `signal >= theta` (shared threshold for simplicity).
        - Robustness >= 0 indicates satisfaction.

    Args:
        req: Trigger / request signal time series (values or (t, value) pairs).
        resp: Response signal time series (values or (t, value) pairs).
        var_req: Variable name for the request signal in RTAMT.
        var_resp: Variable name for the response signal in RTAMT.
        theta: Threshold used in the predicates req>=theta and resp>=theta.
        within: Deadline; interpreted as time units for dense-time semantics, or steps for discrete-time.
        dt: Sampling period for non-timestamped series (RTAMT); used only if timestamps are not provided.
        backend: 'rtamt', 'soft', or 'auto' (prefer rtamt, fallback soft).
        time_semantics: RTAMT semantics ('dense' or 'discrete').
        temp: Softness parameter for the soft backend.

    Notes:
        The soft backend uses smooth max/min approximations and assumes a
        regularly sampled trace. If you pass timestamped traces, timestamps are
        ignored for the soft backend; ensure `dt` matches the sampling.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    if within < 0:
        raise ValueError("within must be non-negative")

    backend_norm = str(backend).strip().lower()
    if backend_norm == "auto":
        chosen: BackendName = prefer_backend(("rtamt", "soft"))
    else:
        chosen = _normalize_backend(backend_norm)
        if chosen not in {"rtamt", "soft"}:
            raise ValueError("backend must be one of: 'auto', 'rtamt', 'soft'")

    if chosen == "rtamt":
        rt = get_backend("rtamt")

        if time_semantics not in {"dense", "discrete"}:
            raise ValueError("time_semantics must be 'dense' or 'discrete'")

        # Format numeric literals defensively: integers stay integers.
        theta_text = str(int(theta)) if float(theta).is_integer() else repr(float(theta))

        if math.isinf(within):
            # Unbounded response: G(req -> F resp)
            spec_text = (
                f"always (({var_req} >= {theta_text}) -> "
                f"(eventually ({var_resp} >= {theta_text})))"
            )
        else:
            if time_semantics == "discrete" and not float(within).is_integer():
                raise ValueError("For discrete-time semantics, 'within' must be an integer number of steps.")
            tau_text = (
                str(int(within))
                if float(within).is_integer()
                else repr(float(within))
            )
            spec_text = (
                f"always (({var_req} >= {theta_text}) -> "
                f"(eventually[0:{tau_text}] ({var_resp} >= {theta_text})))"
            )

        spec = rt.build_stl_spec(
            spec_text,
            var_types={var_req: "float", var_resp: "float"},
            time_semantics=time_semantics,
        )
        return float(rt.evaluate_multi(spec, {var_req: req, var_resp: resp}, dt=dt))

    ensure("soft")
    from . import stl_soft  # type: ignore  # noqa: PLC0415
    import torch  # noqa: PLC0415

    req_vals = _values_from_series(req)
    resp_vals = _values_from_series(resp)
    if not req_vals or not resp_vals:
        return float("inf")
    if len(req_vals) != len(resp_vals):
        raise ValueError("req and resp must have the same number of samples for the soft backend.")

    if math.isinf(within):
        window = len(resp_vals)
    else:
        if time_semantics == "discrete":
            if not float(within).is_integer():
                raise ValueError("For discrete-time semantics, 'within' must be an integer number of steps.")
            window = int(within) + 1
        else:
            steps = int(math.floor(float(within) / float(dt) + 1e-12))
            window = steps + 1
        window = max(1, window)

    r = torch.tensor(req_vals, dtype=torch.float32)
    s = torch.tensor(resp_vals, dtype=torch.float32)

    # Robustness margins for predicates `signal >= theta` are (signal - theta).
    req_m = r - float(theta)
    resp_m = s - float(theta)

    fut = stl_soft.eventually_window(resp_m, window=window, temp=float(temp))
    imp = stl_soft.soft_implies(req_m, fut, temp=float(temp))
    rob = stl_soft.always(imp, temp=float(temp))
    return float(rob.detach().cpu().item())


__all__ = [
    # Backend inspection / selection
    "about",
    "available_backends",
    "ensure",
    "get_backend",
    "is_available",
    "prefer_backend",
    # Lazy submodules
    "rtamt_monitor",
    "moonlight_helper",
    "stl_soft",
    # RTAMT re-exports
    "build_stl_spec",
    "stl_always_upper_bound",
    "stl_response_within",
    "evaluate_series",
    "evaluate_multi",
    "satisfied",
    # MoonLight re-exports
    "load_script_from_text",
    "load_script_from_file",
    "set_domain",
    "get_monitor",
    "build_grid_graph",
    "field_to_signal",
    "as_graph_time_series",
    "monitor_graph_time_series",
    # Soft STL re-exports
    "softmin",
    "softmax",
    "soft_and",
    "soft_or",
    "soft_not",
    "soft_implies",
    "pred_leq",
    "pred_geq",
    "pred_abs_leq",
    "pred_linear_leq",
    "always",
    "eventually",
    "always_window",
    "eventually_window",
    "until_window",
    "release_window",
    "once_window",
    "historically_window",
    "shift_left",
    "STLPenalty",
    "STLPenaltyConfig",
    # Convenience wrappers
    "monitor_global_upper_bound",
    "monitor_response_within",
]
