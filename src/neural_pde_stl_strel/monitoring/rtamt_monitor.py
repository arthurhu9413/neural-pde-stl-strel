from __future__ import annotations

"""Version-robust, optional-dependency helpers for RTAMT (STL monitoring).

RTAMT (Real-Time Analog Monitoring Tool) is a widely used library for
quantitative monitoring of Signal Temporal Logic (STL) specifications over
discrete-time and dense-time signals.

This repository keeps RTAMT as an *optional* dependency. Importing this module
must **not** import :mod:`rtamt` eagerly; the import happens only when you build
or evaluate a specification.

What this module provides
- ``build_stl_spec``: compile an STL formula with variable declarations.
- ``stl_always_upper_bound``: convenience builder for the safety property
  ``G (u <= U)``.
- ``stl_response_within``: convenience builder for the bounded-response
  property ``G (req -> F_[0,w] resp)`` (compiled using RTAMT's ``implies`` keyword).
- ``evaluate_series`` / ``evaluate_multi``: normalize time-series inputs into
  the timestamped shapes RTAMT expects and evaluate offline robustness.
- ``satisfied``: robustness-to-boolean helper using the standard convention
  ``rho >= 0``.

API and input-shape notes
RTAMT's Python API has accumulated minor differences across releases. In
particular, :meth:`evaluate` may accept one (or more) of the following forms:

- ``evaluate(dataset_dict)`` where ``dataset_dict`` looks like
  ``{'time': [...], 'x': [...], 'y': [...]}``.
- ``evaluate(mapping)`` where ``mapping`` looks like
  ``{'x': [(t, v), ...], 'y': [(t, v), ...]}``.
- ``evaluate(['x', x_trace], ['y', y_trace], ...)`` (positional pairs), as used
  in RTAMT's README examples.
- ``evaluate(names, traces)`` (legacy).

The helpers below try these variants automatically to stay resilient.
"""

import math

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, TypeAlias


# Lazy RTAMT import

_RTAMT: Any | None = None  # cached module to avoid repeated imports


def _import_rtamt() -> Any:
    """Import and cache :mod:`rtamt` on first use.

    Raises
    ------
    ImportError
        If RTAMT is not installed (or fails to import).
    """

    global _RTAMT
    if _RTAMT is not None:
        return _RTAMT

    try:
        import rtamt  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised only when missing
        raise ImportError(
            "RTAMT is not available. Install it in a Python 3.10/3.11 environment "
            "with `pip install rtamt`\n"
            "Upstream docs: https://github.com/nickovic/rtamt"
        ) from exc

    _RTAMT = rtamt
    return rtamt


# Time-series normalization

# We accept time series inputs liberally:
# - regular samples: [v0, v1, ...] with a constant dt
# - timestamped samples: [(t0, v0), (t1, v1), ...] (also accepts list-of-lists)
TimeSeries: TypeAlias = Iterable[float] | Iterable[Sequence[float]]


def _is_nonstring_sequence(obj: object) -> bool:
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray))


def _as_time_value_pair(sample: object) -> tuple[float, float] | None:
    """Best-effort conversion of an object to a ``(time, value)`` pair."""

    if _is_nonstring_sequence(sample):
        if len(sample) < 2:
            return None
        try:
            return float(sample[0]), float(sample[1])
        except Exception:
            return None

    # Fallback: try unpacking from an arbitrary iterable.
    try:
        t, v = sample  # type: ignore[misc]
    except Exception:
        return None

    try:
        return float(t), float(v)
    except Exception:
        return None


def _normalize_series(series: TimeSeries, dt: float | None) -> list[tuple[float, float]]:
    """Normalize a time series to a sorted list of ``(t, value)`` pairs.

    Parameters
    series:
        Either regular samples ``[v0, v1, ...]`` or timestamped samples
        ``[(t0, v0), (t1, v1), ...]``.
    dt:
        Sampling period for regular samples. Ignored for timestamped inputs.
        If ``None``, defaults to 1.0.

    Notes
    -----
    - For timestamped inputs, timestamps are sorted in ascending order.
    - Exact duplicate timestamps are coalesced by keeping the *last* value.
      (This is a useful convention for piecewise-constant traces.)
    """

    it = iter(series)
    try:
        first = next(it)
    except StopIteration:
        return []

    # Timestamped form?
    first_pair = _as_time_value_pair(first)
    if first_pair is not None:
        out: list[tuple[float, float]] = [first_pair]
        for el in it:
            pair = _as_time_value_pair(el)
            if pair is None:
                raise TypeError(
                    "Timestamped series elements must be (time, value) pairs; "
                    f"got {type(el)!r}."
                )
            out.append(pair)

        # Sort and coalesce exact duplicates.
        out.sort(key=lambda tv: tv[0])
        dedup: list[tuple[float, float]] = []
        for t, v in out:
            if dedup and t == dedup[-1][0]:
                dedup[-1] = (t, v)
            else:
                dedup.append((t, v))
        return dedup

    # Regular-sampled form.
    step = 1.0 if dt is None else float(dt)
    if step <= 0.0:
        raise ValueError("'dt' must be > 0 for regularly sampled series.")

    try:
        v0 = float(first)  # type: ignore[arg-type]
    except Exception as exc:
        raise TypeError(
            "Regularly sampled series elements must be numeric scalars; "
            f"got {type(first)!r}."
        ) from exc

    out = [(0.0, v0)]
    k = 1
    for v in it:
        out.append((k * step, float(v)))
        k += 1
    return out


# Robustness output coercion


def _coerce_scalar(value: object) -> float:
    """Coerce RTAMT's return value to a plain ``float``.

    RTAMT typically returns a scalar robustness value, but older bindings or
    certain configurations may return small containers such as ``[rho]`` or
    ``[(t, rho)]``. This helper collapses those cases deterministically.
    """

    # Fast path: already numeric (float/int/NumPy scalar).
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        pass

    # Some RTAMT variants expose values via a dict (e.g., named outputs).
    if isinstance(value, Mapping):
        if not value:
            raise ValueError("RTAMT returned an empty mapping; cannot coerce robustness.")
        if "out" in value:
            return _coerce_scalar(value["out"])
        first_key = next(iter(value))
        return _coerce_scalar(value[first_key])

    # Common container forms.
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("RTAMT returned an empty sequence; cannot coerce robustness.")

        # If this looks like [(t, rho), ...], choose the rho at the earliest time.
        first = value[0]
        first_pair = _as_time_value_pair(first)
        if first_pair is not None:
            best_t, best_rho = first_pair[0], first_pair[1]
            for el in value[1:]:
                pair = _as_time_value_pair(el)
                if pair is None:
                    break
                t, rho = pair
                if t < best_t:
                    best_t, best_rho = t, rho
            return float(best_rho)

        # Otherwise treat as [rho, ...] and take the first element.
        return _coerce_scalar(first)

    # Last attempt for custom numeric-ish types.
    return float(value)  # type: ignore[arg-type]


# Specification builders


def _new_spec(time_semantics: Literal["dense", "discrete"] = "dense") -> Any:
    """Create a new RTAMT STL specification object for the requested semantics."""

    rtamt = _import_rtamt()

    if time_semantics == "dense":
        Spec = getattr(rtamt, "StlDenseTimeSpecification", None) or getattr(
            rtamt, "StlDenseTimeOfflineSpecification", None
        )
        if Spec is None:
            raise RuntimeError("Your RTAMT installation lacks dense-time STL support.")
        return Spec()

    if time_semantics == "discrete":
        Spec = getattr(rtamt, "StlDiscreteTimeSpecification", None) or getattr(
            rtamt, "StlDiscreteTimeOfflineSpecification", None
        )
        if Spec is None:
            raise RuntimeError("Your RTAMT installation lacks discrete-time STL support.")
        return Spec()

    raise ValueError("time_semantics must be either 'dense' or 'discrete'.")


def build_stl_spec(
    spec_text: str,
    *,
    var_types: Mapping[str, str] | Sequence[tuple[str, str]] = (),
    io_types: Mapping[str, str] | None = None,
    time_semantics: Literal["dense", "discrete"] = "dense",
    name: str | None = None,
) -> Any:
    """Build and parse an RTAMT STL specification.

    Parameters
    spec_text:
        The STL formula text in RTAMT syntax.
    var_types:
        Variable declarations as ``{"name": "float"}`` or ``[(name, "float"), ...]``.
    io_types:
        Optional map of variable names to IO roles (``"input"`` or ``"output"``).
        This is relevant for IA-STL; unsupported RTAMT builds ignore it.
    time_semantics:
        Either ``"dense"`` (default) or ``"discrete"``.
    name:
        Optional name to attach to the spec (useful in debugging/printing).

    Returns
    -------
    Any
        A parsed RTAMT specification object ready for :meth:`evaluate`.
    """

    spec = _new_spec(time_semantics=time_semantics)
    if name is not None:
        # Many RTAMT builds expose a `.name` attribute; if not, this is harmless.
        try:
            spec.name = str(name)
        except Exception:
            pass

    # Declarations
    decl_items = list(var_types.items()) if isinstance(var_types, Mapping) else list(var_types)
    for var_name, var_type in decl_items:
        spec.declare_var(str(var_name), str(var_type))

    # IO types (best effort)
    if io_types:
        set_io = getattr(spec, "set_var_io_type", None)
        if callable(set_io):
            for var_name, role in io_types.items():
                try:
                    set_io(str(var_name), str(role))
                except Exception:
                    # Unsupported or invalid role; skip silently.
                    pass

    spec.spec = spec_text
    spec.parse()
    return spec


def stl_always_upper_bound(
    var: str = "u",
    u_max: float = 1.0,
    *,
    time_semantics: Literal["dense", "discrete"] = "dense",
) -> Any:
    """Return a compiled spec for the safety property ``G (var <= u_max)``."""

    u_max_f = float(u_max)
    if not math.isfinite(u_max_f):
        raise ValueError("'u_max' must be a finite number.")

    spec = _new_spec(time_semantics=time_semantics)
    spec.declare_var(str(var), "float")
    spec.spec = f"always ({var} <= {u_max_f})"
    spec.parse()
    return spec


def stl_response_within(
    *,
    var_req: str = "req",
    var_resp: str = "resp",
    within: float = 1.0,
    req_threshold: float = 0.0,
    resp_threshold: float = 0.0,
    time_semantics: Literal["dense", "discrete"] = "dense",
) -> Any:
    """Return a compiled bounded-response spec.

    The returned STL formula encodes the response property:

        ``G ( (req >= req_threshold) -> F_[0,within] (resp >= resp_threshold) )``

    RTAMT's concrete syntax uses the keyword ``implies`` (rather than ``->``).

    In the main project code, ``req`` and ``resp`` are typically interpreted as
    *margins* (i.e., ``>= 0`` means the condition holds), hence the default
    thresholds at 0.
    """

    w = float(within)
    if not math.isfinite(w) or w < 0.0:
        raise ValueError("'within' must be a finite, non-negative number.")

    req_thr = float(req_threshold)
    resp_thr = float(resp_threshold)
    if not math.isfinite(req_thr) or not math.isfinite(resp_thr):
        raise ValueError("'req_threshold' and 'resp_threshold' must be finite numbers.")

    spec = _new_spec(time_semantics=time_semantics)
    spec.declare_var(str(var_req), "float")
    spec.declare_var(str(var_resp), "float")
    spec.spec = (
        f"always (({var_req} >= {req_thr}) implies "
        f"(eventually[0:{w}] ({var_resp} >= {resp_thr})))"
    )
    spec.parse()
    return spec


# Evaluation helpers


def _try_build_dataset(series_map: Mapping[str, list[tuple[float, float]]]) -> dict[str, list[float]] | None:
    """Try to convert timestamped traces into RTAMT's dataset dict format.

    Returns ``None`` if:
    - any variable is named "time" (would collide), or
    - variables have different timestamp grids.
    """

    if not series_map or "time" in series_map:
        return None

    first_name = next(iter(series_map))
    base = series_map[first_name]
    if not base:
        return None
    base_times = [t for t, _ in base]

    dataset: dict[str, list[float]] = {"time": base_times}
    dataset[first_name] = [v for _, v in base]

    for name, trace in series_map.items():
        if name == first_name:
            continue
        if [t for t, _ in trace] != base_times:
            return None
        dataset[name] = [v for _, v in trace]
    return dataset


def _evaluate_with_fallbacks(spec: Any, series_map: Mapping[str, list[tuple[float, float]]]) -> float:
    """Evaluate a spec against normalized traces, trying multiple RTAMT signatures."""

    errors: list[BaseException] = []

    # 1) Dataset dict form (works well for discrete-time and some dense-time builds).
    dataset = _try_build_dataset(series_map)
    if dataset is not None:
        try:
            return _coerce_scalar(spec.evaluate(dataset))
        except Exception as exc:
            errors.append(exc)

    # 2) Mapping form: {name: [(t, v), ...], ...}
    try:
        return _coerce_scalar(spec.evaluate(series_map))
    except Exception as exc:
        errors.append(exc)

    # 3) Positional pairs: evaluate(['x', trace], ['y', trace], ...)
    pairs = [[name, trace] for name, trace in series_map.items()]
    try:
        return _coerce_scalar(spec.evaluate(*pairs))  # type: ignore[misc]
    except Exception as exc:
        errors.append(exc)

    # 4) Legacy: evaluate(names, traces)
    names = [name for name, _ in series_map.items()]
    traces = [trace for _, trace in series_map.items()]
    try:
        return _coerce_scalar(spec.evaluate(names, traces))
    except Exception as exc:
        errors.append(exc)

    last = errors[-1] if errors else None
    msg = (
        "Failed to evaluate RTAMT spec with known signatures. Tried: "
        "evaluate(dataset_dict), evaluate(mapping), evaluate(*pairs), evaluate(names, traces)."
    )
    if last is None:
        raise TypeError(msg)
    raise TypeError(msg) from last


def evaluate_series(spec: Any, var: str, series: TimeSeries, *, dt: float = 1.0) -> float:
    """Evaluate robustness for a *single* variable time series."""

    return evaluate_multi(spec, {str(var): series}, dt=float(dt))


def evaluate_multi(
    spec: Any,
    data: Mapping[str, TimeSeries] | Sequence[tuple[str, TimeSeries]],
    *,
    dt: float | Mapping[str, float] = 1.0,
) -> float:
    """Evaluate robustness for a multi-signal trace.

    Parameters
    spec:
        Parsed RTAMT specification.
    data:
        Either ``{name: series, ...}`` or ``[(name, series), ...]``.
    dt:
        Global sampling period for regular-sampled inputs, or a per-variable
        mapping ``{name: dt}``.
    """

    items = list(data.items()) if isinstance(data, Mapping) else list(data)

    if isinstance(dt, Mapping):
        dt_map: Mapping[str, float] = dt
        default_dt: float | None = None
    else:
        dt_map = {}
        default_dt = float(dt)

    series_map: dict[str, list[tuple[float, float]]] = {}
    for name, series in items:
        name_str = str(name)
        this_dt = dt_map.get(name_str, default_dt)
        norm = _normalize_series(series, this_dt)
        if not norm:
            raise ValueError(f"Time series for variable {name_str!r} is empty.")
        series_map[name_str] = norm

    return _evaluate_with_fallbacks(spec, series_map)


def satisfied(robustness: float) -> bool:
    """Return ``True`` iff robustness is non-negative (the standard STL convention)."""

    return float(robustness) >= 0.0


__all__ = [
    "build_stl_spec",
    "stl_always_upper_bound",
    "stl_response_within",
    "evaluate_series",
    "evaluate_multi",
    "satisfied",
]
