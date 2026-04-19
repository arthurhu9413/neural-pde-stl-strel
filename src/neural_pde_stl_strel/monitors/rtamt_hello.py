from __future__ import annotations

"""RTAMT smoke test: minimal offline STL monitoring.

This module intentionally avoids importing :mod:`rtamt` at import time. The
public helper :func:`stl_hello_offline` performs a tiny offline monitoring run
to confirm the optional RTAMT dependency is installed and functional.

Specification
    ``phi := always (u <= 1.0)``

Trace (discrete time, Δt = 1)
    ``u(0)=0.2, u(1)=0.4, u(2)=1.1``

Expected robustness
    Under standard robust STL semantics, the predicate ``u <= 1.0`` has
    robustness ``1.0 - u`` and the ``always`` operator takes the minimum over
    time. Therefore ``rho(phi, u, 0) = min_t (1.0 - u(t)) = -0.1``.

Why so defensive?
    RTAMT's :meth:`~rtamt.StlDenseTimeSpecification.evaluate` call signature and
    return shape have varied slightly across releases. This example tries a few
    common call conventions and normalizes the result to a scalar robustness.
"""

import math
from collections.abc import Iterable, Mapping
from typing import Any

HELLO_VAR: str = "u"
"""Name of the monitored signal."""

HELLO_SPEC: str = "always (u <= 1.0)"
"""STL specification in RTAMT syntax."""

HELLO_TRACE: list[tuple[int, float]] = [(0, 0.2), (1, 0.4), (2, 1.1)]
"""Timestamped signal used by :func:`stl_hello_offline`."""

def _as_float(x: Any) -> float | None:
    """Best-effort cast to :class:`float`.

    Returns ``None`` if the value cannot be converted or is NaN.
    """

    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v):
        return None
    return v


def _reduce_min(values: Iterable[Any]) -> float | None:
    """Extract the minimum numeric value from an iterable.

    RTAMT sometimes returns robustness as a time series of pairs
    ``(t, rho(t))``. This helper accepts either plain numbers or pairs.
    """

    best = math.inf
    found = False

    for item in values:
        candidate: Any = item
        if isinstance(item, (list, tuple)) and item:
            # Common RTAMT shape: (time, value)
            candidate = item[1] if len(item) > 1 else item[0]

        v = _as_float(candidate)
        if v is None:
            continue

        found = True
        if v < best:
            best = v

    return best if found else None


def _coerce_robustness(value: Any) -> float:
    """Coerce RTAMT's output to a scalar robustness.

    For this particular spec (an ``always`` upper bound), the trace-level
    robustness is the minimum margin over the time horizon. If a specific RTAMT
    build returns a robustness time series instead of a single scalar, we take
    the minimum value.
    """

    scalar = _as_float(value)
    if scalar is not None:
        return scalar

    if isinstance(value, Mapping):
        # Prefer common output keys when present.
        for key in ("out", "rob", "rho", "robustness"):
            if key in value:
                try:
                    return _coerce_robustness(value[key])
                except Exception:
                    pass

        reduced = _reduce_min(value.values())
        if reduced is not None:
            return reduced

        # Last resort: recurse into values and take the minimum.
        candidates: list[float] = []
        for v in value.values():
            try:
                candidates.append(_coerce_robustness(v))
            except Exception:
                continue
        if candidates:
            return min(candidates)

        raise TypeError("Could not coerce RTAMT mapping output to a float.")

    if isinstance(value, (list, tuple)):
        reduced = _reduce_min(value)
        if reduced is not None:
            return reduced

        # Last resort: recurse into elements and take the minimum.
        candidates = []
        for item in value:
            try:
                candidates.append(_coerce_robustness(item))
            except Exception:
                continue
        if candidates:
            return min(candidates)

    raise TypeError(f"Unsupported RTAMT return type: {type(value)!r}")

def stl_hello_offline() -> float:
    """Evaluate the hello STL spec in RTAMT and return its robustness.

    Returns
    -------
    float
        The offline robustness for :data:`HELLO_TRACE`. Negative values mean
        the specification is violated; positive values mean it is satisfied.

    Raises
    ------
    ImportError
        If RTAMT is not installed.
    RuntimeError
        If no supported RTAMT ``evaluate`` calling convention works.
    """

    # Optional dependency: import only when invoked.
    try:
        import rtamt  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "rtamt is not installed. Install it with 'pip install rtamt' "
            "or see https://github.com/nickovic/rtamt for build notes."
        ) from e

    # Prefer discrete-time (simple and deterministic), but fall back gracefully.
    spec_cls = getattr(rtamt, "StlDiscreteTimeSpecification", None)
    if spec_cls is None:
        spec_cls = getattr(rtamt, "StlDenseTimeSpecification", None) or getattr(
            rtamt, "StlDenseTimeOfflineSpecification", None
        )
    if spec_cls is None:  # pragma: no cover
        raise RuntimeError("Could not find an STL specification class in rtamt.")

    spec = spec_cls()
    spec.declare_var(HELLO_VAR, "float")
    spec.spec = HELLO_SPEC
    spec.parse()

    # Best-effort: make the discrete-time sampling assumptions explicit.
    set_sp = getattr(spec, "set_sampling_period", None)
    if callable(set_sp):
        try:
            set_sp(1, "s", 0.0)
        except Exception:
            pass

    # Evaluate using the README-documented signature first, then try a few
    # minor variants observed across releases.
    errors: list[Exception] = []
    for call in (
        lambda: spec.evaluate([HELLO_VAR, HELLO_TRACE]),
        lambda: spec.evaluate([[HELLO_VAR, HELLO_TRACE]]),
        lambda: spec.evaluate([HELLO_VAR], [HELLO_TRACE]),
        lambda: spec.evaluate({HELLO_VAR: HELLO_TRACE}),
    ):
        try:
            return _coerce_robustness(call())
        except Exception as e:
            errors.append(e)

    raise RuntimeError(
        "rtamt.evaluate() failed for all supported call signatures."
    ) from errors[-1]


if __name__ == "__main__":  # pragma: no cover
    print(stl_hello_offline())


__all__ = ["stl_hello_offline"]
