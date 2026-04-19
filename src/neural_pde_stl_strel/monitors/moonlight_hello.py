# ruff: noqa: I001
"""MoonLight temporal monitoring hello.

This module is intentionally small but "real": it loads a MoonLight
script, creates a temporal monitor, evaluates it on a toy trace, and
returns a ``(time, value)`` array.

Why this exists
The project treats MoonLight (and Java) as an *optional* dependency.
This file provides a lightweight, deterministic smoke test that answers:

    "Can I run a MoonLight temporal monitor in this environment?"

Monitored specification
We monitor the STL formula (MoonLight script syntax)::

    future := eventually[0,0.4] (x < y)

over a trace where ``x(t) = sin(t)`` and ``y(t) = cos(t)``.

Output convention
MoonLight represents signals (including monitor outputs) as *step-wise*
(piecewise constant) signals by default. Depending on the backend/version,
the raw monitor output may contain only the **change points** of the
verdict rather than one row per input time sample.

For reproducibility (and for easy plotting), :func:`temporal_hello`
resamples the verdict back onto the original input time grid and
normalizes Boolean values to ``{0.0, 1.0}``.

The returned array has shape ``(N, 2)`` with columns:

1. time ``t`` (float)
2. numeric Boolean verdict ``v`` in ``{0.0, 1.0}``
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np


# Keep the script tiny and explicit. Script syntax reference:
# https://github.com/MoonLightSuite/moonlight/wiki/Script-Syntax
_SCRIPT: Final[str] = (
    "signal { real x; real y; }\n"
    "domain boolean;\n"
    "formula cmp = (x < y);\n"
    "formula future = eventually[0,0.4](cmp);\n"
)


def _import_scriptloader() -> Any:
    """Import MoonLight's ScriptLoader or raise a helpful ImportError."""
    try:
        from moonlight import ScriptLoader  # type: ignore
    except Exception as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "MoonLight is not available. Install with `pip install moonlight`\n"
            "and ensure a compatible Java runtime is on PATH (Java 21+ is required).\n"
            "If you use Conda, prefer a standard Python or `pyenv` environment."
        ) from e
    return ScriptLoader


def _monitor_with_best_effort(mon: Any, time_py: list[float], values_py: list[list[float]]) -> Any:
    """Call ``mon.monitor`` using a few common MoonLight Python signatures."""
    fn = getattr(mon, "monitor", None)
    if not callable(fn):
        raise RuntimeError("MoonLight temporal monitor does not expose a callable 'monitor' method.")

    # Modern/typical signature: monitor(time, values)
    try:
        return fn(time_py, values_py)
    except TypeError:
        pass

    # Some builds include an explicit parameter array argument (even if empty).
    for params in ([], (), None):
        try:
            return fn(time_py, values_py, params)
        except TypeError:
            continue

    # Older/alternative signature: a single matrix with time as the first column.
    packed = [[t, *row] for t, row in zip(time_py, values_py)]
    try:
        return fn(packed)
    except TypeError:
        pass

    for params in ([], (), None):
        try:
            return fn(packed, params)
        except TypeError:
            continue

    raise RuntimeError(
        "Unable to call MoonLight temporal monitor. Tried common signatures: "
        "monitor(time, values), monitor(time, values, params), monitor(matrix), monitor(matrix, params)."
    )


def _coerce_time_value_pairs(raw: Any) -> np.ndarray:
    """Coerce MoonLight output to a float ndarray of shape (K, 2).

    Accepts a few plausible shapes:
    - (K, 2): [[t0, v0], [t1, v1], ...]
    - (2, K): [[t0, t1, ...], [v0, v1, ...]]
    - tuple/list of length 2: (times, values)
    """
    if raw is None:
        raise RuntimeError("MoonLight returned None from monitor().")

    # First try the most common case: nested lists of numbers.
    try:
        arr = np.asarray(raw, dtype=float)
    except Exception:
        # Fallback: keep as object first, then handle shapes manually.
        arr = np.asarray(raw, dtype=object)

    # Handle a (times, values) tuple/list.
    if arr.ndim == 1 and isinstance(raw, (list, tuple)) and len(raw) == 2:
        t = np.asarray(raw[0], dtype=float)
        v = np.asarray(raw[1], dtype=float)
        if t.shape != v.shape:
            raise ValueError(f"MoonLight output has mismatched shapes: time{t.shape} vs value{v.shape}")
        arr = np.column_stack([t, v])

    # Handle 2xK by transposing.
    if arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] != 2:
        arr = arr.T

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected MoonLight output shape (K,2); got shape {arr.shape}")

    arr = np.asarray(arr, dtype=float)
    if arr.shape[0] == 0:
        raise ValueError("MoonLight output is empty.")
    if not np.isfinite(arr).all():
        raise ValueError("MoonLight output contains non-finite entries.")
    return arr


def _sample_piecewise_constant(raw_tv: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Sample a piecewise-constant (time,value) signal on a desired time grid."""
    raw_tv = np.asarray(raw_tv, dtype=float)
    if raw_tv.ndim != 2 or raw_tv.shape[1] != 2:
        raise ValueError(f"Expected raw_tv shape (K,2), got {raw_tv.shape}")

    t_raw = raw_tv[:, 0]
    v_raw = raw_tv[:, 1]

    # Ensure time is sorted (MoonLight should already guarantee this, but be defensive).
    order = np.argsort(t_raw, kind="mergesort")
    t_raw = t_raw[order]
    v_raw = v_raw[order]

    # Deduplicate exact-equal timestamps by keeping the *last* value.
    if t_raw.size >= 2:
        dup = np.diff(t_raw) == 0.0
        if np.any(dup):
            # Keep last occurrence per time by reversing, uniquing, reversing back.
            rev_t = t_raw[::-1]
            _, rev_first_idx = np.unique(rev_t, return_index=True)
            keep = (t_raw.size - 1) - rev_first_idx
            keep.sort()
            t_raw = t_raw[keep]
            v_raw = v_raw[keep]

    # For each desired sample time, use the value from the last raw time <= sample time.
    idx = np.searchsorted(t_raw, t_grid, side="right") - 1
    idx = np.clip(idx, 0, t_raw.size - 1)
    return v_raw[idx]


def _normalize_boolean(v: np.ndarray) -> np.ndarray:
    """Normalize MoonLight boolean-ish outputs to {0.0, 1.0}."""
    v = np.asarray(v, dtype=float)
    # Map any positive value to True, everything else to False.
    return (v > 0.0).astype(float)


def temporal_hello() -> np.ndarray:
    """Run a tiny MoonLight temporal monitor and return a (t, v) array.

    Returns
    -------
    np.ndarray
        Array of shape ``(5, 2)`` with columns ``(time, verdict)`` where
        verdict is numeric Boolean in ``{0.0, 1.0}``.

    Raises
    ------
    ImportError
        If MoonLight (or its Java bridge) is not available.
    RuntimeError
        If the MoonLight API is present but cannot be invoked.
    """
    ScriptLoader = _import_scriptloader()

    mls = ScriptLoader.loadFromText(_SCRIPT)
    # Pin Boolean semantics explicitly (safe if already set in the script).
    set_bool = getattr(mls, "setBooleanDomain", None)
    if callable(set_bool):
        try:
            set_bool()
        except Exception:
            # Some builds may not support switching domains via the Python bridge.
            pass

    mon = mls.getMonitor("future")

    # Small, deterministic trace.
    t = np.arange(0.0, 1.0, 0.2, dtype=float)
    x = np.sin(t)
    y = np.cos(t)

    # Convert to plain Python floats for pyjnius.
    time_py = [float(tt) for tt in t]
    values_py = [[float(xx), float(yy)] for xx, yy in zip(x, y)]

    raw = _monitor_with_best_effort(mon, time_py, values_py)
    raw_tv = _coerce_time_value_pairs(raw)

    v_grid = _sample_piecewise_constant(raw_tv, t)
    v_grid = _normalize_boolean(v_grid)

    return np.column_stack([t, v_grid]).astype(float, copy=False)


__all__ = ["temporal_hello"]
