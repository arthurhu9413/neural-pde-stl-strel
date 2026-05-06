"""Minimal 1-D diffusion (heat) sandbox + STL-style robustness utilities.

This module is deliberately small and dependency-free (NumPy only). It provides:

1) A tiny explicit forward-time / centered-space (FTCS) simulator for the 1-D heat equation::

       u_t = alpha * u_xx

2) Simple, STL-inspired robustness helpers for bounds predicates, including
   bounded-time and bounded-space "globally" / "eventually" operators implemented
   as sliding min/max filters (monotone deque, amortized O(n)).

Boundary handling
For toy examples we enforce a *copy* boundary after each time step::

    u[0]  := u[1]
    u[-1] := u[-2]

This is a common quick-and-dirty way to mimic homogeneous Neumann (zero-flux)
boundaries when the endpoints are treated as ghost cells.

Numerics
--------
For a uniform grid with spacing dx and time step dt, the FTCS scheme uses the
dimensionless CFL-like number::

    r = alpha * dt / dx**2

For the heat equation the scheme is conditionally stable only when r <= 1/2.
We emit a RuntimeWarning when r > 1/2.

Robustness semantics note
The sliding-window operators in this file are *trailing* windows. Concretely,
for an input signal rho_phi[i], the "globally" operator implemented here returns::

    out[i] = min_{j in [i-window+1, i]} rho_phi[j]

(and analogously with max for "eventually").

This is convenient for online / prefix monitoring and for demos. If you want the
classic *future-time* bounded STL semantics (min/max over [i+a, i+b]), you can
reverse the time axis and reuse the same primitives.

References
- Gerald W. Recktenwald, *Finite-Difference Approximations to the Heat Equation* (2004).
- A. Donzé and O. Maler, *Robust Satisfaction of Temporal Logic over Real-Valued Signals*
  (FORMATS 2010).

"""

from __future__ import annotations

import warnings
from collections import deque
from typing import Literal

import numpy as np


def _validate_length_steps(length: int, steps: int) -> None:
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")


def _validate_physical_params(*, dt: float, alpha: float, dx: float) -> None:
    """Validate scalar physical / numerical parameters.

    We keep this intentionally strict: non-finite values and non-positive dx are
    almost always programming errors in this repository's toy examples.
    """

    for name, val in (("dt", dt), ("alpha", alpha), ("dx", dx)):
        # np.isfinite supports Python floats as well as NumPy scalars.
        if not np.isfinite(val):
            raise ValueError(f"{name} must be finite, got {val!r}")

    if dx <= 0:
        raise ValueError(f"dx must be > 0, got {dx}")
    if dt < 0:
        raise ValueError(f"dt must be >= 0, got {dt}")
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")


def _as_dtype(dtype: np.dtype | type) -> np.dtype:
    """Normalize a dtype argument to a `numpy.dtype` instance."""
    try:
        return np.dtype(dtype)
    except TypeError as exc:  # pragma: no cover - defensive
        raise TypeError(f"dtype must be convertible to a numpy dtype, got {dtype!r}") from exc


def cfl_number(dt: float, alpha: float, dx: float) -> float:
    """Return the FTCS CFL number r = alpha*dt/dx^2 (for the 1-D heat equation)."""
    _validate_physical_params(dt=dt, alpha=alpha, dx=dx)
    return float(alpha) * float(dt) / (float(dx) ** 2)


def _warn_if_unstable(r: float) -> None:
    """Emit a RuntimeWarning when the FTCS stability condition is violated."""

    if r > 0.5 + 1e-15:
        warnings.warn(
            f"CFL number r={r:.3g} > 0.5; FTCS may be unstable for diffusion/heat.",
            RuntimeWarning,
            stacklevel=2,
        )


def simulate_diffusion(
    length: int,
    steps: int,
    dt: float = 0.1,
    alpha: float = 0.1,
    initial: np.ndarray | None = None,
    *,
    dx: float = 1.0,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """Simulate a 1-D diffusion/heat process using an explicit FTCS scheme.

    Parameters
    length:
        Number of spatial grid points (1-D). Must be positive.
    steps:
        Number of time steps. Must be non-negative.
    dt:
        Time step size.
    alpha:
        Diffusivity parameter (>= 0).
    initial:
        Optional initial condition, shape (length,). If None, uses a unit hotspot at x=0.
    dx:
        Spatial step size (uniform grid).
    dtype:
        Output dtype (any value accepted by ``numpy.dtype``).

    Returns
    -------
    np.ndarray
        Array of shape (steps+1, length), with u[0] equal to the initial condition.

    Notes
    -----
    The interior update is::

        u_i^{n+1} = u_i^n + r * (u_{i-1}^n - 2 u_i^n + u_{i+1}^n)

    where r = alpha*dt/dx^2.

    Boundary handling (toy):
    After each update, we copy the nearest interior value into each boundary cell::

        u[0]  := u[1]
        u[-1] := u[-2]

    This mimics a homogeneous Neumann boundary condition when endpoints are treated as
    ghost cells.
    """

    _validate_length_steps(length, steps)
    dtype_np = _as_dtype(dtype)
    r = cfl_number(dt=dt, alpha=alpha, dx=dx)
    _warn_if_unstable(r)

    u = np.zeros((steps + 1, length), dtype=dtype_np)

    if initial is None:
        u[0, 0] = dtype_np.type(1.0)
    else:
        init_arr = np.asarray(initial, dtype=dtype_np)
        if init_arr.shape != (length,):
            raise ValueError(f"initial must have shape {(length,)}, got {init_arr.shape}")
        u[0] = init_arr

    if steps == 0:
        return u

    # Degenerate spatial grids.
    if length == 1:
        u[1:] = u[0]
        return u

    if length == 2:
        # With only two cells there is no strict "interior". We therefore use a simple,
        # symmetric two-point diffusion update that preserves the mean. This behaves like
        # a Neumann (reflecting) boundary condition implemented with ghost cells.
        for n in range(steps):
            cur = u[n]
            nxt = u[n + 1]
            nxt[0] = cur[0] + (2.0 * r) * (cur[1] - cur[0])
            nxt[1] = cur[1] + (2.0 * r) * (cur[0] - cur[1])
        return u

    # Standard case: at least one interior point.
    for n in range(steps):
        cur = u[n]
        nxt = u[n + 1]

        nxt[1:-1] = cur[1:-1] + r * (cur[0:-2] - 2 * cur[1:-1] + cur[2:])

        # Copy "Neumann" boundaries (treat endpoints as ghost cells).
        nxt[0] = nxt[1]
        nxt[-1] = nxt[-2]

    return u


def simulate_diffusion_with_clipping(
    length: int,
    steps: int,
    dt: float = 0.1,
    alpha: float = 0.1,
    lower: float = -1.0,
    upper: float = 1.0,
    initial: np.ndarray | None = None,
    *,
    dx: float = 1.0,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """Simulate diffusion and hard-clip the state after each step.

    This is meant to emulate the effect of enforcing a hard safety constraint such as
    ``G[0,T] (lower <= u(x,t) <= upper)`` directly on the numerical state.

    Parameters
    lower, upper:
        Clipping bounds. Must satisfy lower <= upper.

    Returns
    -------
    np.ndarray
        Same shape as :func:`simulate_diffusion`.
    """

    _validate_length_steps(length, steps)

    if lower > upper:
        raise ValueError(f"lower must be <= upper, got lower={lower}, upper={upper}")

    # Reuse initial-condition parsing and dtype normalization from the base simulator.
    u0 = simulate_diffusion(length, 0, dt, alpha, initial, dx=dx, dtype=dtype)
    np.clip(u0[0], lower, upper, out=u0[0])

    if steps == 0:
        return u0

    r = cfl_number(dt=dt, alpha=alpha, dx=dx)
    out = np.zeros((steps + 1, length), dtype=u0.dtype)
    out[0] = u0[0]

    if length == 1:
        out[1:] = out[0]
        return out

    if length == 2:
        for n in range(steps):
            cur = out[n]
            nxt = out[n + 1]
            nxt[0] = cur[0] + (2.0 * r) * (cur[1] - cur[0])
            nxt[1] = cur[1] + (2.0 * r) * (cur[0] - cur[1])
            np.clip(nxt, lower, upper, out=nxt)
        return out

    for n in range(steps):
        cur = out[n]
        nxt = out[n + 1]

        nxt[1:-1] = cur[1:-1] + r * (cur[0:-2] - 2 * cur[1:-1] + cur[2:])

        nxt[0] = nxt[1]
        nxt[-1] = nxt[-2]

        # Enforce bounds in-place.
        np.clip(nxt, lower, upper, out=nxt)

    return out


# Robustness utilities


def compute_robustness(signal: np.ndarray, lower: float, upper: float) -> float:
    """Robustness for the predicate ``lower <= x(t) <= upper`` over a 1-D signal.

    The pointwise margin is::

        m(t) = min(x(t) - lower, upper - x(t))

    The overall robustness is ``min_t m(t)``. A positive value means the predicate holds
    with that safety margin; a negative value indicates a violation.

    Notes
    -----
    One-sided bounds are supported by passing +/- np.inf.
    """

    sig = np.asarray(signal, dtype=float)
    if sig.ndim != 1:
        raise ValueError(f"signal must be 1-D, got shape {sig.shape}")
    if sig.size == 0:
        raise ValueError("signal must be non-empty")

    margins = np.minimum(sig - lower, upper - sig)
    return float(margins.min())


def compute_spatiotemporal_robustness(field: np.ndarray, lower: float, upper: float) -> float:
    """Robustness for ``lower <= u(x,t) <= upper`` over a 2-D spatiotemporal field.

    ``field`` is interpreted as a rectangular grid of values. The robustness is the minimum
    margin over all entries.
    """

    arr = np.asarray(field, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"field must be 2-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("field must be non-empty")

    margins = np.minimum(arr - lower, upper - arr)
    return float(margins.min())


def pointwise_bounds_margin(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Return pointwise margins ``min(x-lower, upper-x)`` (vectorized)."""
    arr = np.asarray(values, dtype=float)
    return np.minimum(arr - lower, upper - arr)


def _sliding_extreme(
    x: np.ndarray,
    window: int,
    extreme: Literal["min", "max"],
) -> np.ndarray:
    """Compute a trailing sliding-window min or max over a 1-D array.

    Parameters
    x:
        Input 1-D array (interpreted as a discrete-time signal).
    window:
        Window length (>= 1).
    extreme:
        Either "min" or "max".

    Returns
    -------
    np.ndarray
        Array `out` with the same length as `x`, where::

            out[i] = extreme(x[max(0, i-window+1) : i+1])

    Notes
    -----
    This uses a standard monotone deque (amortized O(n)). NaNs are propagated in the same
    way as NumPy's min/max: if any NaN is present in the current window, the output is NaN.
    """

    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")

    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {arr.shape}")

    if extreme not in {"min", "max"}:
        raise ValueError(f"extreme must be 'min' or 'max', got {extreme!r}")

    if arr.size == 0:
        return np.empty_like(arr, dtype=float)

    is_min = extreme == "min"
    out = np.empty_like(arr, dtype=float)

    nan_mask = np.isnan(arr)
    nan_count = 0

    # Store candidates as (index, value). Maintain monotonic order:
    # - for min: values increasing
    # - for max: values decreasing
    dq: deque[tuple[int, float]] = deque()

    def _pop_while_dominated(val: float) -> None:
        if is_min:
            while dq and val <= dq[-1][1]:
                dq.pop()
        else:
            while dq and val >= dq[-1][1]:
                dq.pop()

    for i, val in enumerate(arr):
        # Update NaN count for the window [i-window+1, i]
        if nan_mask[i]:
            nan_count += 1
        if i >= window and nan_mask[i - window]:
            nan_count -= 1

        # Push current value if it is not NaN.
        if not nan_mask[i]:
            _pop_while_dominated(val)
            dq.append((i, float(val)))

        left = i - window + 1
        while dq and dq[0][0] < left:
            dq.popleft()

        if nan_count > 0:
            out[i] = np.nan
        else:
            # If nan_count == 0, the window has at least one numeric value.
            out[i] = dq[0][1]

    return out


def stl_globally_robustness(rho_phi: np.ndarray, window: int) -> np.ndarray:
    """Robustness for a trailing-window "globally" operator (min filter)."""
    return _sliding_extreme(rho_phi, window=window, extreme="min")


def stl_eventually_robustness(rho_phi: np.ndarray, window: int) -> np.ndarray:
    """Robustness for a trailing-window "eventually" operator (max filter)."""
    return _sliding_extreme(rho_phi, window=window, extreme="max")


def _sliding_extreme_along_axis_1d(
    mat: np.ndarray,
    window: int,
    axis: int,
    extreme: Literal["min", "max"],
) -> np.ndarray:
    """Apply :func:`_sliding_extreme` row-wise (axis=1) or column-wise (axis=0)."""

    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")

    arr = np.asarray(mat, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"mat must be 2-D, got shape {arr.shape}")
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    if axis == 1:
        out = np.empty_like(arr, dtype=float)
        for i in range(arr.shape[0]):
            out[i] = _sliding_extreme(arr[i], window=window, extreme=extreme)
        return out

    # axis == 0
    return _sliding_extreme_along_axis_1d(arr.T, window=window, axis=1, extreme=extreme).T


def stl_spatial_globally_robustness(rho_phi_xt: np.ndarray, window: int) -> np.ndarray:
    """Spatial "globally" (min) over a trailing spatial window.

    ``rho_phi_xt`` is interpreted as a matrix where axis 0 is time and axis 1 is space.
    The result has the same shape, applying a 1-D min filter along the spatial axis.
    """

    return _sliding_extreme_along_axis_1d(rho_phi_xt, window=window, axis=1, extreme="min")


def stl_spatial_eventually_robustness(rho_phi_xt: np.ndarray, window: int) -> np.ndarray:
    """Spatial "eventually" (max) over a trailing spatial window."""
    return _sliding_extreme_along_axis_1d(rho_phi_xt, window=window, axis=1, extreme="max")


def stl_rect_globally_bounds(
    u_xt: np.ndarray,
    lower_bounds: float,
    upper_bounds: float,
    t_window: int,
    x_window: int,
) -> np.ndarray:
    """Rectangular spatiotemporal "globally" bounds for a band predicate.

    This corresponds to a trailing-window variant of::

        G_{t in [t-t_window+1, t]} G_{x in [x-x_window+1, x]} (lower <= u(t,x) <= upper)

    Implemented as: pointwise margins -> spatial min filter -> temporal min filter.
    """

    rho = pointwise_bounds_margin(u_xt, lower_bounds, upper_bounds)
    rho_space = stl_spatial_globally_robustness(rho, window=x_window)
    rho_rect = _sliding_extreme_along_axis_1d(rho_space, window=t_window, axis=0, extreme="min")
    return rho_rect


def stl_rect_eventually_bounds(
    u_xt: np.ndarray,
    lower_bounds: float,
    upper_bounds: float,
    t_window: int,
    x_window: int,
) -> np.ndarray:
    """Rectangular spatiotemporal "eventually" bounds for a band predicate.

    This corresponds to a trailing-window variant of::

        F_{t in [t-t_window+1, t]} F_{x in [x-x_window+1, x]} (lower <= u(t,x) <= upper)

    Implemented as: pointwise margins -> spatial max filter -> temporal max filter.
    """

    rho = pointwise_bounds_margin(u_xt, lower_bounds, upper_bounds)
    rho_space = stl_spatial_eventually_robustness(rho, window=x_window)
    rho_rect = _sliding_extreme_along_axis_1d(rho_space, window=t_window, axis=0, extreme="max")
    return rho_rect


__all__ = [
    "cfl_number",
    "simulate_diffusion",
    "simulate_diffusion_with_clipping",
    "compute_robustness",
    "compute_spatiotemporal_robustness",
    "pointwise_bounds_margin",
    "stl_globally_robustness",
    "stl_eventually_robustness",
    "stl_spatial_globally_robustness",
    "stl_spatial_eventually_robustness",
    "stl_rect_globally_bounds",
    "stl_rect_eventually_bounds",
]
