# src/neural_pde_stl_strel/datasets/stlnet_synthetic.py
r"""neural_pde_stl_strel.datasets.stlnet_synthetic

This module provides:

* :class:`SyntheticSTLNetDataset` - a compact 1-D synthetic trace
  ``(t, v(t))`` useful for unit tests and small STL demos.
* :class:`BoundedAtomicSpec` - a minimal, NumPy-only robustness evaluator for a
  bounded fragment of Signal Temporal Logic (STL) over scalar signals.

Design goals
- **Deterministic** (within a fixed NumPy build): if the user seeds NumPy's
  global RNG (``np.random.seed(seed)``) or passes an explicit RNG object,
  two instances created with the same parameters produce identical samples.
- **Stable tests**: noisy values are rounded to 15 decimal places to reduce
  spurious 1-ULP drift in floating-point arithmetic across platforms.
- **Tiny and fast**: everything is vectorized, NumPy-only, and CPU-friendly.
- **Self-contained STL hooks**: unit tests and toy demos do not depend on RTAMT.

The synthetic trace
The dataset is the normalized sinusoid

.. math::

    v(t) = 0.5(\sin(2\pi t) + 1) + \sigma\,\varepsilon(t)

where ``t`` is a uniform grid on ``[0, 1]`` of length ``n`` and
``\varepsilon(t) ~ \mathcal{N}(0, 1)`` i.i.d. With ``noise=0`` and ``n=33``, the
signal hits the expected quarter-point landmarks:

``[0.5, 1.0, 0.5, 0.0, 0.5]`` at indices ``[0, 8, 16, 24, 32]``.

The STL fragment
We implement a very small bounded fragment that matches the repository tests:

* Atomic predicates:

  * ``v <= c``  ->  robustness margin ``ρ = c - v``
  * ``v >= c``  ->  robustness margin ``ρ = v - c``

* Temporal operators over an **inclusive** sample interval ``[start, horizon]``:

  * ``G[start,horizon] φ`` (always)     ->  ``ρ = min ρ_k``
  * ``F[start,horizon] φ`` (eventually) ->  ``ρ = max ρ_k``

Boolean satisfaction is derived from robustness. By default we use a **strict**
convention (``ρ > 0``). This is convenient in many ML settings because it turns
``v <= c`` into a strict bound ``v < c`` when interpreted as a hard constraint.
For the conventional non-strict semantics (``ρ >= 0``), pass ``strict=False``
when calling :meth:`BoundedAtomicSpec.satisfied`.

Public API
- :class:`SyntheticSTLNetDataset`
- :class:`BoundedAtomicSpec`
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Helpers

def _is_int_like(x: object) -> bool:
    """Return True for real integer scalars (reject bool)."""
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


def _sliding_window(x: np.ndarray, window: int) -> np.ndarray:
    """Return overlapping 1-D windows.

    Parameters
    x:
        1-D NumPy array.
    window:
        Positive window size.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(n - window + 1, window)`` when ``window <= n``.
        If ``window > n``, returns an empty array of shape ``(0, window)``.

    Notes
    -----
    Uses :func:`numpy.lib.stride_tricks.sliding_window_view` when available
    (NumPy >= 1.20). On older NumPy versions, falls back to a safe
    advanced-indexing implementation.
    """
    if not _is_int_like(window):
        raise TypeError(f"window must be an integer; got {type(window).__name__}")
    window_i = int(window)
    if window_i <= 0:
        raise ValueError(f"window must be positive; got {window_i}")

    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D; got shape {x.shape}")

    n = int(x.shape[0])
    if window_i > n:
        return np.empty((0, window_i), dtype=x.dtype)

    # Prefer a view-based implementation when available.
    try:  # pragma: no cover - version-dependent import
        from numpy.lib.stride_tricks import sliding_window_view  # type: ignore
    except ImportError:  # pragma: no cover - NumPy < 1.20
        sliding_window_view = None  # type: ignore[assignment]

    if sliding_window_view is not None:  # pragma: no cover - exercised on modern NumPy
        return sliding_window_view(x, window_shape=window_i)

    # Fallback: advanced indexing (allocates, but our traces are tiny).
    starts = np.arange(0, n - window_i + 1, dtype=int)
    idx = starts[:, None] + np.arange(window_i, dtype=int)[None, :]
    return x[idx]


# Minimal bounded STL robustness

@dataclass(frozen=True)
class BoundedAtomicSpec:
    """Bounded robustness for one atomic predicate under ``G`` / ``F``.

    This is intentionally a *small* subset of STL for tests and demos.

    Formula forms
    For a scalar signal ``v`` and constant ``c``:

    * ``G[start,horizon] (v <= c)``
    * ``G[start,horizon] (v >= c)``
    * ``F[start,horizon] (v <= c)``
    * ``F[start,horizon] (v >= c)``

    where ``start`` and ``horizon`` are measured in **samples** and the time
    interval is inclusive.

    Attributes
    temporal:
        Either ``"always"`` (□ / :math:`G`) or ``"eventually"`` (◇ / :math:`F`).
    op:
        Either ``"<="`` or ``">="``.
    threshold:
        The scalar threshold ``c``.
    horizon:
        Upper bound of the interval, in samples (``>= 0``).
    start:
        Lower bound of the interval, in samples (``>= 0`` and ``<= horizon``).

    Notes
    -----
    The robustness margin of the atomic predicate is:

    * ``v <= c``  ->  ``c - v``
    * ``v >= c``  ->  ``v - c``

    Then ``G`` takes a minimum and ``F`` takes a maximum over the interval.
    """

    temporal: str  # "always" | "eventually"
    op: str  # "<=" | ">="
    threshold: float
    horizon: int = 0
    start: int = 0

    def __post_init__(self) -> None:
        if self.temporal not in {"always", "eventually"}:
            raise ValueError(
                f"temporal must be 'always' or 'eventually'; got {self.temporal!r}"
            )
        if self.op not in {"<=", ">="}:
            raise ValueError(f"op must be '<=' or '>='; got {self.op!r}")

        # Horizon validation. (Reject bool and non-integers.)
        if not _is_int_like(self.horizon) or int(self.horizon) < 0:
            raise ValueError(
                f"horizon must be a non-negative integer; got {self.horizon!r}"
            )

        # Start validation.
        if not _is_int_like(self.start) or int(self.start) < 0:
            raise ValueError(
                f"start must be a non-negative integer; got {self.start!r}"
            )

        if int(self.start) > int(self.horizon):
            raise ValueError(
                f"Require start <= horizon; got start={self.start!r}, horizon={self.horizon!r}"
            )

    # Presentation helpers (useful for reports / logs)

    def to_stl(self, *, signal: str = "v") -> str:
        """Render the spec as a compact STL string."""
        op = self.op
        temporal = "G" if self.temporal == "always" else "F"
        lo = int(self.start)
        hi = int(self.horizon)
        return f"{temporal}[{lo},{hi}] ({signal} {op} {self.threshold})"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.to_stl()

    # Robust semantics

    def _robustness_on_windows(self, wins: np.ndarray) -> np.ndarray:
        """Compute robustness given pre-built windows.

        Parameters
        wins:
            Array of shape ``(num_windows, horizon+1)``.

        Returns
        -------
        np.ndarray
            Robustness per window.
        """
        if wins.size == 0:
            return np.empty((0,), dtype=float)
        if wins.ndim != 2:
            raise ValueError(f"wins must be 2-D; got shape {wins.shape}")

        wins_f = np.asarray(wins, dtype=float)
        expected_cols = int(self.horizon) + 1
        if wins_f.shape[1] != expected_cols:
            raise ValueError(
                f"wins must have shape (num_windows, {expected_cols}); got {wins_f.shape}"
            )

        # Atomic margins.
        if self.op == "<=":
            r = float(self.threshold) - wins_f
        else:  # ">="
            r = wins_f - float(self.threshold)

        # Restrict to the requested interval within each window.
        lo = int(self.start)
        r_int = r[:, lo:]

        # Temporal reduction.
        if self.temporal == "always":
            return np.min(r_int, axis=1)
        return np.max(r_int, axis=1)

    def robustness(self, v: np.ndarray, stride: int = 1) -> np.ndarray:
        """Evaluate robust semantics on a discrete-time scalar signal.

        Parameters
        v:
            1-D array of samples ``v[0], ..., v[n-1]``.
        stride:
            Keep one robustness value every ``stride`` windows (must be ``>= 1``).

        Returns
        -------
        np.ndarray
            Robustness value per window, after applying ``stride``.
        """
        if not _is_int_like(stride):
            raise TypeError(f"stride must be an integer; got {type(stride).__name__}")
        stride_i = int(stride)
        if stride_i <= 0:
            raise ValueError("stride must be a positive integer")

        v_arr = np.asarray(v, dtype=float)
        if v_arr.ndim != 1:
            raise ValueError(f"v must be 1-D (scalar signal); got shape {v_arr.shape}")

        window = int(self.horizon) + 1
        wins = _sliding_window(v_arr, window)
        wins = wins[::stride_i, :]
        return self._robustness_on_windows(wins)

    def satisfied(self, v: np.ndarray, stride: int = 1, *, strict: bool = True) -> np.ndarray:
        """Boolean satisfaction derived from robustness.

        Parameters
        v:
            1-D array of scalar samples.
        stride:
            Keep one satisfaction value every ``stride`` windows.
        strict:
            If ``True`` (default), return ``ρ > 0``. If ``False``, return
            ``ρ >= 0``.
        """
        rho = self.robustness(v, stride=stride)
        return (rho > 0.0) if strict else (rho >= 0.0)


# Synthetic dataset

class SyntheticSTLNetDataset:
    """A compact, NumPy-only synthetic trace ``(t, v(t))``.

    Parameters
    length:
        Number of samples ``n`` (``>= 0``). When ``n == 1`` the single time
        stamp is exactly ``0.0``.
    noise:
        Standard deviation ``σ`` of i.i.d. Gaussian noise. Must be non-negative.
        Default is ``0.05``.
    rng:
        Optional NumPy RNG (``Generator``-like or ``RandomState``-like). When
        ``None`` (default), draws come from NumPy's global RNG (and therefore
        respect ``np.random.seed``).

    Notes
    -----
    * The dataset is intentionally light-weight; it is *not* a full STLnet
      reproduction and is primarily used as executable documentation in tests.
    * ``np.round(..., decimals=15)`` is applied to the noisy trace (when
      ``noise > 0``) to reduce platform-dependent 1-ULP differences.
    """

    __slots__ = ("_data",)

    def __init__(self, length: int = 100, noise: float = 0.05, rng: object | None = None) -> None:
        if not _is_int_like(length):
            raise TypeError(f"length must be an integer; got {type(length).__name__}")
        n = int(length)
        if n < 0:
            raise ValueError(f"length must be non-negative; got {n}")

        noise_f = float(noise)
        if noise_f < 0.0:
            raise ValueError(f"noise must be non-negative; got {noise_f}")

        # Time grid (uniform). Keep exact endpoints for n>1.
        if n == 0:
            t = np.empty((0,), dtype=float)
        elif n == 1:
            t = np.array([0.0], dtype=float)
        else:
            t = np.linspace(0.0, 1.0, num=n, dtype=float)

        clean = 0.5 * (np.sin(2.0 * np.pi * t) + 1.0)

        if n == 0 or noise_f == 0.0:
            v = clean
        else:
            # Draw noise without disturbing the global RNG unless rng is None.
            if rng is None:
                eps = np.random.randn(n)  # respects external np.random.seed
            else:
                # Support both Generator.standard_normal and RandomState.randn.
                if hasattr(rng, "standard_normal"):
                    eps = rng.standard_normal(n)  # type: ignore[attr-defined]
                elif hasattr(rng, "randn"):
                    eps = rng.randn(n)  # type: ignore[attr-defined]
                else:
                    raise TypeError(
                        "rng must be a NumPy Generator/RandomState (or compatible)"
                    )

            v = clean + noise_f * np.asarray(eps, dtype=float)
            v = np.round(v, decimals=15)

        self._data = np.stack((t, v), axis=1) if n > 0 else np.empty((0, 2), dtype=float)

    # Sequence protocol

    def __len__(self) -> int:  # pragma: no cover - trivial
        return int(self._data.shape[0])

    def __getitem__(self, idx: int) -> tuple[float, float]:
        # NumPy handles negative indices and bounds checks.
        t, v = self._data[idx]
        return float(t), float(v)

    def __iter__(self):  # pragma: no cover - trivial
        for i in range(len(self)):
            yield self[i]

    # Convenient accessors

    @property
    def array(self) -> np.ndarray:
        """Underlying ``(n, 2)`` array **view** with columns ``[t, v]``."""
        return self._data

    @property
    def t(self) -> np.ndarray:
        """Time stamps as a **view** (1-D float array)."""
        return self._data[:, 0]

    @property
    def v(self) -> np.ndarray:
        """Values as a **view** (1-D float array)."""
        return self._data[:, 1]

    # Window utilities

    def windows(self, length: int, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Return overlapping windows of time and value.

        Parameters
        length:
            Window length (positive integer).
        stride:
            Keep one out of every ``stride`` windows (positive integer).

        Returns
        -------
        (t_win, v_win):
            Two arrays of shape ``(num_windows, length)``.
        """
        if not _is_int_like(length) or int(length) <= 0:
            raise ValueError(f"length must be a positive integer; got {length!r}")
        if not _is_int_like(stride) or int(stride) <= 0:
            raise ValueError(f"stride must be a positive integer; got {stride!r}")

        win = int(length)
        step = int(stride)

        t_win = _sliding_window(self.t, win)
        v_win = _sliding_window(self.v, win)
        return t_win[::step, :], v_win[::step, :]

    def windowed_robustness(
        self,
        spec: BoundedAtomicSpec,
        stride: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return windows paired with robustness values.

        Returns
        -------
        (t_win, v_win, rho):
            Time/value windows and the robustness of ``spec`` on each window.
        """
        win_len = int(spec.horizon) + 1
        t_win, v_win = self.windows(win_len, stride=stride)
        rho = spec._robustness_on_windows(v_win)
        return t_win, v_win, rho


__all__ = ["SyntheticSTLNetDataset", "BoundedAtomicSpec"]
