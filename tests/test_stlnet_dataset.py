from __future__ import annotations

"""Unit tests for the STLnet-inspired synthetic dataset utilities.

This file intentionally stays **NumPy-only** (no torch) so the tests run fast
on CPU and can act as executable documentation.

Robust semantics used here (discrete time, min/max)
We implement the standard quantitative (robustness) semantics of bounded
Signal Temporal Logic (STL) as defined in Fainekos & Pappas (2009) and
Donzé (2013), restricted to atomic predicates under a single temporal
operator:

* Atomic predicate robustness:

  * ``v <= c``  ->  ``ρ = c - v``
  * ``v >= c``  ->  ``ρ = v - c``

* Temporal operators over an inclusive sample interval ``[start, horizon]``:

  * ``G[start,horizon] φ``  (always / □)     ->  ``ρ = min_{k in [start,horizon]} ρ_k``
  * ``F[start,horizon] φ``  (eventually / ◇) ->  ``ρ = max_{k in [start,horizon]} ρ_k``

Boolean satisfaction is ``ρ > 0`` (strict) or ``ρ >= 0`` (non-strict).

Note on boolean satisfaction
Many STL references treat boolean satisfaction as ``ρ >= 0`` (with ``ρ = 0``
being the boundary case).  In this repository, the convenience helper
``BoundedAtomicSpec.satisfied`` is intentionally stricter by default and
requires ``ρ > 0`` (strictly positive margin).  Passing ``strict=False``
recovers the conventional non-strict semantics (``ρ >= 0``).  The tests
below exercise both conventions explicitly.

Covered
-------
* ``SyntheticSTLNetDataset`` time grid, noiseless landmarks, and iteration
* Noise handling and deterministic RNG behavior (Generator / RandomState / global)
* Sliding windows and bounded robust STL semantics via ``BoundedAtomicSpec``
* The ``start`` parameter for non-zero interval lower bounds (e.g. F[start,horizon])
* Presentation helper ``to_stl()``
* Dataset registry round-trip and canonicalization

These tests are deterministic, lightweight, and network-free.
"""

import copy
import math
import pathlib
import sys
from dataclasses import FrozenInstanceError

# Import setup

# Ensure the in-repo package is importable without installing the wheel.
_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pytest

import neural_pde_stl_strel.datasets as dshub
from neural_pde_stl_strel.datasets import (
    BoundedAtomicSpec,
    DatasetInfo,
    SyntheticSTLNetDataset,
    create_dataset,
    get_dataset_cls,
    register_dataset,
)

# Tight but robust absolute tolerance for analytical landmark checks.
TOL = 1e-12


def _isclose(a: float, b: float, tol: float = TOL) -> bool:
    """Absolute-only float comparison (stable across platforms)."""
    return math.isclose(a, b, rel_tol=0.0, abs_tol=tol)


def _clean_value(t: float) -> float:
    """Noiseless reference signal used by ``SyntheticSTLNetDataset``.

    .. math::

        v(t) = 0.5(\\sin(2\\pi t) + 1)
    """
    return 0.5 * (math.sin(2.0 * math.pi * t) + 1.0)


def _manual_robustness(
    v: np.ndarray,
    *,
    temporal: str,
    op: str,
    threshold: float,
    horizon: int,
    start: int = 0,
    stride: int = 1,
) -> np.ndarray:
    """Reference implementation of the bounded STL robust semantics.

    This mirrors ``BoundedAtomicSpec`` exactly:

    - ``temporal`` in {"always", "eventually"}
    - ``op`` in {"<=", ">="}
    - ``horizon`` H and ``start`` S are measured in *samples*,
      so the sliding-window length is ``H + 1`` and the temporal
      aggregation covers columns ``[S, H]`` (inclusive).
    - ``stride`` sub-samples the set of windows.

    Returns
    -------
    np.ndarray
        Robustness per window, after applying stride.
    """
    v = np.asarray(v, dtype=float).reshape(-1)
    if int(horizon) != horizon or horizon < 0:
        raise ValueError("horizon must be a non-negative int")
    if int(start) != start or start < 0:
        raise ValueError("start must be a non-negative int")
    if start > horizon:
        raise ValueError("start must be <= horizon")
    if stride <= 0:
        raise ValueError("stride must be >= 1")

    window = int(horizon) + 1
    if window <= 0:
        raise ValueError("window must be >= 1")
    if v.size < window:
        return np.empty((0,), dtype=float)

    # Use a view-based sliding window (NumPy >= 1.20).
    wins = np.lib.stride_tricks.sliding_window_view(v, window_shape=window)[::stride]

    if op == "<=":
        r = threshold - wins
    elif op == ">=":
        r = wins - threshold
    else:
        raise ValueError("op must be '<=' or '>='")

    # Restrict to the [start, horizon] interval within each window.
    r = r[:, int(start):]

    if temporal == "always":
        return np.min(r, axis=1)
    if temporal == "eventually":
        return np.max(r, axis=1)
    raise ValueError("temporal must be 'always' or 'eventually'")


# SyntheticSTLNetDataset -- construction, indexing, time grid


@pytest.mark.parametrize("n", [0, 1, 2, 5, 33])
def test_dataset_len_types_and_vector_views(n: int) -> None:
    ds = SyntheticSTLNetDataset(length=n, noise=0.0)
    assert len(ds) == n

    # ``t`` / ``v`` are 1-D float arrays of length n.
    assert isinstance(ds.t, np.ndarray)
    assert isinstance(ds.v, np.ndarray)
    assert ds.t.shape == (n,)
    assert ds.v.shape == (n,)
    assert np.issubdtype(ds.t.dtype, np.floating)
    assert np.issubdtype(ds.v.dtype, np.floating)

    if n == 0:
        with pytest.raises(IndexError):
            _ = ds[0]
        return

    # __getitem__ returns a 2-tuple of Python floats.
    t0, v0 = ds[0]
    assert isinstance(t0, float)
    assert isinstance(v0, float)

    # Vector views match scalar indexing.
    assert np.allclose(ds.t, [ds[i][0] for i in range(n)])
    assert np.allclose(ds.v, [ds[i][1] for i in range(n)])


@pytest.mark.parametrize("n", [1, 2, 5, 33])
def test_time_grid_endpoints_monotonic_and_uniform(n: int) -> None:
    ds = SyntheticSTLNetDataset(length=n, noise=0.0)

    if n == 1:
        assert _isclose(ds.t[0], 0.0)
        return

    assert _isclose(ds.t[0], 0.0)
    assert _isclose(ds.t[-1], 1.0)

    diffs = np.diff(ds.t)
    assert np.all(diffs > 0.0)
    step = 1.0 / (n - 1)
    assert np.allclose(diffs, step, atol=TOL, rtol=0.0)


def test_sequence_indexing_semantics_and_bounds() -> None:
    n = 5
    ds = SyntheticSTLNetDataset(length=n, noise=0.0)

    assert ds[-1] == ds[n - 1]
    assert ds[-n] == ds[0]

    with pytest.raises(IndexError):
        _ = ds[n]
    with pytest.raises(IndexError):
        _ = ds[-(n + 1)]


def test_length_one_semantics() -> None:
    """With n=1 the single sample is at t=0, v(0) = 0.5."""
    ds = SyntheticSTLNetDataset(length=1, noise=0.0)
    t, v = ds[0]
    assert _isclose(t, 0.0)
    assert _isclose(v, 0.5)


def test_noiseless_quarter_point_landmarks_and_bounds_n33() -> None:
    """n=33 gives t in {0, 1/32, ..., 1}, hitting quarter-points exactly."""
    ds = SyntheticSTLNetDataset(length=33, noise=0.0)

    idxs = [0, 8, 16, 24, 32]
    expected = [0.5, 1.0, 0.5, 0.0, 0.5]
    for i, e in zip(idxs, expected):
        assert _isclose(ds.v[i], e), (i, ds.t[i], ds.v[i], e)

    assert float(ds.v.min()) >= -TOL
    assert float(ds.v.max()) <= 1.0 + TOL


@pytest.mark.parametrize("n", [2, 5, 33])
def test_noiseless_values_match_closed_form(n: int) -> None:
    ds = SyntheticSTLNetDataset(length=n, noise=0.0)
    for i in range(n):
        t, v = ds[i]
        assert _isclose(v, _clean_value(t))


def test_array_property_shape_and_consistency() -> None:
    """The ``array`` property exposes the underlying (n, 2) data view."""
    n = 10
    ds = SyntheticSTLNetDataset(length=n, noise=0.0)
    arr = ds.array
    assert arr.shape == (n, 2)
    assert np.issubdtype(arr.dtype, np.floating)
    assert np.array_equal(arr[:, 0], ds.t)
    assert np.array_equal(arr[:, 1], ds.v)


def test_iter_protocol() -> None:
    """Iterating yields the same (t, v) pairs as indexed access."""
    n = 7
    ds = SyntheticSTLNetDataset(length=n, noise=0.0)
    pairs = list(ds)
    assert len(pairs) == n
    for i, (t, v) in enumerate(pairs):
        assert isinstance(t, float)
        assert isinstance(v, float)
        assert (t, v) == ds[i]


def test_invalid_dataset_parameters_raise() -> None:
    with pytest.raises((TypeError, ValueError)):
        _ = SyntheticSTLNetDataset(length=-1, noise=0.0)
    with pytest.raises((TypeError, ValueError)):
        _ = SyntheticSTLNetDataset(length=3.7, noise=0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        _ = SyntheticSTLNetDataset(length=8, noise=-0.1)


def test_rng_type_validation() -> None:
    class BadRNG:
        pass

    with pytest.raises(TypeError):
        _ = SyntheticSTLNetDataset(length=8, noise=0.1, rng=BadRNG())  # type: ignore[arg-type]


# Noise handling and RNG reproducibility


def test_no_nan_or_inf_even_with_noise() -> None:
    ds = SyntheticSTLNetDataset(length=33, noise=0.3, rng=np.random.default_rng(7))
    assert np.isfinite(ds.v).all()


def test_rng_reproducibility_and_linear_noise_scaling_generator() -> None:
    length = 16
    noise_a = 0.1
    noise_b = 0.2

    g = np.random.default_rng(2024)
    state = copy.deepcopy(g.bit_generator.state)

    a = SyntheticSTLNetDataset(length=length, noise=noise_a, rng=g)
    # Reset a new generator to *the same* state so it draws identical eps.
    g2 = np.random.default_rng()
    g2.bit_generator.state = state
    b = SyntheticSTLNetDataset(length=length, noise=noise_b, rng=g2)

    # Same times, residuals scale linearly with the noise amplitude.
    for i in range(length):
        t, va = a[i]
        _, vb = b[i]
        clean = _clean_value(t)
        ra = va - clean
        rb = vb - clean
        if abs(ra) > 1e-15:  # avoid 0/0 on exact zeros
            assert _isclose(rb, (noise_b / noise_a) * ra, tol=1e-11)


def test_rng_reproducibility_randomstate() -> None:
    length = 10
    r1 = np.random.RandomState(12345)
    r2 = np.random.RandomState(12345)

    d1 = SyntheticSTLNetDataset(length=length, noise=0.5, rng=r1)
    d2 = SyntheticSTLNetDataset(length=length, noise=0.5, rng=r2)

    assert [d1[i] for i in range(length)] == [d2[i] for i in range(length)]


def test_global_numpy_seed_reproducibility_is_restored() -> None:
    """rng=None uses the global NumPy RNG; we restore state afterward."""
    state = np.random.get_state()
    try:
        np.random.seed(2024)
        a = SyntheticSTLNetDataset(length=12, noise=0.2)
        np.random.seed(2024)
        b = SyntheticSTLNetDataset(length=12, noise=0.2)
        assert [a[i] for i in range(len(a))] == [b[i] for i in range(len(b))]
    finally:
        np.random.set_state(state)


# Sliding windows


@pytest.mark.parametrize(
    "n,win,stride,expected_windows",
    [
        (9, 5, 2, 3),   # raw windows=5 -> keep 0,2,4
        (10, 3, 3, 3),  # raw windows=8 -> keep 0,3,6
        (5, 5, 1, 1),
        (4, 5, 1, 0),   # window longer than trace -> empty
    ],
)
def test_windows_shape_stride_and_alignment(
    n: int, win: int, stride: int, expected_windows: int,
) -> None:
    ds = SyntheticSTLNetDataset(length=n, noise=0.0)
    t_win, v_win = ds.windows(length=win, stride=stride)

    assert t_win.shape == v_win.shape == (expected_windows, win)
    assert np.issubdtype(t_win.dtype, np.floating)
    assert np.issubdtype(v_win.dtype, np.floating)

    if expected_windows:
        # The first window is always the first `win` samples.
        assert np.allclose(t_win[0], ds.t[:win])
        assert np.allclose(v_win[0], ds.v[:win])


def test_windows_invalid_parameters_raise() -> None:
    ds = SyntheticSTLNetDataset(length=8, noise=0.0)
    with pytest.raises(ValueError):
        _ = ds.windows(length=0)
    with pytest.raises(ValueError):
        _ = ds.windows(length=-1)
    with pytest.raises(ValueError):
        _ = ds.windows(length=3.7)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        _ = ds.windows(length=3, stride=0)


# BoundedAtomicSpec -- validation, frozenness, presentation


def test_bounded_atomic_spec_validation_and_frozenness() -> None:
    spec = BoundedAtomicSpec(temporal="always", op="<=", threshold=0.0, horizon=0)

    # Frozen dataclass (immutability aids configs and reproducibility).
    with pytest.raises(FrozenInstanceError):
        spec.threshold = 1.0  # type: ignore[misc]

    # Invalid temporal operator.
    with pytest.raises(ValueError):
        _ = BoundedAtomicSpec(temporal="G", op="<=", threshold=0.0, horizon=0)

    # Invalid comparison operator.
    with pytest.raises(ValueError):
        _ = BoundedAtomicSpec(temporal="always", op="<", threshold=0.0, horizon=0)

    # Negative horizon.
    with pytest.raises(ValueError):
        _ = BoundedAtomicSpec(temporal="always", op="<=", threshold=0.0, horizon=-1)


def test_bounded_atomic_spec_start_validation() -> None:
    """The ``start`` parameter must satisfy 0 <= start <= horizon."""
    # Negative start is rejected.
    with pytest.raises(ValueError):
        _ = BoundedAtomicSpec(
            temporal="always", op="<=", threshold=0.0, horizon=5, start=-1,
        )

    # start > horizon is rejected.
    with pytest.raises(ValueError):
        _ = BoundedAtomicSpec(
            temporal="always", op="<=", threshold=0.0, horizon=3, start=4,
        )

    # start == horizon is valid (single-sample temporal window).
    spec = BoundedAtomicSpec(
        temporal="always", op="<=", threshold=1.0, horizon=3, start=3,
    )
    assert spec.start == 3
    assert spec.horizon == 3


def test_bounded_atomic_spec_to_stl_rendering() -> None:
    """``to_stl()`` produces compact STL strings for reports and logs.

    Examples matching the paper notation:

    * ``G[0,4] (v <= 1.0)``
    * ``F[2,4] (temp <= 25.0)``
    """
    spec_g = BoundedAtomicSpec(
        temporal="always", op="<=", threshold=1.0, horizon=4, start=0,
    )
    assert spec_g.to_stl() == "G[0,4] (v <= 1.0)"
    assert spec_g.to_stl(signal="u_max") == "G[0,4] (u_max <= 1.0)"

    spec_f = BoundedAtomicSpec(
        temporal="eventually", op="<=", threshold=25.0, horizon=4, start=2,
    )
    assert spec_f.to_stl() == "F[2,4] (v <= 25.0)"
    assert spec_f.to_stl(signal="temp") == "F[2,4] (temp <= 25.0)"

    spec_ge = BoundedAtomicSpec(
        temporal="always", op=">=", threshold=0.0, horizon=9,
    )
    assert spec_ge.to_stl() == "G[0,9] (v >= 0.0)"


# BoundedAtomicSpec -- robustness computation


def test_bounded_atomic_spec_robustness_matches_manual_reference() -> None:
    v = np.array([0.0, 0.5, 1.0, 0.5], dtype=float)

    for temporal in ("always", "eventually"):
        for op in ("<=", ">="):
            spec = BoundedAtomicSpec(
                temporal=temporal, op=op, threshold=0.6, horizon=2,
            )
            rho = spec.robustness(v, stride=1)
            ref = _manual_robustness(
                v, temporal=temporal, op=op,
                threshold=0.6, horizon=2, stride=1,
            )
            assert rho.shape == ref.shape
            assert np.allclose(rho, ref, atol=TOL, rtol=0.0)

    # Stride should simply sub-sample the windowed robustness.
    spec2 = BoundedAtomicSpec(
        temporal="always", op="<=", threshold=0.6, horizon=1,
    )
    rho2 = spec2.robustness(v, stride=2)
    ref2 = _manual_robustness(
        v, temporal="always", op="<=", threshold=0.6, horizon=1, stride=2,
    )
    assert np.allclose(rho2, ref2, atol=TOL, rtol=0.0)


def test_bounded_atomic_spec_with_nonzero_start() -> None:
    """Test robustness when ``start > 0``, corresponding to G[S,H] or F[S,H].

    This mirrors the paper's eventual-cooling spec ``F[0.8,1](s(t) <= U_cool)``
    where the temporal interval does not begin at zero.

    Example: with v = [10, 8, 6, 4, 2], horizon=4, start=2 and op="<=",
    threshold=5, the window is [10,8,6,4,2] but the temporal aggregation
    only considers positions [2,3,4] => values [6,4,2], margins [-1,1,3].
    """
    v = np.array([10.0, 8.0, 6.0, 4.0, 2.0], dtype=float)

    # F[2,4](v <= 5): max of margins at positions 2..4 = max(-1, 1, 3) = 3.
    spec_f = BoundedAtomicSpec(
        temporal="eventually", op="<=", threshold=5.0, horizon=4, start=2,
    )
    rho_f = spec_f.robustness(v)
    ref_f = _manual_robustness(
        v, temporal="eventually", op="<=",
        threshold=5.0, horizon=4, start=2,
    )
    assert rho_f.shape == (1,)
    assert np.allclose(rho_f, ref_f, atol=TOL, rtol=0.0)
    assert _isclose(float(rho_f[0]), 3.0)

    # G[2,4](v <= 5): min of margins at positions 2..4 = min(-1, 1, 3) = -1.
    spec_g = BoundedAtomicSpec(
        temporal="always", op="<=", threshold=5.0, horizon=4, start=2,
    )
    rho_g = spec_g.robustness(v)
    ref_g = _manual_robustness(
        v, temporal="always", op="<=",
        threshold=5.0, horizon=4, start=2,
    )
    assert rho_g.shape == (1,)
    assert np.allclose(rho_g, ref_g, atol=TOL, rtol=0.0)
    assert _isclose(float(rho_g[0]), -1.0)


def test_bounded_atomic_spec_start_equals_horizon() -> None:
    """When start == horizon the temporal window contains a single sample."""
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)

    # G[3,3](v <= 3.5): each window has 4 samples; only index 3 is used.
    # Windows: [1,2,3,4] -> margin at idx 3 = 3.5-4 = -0.5
    #          [2,3,4,5] -> margin at idx 3 = 3.5-5 = -1.5
    spec = BoundedAtomicSpec(
        temporal="always", op="<=", threshold=3.5, horizon=3, start=3,
    )
    rho = spec.robustness(v)
    ref = _manual_robustness(
        v, temporal="always", op="<=",
        threshold=3.5, horizon=3, start=3,
    )
    assert rho.shape == ref.shape == (2,)
    assert np.allclose(rho, ref, atol=TOL, rtol=0.0)
    assert _isclose(float(rho[0]), -0.5)
    assert _isclose(float(rho[1]), -1.5)


def test_bounded_atomic_spec_empty_when_horizon_exceeds_trace() -> None:
    v = np.array([0.0, 1.0], dtype=float)
    spec = BoundedAtomicSpec(
        temporal="always", op="<=", threshold=1.0, horizon=10,
    )
    rho = spec.robustness(v)
    assert rho.shape == (0,)


def test_bounded_atomic_spec_horizon_zero_single_sample_windows() -> None:
    """With horizon=0 each window is a single sample (pointwise check)."""
    v = np.array([0.2, 0.5, 0.8], dtype=float)
    spec = BoundedAtomicSpec(
        temporal="always", op="<=", threshold=0.6, horizon=0,
    )
    rho = spec.robustness(v)
    ref = _manual_robustness(
        v, temporal="always", op="<=", threshold=0.6, horizon=0,
    )
    # Three single-sample windows: margins [0.4, 0.1, -0.2].
    assert rho.shape == (3,)
    assert np.allclose(rho, ref, atol=TOL, rtol=0.0)
    assert _isclose(float(rho[0]), 0.4)
    assert _isclose(float(rho[1]), 0.1)
    assert _isclose(float(rho[2]), -0.2)


def test_bounded_atomic_spec_rejects_bad_stride() -> None:
    v = np.array([0.0, 1.0, 2.0], dtype=float)
    spec = BoundedAtomicSpec(
        temporal="always", op="<=", threshold=2.0, horizon=1,
    )

    with pytest.raises(ValueError):
        _ = spec.robustness(v, stride=0)
    with pytest.raises(ValueError):
        _ = spec.robustness(v, stride=-1)


# BoundedAtomicSpec -- satisfaction (strict and non-strict)


def test_satisfied_strict_requires_positive_robustness() -> None:
    """``satisfied(strict=True)`` returns ``ρ > 0``, not ``ρ >= 0``.

    This means equality with the threshold yields ``False``.
    """
    v = np.array([0.0, 1.0], dtype=float)

    # G[0,1](v <= 1.0): margins [1.0, 0.0], min = 0.0.
    spec_eq = BoundedAtomicSpec(
        temporal="always", op="<=", threshold=1.0, horizon=1,
    )
    rho_eq = spec_eq.robustness(v)
    assert np.all(rho_eq >= -TOL)
    assert not np.any(spec_eq.satisfied(v))  # strict by default

    # Loosen the threshold slightly and satisfaction should hold.
    spec_loose = BoundedAtomicSpec(
        temporal="always", op="<=", threshold=1.0 + 1e-3, horizon=1,
    )
    assert np.all(spec_loose.satisfied(v))


def test_satisfied_nonstrict_accepts_zero_robustness() -> None:
    """``satisfied(strict=False)`` returns ``ρ >= 0``."""
    v = np.array([0.0, 1.0], dtype=float)

    # G[0,1](v <= 1.0): min margin is 0.0, so non-strict should be True.
    spec = BoundedAtomicSpec(
        temporal="always", op="<=", threshold=1.0, horizon=1,
    )
    rho = spec.robustness(v)
    assert _isclose(float(rho[0]), 0.0)

    sat_strict = spec.satisfied(v, strict=True)
    sat_nonstrict = spec.satisfied(v, strict=False)

    assert not bool(sat_strict[0])
    assert bool(sat_nonstrict[0])


# CPS-flavored temporal property examples


def test_eventually_encodes_a_simple_cooling_liveness_property() -> None:
    r"""A "liveness" example mirroring common CPS specifications.

    Informal intent:
        "If the system starts hot, it should *eventually* cool below a
        threshold."

    In STL notation:

    .. math::

        \varphi_{\text{cool}} = \mathbf{F}_{[0,4]}(\text{temp} \le 25)

    Here we use the bounded fragment implemented by ``BoundedAtomicSpec``.
    """
    # The trace cools below the threshold (strictly), so ρ > 0.
    temp = np.array([50.0, 40.0, 30.0, 24.0, 23.0], dtype=float)
    spec = BoundedAtomicSpec(
        temporal="eventually", op="<=", threshold=25.0, horizon=4,
    )
    rho = spec.robustness(temp)
    sat = spec.satisfied(temp)

    assert rho.shape == (1,)
    assert sat.shape == (1,)
    assert rho.dtype == float
    assert sat.dtype == bool
    assert rho[0] > 0.0
    assert bool(sat[0]) is True

    # If it only reaches the threshold (equality) but never goes below,
    # ρ = 0 and ``satisfied`` (strict) must be False.
    temp_eq = np.array([50.0, 40.0, 30.0, 26.0, 25.0], dtype=float)
    rho_eq = spec.robustness(temp_eq)
    sat_eq = spec.satisfied(temp_eq)

    assert rho_eq.shape == (1,)
    assert sat_eq.shape == (1,)
    assert _isclose(float(rho_eq[0]), 0.0)
    assert bool(sat_eq[0]) is False


def test_eventually_cooling_with_nonzero_start() -> None:
    r"""Eventual cooling with a later start time.

    Informal intent (matching the repository's cooling-example discussion):
        "After some initial period, the temperature should *eventually*
        drop below a threshold."

    In STL notation:

    .. math::

        \varphi = \mathbf{F}_{[3,6]}(\text{temp} \le 25)

    The temporal interval [3, 6] skips the first three samples and checks
    that at least one sample in positions 3..6 satisfies ``temp <= 25``.
    """
    # Trace: slow cooling, dips below 25 only at position 5.
    temp = np.array([80.0, 60.0, 40.0, 30.0, 28.0, 22.0, 21.0], dtype=float)
    spec = BoundedAtomicSpec(
        temporal="eventually", op="<=", threshold=25.0, horizon=6, start=3,
    )
    rho = spec.robustness(temp)
    ref = _manual_robustness(
        temp, temporal="eventually", op="<=",
        threshold=25.0, horizon=6, start=3,
    )

    assert rho.shape == (1,)
    assert np.allclose(rho, ref, atol=TOL, rtol=0.0)
    # At positions 3..6: margins = [25-30, 25-28, 25-22, 25-21] = [-5, -3, 3, 4].
    # max = 4.0 > 0, so satisfied.
    assert _isclose(float(rho[0]), 4.0)
    assert bool(spec.satisfied(temp)[0]) is True


def test_safety_bound_always_property() -> None:
    r"""A safety specification: the signal must never exceed a bound.

    In STL notation (matching the 1D diffusion case study in the README):

    .. math::

        \varphi_{\text{safe}} = \mathbf{G}_{[0,H]}(s(t) \le U_{\max})

    Robustness: :math:`\rho = \min_{t}(U_{\max} - s(t))`.
    """
    # A signal that stays below 1.0.
    s = np.array([0.1, 0.5, 0.9, 0.7, 0.3], dtype=float)
    spec = BoundedAtomicSpec(
        temporal="always", op="<=", threshold=1.0, horizon=4,
    )
    rho = spec.robustness(s)

    assert rho.shape == (1,)
    # min margin = 1.0 - 0.9 = 0.1 > 0 -> satisfied.
    assert _isclose(float(rho[0]), 0.1)
    assert bool(spec.satisfied(s)[0]) is True

    # A signal that breaches the bound.
    s_bad = np.array([0.1, 0.5, 1.05, 0.7, 0.3], dtype=float)
    rho_bad = spec.robustness(s_bad)
    # min margin = 1.0 - 1.05 = -0.05 < 0 -> violated.
    assert _isclose(float(rho_bad[0]), -0.05)
    assert bool(spec.satisfied(s_bad)[0]) is False


# Windowed robustness on dataset


def test_windowed_robustness_matches_windows_and_manual_reference() -> None:
    ds = SyntheticSTLNetDataset(length=9, noise=0.0)
    spec = BoundedAtomicSpec(
        temporal="always", op="<=", threshold=1.0, horizon=4,
    )

    win_len = int(spec.horizon) + 1
    t_win, v_win = ds.windows(length=win_len, stride=2)
    t2, v2, rho = ds.windowed_robustness(spec, stride=2)

    assert np.allclose(t2, t_win)
    assert np.allclose(v2, v_win)

    ref = _manual_robustness(
        ds.v, temporal="always", op="<=",
        threshold=1.0, horizon=4, stride=2,
    )
    assert np.allclose(rho, ref, atol=TOL, rtol=0.0)


def test_windowed_robustness_with_nonzero_start() -> None:
    """``windowed_robustness`` must respect the spec's ``start`` parameter."""
    ds = SyntheticSTLNetDataset(length=9, noise=0.0)
    spec = BoundedAtomicSpec(
        temporal="eventually", op="<=", threshold=0.5, horizon=4, start=2,
    )

    _, _, rho = ds.windowed_robustness(spec, stride=1)
    ref = _manual_robustness(
        ds.v, temporal="eventually", op="<=",
        threshold=0.5, horizon=4, start=2, stride=1,
    )
    assert rho.shape == ref.shape
    assert np.allclose(rho, ref, atol=TOL, rtol=0.0)


# Dataset registry


def test_dataset_registry_round_trip_and_canonicalization() -> None:
    """The dataset hub supports name-based lookup with case/underscore/dash
    insensitivity via a canonicalization function."""
    info = DatasetInfo(
        name="UNIT-STLNET_SYNTH",
        target=".stlnet_synthetic:SyntheticSTLNetDataset",
        summary="Unit-test registration for canonicalization checks.",
        tags=("unit-tests",),
    )

    # Snapshot/restore to avoid leaking state across the test suite.
    before = dict(dshub._REGISTRY)  # type: ignore[attr-defined]
    try:
        register_dataset(info)

        for key in ("unit_stlnet_synth", "UNIT-STLNET-SYNTH", "UnitStlNetSynth"):
            cls = get_dataset_cls(key)
            assert cls is SyntheticSTLNetDataset

            ds = create_dataset(key, length=3, noise=0.0)
            assert isinstance(ds, SyntheticSTLNetDataset)
            assert len(ds) == 3
    finally:
        dshub._REGISTRY.clear()  # type: ignore[attr-defined]
        dshub._REGISTRY.update(before)  # type: ignore[attr-defined]
