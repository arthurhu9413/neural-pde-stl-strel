# tests/test_pde_robustness.py
"""Tests for STL-style robustness helpers in :mod:`neural_pde_stl_strel.pde_example`.

This repository uses *quantitative semantics* ("robustness") for simple band
predicates of the form:

    lower ≤ x ≤ upper

The pointwise robustness margin is:

    ρ(x) = min(x - lower, upper - x)

Positive values mean the predicate is satisfied with slack; negative values mean
violation magnitude. For a whole 1-D signal, the scalar robustness is the minimum
margin over time. For a spatio-temporal field (time × space), it is the minimum
over all samples.

The module also includes tiny sliding-window helpers that act like on-line
implementations of temporal/spatial "Globally" (G) and "Eventually" (F):

* G = trailing window min
* F = trailing window max

Note: the repo demo docs call:

    pytest -q tests/test_pde_robustness.py::test_robustness_monotone_in_bounds

Keep that test CPU-friendly and fast.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest


# Support running tests from a source checkout without an editable install.
try:  # pragma: no cover - import convenience
    from neural_pde_stl_strel import pde_example as pe  # type: ignore
except Exception:  # pragma: no cover - import convenience
    ROOT = pathlib.Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if SRC.exists():
        sys.path.insert(0, str(SRC))
    from neural_pde_stl_strel import pde_example as pe  # type: ignore  # noqa: E402


ATOL = 1e-12


def _scalar_band_robustness_loop(signal: np.ndarray, lower: float, upper: float) -> float:
    """Literal definition: min_i min(x_i - lower, upper - x_i)."""

    sig = np.asarray(signal, dtype=float)
    if sig.ndim != 1:
        raise ValueError("signal must be 1-D")
    if sig.size == 0:
        raise ValueError("signal must be non-empty")
    return float(min(min(float(x) - lower, upper - float(x)) for x in sig))


def _trailing_window_extreme_1d(x: np.ndarray, window: int, *, extreme: str) -> np.ndarray:
    """Naive trailing-window min/max reference for 1-D arrays."""

    if window <= 0:
        raise ValueError("window must be positive")
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("x must be 1-D")

    out = np.empty_like(arr, dtype=float)
    for i in range(arr.size):
        j0 = max(0, i - window + 1)
        sl = arr[j0 : i + 1]
        out[i] = sl.min() if extreme == "min" else sl.max()
    return out


def _trailing_window_extreme_axis1(
    mat: np.ndarray, window: int, *, extreme: str
) -> np.ndarray:
    """Naive trailing-window min/max along axis=1 for 2-D arrays."""

    arr = np.asarray(mat, dtype=float)
    if arr.ndim != 2:
        raise ValueError("mat must be 2-D")
    out = np.empty_like(arr, dtype=float)
    for r in range(arr.shape[0]):
        out[r] = _trailing_window_extreme_1d(arr[r], window, extreme=extreme)
    return out


def _rect_window_extreme_bounds(
    u_xt: np.ndarray,
    lower: float,
    upper: float,
    *,
    t_window: int,
    x_window: int,
    extreme: str,
) -> np.ndarray:
    """Naive reference for rectangular trailing-window robustness over bounds."""

    if t_window <= 0 or x_window <= 0:
        raise ValueError("window sizes must be positive")

    u = np.asarray(u_xt, dtype=float)
    if u.ndim != 2:
        raise ValueError("u_xt must be 2-D")

    rho = pe.pointwise_bounds_margin(u, lower, upper)
    out = np.empty_like(rho, dtype=float)

    for t in range(u.shape[0]):
        t0 = max(0, t - t_window + 1)
        for x in range(u.shape[1]):
            x0 = max(0, x - x_window + 1)
            block = rho[t0 : t + 1, x0 : x + 1]
            out[t, x] = block.min() if extreme == "min" else block.max()
    return out


# Scalar robustness: 1-D signals


@pytest.mark.parametrize(
    "sig, lower, upper, expected",
    [
        ([0.2, 0.4, 0.6], 0.0, 1.0, 0.2),
        ([0.0, 1.0], 0.0, 1.0, 0.0),  # exactly on the bounds -> zero robustness
        ([0.5, 0.5], 0.0, 1.0, 0.5),  # centered in interval -> margin = 0.5
        ([-0.1, 0.2], 0.0, 1.0, -0.1),
        ([0.2, 1.2], 0.0, 1.0, -0.2),
    ],
    ids=[
        "typical",
        "on_bounds",
        "centered",
        "below_lower",
        "above_upper",
    ],
)
def test_compute_robustness_matches_literal_definition(
    sig: list[float],
    lower: float,
    upper: float,
    expected: float,
) -> None:
    arr = np.array(sig, dtype=float)
    got = pe.compute_robustness(arr, lower=lower, upper=upper)
    assert isinstance(got, float)
    assert got == pytest.approx(expected, abs=ATOL)
    assert got == pytest.approx(_scalar_band_robustness_loop(arr, lower, upper), abs=ATOL)


def test_compute_robustness_degenerate_interval() -> None:
    sig = np.array([0.2, 0.3, 0.6], dtype=float)
    # Predicate is x == 0.3.
    got = pe.compute_robustness(sig, lower=0.3, upper=0.3)
    assert got == pytest.approx(-0.3, abs=ATOL)


def test_compute_robustness_one_sided_bounds() -> None:
    sig = np.array([-1.0, 0.0, 1.0], dtype=float)

    # Upper-only: (-inf ≤ x ≤ U)  =>  min_t (U - x(t)) = U - max_t x(t)
    upper = 1.0
    got_upper_only = pe.compute_robustness(sig, lower=-np.inf, upper=upper)
    assert got_upper_only == pytest.approx(upper - sig.max(), abs=ATOL)

    # Lower-only: (L ≤ x ≤ +inf)  =>  min_t (x(t) - L) = min_t x(t) - L
    lower = -1.0
    got_lower_only = pe.compute_robustness(sig, lower=lower, upper=np.inf)
    assert got_lower_only == pytest.approx(sig.min() - lower, abs=ATOL)


def test_compute_robustness_propagates_nan() -> None:
    sig = np.array([0.0, np.nan, 1.0], dtype=float)
    got = pe.compute_robustness(sig, lower=0.0, upper=1.0)
    assert np.isnan(got)


@pytest.mark.parametrize(
    "bad",
    [
        np.array([], dtype=float),
        np.zeros((0,), dtype=float),
        np.zeros((1, 1), dtype=float),
    ],
    ids=["empty", "empty_1d", "not_1d"],
)
def test_compute_robustness_rejects_invalid_inputs(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        _ = pe.compute_robustness(bad, lower=0.0, upper=1.0)


def test_compute_robustness_does_not_mutate_and_accepts_ints() -> None:
    sig_f = np.array([0.2, 0.4, 0.6], dtype=float)
    sig_i = np.array([0, 1, 2], dtype=int)
    sig_f_copy = sig_f.copy()

    got_f = pe.compute_robustness(sig_f, 0.0, 1.0)
    got_i = pe.compute_robustness(sig_i, 0.0, 3.0)

    assert np.array_equal(sig_f, sig_f_copy)
    assert isinstance(got_f, float) and isinstance(got_i, float)
    assert got_i == pytest.approx(min(sig_i.min() - 0.0, 3.0 - sig_i.max()), abs=ATOL)


@pytest.mark.parametrize("shift", [-2.0, -0.5, 0.0, 0.42, 3.3])
def test_compute_robustness_translation_invariant(shift: float) -> None:
    sig = np.array([-0.2, 0.1, 0.9, 1.3], dtype=float)
    lower, upper = 0.0, 1.0
    base = pe.compute_robustness(sig, lower, upper)
    shifted = pe.compute_robustness(sig + shift, lower + shift, upper + shift)
    assert shifted == pytest.approx(base, abs=ATOL)


@pytest.mark.parametrize("scale", [0.2, 0.5, 1.0, 2.0, 10.0])
def test_compute_robustness_positive_homogeneous(scale: float) -> None:
    sig = np.array([0.2, 0.4, 0.6], dtype=float)
    lower, upper = 0.0, 1.0
    base = pe.compute_robustness(sig, lower, upper)
    scaled = pe.compute_robustness(scale * sig, scale * lower, scale * upper)
    assert scaled == pytest.approx(scale * base, abs=ATOL)


def test_compute_robustness_order_invariant() -> None:
    sig = np.array([0.25, 0.75, 0.4, 0.6], dtype=float)
    rev = sig[::-1].copy()
    lower, upper = 0.0, 1.0
    assert pe.compute_robustness(sig, lower, upper) == pytest.approx(
        pe.compute_robustness(rev, lower, upper),
        abs=ATOL,
    )


def test_robustness_monotone_in_bounds() -> None:
    """Tightening bounds cannot increase robustness; widening cannot decrease."""

    sig = np.array([0.2, 0.4, 0.6], dtype=float)
    base = pe.compute_robustness(sig, 0.0, 1.0)

    tighter = pe.compute_robustness(sig, 0.1, 0.7)
    wider = pe.compute_robustness(sig, -1.0, 2.0)

    assert tighter <= base + ATOL
    assert wider >= base - ATOL


# Scalar robustness: 2-D fields (time × space)


def test_compute_spatiotemporal_robustness_agrees_with_flatten() -> None:
    mat = np.array([[0.5, 0.6, 0.7], [0.2, 0.4, 0.9]], dtype=float)
    lower, upper = 0.0, 1.0
    r2d = pe.compute_spatiotemporal_robustness(mat, lower, upper)
    r1d = pe.compute_robustness(mat.ravel(), lower, upper)
    assert r2d == pytest.approx(r1d, abs=ATOL)


def test_compute_spatiotemporal_robustness_constant_and_typical() -> None:
    mat = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=float)
    assert pe.compute_spatiotemporal_robustness(mat, 0.0, 1.0) == pytest.approx(0.2, abs=ATOL)

    const = np.full((3, 4), 0.5, dtype=float)
    assert pe.compute_spatiotemporal_robustness(const, 0.0, 1.0) == pytest.approx(0.5, abs=ATOL)


def test_compute_spatiotemporal_transpose_invariant_and_no_mutation() -> None:
    mat = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.0, 0.5, 0.6, 0.7],
            [-0.1, 0.2, 0.9, 1.2],
        ],
        dtype=float,
    )
    lo, hi = -0.5, 0.75
    mat_copy = mat.copy()

    r = pe.compute_spatiotemporal_robustness(mat, lo, hi)
    r_t = pe.compute_spatiotemporal_robustness(mat.T, lo, hi)

    assert r == pytest.approx(r_t, abs=ATOL)
    assert np.array_equal(mat, mat_copy)


def test_compute_spatiotemporal_degenerate_interval() -> None:
    mat = np.array([[0.2, 0.3, 0.6], [0.1, 0.4, 0.9]], dtype=float)
    got = pe.compute_spatiotemporal_robustness(mat, lower=0.3, upper=0.3)
    assert got == pytest.approx(-0.6, abs=ATOL)


@pytest.mark.parametrize(
    "bad",
    [
        np.array([], dtype=float),
        np.zeros((0, 3), dtype=float),
        np.zeros((3, 0), dtype=float),
        np.zeros((2,), dtype=float),
        np.zeros((1, 1, 1), dtype=float),
    ],
    ids=["empty", "empty_rows", "empty_cols", "not_2d", "not_2d_3d"],
)
def test_compute_spatiotemporal_robustness_rejects_invalid_inputs(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        _ = pe.compute_spatiotemporal_robustness(bad, lower=0.0, upper=1.0)


def test_spatiotemporal_robustness_monotone_in_bounds() -> None:
    mat = np.array([[0.2, 0.4, 0.6], [0.1, 0.3, 0.5]], dtype=float)
    base = pe.compute_spatiotemporal_robustness(mat, 0.0, 1.0)
    tighter = pe.compute_spatiotemporal_robustness(mat, 0.1, 0.7)
    wider = pe.compute_spatiotemporal_robustness(mat, -1.0, 2.0)
    assert tighter <= base + ATOL
    assert wider >= base - ATOL


# Sliding-window temporal/spatial operators (toy STL "G" / "F")


@pytest.mark.parametrize("window", [1, 2, 3, 10], ids=lambda w: f"w={w}")
def test_stl_temporal_globally_matches_naive(window: int) -> None:
    rho = np.array([1.0, 0.5, -2.0, 3.0], dtype=float)
    expected = _trailing_window_extreme_1d(rho, window, extreme="min")
    got = pe.stl_globally_robustness(rho, window=window)
    assert np.allclose(got, expected, atol=ATOL, rtol=0.0)


@pytest.mark.parametrize("window", [1, 2, 3, 10], ids=lambda w: f"w={w}")
def test_stl_temporal_eventually_matches_naive(window: int) -> None:
    rho = np.array([1.0, 0.5, -2.0, 3.0], dtype=float)
    expected = _trailing_window_extreme_1d(rho, window, extreme="max")
    got = pe.stl_eventually_robustness(rho, window=window)
    assert np.allclose(got, expected, atol=ATOL, rtol=0.0)


@pytest.mark.parametrize("window", [0, -1])
def test_stl_temporal_ops_reject_nonpositive_window(window: int) -> None:
    rho = np.array([1.0, 2.0], dtype=float)
    with pytest.raises(ValueError):
        _ = pe.stl_globally_robustness(rho, window=window)
    with pytest.raises(ValueError):
        _ = pe.stl_eventually_robustness(rho, window=window)


def test_stl_temporal_eventually_cooling_example_full_window() -> None:
    """A small, concrete example mirroring an "eventually cool" pattern.

    If the predicate is (temp ≤ 25), the pointwise margin is (25 - temp).
    For a full-horizon trailing F, the final robustness equals:

        max_t (25 - temp[t]) = 25 - min_t temp[t]
    """

    temp = np.array([50.0, 40.0, 30.0, 24.0, 20.0], dtype=float)
    rho = pe.pointwise_bounds_margin(temp, lower=-np.inf, upper=25.0)
    r_f = pe.stl_eventually_robustness(rho, window=temp.size)
    assert r_f[-1] == pytest.approx(25.0 - temp.min(), abs=ATOL)


def test_stl_spatial_ops_match_naive_along_axis1() -> None:
    rho_xt = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [0.0, -1.0, 2.0, 1.0],
        ],
        dtype=float,
    )
    window = 3
    expected_min = _trailing_window_extreme_axis1(rho_xt, window, extreme="min")
    expected_max = _trailing_window_extreme_axis1(rho_xt, window, extreme="max")

    got_min = pe.stl_spatial_globally_robustness(rho_xt, window=window)
    got_max = pe.stl_spatial_eventually_robustness(rho_xt, window=window)

    assert np.allclose(got_min, expected_min, atol=ATOL, rtol=0.0)
    assert np.allclose(got_max, expected_max, atol=ATOL, rtol=0.0)


def test_stl_spatial_ops_reject_non_2d() -> None:
    rho = np.array([1.0, 2.0, 3.0], dtype=float)
    with pytest.raises(ValueError):
        _ = pe.stl_spatial_globally_robustness(rho, window=2)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        _ = pe.stl_spatial_eventually_robustness(rho, window=2)  # type: ignore[arg-type]


def test_stl_rect_globally_bounds_matches_naive_rectangle_min() -> None:
    u_xt = np.array(
        [
            [0.0, 0.2, 0.4, 0.6],
            [0.1, 0.3, 0.5, 0.7],
            [0.2, 0.4, 0.6, 0.8],
        ],
        dtype=float,
    )
    lower, upper = 0.0, 1.0
    t_window, x_window = 2, 3

    expected = _rect_window_extreme_bounds(
        u_xt,
        lower,
        upper,
        t_window=t_window,
        x_window=x_window,
        extreme="min",
    )
    got = pe.stl_rect_globally_bounds(u_xt, lower, upper, t_window=t_window, x_window=x_window)
    assert np.allclose(got, expected, atol=ATOL, rtol=0.0)


def test_stl_rect_eventually_bounds_matches_naive_rectangle_max() -> None:
    u_xt = np.array(
        [
            [0.0, 0.2, 0.4, 0.6],
            [0.1, 0.3, 0.5, 0.7],
            [0.2, 0.4, 0.6, 0.8],
        ],
        dtype=float,
    )
    lower, upper = 0.0, 1.0
    t_window, x_window = 2, 3

    expected = _rect_window_extreme_bounds(
        u_xt,
        lower,
        upper,
        t_window=t_window,
        x_window=x_window,
        extreme="max",
    )
    got = pe.stl_rect_eventually_bounds(u_xt, lower, upper, t_window=t_window, x_window=x_window)
    assert np.allclose(got, expected, atol=ATOL, rtol=0.0)


def test_stl_rect_full_window_recovers_global_min_or_max_at_last_cell() -> None:
    """Full windows cover the entire past rectangle at (t=-1, x=-1)."""

    u_xt = np.array(
        [
            [0.2, 0.4],
            [0.1, 0.9],
        ],
        dtype=float,
    )
    lower, upper = 0.0, 1.0
    t_window, x_window = u_xt.shape

    rho = pe.pointwise_bounds_margin(u_xt, lower, upper)
    global_min = float(rho.min())
    global_max = float(rho.max())

    got_g = pe.stl_rect_globally_bounds(u_xt, lower, upper, t_window=t_window, x_window=x_window)
    got_f = pe.stl_rect_eventually_bounds(u_xt, lower, upper, t_window=t_window, x_window=x_window)

    assert got_g[-1, -1] == pytest.approx(global_min, abs=ATOL)
    assert got_f[-1, -1] == pytest.approx(global_max, abs=ATOL)
