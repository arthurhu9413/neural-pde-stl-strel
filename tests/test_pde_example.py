"""Unit tests for the NumPy-only PDE sandbox.

This repository includes a tiny, dependency-light diffusion (heat) equation simulator in
`neural_pde_stl_strel.pde_example`. These tests intentionally serve *two* purposes:

1) Verify the simulator and simple robustness helpers behave as expected.
2) Provide executable "toy examples" that can be used in demos when explaining how
   (spatio-)temporal specifications relate to PDE traces (e.g., global safety bounds and
   eventual cooling).

The simulator is deliberately simple (forward-Euler in time, centered differences in space)
so these tests focus on properties we *expect* from that discrete scheme when the diffusion
number r = alpha * dt / dx^2 is stable (r <= 1/2).
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest


# Support running tests from a source checkout without requiring an editable install.
try:  # pragma: no cover
    import neural_pde_stl_strel.pde_example as pe
except ModuleNotFoundError:  # pragma: no cover
    ROOT = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "src"))
    import neural_pde_stl_strel.pde_example as pe


TOL = 1e-12


def _diffusion_number(*, alpha: float, dt: float, dx: float) -> float:
    """Return r = alpha * dt / dx^2 used by the explicit FTCS diffusion scheme."""

    return float(alpha) * float(dt) / (float(dx) ** 2)


def _assert_nonincreasing(arr: np.ndarray, *, atol: float = TOL) -> None:
    """Assert arr[i+1] <= arr[i] (up to atol) for all i."""

    diffs = np.diff(arr)
    assert np.all(diffs <= atol), f"Expected nonincreasing sequence, max Δ={diffs.max()}"


def _assert_nondecreasing(arr: np.ndarray, *, atol: float = TOL) -> None:
    """Assert arr[i+1] >= arr[i] (up to atol) for all i."""

    diffs = np.diff(arr)
    assert np.all(diffs >= -atol), f"Expected nondecreasing sequence, min Δ={diffs.min()}"


# Finite-difference diffusion simulator


def test_simulate_diffusion_shape_dtype_and_finiteness() -> None:
    u = pe.simulate_diffusion(length=3, steps=5, dt=0.1, alpha=0.1)

    assert u.shape == (6, 3)
    assert u.dtype == np.float64
    assert np.isfinite(u).all()


def test_simulate_diffusion_zero_steps_returns_initial() -> None:
    """If steps == 0, the simulator should return exactly one frame (the initial state)."""

    # Default initial condition: a single hot spot at the left boundary.
    u0 = pe.simulate_diffusion(length=4, steps=0)
    assert u0.shape == (1, 4)
    np.testing.assert_allclose(
        u0[0],
        np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        atol=TOL,
        rtol=0.0,
    )

    # Custom initial condition is respected exactly when steps == 0.
    init = np.array([0.2, -0.1, 0.4, 0.0], dtype=float)
    init_copy = init.copy()
    u0b = pe.simulate_diffusion(length=4, steps=0, initial=init)
    np.testing.assert_allclose(u0b[0], init, atol=TOL, rtol=0.0)
    # Defensive: the helper should not mutate the caller's array.
    np.testing.assert_array_equal(init, init_copy)


def test_simulate_diffusion_length_one_is_constant() -> None:
    """With a single spatial cell there are no neighbors, so the state never changes."""

    u = pe.simulate_diffusion(length=1, steps=5, dt=0.1, alpha=0.1)
    assert u.shape == (6, 1)
    np.testing.assert_allclose(u[:, 0], u[0, 0], atol=TOL, rtol=0.0)


def test_simulate_diffusion_rejects_bad_inputs() -> None:
    """Basic argument validation (used in demos as a quick smoke test)."""

    # Geometry/time parameters.
    with pytest.raises(ValueError):
        pe.simulate_diffusion(length=0, steps=1)
    with pytest.raises(ValueError):
        pe.simulate_diffusion(length=3, steps=-1)
    # dx=0 is nonsensical and should fail fast (current implementation hits division by zero).
    # If the simulator later adds explicit validation, ValueError is also acceptable.
    with pytest.raises((ValueError, ZeroDivisionError)):
        pe.simulate_diffusion(length=3, steps=1, dx=0.0)

    # Initial state must be 1-D with matching length.
    with pytest.raises(ValueError):
        pe.simulate_diffusion(length=3, steps=1, initial=np.zeros((3, 3)))
    with pytest.raises(ValueError):
        pe.simulate_diffusion(length=3, steps=1, initial=np.zeros(4))


def test_interior_update_is_convex_combination_when_stable() -> None:
    """For r <= 1/2, each interior update is a convex combination of neighbors."""

    rng = np.random.default_rng(42)
    length = 9
    init = rng.normal(size=length)
    dt, alpha, dx = 0.1, 0.1, 1.0

    r = _diffusion_number(alpha=alpha, dt=dt, dx=dx)
    assert r <= 0.5

    u = pe.simulate_diffusion(
        length=length,
        steps=1,
        dt=dt,
        alpha=alpha,
        dx=dx,
        initial=init,
    )

    # Interior points: u_i^{n+1} = r*u_{i-1}^n + (1-2r)*u_i^n + r*u_{i+1}^n
    # With r <= 1/2, weights are nonnegative and sum to 1.
    u1 = u[1]
    for i in range(1, length - 1):
        mn = float(np.min(init[i - 1 : i + 2]))
        mx = float(np.max(init[i - 1 : i + 2]))
        assert mn - TOL <= u1[i] <= mx + TOL


def test_discrete_maximum_principle_for_stable_r() -> None:
    """For stable r, the global max cannot increase and the global min cannot decrease."""

    rng = np.random.default_rng(0)
    init = rng.normal(size=25)
    dt, alpha, dx = 0.1, 0.1, 1.0
    assert _diffusion_number(alpha=alpha, dt=dt, dx=dx) <= 0.5

    u = pe.simulate_diffusion(length=init.size, steps=10, dt=dt, alpha=alpha, dx=dx, initial=init)
    max_t = u.max(axis=1)
    min_t = u.min(axis=1)

    _assert_nonincreasing(max_t)
    _assert_nondecreasing(min_t)
    _assert_nonincreasing(max_t - min_t)  # peak-to-peak never increases


def test_length3_first_step_is_uniform_equal_to_r() -> None:
    """Exact check on a tiny grid.

    With the default initial condition [1, 0, 0] and Neumann-style boundary copying,
    after one step every cell becomes r = alpha*dt/dx^2.
    """

    dt, alpha, dx = 0.2, 0.2, 1.0
    r = _diffusion_number(alpha=alpha, dt=dt, dx=dx)
    u = pe.simulate_diffusion(length=3, steps=1, dt=dt, alpha=alpha, dx=dx)
    np.testing.assert_allclose(u[1], np.full(3, r), atol=TOL, rtol=0.0)


def test_simulate_diffusion_emits_runtime_warning_when_unstable() -> None:
    """Choose r > 1/2 to trigger the stability warning (used in demos)."""

    dt, alpha, dx = 1.1, 0.6, 1.0
    assert _diffusion_number(alpha=alpha, dt=dt, dx=dx) > 0.5

    with pytest.warns(RuntimeWarning, match="CFL number r=.*unstable"):
        u = pe.simulate_diffusion(length=5, steps=1, dt=dt, alpha=alpha, dx=dx)
    assert u.shape == (2, 5)


def test_simulate_diffusion_respects_dtype_override() -> None:
    u = pe.simulate_diffusion(length=4, steps=2, dt=0.1, alpha=0.1, dtype=np.float32)
    assert u.dtype == np.float32
    assert np.isfinite(u).all()


# Simulator with per-step clipping (a simple form of "safety shielding")


def test_diffusion_with_clipping_shape_and_bounds() -> None:
    u = pe.simulate_diffusion_with_clipping(
        length=4,
        steps=3,
        dt=0.1,
        alpha=0.1,
        lower=-0.25,
        upper=0.25,
    )
    assert u.shape == (4, 4)
    assert (u >= -0.25 - TOL).all()
    assert (u <= 0.25 + TOL).all()


def test_first_frame_is_also_clipped() -> None:
    u = pe.simulate_diffusion_with_clipping(length=3, steps=0, lower=0.0, upper=0.2)
    assert u.shape == (1, 3)
    np.testing.assert_allclose(u[0], np.array([0.2, 0.0, 0.0]), atol=TOL, rtol=0.0)


def test_monotone_decay_when_clipped() -> None:
    """End-to-end toy example: safety bound + eventual cooling on a diffusion trace.

    We treat u(x,t) as "temperature" on a 1-D spatial grid.

    * Safety (always) bound (STL sketch):   G(u <= U_max)
      We monitor this via max_x u(t) and the quantitative margin U_max - max_{x,t} u.

    * Eventual cooling (STL sketch):        F(max_x u <= eps)
      Here we simply assert the maximum eventually drops below a small threshold.

    Per-step clipping enforces the hard bound u <= U_max (and u >= lower) at every frame.
    """

    length, steps = 51, 50
    dt, alpha, dx = 0.1, 0.1, 1.0
    lower, upper = 0.0, 0.5

    # Baseline run violates the bound at t=0 because the default initial condition has u=1.
    u_base = pe.simulate_diffusion(length=length, steps=steps, dt=dt, alpha=alpha, dx=dx)
    max_base = u_base.max(axis=1)
    assert max_base[0] == pytest.approx(1.0, abs=TOL)

    # Clipped run enforces the bound at every frame.
    u_clip = pe.simulate_diffusion_with_clipping(
        length=length,
        steps=steps,
        dt=dt,
        alpha=alpha,
        dx=dx,
        lower=lower,
        upper=upper,
    )
    max_clip = u_clip.max(axis=1)

    assert (u_clip >= lower - TOL).all()
    assert (u_clip <= upper + TOL).all()
    _assert_nonincreasing(max_clip)

    # Quantitative satisfaction margin for G(max_x u <= U_max).
    rob_base = pe.compute_robustness(max_base, lower=-np.inf, upper=upper)
    rob_clip = pe.compute_robustness(max_clip, lower=-np.inf, upper=upper)
    assert rob_base == pytest.approx(upper - 1.0, abs=TOL)  # -0.5
    assert rob_clip == pytest.approx(0.0, abs=TOL)

    # "Eventually" cooling: after enough time the max temperature becomes very small.
    assert max_clip[-1] < 4e-3


def test_per_step_clipping_never_worsens_robustness() -> None:
    """Clipping should (weakly) increase the robustness for a rectangular bound spec."""

    lower, upper = 0.0, 0.5
    base_u = pe.simulate_diffusion(length=5, steps=5, dt=0.1, alpha=0.1)
    base_rob = pe.compute_spatiotemporal_robustness(base_u, lower=lower, upper=upper)
    u_clip = pe.simulate_diffusion_with_clipping(
        length=5,
        steps=5,
        dt=0.1,
        alpha=0.1,
        lower=lower,
        upper=upper,
    )
    clip_rob = pe.compute_spatiotemporal_robustness(u_clip, lower=lower, upper=upper)

    assert base_rob == pytest.approx(-0.5, abs=TOL)
    assert clip_rob == pytest.approx(0.0, abs=TOL)
    assert clip_rob >= base_rob - TOL
