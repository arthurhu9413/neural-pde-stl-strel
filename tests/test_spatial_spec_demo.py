"""Tests for the SpaTiaL-based spatial-temporal monitoring *demo*.

Repository-aligned intent
The repository should contain **actually-run examples** with:

* the *specifications written out* (STL / spatial-STL variants),
* evidence of both **satisfaction** and **falsification**, and
* **plots/figures** that make the behavior and monitoring results easy to see.

This test module turns :mod:`neural_pde_stl_strel.monitors.spatial_demo` into such a
toy example.

Demo property (written explicitly)
The demo scene is a moving circular *agent* and a static circular *goal*.  We
monitor the bounded eventually property:

    φ := F[0, T-1] ( distance(agent, goal) <= ε )

with **quantitative** (real-valued) robustness semantics.  Intuitively, the
property says: "within the horizon, the agent gets within ε of the goal".

What these tests cover
1) The demo runs quickly and returns a finite ``float`` robustness value.
2) A short horizon falsifies the property; a long enough horizon satisfies it.
3) Robustness for an ``eventually`` property is monotone (non-decreasing) as
   the horizon grows.
4) We generate a small diagnostic figure (robustness vs. time) to ensure the
   repo can produce the sort of plots expected for a clear technical demonstration.
5) If the optional SpaTiaL stack is installed, we sanity-check agreement
   against an exact, dependency-free geometric calculation.

Naming note
This file is named ``test_spatial_spec_demo.py`` for historical reasons in this
repository. It tests the **SpaTiaL monitoring library** (import name
``spatial``) used by :mod:`neural_pde_stl_strel.monitors.spatial_demo`, not the
separate ``spatial_spec`` package.
"""

from __future__ import annotations

import math
import os
import pathlib
import sys
from typing import Iterable, List

import numpy as np
import pytest


# Ensure headless plotting works in CI.
os.environ.setdefault("MPLBACKEND", "Agg")


# Support running tests from a source checkout without installation.
try:  # pragma: no cover - import convenience
    from neural_pde_stl_strel.monitors import spatial_demo  # type: ignore
except Exception:  # pragma: no cover - import convenience
    ROOT = pathlib.Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if SRC.exists():
        sys.path.insert(0, str(SRC))
    from neural_pde_stl_strel.monitors import spatial_demo  # type: ignore  # noqa: E402


def _exact_eventually_reach_robustness(cfg: spatial_demo.ToyScene) -> float:
    """Exact robustness for the demo property under discrete-time semantics.

    The demo property is:

        F[0,T-1] ( distance(agent, goal) <= eps )

    For quantitative semantics, the robustness is:

        max_{t in [0,T-1]} ( eps - d(t) )

    where ``d(t)`` is the *non-negative* separation between the two discs
    (0 if overlapping/touching).

    This helper computes that quantity by enumerating the discrete horizon.
    """

    assert cfg.T >= 1, "ToyScene.T must be >= 1 for a bounded temporal formula"

    # Mirror the demo's "touching" convention: eps==0 -> tiny positive epsilon.
    eps = float(cfg.reach_eps if cfg.reach_eps > 0.0 else 1e-12)

    v = float(cfg.agent_speed)
    x_goal = float(cfg.goal_pos[0])
    r_sum = float(cfg.agent_radius + cfg.goal_radius)

    best = -float("inf")
    for t in range(int(cfg.T)):
        # Agent moves along +x axis from x=0.
        x_agent = v * float(t)
        center_dist = abs(x_goal - x_agent)
        sep = max(0.0, center_dist - r_sum)
        best = max(best, eps - sep)

    return float(best)


def _robustness_trace(cfg: spatial_demo.ToyScene) -> np.ndarray:
    """Return the per-time robustness trace: ρ(t) = eps - distance(t)."""

    eps = float(cfg.reach_eps if cfg.reach_eps > 0.0 else 1e-12)
    v = float(cfg.agent_speed)
    x_goal = float(cfg.goal_pos[0])
    r_sum = float(cfg.agent_radius + cfg.goal_radius)

    out = np.empty(int(cfg.T), dtype=float)
    for i, t in enumerate(range(int(cfg.T))):
        x_agent = v * float(t)
        center_dist = abs(x_goal - x_agent)
        sep = max(0.0, center_dist - r_sum)
        out[i] = eps - sep

    return out


def _pairwise_non_decreasing(xs: Iterable[float]) -> bool:
    it = iter(xs)
    try:
        prev = next(it)
    except StopIteration:
        return True
    for x in it:
        if x < prev:
            return False
        prev = x
    return True


def test_spatial_demo_smoke_returns_finite_float() -> None:
    """The demo should run quickly and return a finite float.

    This is a minimal regression test that should pass regardless of whether
    the optional SpaTiaL stack is installed (the demo has a no-dependency
    analytical fallback).
    """

    val = spatial_demo.run_demo(T=5)
    assert isinstance(val, float)
    assert math.isfinite(val)


def test_spatial_demo_eventually_reach_falsified_then_satisfied(monkeypatch: pytest.MonkeyPatch) -> None:
    """Show one falsifying run and one satisfying run of the same spec.

    We *force* the analytical fallback path for determinism and to ensure the
    tests are robust when SpaTiaL isn't installed.
    """

    monkeypatch.setattr(spatial_demo, "Spatial", None, raising=False)

    # Use a slightly positive eps for a clear margin.
    cfg_short = spatial_demo.ToyScene(T=10, reach_eps=0.10)
    cfg_long = spatial_demo.ToyScene(T=35, reach_eps=0.10)

    r_short = spatial_demo.evaluate_formula(cfg_short)
    r_long = spatial_demo.evaluate_formula(cfg_long)

    assert r_short == pytest.approx(_exact_eventually_reach_robustness(cfg_short), abs=1e-12)
    assert r_long == pytest.approx(_exact_eventually_reach_robustness(cfg_long), abs=1e-12)

    assert r_short < 0.0, "Short horizon should fail to reach the goal"
    assert r_long > 0.0, "Long horizon should reach the goal within eps"


def test_spatial_demo_eventually_monotone_in_horizon(monkeypatch: pytest.MonkeyPatch) -> None:
    """Eventually robustness should be non-decreasing as the horizon grows."""

    monkeypatch.setattr(spatial_demo, "Spatial", None, raising=False)

    cfgs: List[spatial_demo.ToyScene] = [
        spatial_demo.ToyScene(T=T, reach_eps=0.10) for T in (5, 10, 20, 30, 35)
    ]
    rs = [spatial_demo.evaluate_formula(cfg) for cfg in cfgs]

    assert _pairwise_non_decreasing(rs), f"Expected non-decreasing robustness, got {rs!r}"


def test_spatial_demo_can_generate_a_figure(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Generate a small plot of robustness vs time for the demo spec.

    This mirrors the sort of lightweight figure that is useful in
    the repository/report.
    """

    monkeypatch.setattr(spatial_demo, "Spatial", None, raising=False)

    cfg = spatial_demo.ToyScene(T=35, reach_eps=0.10)
    trace = _robustness_trace(cfg)
    r = spatial_demo.evaluate_formula(cfg)

    assert r == pytest.approx(float(np.max(trace)), abs=1e-12)
    assert r > 0.0

    # Local import to avoid importing pyplot at module import time.
    import matplotlib.pyplot as plt

    t = np.arange(cfg.T, dtype=int)
    t_star = int(np.argmax(trace))

    fig, ax = plt.subplots()
    ax.plot(t, trace)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.scatter([t_star], [trace[t_star]])
    ax.set_xlabel("time step t")
    ax.set_ylabel("robustness ρ(t)")
    ax.set_title("SpaTiaL demo: ρ(t) = ε − distance(agent, goal)")
    fig.tight_layout()

    out = tmp_path / "spatial_demo_robustness_trace.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    assert out.exists() and out.stat().st_size > 0


@pytest.mark.skipif(getattr(spatial_demo, "Spatial", None) is None, reason="SpaTiaL (spatial-lib) not installed")
def test_spatial_demo_spatial_lib_agrees_with_exact_geometry() -> None:
    """If SpaTiaL is installed, it should match exact disc-distance semantics.

    SpaTiaL represents circles as polygons (via Shapely). For this toy problem
    the result should agree very closely with the exact, analytic computation.
    """

    cfg = spatial_demo.ToyScene(T=35, reach_eps=0.10)
    r_expected = _exact_eventually_reach_robustness(cfg)
    r_spatial = spatial_demo.evaluate_formula(cfg)

    # Conservative tolerance: polygonal approximations can introduce tiny error.
    assert r_spatial == pytest.approx(r_expected, abs=1e-2)
