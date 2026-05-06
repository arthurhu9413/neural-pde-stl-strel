from __future__ import annotations

"""SpaTiaL spatial-temporal monitoring demo (object-centric).

This module is a tiny, CPU-friendly *example you can actually run*, designed
for demos and CI tests. It constructs a toy 2D scene with

- a circular *agent* moving at constant velocity along the +x axis, and
- a static circular *goal*,

and evaluates the **bounded eventually** spatio-temporal property

    φ := F[0, T-1] ( distance(agent, goal) <= eps )

under *quantitative semantics* (real-valued robustness).

Semantics (discrete time)
SpaTiaL's quantitative semantics for ``F`` (eventually) over a finite horizon
``[0, T-1]`` is:

    ρ(φ) = max_{t = 0..T-1} ρ_t,
    ρ_t  = eps - dist(agent(t), goal)

where ``dist`` is the **shape separation distance** (0 when the shapes touch or
overlap).

Design goals
- **Runnable example with the spec written out.** The spec above is exactly
  what this demo monitors.
- **Dependency-light by default.** SpaTiaL is an *optional* dependency. If it is
  unavailable (or if a local SpaTiaL grammar differs), we fall back to an
  analytical computation that mirrors the discrete-time semantics for this
  scene.
- **Version-tolerant.** We try a small set of equivalent formula spellings and
  call signatures to handle minor API/grammar differences across SpaTiaL
  releases.

High-level data flow
ToyScene cfg
  -> (optional) SpaTiaL geometry objects (DynamicObject/StaticObject)
  -> parse φ (try a few grammar spellings)
  -> interpret φ over [0, T-1]
  -> robustness (float)

Fallback (no SpaTiaL)
  -> compute separation distance at each integer t
  -> return max_t (eps - separation)

Notes
-----
- ``reach_eps = 0`` is interpreted as "touching". SpaTiaL currently asserts
  ``eps > 0`` in its distance predicate, so we substitute a tiny positive
  epsilon when needed.

References
- SpaTiaL project: https://github.com/KTH-RPL-Planiacs/SpaTiaL
- SpaTiaL docs: https://kth-rpl-planiacs.github.io/SpaTiaL/
"""

from dataclasses import dataclass
import math
from typing import Any

# Optional dependency: SpaTiaL (module name: "spatial").
# In this repository we typically install it from Git (see docs/INSTALL_EXTRAS.md).
try:  # pragma: no cover - optional dependency path
    from spatial.logic import Spatial  # type: ignore
except Exception:  # pragma: no cover
    Spatial = None  # type: ignore[assignment]


_TINY_EPS: float = 1e-12


@dataclass(slots=True)
class ToyScene:
    """Parameters for the toy scene.

    Attributes
    T:
        Number of *discrete* time steps in the horizon; monitoring is done over
        ``t = 0, 1, ..., T-1``.
    agent_speed:
        Constant speed of the agent (units per step) along the +x axis.
    agent_radius:
        Radius of the agent circle.
    goal_pos:
        Goal center ``(x, y)``.
    goal_radius:
        Radius of the goal circle.
    reach_eps:
        Tolerance ``eps`` in ``distance(agent, goal) <= eps``.

        If ``reach_eps <= 0``, we interpret it as "touching" and substitute a
        tiny positive epsilon for compatibility with SpaTiaL's current
        distance predicate implementation.
    """

    T: int = 50
    agent_speed: float = 0.35
    agent_radius: float = 0.30
    goal_pos: tuple[float, float] = (12.0, 0.0)
    goal_radius: float = 0.40
    reach_eps: float = 0.0

    def __post_init__(self) -> None:
        # Normalize common numeric inputs (e.g., numpy scalars) and validate.
        self.T = int(self.T)
        self.agent_speed = float(self.agent_speed)
        self.agent_radius = float(self.agent_radius)
        self.goal_radius = float(self.goal_radius)
        gx, gy = self.goal_pos
        self.goal_pos = (float(gx), float(gy))
        self.reach_eps = float(self.reach_eps)

        if self.T <= 0:
            raise ValueError(f"T must be a positive integer; got {self.T!r}")
        if not math.isfinite(self.agent_speed):
            raise ValueError(f"agent_speed must be finite; got {self.agent_speed!r}")
        if not math.isfinite(self.agent_radius) or self.agent_radius < 0.0:
            raise ValueError(f"agent_radius must be finite and non-negative; got {self.agent_radius!r}")
        if not math.isfinite(self.goal_radius) or self.goal_radius < 0.0:
            raise ValueError(f"goal_radius must be finite and non-negative; got {self.goal_radius!r}")
        if not (math.isfinite(self.goal_pos[0]) and math.isfinite(self.goal_pos[1])):
            raise ValueError(f"goal_pos must be finite; got {self.goal_pos!r}")
        # Negative eps is nonsensical for a distance upper-bound and would not
        # match the SpaTiaL branch (which requires eps > 0). Enforce >= 0.
        if not math.isfinite(self.reach_eps) or self.reach_eps < 0.0:
            raise ValueError(f"reach_eps must be finite and >= 0; got {self.reach_eps!r}")


def build_scene(T: int = 50) -> ToyScene:
    """Convenience constructor used by demos/tests."""

    return ToyScene(T=T)


# Small helpers


def _safe_eps(eps: float) -> float:
    """Map a user epsilon to a SpaTiaL-safe epsilon.

    SpaTiaL's distance predicate currently asserts ``eps > 0``. For the common
    "touching" case ``eps = 0`` we use a tiny positive number.
    """

    # The ``eps > 0`` check also guards against NaN.
    return eps if (eps > 0.0) else _TINY_EPS


def _agent_center(cfg: ToyScene, t: int) -> tuple[float, float]:
    """Agent center at integer time ``t``."""

    return (cfg.agent_speed * float(t), 0.0)


def _separation_distance(cfg: ToyScene, t: int) -> float:
    """Separation distance between agent(t) and goal (0 if touching/overlap)."""

    ax, ay = _agent_center(cfg, t)
    gx, gy = cfg.goal_pos
    center_d = math.hypot(gx - ax, gy - ay)
    r_sum = cfg.agent_radius + cfg.goal_radius
    return max(0.0, center_d - r_sum)


def robustness_trace(cfg: ToyScene, *, eps: float | None = None) -> list[float]:
    """Return the per-time-step predicate robustness trace.

    The trace is:

        ρ_t = eps - dist(agent(t), goal)

    for ``t = 0..T-1``.

    This is useful for plotting/figures. The property robustness for
    ``F[0, T-1]`` is simply ``max(trace)``.
    """

    eps_val = _safe_eps(cfg.reach_eps if eps is None else float(eps))
    return [eps_val - _separation_distance(cfg, t) for t in range(cfg.T)]


def _analytical_eventually_robustness(cfg: ToyScene, *, eps: float | None = None) -> float:
    """Dependency-free robustness for ``F[0,T-1] (distance(agent, goal) <= eps)``."""

    return max(robustness_trace(cfg, eps=eps))


# SpaTiaL-backed evaluation


def _build_spatial_objects(cfg: ToyScene) -> dict[str, Any]:
    """Construct SpaTiaL geometry objects for the scene."""

    # Import SpaTiaL geometry lazily so the module remains importable without it.
    try:
        import numpy as np
        from spatial.geometry import (  # type: ignore
            Circle,
            DynamicObject,
            PolygonCollection,
            StaticObject,
        )
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "SpaTiaL geometry backend is not available.\n"
            "Install SpaTiaL (Linux/macOS) via:\n"
            "  pip install \"spatial @ git+https://github.com/"
            "KTH-RPL-Planiacs/SpaTiaL.git#subdirectory=spatial-lib\" shapely lark\n"
            "If you do not want optional dependencies, this demo will still run\n"
            "using the analytical fallback (SpaTiaL missing)."
        ) from e

    agent = DynamicObject()
    for t in range(cfg.T):
        ax, ay = _agent_center(cfg, t)
        center = np.array([ax, ay], dtype=float)
        footprint = PolygonCollection({Circle(center, float(cfg.agent_radius))})
        agent.addObject(footprint, time=t)

    gx, gy = cfg.goal_pos
    goal_center = np.array([gx, gy], dtype=float)
    goal_shape = PolygonCollection({Circle(goal_center, float(cfg.goal_radius))})
    goal = StaticObject(goal_shape)

    return {"agent": agent, "goal": goal}


def _first_parsed(sp: Any, candidates: list[str]) -> tuple[str, Any]:
    """Return the first (formula_str, parse_tree) that parses successfully."""

    for s in candidates:
        try:
            tree = sp.parse(s)
        except Exception:
            continue
        if tree is not None:
            return s, tree

    tried = "\n".join(candidates)
    raise ValueError(f"Could not parse any SpaTiaL formula; tried:\n{tried}")


def _make_spatial_interpreter() -> Any:
    """Create a quantitative SpaTiaL interpreter (version-tolerant)."""

    if Spatial is None:  # pragma: no cover - handled by caller
        raise RuntimeError("SpaTiaL is not installed")

    # Prefer the keyword-argument API (documented). Fall back to positional.
    try:
        return Spatial(quantitative=True)  # type: ignore[misc]
    except TypeError:  # pragma: no cover
        return Spatial(True)  # type: ignore[misc]


def _register_variables(sp: Any, vars_map: dict[str, Any]) -> None:
    """Register variables with SpaTiaL across minor API differences."""

    assign = getattr(sp, "assign_variable", None) or getattr(sp, "assignVariable", None)
    if not callable(assign):  # pragma: no cover
        raise AttributeError("SpaTiaL Spatial object has no assign_variable method")

    for name, obj in vars_map.items():
        assign(name, obj)

    # Some SpaTiaL versions also require updating the temporal interpreter vars.
    for meth in ("update_variables", "updateVariables", "set_variables", "setVariables"):
        fn = getattr(sp, meth, None)
        if callable(fn):
            try:
                fn(vars_map)
                break
            except Exception:
                continue


def _interpret(sp: Any, tree: Any, *, lower: int, upper: int) -> float:
    """Call SpaTiaL interpret() robustly and normalize to float."""

    try:
        val = sp.interpret(tree, lower=lower, upper=upper)
    except TypeError:  # pragma: no cover
        val = sp.interpret(tree, lower, upper)

    if val is None:  # pragma: no cover
        raise RuntimeError("SpaTiaL interpret() returned None")
    return float(val)


def _formula_candidates(*, upper: int, eps: float) -> list[str]:
    """Small set of equivalent formula spellings across SpaTiaL releases."""

    # Use a stable float spelling; Lark parses scientific notation as well.
    eps_s = format(float(eps), ".17g")

    # Prefer bounded eventually if supported; then fall back to unbounded syntax
    # (still evaluated over [lower, upper] by interpret()).
    return [
        f"F[0, {upper}] ( distance(agent, goal) <= {eps_s} )",
        f"F[0,{upper}](distance(agent, goal) <= {eps_s})",
        f"eventually [0, {upper}] ( distance(agent, goal) <= {eps_s} )",
        f"eventually[0,{upper}](distance(agent, goal) <= {eps_s})",
        f"F ( distance(agent, goal) <= {eps_s} )",
        f"eventually ( distance(agent, goal) <= {eps_s} )",
    ]


# Public API


def evaluate_formula(cfg: ToyScene, *, prefer_spatial: bool = True, strict: bool = False) -> float:
    """Evaluate ``φ`` and return its robustness.

    Parameters
    cfg:
        Scene configuration.
    prefer_spatial:
        If ``True`` (default), try SpaTiaL first when available.
    strict:
        If ``True``, raise an exception if SpaTiaL is available but cannot be
        used (e.g., parse/interpret errors). If ``False`` (default), fall back
        to the analytical computation.

    Returns
    -------
    float
        Robustness value; ``>= 0`` means the property is satisfied.
    """

    eps = _safe_eps(cfg.reach_eps)
    upper = cfg.T - 1

    # If SpaTiaL is missing, or user requests the fallback, use the analytical path.
    if (not prefer_spatial) or Spatial is None:
        return _analytical_eventually_robustness(cfg, eps=eps)

    try:
        vars_map = _build_spatial_objects(cfg)
        sp = _make_spatial_interpreter()
        _register_variables(sp, vars_map)
        _, tree = _first_parsed(sp, _formula_candidates(upper=upper, eps=eps))
        return _interpret(sp, tree, lower=0, upper=upper)
    except Exception:
        if strict:
            raise
        return _analytical_eventually_robustness(cfg, eps=eps)


def run_demo(T: int = 50) -> float:
    """Run the default demo and return the robustness."""

    return evaluate_formula(ToyScene(T=T))


def _cli(argv: list[str] | None = None) -> int:  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--T", type=int, default=50, help="Number of time steps")
    p.add_argument("--speed", type=float, default=0.35, help="Agent speed (units/step)")
    p.add_argument("--agent-radius", type=float, default=0.30, help="Agent radius")
    p.add_argument("--goal-x", type=float, default=12.0, help="Goal x coordinate")
    p.add_argument("--goal-y", type=float, default=0.0, help="Goal y coordinate")
    p.add_argument("--goal-radius", type=float, default=0.40, help="Goal radius")
    p.add_argument("--eps", type=float, default=0.0, help="Reach tolerance eps (0 means touching)")
    p.add_argument("--no-spatial", action="store_true", help="Skip SpaTiaL and use analytical fallback")
    p.add_argument("--strict", action="store_true", help="Fail if SpaTiaL is available but unusable")
    p.add_argument("--plot", type=str, default="", help="If set, save a robustness trace plot to this path")

    args = p.parse_args(argv)

    cfg = ToyScene(
        T=args.T,
        agent_speed=args.speed,
        agent_radius=args.agent_radius,
        goal_pos=(args.goal_x, args.goal_y),
        goal_radius=args.goal_radius,
        reach_eps=args.eps,
    )

    rob = evaluate_formula(cfg, prefer_spatial=not args.no_spatial, strict=args.strict)
    sat = "SAT" if rob >= 0.0 else "UNSAT"
    print(f"robustness={rob:.6g}  ({sat})")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            raise RuntimeError("matplotlib is required for --plot") from e

        trace = robustness_trace(cfg)
        plt.figure()
        plt.plot(range(cfg.T), trace)
        plt.axhline(0.0, linestyle="--")
        plt.title("SpaTiaL demo: predicate robustness over time")
        plt.xlabel("t")
        plt.ylabel("eps - separation")
        plt.tight_layout()
        plt.savefig(args.plot)
        plt.close()
        print(f"wrote plot: {args.plot}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())


__all__ = ["ToyScene", "build_scene", "evaluate_formula", "run_demo"]
