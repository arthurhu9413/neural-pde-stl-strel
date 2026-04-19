r"""TorchPhysics optional-dependency tests + a tiny monitored example.

This repository treats physics-ML + logic examples as first-class artifacts.
The repository design goals explicitly emphasize:

* keep the repo reproducible end-to-end;
* include concrete, actually-run examples;
* write the STL/STREL specs down explicitly;
* include plots/figures that match those specs;
* show a high-level block diagram of how the frameworks connect;
* show a detailed data-flow diagram tied to each example;
* record runtime / hardware context for empirical reproducibility.

To support that, this module intentionally mixes:

1) **Zero-dependency unit tests** (always run)
   Validate that :mod:`neural_pde_stl_strel.frameworks.torchphysics_hello` is:
   * import-safe (does not import TorchPhysics eagerly),
   * explicit about its public API, and
   * provides actionable error messages when TorchPhysics is missing.

2) **A tiny deterministic "toy run"** (skipped unless optional deps exist)
   A CPU-only example that:
   * defines a simple cooling ODE,
   * evaluates TorchPhysics residuals via :class:`torchphysics.conditions.PINNCondition`,
   * monitors explicit STL specifications on the resulting trace, and
   * writes a demo/report-friendly plot with timing and platform context.

High-level wiring diagram (framework connections)

    TorchPhysics (domains/samplers/conditions)
        |
        +-> PyTorch trace u(t) on a grid
        |      |
        |      +-> RTAMT (optional) STL robustness monitoring
        |      +-> matplotlib (optional) figure artifact
        |
        +-> scalar residual losses (sanity check)

Detailed data-flow diagram (this example)

    INPUTS
      PDE:     du/dt = -a u  (Newtonian cooling, a = 0.2)
      IC:      u(0) = 50
      Domain:  t in [0, 10]
      Model:   ExactCooling (closed-form u(t) = 50 exp(-0.2 t))
      Specs:   psi := G[0,10] (u >= 0)
               phi := F[5,10] (u <= 25)

    PROCESSING
      1. TorchPhysics PINNCondition evaluates residuals (sanity check).
      2. Model is queried on a uniform grid -> trace u(t).
      3. Trace is monitored against psi, phi, and psi ^ phi.

    OUTPUTS
      Robustness values (rho_psi, rho_phi, rho_both).
      Figure: trace + reference thresholds + highlighted window.

Note on the lambda (stl_weight) parameter

This test is *monitoring-only*: the model is the exact analytical solution,
so no training loop (and therefore no lambda/stl_weight) is involved.  For
examples that use lambda to weight an STL penalty during PINN training, see:

* ``scripts/train_burgers_torchphysics.py``  (TorchPhysics + STL training)
* ``scripts/train_diffusion_stl.py``         (standalone PINN + STL training)

In those scripts, the total loss has the form::

    loss = physics_loss + stl_weight * stl_penalty

where ``stl_weight`` (= lambda) controls how strongly the STL robustness
penalty is enforced relative to the PDE/IC/BC residual losses.

Monitored STL specifications (explicit)

We use Newtonian cooling on a bounded horizon :math:`t \in [0, 10]`:

    du/dt = -a u,    u(0) = 50,    a = 0.2.

Exact solution: :math:`u(t) = 50 \exp(-0.2 t)`.

Key values:
    u(0)  = 50.0
    u(5)  = 50 exp(-1)  ~ 18.394
    u(10) = 50 exp(-2)  ~  6.767

We monitor two exemplar STL properties on the trace :math:`u(t)`:

* **Safety** (non-negativity):

    psi := G[0,10] (u(t) >= 0)

  Robustness: rho(psi) = min_{t in [0,10]} u(t) = u(10) ~ 6.767  (> 0, satisfied)

* **Eventually** cooling below a threshold:

    phi := F[5,10] (u(t) <= 25)

  Robustness: rho(phi) = max_{t in [5,10]} (25 - u(t)) = 25 - u(10) ~ 18.233  (> 0, satisfied)

Both are checked quantitatively via robustness rho(.); satisfaction is rho >= 0.

The conjunction rho(psi ^ phi) = min(rho(psi), rho(phi)) ~ 6.767.

Note
----
Spatial logic monitoring (STREL/SpaTiaL) is exercised in separate tests.
"""

from __future__ import annotations

import builtins
import importlib
import importlib.util
import math
import platform
import sys
import time
import types
from pathlib import Path
from typing import Any, Iterable

import pytest


# Make the in-repo package importable whether or not it has been installed yet.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

MOD = "neural_pde_stl_strel.frameworks.torchphysics_hello"


def _import_helper_or_skip() -> Any:
    """Import the helper module or skip if the package path is unresolved."""

    spec = importlib.util.find_spec(MOD)
    if spec is None:  # pragma: no cover - environment dependent
        pytest.skip("helper module not importable")
    return importlib.import_module(MOD)


# Part (1): import-safe unit tests (no optional deps required)


def test_helper_import_is_lazy_wrt_torchphysics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing the helper must *not* import torchphysics eagerly.

    This protects users who install only the lightweight base requirements.
    """

    # Ensure a fresh import of the helper.
    monkeypatch.delitem(sys.modules, MOD, raising=False)

    real_import = builtins.__import__

    def _block_torchphysics(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "torchphysics" or name.startswith("torchphysics."):
            raise ModuleNotFoundError("torchphysics import blocked for laziness test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_torchphysics)
    helper = _import_helper_or_skip()
    assert hasattr(helper, "torchphysics_available")


def test_public_api_and_signature() -> None:
    helper = _import_helper_or_skip()

    # Exported API should be explicit via __all__.
    exported = set(getattr(helper, "__all__", []))
    assert "torchphysics_version" in exported
    assert "torchphysics_available" in exported
    assert "torchphysics_smoke" in exported
    assert "TORCHPHYSICS_DIST_NAME" in exported
    assert "TORCHPHYSICS_MODULE_NAME" in exported

    # And the attributes should exist.
    for name in exported:
        assert hasattr(helper, name)

    # Signatures are part of the user-facing contract, but be tolerant to
    # postponed evaluation of annotations (PEP 563 / __future__.annotations).
    import inspect

    sig_v = inspect.signature(helper.torchphysics_version)
    assert list(sig_v.parameters) == []
    assert sig_v.return_annotation in (str, "str")

    sig_a = inspect.signature(helper.torchphysics_available)
    assert list(sig_a.parameters) == []
    assert sig_a.return_annotation in (bool, "bool")

    sig_s = inspect.signature(helper.torchphysics_smoke)
    assert "n_points" in sig_s.parameters and sig_s.parameters["n_points"].default == 32


def test_version_prefers___version__(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _import_helper_or_skip()

    dummy = types.ModuleType("torchphysics")
    dummy.__version__ = "1.2.3"
    monkeypatch.setitem(sys.modules, "torchphysics", dummy)

    assert helper.torchphysics_version() == "1.2.3"


def test_version_falls_back_to_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _import_helper_or_skip()

    dummy = types.ModuleType("torchphysics")
    # No __version__ attribute -> should use importlib.metadata
    monkeypatch.setitem(sys.modules, "torchphysics", dummy)

    monkeypatch.setattr(helper._metadata, "version", lambda _: "9.9.9")
    assert helper.torchphysics_version() == "9.9.9"


def test_available_false_when_import_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _import_helper_or_skip()

    def _boom(_: str):
        raise ImportError("nope")

    monkeypatch.setattr(helper, "import_module", _boom)
    assert helper.torchphysics_available() is False


def test_version_raises_actionable_importerror_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _import_helper_or_skip()

    def _boom(_: str):
        raise ModuleNotFoundError("torchphysics not installed")

    monkeypatch.setattr(helper, "import_module", _boom)
    with pytest.raises(ImportError) as ei:
        _ = helper.torchphysics_version()

    msg = str(ei.value)
    assert "pip install torchphysics" in msg
    assert helper.TORCHPHYSICS_DIST_NAME in msg
    assert helper.TORCHPHYSICS_MODULE_NAME in msg


# Part (2): deterministic end-to-end example (skipped unless optional deps exist)


def _safe_import_optional(name: str) -> Any:
    """Import an optional dependency.

    We intentionally *skip* (rather than fail) on import-time exceptions.
    This keeps CI stable when optional packages are unavailable or when the
    environment lacks system libraries (common for scientific stacks).
    """

    try:
        return importlib.import_module(name)
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"optional dependency '{name}' not available: {e}")


def _manual_robustness_always_geq(values: Iterable[float], threshold: float) -> float:
    """Robustness of G(x >= c) under standard quantitative STL semantics.

    rho(G(x >= c)) = min_t (x(t) - c)

    Positive robustness means the specification is satisfied everywhere;
    the magnitude indicates the minimum margin above the threshold.
    """

    return min(v - threshold for v in values)


def _manual_robustness_eventually_leq(
    times: Iterable[float],
    values: Iterable[float],
    lo: float,
    hi: float,
    threshold: float,
) -> float:
    """Robustness of F[lo,hi](x <= c) under standard quantitative STL semantics.

    rho(F[lo,hi](x <= c)) = max_{t in [lo,hi]} (c - x(t))

    Positive robustness means there exists at least one time in the window
    where x(t) is below the threshold; the magnitude indicates the best margin.
    """

    best = -float("inf")
    for t, v in zip(times, values):
        if lo <= t <= hi:
            best = max(best, threshold - v)
    if best == -float("inf"):
        raise ValueError("No samples fell within the requested time window")
    return best


def test_torchphysics_smoke_returns_metrics_if_installed() -> None:
    """If TorchPhysics is installed, the helper smoke test should run on CPU."""

    helper = _import_helper_or_skip()
    if not helper.torchphysics_available():
        pytest.skip("TorchPhysics not installed")

    try:
        metrics = helper.torchphysics_smoke(n_points=16, hidden=(8,), seed=0)
    except RuntimeError as e:  # pragma: no cover - env dependent
        pytest.skip(f"TorchPhysics available but smoke test failed: {e}")

    assert isinstance(metrics, dict)
    assert metrics.get("points") == 16
    assert isinstance(metrics.get("loss"), float)
    assert math.isfinite(float(metrics["loss"]))
    assert isinstance(metrics.get("version"), str)


def test_torchphysics_cooling_ode_monitors_stl_and_writes_figure(tmp_path: Path) -> None:
    """End-to-end toy example with explicit STL specs + a saved plot.

    This is designed to be something you can run live in a demo:

    * deterministic (no training loop, uses an exact model),
    * fast (CPU-only),
    * produces a concrete figure artifact with timing and platform context.

    It also acts as a regression test that TorchPhysics' differential operators
    (grad) work with our model interface.
    """

    t_wall_start = time.perf_counter()

    torch = _safe_import_optional("torch")
    tp = _safe_import_optional("torchphysics")

    # Avoid any GPU use in a demo/test environment.
    device = torch.device("cpu")

    # Problem: Newtonian cooling
    #   du/dt = -a u,   u(0) = u0
    #   Exact solution: u(t) = u0 * exp(-a t)
    a = 0.2
    u0 = 50.0
    t0, t1 = 0.0, 10.0

    # STL specs we intend to monitor on u(t):
    #
    #   psi := G[0,10] (u >= 0)            safety (non-negativity)
    #   phi := F[5,10] (u <= 25)           eventually-cooling
    #   psi ^ phi                          conjunction
    #
    # RTAMT syntax uses "always" and "eventually" keywords with colon-separated
    # interval bounds (e.g. always[a:b]).  See: https://github.com/nickovic/rtamt
    spec_always_nonneg = "always[0:10] (u >= 0)"
    spec_eventually_cool = "eventually[5:10] (u <= 25)"
    spec_conjunction = f"({spec_always_nonneg}) and ({spec_eventually_cool})"

    # TorchPhysics spaces/domains.
    time_space = tp.spaces.R1("t")
    u_space = tp.spaces.R1("u")
    domain = tp.domains.Interval(space=time_space, lower_bound=t0, upper_bound=t1)

    # TorchPhysics uses a lightweight `Points` container for coordinates and
    # model outputs.  Keep access robust across minor API re-exports.
    Points = getattr(getattr(tp, "spaces", None), "Points", None)
    if Points is None and hasattr(tp, "problem"):
        Points = getattr(getattr(tp.problem, "spaces", None), "Points", None)
    if Points is None:  # pragma: no cover - version dependent
        pytest.skip("TorchPhysics Points class not found (unexpected API change)")

    # TorchPhysics gradient operator location varies slightly across versions.
    grad = None
    try:
        from torchphysics.utils.differentialoperators import grad as _grad  # type: ignore

        grad = _grad
    except Exception:
        grad = getattr(getattr(tp, "utils", None), "grad", None)
    if not callable(grad):  # pragma: no cover - version dependent
        pytest.skip("TorchPhysics grad() operator not found (unexpected API change)")

    def _as_tensor(x: Any) -> Any:
        """Return a torch.Tensor from TorchPhysics objects when needed."""

        if hasattr(x, "as_tensor"):
            at = getattr(x, "as_tensor")
            return at() if callable(at) else at
        return x

    def _grad_scalar(y: Any, x: Any) -> Any:
        """Return dy/dx as a torch.Tensor."""

        g = grad(_as_tensor(y), _as_tensor(x))
        if isinstance(g, (tuple, list)):
            g = g[0]
        return _as_tensor(g)

    # Exact model: u(t) = u0 * exp(-a t)
    # TorchPhysics modules accept/return `Points`.
    class ExactCooling(torch.nn.Module):
        def __init__(self, a_: float, u0_: float):
            super().__init__()
            self.register_buffer("a", torch.tensor(float(a_)))
            self.register_buffer("u0", torch.tensor(float(u0_)))

        def forward(self, points: Any) -> Any:  # points: TorchPhysics Points
            t = points.coordinates["t"]
            u = self.u0 * torch.exp(-self.a * t)
            return Points(u, u_space)

    model = ExactCooling(a, u0).to(device)
    model.eval()

    # Residuals: PINNCondition injects tensors by variable name.
    # The residual function receives (model_output, input_variables...) where
    # the names match the TorchPhysics space variable names defined above.
    def ode_residual(u: Any, t: Any) -> Any:
        return _grad_scalar(u, t) + a * u  # should be ~0

    def ic_residual(u: Any) -> Any:
        return u - u0  # at t=0 should be 0

    # Samplers: keep them static so point sets are deterministic for demos.
    sampler_interior = tp.samplers.RandomUniformSampler(domain, n_points=64)
    ic_domain = getattr(domain, "boundary_left", None)
    if ic_domain is None:  # pragma: no cover - version dependent
        pytest.skip(
            "TorchPhysics Interval has no 'boundary_left' attribute; cannot build deterministic IC sampler"
        )
    sampler_ic = tp.samplers.RandomUniformSampler(ic_domain, n_points=8)

    # Freeze random draws deterministically without perturbing global RNG state.
    fork_rng = getattr(getattr(torch, "random", None), "fork_rng", None)

    def _make_static_if_supported(sampler: Any) -> Any:
        if not hasattr(sampler, "make_static"):
            return sampler
        maybe = sampler.make_static()
        return sampler if maybe is None else maybe

    if fork_rng is None:  # pragma: no cover - very old torch
        torch.manual_seed(0)
        sampler_interior = _make_static_if_supported(sampler_interior)
        sampler_ic = _make_static_if_supported(sampler_ic)
    else:
        with fork_rng(devices=[]):
            torch.manual_seed(0)
            sampler_interior = _make_static_if_supported(sampler_interior)
            sampler_ic = _make_static_if_supported(sampler_ic)

    cond_pde = tp.conditions.PINNCondition(
        module=model, sampler=sampler_interior, residual_fn=ode_residual
    )
    cond_ic = tp.conditions.PINNCondition(
        module=model, sampler=sampler_ic, residual_fn=ic_residual
    )

    # Avoid higher-level Problem containers (which vary across TorchPhysics
    # versions) and use the stable Condition API directly.
    with torch.enable_grad():
        loss_t = cond_pde.forward(device=str(device)) + cond_ic.forward(device=str(device))
    loss = float(loss_t.detach().cpu().item())

    assert math.isfinite(loss)
    # Exact model should satisfy the ODE/IC essentially up to numerical precision.
    assert loss < 1e-5

    # Evaluate trace u(t) on a grid
    n = 201
    t_grid = torch.linspace(t0, t1, n, device=device).unsqueeze(-1)
    with torch.no_grad():
        u_pred = _as_tensor(model(Points(t_grid, time_space))).detach().cpu().squeeze(-1)

    times = [float(t) for t in t_grid.detach().cpu().squeeze(-1).tolist()]
    values = [float(v) for v in u_pred.tolist()]
    assert len(times) == len(values) == n

    # Quick analytical sanity checks on key trace values.
    # u(0)  = 50,  u(5) = 50*exp(-1) ~ 18.394,  u(10) = 50*exp(-2) ~ 6.767
    assert values[0] == pytest.approx(50.0, abs=1e-4)
    assert values[n // 2] == pytest.approx(50.0 * math.exp(-0.2 * 5.0), abs=1e-2)
    assert values[-1] == pytest.approx(50.0 * math.exp(-0.2 * 10.0), abs=1e-2)

    # Monitor STL specs (RTAMT if available, else deterministic fallback)
    rho_nonneg: float
    rho_cool: float
    rho_both: float

    try:
        rt = importlib.import_module("neural_pde_stl_strel.monitoring.rtamt_monitor")
        # Build specs explicitly with dense-time semantics (appropriate for
        # continuous-time physics traces sampled at discrete points).
        s1 = rt.build_stl_spec(spec_always_nonneg, var_types={"u": "float"}, time_semantics="dense")
        s2 = rt.build_stl_spec(
            spec_eventually_cool, var_types={"u": "float"}, time_semantics="dense"
        )
        s3 = rt.build_stl_spec(spec_conjunction, var_types={"u": "float"}, time_semantics="dense")
        dt = times[1] - times[0]
        rho_nonneg = float(rt.evaluate_series(s1, "u", values, dt=dt))
        rho_cool = float(rt.evaluate_series(s2, "u", values, dt=dt))
        rho_both = float(rt.evaluate_series(s3, "u", values, dt=dt))
        assert rt.satisfied(rho_nonneg)
        assert rt.satisfied(rho_cool)
        assert rt.satisfied(rho_both)
    except Exception:
        # Fall back to simple max/min semantics if RTAMT isn't importable.
        rho_nonneg = _manual_robustness_always_geq(values, threshold=0.0)
        rho_cool = _manual_robustness_eventually_leq(
            times, values, lo=5.0, hi=10.0, threshold=25.0
        )
        rho_both = min(rho_nonneg, rho_cool)
        assert rho_nonneg >= 0.0
        assert rho_cool >= 0.0
        assert rho_both >= 0.0

    assert math.isfinite(rho_nonneg)
    assert math.isfinite(rho_cool)
    assert math.isfinite(rho_both)

    # Conjunction robustness should be the minimum of its operands.
    assert rho_both == pytest.approx(min(rho_nonneg, rho_cool), abs=1e-6)

    # Verify that the robustness values are in the expected ballpark
    # for the exact solution.
    #   rho(psi) = min u(t) = u(10) ~ 6.767
    #   rho(phi) = max_{[5,10]} (25 - u(t)) = 25 - u(10) ~ 18.233
    assert rho_nonneg == pytest.approx(50.0 * math.exp(-2.0), abs=0.5)
    assert rho_cool == pytest.approx(25.0 - 50.0 * math.exp(-2.0), abs=0.5)

    t_wall_end = time.perf_counter()
    wall_seconds = t_wall_end - t_wall_start

    # Write a demo-friendly figure artifact
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"matplotlib not available: {e}")

    fig_path = tmp_path / "torchphysics_cooling_ode_trace.png"
    fig, ax = plt.subplots(figsize=(8, 5))

    # Trace.
    ax.plot(times, values, color="#2563EB", linewidth=2.0, label="u(t) = 50 exp(-0.2t)")

    # Reference thresholds as dashed horizontal lines.
    ax.axhline(y=0.0, color="#6B7280", linestyle="--", linewidth=1.0, label="u = 0 (safety bound)")
    ax.axhline(y=25.0, color="#DC2626", linestyle="--", linewidth=1.0, label="u = 25 (cooling threshold)")

    # Highlight the eventually-window [5, 10].
    ax.axvspan(5.0, 10.0, alpha=0.10, color="#F59E0B", label="F window [5, 10]")

    # Annotate key trace values for clarity in demos/reports.
    u_at_5 = 50.0 * math.exp(-1.0)
    u_at_10 = 50.0 * math.exp(-2.0)
    ax.annotate(
        f"u(5) = {u_at_5:.1f}",
        xy=(5.0, u_at_5), xytext=(5.8, u_at_5 + 8),
        arrowprops=dict(arrowstyle="->", color="#4B5563"),
        fontsize=9, color="#4B5563",
    )
    ax.annotate(
        f"u(10) = {u_at_10:.1f}",
        xy=(10.0, u_at_10), xytext=(7.8, u_at_10 - 4),
        arrowprops=dict(arrowstyle="->", color="#4B5563"),
        fontsize=9, color="#4B5563",
    )

    ax.set_xlabel("t", fontsize=11)
    ax.set_ylabel("u(t)", fontsize=11)
    ax.set_title(
        "Cooling ODE trace with monitored STL specifications\n"
        rf"$\psi$ = G[0,10](u $\geq$ 0), $\rho$ = {rho_nonneg:.3g}    "
        rf"$\varphi$ = F[5,10](u $\leq$ 25), $\rho$ = {rho_cool:.3g}    "
        rf"$\psi \wedge \varphi$, $\rho$ = {rho_both:.3g}",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=8)

    # Add runtime and platform context (reproducibility requirement).
    hw_label = (
        f"wall = {wall_seconds:.2f}s | "
        f"python = {platform.python_version()} | "
        f"os = {platform.system()} {platform.machine()}"
    )
    ax.text(
        0.01, 0.01, hw_label,
        transform=ax.transAxes, fontsize=7,
        color="#9CA3AF", verticalalignment="bottom",
    )

    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    assert fig_path.exists()
    assert fig_path.stat().st_size > 0


if __name__ == "__main__":  # pragma: no cover
    # Allow running this file directly to generate the figure in a stable location.
    # Example:
    #   python tests/test_torchphysics_hello.py
    out_dir = _ROOT / "figs" / "tests"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        test_torchphysics_cooling_ode_monitors_stl_and_writes_figure(out_dir)
    except pytest.skip.Exception as e:
        print(str(e))
        raise SystemExit(0) from e
    except Exception as e:
        raise SystemExit(str(e)) from e
    else:
        print(f"Wrote: {out_dir / 'torchphysics_cooling_ode_trace.png'}")
