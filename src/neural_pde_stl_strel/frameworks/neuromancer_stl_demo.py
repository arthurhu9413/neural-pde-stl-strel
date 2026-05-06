from __future__ import annotations

"""neural_pde_stl_strel.frameworks.neuromancer_stl_demo

Neuromancer × STL demo (toy signal)

This module is a *minimal, fast, CPU-friendly* example of how Signal Temporal
Logic (STL) properties can be:

1) **enforced during training** (as a differentiable penalty or a constraint), and
2) **audited after training** (as an offline robustness score).

The demo deliberately stays at the level of a 1-D time series so that the
*data flow* is unmistakable. In a PINN/NODE/PDE setting, you would replace the
toy signal ``y_hat(t)`` with the state/field predicted by the physics-ML
framework and apply the *same monitoring logic* to the resulting trajectories
(and, for spatial STL/STREL, to the field sampled over space).

STL properties used
We focus on two canonical templates that cover a large portion of the
feedback ("bounds" and "eventually" properties):

Safety bound (Always)
^^^^^^^^^^^^^^^^^^^^^

    φ_safe := G ( y(t) <= bound )

Under discrete-time semantics on a finite trace ``y[0:N]``, the robustness is:

    ρ(φ_safe, y) = min_{k=0..N-1} (bound - y[k])

- ``ρ > 0`` means the spec is satisfied with margin.
- ``ρ < 0`` means the spec is violated; ``-ρ`` is the violation margin.

Eventual target (Eventually)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    φ_evt := F[a,b] ( y(t) <= evt_bound )

Robustness on a finite trace is:

    ρ(φ_evt, y) = max_{k=a..b} (evt_bound - y[k])

This corresponds to statements like the cooling example
"eventually after some time, the temperature is below a threshold".

We provide both:

- a differentiable surrogate via log-sum-exp soft-min/soft-max (for training), and
- an exact closed-form robustness (for offline reporting).

High-level block diagram
The demo follows the same pattern you want for PINNs/NODEs in physics-ML:

    (t, y_true) ──▶ model f_θ (PyTorch or Neuromancer Node) ──▶ y_hat(t)
         │                                                      │
         │                                                      ├─▶ STL monitor (offline) ─▶ robustness ρ
         │                                                      │
         └── loss = MSE(y_hat, y_true) + λ · STL_penalty(y_hat) ◀── backprop

Here ``λ`` is the STL weight. In this file it is stored as ``DemoConfig.weight``
(the STL penalty weight).

What the demo returns
``train_demo(cfg)`` returns metrics for:

- a plain **PyTorch** training loop with an STL penalty, and
- an optional **Neuromancer** variant using ``PenaltyLoss`` and an inequality
  constraint (if Neuromancer is installed).

The metrics are chosen so it is easy to document *what was run* and *what was
monitored*: MSE, max/mean pointwise violation, and robustness values.

Public API (used by tests)
- :class:`DemoConfig`
- :func:`train_demo`
- :func:`stl_violation`
- :func:`stl_offline_robustness`

Optional dependencies
- ``neuromancer``: enables the Neuromancer training path.
- ``rtamt``: enables an offline monitor cross-check for the safety property.

References
- Neuromancer: https://github.com/pnnl/neuromancer
- RTAMT:       https://github.com/nickovic/rtamt
"""

import math
import time
from dataclasses import dataclass
from typing import Any

# Torch is required for this demo.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

# Optional STL monitor (offline cross-check).
try:  # pragma: no cover
    import rtamt  # type: ignore
except Exception:  # pragma: no cover
    rtamt = None  # type: ignore[assignment]


# Configuration


@dataclass(frozen=True, slots=True)
class DemoConfig:
    """Configuration for the toy demo.

    Notes
    -----
    - ``weight`` is the STL penalty/constraint scaling factor (λ).
    - ``soft_beta`` controls the softness of the differentiable surrogate.
    - The optional eventual property is *disabled by default* (set
      ``eventually_bound`` to enable it).
    """

    # Data / optimization
    n: int = 256
    epochs: int = 200
    lr: float = 1e-3
    device: str = "cpu"
    seed: int = 7

    # Model architecture (shared across both training paths)
    hidden: int = 64
    depth: int = 2  # number of hidden layers

    # Safety spec parameters (Always)
    bound: float = 0.8
    weight: float = 100.0
    soft_beta: float = 25.0

    # Optional eventual spec parameters (Eventually)
    eventually_bound: float | None = None
    eventually_window: tuple[int, int] | None = None  # inclusive indices (a, b)
    eventually_weight: float = 0.0  # set >0 to include as a training penalty

    # Loss design for the safety spec
    use_soft_stl_in_loss: bool = True

    # Neuromancer toggles (kept tiny so it is safe to run in CI)
    nm_batch_size: int = 64
    nm_epochs: int = 50
    nm_lr: float = 5e-4

    # Optional artifact output (e.g., for slides/report figures)
    plot_path: str | None = None

    @property
    def lambda_stl(self) -> float:
        """Alias for ``weight`` (λ) to match paper/email terminology."""

        return self.weight


# Utilities


def _require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for this demo, but is not available.")


def _set_seed(seed: int) -> None:
    """Best-effort seeding for deterministic behavior."""

    _require_torch()
    torch.manual_seed(seed)


def _make_data(n: int, *, device: str) -> dict[str, "torch.Tensor"]:
    """Generate a deterministic toy signal dataset.

    We default to a sine wave because it is bounded and makes it easy to
    interpret the safety bound ``y <= bound``.
    """

    _require_torch()
    t = torch.linspace(0.0, 2.0 * math.pi, n, device=device).reshape(n, 1)
    y_true = torch.sin(t)
    return {"t": t, "y_true": y_true}


def _build_mlp(*, insize: int, outsize: int, hidden: int, depth: int) -> "nn.Module":
    """Simple MLP used by both the PyTorch and Neuromancer paths."""

    _require_torch()
    if depth < 0:
        raise ValueError(f"depth must be >= 0, got {depth}")

    if depth == 0:
        return nn.Sequential(nn.Linear(insize, outsize))

    layers: list[nn.Module] = [nn.Linear(insize, hidden), nn.Tanh()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
    layers.append(nn.Linear(hidden, outsize))
    return nn.Sequential(*layers)


def _normalize_inclusive_window(window: tuple[int, int], *, n: int) -> tuple[int, int]:
    """Validate/normalize an inclusive index window ``(a, b)`` for length ``n``."""

    a, b = window
    if n <= 0:
        raise ValueError("n must be positive")
    if a < 0 or b < 0:
        raise ValueError(f"window indices must be non-negative, got {window}")
    if a > b:
        raise ValueError(f"window must satisfy a <= b, got {window}")
    if b >= n:
        raise ValueError(f"window end b={b} must be < n={n}")
    return a, b


# STL helpers


def stl_violation(u: "torch.Tensor", bound: float) -> "torch.Tensor":
    """Elementwise violation of the predicate ``u <= bound``.

    Returns
    -------
    torch.Tensor
        ``(u - bound)+`` (ReLU), same shape as ``u``.

    Notes
    -----
    This is the atomic predicate violation used by many STL penalties.
    Temporal operators (e.g., ``G``/``F``) are handled by aggregation.
    """

    _require_torch()
    return torch.relu(u - bound)


def _softmin(x: "torch.Tensor", *, beta: float = 25.0, dim: int | None = None) -> "torch.Tensor":
    """Smooth approximation of ``min(x)`` using log-sum-exp.

    Definition
    ``softmin_beta(x) = -(1/beta) * logsumexp(-beta * x)``

    As ``beta -> +∞``, ``softmin_beta(x) -> min(x)``.

    Parameters
    x:
        Input tensor.
    beta:
        Inverse temperature (larger = sharper min).
    dim:
        Dimension to reduce. If ``None``, ``x`` is flattened.
    """

    _require_torch()
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    if dim is None:
        x = x.reshape(-1)
        dim = 0
    return -(torch.logsumexp(-beta * x, dim=dim) / beta)


def _softmax(x: "torch.Tensor", *, beta: float = 25.0, dim: int | None = None) -> "torch.Tensor":
    """Smooth approximation of ``max(x)`` using log-sum-exp.

    Definition
    ``softmax_beta(x) = (1/beta) * logsumexp(beta * x)``

    As ``beta -> +∞``, ``softmax_beta(x) -> max(x)``.

    Parameters are the same as :func:`_softmin`.
    """

    _require_torch()
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    if dim is None:
        x = x.reshape(-1)
        dim = 0
    return torch.logsumexp(beta * x, dim=dim) / beta


def _stl_always_soft_robustness(u: "torch.Tensor", bound: float, *, beta: float = 25.0) -> "torch.Tensor":
    """Differentiable robustness surrogate for ``G (u <= bound)``.

    Exact discrete-time robustness for the safety property is:

    ``min_t (bound - u[t])``.

    We approximate the ``min`` with :func:`_softmin`.

    Returns
    -------
    torch.Tensor
        A scalar tensor (robustness surrogate). Positive is satisfied.

    Notes
    -----
    This function is intentionally module-private but is used by unit tests,
    so the name/signature should remain stable.
    """

    _require_torch()
    r = bound - u.reshape(-1)  # predicate robustness per sample
    return _softmin(r, beta=beta, dim=0)


def _stl_eventually_soft_robustness(
    u: "torch.Tensor",
    bound: float,
    *,
    beta: float = 25.0,
    window: tuple[int, int] | None = None,
) -> "torch.Tensor":
    """Differentiable robustness surrogate for ``F[a,b] (u <= bound)``.

    Exact robustness is ``max_{t in [a,b]} (bound - u[t])``.

    We approximate the ``max`` with :func:`_softmax`.
    """

    _require_torch()
    u_flat = u.reshape(-1)
    r = bound - u_flat

    if window is not None:
        a, b = _normalize_inclusive_window(window, n=int(u_flat.numel()))
        r = r[a : b + 1]

    return _softmax(r, beta=beta, dim=0)


def stl_offline_robustness(u: "torch.Tensor", bound: float) -> float:
    """Exact robustness for ``G (u <= bound)`` on a finite discrete-time trace.

    The mathematically exact robustness is the closed form:

        ``rho = min_t (bound - u[t])``

    If RTAMT is installed, we optionally cross-check that the library monitor
    agrees with the closed form and use it only when it matches (to avoid
    surprising discrepancies across RTAMT versions).
    """

    _require_torch()
    u_flat = u.reshape(-1)
    rho_closed = float((bound - u_flat).min().item())

    if rtamt is None:
        return rho_closed

    # Optional RTAMT cross-check.
    try:  # pragma: no cover
        spec = rtamt.StlDiscreteTimeSpecification()
        spec.declare_var("y", "float")
        n = int(u_flat.numel())
        spec.spec = f"always[0,{max(0, n - 1)}](y <= {float(bound)})"
        spec.parse()
        spec.pastify()  # required for future operators in online evaluation

        rob: float | None = None
        for k, val in enumerate(u_flat.tolist()):
            rob = float(spec.update(k, [("y", float(val))]))

        if rob is None or not math.isfinite(rob):
            return rho_closed

        if abs(rob - rho_closed) <= 1e-6 * (1.0 + abs(rho_closed)):
            return float(rob)

        return rho_closed

    except Exception:
        return rho_closed


def stl_eventually_offline_robustness(
    u: "torch.Tensor",
    bound: float,
    *,
    window: tuple[int, int] | None = None,
) -> float:
    """Exact robustness for ``F[a,b] (u <= bound)`` on a finite discrete-time trace."""

    _require_torch()
    u_flat = u.reshape(-1)
    r = bound - u_flat

    if window is not None:
        a, b = _normalize_inclusive_window(window, n=int(u_flat.numel()))
        r = r[a : b + 1]

    return float(r.max().item())


# PyTorch baseline


def _maybe_plot(cfg: DemoConfig, data: dict[str, "torch.Tensor"], y_hat: "torch.Tensor") -> None:
    """Optionally save a simple time-series plot.

    Plotting is kept optional
    so the repository remains lightweight.
    """

    if cfg.plot_path is None:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return

    t = data["t"].detach().cpu().reshape(-1)
    y_true = data["y_true"].detach().cpu().reshape(-1)
    y_pred = y_hat.detach().cpu().reshape(-1)

    max_viol = float(torch.relu(y_pred - float(cfg.bound)).max().item())
    rho_safe = stl_offline_robustness(y_pred, cfg.bound)

    title = f"Toy STL safety: G(y<=bound),  rho={rho_safe:+.3f},  max_violation={max_viol:.3f}"

    if cfg.eventually_bound is not None:
        rho_evt = stl_eventually_offline_robustness(y_pred, float(cfg.eventually_bound), window=cfg.eventually_window)
        if cfg.eventually_window is None:
            title += f" |  rho_evt(F(y<=evt))={rho_evt:+.3f}"
        else:
            a, b = cfg.eventually_window
            title += f" |  rho_evt(F[{a},{b}](y<=evt))={rho_evt:+.3f}"

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t, y_true, label="y_true")
    ax.plot(t, y_pred, label="y_hat")
    ax.axhline(float(cfg.bound), linestyle="--", label="bound")
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(cfg.plot_path, dpi=200)
    plt.close(fig)


def _train_pytorch(cfg: DemoConfig, data: dict[str, "torch.Tensor"]) -> dict[str, float]:
    """Train the toy model in plain PyTorch with an STL penalty."""

    _require_torch()
    _set_seed(cfg.seed)

    device = torch.device(cfg.device)
    net = _build_mlp(insize=1, outsize=1, hidden=cfg.hidden, depth=cfg.depth).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    t = data["t"]
    y_true = data["y_true"]

    start = time.perf_counter()
    for _ in range(cfg.epochs):
        opt.zero_grad(set_to_none=True)
        y_hat = net(t)
        mse = F.mse_loss(y_hat, y_true)

        # Safety penalty: encourage ρ(G(y<=bound), y_hat) >= 0.
        if cfg.use_soft_stl_in_loss:
            rho_safe_soft = _stl_always_soft_robustness(y_hat, cfg.bound, beta=cfg.soft_beta)
            safe_penalty = F.relu(-rho_safe_soft)  # penalize only violations
        else:
            # A simpler baseline: average pointwise predicate violation.
            safe_penalty = stl_violation(y_hat, cfg.bound).mean()

        loss = mse + cfg.lambda_stl * safe_penalty

        # Optional eventual penalty: encourage ρ(F[a,b](y<=evt_bound), y_hat) >= 0.
        if cfg.eventually_bound is not None and cfg.eventually_weight > 0.0:
            rho_evt_soft = _stl_eventually_soft_robustness(
                y_hat,
                float(cfg.eventually_bound),
                beta=cfg.soft_beta,
                window=cfg.eventually_window,
            )
            evt_penalty = F.relu(-rho_evt_soft)
            loss = loss + cfg.eventually_weight * evt_penalty

        loss.backward()
        opt.step()

    train_time = time.perf_counter() - start

    with torch.no_grad():
        y_hat = net(t)
        mse_val = float(F.mse_loss(y_hat, y_true).item())
        viol = stl_violation(y_hat, cfg.bound)
        max_viol = float(viol.max().item())
        mean_viol = float(viol.mean().item())
        rho_safe = stl_offline_robustness(y_hat, cfg.bound)

    metrics: dict[str, float] = {
        "final_mse": mse_val,
        "max_violation": max_viol,
        "mean_violation": mean_viol,
        "stl_robustness": float(rho_safe),
        "train_seconds": float(train_time),
    }

    if cfg.eventually_bound is not None:
        rho_evt = stl_eventually_offline_robustness(y_hat, float(cfg.eventually_bound), window=cfg.eventually_window)
        metrics["eventually_robustness"] = float(rho_evt)

    _maybe_plot(cfg, data, y_hat)

    return metrics


# Neuromancer variant (optional)


def _train_neuromancer(cfg: DemoConfig, data: dict[str, "torch.Tensor"]) -> dict[str, float] | None:
    """Train the toy model using Neuromancer's symbolic API.

    Returns ``None`` if Neuromancer is not installed or if its API differs from
    what we expect. This keeps the repository portable.

    Input/output mapping:

    Inputs
      - data (t, y_true)
      - network architecture (Neuromancer blocks.MLP)
      - safety specification (bound, λ)

    Output
      - trained parameters (within the Neuromancer ``Problem`` / ``Node``) and
        evaluation metrics.
    """

    try:
        import neuromancer as nm  # type: ignore
        from neuromancer.constraint import variable  # type: ignore
        from neuromancer.dataset import DictDataset  # type: ignore
        from neuromancer.loss import PenaltyLoss  # type: ignore
        from neuromancer.modules import blocks  # type: ignore
        from neuromancer.problem import Problem  # type: ignore
        from neuromancer.system import Node  # type: ignore
    except Exception:
        return None

    _require_torch()
    _set_seed(cfg.seed)

    device = torch.device(cfg.device)

    try:
        # Resolve Neuromancer's "linear" map if it exists; otherwise omit it.
        linear_map = None
        try:
            linear_map = nm.slim.maps["linear"]  # type: ignore[attr-defined]
        except Exception:
            linear_map = None

        mlp_kwargs: dict[str, Any] = {
            "insize": 1,
            "outsize": 1,
            "hsizes": [cfg.hidden] * max(1, cfg.depth),
            "nonlin": nn.Tanh,
        }
        if linear_map is not None:
            mlp_kwargs["linear_map"] = linear_map

        func = blocks.MLP(**mlp_kwargs)
        node = Node(func, input_keys=["t"], output_keys=["y_hat"], name="regressor")

        # Symbolic variables (names must match dataset keys / node output keys).
        y_hat = variable("y_hat")
        y_true = variable("y_true")

        # Objective: supervised fit.
        try:
            mse_obj = ((y_hat - y_true) ** 2).mean().minimize(weight=1.0, name="mse")
        except Exception:
            mse_obj = ((y_hat - y_true) ** 2).minimize(weight=1.0, name="mse")

        # Constraint: pointwise safety bound.
        # Neuromancer's Constraint supports weighting via multiplication.
        safety = cfg.lambda_stl * (y_hat <= float(cfg.bound))

        loss = PenaltyLoss(objectives=[mse_obj], constraints=[safety])
        problem = Problem(nodes=[node], loss=loss).to(device)

        # Dataset / loader.
        train_data = DictDataset({"t": data["t"], "y_true": data["y_true"]}, name="train")
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=min(cfg.nm_batch_size, len(train_data)),
            shuffle=False,
            num_workers=0,
            collate_fn=train_data.collate_fn,
        )

        optimizer = torch.optim.Adam(problem.parameters(), lr=cfg.nm_lr)

        start = time.perf_counter()
        for _ in range(cfg.nm_epochs):
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)

                out: Any = problem(batch)

                # Neuromancer versions differ: sometimes `problem(batch)` is a dict
                # with a `loss` key, sometimes it is the scalar loss tensor.
                if isinstance(out, dict):
                    loss_tensor = out.get("loss") or out.get("train_loss") or out.get("total_loss")
                    if loss_tensor is None:
                        raise RuntimeError("Neuromancer Problem did not return a loss tensor.")
                else:
                    loss_tensor = out

                if not isinstance(loss_tensor, torch.Tensor):
                    raise RuntimeError("Neuromancer loss was not a torch.Tensor.")

                loss_tensor.backward()
                optimizer.step()

        train_time = time.perf_counter() - start

        # Final metrics (evaluate on the full grid).
        with torch.no_grad():
            y_pred = node({"t": data["t"]})["y_hat"]
            mse_val = float(F.mse_loss(y_pred, data["y_true"]).item())
            viol = stl_violation(y_pred, cfg.bound)
            max_viol = float(viol.max().item())
            mean_viol = float(viol.mean().item())
            rho_safe = stl_offline_robustness(y_pred, cfg.bound)

        metrics: dict[str, float] = {
            "final_mse": mse_val,
            "max_violation": max_viol,
            "mean_violation": mean_viol,
            "stl_robustness": float(rho_safe),
            "train_seconds": float(train_time),
        }

        if cfg.eventually_bound is not None:
            rho_evt = stl_eventually_offline_robustness(
                y_pred, float(cfg.eventually_bound), window=cfg.eventually_window,
            )
            metrics["eventually_robustness"] = float(rho_evt)

        return metrics

    except Exception:
        # The Neuromancer path is optional; never fail the repository because of
        # an upstream API mismatch.
        return None


# Public API


def train_demo(cfg: DemoConfig) -> dict[str, dict[str, float] | None]:
    """Run the PyTorch demo and (optionally) the Neuromancer demo."""

    _require_torch()

    device = str(torch.device(cfg.device))
    data = _make_data(cfg.n, device=device)

    metrics_pt = _train_pytorch(cfg, data)
    metrics_nm = _train_neuromancer(cfg, data)

    return {"pytorch": metrics_pt, "neuromancer": metrics_nm}


__all__ = [
    "DemoConfig",
    "train_demo",
    "stl_violation",
    "stl_offline_robustness",
]
