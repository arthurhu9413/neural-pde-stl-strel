#!/usr/bin/env python3
"""scripts/train_neuromancer_stl.py

Toy Neuromancer/STL demo: learn a 1D signal while enforcing a safety bound.

This file is intentionally **self-contained** and **CPU-friendly**. It is meant to
be easy to read in a meeting, easy to run on a laptop, and easy to cite in a
report.

Summary
**Inputs**
  1) A time grid and ground-truth signal (t_i, y_i)
  2) A model family f_θ (MLP)
  3) An STL specification (written explicitly below)
  4) A penalty weight λ (CLI: --stl-weight)

**Output**
  - trained weights (optional checkpoint)
  - plots (optional)
  - a JSON record with: configuration, environment snapshot, STL formulas, and metrics

STL specifications (written explicitly)
Primary safety guard (trained + monitored):
  φ_safe := G ( y(t) <= U_max )

We enforce φ_safe during training via a simple differentiable penalty:
  L_total(θ) = MSE(f_θ(t), y(t)) + λ · mean_t [ max(0, f_θ(t) - U_max) ]

We also report the *hard* (classical) robustness of φ_safe on the final trajectory:
  ρ_safe = min_t (U_max - f_θ(t))
  φ_safe is satisfied  <=>  ρ_safe >= 0

Optional additional (monitor-only) specs (computed on the final trajectory only):
  φ_ev_below := F_[t0,t1] ( y(t) <= c_below )
    robustness: ρ = max_{t∈[t0,t1]} (c_below - y(t))

  φ_ev_above := F_[t0,t1] ( y(t) >= c_above )
    robustness: ρ = max_{t∈[t0,t1]} (y(t) - c_above)

These "eventually" examples are included because they illustrate the qualitative
difference between safety (always) and reachability (eventually) properties.

Data-flow block diagram (implementation-level)
          (t_i) ───────────────┐
                               v
                           f_θ(t) ──> y_hat(t)
                               │          │
                               │          ├─ MSE(y_hat, y_true)
                               │          │
                               │          └─ STL penalty: max(0, y_hat - U_max)
                               │
                               └──────> SGD/Adam update of θ

After training:
  y_hat(t) ──> monitor φ_safe (and optional φ_ev_*) ──> robustness scores / plots

Notes on dependencies
- Neuromancer is an *optional* dependency. If it is not installed, the script
  still runs the pure PyTorch path and records `neuromancer=null` in the JSON.
- RTAMT is an *optional* dependency. If missing, RTAMT robustness fields are null.
- By default we cap torch CPU thread pools to 1 thread for reproducibility and
  CPU-friendliness. Override with --torch-threads / --torch-interop-threads.

Usage
-----
PyTorch only (fastest):
  python scripts/train_neuromancer_stl.py --pretty

Enable RTAMT audit (if installed):
  python scripts/train_neuromancer_stl.py --rtamt --pretty

Also run Neuromancer (if installed):
  python scripts/train_neuromancer_stl.py --neuromancer --pretty

Save a checkpoint + plots into the run directory:
  python scripts/train_neuromancer_stl.py --save-model runs/tmp/model.pt --plot

Suppress the default "eventually" example specs:
  python scripts/train_neuromancer_stl.py --no-eventually-specs

"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


def _ensure_repo_src_on_path() -> None:
    """
    Ensure `src/` is on sys.path so `neural_pde_stl_strel.*` imports work when invoked as:
      python scripts/train_neuromancer_stl.py
    """
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


_ensure_repo_src_on_path()


# Configuration dataclasses


@dataclass(frozen=True)
class PTConfig:
    """PyTorch training configuration (also used to parameterize the Neuromancer run)."""

    n: int = 256
    t_start: float = 0.0
    t_end: float = float(2.0 * math.pi)
    epochs: int = 200
    lr: float = 1e-2
    bound: float = 0.8
    stl_weight: float = 5.0  # λ
    device: str = "cpu"
    seed: int = 0

    # Fixed architecture for this demo (also matches repo docs/audit scripts)
    hidden: int = 64
    depth: int = 2


@dataclass(frozen=True)
class MonitorConfig:
    """Monitoring-only spec configuration."""

    enable_eventually: bool = True
    eventually_below: float = 0.3
    eventually_above: float = 0.9
    # If None, use [t_start, t_end] (the full horizon).
    eventually_window: tuple[float, float] | None = None


@dataclass(frozen=True)
class NMConfig:
    """Neuromancer run configuration."""

    enabled: bool = False
    epochs: int | None = None  # default: use PTConfig.epochs
    lr: float | None = None  # default: use PTConfig.lr
    batch_size: int = 64


@dataclass(frozen=True)
class RunConfig:
    """Top-level run settings."""

    out: str | None = None
    save_model: str | None = None
    plot: bool = False
    plot_dir: str | None = None
    include_series: bool = False
    include_curves: bool = False
    rtamt: bool = False
    pretty: bool = False
    quiet: bool = False

    # CPU-friendliness/reproducibility: torch thread caps (set early in main).
    # Use <=0 to leave torch defaults unchanged.
    torch_threads: int | None = 1
    torch_interop_threads: int | None = 1


# Small helpers


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_out_path() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / "neuromancer_stl" / f"run_{ts}.json"


def _mkdir_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _git_sha_short() -> str | None:
    try:
        out = subprocess.check_output(  # noqa: S603,S607
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def _cpu_model_linux() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return None
    try:
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                _, val = line.split(":", 1)
                return val.strip() or None
    except Exception:
        return None
    return None


def _total_ram_bytes_linux() -> int | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    try:
        for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    kb = int(parts[1])
                    return kb * 1024
    except Exception:
        return None
    return None


def _bytes_to_gib(n: int | None) -> float | None:
    if n is None:
        return None
    return float(n) / (1024.0**3)


def _as_jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_as_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _as_jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "__dataclass_fields__"):
        return _as_jsonable(asdict(obj))
    return str(obj)


# Optional dependency helpers


def _import_torch():
    try:
        import torch
        from torch import nn
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for this script. "
            "Install the repo requirements (see requirements.txt)."
        ) from e
    return torch, nn


def _gather_env(*, torch) -> dict[str, Any]:
    env: dict[str, Any] = {
        "timestamp_utc": _utc_now_iso(),
        "git_sha": _git_sha_short(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "cpu_model": _cpu_model_linux(),
        "total_ram_gib": _bytes_to_gib(_total_ram_bytes_linux()),
        "cwd": os.getcwd(),
        "torch": getattr(torch, "__version__", None),
        "torch_threads": int(torch.get_num_threads()) if hasattr(torch, "get_num_threads") else None,
        "torch_interop_threads": (
            int(torch.get_num_interop_threads()) if hasattr(torch, "get_num_interop_threads") else None
        ),
        "cuda_available": bool(getattr(torch.cuda, "is_available", lambda: False)()),
        "cuda_device_count": int(getattr(torch.cuda, "device_count", lambda: 0)()),
    }
    if env["cuda_available"]:
        try:
            env["cuda_device_name0"] = str(torch.cuda.get_device_name(0))
        except Exception:
            env["cuda_device_name0"] = None

    # Neuromancer version (optional; uses repo helper for robust import probing)
    try:
        from neural_pde_stl_strel.frameworks.neuromancer_hello import neuromancer_version

        env["neuromancer"] = neuromancer_version()
    except Exception:
        env["neuromancer"] = None

    # RTAMT availability
    try:
        import rtamt  # noqa: F401

        env["rtamt_available"] = True
    except Exception:
        env["rtamt_available"] = False

    return env


# Dataset + specs


def _set_seed(*, torch, seed: int) -> None:
    # Deterministic-ish for a toy script; full determinism isn't guaranteed across all ops/devices.
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _make_data(*, torch, cfg: PTConfig):
    device = torch.device(cfg.device)
    t = torch.linspace(cfg.t_start, cfg.t_end, cfg.n, device=device, dtype=torch.float32).view(-1, 1)
    y = torch.sin(t)
    return {"t": t, "y": y}


def _hinge_violation(*, torch, y_hat, bound: float):
    # Differentiable surrogate for (y_hat <= bound)
    return torch.relu(y_hat - float(bound))


def _robustness_always_upper(*, torch, y_hat, bound: float) -> float:
    # ρ = min_t (bound - y_hat(t))
    return float((float(bound) - y_hat).min().item())


def _mask_time_window(*, torch, t, t0: float, t1: float):
    lo = min(float(t0), float(t1))
    hi = max(float(t0), float(t1))
    return (t >= lo) & (t <= hi)


def _robustness_eventually_upper(*, torch, t, y_hat, threshold: float, window: tuple[float, float]) -> float:
    # For predicate (y_hat <= threshold) under F_[t0,t1]:
    # ρ = max_{t in window} (threshold - y_hat(t))
    t0, t1 = window
    mask = _mask_time_window(torch=torch, t=t, t0=t0, t1=t1)
    y_w = y_hat[mask]
    if y_w.numel() == 0:
        raise ValueError("Empty time window for eventually spec; adjust --eventually-window.")
    return float((float(threshold) - y_w).max().item())


def _robustness_eventually_lower(*, torch, t, y_hat, threshold: float, window: tuple[float, float]) -> float:
    # For predicate (y_hat >= threshold) under F_[t0,t1]:
    # ρ = max_{t in window} (y_hat(t) - threshold)
    t0, t1 = window
    mask = _mask_time_window(torch=torch, t=t, t0=t0, t1=t1)
    y_w = y_hat[mask]
    if y_w.numel() == 0:
        raise ValueError("Empty time window for eventually spec; adjust --eventually-window.")
    return float((y_w - float(threshold)).max().item())


# RTAMT monitoring (optional)


def _rtamt_eval(
    *,
    spec_text: str,
    var: str,
    t: list[float],
    y: list[float],
    time_semantics: Literal["dense", "discrete"] = "dense",
) -> float | None:
    """
    Evaluate robustness with RTAMT if available; otherwise return None.

    We use the repository helper (neural_pde_stl_strel.monitoring.rtamt_monitor) so that
    RTAMT remains an optional dependency.
    """
    try:
        from neural_pde_stl_strel.monitoring import rtamt_monitor as rm
    except Exception:
        return None

    try:
        spec = rm.build_stl_spec(spec_text=spec_text, var=var, time_semantics=time_semantics)
        series = list(zip(t, y))
        rob = rm.evaluate_series(spec, {var: series})
        return float(rob)
    except Exception:
        return None


# Training: pure PyTorch


class _MLP:
    """Tiny tanh MLP: 1 -> hidden -> ... -> hidden -> 1."""

    def __init__(self, *, nn, hidden: int, depth: int):
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers: list[Any] = []
        in_dim = 1
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.Tanh())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def to(self, device):
        self.net.to(device)
        return self

    def __call__(self, x):
        return self.net(x)

    def parameters(self):
        return self.net.parameters()

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        return self.net.load_state_dict(sd)


def _train_pytorch(
    *,
    torch,
    nn,
    cfg: PTConfig,
    mon: MonitorConfig,
    run: RunConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _set_seed(torch=torch, seed=cfg.seed)
    data = _make_data(torch=torch, cfg=cfg)
    t = data["t"]
    y = data["y"]

    model = _MLP(nn=nn, hidden=cfg.hidden, depth=cfg.depth).to(torch.device(cfg.device))
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))

    curves: dict[str, list[float]] = {"mse": [], "violation_mean": [], "robustness_safe": []}

    t0 = time.perf_counter()
    for _epoch in range(int(cfg.epochs)):
        opt.zero_grad(set_to_none=True)
        y_hat = model(t)
        mse = torch.mean((y_hat - y) ** 2)
        violation = _hinge_violation(torch=torch, y_hat=y_hat, bound=cfg.bound)
        violation_mean = torch.mean(violation)
        loss = mse + float(cfg.stl_weight) * violation_mean
        loss.backward()
        opt.step()

        if run.include_curves:
            curves["mse"].append(float(mse.item()))
            curves["violation_mean"].append(float(violation_mean.item()))
            curves["robustness_safe"].append(_robustness_always_upper(torch=torch, y_hat=y_hat, bound=cfg.bound))

    train_seconds = time.perf_counter() - t0

    # Final evaluation
    with torch.no_grad():
        y_hat = model(t)
        mse = float(torch.mean((y_hat - y) ** 2).item())
        violation = _hinge_violation(torch=torch, y_hat=y_hat, bound=cfg.bound)
        final_violation = float(torch.mean(violation).item())
        max_violation = float(torch.max(violation).item())
        robustness_safe = _robustness_always_upper(torch=torch, y_hat=y_hat, bound=cfg.bound)
        satisfied_safe = bool(robustness_safe >= 0.0)

        # Optional additional specs (monitor-only)
        ev_window = mon.eventually_window or (cfg.t_start, cfg.t_end)
        ev_metrics: dict[str, Any] = {}
        if mon.enable_eventually:
            ev_below_rob = _robustness_eventually_upper(
                torch=torch, t=t, y_hat=y_hat, threshold=mon.eventually_below, window=ev_window
            )
            ev_above_rob = _robustness_eventually_lower(
                torch=torch, t=t, y_hat=y_hat, threshold=mon.eventually_above, window=ev_window
            )
            ev_metrics = {
                "below": {
                    "threshold": float(mon.eventually_below),
                    "window": [float(ev_window[0]), float(ev_window[1])],
                    "robustness": float(ev_below_rob),
                    "satisfied": bool(ev_below_rob >= 0.0),
                },
                "above": {
                    "threshold": float(mon.eventually_above),
                    "window": [float(ev_window[0]), float(ev_window[1])],
                    "robustness": float(ev_above_rob),
                    "satisfied": bool(ev_above_rob >= 0.0),
                },
            }

        # Optional RTAMT audit (dense-time, timestamped by the actual t grid)
        rtamt_safe = None
        rtamt_ev_below = None
        rtamt_ev_above = None
        if run.rtamt:
            t_list = [float(x) for x in t.detach().cpu().view(-1).tolist()]
            y_list = [float(x) for x in y_hat.detach().cpu().view(-1).tolist()]
            rtamt_safe = _rtamt_eval(spec_text=f"always (y <= {float(cfg.bound)})", var="y", t=t_list, y=y_list)
            if mon.enable_eventually:
                win0, win1 = ev_window
                rtamt_ev_below = _rtamt_eval(
                    spec_text=f"eventually[{float(win0)}:{float(win1)}] (y <= {float(mon.eventually_below)})",
                    var="y",
                    t=t_list,
                    y=y_list,
                )
                rtamt_ev_above = _rtamt_eval(
                    spec_text=f"eventually[{float(win0)}:{float(win1)}] (y >= {float(mon.eventually_above)})",
                    var="y",
                    t=t_list,
                    y=y_list,
                )
                if ev_metrics:
                    ev_metrics["below"]["rtamt"] = rtamt_ev_below
                    ev_metrics["above"]["rtamt"] = rtamt_ev_above

        n_params = int(sum(p.numel() for p in model.parameters()))
        metrics: dict[str, Any] = {
            "final_mse": float(mse),
            "final_violation": float(final_violation),
            "max_violation": float(max_violation),
            "robustness_min": float(robustness_safe),
            "satisfied": satisfied_safe,
            "train_seconds": float(train_seconds),
            "n_params": n_params,
            "rtamt_robustness": rtamt_safe,  # backward-compat alias for φ_safe
            "eventually": ev_metrics,
            "rtamt": {
                "safe": rtamt_safe,
                "eventually_below": rtamt_ev_below,
                "eventually_above": rtamt_ev_above,
            },
        }

        artifacts: dict[str, Any] = {
            "model": model,
            "data": data,
            "y_hat": y_hat,
            "curves": curves if run.include_curves else None,
        }
        if run.include_series:
            artifacts["series"] = {
                "t": [float(x) for x in t.detach().cpu().view(-1).tolist()],
                "y_true": [float(x) for x in y.detach().cpu().view(-1).tolist()],
                "y_hat": [float(x) for x in y_hat.detach().cpu().view(-1).tolist()],
            }

    return metrics, artifacts


# Training: Neuromancer (optional)


def _train_neuromancer(
    *,
    torch,
    nn,
    cfg: PTConfig,
    mon: MonitorConfig,
    run: RunConfig,
    nm_cfg: NMConfig,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    """
    Try to run a Neuromancer version of the same objective/constraint.

    Returns (metrics, artifacts, error_message). On failure, metrics/artifacts are None
    and error_message contains a short reason.
    """
    if not nm_cfg.enabled:
        return None, None, None

    try:
        import neuromancer as nm  # type: ignore
    except Exception as e:
        return None, None, f"Neuromancer import failed: {type(e).__name__}: {e}"

    # Resolve key classes (keep imports local to remain optional).
    try:
        DictDataset = nm.dataset.DictDataset
        Node = nm.system.Node
        variable = nm.constraint.variable
        PenaltyLoss = nm.loss.PenaltyLoss
        Problem = nm.problem.Problem
    except Exception as e:
        return None, None, f"Neuromancer API mismatch: {type(e).__name__}: {e}"

    data = _make_data(torch=torch, cfg=cfg)
    t = data["t"]
    y = data["y"]

    # Build an MLP in Neuromancer style if available; otherwise fall back to a plain nn.Sequential.
    try:
        blocks = nm.modules.blocks
        # Many Neuromancer examples use SLiM linear maps; use if present, otherwise default.
        try:
            linear_map = nm.slim.maps["linear"]
        except Exception:
            linear_map = None

        hsizes = [int(cfg.hidden)] * int(cfg.depth)
        try:
            func = blocks.MLP(
                insize=1,
                outsize=1,
                hsizes=hsizes,
                nonlin=nn.Tanh,
                linear_map=linear_map,
            )
        except TypeError:
            # Fallback for older signature patterns.
            func = blocks.MLP(1, 1, hsizes, nn.Tanh, linear_map=linear_map)
    except Exception:
        # Fallback MLP with the same architecture.
        func = _MLP(nn=nn, hidden=cfg.hidden, depth=cfg.depth).net

    node = Node(func, ["t"], ["y_hat"], name="f")

    # Symbolic vars for objective and constraint
    y_hat = variable("y_hat")
    y_true = variable("y")

    obj = ((y_hat - y_true) ** 2).mean().minimize(weight=1.0)

    # Weighted constraint (penalty method)
    con = float(cfg.stl_weight) * (y_hat <= float(cfg.bound))

    loss = PenaltyLoss(objectives=[obj], constraints=[con])
    problem = Problem(nodes=[node], loss=loss)
    problem.to(torch.device(cfg.device))

    dataset = DictDataset(data, name="train")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(nm_cfg.batch_size),
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )
    lr = float(nm_cfg.lr if nm_cfg.lr is not None else cfg.lr)
    epochs = int(nm_cfg.epochs if nm_cfg.epochs is not None else cfg.epochs)
    opt = torch.optim.Adam(problem.parameters(), lr=lr)

    t0 = time.perf_counter()
    for _epoch in range(epochs):
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            out = problem(batch)
            out["loss"].backward()
            opt.step()
    train_seconds = time.perf_counter() - t0

    with torch.no_grad():
        y_hat_t = node({"t": t})["y_hat"]
        mse = float(torch.mean((y_hat_t - y) ** 2).item())
        violation = _hinge_violation(torch=torch, y_hat=y_hat_t, bound=cfg.bound)
        final_violation = float(torch.mean(violation).item())
        max_violation = float(torch.max(violation).item())
        robustness_safe = _robustness_always_upper(torch=torch, y_hat=y_hat_t, bound=cfg.bound)
        satisfied_safe = bool(robustness_safe >= 0.0)

        ev_window = mon.eventually_window or (cfg.t_start, cfg.t_end)
        ev_metrics: dict[str, Any] = {}
        if mon.enable_eventually:
            ev_below_rob = _robustness_eventually_upper(
                torch=torch, t=t, y_hat=y_hat_t, threshold=mon.eventually_below, window=ev_window
            )
            ev_above_rob = _robustness_eventually_lower(
                torch=torch, t=t, y_hat=y_hat_t, threshold=mon.eventually_above, window=ev_window
            )
            ev_metrics = {
                "below": {
                    "threshold": float(mon.eventually_below),
                    "window": [float(ev_window[0]), float(ev_window[1])],
                    "robustness": float(ev_below_rob),
                    "satisfied": bool(ev_below_rob >= 0.0),
                },
                "above": {
                    "threshold": float(mon.eventually_above),
                    "window": [float(ev_window[0]), float(ev_window[1])],
                    "robustness": float(ev_above_rob),
                    "satisfied": bool(ev_above_rob >= 0.0),
                },
            }

        rtamt_safe = None
        rtamt_ev_below = None
        rtamt_ev_above = None
        if run.rtamt:
            t_list = [float(x) for x in t.detach().cpu().view(-1).tolist()]
            y_list = [float(x) for x in y_hat_t.detach().cpu().view(-1).tolist()]
            rtamt_safe = _rtamt_eval(spec_text=f"always (y <= {float(cfg.bound)})", var="y", t=t_list, y=y_list)
            if mon.enable_eventually:
                win0, win1 = ev_window
                rtamt_ev_below = _rtamt_eval(
                    spec_text=f"eventually[{float(win0)}:{float(win1)}] (y <= {float(mon.eventually_below)})",
                    var="y",
                    t=t_list,
                    y=y_list,
                )
                rtamt_ev_above = _rtamt_eval(
                    spec_text=f"eventually[{float(win0)}:{float(win1)}] (y >= {float(mon.eventually_above)})",
                    var="y",
                    t=t_list,
                    y=y_list,
                )
                if ev_metrics:
                    ev_metrics["below"]["rtamt"] = rtamt_ev_below
                    ev_metrics["above"]["rtamt"] = rtamt_ev_above

        n_params = int(sum(p.numel() for p in problem.parameters()))
        metrics: dict[str, Any] = {
            "final_mse": float(mse),
            "final_violation": float(final_violation),
            "max_violation": float(max_violation),
            "robustness_min": float(robustness_safe),
            "satisfied": satisfied_safe,
            "train_seconds": float(train_seconds),
            "n_params": n_params,
            "rtamt_robustness": rtamt_safe,  # backward-compat alias for φ_safe
            "eventually": ev_metrics,
            "rtamt": {
                "safe": rtamt_safe,
                "eventually_below": rtamt_ev_below,
                "eventually_above": rtamt_ev_above,
            },
        }
        artifacts: dict[str, Any] = {
            "data": data,
            "y_hat": y_hat_t,
        }
        if run.include_series:
            artifacts["series"] = {
                "t": [float(x) for x in t.detach().cpu().view(-1).tolist()],
                "y_true": [float(x) for x in y.detach().cpu().view(-1).tolist()],
                "y_hat": [float(x) for x in y_hat_t.detach().cpu().view(-1).tolist()],
            }

    return metrics, artifacts, None


# Plotting (optional)


def _maybe_plot_timeseries(*, out_png: Path, t: list[float], y_true: list[float], y_hat: list[float], bound: float):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t, y_true, label="y_true")
    ax.plot(t, y_hat, label="y_hat")
    ax.axhline(bound, linestyle="--", label="bound")
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _maybe_plot_curves(*, out_png: Path, curves: dict[str, list[float]] | None) -> list[Path]:
    """
    Plot training curves.

    This function writes one PNG per curve (to keep each figure simple) and returns
    the list of files written. If matplotlib is unavailable or curves is None/empty,
    returns an empty list.
    """
    if not curves:
        return []

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    written: list[Path] = []
    for key, values in curves.items():
        if not values:
            continue
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(list(range(len(values))), values)
        ax.set_xlabel("epoch")
        ax.set_ylabel(key)
        ax.set_title(key)
        fig.tight_layout()
        out = out_png.with_name(f"{out_png.stem}_{key}{out_png.suffix}")
        fig.savefig(out, dpi=180)
        plt.close(fig)
        written.append(out)

    return written


# CLI


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a toy MLP with an STL-style safety penalty; optionally run Neuromancer + RTAMT audit.",
    )

    # Core training config
    p.add_argument("--n", type=int, default=PTConfig.n, help="Number of time samples.")
    p.add_argument("--t-start", type=float, default=PTConfig.t_start, help="Start time for the grid.")
    p.add_argument("--t-end", type=float, default=PTConfig.t_end, help="End time for the grid.")
    p.add_argument("--epochs", type=int, default=PTConfig.epochs, help="Training epochs.")
    p.add_argument("--lr", type=float, default=PTConfig.lr, help="Learning rate.")
    p.add_argument("--bound", type=float, default=PTConfig.bound, help="Safety upper bound U_max.")
    p.add_argument("--stl-weight", type=float, default=PTConfig.stl_weight, help="λ (weight on STL penalty).")
    p.add_argument("--device", type=str, default=PTConfig.device, help='Torch device, e.g. "cpu" or "cuda".')
    p.add_argument("--seed", type=int, default=PTConfig.seed, help="Random seed.")

    # Monitor-only specs
    p.add_argument(
        "--no-eventually-specs",
        action="store_true",
        help="Disable the default monitor-only eventually specs.",
    )
    p.add_argument(
        "--eventually-below",
        type=float,
        default=MonitorConfig.eventually_below,
        help="Threshold c for monitor-only spec F(y <= c). (Default: 0.3)",
    )
    p.add_argument(
        "--eventually-above",
        type=float,
        default=MonitorConfig.eventually_above,
        help="Threshold c for monitor-only spec F(y >= c). (Default: 0.9)",
    )
    p.add_argument(
        "--eventually-window",
        type=float,
        nargs=2,
        default=None,
        metavar=("T0", "T1"),
        help="Time window [T0, T1] for eventually specs (dense time). Default: full horizon.",
    )

    # Neuromancer
    p.add_argument(
        "--neuromancer",
        action="store_true",
        help="Also run the Neuromancer symbolic-API path (if installed).",
    )
    p.add_argument(
        "--no-nm",
        action="store_true",
        help="(Deprecated) Disable Neuromancer. Has effect only if --neuromancer is set.",
    )
    p.add_argument("--nm-epochs", type=int, default=None, help="Neuromancer epochs (default: --epochs).")
    p.add_argument("--nm-lr", type=float, default=None, help="Neuromancer learning rate (default: --lr).")
    p.add_argument("--nm-batch-size", type=int, default=NMConfig.batch_size, help="Neuromancer batch size.")

    # Output / artifacts
    p.add_argument("--out", type=str, default=None, help="Output JSON path (default: runs/neuromancer_stl/run_*.json)")
    p.add_argument("--save-model", type=str, default=None, help="Save PyTorch model state_dict to this path.")
    p.add_argument("--plot", action="store_true", help="Write PNG figures to --plot-dir (or run directory).")
    p.add_argument("--plot-dir", type=str, default=None, help="Directory for plots (default: JSON output directory).")
    p.add_argument(
        "--include-series",
        action="store_true",
        help="Include the full (t, y_true, y_hat) series in the JSON output.",
    )
    p.add_argument(
        "--include-curves",
        action="store_true",
        help="Include per-epoch curves in JSON output and enable curve plots.",
    )

    # Resource control
    p.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Set torch.set_num_threads(N). Use <=0 to leave default.",
    )
    p.add_argument(
        "--torch-interop-threads",
        type=int,
        default=1,
        help="Set torch.set_num_interop_threads(N). Use <=0 to leave default.",
    )

    # Audit
    p.add_argument("--rtamt", action="store_true", help="Compute RTAMT robustness (optional dependency).")

    # UX
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON to stdout.")
    p.add_argument("--quiet", action="store_true", help="Suppress non-error stdout output.")

    return p


def _parse_configs(args: argparse.Namespace) -> tuple[PTConfig, MonitorConfig, NMConfig, RunConfig]:
    pt = PTConfig(
        n=int(args.n),
        t_start=float(args.t_start),
        t_end=float(args.t_end),
        epochs=int(args.epochs),
        lr=float(args.lr),
        bound=float(args.bound),
        stl_weight=float(args.stl_weight),
        device=str(args.device),
        seed=int(args.seed),
    )

    mon = MonitorConfig(
        enable_eventually=not bool(args.no_eventually_specs),
        eventually_below=float(args.eventually_below),
        eventually_above=float(args.eventually_above),
        eventually_window=tuple(float(x) for x in args.eventually_window) if args.eventually_window else None,
    )

    nm_enabled = bool(args.neuromancer)
    if bool(args.no_nm) and nm_enabled:
        raise ValueError("Conflicting flags: --neuromancer and --no-nm")

    nm = NMConfig(
        enabled=nm_enabled and not bool(args.no_nm),
        epochs=int(args.nm_epochs) if args.nm_epochs is not None else None,
        lr=float(args.nm_lr) if args.nm_lr is not None else None,
        batch_size=int(args.nm_batch_size),
    )

    run = RunConfig(
        out=args.out,
        save_model=args.save_model,
        plot=bool(args.plot),
        plot_dir=args.plot_dir,
        include_series=bool(args.include_series),
        include_curves=bool(args.include_curves),
        rtamt=bool(args.rtamt),
        pretty=bool(args.pretty),
        quiet=bool(args.quiet),
        torch_threads=int(args.torch_threads),
        torch_interop_threads=int(args.torch_interop_threads),
    )

    return pt, mon, nm, run


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        pt_cfg, mon_cfg, nm_cfg, run_cfg = _parse_configs(args)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    torch, nn = _import_torch()

    # CPU-friendliness/reproducibility: cap torch thread pools early (prevents oversubscription).
    if run_cfg.torch_threads is not None and int(run_cfg.torch_threads) > 0:
        try:
            torch.set_num_threads(int(run_cfg.torch_threads))
        except Exception:
            pass
    if run_cfg.torch_interop_threads is not None and int(run_cfg.torch_interop_threads) > 0:
        try:
            torch.set_num_interop_threads(int(run_cfg.torch_interop_threads))
        except Exception:
            pass

    out_path = Path(run_cfg.out) if run_cfg.out else _default_out_path()
    _mkdir_parent(out_path)
    run_dir = out_path.parent

    # Run: PyTorch
    env = _gather_env(torch=torch)
    pyt_metrics, pyt_art = _train_pytorch(torch=torch, nn=nn, cfg=pt_cfg, mon=mon_cfg, run=run_cfg)

    # Save checkpoint (PyTorch path only)
    ckpt_path: Path | None = None
    if run_cfg.save_model:
        ckpt_path = Path(run_cfg.save_model)
        _mkdir_parent(ckpt_path)
        torch.save(pyt_art["model"].state_dict(), ckpt_path)

    # Run: Neuromancer (optional)
    nm_metrics: dict[str, Any] | None = None
    nm_art: dict[str, Any] | None = None
    nm_error: str | None = None
    if nm_cfg.enabled:
        nm_metrics, nm_art, nm_error = _train_neuromancer(
            torch=torch, nn=nn, cfg=pt_cfg, mon=mon_cfg, run=run_cfg, nm_cfg=nm_cfg
        )

    # Plots (optional)
    plot_paths: dict[str, Any] = {}
    if run_cfg.plot:
        plot_dir = Path(run_cfg.plot_dir) if run_cfg.plot_dir else run_dir
        plot_dir.mkdir(parents=True, exist_ok=True)

        # PyTorch plots
        t_list = [float(x) for x in pyt_art["data"]["t"].detach().cpu().view(-1).tolist()]
        y_true_list = [float(x) for x in pyt_art["data"]["y"].detach().cpu().view(-1).tolist()]
        y_hat_list = [float(x) for x in pyt_art["y_hat"].detach().cpu().view(-1).tolist()]
        p_png = plot_dir / "pytorch_timeseries.png"
        _maybe_plot_timeseries(
            out_png=p_png, t=t_list, y_true=y_true_list, y_hat=y_hat_list, bound=float(pt_cfg.bound)
        )
        plot_paths["pytorch_timeseries"] = str(p_png)

        if run_cfg.include_curves:
            c_png = plot_dir / "pytorch_curves.png"
            created = _maybe_plot_curves(out_png=c_png, curves=pyt_art.get("curves"))
            if created:
                plot_paths["pytorch_curves"] = [str(p) for p in created]

        # Neuromancer plots (if run succeeded)
        if nm_cfg.enabled and nm_art is not None:
            t_list = [float(x) for x in nm_art["data"]["t"].detach().cpu().view(-1).tolist()]
            y_true_list = [float(x) for x in nm_art["data"]["y"].detach().cpu().view(-1).tolist()]
            y_hat_list = [float(x) for x in nm_art["y_hat"].detach().cpu().view(-1).tolist()]
            nm_png = plot_dir / "neuromancer_timeseries.png"
            _maybe_plot_timeseries(
                out_png=nm_png, t=t_list, y_true=y_true_list, y_hat=y_hat_list, bound=float(pt_cfg.bound)
            )
            plot_paths["neuromancer_timeseries"] = str(nm_png)

    # Compose report
    specs: dict[str, Any] = {
        "phi_safe": {
            "stl": "G (y(t) <= U_max)",
            "robustness": "rho_safe = min_t (U_max - y(t))",
            "lambda": float(pt_cfg.stl_weight),
            "U_max": float(pt_cfg.bound),
        },
        "phi_ev_below": {
            "stl": "F_[t0,t1] (y(t) <= c_below)",
            "robustness": "rho = max_{t∈[t0,t1]} (c_below - y(t))",
        },
        "phi_ev_above": {
            "stl": "F_[t0,t1] (y(t) >= c_above)",
            "robustness": "rho = max_{t∈[t0,t1]} (y(t) - c_above)",
        },
    }

    report: dict[str, Any] = {
        "config": {
            "pytorch": _as_jsonable(pt_cfg),
            "monitor": _as_jsonable(mon_cfg),
            "neuromancer": _as_jsonable(nm_cfg),
            "run": _as_jsonable(run_cfg),
        },
        "env": _as_jsonable(env),
        "specs": _as_jsonable(specs),
        "pytorch": _as_jsonable(pyt_metrics),
        "neuromancer": _as_jsonable(nm_metrics),
        "neuromancer_error": nm_error,
        "artifacts": _as_jsonable(
            {
                "out": str(out_path),
                "checkpoint": str(ckpt_path) if ckpt_path else None,
                "plots": plot_paths if plot_paths else None,
            }
        ),
    }

    # Write JSON to disk
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    if not run_cfg.quiet:
        print(f"Wrote: {out_path}")
        if ckpt_path is not None:
            print(f"Wrote checkpoint: {ckpt_path}")
        if plot_paths:
            print("Wrote plots:")
            for k, v in plot_paths.items():
                print(f"  {k}: {v}")

    if run_cfg.pretty:
        print(json.dumps(report, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
