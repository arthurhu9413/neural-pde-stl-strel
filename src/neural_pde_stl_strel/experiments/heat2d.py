# ruff: noqa: I001
from __future__ import annotations

"""2-D heat-equation PINN experiment with optional STL regularization.

This module implements a CPU-friendly Physics-Informed Neural Network (PINN) baseline for the
2-D heat equation, with an **optional differentiable STL penalty** that can be used as a soft
constraint during training.

PDE + IC/BC
We solve the heat equation on a rectangular domain Ω × [t_min, t_max]:

    u_t(x, y, t) = α · (u_xx(x, y, t) + u_yy(x, y, t))

with:
* **Dirichlet boundary condition**: u(x, y, t) = 0 for (x, y) ∈ ∂Ω
* **Initial condition**: a Gaussian bump at t = 0, configurable via amplitude/sharpness.

STL (optional)
This experiment can add a differentiable STL penalty computed on a coarse evaluation grid.
A common **safety** property is an upper bound:

    φ_safe := G_[t0,t1] (u(x,y,t) ≤ U_max)

and a common **liveness/cooling** property is an eventual bound:

    φ_cool := F_[t0,t1] (u(x,y,t) ≤ U_cool)

Spatial quantification is approximated over a sampled spatial grid:
* ``space_op = "forall"``  -> soft minimum over space (∀ x,y)
* ``space_op = "exists"``  -> soft maximum over space (∃ x,y)

The scalar robustness ρ is fed to a smooth penalty φ(ρ) (e.g., softplus hinge). The overall loss:

    L = L_PDE + w_BCIC · (L_BC + L_IC) + λ · φ(ρ)

where ``λ = stl_weight`` is the STL regularization weight (the "lambda parameter" referenced in
project discussions).

Dataflow sketch
Inputs
  * PDE + domain + IC/BC
  * PINN architecture (MLP)
  * (Optional) STL specification + discretization choices

Forward / losses
  coords ──► PINN u(x,y,t) ──► PDE residual r ──► L_PDE
    │               │
    │               ├────────► BC/IC targets ───► L_BC + L_IC
    │
    └───────────────└────────► (Optional) u on coarse grid ─► ρ ─► L_STL

Optimization
  L_total ─► backprop ─► Adam ─► updated PINN parameters

Outputs (saved under io.run_dir / io.results_dir)
* ``heat2d_<tag>.csv``: training log (loss terms, robustness, timings)
* ``heat2d_<tag>_field.pt``: final predicted field on the full cartesian grid (torch save)
* Optional figures/frames (2-D heatmaps, time-series summaries)
* Optional packed ``.npy`` + ``heat2d_dt.txt`` sidecars to support STREL/MoonLight tooling.

Optional STREL audit (MoonLight)
If ``strel_audit.run: true`` is provided, this experiment will *best-effort* invoke
``scripts/eval_heat2d_moonlight.py`` (if present) to monitor a STREL specification on the
exported field and write a JSON summary. This keeps STREL as an optional dependency and avoids
hard-requiring Java/JNI for the core PINN baseline.

"""

from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from ..models.mlp import MLP
from ..physics.heat2d import (
    bc_ic_heat2d,
    gaussian_ic,
    make_dirichlet_mask,
    MaskedModel,
    residual_heat2d,
)
from ..training.grids import grid2d
from ..utils.logger import CSVLogger
from ..utils.seed import seed_everything

try:
    from ..monitoring.stl_soft import (
        STLPenalty,
        always,
        always_window,
        eventually,
        eventually_window,
        pred_leq,
        softmax,
        softmin,
    )

    _HAS_STL = True
except Exception:  # pragma: no cover - optional (but included in this repo)
    _HAS_STL = False

__all__ = ["Heat2DConfig", "run_heat2d"]


# Config


@dataclass(frozen=True)
class STRELAuditConfig:
    """Optional post-hoc STREL (MoonLight) monitoring configuration.

    Notes
    -----
    * This is *optional*. The core experiment should run without MoonLight/Java.
    * Paths are interpreted relative to the run directory, with a small convenience
      rule: a leading "results/" prefix is stripped (so configs remain readable
      whether they are run directly or via scripts/run_experiment.py, which nests
      outputs under a timestamped run directory).
    """

    export: bool = True
    run: bool = False

    # Exported field sidecars (for scripts/eval_heat2d_moonlight.py)
    packed_field_path: str = "heat2d_field_xy_t.npy"
    dt_path: str = "heat2d_dt.txt"
    layout: Literal["xy_t", "t_xy"] = "xy_t"

    # MoonLight spec invocation
    mls_path: str = "scripts/specs/contain_hotspot.mls"
    formula: tuple[str, ...] = ("contain_hotspot",)

    # Optional binarization of the field to a boolean predicate hot := (u ≥ θ)
    binarize: bool | None = True
    threshold: float | None = None

    # Control eval script behavior
    moonlight: bool = True
    suppress_java_output: bool = True
    plots: bool = True
    sweep: bool = False

    # Output artifacts
    json_out: str = "heat2d_strel.json"
    fig_dir: str = "strel_figs"


@dataclass(frozen=True)
class Heat2DConfig:
    """Configuration for the heat2d experiment.

    The YAML config is expected to have top-level sections:

      model, grid, optim, physics, rar, stl, io, (optional) strel_audit
    """

    # Repro + hardware
    seed: int = 0
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"
    dtype: str = "float32"

    # Model
    hidden: tuple[int, ...] = (64, 64, 64)
    activation: str = "tanh"
    out_activation: str = "identity"

    # Dense training / export grid
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0
    n_x: int = 64
    n_y: int = 64
    n_t: int = 16

    # Optim
    lr: float = 1e-3
    epochs: int = 200
    batch: int = 4096
    grad_clip: float = 1.0

    # Performance toggles (keep conservative defaults: higher-order AD can be fragile)
    amp: bool = False
    compile: bool = False

    # Physics
    alpha: float = 0.1
    bcic_weight: float = 10.0
    n_boundary: int = 512
    n_initial: int = 512
    ic_amplitude: float = 1.0
    ic_sharpness: float = 50.0
    ic_center_x: float = 0.5
    ic_center_y: float = 0.5
    use_dirichlet_mask_pow: int = 0  # 0 disables; otherwise enforce u=0 on boundary via output transform

    # RAR (Residual-based Adaptive Refinement; optional)
    rar_pool: int = 0
    rar_hard_frac: float = 0.1
    rar_every: int = 10

    # STL (optional; differentiable)
    stl_use: bool = False
    stl_weight: float = 1.0
    stl_margin: float = 0.0
    stl_beta: float = 10.0
    stl_temp: float = 0.1
    stl_kind: Literal["softplus", "logistic", "hinge", "sqhinge"] = "softplus"

    stl_u_min: float | None = None
    stl_u_max: float | None = 1.0

    # STL evaluation grid + operators
    stl_x_min: float = 0.0
    stl_x_max: float = 1.0
    stl_y_min: float = 0.0
    stl_y_max: float = 1.0
    stl_nx: int = 32
    stl_ny: int = 32
    stl_nt: int = 16
    stl_t_min: float = 0.0
    stl_t_max: float = 1.0
    stl_operator: Literal["always", "eventually"] = "always"
    stl_space_op: Literal["forall", "exists"] = "forall"
    stl_window: int = 0
    stl_stride: int = 1
    stl_every: int = 5

    # IO
    results_dir: str = "results"
    tag: str = "run"
    save_ckpt: bool = True
    save_frames: bool = True
    frames_idx: tuple[int, ...] = (0, 5, 10, 15)
    save_figs: bool = True
    print_every: int = 10

    # Optional STREL audit configuration
    strel_audit: STRELAuditConfig = field(default_factory=STRELAuditConfig)


# Helpers


def _select_device(device: str) -> torch.device:
    dev = (device or "auto").strip().lower()
    if dev == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(dev)


def _resolve_dtype(name: str) -> torch.dtype:
    key = (name or "float32").strip().lower()
    if key in {"float32", "fp32", "float"}:
        return torch.float32
    if key in {"float64", "fp64", "double"}:
        return torch.float64
    raise ValueError(f"Unsupported dtype '{name}'. Use 'float32' or 'float64'.")


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _parse_int_tuple(x: Any, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if x is None:
        return default
    if isinstance(x, (list, tuple)):
        return tuple(int(v) for v in x)
    if isinstance(x, str):
        parts = [p.strip() for p in x.replace(";", ",").split(",") if p.strip()]
        return tuple(int(v) for v in parts)
    return (int(x),)


def _repo_root() -> Path | None:
    # Best effort: repo_root/src/neural_pde_stl_strel/experiments/heat2d.py -> parents[3]
    try:
        return Path(__file__).resolve().parents[3]
    except Exception:
        return None


def _resolve_results_path(path_str: str, *, results_dir: Path) -> Path:
    """Interpret a (possibly-relative) path w.r.t. the run directory.

    Convenience: if the path starts with "results/", drop that prefix.
    """
    p = Path(path_str)
    if p.is_absolute():
        return p
    parts = p.parts
    if parts and parts[0] == "results":
        p = Path(*parts[1:])
    return results_dir / p


def _resolve_project_path(path_str: str) -> Path:
    """Resolve a project-relative path (best effort)."""
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p
    root = _repo_root()
    if root is not None:
        cand = root / p
        if cand.exists():
            return cand
    return p


def _maybe_compile_heat2d(
    model_eager: nn.Module,
    *,
    enable: bool,
    device: torch.device,
    dtype: torch.dtype,
    alpha: float,
) -> tuple[nn.Module, bool]:
    """Best-effort torch.compile with a tiny smoke test for higher-order AD.

    Rationale
    ---------
    PINNs for PDEs with second derivatives require "double backward" through AD graphs.
    Depending on the PyTorch version/backend, `torch.compile` may fail or be unstable for
    such higher-order derivative workloads. We therefore:
      1) attempt compilation, and
      2) run a small forward+residual+backward smoke test;
    if anything fails, we fall back to eager mode.
    """
    if not enable:
        return model_eager, False
    if not hasattr(torch, "compile"):
        print("[heat2d] torch.compile unavailable; continuing in eager mode.")
        return model_eager, False

    try:
        compiled = torch.compile(model_eager)  # type: ignore[attr-defined]
    except Exception as e:  # pragma: no cover
        print(f"[heat2d] torch.compile failed; continuing eager. ({e})")
        return model_eager, False

    # Smoke test: 4 points, ensure backward through residual works.
    try:
        model_eager.zero_grad(set_to_none=True)
        pts = torch.tensor(
            [
                [0.1, 0.2, 0.0],
                [0.7, 0.3, 0.5],
                [0.4, 0.9, 0.8],
                [0.6, 0.6, 1.0],
            ],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        res = residual_heat2d(compiled, pts, alpha=alpha)
        loss = res.square().mean()
        loss.backward()
        model_eager.zero_grad(set_to_none=True)
    except Exception as e:  # pragma: no cover
        print(f"[heat2d] compiled smoke test failed; continuing eager. ({e})")
        return model_eager, False

    return compiled, True


def _spatial_reduce(m: torch.Tensor, *, op: Literal["forall", "exists"], temp: float) -> torch.Tensor:
    # m is expected to be shape (N_space,) (after time reduction).
    if op == "forall":
        return softmin(m, temp=temp, dim=-1)
    return softmax(m, temp=temp, dim=-1)


def _stl_loss_and_robustness(cfg: Heat2DConfig, model: nn.Module, penalty: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (loss_stl, robustness) for the configured STL spec."""
    if (not _HAS_STL) or (not cfg.stl_use):
        z = torch.tensor(0.0, device=next(model.parameters()).device)
        return z, z

    # Build evaluation grid (cartesian table, time varies fastest).
    p0 = next(model.parameters())
    _, _, _, XYT = grid2d(
        x_min=cfg.stl_x_min,
        x_max=cfg.stl_x_max,
        y_min=cfg.stl_y_min,
        y_max=cfg.stl_y_max,
        t_min=cfg.stl_t_min,
        t_max=cfg.stl_t_max,
        n_x=cfg.stl_nx,
        n_y=cfg.stl_ny,
        n_t=cfg.stl_nt,
        device=p0.device,
        dtype=p0.dtype,
        return_cartesian=True,
    )

    # u has shape (N_xy, N_t)
    u = model(XYT).reshape(cfg.stl_nx * cfg.stl_ny, cfg.stl_nt)

    # Predicate margins (positive means satisfied).
    margins: list[torch.Tensor] = []
    if cfg.stl_u_max is not None:
        margins.append(pred_leq(u, float(cfg.stl_u_max)))  # u_max - u
    if cfg.stl_u_min is not None:
        margins.append(pred_leq(-u, -float(cfg.stl_u_min)))  # u - u_min

    if not margins:
        z = torch.tensor(0.0, device=u.device, dtype=u.dtype)
        return z, z

    if len(margins) == 1:
        m = margins[0]
    else:
        # Combine multiple predicates via a soft minimum across the predicate axis.
        m = softmin(torch.stack(margins, dim=0), temp=cfg.stl_temp, dim=0)

    # Temporal operator over time axis.
    if cfg.stl_operator == "always":
        if cfg.stl_window and cfg.stl_window > 0:
            m_w = always_window(
                m, temp=cfg.stl_temp, time_dim=-1,
                window=int(cfg.stl_window), stride=int(cfg.stl_stride),
            )
            m_t = softmin(m_w, temp=cfg.stl_temp, dim=-1)  # min over windows
        else:
            m_t = always(m, temp=cfg.stl_temp, time_dim=-1)
    elif cfg.stl_operator == "eventually":
        if cfg.stl_window and cfg.stl_window > 0:
            m_w = eventually_window(
                m, temp=cfg.stl_temp, time_dim=-1, window=int(cfg.stl_window), stride=int(cfg.stl_stride)
            )
            m_t = softmax(m_w, temp=cfg.stl_temp, dim=-1)  # max over windows
        else:
            m_t = eventually(m, temp=cfg.stl_temp, time_dim=-1)
    else:  # pragma: no cover
        raise ValueError(f"Unknown stl_operator: {cfg.stl_operator}")

    # Spatial quantification over the flattened (x,y) grid.
    rob = _spatial_reduce(m_t, op=cfg.stl_space_op, temp=cfg.stl_temp)

    return penalty(rob), rob


def _parse_config(cfg_dict: dict[str, Any]) -> Heat2DConfig:
    model = cfg_dict.get("model", {}) or {}
    grid = cfg_dict.get("grid", {}) or {}
    optim = cfg_dict.get("optim", {}) or {}
    physics = cfg_dict.get("physics", {}) or {}
    rar = cfg_dict.get("rar", {}) or {}
    stl = cfg_dict.get("stl", {}) or {}
    io = cfg_dict.get("io", {}) or {}
    strel = cfg_dict.get("strel_audit", {}) or {}

    # STL: tolerate older key names (temp) vs newer (smoothmin)
    stl_temp = stl.get("smoothmin", stl.get("temp", None))
    if stl_temp is None:
        stl_temp = 0.1
    # STL penalty kind: allow either stl.kind or stl.penalty
    stl_kind = str(stl.get("kind", stl.get("penalty", "softplus"))).strip().lower()

    # STREL: parse formula (string or list) and optional flags
    fml_raw = strel.get("formula", strel.get("formulas", ("contain_hotspot",)))
    if isinstance(fml_raw, (list, tuple)):
        formulas = tuple(str(x) for x in fml_raw)
    else:
        formulas = (str(fml_raw),)

    strel_cfg = STRELAuditConfig(
        export=bool(strel.get("export", bool(strel))),
        run=bool(strel.get("run", False)),
        packed_field_path=str(strel.get("packed_field_path", "heat2d_field_xy_t.npy")),
        dt_path=str(strel.get("dt_path", "heat2d_dt.txt")),
        layout=str(strel.get("layout", "xy_t")),
        mls_path=str(strel.get("mls_path", "scripts/specs/contain_hotspot.mls")),
        formula=formulas,
        binarize=strel.get("binarize", True),
        threshold=strel.get("hot_threshold", strel.get("threshold", None)),
        moonlight=bool(strel.get("moonlight", True)),
        suppress_java_output=bool(strel.get("suppress_java_output", True)),
        plots=bool(strel.get("plots", True)),
        sweep=bool(strel.get("sweep", False)),
        json_out=str(strel.get("json_out", "heat2d_strel.json")),
        fig_dir=str(strel.get("fig_dir", "strel_figs")),
    )

    return Heat2DConfig(
        seed=int(cfg_dict.get("seed", 0)),
        device=str(cfg_dict.get("device", "auto")),
        dtype=str(cfg_dict.get("dtype", "float32")),
        hidden=_parse_int_tuple(model.get("hidden"), default=(64, 64, 64)),
        activation=str(model.get("activation", "tanh")),
        out_activation=str(model.get("out_activation", model.get("out_act", "identity"))),
        x_min=float(grid.get("x_min", 0.0)),
        x_max=float(grid.get("x_max", 1.0)),
        y_min=float(grid.get("y_min", 0.0)),
        y_max=float(grid.get("y_max", 1.0)),
        t_min=float(grid.get("t_min", 0.0)),
        t_max=float(grid.get("t_max", 1.0)),
        n_x=int(grid.get("n_x", 64)),
        n_y=int(grid.get("n_y", 64)),
        n_t=int(grid.get("n_t", 16)),
        lr=float(optim.get("lr", 1e-3)),
        epochs=int(optim.get("epochs", 200)),
        batch=int(optim.get("batch", 4096)),
        grad_clip=float(optim.get("grad_clip", 1.0)),
        amp=bool(optim.get("amp", False)),
        compile=bool(optim.get("compile", False)),
        alpha=float(physics.get("alpha", 0.1)),
        bcic_weight=float(physics.get("bcic_weight", 10.0)),
        n_boundary=int(physics.get("n_boundary", 512)),
        n_initial=int(physics.get("n_initial", 512)),
        ic_amplitude=float(physics.get("ic_amplitude", 1.0)),
        ic_sharpness=float(physics.get("ic_sharpness", 50.0)),
        ic_center_x=float(physics.get("ic_center_x", 0.5)),
        ic_center_y=float(physics.get("ic_center_y", 0.5)),
        use_dirichlet_mask_pow=int(physics.get("use_dirichlet_mask_pow", 0)),
        rar_pool=int(rar.get("pool", 0)),
        rar_hard_frac=float(rar.get("hard_frac", 0.1)),
        rar_every=int(rar.get("every", 10)),
        stl_use=bool(stl.get("use", False)),
        stl_weight=float(stl.get("weight", stl.get("stl_weight", 1.0))),
        stl_margin=float(stl.get("margin", 0.0)),
        stl_beta=float(stl.get("beta", 10.0)),
        stl_temp=float(stl_temp),
        stl_kind=stl_kind,  # type: ignore[arg-type]
        stl_u_min=(None if stl.get("u_min", None) is None else float(stl.get("u_min"))),
        stl_u_max=(None if stl.get("u_max", None) is None else float(stl.get("u_max"))),
        stl_x_min=float(stl.get("x_min", 0.0)),
        stl_x_max=float(stl.get("x_max", 1.0)),
        stl_y_min=float(stl.get("y_min", 0.0)),
        stl_y_max=float(stl.get("y_max", 1.0)),
        stl_nx=int(stl.get("n_x", 32)),
        stl_ny=int(stl.get("n_y", 32)),
        stl_nt=int(stl.get("n_t", 16)),
        stl_t_min=float(stl.get("t_min", 0.0)),
        stl_t_max=float(stl.get("t_max", 1.0)),
        stl_operator=str(stl.get("operator", "always")).strip().lower(),  # type: ignore[arg-type]
        stl_space_op=str(stl.get("space_op", "forall")).strip().lower(),  # type: ignore[arg-type]
        stl_window=int(stl.get("window", 0)),
        stl_stride=int(stl.get("stride", 1)),
        stl_every=int(stl.get("every", 5)),
        results_dir=str(io.get("run_dir", io.get("results_dir", "results"))),
        tag=str(cfg_dict.get("tag", io.get("tag", "run"))),
        save_ckpt=bool(io.get("save_ckpt", True)),
        save_frames=bool(io.get("save_frames", True)),
        frames_idx=_parse_int_tuple(io.get("frames_idx"), default=(0, 5, 10, 15)),
        save_figs=bool(io.get("save_figs", True)),
        print_every=int(io.get("print_every", 10)),
        strel_audit=strel_cfg,
    )


def _validate_cfg(cfg: Heat2DConfig) -> None:
    """Lightweight config validation with actionable error messages."""
    if cfg.n_x < 2 or cfg.n_y < 2:
        raise ValueError("grid.n_x and grid.n_y must be >= 2.")
    if cfg.n_t < 1:
        raise ValueError("grid.n_t must be >= 1.")
    if cfg.x_max <= cfg.x_min:
        raise ValueError("grid.x_max must be > grid.x_min.")
    if cfg.y_max <= cfg.y_min:
        raise ValueError("grid.y_max must be > grid.y_min.")
    if cfg.t_max < cfg.t_min:
        raise ValueError("grid.t_max must be >= grid.t_min.")
    if cfg.n_t > 1 and cfg.t_max == cfg.t_min:
        raise ValueError("grid.t_max must be > grid.t_min when grid.n_t > 1.")
    if cfg.epochs < 0:
        raise ValueError("optim.epochs must be >= 0.")
    if cfg.batch <= 0:
        raise ValueError("optim.batch must be > 0.")
    if cfg.grad_clip < 0:
        raise ValueError("optim.grad_clip must be >= 0.")
    if cfg.alpha <= 0:
        raise ValueError("physics.alpha must be > 0.")
    if cfg.bcic_weight < 0:
        raise ValueError("physics.bcic_weight must be >= 0.")
    if cfg.n_boundary < 0 or cfg.n_initial < 0:
        raise ValueError("physics.n_boundary and physics.n_initial must be >= 0.")
    if cfg.rar_pool < 0:
        raise ValueError("rar.pool must be >= 0.")
    if cfg.rar_pool > 0 and not (0.0 < cfg.rar_hard_frac <= 1.0):
        raise ValueError("rar.hard_frac must be in (0, 1] when rar.pool > 0.")
    if cfg.stl_use:
        if not _HAS_STL:
            raise RuntimeError("stl.use is true but monitoring.stl_soft could not be imported.")
        if cfg.stl_temp <= 0:
            raise ValueError("stl.smoothmin (temp) must be > 0.")
        if cfg.stl_beta <= 0:
            raise ValueError("stl.beta must be > 0.")
        if cfg.stl_weight < 0:
            raise ValueError("stl.stl_weight must be >= 0.")
        if cfg.stl_kind not in {"softplus", "logistic", "hinge", "sqhinge"}:
            raise ValueError("stl.kind must be one of: softplus, logistic, hinge, sqhinge.")
        if cfg.stl_u_min is not None and cfg.stl_u_max is not None and cfg.stl_u_min > cfg.stl_u_max:
            raise ValueError("stl.u_min must be <= stl.u_max.")
        if cfg.stl_nx < 1 or cfg.stl_ny < 1 or cfg.stl_nt < 1:
            raise ValueError("stl.n_x, stl.n_y, and stl.n_t must be >= 1.")
        if cfg.stl_operator not in {"always", "eventually"}:
            raise ValueError("stl.operator must be 'always' or 'eventually'.")
        if cfg.stl_space_op not in {"forall", "exists"}:
            raise ValueError("stl.space_op must be 'forall' or 'exists'.")
        if cfg.stl_window < 0 or cfg.stl_stride < 1:
            raise ValueError("stl.window must be >= 0 and stl.stride must be >= 1.")


def _export_packed_field_and_dt(
    *,
    cfg: Heat2DConfig,
    results_dir: Path,
    U: torch.Tensor,
    T: torch.Tensor,
) -> tuple[Path, Path, float]:
    """Export packed `.npy` field and `heat2d_dt.txt` sidecar for STREL tooling."""
    # dt from uniform time grid
    if cfg.n_t >= 2:
        dt = float((T[0, 0, 1] - T[0, 0, 0]).item())
    else:
        dt = 0.0

    field_path = _resolve_results_path(cfg.strel_audit.packed_field_path, results_dir=results_dir)
    dt_path = _resolve_results_path(cfg.strel_audit.dt_path, results_dir=results_dir)

    np.save(field_path, U.detach().cpu().numpy())
    dt_path.write_text(f"{dt:.16g}\n", encoding="utf-8")
    return field_path, dt_path, dt


def _maybe_run_strel_audit(
    *,
    cfg: Heat2DConfig,
    results_dir: Path,
    packed_field_path: Path,
    dt: float,
) -> Path | None:
    """Optionally run scripts/eval_heat2d_moonlight.py (best effort)."""
    if not cfg.strel_audit.run:
        return None

    eval_script = _resolve_project_path("scripts/eval_heat2d_moonlight.py")
    if not eval_script.exists():
        print(f"[heat2d] STREL audit requested, but eval script not found: {eval_script}")
        return None

    mls_path = _resolve_project_path(cfg.strel_audit.mls_path)
    if not mls_path.exists():
        print(f"[heat2d] STREL audit requested, but .mls spec not found: {mls_path}")
        return None

    out_json = _resolve_results_path(cfg.strel_audit.json_out, results_dir=results_dir)
    fig_dir = _resolve_results_path(cfg.strel_audit.fig_dir, results_dir=results_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        sys.executable,
        str(eval_script),
        "--field",
        str(packed_field_path),
        "--layout",
        str(cfg.strel_audit.layout),
        "--dt",
        str(dt),
        "--mls",
        str(mls_path),
        "--out-json",
        str(out_json),
        "--fig-dir",
        str(fig_dir),
    ]

    # Formula(s): allow parameterized calls.
    if cfg.strel_audit.formula:
        cmd += ["--formula", *list(cfg.strel_audit.formula)]

    # Binarization controls
    if cfg.strel_audit.binarize is True:
        cmd += ["--binarize"]
    elif cfg.strel_audit.binarize is False:
        cmd += ["--no-binarize"]

    if cfg.strel_audit.threshold is not None:
        cmd += ["--threshold", str(float(cfg.strel_audit.threshold))]

    if not cfg.strel_audit.moonlight:
        cmd += ["--no-moonlight"]
    if not cfg.strel_audit.plots:
        cmd += ["--no-plots"]
    if not cfg.strel_audit.sweep:
        cmd += ["--no-sweep"]
    if not cfg.strel_audit.suppress_java_output:
        cmd += ["--no-suppress-java-output"]

    print("[heat2d] Running STREL audit:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"[heat2d] STREL audit failed (continuing). ({e})")
        return None

    return out_json


# Runner


def run_heat2d(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    """Run the heat2d experiment. Returns a small metrics/artifacts dict."""
    cfg = _parse_config(cfg_dict)

    _validate_cfg(cfg)

    seed_everything(cfg.seed)
    device = _select_device(cfg.device)
    dtype = _resolve_dtype(cfg.dtype)

    # MPS has limited float64 support; fall back to float32 if needed.
    if device.type == "mps" and dtype == torch.float64:
        print("[heat2d] float64 on MPS is not well-supported; using float32 instead.")
        dtype = torch.float32

    torch.set_default_dtype(dtype)

    results_dir = _ensure_dir(cfg.results_dir)

    # Training log CSV
    csv_path = results_dir / f"heat2d_{cfg.tag}.csv"
    log = CSVLogger(
        csv_path,
        header=[
            "epoch",
            "lr",
            "wall_time_s",
            "epoch_time_s",
            "loss",
            "loss_pde",
            "loss_bc",
            "loss_ic",
            "loss_stl",
            "stl_rob",
        ],
    )

    # Full cartesian grid (used both for minibatch sampling and for final export)
    X, Y, T, XYT = grid2d(
        x_min=cfg.x_min,
        x_max=cfg.x_max,
        y_min=cfg.y_min,
        y_max=cfg.y_max,
        t_min=cfg.t_min,
        t_max=cfg.t_max,
        n_x=cfg.n_x,
        n_y=cfg.n_y,
        n_t=cfg.n_t,
        device=device,
        dtype=dtype,
        return_cartesian=True,
    )
    n_total = XYT.shape[0]
    batch = min(int(cfg.batch), int(n_total))

    # Model
    model_eager: nn.Module = MLP(
        in_dim=3,
        hidden=list(cfg.hidden),
        out_dim=1,
        activation=cfg.activation,
        out_activation=cfg.out_activation,
    ).to(device)

    if cfg.use_dirichlet_mask_pow and cfg.use_dirichlet_mask_pow > 0:
        mask = make_dirichlet_mask(
            x_min=cfg.x_min,
            x_max=cfg.x_max,
            y_min=cfg.y_min,
            y_max=cfg.y_max,
            pow=int(cfg.use_dirichlet_mask_pow),
        )
        model_eager = MaskedModel(model_eager, mask=mask).to(device)

    # Optimizer always uses eager parameters (compiled wrapper shares these).
    opt = Adam(model_eager.parameters(), lr=float(cfg.lr))

    # Optional compile (best effort + smoke test)
    model, compiled_ok = _maybe_compile_heat2d(
        model_eager,
        enable=bool(cfg.compile),
        device=device,
        dtype=dtype,
        alpha=float(cfg.alpha),
    )
    if cfg.compile and not compiled_ok:
        # keep cfg.compile unchanged (frozen dataclass), but record status via compiled_ok.
        pass

    # Optional AMP (only for CUDA)
    use_amp = bool(cfg.amp) and (device.type == "cuda")
    scaler: torch.amp.GradScaler | None
    if use_amp:
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = None

    # STL penalty object
    if _HAS_STL and cfg.stl_use:
        penalty = STLPenalty(
            weight=float(cfg.stl_weight), margin=float(cfg.stl_margin),
            kind=cfg.stl_kind, beta=float(cfg.stl_beta),
        )
    else:
        penalty = None

    # IC function (used by bc_ic_heat2d)
    ic_fn = lambda x, y: gaussian_ic(  # noqa: E731
        x,
        y,
        center=(float(cfg.ic_center_x), float(cfg.ic_center_y)),
        sharpness=float(cfg.ic_sharpness),
        amplitude=float(cfg.ic_amplitude),
    )

    t0_wall = time.perf_counter()

    # Main loop
    for epoch in range(int(cfg.epochs)):
        t_epoch0 = time.perf_counter()
        model.train()

        # Minibatch from full grid (deterministic given seed)
        idx = torch.randint(0, n_total, (batch,), device=device)
        coords = XYT[idx].detach().clone().requires_grad_(True)

        with (torch.amp.autocast("cuda") if use_amp else nullcontext()):
            # PDE residual loss
            res = residual_heat2d(model, coords, alpha=float(cfg.alpha))
            loss_pde = res.square().mean()

            # Boundary + initial losses (note: bc_ic_heat2d returns a tuple)
            loss_bc, loss_ic = bc_ic_heat2d(
                model,
                x_min=float(cfg.x_min),
                x_max=float(cfg.x_max),
                y_min=float(cfg.y_min),
                y_max=float(cfg.y_max),
                t_min=float(cfg.t_min),
                t_max=float(cfg.t_max),
                device=device,
                dtype=dtype,
                n_boundary=int(cfg.n_boundary),
                n_initial=int(cfg.n_initial),
                ic=ic_fn,
                seed=int(cfg.seed) + 1009 * epoch,
            )
            loss_bcic = loss_bc + loss_ic

            # Optional STL penalty (coarser and less frequent by default)
            if _HAS_STL and cfg.stl_use and penalty is not None and (epoch % max(1, int(cfg.stl_every)) == 0):
                loss_stl, rob = _stl_loss_and_robustness(cfg, model, penalty)
            else:
                loss_stl = torch.zeros((), device=device, dtype=dtype)
                rob = torch.tensor(float("nan"), device=device, dtype=dtype)

            loss = loss_pde + float(cfg.bcic_weight) * loss_bcic + loss_stl

        # Parameter update
        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model_eager.parameters(), float(cfg.grad_clip))
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model_eager.parameters(), float(cfg.grad_clip))
            opt.step()

        # RAR: add high-residual points to a pool and train on a hard subset
        if cfg.rar_pool and cfg.rar_pool > 0 and cfg.rar_every > 0 and (epoch % int(cfg.rar_every) == 0) and epoch > 0:
            with torch.no_grad():
                pool_idx = torch.randint(0, n_total, (int(cfg.rar_pool),), device=device)
                pool = XYT[pool_idx].detach().clone().requires_grad_(True)
            res_pool = residual_heat2d(model, pool, alpha=float(cfg.alpha)).squeeze(-1).abs()
            k = max(1, int(float(cfg.rar_hard_frac) * int(cfg.rar_pool)))
            hard_idx = torch.topk(res_pool, k=k, largest=True).indices
            hard = pool[hard_idx].detach().clone().requires_grad_(True)

            with (torch.amp.autocast("cuda") if use_amp else nullcontext()):
                res_hard = residual_heat2d(model, hard, alpha=float(cfg.alpha))
                loss_hard = res_hard.square().mean()

            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss_hard).backward()
                if cfg.grad_clip and cfg.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model_eager.parameters(), float(cfg.grad_clip))
                scaler.step(opt)
                scaler.update()
            else:
                loss_hard.backward()
                if cfg.grad_clip and cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model_eager.parameters(), float(cfg.grad_clip))
                opt.step()

        # Logging
        lr_now = float(opt.param_groups[0]["lr"])
        wall_now = time.perf_counter() - t0_wall
        epoch_dt = time.perf_counter() - t_epoch0
        log.append(
            [
                epoch,
                lr_now,
                wall_now,
                epoch_dt,
                float(loss.detach().cpu()),
                float(loss_pde.detach().cpu()),
                float(loss_bc.detach().cpu()),
                float(loss_ic.detach().cpu()),
                float(loss_stl.detach().cpu()),
                float(rob.detach().cpu()) if torch.isfinite(rob).item() else float("nan"),
            ]
        )

        if (epoch % max(1, int(cfg.print_every)) == 0) or (epoch == int(cfg.epochs) - 1):
            loss_val = float(loss.detach().cpu().item())
            loss_pde_val = float(loss_pde.detach().cpu().item())
            loss_bc_val = float(loss_bc.detach().cpu().item())
            loss_ic_val = float(loss_ic.detach().cpu().item())
            loss_stl_val = float(loss_stl.detach().cpu().item())
            rob_val = float(rob.detach().cpu().item()) if torch.isfinite(rob).item() else float("nan")
            rob_str = f"{rob_val: .3e}" if torch.isfinite(rob).item() else "   n/a"
            print(
                f"[heat2d] epoch={epoch:04d} lr={lr_now:.2e} "
                f"loss={loss_val:.4e} pde={loss_pde_val:.4e} "
                f"bc={loss_bc_val:.4e} ic={loss_ic_val:.4e} "
                f"stl={loss_stl_val:.4e} rob={rob_str} "
                f"({epoch_dt:.2f}s)"
            )

    total_train_s = time.perf_counter() - t0_wall

    # Artifacts
    artifacts: list[str] = [str(csv_path)]

    if cfg.save_ckpt:
        ckpt_path = results_dir / f"heat2d_{cfg.tag}.pt"
        torch.save({"model": model_eager.state_dict(), "config": asdict(cfg)}, ckpt_path)
        artifacts.append(str(ckpt_path))

    # Save final field on full grid
    model_eager.eval()
    with torch.no_grad():
        U = model_eager(XYT).reshape(cfg.n_x, cfg.n_y, cfg.n_t).detach().cpu()
        X_cpu, Y_cpu, T_cpu = X.detach().cpu(), Y.detach().cpu(), T.detach().cpu()

    field_path = results_dir / f"heat2d_{cfg.tag}_field.pt"
    torch.save(
        {"u": U, "X": X_cpu, "Y": Y_cpu, "T": T_cpu,
         "alpha": float(cfg.alpha), "config": asdict(cfg)},
        field_path,
    )
    artifacts.append(str(field_path))

    # Export packed field + dt for STREL tooling (optional but inexpensive)
    packed_field_path: Path | None = None
    dt_val: float | None = None
    if cfg.strel_audit.export:
        packed_field_path, _, dt_val = _export_packed_field_and_dt(cfg=cfg, results_dir=results_dir, U=U, T=T_cpu)
        artifacts.append(str(packed_field_path))
        artifacts.append(str(_resolve_results_path(cfg.strel_audit.dt_path, results_dir=results_dir)))

    # Simple time-series summary (u_min/u_mean/u_max over space per time)
    u_np = U.numpy()
    t_np = T_cpu[0, 0, :].numpy()
    u_min = u_np.reshape(-1, cfg.n_t).min(axis=0)
    u_max = u_np.reshape(-1, cfg.n_t).max(axis=0)
    u_mean = u_np.reshape(-1, cfg.n_t).mean(axis=0)
    stats_path = results_dir / f"heat2d_{cfg.tag}_u_stats.csv"
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("t,u_min,u_mean,u_max\n")
        for ti, mn, me, mx in zip(t_np, u_min, u_mean, u_max):
            f.write(f"{float(ti):.16g},{float(mn):.16g},{float(me):.16g},{float(mx):.16g}\n")
    artifacts.append(str(stats_path))

    # Frames + figures (optional)
    fig_paths: list[str] = []
    frame_paths: list[str] = []

    plt = None
    if cfg.save_figs:
        try:
            import matplotlib.pyplot as plt  # noqa: WPS433 (optional dependency)
        except Exception as e:  # pragma: no cover
            print(f"[heat2d] matplotlib unavailable; skipping figures. ({e})")
            plt = None  # type: ignore[assignment]

    if cfg.save_frames or cfg.save_figs:
        for k in cfg.frames_idx:
            if not (0 <= int(k) < int(cfg.n_t)):
                continue
            u_k = u_np[:, :, int(k)]
            t_k = float(t_np[int(k)])
            base = results_dir / f"heat2d_{cfg.tag}_t{k:03d}"

            if cfg.save_frames:
                npy_path = str(base) + "_u.npy"
                np.save(npy_path, u_k)
                frame_paths.append(npy_path)

            if cfg.save_figs and plt is not None:
                # Field heatmap
                fig = plt.figure()
                plt.imshow(
                    u_k.T,
                    origin="lower",
                    aspect="auto",
                    extent=[cfg.x_min, cfg.x_max, cfg.y_min, cfg.y_max],
                )
                plt.title(f"u(x,y,t) at t={t_k:.3f}")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.colorbar(label="u")
                out_u = str(base) + "_u.png"
                fig.savefig(out_u, dpi=200, bbox_inches="tight")
                plt.close(fig)
                fig_paths.append(out_u)

                # Gradient magnitude heatmap (visual edge/decay intuition)
                # Use simple central differences in index space.
                ux = np.zeros_like(u_k)
                uy = np.zeros_like(u_k)
                ux[1:-1, :] = 0.5 * (u_k[2:, :] - u_k[:-2, :])
                uy[:, 1:-1] = 0.5 * (u_k[:, 2:] - u_k[:, :-2])
                gradmag = np.sqrt(ux * ux + uy * uy)

                fig = plt.figure()
                plt.imshow(
                    gradmag.T,
                    origin="lower",
                    aspect="auto",
                    extent=[cfg.x_min, cfg.x_max, cfg.y_min, cfg.y_max],
                )
                plt.title(f"|∇u| at t={t_k:.3f}")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.colorbar(label="|∇u|")
                out_g = str(base) + "_gradmag.png"
                fig.savefig(out_g, dpi=200, bbox_inches="tight")
                plt.close(fig)
                fig_paths.append(out_g)

        # Time-series plot
        if cfg.save_figs and plt is not None:
            fig = plt.figure()
            plt.plot(t_np, u_max, label="max u")
            plt.plot(t_np, u_mean, label="mean u")
            plt.plot(t_np, u_min, label="min u")
            if cfg.stl_use and cfg.stl_u_max is not None:
                plt.axhline(float(cfg.stl_u_max), linestyle="--", label="U_max")
            plt.xlabel("t")
            plt.ylabel("u-statistic over space")
            plt.title("Heat2D: spatial statistics over time")
            plt.legend()
            out_ts = results_dir / f"heat2d_{cfg.tag}_u_stats.png"
            fig.savefig(out_ts, dpi=200, bbox_inches="tight")
            plt.close(fig)
            fig_paths.append(str(out_ts))

    artifacts.extend(frame_paths)
    artifacts.extend(fig_paths)

    # Optional STREL audit run
    strel_json: str | None = None
    if cfg.strel_audit.run and (packed_field_path is not None) and (dt_val is not None):
        out = _maybe_run_strel_audit(cfg=cfg, results_dir=results_dir, packed_field_path=packed_field_path, dt=dt_val)
        if out is not None:
            strel_json = str(out)
            artifacts.append(strel_json)

    # Final robustness for the configured STL spec (if enabled)
    final_rob: float | None = None
    if _HAS_STL and cfg.stl_use and penalty is not None:
        model_eager.eval()
        with torch.no_grad():
            _, rob = _stl_loss_and_robustness(cfg, model_eager, penalty)
            final_rob = float(rob.detach().cpu())

    return {
        "experiment": "heat2d",
        "tag": cfg.tag,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "compiled": bool(compiled_ok),
        "train_time_s": float(total_train_s),
        "final_stl_rob": final_rob,
        "artifacts": artifacts,
        "num_figures": len(fig_paths),
        "strel_json": strel_json,
    }
