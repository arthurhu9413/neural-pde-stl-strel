#!/usr/bin/env python3
# ruff: noqa: I001
from __future__ import annotations

"""Train TorchPhysics Burgers' PINN with STL monitoring/regularization.

This script is a lightweight example that matches the project theme:
*physics-ML + formal specification monitoring.*

We solve the 1-D *viscous Burgers' equation* on a rectangular space-time domain:

    u_t + u u_x − ν u_xx = 0,   (x, t) ∈ [x_min, x_max] × [t_min, t_max]

with typical PINN-style constraints:
  * initial condition: u(x, t_min) = u0(x)
  * boundary conditions: u(x_min, t) = u(x_max, t) = 0

We additionally **monitor** (and optionally **train with**) differentiable STL constraints
over a *discretized* space-time grid. This is specifically meant to support the
project requirements:

  * provide **concrete, actually-run examples** with **specs written out**
  * show **plots/figures** of PDE behavior plus **monitoring results**
  * clearly explain the **λ** parameter (here: `stl_weight`) and data-flow
  * record **runtime + environment/hardware info** for empirical reproducibility

High-level block diagram (framework + connections)

   +--------------------+          +------------------------+
   | TorchPhysics       |          | STL / monitoring        |
   | - spaces/domains   |          | - differentiable robust |
   | - samplers         |          |   semantics (softmin/max|
   | - PINN conditions  |          |   approximations)       |
   +---------+----------+          +-----------+------------+
             |                                 |
             v                                 v
      sampled points (x,t)                 spec φ over u(x,t)
             |                                 |
             +------------------+--------------+
                                v
                      +------------------+
                      | PINN uθ(x,t)     |
                      | (TorchPhysics    |
                      |  model)          |
                      +--------+---------+
                               |
                               v
                     +--------------------+
                     | total loss         |
                     |  = PDE + IC + BC   |
                     |    + λ·STL penalty |
                     +---------+----------+
                               |
                               v
                          optimizer
                               |
                               v
                       trained parameters θ*

λ (lambda) corresponds to **`stl_weight`** (and optionally
`stl_event_weight`) in this script: it scales how strongly STL robustness is
enforced during training.

Example data-flow diagram (this specific script)

Inputs:
  * PDE: Burgers' equation + ν
  * IC/BC: u0(x), boundary u=0
  * NN architecture: MLP (TorchPhysics FCN)
  * STL specs (examples):
      (S1) Safety bound:     G_[t_min,t_max] ( max_x |u(x,t)| ≤ U_max )
      (S2) Eventual bound:   F_[t_e0,t_e1]   ( max_x |u(x,t)| ≤ U_event )

Outputs:
  * trained model checkpoint: burgers1d_<tag>.pt
  * evaluated field: burgers1d_<tag>_field.pt
  * plots: field heatmap, profiles, max|u| time-series with thresholds
  * monitoring summary JSON + timeseries CSV (robustness/margins)

Notes on discretization / quantifiers (space + time)

The spatial universal quantifier "for all x" and temporal operators are approximated
on a fixed grid of size (stl_grid_nt × stl_grid_nx) for differentiable training
(and on a denser (eval_nt × eval_nx) grid for reporting).

This is the standard "sampled semantics" approach: bigger grids approximate the
continuous semantics more closely.

Dependencies

This script is *optional* and only runs if TorchPhysics + PyTorch Lightning are installed.

Recommended:
  pip install -r requirements-extra.txt

If missing, the script writes a placeholder artifact explaining what to install.

"""

import argparse
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

def _ensure_repo_src_on_path() -> None:
    """Ensure `src/` is importable when running `python scripts/...py`."""
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if src.exists():
        src_str = str(src)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)


_ensure_repo_src_on_path()

try:
    from neural_pde_stl_strel.utils.seed import seed_everything as _seed_everything
except Exception:  # pragma: no cover
    _seed_everything = None

try:
    import importlib.metadata as _importlib_metadata  # py3.8+
except Exception:  # pragma: no cover
    _importlib_metadata = None

try:
    import torch
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyTorch is required for this script.") from e

try:
    import pytorch_lightning as pl
except Exception:  # pragma: no cover
    pl = None

try:
    import torchphysics as tp
except Exception:  # pragma: no cover
    tp = None

def stl_softmax(x: torch.Tensor, *, temp: float, dim: int, keepdim: bool = False) -> torch.Tensor:
    """Smooth max via log-sum-exp: τ log ∑ exp(x/τ). Approaches max as τ->0+."""
    if temp <= 0:
        raise ValueError("temp must be > 0 for stl_softmax")
    tau = x.new_tensor(float(temp))
    return tau * torch.logsumexp(x / tau, dim=dim, keepdim=keepdim)


def stl_softmin(x: torch.Tensor, *, temp: float, dim: int, keepdim: bool = False) -> torch.Tensor:
    """Smooth min via -softmax(-x). Approaches min as τ->0+."""
    return -stl_softmax(-x, temp=temp, dim=dim, keepdim=keepdim)


def _time_window_indices(t_grid: torch.Tensor, *, t0: float, t1: float) -> slice:
    """
    Return a slice selecting indices in [t0,t1] (clamped) for a uniform 1D grid.
    """
    if t_grid.ndim != 1:
        raise ValueError("t_grid must be 1D")
    n = int(t_grid.numel())
    if n < 2:
        return slice(0, n)
    t_min = float(t_grid[0].item())
    t_max = float(t_grid[-1].item())
    dt = (t_max - t_min) / (n - 1)

    def _idx(t: float) -> int:
        return int(round((t - t_min) / dt))

    i0 = max(0, min(n - 1, _idx(t0)))
    i1 = max(0, min(n - 1, _idx(t1)))
    if i1 < i0:
        i0, i1 = i1, i0
    return slice(i0, i1 + 1)

@dataclass(frozen=True)
class Config:
    # Domain
    x_min: float = 0.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0
    nu: float = 0.01 / math.pi

    # Model
    hidden: Tuple[int, ...] = (64, 64, 64)
    activation: Literal["tanh", "gelu", "sin"] = "tanh"
    normalize_inputs: bool = True

    # Sampling (PINN conditions)
    n_pde: int = 4096
    n_bc: int = 512
    n_ic: int = 512

    # Optimization / training
    lr: float = 1e-3
    max_steps: int = 5000
    precision: Literal[32, 16] = 32
    deterministic: bool = True
    seed: int = 1234
    threads: int = 1
    device: Literal["auto", "cpu", "cuda"] = "auto"
    progress_bar: bool = True

    # STL (λ is stl_weight)
    stl_weight: float = 1.0  # λ for always (safety) spec
    stl_event_weight: float = 0.0  # λ for eventual spec (optional)
    stl_warmup: int = 0  # steps with STL weights set to 0
    stl_temp: float = 0.02  # τ for softmin/softmax
    u_max: float = 1.0  # U_max in (S1)

    # Eventual spec params (S2)
    t_event_start: float = 0.5
    t_event_end: float = 1.0
    u_event_max: float = 0.5

    # STL grid used during training (differentiable monitoring)
    stl_grid_nx: int = 64
    stl_grid_nt: int = 64

    # Evaluation grid (for plots + hard monitoring summary)
    eval_nx: int = 256
    eval_nt: int = 256

    # Output
    tag: str = "burgers_demo"
    results_dir: str = "results/torchphysics_burgers"
    save_ckpt: bool = True
    save_figs: bool = True


def _add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_str: str) -> None:
    """Add --<name> / --no-<name> flags. Name should be a valid CLI token."""
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_str)
    group.add_argument(f"--no-{name}", dest=dest, action="store_false", help=f"Disable: {help_str}")
    parser.set_defaults(**{dest: default})


def _parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    p = argparse.ArgumentParser(
        description="Train TorchPhysics PINN for 1D viscous Burgers' equation with STL monitoring.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # IO
    p.add_argument("--tag", type=str, default=Config.tag, help="Run tag (subfolder name).")
    p.add_argument("--results_dir", type=str, default=Config.results_dir, help="Base results folder.")
    _add_bool_arg(p, "save-ckpt", Config.save_ckpt, "Save a model checkpoint (.pt).")
    _add_bool_arg(p, "save-figs", Config.save_figs, "Save figures (.png).")

    # System
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--threads", type=int, default=Config.threads)
    p.add_argument("--device", type=str, default=Config.device, choices=["auto", "cpu", "cuda"])
    _add_bool_arg(p, "deterministic", Config.deterministic, "Use deterministic algorithms where possible.")
    _add_bool_arg(p, "progress-bar", Config.progress_bar, "Show Lightning progress bar.")

    # Domain
    p.add_argument("--x_min", type=float, default=Config.x_min)
    p.add_argument("--x_max", type=float, default=Config.x_max)
    p.add_argument("--t_min", type=float, default=Config.t_min)
    p.add_argument("--t_max", type=float, default=Config.t_max)
    p.add_argument("--nu", type=float, default=Config.nu)

    # Model
    p.add_argument("--hidden", nargs="+", type=int, default=list(Config.hidden))
    p.add_argument("--activation", type=str, choices=["tanh", "gelu", "sin"], default=Config.activation)
    _add_bool_arg(p, "normalize-inputs", Config.normalize_inputs, "Add TorchPhysics NormalizationLayer (if available).")

    # Training
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--max_steps", type=int, default=Config.max_steps)
    p.add_argument("--precision", type=int, choices=[16, 32], default=int(Config.precision))

    # Sampling
    p.add_argument("--n_pde", type=int, default=Config.n_pde)
    p.add_argument("--n_bc", type=int, default=Config.n_bc)
    p.add_argument("--n_ic", type=int, default=Config.n_ic)

    # STL
    p.add_argument("--stl_weight", type=float, default=Config.stl_weight, help="λ for always/safety spec (S1).")
    p.add_argument("--stl_event_weight", type=float, default=Config.stl_event_weight, help="λ for eventual spec (S2).")
    p.add_argument("--stl_warmup", type=int, default=Config.stl_warmup, help="Warmup steps with STL disabled.")
    p.add_argument("--stl_temp", type=float, default=Config.stl_temp, help="τ for softmin/softmax.")
    p.add_argument("--u_max", type=float, default=Config.u_max, help="U_max bound in (S1).")
    p.add_argument("--u_event_max", type=float, default=Config.u_event_max, help="U_event bound in (S2).")
    p.add_argument("--t_event_start", type=float, default=Config.t_event_start)
    p.add_argument("--t_event_end", type=float, default=Config.t_event_end)
    p.add_argument("--stl_grid_nx", type=int, default=Config.stl_grid_nx)
    p.add_argument("--stl_grid_nt", type=int, default=Config.stl_grid_nt)

    # Eval
    p.add_argument("--eval_nx", type=int, default=Config.eval_nx)
    p.add_argument("--eval_nt", type=int, default=Config.eval_nt)

    a = p.parse_args(argv)

    cfg = Config(
        # IO
        tag=str(a.tag),
        results_dir=str(a.results_dir),
        save_ckpt=bool(a.save_ckpt),
        save_figs=bool(a.save_figs),
        # System
        seed=int(a.seed),
        threads=int(a.threads),
        device=str(a.device),
        deterministic=bool(a.deterministic),
        progress_bar=bool(a.progress_bar),
        # Domain
        x_min=float(a.x_min),
        x_max=float(a.x_max),
        t_min=float(a.t_min),
        t_max=float(a.t_max),
        nu=float(a.nu),
        # Model
        hidden=tuple(int(x) for x in a.hidden),
        activation=str(a.activation),  # type: ignore[arg-type]
        normalize_inputs=bool(a.normalize_inputs),
        # Training
        lr=float(a.lr),
        max_steps=int(a.max_steps),
        precision=int(a.precision),  # type: ignore[arg-type]
        # Sampling
        n_pde=int(a.n_pde),
        n_bc=int(a.n_bc),
        n_ic=int(a.n_ic),
        # STL
        stl_weight=float(a.stl_weight),
        stl_event_weight=float(a.stl_event_weight),
        stl_warmup=int(a.stl_warmup),
        stl_temp=float(a.stl_temp),
        u_max=float(a.u_max),
        u_event_max=float(a.u_event_max),
        t_event_start=float(a.t_event_start),
        t_event_end=float(a.t_event_end),
        stl_grid_nx=int(a.stl_grid_nx),
        stl_grid_nt=int(a.stl_grid_nt),
        # Eval
        eval_nx=int(a.eval_nx),
        eval_nt=int(a.eval_nt),
    )

    # Basic validation (avoid silent nonsense)
    if cfg.x_max <= cfg.x_min:
        raise ValueError("Require x_max > x_min.")
    if cfg.t_max <= cfg.t_min:
        raise ValueError("Require t_max > t_min.")
    if cfg.nu <= 0:
        raise ValueError("Require nu > 0.")
    if cfg.max_steps <= 0:
        raise ValueError("Require max_steps > 0.")
    if cfg.stl_temp <= 0:
        raise ValueError("Require stl_temp > 0.")
    if cfg.u_max < 0 or cfg.u_event_max < 0:
        raise ValueError("Require nonnegative thresholds u_max/u_event_max.")
    if cfg.stl_weight < 0 or cfg.stl_event_weight < 0:
        raise ValueError("Require nonnegative STL weights.")
    if cfg.stl_warmup < 0:
        raise ValueError("Require stl_warmup >= 0.")
    if cfg.stl_grid_nx < 2 or cfg.stl_grid_nt < 2:
        raise ValueError("Require stl_grid_nx/stl_grid_nt >= 2.")
    if cfg.eval_nx < 2 or cfg.eval_nt < 2:
        raise ValueError("Require eval_nx/eval_nt >= 2.")
    return cfg

def _pkg_version(name: str) -> Optional[str]:
    if _importlib_metadata is None:
        return None
    try:
        return _importlib_metadata.version(name)
    except Exception:
        return None


def _env_summary(device: torch.device) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": getattr(torch, "get_num_interop_threads", lambda: None)(),
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "torchphysics": _pkg_version("torchphysics"),
        "pytorch_lightning": _pkg_version("pytorch-lightning") or _pkg_version("pytorch_lightning"),
    }
    if torch.cuda.is_available():
        try:
            idx = device.index if device.type == "cuda" else 0
            if idx is None:
                idx = 0
            info["cuda_device_name"] = torch.cuda.get_device_name(idx)
            info["cuda_capability"] = torch.cuda.get_device_capability(idx)
        except Exception:
            pass
    return info


def _select_device(device_pref: str) -> torch.device:
    if device_pref == "cpu":
        return torch.device("cpu")
    if device_pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _setup_system(cfg: Config) -> torch.device:
    if cfg.threads and cfg.threads > 0:
        torch.set_num_threads(int(cfg.threads))

    if _seed_everything is not None:
        _seed_everything(cfg.seed, deterministic=cfg.deterministic)
    else:  # lightweight fallback
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

    # Make deterministic choice explicit (in case fallback seed didn't set it)
    try:
        torch.use_deterministic_algorithms(bool(cfg.deterministic))
    except Exception:
        # Not all builds support this call; ignore.
        pass

    return _select_device(cfg.device)

def _tp_get_grad():
    """Return a grad() function that works across TorchPhysics versions."""
    if tp is None:
        raise RuntimeError("TorchPhysics not available.")
    # Newer versions often export grad under tp.utils
    if hasattr(tp, "utils") and hasattr(tp.utils, "grad"):
        return tp.utils.grad
    # Older versions use torchphysics.utils.differentialoperators
    try:
        from torchphysics.utils.differentialoperators import grad as _grad  # type: ignore

        return _grad
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Could not locate TorchPhysics grad() operator.") from e


def _tp_get_PINNCondition():
    if tp is None:
        raise RuntimeError("TorchPhysics not available.")
    # Common: tp.conditions.PINNCondition
    if hasattr(tp, "conditions") and hasattr(tp.conditions, "PINNCondition"):
        return tp.conditions.PINNCondition
    # Older layouts: tp.problem.conditions.PINNCondition
    if hasattr(tp, "problem") and hasattr(tp.problem, "conditions") and hasattr(tp.problem.conditions, "PINNCondition"):
        return tp.problem.conditions.PINNCondition
    raise RuntimeError("Could not find TorchPhysics PINNCondition class.")


def _tp_make_points(coords: Dict[str, torch.Tensor], space) -> Any:
    """Points.from_coordinates signature differs slightly across TorchPhysics versions."""
    if tp is None:
        raise RuntimeError("TorchPhysics not available.")
    Points = getattr(getattr(tp, "spaces", None), "Points", None)
    if Points is None:
        raise RuntimeError("Could not find TorchPhysics Points class.")
    fn = getattr(Points, "from_coordinates", None)
    if fn is None:
        raise RuntimeError("TorchPhysics Points.from_coordinates not found.")
    try:
        return fn(coords, space)
    except TypeError:
        return fn(coords)


def _tp_make_FCN(input_space, output_space, *, hidden: Tuple[int, ...], act: torch.nn.Module):
    if tp is None:
        raise RuntimeError("TorchPhysics not available.")
    FCN = getattr(getattr(tp, "models", None), "FCN", None)
    if FCN is None:
        raise RuntimeError("TorchPhysics models.FCN not found.")

    # Try most common signatures
    for kwargs in (
        {"input_space": input_space, "output_space": output_space, "hidden": hidden, "activations": act},
        {"input_space": input_space, "output_space": output_space, "hidden": hidden, "activation": act},
        {"input_space": input_space, "output_space": output_space, "hidden": hidden},
    ):
        try:
            net = FCN(**kwargs)
            # If activations/activation not accepted above, attempt to set after.
            if "activations" not in kwargs and "activation" not in kwargs:
                if hasattr(net, "activations"):
                    net.activations = act
                elif hasattr(net, "activation"):
                    net.activation = act
            return net
        except TypeError:
            continue

    # Final fallback: positional
    try:
        return FCN(input_space, output_space, hidden=hidden, activations=act)
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Failed to construct TorchPhysics FCN; API mismatch?") from e


def _activation_module(name: str) -> torch.nn.Module:
    if name == "tanh":
        return torch.nn.Tanh()
    if name == "gelu":
        return torch.nn.GELU()
    if name == "sin":

        class Sin(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
                return torch.sin(x)

        return Sin()
    raise ValueError(f"Unknown activation: {name}")


def _lightning_precision(cfg: Config, device: torch.device) -> Any:
    """Map requested precision to a Lightning-friendly value."""
    if int(cfg.precision) == 16 and device.type == "cuda":
        return "16-mixed"
    # On CPU, avoid fp16 by default.
    return 32

def _maybe_write_placeholder(out_dir: Path, reason: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    msg = {
        "status": "skipped",
        "reason": reason,
        "how_to_install": "pip install -r requirements-extra.txt",
        "note": "TorchPhysics + PyTorch Lightning are optional extras; base repo runs without them.",
    }
    (out_dir / "SKIPPED_torchphysics_burgers.json").write_text(json.dumps(msg, indent=2), encoding="utf-8")
    print(json.dumps(msg, indent=2))

class _STLWeightWarmupCallback(pl.Callback):  # type: ignore[misc]
    """
    Updates `weight_state` dict after `warmup_steps` training steps.

    We keep STL weights *outside* TorchPhysics Condition weight semantics, so the
    STL residual functions can:
      * short-circuit computation when weight is 0
      * apply sqrt(weight) scaling in a way that's stable and explicit
    """

    def __init__(self, *, weight_state: Dict[str, float], final_weights: Dict[str, float], warmup_steps: int):
        super().__init__()
        self.weight_state = weight_state
        self.final_weights = final_weights
        self.warmup_steps = int(warmup_steps)
        self._done = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:  # noqa: ANN001
        if self._done:
            return
        if int(getattr(trainer, "global_step", 0)) >= self.warmup_steps:
            self.weight_state.update(self.final_weights)
            self._done = True

def main(argv: Optional[Sequence[str]] = None) -> int:
    cfg = _parse_args(argv)

    out_dir = Path(cfg.results_dir).expanduser().resolve() / cfg.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Always write config first for reproducibility
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True), encoding="utf-8")

    if tp is None or pl is None:
        _maybe_write_placeholder(
            out_dir,
            reason=f"Missing deps: torchphysics={tp is not None}, pytorch_lightning={pl is not None}",
        )
        return 0

    device = _setup_system(cfg)
    env = _env_summary(device)
    (out_dir / "env.json").write_text(json.dumps(env, indent=2, sort_keys=True), encoding="utf-8")

    X = tp.spaces.R1("x")
    T = tp.spaces.R1("t")
    U = tp.spaces.R1("u")

    Omega_x = tp.domains.Interval(space=X, lower_bound=cfg.x_min, upper_bound=cfg.x_max)
    Omega_t = tp.domains.Interval(space=T, lower_bound=cfg.t_min, upper_bound=cfg.t_max)
    Omega = Omega_x * Omega_t

    act = _activation_module(cfg.activation)
    fcn = _tp_make_FCN(X * T, U, hidden=cfg.hidden, act=act)

    layers = []
    if cfg.normalize_inputs:
        Norm = getattr(getattr(tp, "models", None), "NormalizationLayer", None)
        if Norm is not None:
            try:
                layers.append(Norm(Omega))
            except Exception:
                # NormalizationLayer exists but signature differs; just skip.
                pass

    if layers:
        Seq = getattr(getattr(tp, "models", None), "Sequential", None)
        if Seq is None:
            # Fall back to torch.nn.Sequential if TorchPhysics doesn't provide one.
            model = torch.nn.Sequential(*layers, fcn)  # type: ignore[arg-type]
        else:
            model = Seq(*layers, fcn)
    else:
        model = fcn

    grad = _tp_get_grad()
    PINNCondition = _tp_get_PINNCondition()

    nu = float(cfg.nu)

    def u0(x: torch.Tensor) -> torch.Tensor:
        # Smooth initial condition on [x_min, x_max]; standard demo uses -sin(pi x) on [0,1]
        x01 = (x - float(cfg.x_min)) / (float(cfg.x_max) - float(cfg.x_min))
        return -torch.sin(math.pi * x01)

    def residual_pde(u: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        u_t = grad(u, t)
        u_x = grad(u, x)
        u_xx = grad(u_x, x)
        return u_t + u * u_x - nu * u_xx

    def residual_bc(u: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return u  # enforce u=0 at x boundaries

    def residual_ic(u: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return u - u0(x)

    S_pde = tp.samplers.RandomUniformSampler(domain=Omega, n_points=int(cfg.n_pde))
    S_bc = tp.samplers.RandomUniformSampler(domain=Omega_x.boundary * Omega_t, n_points=int(cfg.n_bc))
    S_ic = tp.samplers.RandomUniformSampler(domain=Omega_x * Omega_t.boundary_left, n_points=int(cfg.n_ic))

    stl_x = torch.linspace(cfg.x_min, cfg.x_max, int(cfg.stl_grid_nx), device=device).view(-1)
    stl_t = torch.linspace(cfg.t_min, cfg.t_max, int(cfg.stl_grid_nt), device=device).view(-1)
    TT, XX = torch.meshgrid(stl_t, stl_x, indexing="ij")
    stl_points = _tp_make_points({"x": XX.reshape(-1, 1), "t": TT.reshape(-1, 1)}, X * T)
    S_stl = tp.samplers.DataSampler(stl_points)

    # Mutable STL weights (warmup can set these to 0 then restore).
    stl_weight_state: Dict[str, float] = {"always": float(cfg.stl_weight), "event": float(cfg.stl_event_weight)}
    stl_final_weights = dict(stl_weight_state)

    if cfg.stl_warmup > 0:
        stl_weight_state["always"] = 0.0
        stl_weight_state["event"] = 0.0

    event_sl = _time_window_indices(stl_t.detach().cpu(), t0=float(cfg.t_event_start), t1=float(cfg.t_event_end))

    def _stl_robustness_always(u_grid: torch.Tensor) -> torch.Tensor:
        # signal(t) = max_x |u(x,t)|
        s_t = stl_softmax(u_grid.abs(), temp=float(cfg.stl_temp), dim=1)
        margins_t = float(cfg.u_max) - s_t
        return stl_softmin(margins_t, temp=float(cfg.stl_temp), dim=0)

    def _stl_robustness_eventual(u_grid: torch.Tensor) -> torch.Tensor:
        s_t = stl_softmax(u_grid.abs(), temp=float(cfg.stl_temp), dim=1)
        margins_t = float(cfg.u_event_max) - s_t
        return stl_softmax(margins_t[event_sl], temp=float(cfg.stl_temp), dim=0)

    def residual_stl_always(u: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        w = float(stl_weight_state["always"])
        if w <= 0.0:
            return torch.zeros_like(u)
        u_grid = u.reshape(int(cfg.stl_grid_nt), int(cfg.stl_grid_nx))
        rho = _stl_robustness_always(u_grid)
        viol = torch.relu(-rho)  # hinge on negative robustness
        # PINNCondition typically computes MSE(residual), so scale by sqrt(w)
        r = viol * math.sqrt(w)
        return r.expand_as(u)

    def residual_stl_event(u: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        w = float(stl_weight_state["event"])
        if w <= 0.0:
            return torch.zeros_like(u)
        u_grid = u.reshape(int(cfg.stl_grid_nt), int(cfg.stl_grid_nx))
        rho = _stl_robustness_eventual(u_grid)
        viol = torch.relu(-rho)
        r = viol * math.sqrt(w)
        return r.expand_as(u)

    cond_pde = PINNCondition(module=model, sampler=S_pde, residual_fn=residual_pde, name="pde")
    cond_bc = PINNCondition(module=model, sampler=S_bc, residual_fn=residual_bc, name="bc")
    cond_ic = PINNCondition(module=model, sampler=S_ic, residual_fn=residual_ic, name="ic")

    train_conditions = [cond_pde, cond_bc, cond_ic]

    # Add STL conditions if either weight > 0 or warmup is requested (so it can activate later).
    if float(cfg.stl_weight) > 0.0 or int(cfg.stl_warmup) > 0:
        cond_stl = PINNCondition(module=model, sampler=S_stl, residual_fn=residual_stl_always, name="stl_always")
        train_conditions.append(cond_stl)

    if float(cfg.stl_event_weight) > 0.0 or int(cfg.stl_warmup) > 0:
        cond_stl_e = PINNCondition(module=model, sampler=S_stl, residual_fn=residual_stl_event, name="stl_event")
        train_conditions.append(cond_stl_e)

    OptimizerSetting = getattr(tp, "OptimizerSetting", None) or getattr(tp.solver, "OptimizerSetting", None)
    if OptimizerSetting is None:
        raise RuntimeError("Could not find TorchPhysics OptimizerSetting.")

    opt_setting = OptimizerSetting(optimizer_class=torch.optim.Adam, lr=float(cfg.lr))
    solver = tp.solver.Solver(train_conditions=train_conditions, optimizer_setting=opt_setting)

    callbacks = []
    if cfg.stl_warmup > 0:
        callbacks.append(
            _STLWeightWarmupCallback(
                weight_state=stl_weight_state,
                final_weights=stl_final_weights,
                warmup_steps=int(cfg.stl_warmup),
            )
        )

    trainer = pl.Trainer(
        max_steps=int(cfg.max_steps),
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1,
        precision=_lightning_precision(cfg, device),
        deterministic=bool(cfg.deterministic),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=bool(cfg.progress_bar),
        default_root_dir=str(out_dir),
        log_every_n_steps=50,
        callbacks=callbacks,
    )

    t0_wall = time.perf_counter()
    trainer.fit(solver)
    t1_wall = time.perf_counter()
    runtime_sec = float(t1_wall - t0_wall)

    model.eval()
    eval_x = torch.linspace(cfg.x_min, cfg.x_max, int(cfg.eval_nx), device=device).view(-1, 1)
    eval_t = torch.linspace(cfg.t_min, cfg.t_max, int(cfg.eval_nt), device=device).view(-1, 1)
    TT2, XX2 = torch.meshgrid(eval_t.view(-1), eval_x.view(-1), indexing="ij")
    GX = _tp_make_points({"x": XX2.reshape(-1, 1), "t": TT2.reshape(-1, 1)}, X * T)

    with torch.no_grad():
        U_pred = model(GX).reshape(int(cfg.eval_nt), int(cfg.eval_nx)).detach().cpu()

    absU = U_pred.abs()
    max_abs_over_x = absU.max(dim=1).values  # shape [eval_nt]
    max_abs_over_xt = float(max_abs_over_x.max().item())

    # Hard robustness for (S1): u_max - max_{t} max_{x} |u|
    rho_always_hard = float(cfg.u_max) - max_abs_over_xt
    sat_always = rho_always_hard >= 0.0

    # Hard robustness for (S2): max_{t in window} (u_event_max - max_x|u|)
    # = u_event_max - min_{t in window} max_x|u|
    eval_t_cpu = eval_t.detach().cpu().view(-1)
    win = _time_window_indices(eval_t_cpu, t0=float(cfg.t_event_start), t1=float(cfg.t_event_end))
    min_max_abs_in_win = float(max_abs_over_x[win].min().item())
    rho_event_hard = float(cfg.u_event_max) - min_max_abs_in_win
    sat_event = rho_event_hard >= 0.0

    # Optional PDE residual RMSE on a small random sample (cost/accuracy signal)
    pde_rmse: Optional[float] = None
    try:
        sampler_err = tp.samplers.RandomUniformSampler(domain=Omega, n_points=min(2048, int(cfg.n_pde)))
        pts = sampler_err.sample_points(device=str(device)) if hasattr(sampler_err, "sample_points") else sampler_err()
        coords = getattr(pts, "coordinates", None)
        if isinstance(coords, dict) and "x" in coords and "t" in coords:
            x_err = coords["x"].detach().clone().requires_grad_(True)
            t_err = coords["t"].detach().clone().requires_grad_(True)
            pts2 = _tp_make_points({"x": x_err, "t": t_err}, X * T)
            u_err = model(pts2)
            r = residual_pde(u_err, x_err, t_err)
            pde_rmse = float(torch.sqrt(torch.mean(r**2)).detach().cpu().item())
    except Exception:
        pde_rmse = None

    ckpt_path = out_dir / f"burgers1d_{cfg.tag}.pt"
    field_path = out_dir / f"burgers1d_{cfg.tag}_field.pt"
    ts_csv = out_dir / f"burgers1d_{cfg.tag}_timeseries.csv"
    summary_path = out_dir / f"burgers1d_{cfg.tag}_summary.json"

    # artifacts list is used in the summary JSON
    artifacts = []
    if cfg.save_ckpt:
        torch.save(
            {"state_dict": model.state_dict(), "config": asdict(cfg), "env": env},
            ckpt_path,
        )
        artifacts.append(str(ckpt_path))

    torch.save(
        {
            "x": eval_x.detach().cpu(),
            "t": eval_t.detach().cpu(),
            "u": U_pred,
            "nu": float(cfg.nu),
            "config": asdict(cfg),
        },
        field_path,
    )
    artifacts.append(str(field_path))

    # timeseries CSV: t, max|u|, margins for both specs
    lines = ["t,max_abs_u,margin_always,margin_event"]
    t_list = eval_t.detach().cpu().view(-1).tolist()
    max_list = max_abs_over_x.detach().cpu().tolist()
    for tval, mval in zip(t_list, max_list, strict=True):
        lines.append(
            f"{tval:.8g},{mval:.8g},{(float(cfg.u_max)-mval):.8g},{(float(cfg.u_event_max)-mval):.8g}"
        )
    ts_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifacts.append(str(ts_csv))

    # figures
    fig_field = out_dir / f"burgers1d_{cfg.tag}_field.png"
    fig_profiles = out_dir / f"burgers1d_{cfg.tag}_profiles.png"
    fig_ts = out_dir / f"burgers1d_{cfg.tag}_maxabs_timeseries.png"

    if cfg.save_figs:
        import matplotlib.pyplot as plt

        # Field heatmap
        plt.figure()
        plt.imshow(
            U_pred.numpy(),
            extent=[float(cfg.x_min), float(cfg.x_max), float(cfg.t_min), float(cfg.t_max)],
            origin="lower",
            aspect="auto",
        )
        plt.xlabel("x")
        plt.ylabel("t")
        plt.title("Burgers' PINN: u(x,t)")
        plt.colorbar(label="u")
        plt.tight_layout()
        plt.savefig(fig_field, dpi=200)
        plt.close()
        artifacts.append(str(fig_field))

        # Profiles at a few times
        plt.figure()
        x_cpu = eval_x.detach().cpu().view(-1).numpy()
        t_cpu = eval_t.detach().cpu().view(-1).numpy()
        for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
            t_target = float(cfg.t_min) + frac * (float(cfg.t_max) - float(cfg.t_min))
            idx = int(round((t_target - float(cfg.t_min)) / (float(cfg.t_max) - float(cfg.t_min)) * (cfg.eval_nt - 1)))
            idx = max(0, min(int(cfg.eval_nt) - 1, idx))
            plt.plot(x_cpu, U_pred[idx, :].numpy(), label=f"t={t_cpu[idx]:.3g}")
        plt.xlabel("x")
        plt.ylabel("u(x,t)")
        plt.title("Burgers' PINN profiles at selected times")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_profiles, dpi=200)
        plt.close()
        artifacts.append(str(fig_profiles))

        # STL signal time-series: max_x |u|
        plt.figure()
        plt.plot(t_cpu, max_abs_over_x.numpy(), label="max_x |u(x,t)|")
        plt.axhline(float(cfg.u_max), linestyle="--", label="U_max (S1)")
        plt.axhline(float(cfg.u_event_max), linestyle="--", label="U_event (S2)")
        plt.xlabel("t")
        plt.ylabel("max_x |u(x,t)|")
        plt.title("STL monitoring signal over time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_ts, dpi=200)
        plt.close()
        artifacts.append(str(fig_ts))

    # Summary JSON (include itself in artifacts list)
    artifacts.append(str(summary_path))
    summary = {
        "config": asdict(cfg),
        "env": env,
        "runtime_sec": runtime_sec,
        "specs": {
            "S1_always_safety": f"G_[{cfg.t_min},{cfg.t_max}] (max_x |u(x,t)| <= {cfg.u_max})",
            "S2_eventual": f"F_[{cfg.t_event_start},{cfg.t_event_end}] (max_x |u(x,t)| <= {cfg.u_event_max})",
            "notes": "Robustness is approximated on grids; training uses softmin/softmax, reporting uses hard max/min.",
        },
        "metrics": {
            "pde_rmse_sample": pde_rmse,
            "max_abs_over_xt": max_abs_over_xt,
            "rho_always_hard": rho_always_hard,
            "sat_always": sat_always,
            "rho_event_hard": rho_event_hard,
            "sat_event": sat_event,
        },
        "artifacts": artifacts,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    # Console summary (quick copy-paste into report)
    print("\n=== Burgers TorchPhysics + STL summary ===")
    print(f"out_dir: {out_dir}")
    print(f"runtime_sec: {runtime_sec:.3f}")
    print(f"pde_rmse_sample: {pde_rmse}")
    print(f"max_abs_over_xt: {max_abs_over_xt:.6g}")
    print(f"S1 rho_always_hard: {rho_always_hard:.6g}   satisfied={sat_always}")
    print(f"S2 rho_event_hard:  {rho_event_hard:.6g}   satisfied={sat_event}")
    print("Artifacts:")
    for a_path in artifacts:
        print(f"  - {a_path}")
    print("=========================================\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
