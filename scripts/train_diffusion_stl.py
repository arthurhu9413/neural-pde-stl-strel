#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ruff: noqa: I001
from __future__ import annotations

"""
train_diffusion_stl.py

Train a 1-D diffusion / heat-equation PINN with optional *differentiable* STL regularization.

This is a lightweight example (pure PyTorch + the small helpers in `neural_pde_stl_strel`)
supporting the neural-PDE + formal-specification research direction.

PDE problem (Diffusion1D / heat equation)

We learn a scalar field u(x,t) on a space-time rectangle:

    u_t = α u_xx,        (x,t) ∈ [x_min,x_max] × [t_min,t_max]

with homogeneous Dirichlet boundary conditions and a sine initial condition:

    u(x_min,t) = 0
    u(x_max,t) = 0
    u(x,t_min) = sin(π·(x-x_min)/(x_max-x_min))

The analytic solution for this setup is:

    u(x,t) = exp(-α·(π/L)^2·(t-t_min)) · sin(π·(x-x_min)/L),   where L = x_max-x_min

We use this analytic solution **only** for post-training evaluation plots/metrics.

Key features

- Prints and saves the exact STL specs monitored/trained.
- Lambda (--stl-weight) scales the STL penalty in the total loss.
- Spatial aggregation (--stl-spatial) controls how the field is reduced to a
  scalar time-series: amax (audit), softmax (training), mean, or point.
- Saves u(x,t) heatmaps, loss/robustness curves, and threshold overlays.
- Saves env.json with hardware/software info and per-epoch wall-clock times.

STL conventions used here

We use *sampled quantitative semantics* (robustness):

- Atomic predicate robustness:
    (u ≤ c)  has robustness   ρ = c - u
    (u ≥ c)  has robustness   ρ = u - c

- Temporal operators over a time-series of margins m(t):
    Always:     ρ(G m) = min_t m(t)
    Eventually: ρ(F m) = max_t m(t)

For differentiability we use smooth min/max (LogSumExp) with temperature τ:

    softmax_τ(x) := τ · log Σ_i exp(x_i / τ)
    softmin_τ(x) := -softmax_τ(-x)

Smaller τ gives a closer approximation to hard max/min, but is less smooth.

Outputs (by default)

A new run directory under `--results-dir`:

    results/diffusion1d--<tag>--<timestamp>/

containing:
  - env.json                       (hardware/software summary)
  - config.json                    (effective arguments)
  - spec.txt                       (human-readable spec string(s))
  - diffusion1d_<tag>.csv          (per-epoch logs)
  - diffusion1d_<tag>.pt           (checkpoint with model + optimizer)
  - diffusion1d_<tag>_field.pt     (dense u(x,t) field for plotting/monitoring)
  - monitoring.json                (post-training robustness for several specs)
  - figs/*.png                     (plots/figures; disable via --no-plots)

Example usage

Baseline (no STL penalty):
    python scripts/train_diffusion_stl.py --tag baseline --no-stl

STL-regularized training (smooth worst-case upper bound):
    python scripts/train_diffusion_stl.py --tag stl --stl-weight 5 --stl-spatial softmax --stl-u-max 1.05

Audit the final field with an "eventually cooling" spec (default extra monitoring):
    (done automatically unless you pass --no-extra-monitoring)

"""

import argparse
import json
import math
import os
import platform
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, NoReturn, Sequence

import torch

# Local-import helper (allows running from source checkout without pip install)


def _ensure_src_on_path() -> None:
    """
    Allow running from a source checkout without `pip install -e .`.

    If `neural_pde_stl_strel` is already importable, do nothing.
    Else add `<repo_root>/src` to `sys.path`.
    """
    try:
        import neural_pde_stl_strel  # noqa: F401

        return
    except Exception:
        repo_root = Path(__file__).resolve().parents[1]
        src_dir = repo_root / "src"
        if src_dir.is_dir():
            sys.path.insert(0, str(src_dir))


_ensure_src_on_path()

# Repo imports (kept after sys.path adjustment)

from neural_pde_stl_strel.models.mlp import MLP  # noqa: E402
from neural_pde_stl_strel.monitoring.stl_soft import (  # noqa: E402
    STLPenalty,
    always,
    always_window,
    eventually,
    eventually_window,
    pred_geq,
    pred_leq,
    soft_and,
    softmax as stl_softmax,
    softmin as stl_softmin,
)
from neural_pde_stl_strel.physics.diffusion1d import (  # noqa: E402
    MaskedModel,
    boundary_loss,
    make_dirichlet_mask_1d,
    residual_loss,
    sine_solution,
)
from neural_pde_stl_strel.training.grids import grid1d, sample_interior_1d  # noqa: E402
from neural_pde_stl_strel.utils.logger import CSVLogger  # noqa: E402
from neural_pde_stl_strel.utils.seed import seed_everything  # noqa: E402

SpatialAgg = Literal["mean", "softmax", "amax", "softmin", "amin", "point"]
STLPredSpec = Literal["upper", "lower", "range"]
STLOuter = Literal["always", "eventually"]
TimeMode = Literal["all", "window", "interval"]
PenaltyKind = Literal["softplus", "logistic", "hinge", "sqhinge"]


# Config


@dataclass
class Args:
    # model
    hidden: tuple[int, ...] = (64, 64, 64)
    activation: str = "tanh"
    out_act: str | None = None

    # grid/domain (dense field used for final evaluation + saved checkpoint)
    nx: int = 128
    nt: int = 64
    x_min: float = 0.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    # optimization
    lr: float = 2e-3
    epochs: int = 200
    batch: int = 4096
    opt: str = "adam"  # adam | adamw
    weight_decay: float = 0.0
    sched: str = "none"  # none | onecycle | cosine
    grad_clip: float = 0.0

    # physics
    alpha: float = 0.1
    n_boundary: int = 256
    n_initial: int = 512
    sample_method: str = "sobol"  # sobol | uniform
    w_boundary: float = 1.0
    w_initial: float = 1.0
    dirichlet_mask: bool = False  # enforce homogeneous Dirichlet BC via masking

    # STL (training-time semantics)
    stl_use: bool = True
    stl_weight: float = 5.0  # λ multiplier on STL penalty term
    stl_temp: float = 0.05  # τ for smooth min/max (smaller => closer to hard)
    stl_spatial: SpatialAgg = "softmax"
    stl_x_star: float = 0.5  # used when stl_spatial="point"

    # predicate: 'upper' enforces u <= u_max, 'lower' enforces u >= u_min,
    #            'range' enforces both (conjunction).
    stl_spec: STLPredSpec = "upper"
    stl_u_max: float = 1.05  # training-time safety bound (slightly above true peak=1.0)
    stl_u_min: float = 0.0

    # temporal aggregation
    stl_outer: STLOuter = "always"
    stl_time_mode: TimeMode = "all"
    stl_window: int = 16  # for time_mode='window': sliding window length (in coarse steps)
    stl_stride: int = 1  # for time_mode='window': sliding window stride
    stl_t0: float = 0.0  # for time_mode='interval': time units in [t_min, t_max]
    stl_t1: float = 1.0
    stl_every: int = 1  # evaluate STL every N epochs
    stl_nx: int = 64  # coarse grid for STL (space)
    stl_nt: int = 64  # coarse grid for STL (time)
    stl_warmup: int = 0  # epochs to wait before adding STL loss

    # STL penalty shape
    stl_penalty: PenaltyKind = "softplus"
    stl_beta: float = 10.0
    stl_margin: float = 0.0

    # monitoring extras (post-training audit specs)
    extra_monitoring: bool = True
    u_max_eval: float = 1.0  # audit-time stricter bound
    cool_t0: float = 0.8  # audit interval start (time units)
    cool_t1: float = 1.0  # audit interval end (time units)
    cool_u_max_loose: float = 0.40  # should be satisfiable for default alpha,t_max
    cool_u_max_tight: float = 0.30  # should be falsifiable for default alpha,t_max

    # system / numerics
    device: str | None = None  # "cpu" | "cuda" | "mps" | None(auto)
    dtype: str = "float32"  # float32 | float64
    amp: bool = False  # CUDA AMP (not recommended for PDE residuals; off by default)
    compile: bool = False
    seed: int = 0
    print_every: int = 25
    threads: int = 0  # 0 => leave default

    # I/O
    results_dir: str = "results"
    tag: str = "run"
    run_dir: str | None = None  # override output directory (disables auto timestamp)
    timestamp: bool = True
    save_ckpt: bool = True
    save_field: bool = True
    save_plots: bool = True
    plot_dpi: int = 180
    plot_formats: tuple[str, ...] = ("png",)
    resume: str | None = None


# CLI


def parse_args(argv: Sequence[str] | None = None) -> Args:
    p = argparse.ArgumentParser(
        description="Train a 1-D diffusion PINN with differentiable STL regularization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # model
    p.add_argument("--hidden", type=int, nargs="+", default=(64, 64, 64))
    p.add_argument("--activation", type=str, default="tanh")
    p.add_argument("--out-act", type=str, default=None)

    # grid/domain
    p.add_argument("--nx", type=int, default=128)
    p.add_argument("--nt", type=int, default=64)
    p.add_argument("--x-min", type=float, default=0.0)
    p.add_argument("--x-max", type=float, default=1.0)
    p.add_argument("--t-min", type=float, default=0.0)
    p.add_argument("--t-max", type=float, default=1.0)

    # optimization
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--opt", type=str, default="adam", choices=["adam", "adamw"])
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--sched", type=str, default="none", choices=["none", "onecycle", "cosine"])
    p.add_argument("--grad-clip", type=float, default=0.0)

    # physics
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--n-boundary", type=int, default=256)
    p.add_argument("--n-initial", type=int, default=512)
    p.add_argument("--sample-method", type=str, default="sobol", choices=["sobol", "uniform"])
    p.add_argument("--w-boundary", type=float, default=1.0)
    p.add_argument("--w-initial", type=float, default=1.0)
    p.add_argument("--dirichlet-mask", action="store_true", help="enforce BCs by construction")

    # STL (train-time)
    stl = p.add_argument_group("STL (training-time)")
    stl.add_argument("--no-stl", action="store_true", help="disable STL loss")
    stl.add_argument("--stl-weight", type=float, default=5.0, help="λ: scales STL penalty term")
    stl.add_argument("--stl-temp", type=float, default=0.05, help="τ for smooth min/max")
    stl.add_argument(
        "--stl-spatial",
        type=str,
        default="softmax",
        choices=["mean", "softmax", "amax", "softmin", "amin", "point"],
        help="spatial reduction to produce a 1-D time signal from u(x,t)",
    )
    stl.add_argument(
        "--stl-x-star",
        type=float,
        default=0.5,
        help="x* used when --stl-spatial point (nearest grid location)",
    )
    stl.add_argument(
        "--stl-spec",
        type=str,
        default="upper",
        choices=["upper", "lower", "range"],
        help="predicate type: upper(u<=Umax), lower(u>=Umin), range(Umin<=u<=Umax)",
    )
    stl.add_argument("--stl-u-max", type=float, default=1.05)
    stl.add_argument("--stl-u-min", type=float, default=0.0)
    stl.add_argument("--stl-outer", type=str, default="always", choices=["always", "eventually"])
    stl.add_argument("--stl-time-mode", type=str, default="all", choices=["all", "window", "interval"])
    stl.add_argument("--stl-window", type=int, default=16)
    stl.add_argument("--stl-stride", type=int, default=1)
    stl.add_argument("--stl-t0", type=float, default=0.0)
    stl.add_argument("--stl-t1", type=float, default=1.0)
    stl.add_argument("--stl-every", type=int, default=1)
    stl.add_argument("--stl-nx", type=int, default=64)
    stl.add_argument("--stl-nt", type=int, default=64)
    stl.add_argument("--stl-warmup", type=int, default=0)

    stl.add_argument(
        "--stl-penalty",
        type=str,
        default="softplus",
        choices=["softplus", "logistic", "hinge", "sqhinge", "relu", "exp"],
        help="shape of penalty applied to robustness (rob>margin desired)",
    )
    stl.add_argument("--stl-beta", type=float, default=10.0)
    stl.add_argument("--stl-margin", type=float, default=0.0)

    # post-training monitoring extras
    mon = p.add_argument_group("Monitoring extras (post-training)")
    mon.add_argument("--no-extra-monitoring", action="store_true", help="skip additional audit specs")
    mon.add_argument("--u-max-eval", type=float, default=1.0)
    mon.add_argument("--cool-t0", type=float, default=0.8)
    mon.add_argument("--cool-t1", type=float, default=1.0)
    mon.add_argument("--cool-u-max-loose", type=float, default=0.40)
    mon.add_argument("--cool-u-max-tight", type=float, default=0.30)

    # system
    p.add_argument("--device", type=str, default=None, help="cpu | cuda | mps | auto(None)")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--amp", action="store_true", help="CUDA mixed precision (not recommended)")
    p.add_argument("--compile", action="store_true", help="torch.compile the model (if available)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--print-every", type=int, default=25)
    p.add_argument("--threads", type=int, default=0)

    # I/O
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--tag", type=str, default="run")
    p.add_argument("--run-dir", type=str, default=None, help="override output directory")
    p.add_argument("--no-timestamp", action="store_true", help="disable auto timestamp naming")
    p.add_argument("--no-ckpt", action="store_true", help="do not save model checkpoint")
    p.add_argument("--no-field", action="store_true", help="do not save dense field checkpoint")
    p.add_argument("--no-plots", action="store_true", help="do not save plots/figures")
    p.add_argument("--plot-dpi", type=int, default=180)
    p.add_argument("--plot-formats", type=str, nargs="+", default=["png"])
    p.add_argument("--resume", type=str, default=None, help="path to .pt checkpoint to resume")

    ns = p.parse_args(argv)

    # Backwards-compatible aliases (older drafts used relu/exp names).
    penalty = str(ns.stl_penalty).lower()
    penalty_aliases = {"relu": "hinge", "exp": "logistic"}
    if penalty in penalty_aliases:
        print(
            f"WARNING: --stl-penalty {penalty!r} is an alias for "
            f"{penalty_aliases[penalty]!r}; using the canonical name.",
            file=sys.stderr,
        )
        penalty = penalty_aliases[penalty]

    return Args(
        hidden=tuple(int(x) for x in ns.hidden),
        activation=str(ns.activation),
        out_act=ns.out_act,
        nx=int(ns.nx),
        nt=int(ns.nt),
        x_min=float(ns.x_min),
        x_max=float(ns.x_max),
        t_min=float(ns.t_min),
        t_max=float(ns.t_max),
        lr=float(ns.lr),
        epochs=int(ns.epochs),
        batch=int(ns.batch),
        opt=str(ns.opt),
        weight_decay=float(ns.weight_decay),
        sched=str(ns.sched),
        grad_clip=float(ns.grad_clip),
        alpha=float(ns.alpha),
        n_boundary=int(ns.n_boundary),
        n_initial=int(ns.n_initial),
        sample_method=str(ns.sample_method),
        w_boundary=float(ns.w_boundary),
        w_initial=float(ns.w_initial),
        dirichlet_mask=bool(ns.dirichlet_mask),
        stl_use=not bool(ns.no_stl),
        stl_weight=float(ns.stl_weight),
        stl_temp=float(ns.stl_temp),
        stl_spatial=str(ns.stl_spatial),  # type: ignore[assignment]
        stl_x_star=float(ns.stl_x_star),
        stl_spec=str(ns.stl_spec),  # type: ignore[assignment]
        stl_u_max=float(ns.stl_u_max),
        stl_u_min=float(ns.stl_u_min),
        stl_outer=str(ns.stl_outer),  # type: ignore[assignment]
        stl_time_mode=str(ns.stl_time_mode),  # type: ignore[assignment]
        stl_window=int(ns.stl_window),
        stl_stride=int(ns.stl_stride),
        stl_t0=float(ns.stl_t0),
        stl_t1=float(ns.stl_t1),
        stl_every=int(ns.stl_every),
        stl_nx=int(ns.stl_nx),
        stl_nt=int(ns.stl_nt),
        stl_warmup=int(ns.stl_warmup),
        stl_penalty=penalty,  # type: ignore[assignment]
        stl_beta=float(ns.stl_beta),
        stl_margin=float(ns.stl_margin),
        extra_monitoring=not bool(ns.no_extra_monitoring),
        u_max_eval=float(ns.u_max_eval),
        cool_t0=float(ns.cool_t0),
        cool_t1=float(ns.cool_t1),
        cool_u_max_loose=float(ns.cool_u_max_loose),
        cool_u_max_tight=float(ns.cool_u_max_tight),
        device=ns.device,
        dtype=str(ns.dtype),
        amp=bool(ns.amp),
        compile=bool(ns.compile),
        seed=int(ns.seed),
        print_every=int(ns.print_every),
        threads=int(ns.threads),
        results_dir=str(ns.results_dir),
        tag=str(ns.tag),
        run_dir=ns.run_dir,
        timestamp=not bool(ns.no_timestamp),
        save_ckpt=not bool(ns.no_ckpt),
        save_field=not bool(ns.no_field),
        save_plots=not bool(ns.no_plots),
        plot_dpi=int(ns.plot_dpi),
        plot_formats=tuple(str(f).lower() for f in ns.plot_formats),
        resume=ns.resume,
    )


# Small utilities


def _fatal(msg: str, *, code: int = 2) -> NoReturn:
    """Print an error message and exit with a non-zero code."""
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(int(code))


def _as_torch_dtype(name: str) -> torch.dtype:
    try:
        return getattr(torch, name)
    except AttributeError as e:  # pragma: no cover
        raise ValueError(f"Unknown dtype: {name!r}") from e


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def _make_run_dir(cfg: Args) -> Path:
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if cfg.run_dir is not None:
        run_dir = Path(cfg.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    if cfg.timestamp:
        name = f"diffusion1d--{cfg.tag}--{_timestamp()}"
    else:
        name = f"diffusion1d--{cfg.tag}"
    run_dir = results_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _device_from_cfg(cfg: Args) -> torch.device:
    if cfg.device is None or str(cfg.device).lower() in {"auto", ""}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return torch.device("mps")
        return torch.device("cpu")

    dev = str(cfg.device).lower()
    if dev == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda requested but CUDA is unavailable; using CPU.", file=sys.stderr)
        return torch.device("cpu")
    return torch.device(dev)


def _env_summary(*, device: torch.device, cfg: Args) -> dict[str, Any]:
    """Collect a lightweight env/hardware summary for reproducibility."""
    info: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "cmd": " ".join(sys.argv),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        # Memory info (best-effort; Linux reads /proc/meminfo).
        "mem_total_gb": None,
        "mem_available_gb": None,
        "torch_version": torch.__version__,
        "torch_default_dtype": str(torch.get_default_dtype()).replace("torch.", ""),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "device": str(device),
        "seed": cfg.seed,
    }

    # Populate memory figures when possible.
    try:
        if Path("/proc/meminfo").is_file():
            mem_kb: dict[str, int] = {}
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if ":" not in line:
                    continue
                k, rest = line.split(":", 1)
                parts = rest.strip().split()
                if not parts:
                    continue
                try:
                    mem_kb[k] = int(parts[0])
                except ValueError:
                    continue
            if "MemTotal" in mem_kb:
                info["mem_total_gb"] = float(mem_kb["MemTotal"]) / 1e6
            if "MemAvailable" in mem_kb:
                info["mem_available_gb"] = float(mem_kb["MemAvailable"]) / 1e6
    except Exception:
        pass

    if device.type == "cuda":
        try:
            info["cuda_available"] = True
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(device.index or 0)
            try:
                props = torch.cuda.get_device_properties(device.index or 0)
                info["cuda_total_memory_gb"] = float(props.total_memory) / (1024**3)
            except Exception:
                pass
            info["cuda_capability"] = torch.cuda.get_device_capability(device.index or 0)
        except Exception:  # pragma: no cover
            pass
    else:
        info["cuda_available"] = bool(torch.cuda.is_available())

    if hasattr(torch.backends, "mps"):
        info["mps_available"] = bool(torch.backends.mps.is_available())  # type: ignore[attr-defined]

    return info


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)


def _maybe_compile(model: torch.nn.Module, *, enabled: bool) -> torch.nn.Module:
    if not enabled:
        return model
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:  # pragma: no cover
        print("WARNING: torch.compile not found in this PyTorch build; skipping.", file=sys.stderr)
        return model
    try:
        return compile_fn(model, mode="reduce-overhead")  # type: ignore[misc]
    except Exception as e:  # pragma: no cover
        print(f"WARNING: torch.compile failed ({e}); continuing without compile.", file=sys.stderr)
        return model


def _validate_cfg(cfg: Args) -> None:
    if cfg.x_max <= cfg.x_min:
        _fatal("Expected x_max > x_min")
    if cfg.t_max <= cfg.t_min:
        _fatal("Expected t_max > t_min")
    if cfg.nx < 2 or cfg.nt < 2:
        _fatal("Expected nx, nt >= 2")
    if cfg.epochs < 1:
        _fatal("Expected epochs >= 1")
    if cfg.batch < 1:
        _fatal("Expected batch >= 1")
    if cfg.alpha <= 0:
        _fatal("Expected alpha > 0")
    if cfg.sample_method not in {"sobol", "uniform"}:
        _fatal("sample_method must be 'sobol' or 'uniform'")

    if cfg.stl_use:
        if cfg.stl_temp <= 0:
            _fatal("stl_temp must be > 0")
        if cfg.stl_every < 1:
            _fatal("stl_every must be >= 1")
        if cfg.stl_nx < 2 or cfg.stl_nt < 2:
            _fatal("stl_nx and stl_nt must be >= 2")
        if cfg.stl_warmup < 0:
            _fatal("stl_warmup must be >= 0")
        if cfg.stl_time_mode == "window":
            if cfg.stl_window < 1:
                _fatal("stl_window must be >= 1")
            if cfg.stl_stride < 1:
                _fatal("stl_stride must be >= 1")
        if cfg.stl_time_mode == "interval":
            if not (math.isfinite(cfg.stl_t0) and math.isfinite(cfg.stl_t1)):
                _fatal("stl_t0 and stl_t1 must be finite numbers")
        if cfg.stl_spatial == "point":
            if not math.isfinite(cfg.stl_x_star):
                _fatal("stl_x_star must be finite when stl_spatial='point'")
            if not (cfg.x_min <= cfg.stl_x_star <= cfg.x_max):
                print(
                    "WARNING: stl_x_star is outside [x_min,x_max]; the nearest grid point will be used.",
                    file=sys.stderr,
                )

    if cfg.threads < 0:
        _fatal("threads must be >= 0")


def _spatial_pair(mode: SpatialAgg) -> SpatialAgg:
    """Return the complementary reduction for enforcing range bounds over all x."""
    pair: dict[SpatialAgg, SpatialAgg] = {
        "softmax": "softmin",
        "amax": "amin",
        "softmin": "softmax",
        "amin": "amax",
        "mean": "mean",
        "point": "point",
    }
    return pair[mode]


def _nearest_x_index(x_axis: torch.Tensor, x_star: float) -> int:
    x_cpu = x_axis.detach().float().cpu()
    return int(torch.argmin((x_cpu - float(x_star)).abs()).item())


def _spatial_reduce(
    u_xt: torch.Tensor,
    *,
    x_axis: torch.Tensor,
    mode: SpatialAgg,
    temp: float,
    x_star: float,
) -> torch.Tensor:
    """Reduce u(x,t) over x to obtain a 1-D signal over time."""
    if u_xt.ndim != 2:
        raise ValueError(f"Expected u_xt to have shape (Nx,Nt), got {tuple(u_xt.shape)}")
    if mode == "mean":
        return u_xt.mean(dim=0)
    if mode == "amax":
        return u_xt.amax(dim=0)
    if mode == "softmax":
        return stl_softmax(u_xt, temp=float(temp), dim=0, keepdim=False)
    if mode == "amin":
        return u_xt.amin(dim=0)
    if mode == "softmin":
        return stl_softmin(u_xt, temp=float(temp), dim=0, keepdim=False)
    if mode == "point":
        idx = _nearest_x_index(x_axis, x_star)
        return u_xt[idx, :]
    raise ValueError(f"Unknown spatial mode: {mode!r}")


def _time_slice(*, nt: int, t_min: float, t_max: float, t0: float, t1: float) -> slice:
    """Return a slice selecting the closed interval [t0,t1] (in time units)."""
    if nt < 2:
        return slice(0, nt)

    lo = max(min(t0, t1), t_min)
    hi = min(max(t0, t1), t_max)

    if hi <= lo:
        frac = 0.0 if t_max == t_min else (lo - t_min) / (t_max - t_min)
        idx = int(round(frac * (nt - 1)))
        idx = max(0, min(nt - 1, idx))
        return slice(idx, idx + 1)

    frac0 = (lo - t_min) / (t_max - t_min)
    frac1 = (hi - t_min) / (t_max - t_min)

    i0 = int(math.floor(frac0 * (nt - 1)))
    i1 = int(math.ceil(frac1 * (nt - 1)))

    i0 = max(0, min(nt - 1, i0))
    i1 = max(0, min(nt - 1, i1))

    return slice(i0, i1 + 1)


def _build_predicate_margins(*, u_xt: torch.Tensor, x_axis: torch.Tensor,
    cfg: Args) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Build per-time predicate margins m(t) whose sign indicates satisfaction.

    Returns:
      - margins: Tensor of shape (Nt,)
      - debug: traces used to construct margins (useful for plots)
    """
    tau = float(cfg.stl_temp)

    debug: dict[str, Any] = {
        "stl_spec": cfg.stl_spec,
        "stl_spatial": cfg.stl_spatial,
        "stl_x_star": float(cfg.stl_x_star),
        "stl_u_max": float(cfg.stl_u_max),
        "stl_u_min": float(cfg.stl_u_min),
    }

    if cfg.stl_spec == "upper":
        s = _spatial_reduce(u_xt, x_axis=x_axis, mode=cfg.stl_spatial, temp=tau, x_star=cfg.stl_x_star)
        m = pred_leq(s, float(cfg.stl_u_max))
        debug["signal"] = s.detach()
        debug["signal_kind"] = "upper_signal"
        return m, debug

    if cfg.stl_spec == "lower":
        s = _spatial_reduce(u_xt, x_axis=x_axis, mode=cfg.stl_spatial, temp=tau, x_star=cfg.stl_x_star)
        m = pred_geq(s, float(cfg.stl_u_min))
        debug["signal"] = s.detach()
        debug["signal_kind"] = "lower_signal"
        return m, debug

    if cfg.stl_spec == "range":
        mode_u = cfg.stl_spatial
        mode_l = _spatial_pair(cfg.stl_spatial)

        su = _spatial_reduce(u_xt, x_axis=x_axis, mode=mode_u, temp=tau, x_star=cfg.stl_x_star)
        sl = _spatial_reduce(u_xt, x_axis=x_axis, mode=mode_l, temp=tau, x_star=cfg.stl_x_star)

        mu = pred_leq(su, float(cfg.stl_u_max))
        ml = pred_geq(sl, float(cfg.stl_u_min))

        m = soft_and(mu, ml, temp=tau)
        debug["signal_upper"] = su.detach()
        debug["signal_lower"] = sl.detach()
        debug["signal_kind"] = "range_signals"
        debug["stl_spatial_upper"] = mode_u
        debug["stl_spatial_lower"] = mode_l
        return m, debug

    raise ValueError(f"Unknown stl_spec: {cfg.stl_spec!r}")


def _reduce_temporal(margins: torch.Tensor, *, cfg: Args) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Apply the temporal operator to margins and return scalar robustness.
    """
    if margins.ndim != 1:
        raise ValueError(f"Expected margins to be 1-D (Nt,), got {tuple(margins.shape)}")

    tau = float(cfg.stl_temp)
    nt = int(margins.shape[0])

    debug: dict[str, Any] = {
        "stl_outer": cfg.stl_outer,
        "stl_time_mode": cfg.stl_time_mode,
        "stl_window": int(cfg.stl_window),
        "stl_stride": int(cfg.stl_stride),
        "stl_t0": float(cfg.stl_t0),
        "stl_t1": float(cfg.stl_t1),
    }

    if cfg.stl_time_mode == "interval":
        sl = _time_slice(nt=nt, t_min=cfg.t_min, t_max=cfg.t_max, t0=cfg.stl_t0, t1=cfg.stl_t1)
        margins_use = margins[sl]
        debug["time_slice"] = [int(sl.start or 0), int((sl.stop or nt) - 1)]
    else:
        margins_use = margins

    if cfg.stl_time_mode == "window":
        window = int(cfg.stl_window)
        stride = int(cfg.stl_stride)
        if cfg.stl_outer == "always":
            rob_seq = always_window(margins_use, window=window, stride=stride, temp=tau, time_dim=0, keepdim=False)
            rob = stl_softmin(rob_seq, temp=tau, dim=0, keepdim=False) if rob_seq.numel() else rob_seq.new_tensor(0.0)
        else:
            rob_seq = eventually_window(margins_use, window=window, stride=stride, temp=tau, time_dim=0, keepdim=False)
            rob = stl_softmax(rob_seq, temp=tau, dim=0, keepdim=False) if rob_seq.numel() else rob_seq.new_tensor(0.0)

        debug["rob_trace"] = rob_seq.detach()
        return rob, debug

    if cfg.stl_outer == "always":
        rob = always(margins_use, temp=tau, time_dim=0)
    else:
        rob = eventually(margins_use, temp=tau, time_dim=0)

    return rob, debug


def compute_training_robustness(u_xt: torch.Tensor, *, x_axis: torch.Tensor,
    cfg: Args) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute scalar robustness for the *training* spec."""
    margins, dbg_pred = _build_predicate_margins(u_xt=u_xt, x_axis=x_axis, cfg=cfg)
    rob, dbg_time = _reduce_temporal(margins, cfg=cfg)
    dbg = {"predicate": dbg_pred, "temporal": dbg_time}
    return rob, dbg


def _spec_to_str(
    *,
    stl_outer: STLOuter,
    stl_spec: STLPredSpec,
    spatial: SpatialAgg,
    x_star: float,
    u_max: float,
    u_min: float,
    time_mode: TimeMode,
    t0: float,
    t1: float,
    window: int,
    stride: int,
    tau: float,
    domain: tuple[float, float, float, float],
) -> str:
    x_min, x_max, t_min, t_max = domain

    # Spatial part (how we turn u(x,t) into a 1-D signal over time).
    if spatial == "mean":
        s = "mean_x u(x,t)"
    elif spatial == "amax":
        s = "max_x u(x,t)"
    elif spatial == "softmax":
        s = f"softmax_x^tau u(x,t)  (tau={tau:g})"
    elif spatial == "amin":
        s = "min_x u(x,t)"
    elif spatial == "softmin":
        s = f"softmin_x^tau u(x,t)  (tau={tau:g})"
    elif spatial == "point":
        s = f"u(x*={x_star:g},t)"
    else:  # pragma: no cover
        s = f"<unknown spatial={spatial}>"

    # Predicate part.
    if stl_spec == "upper":
        pred = f"({s} <= {u_max:g})"
    elif stl_spec == "lower":
        pred = f"({s} >= {u_min:g})"
    else:
        # Range: by construction we use paired reductions for upper/lower when spatial is max/min-like.
        if spatial in {"softmax", "amax", "softmin", "amin"}:
            s_u = "max_x u(x,t)" if spatial in {"amax", "amin"} else f"softmax_x^tau u(x,t) (tau={tau:g})"
            s_l = "min_x u(x,t)" if spatial in {"amax", "amin"} else f"softmin_x^tau u(x,t) (tau={tau:g})"
            pred = f"({s_l} >= {u_min:g}) AND ({s_u} <= {u_max:g})"
        else:
            pred = f"({u_min:g} <= {s} <= {u_max:g})"

    # Temporal part.
    if time_mode == "all":
        interval = f"[{t_min:g},{t_max:g}]"
    elif time_mode == "interval":
        interval = f"[{t0:g},{t1:g}] (clamped to [{t_min:g},{t_max:g}])"
    else:
        interval = f"[windows: len={window}, stride={stride}] over t∈[{t_min:g},{t_max:g}]"

    op = "G" if stl_outer == "always" else "F"
    return (
        f"{op}_{interval} {pred}\n"
        f"  Robustness sign: >=0 satisfies, <0 violates.\n"
        f"  Spatial sampling: x-grid sampled in [x_min={x_min:g}, x_max={x_max:g}]."
    )


def _import_matplotlib():
    # Headless-safe plotting.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433

    return plt


def _plot_training_curves(*, hist: dict[str, list[float]], out_dir: Path, dpi: int, fmts: tuple[str, ...]) -> None:
    if not hist["epoch"]:
        return

    plt = _import_matplotlib()

    # Losses
    fig = plt.figure()
    plt.plot(hist["epoch"], hist["loss"], label="total")
    plt.plot(hist["epoch"], hist["loss_pde"], label="pde")
    plt.plot(hist["epoch"], hist["loss_bcic"], label="bc+ic")
    if any(math.isfinite(x) and x != 0.0 for x in hist["loss_stl"]):
        plt.plot(hist["epoch"], hist["loss_stl"], label="stl")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    for ext in fmts:
        fig.savefig(out_dir / f"loss_curves.{ext}", dpi=dpi)
    plt.close(fig)

    # Robustness
    if any(math.isfinite(x) for x in hist["robustness"]):
        fig = plt.figure()
        plt.plot(hist["epoch"], hist["robustness"], label="robustness")
        plt.axhline(0.0, linestyle="--", linewidth=1.0)
        plt.xlabel("epoch")
        plt.ylabel("robustness (train spec)")
        plt.legend()
        plt.tight_layout()
        for ext in fmts:
            fig.savefig(out_dir / f"robustness_curve.{ext}", dpi=dpi)
        plt.close(fig)

    # Epoch time
    if any(x > 0 for x in hist["epoch_time_s"]):
        fig = plt.figure()
        plt.plot(hist["epoch"], hist["epoch_time_s"], label="epoch wall time (s)")
        plt.xlabel("epoch")
        plt.ylabel("seconds")
        plt.legend()
        plt.tight_layout()
        for ext in fmts:
            fig.savefig(out_dir / f"epoch_time.{ext}", dpi=dpi)
        plt.close(fig)


def _plot_field_heatmap(
    *, U: torch.Tensor, X: torch.Tensor, T: torch.Tensor,
    out_dir: Path, title: str, fname_stem: str,
    dpi: int, fmts: tuple[str, ...]
) -> None:
    plt = _import_matplotlib()

    u = U.detach().cpu().float().numpy()
    x = X.detach().cpu().float().numpy()
    t = T.detach().cpu().float().numpy()

    if x.ndim == 2 and t.ndim == 2:
        x_min = float(x.min())
        x_max = float(x.max())
        t_min = float(t.min())
        t_max = float(t.max())
    else:
        x_min = float(x[0])
        x_max = float(x[-1])
        t_min = float(t[0])
        t_max = float(t[-1])

    fig = plt.figure()
    plt.imshow(u, origin="lower", aspect="auto", extent=[t_min, t_max, x_min, x_max])
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title(title)
    plt.tight_layout()
    for ext in fmts:
        fig.savefig(out_dir / f"{fname_stem}.{ext}", dpi=dpi)
    plt.close(fig)


def _plot_time_series(
    *,
    t_axis: torch.Tensor,
    series: dict[str, torch.Tensor],
    thresholds: dict[str, float] | None,
    interval: tuple[float, float] | None,
    out_dir: Path,
    title: str,
    fname_stem: str,
    dpi: int,
    fmts: tuple[str, ...],
) -> None:
    plt = _import_matplotlib()

    tt = t_axis.detach().cpu().float().numpy()

    fig = plt.figure()
    for name, s in series.items():
        plt.plot(tt, s.detach().cpu().float().numpy(), label=name)

    if thresholds:
        for name, val in thresholds.items():
            plt.axhline(float(val), linestyle="--", linewidth=1.0, label=name)

    if interval is not None:
        t0, t1 = interval
        plt.axvspan(float(t0), float(t1), alpha=0.15, label="interval")

    plt.xlabel("t")
    plt.ylabel("value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    for ext in fmts:
        fig.savefig(out_dir / f"{fname_stem}.{ext}", dpi=dpi)
    plt.close(fig)


def _relative_l2(u: torch.Tensor, v: torch.Tensor) -> float:
    num = torch.linalg.norm((u - v).reshape(-1), ord=2)
    den = torch.linalg.norm(v.reshape(-1), ord=2).clamp_min(1e-12)
    return float((num / den).detach().cpu().item())


def _monitor_scalar_trace_hard(trace: torch.Tensor, *, pred: STLPredSpec, u_max: float, u_min: float, outer: STLOuter,
    time_slice: slice | None) -> float:
    """
    Hard min/max monitoring on a 1-D trace.

    Returns scalar robustness (>=0 satisfied).
    """
    s = trace
    if time_slice is not None:
        s = s[time_slice]

    if pred == "upper":
        margins = u_max - s
    elif pred == "lower":
        margins = s - u_min
    else:
        margins = torch.minimum(u_max - s, s - u_min)

    if margins.numel() == 0:
        return 0.0

    if outer == "always":
        return float(margins.min().detach().cpu().item())
    return float(margins.max().detach().cpu().item())


def main(argv: Sequence[str] | None = None) -> int:
    cfg = parse_args(argv)
    _validate_cfg(cfg)

    # Ensure timely logs even when stdout is not a TTY (e.g., in batch runs).
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    if cfg.threads > 0:
        torch.set_num_threads(int(cfg.threads))
        torch.set_num_interop_threads(max(1, int(cfg.threads // 2)))

    seed_everything(int(cfg.seed))
    random.seed(int(cfg.seed))

    device = _device_from_cfg(cfg)
    dtype = _as_torch_dtype(cfg.dtype)
    torch.set_default_dtype(dtype)

    if cfg.amp and (device.type != "cuda" or dtype != torch.float32):
        print("WARNING: AMP requested but only supported/recommended for CUDA float32; disabling AMP.", file=sys.stderr)
        use_amp = False
    else:
        use_amp = bool(cfg.amp) and device.type == "cuda"

    run_dir = _make_run_dir(cfg)
    figs_dir = run_dir / "figs"
    if cfg.save_plots:
        figs_dir.mkdir(parents=True, exist_ok=True)

    _write_json(run_dir / "config.json", asdict(cfg))
    _write_json(run_dir / "env.json", _env_summary(device=device, cfg=cfg))

    spec_str = _spec_to_str(
        stl_outer=cfg.stl_outer,
        stl_spec=cfg.stl_spec,
        spatial=cfg.stl_spatial,
        x_star=cfg.stl_x_star,
        u_max=cfg.stl_u_max,
        u_min=cfg.stl_u_min,
        time_mode=cfg.stl_time_mode,
        t0=cfg.stl_t0,
        t1=cfg.stl_t1,
        window=cfg.stl_window,
        stride=cfg.stl_stride,
        tau=cfg.stl_temp,
        domain=(cfg.x_min, cfg.x_max, cfg.t_min, cfg.t_max),
    )
    spatial_desc = str(cfg.stl_spatial)
    if cfg.stl_spec == "range":
        spatial_desc = f"{cfg.stl_spatial} (upper), {_spatial_pair(cfg.stl_spatial)} (lower)"

    spec_str += (
        f"\n  Monitor grid (training): Nx={cfg.stl_nx}, Nt={cfg.stl_nt}. "
        f"Spatial aggregator: {spatial_desc}. Temp tau={cfg.stl_temp:g}."
    )
    (run_dir / "spec.txt").write_text(spec_str + "\n", encoding="utf-8")

    print("=" * 88)
    print("Diffusion1D PINN + STL")
    print(f"Run dir: {run_dir}")
    print(f"Device:  {device}  | dtype: {dtype}")
    print(f"Model:   MLP hidden={cfg.hidden} act={cfg.activation} out_act={cfg.out_act}")
    if cfg.stl_use:
        print("Training spec:")
        print(spec_str)
        print(
            f"STL penalty: kind={cfg.stl_penalty}, beta={cfg.stl_beta:g},"
            f" margin={cfg.stl_margin:g}, lambda={cfg.stl_weight:g}"
        )
    else:
        print("Training spec: (STL disabled)")
    print("=" * 88)

    base = MLP(
        in_dim=2,
        out_dim=1,
        hidden=list(cfg.hidden),
        activation=cfg.activation,
        out_activation=cfg.out_act,
        dtype=dtype,
        device=device,
    )
    if cfg.dirichlet_mask:
        mask = make_dirichlet_mask_1d(x_left=cfg.x_min, x_right=cfg.x_max)
        model: torch.nn.Module = MaskedModel(base, mask)
    else:
        model = base

    model = model.to(device=device, dtype=dtype)
    model = _maybe_compile(model, enabled=bool(cfg.compile))

    n_params = _count_parameters(model)
    print(f"Trainable parameters: {n_params:,}")

    if cfg.opt == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.sched == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=cfg.lr, total_steps=cfg.epochs,
            pct_start=0.1, anneal_strategy="cos",
        )
    elif cfg.sched == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    else:
        sched = None

    # AMP scaler (CUDA only). Prefer the newer torch.amp API when available.
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:  # pragma: no cover
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 1
    if cfg.resume is not None:
        ckpt = torch.load(cfg.resume, map_location=device)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        if "sched" in ckpt and sched is not None:
            try:
                sched.load_state_dict(ckpt["sched"])
            except Exception:
                pass
        if "scaler" in ckpt and use_amp:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception:
                pass
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
        print(f"Resumed from {cfg.resume} (starting at epoch {start_epoch})")

    stl_penalty = STLPenalty(
        weight=float(cfg.stl_weight), margin=float(cfg.stl_margin),
        kind=cfg.stl_penalty, beta=float(cfg.stl_beta), reduction="mean",
    )

    Xs_c, Ts_c, XTs_c = grid1d(
        cfg.stl_nx,
        cfg.stl_nt,
        x_min=cfg.x_min,
        x_max=cfg.x_max,
        t_min=cfg.t_min,
        t_max=cfg.t_max,
        device=device,
        dtype=dtype,
        return_cartesian=False,
    )
    x_axis_c = Xs_c[:, 0].detach()

    csv_path = run_dir / f"diffusion1d_{cfg.tag}.csv"
    logger = CSVLogger(
        csv_path,
        header=[
            "epoch",
            "lr",
            "loss",
            "loss_pde",
            "loss_bcic",
            "loss_stl",
            "robustness",
            "epoch_time_s",
        ],
    )

    hist: dict[str, list[float]] = {
        "epoch": [],
        "lr": [],
        "loss": [],
        "loss_pde": [],
        "loss_bcic": [],
        "loss_stl": [],
        "robustness": [],
        "epoch_time_s": [],
    }

    train_start = time.perf_counter()

    for epoch in range(start_epoch, cfg.epochs + 1):
        t0_epoch = time.perf_counter()

        model.train()
        opt.zero_grad(set_to_none=True)

        coords = sample_interior_1d(
            cfg.batch,
            x_min=cfg.x_min,
            x_max=cfg.x_max,
            t_min=cfg.t_min,
            t_max=cfg.t_max,
            device=device,
            dtype=dtype,
            method=cfg.sample_method,  # type: ignore[arg-type]
            seed=int(cfg.seed) + epoch,
        )
        coords = coords.requires_grad_(True)

        autocast_ctx = torch.cuda.amp.autocast(enabled=True) if use_amp else nullcontext()

        with autocast_ctx:
            loss_pde = residual_loss(model, coords, alpha=cfg.alpha, reduction="mean")
            loss_bcic = boundary_loss(
                model,
                x_left=cfg.x_min,
                x_right=cfg.x_max,
                t_min=cfg.t_min,
                t_max=cfg.t_max,
                device=device,
                n_boundary=cfg.n_boundary,
                n_initial=cfg.n_initial,
                dtype=dtype,
                method=cfg.sample_method,  # type: ignore[arg-type]
                seed=int(cfg.seed) + 10_000 + epoch,
                w_boundary=cfg.w_boundary,
                w_initial=cfg.w_initial,
            )

            loss_stl = torch.zeros((), device=device, dtype=dtype)
            rob_val = float("nan")

            if cfg.stl_use and epoch >= cfg.stl_warmup and (epoch % cfg.stl_every == 0):
                Uc = model(XTs_c).reshape(cfg.stl_nx, cfg.stl_nt)
                rob, _dbg = compute_training_robustness(Uc, x_axis=x_axis_c, cfg=cfg)
                loss_stl = stl_penalty(rob)
                rob_val = float(rob.detach().cpu().item())

            loss = loss_pde + loss_bcic + loss_stl

        if use_amp:
            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.grad_clip))
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.grad_clip))
            opt.step()

        if sched is not None:
            try:
                sched.step()
            except Exception:
                pass

        epoch_time = time.perf_counter() - t0_epoch
        lr_now = float(opt.param_groups[0]["lr"])

        logger.append(
            {
                "epoch": epoch,
                "lr": lr_now,
                "loss": float(loss.detach().cpu().item()),
                "loss_pde": float(loss_pde.detach().cpu().item()),
                "loss_bcic": float(loss_bcic.detach().cpu().item()),
                "loss_stl": float(loss_stl.detach().cpu().item()),
                "robustness": rob_val,
                "epoch_time_s": float(epoch_time),
            }
        )

        hist["epoch"].append(float(epoch))
        hist["lr"].append(lr_now)
        hist["loss"].append(float(loss.detach().cpu().item()))
        hist["loss_pde"].append(float(loss_pde.detach().cpu().item()))
        hist["loss_bcic"].append(float(loss_bcic.detach().cpu().item()))
        hist["loss_stl"].append(float(loss_stl.detach().cpu().item()))
        hist["robustness"].append(float(rob_val))
        hist["epoch_time_s"].append(float(epoch_time))

        if epoch == 1 or epoch == cfg.epochs or (cfg.print_every and epoch % cfg.print_every == 0):
            msg = (
                f"[{epoch:4d}/{cfg.epochs}] "
                f"loss={hist['loss'][-1]:.3e} "
                f"pde={hist['loss_pde'][-1]:.3e} "
                f"bcic={hist['loss_bcic'][-1]:.3e} "
                f"stl={hist['loss_stl'][-1]:.3e} "
                f"rob={hist['robustness'][-1]: .3e} "
                f"lr={lr_now:.2e} "
                f"time={epoch_time:.2f}s"
            )
            print(msg)

    train_wall = time.perf_counter() - train_start
    print(f"Training wall time: {train_wall:.2f}s")

    ckpt_path = run_dir / f"diffusion1d_{cfg.tag}.pt"
    if cfg.save_ckpt:
        torch.save(
            {
                "epoch": cfg.epochs,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict() if sched is not None else None,
                "scaler": scaler.state_dict() if use_amp else None,
                "config": asdict(cfg),
            },
            ckpt_path,
        )

    field_path = run_dir / f"diffusion1d_{cfg.tag}_field.pt"

    with torch.inference_mode():
        X, T, XT = grid1d(
            cfg.nx,
            cfg.nt,
            x_min=cfg.x_min,
            x_max=cfg.x_max,
            t_min=cfg.t_min,
            t_max=cfg.t_max,
            device=device,
            dtype=dtype,
            return_cartesian=False,
        )
        t_eval0 = time.perf_counter()
        U = model(XT).reshape(cfg.nx, cfg.nt)
        infer_wall = time.perf_counter() - t_eval0

        # Analytic solution assumes t=0 at the initial condition; shift by t_min.
        U_true = sine_solution(
            X,
            T - float(cfg.t_min),
            alpha=cfg.alpha,
            x_left=cfg.x_min,
            x_right=cfg.x_max,
            amplitude=1.0,
        )
        err = U - U_true

        metrics = {
            "rel_l2": _relative_l2(U, U_true),
            "max_abs": float(err.abs().max().detach().cpu().item()),
            "mean_abs": float(err.abs().mean().detach().cpu().item()),
            "infer_wall_s": float(infer_wall),
            "train_wall_s": float(train_wall),
            "n_params": int(n_params),
        }

    if cfg.save_field:
        torch.save(
            {
                "u": U.detach().cpu(),
                "X": X.detach().cpu(),
                "T": T.detach().cpu(),
                "alpha": float(cfg.alpha),
                "u_max": float(cfg.stl_u_max),
                "u_max_eval": float(cfg.u_max_eval),
                "config": asdict(cfg),
                "metrics": metrics,
            },
            field_path,
        )

    monitoring: dict[str, Any] = {"metrics": metrics, "specs": []}

    with torch.inference_mode():
        x_axis = X[:, 0].detach()
        t_axis = T[0, :].detach()
        s_max = U.amax(dim=0)
        s_min = U.amin(dim=0)

        idx_star = _nearest_x_index(x_axis, cfg.stl_x_star)
        s_star = U[idx_star, :]

        sl_cool = _time_slice(nt=cfg.nt, t_min=cfg.t_min, t_max=cfg.t_max, t0=cfg.cool_t0, t1=cfg.cool_t1)

        def _add_spec(
            name: str,
            *,
            trace: torch.Tensor,
            pred: STLPredSpec,
            u_max: float,
            u_min: float,
            outer: STLOuter,
            time_slice: slice | None,
            desc: str,
            meta: dict[str, Any] | None = None,
        ) -> None:
            rob = _monitor_scalar_trace_hard(
                trace,
                pred=pred,
                u_max=float(u_max),
                u_min=float(u_min),
                outer=outer,
                time_slice=time_slice,
            )

            slice_info: dict[str, int] | None = None
            if time_slice is not None:
                start = 0 if time_slice.start is None else int(time_slice.start)
                stop = int((time_slice.stop or int(trace.numel())) - 1)
                slice_info = {"start": start, "stop": stop}

            entry: dict[str, Any] = {
                "name": name,
                "desc": desc,
                "outer": outer,
                "pred": pred,
                "u_max": float(u_max),
                "u_min": float(u_min),
                "time_slice": slice_info,
                "robustness_hard": float(rob),
                "satisfied": bool(rob >= 0.0),
            }
            if meta:
                entry.update(meta)
            monitoring["specs"].append(entry)

        _add_spec(
            "safety_max_alltime",
            trace=s_max,
            pred="upper",
            u_max=float(cfg.u_max_eval),
            u_min=0.0,
            outer="always",
            time_slice=None,
            desc=f"G_[{cfg.t_min:g},{cfg.t_max:g}] (max_x u(x,t) <= {cfg.u_max_eval:g})",
            meta={"spatial": "amax"},
        )

        if cfg.extra_monitoring:
            _add_spec(
                "cool_max_loose",
                trace=s_max,
                pred="upper",
                u_max=float(cfg.cool_u_max_loose),
                u_min=0.0,
                outer="eventually",
                time_slice=sl_cool,
                desc=f"F_[{cfg.cool_t0:g},{cfg.cool_t1:g}] (max_x u(x,t) <= {cfg.cool_u_max_loose:g})",
                meta={"spatial": "amax"},
            )
            _add_spec(
                "cool_max_tight",
                trace=s_max,
                pred="upper",
                u_max=float(cfg.cool_u_max_tight),
                u_min=0.0,
                outer="eventually",
                time_slice=sl_cool,
                desc=f"F_[{cfg.cool_t0:g},{cfg.cool_t1:g}] (max_x u(x,t) <= {cfg.cool_u_max_tight:g})",
                meta={"spatial": "amax"},
            )
            _add_spec(
                "cool_point_loose",
                trace=s_star,
                pred="upper",
                u_max=float(cfg.cool_u_max_loose),
                u_min=0.0,
                outer="eventually",
                time_slice=sl_cool,
                desc=(
                    f"F_[{cfg.cool_t0:g},{cfg.cool_t1:g}]"
                    f" (u(x*={float(x_axis[idx_star]):g},t)"
                    f" <= {cfg.cool_u_max_loose:g})"
                ),
                meta={
                    "spatial": "point",
                    "x_idx": int(idx_star),
                    "x_value": float(x_axis[idx_star].cpu().item()),
                },
            )
            _add_spec(
                "cool_point_tight",
                trace=s_star,
                pred="upper",
                u_max=float(cfg.cool_u_max_tight),
                u_min=0.0,
                outer="eventually",
                time_slice=sl_cool,
                desc=(
                    f"F_[{cfg.cool_t0:g},{cfg.cool_t1:g}]"
                    f" (u(x*={float(x_axis[idx_star]):g},t)"
                    f" <= {cfg.cool_u_max_tight:g})"
                ),
                meta={
                    "spatial": "point",
                    "x_idx": int(idx_star),
                    "x_value": float(x_axis[idx_star].cpu().item()),
                },
            )

    _write_json(run_dir / "monitoring.json", monitoring)

    if cfg.save_plots:
        _plot_training_curves(hist=hist, out_dir=figs_dir, dpi=int(cfg.plot_dpi), fmts=cfg.plot_formats)

        _plot_field_heatmap(
            U=U, X=X, T=T, out_dir=figs_dir, title="Predicted field u(x,t)",
            fname_stem="field_pred", dpi=int(cfg.plot_dpi), fmts=cfg.plot_formats,
        )
        _plot_field_heatmap(
            U=U_true, X=X, T=T, out_dir=figs_dir, title="Analytic solution u*(x,t)",
            fname_stem="field_true", dpi=int(cfg.plot_dpi), fmts=cfg.plot_formats,
        )
        _plot_field_heatmap(
            U=err, X=X, T=T, out_dir=figs_dir, title="Error u(x,t) - u*(x,t)",
            fname_stem="field_error", dpi=int(cfg.plot_dpi), fmts=cfg.plot_formats,
        )

        _plot_time_series(
            t_axis=t_axis,
            series={
                "max_x u(x,t)": s_max,
                "min_x u(x,t)": s_min,
                f"u(x*≈{float(x_axis[idx_star]):g},t)": s_star,
            },
            thresholds={
                "U_max_eval": float(cfg.u_max_eval),
                "cool_loose": float(cfg.cool_u_max_loose),
                "cool_tight": float(cfg.cool_u_max_tight),
            }
            if cfg.extra_monitoring
            else {"U_max_eval": float(cfg.u_max_eval)},
            interval=(
                max(min(float(cfg.cool_t0), float(cfg.cool_t1)), float(cfg.t_min)),
                min(max(float(cfg.cool_t0), float(cfg.cool_t1)), float(cfg.t_max)),
            )
            if cfg.extra_monitoring
            else None,
            out_dir=figs_dir,
            title="Monitored time-series (spatial aggregation vs point)",
            fname_stem="time_series",
            dpi=int(cfg.plot_dpi),
            fmts=cfg.plot_formats,
        )

    print("=" * 88)
    print("DONE")
    print(f"Logs:        {csv_path}")
    if cfg.save_ckpt:
        print(f"Checkpoint:  {ckpt_path}")
    if cfg.save_field:
        print(f"Field:       {field_path}")
    print(f"Monitoring:  {run_dir / 'monitoring.json'}")
    if cfg.save_plots:
        print(f"Figures:     {figs_dir}")
    print(
        f"Metrics:     rel_l2={metrics['rel_l2']:.3e},"
        f" max_abs={metrics['max_abs']:.3e},"
        f" infer_wall={metrics['infer_wall_s']:.3f}s"
    )
    print("=" * 88)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
