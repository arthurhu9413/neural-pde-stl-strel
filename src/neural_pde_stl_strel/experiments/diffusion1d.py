# ruff: noqa: I001
from __future__ import annotations

"""neural_pde_stl_strel.experiments.diffusion1d

1-D diffusion/heat PINN with STL monitoring and optional STL-regularized training.

This experiment trains a compact neural field :math:`u_\theta(x,t)` to satisfy the PDE

    ∂u/∂t = α ∂²u/∂x²,   (x,t) ∈ [x_min, x_max] × [t_min, t_max],

using standard PINN losses:

* PDE residual loss over interior collocation points
* boundary/initial condition losses (Dirichlet + sine initial condition by default)

STL safety property
We monitor (and optionally regularize) a global-in-time upper bound:

    ϕ_safe := G_[t_min,t_max] ( Agg_x u(x,t) ≤ U_max )

where ``Agg_x`` approximates a universal quantifier over space by reducing the
field into a scalar time series s(t):

    s(t) = Agg_x u(x,t)

Supported spatial aggregators:

* ``"amax"``   : hard max over x (exact on the discrete grid)
* ``"mean"``   : mean over x
* ``"softmax"``: log-sum-exp smooth max over x with temperature ``τ``

Training objective
The objective matches the report/slide deck data-flow:

    L_total = L_PDE + w_bcic·L_BC/IC + λ·Penalty(ρ(ϕ_safe, uθ))

where ρ is a (soft) robustness value (positive => satisfied). The STL term is
computed on a *monitor grid* (N_x^mon × N_t^mon) decoupled from the collocation
batch.

Artifacts
---------
Stable outputs are written under ``results_dir``:

* ``diffusion1d_{tag}.csv``          per-epoch log (losses, robustness, timing)
* ``diffusion1d_{tag}.pt``           checkpoint (state_dict + config + final metrics)
* ``diffusion1d_{tag}_field.pt``     final field snapshot dict(u, X, T, ...)
* ``diffusion1d_{tag}_metrics.json`` small summary (errors, robustness, runtime)

When ``io.run_dir`` is provided (e.g., via ``scripts/run_experiment.py``), the
stable artifacts are also copied into that run directory for provenance.
"""

from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import time
from typing import Any, Mapping

import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim

from ..models.mlp import MLP
from ..physics.diffusion1d import boundary_loss, residual_loss, sine_solution
from ..training.grids import grid1d, sample_interior_1d
from ..utils.logger import CSVLogger
from ..utils.seed import seed_everything

# Optional differentiable STL (soft semantics)
try:  # keep the experiment runnable even if optional monitoring deps are absent
    from ..monitoring.stl_soft import STLPenalty, always, eventually, pred_leq

    _HAS_STL = True
except Exception:  # pragma: no cover
    _HAS_STL = False

__all__ = ["Diffusion1DConfig", "run_diffusion1d"]


# Configuration


@dataclass
class Diffusion1DConfig:

    hidden: tuple[int, ...] = (64, 64, 64)
    activation: str = "tanh"
    out_activation: str | None = None

    n_x: int = 128
    n_t: int = 64
    x_min: float = 0.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    lr: float = 1e-3
    epochs: int = 200
    batch: int = 4096
    weight_decay: float = 0.0
    bcic_weight: float = 1.0

    alpha: float = 0.1

    n_boundary: int = 256
    n_initial: int = 512
    sample_method: str = "sobol"  # "sobol" | "uniform"

    device: str | None = None  # "cuda" | "mps" | "cpu" | None/"auto"
    dtype: str = "float32"  # "float32" | "float64" | "bf16" | ...
    amp: bool = False  # mixed precision (off by default due to higher-order grads)
    compile: bool = False  # torch.compile if available
    print_every: int = 25

    stl_use: bool = False
    stl_weight: float = 0.0  # λ
    stl_u_max_train: float = 1.0
    stl_u_max_eval: float | None = None
    stl_temp: float = 0.1
    stl_reduce_x: str = "mean"  # mean | softmax | amax
    stl_every: int = 1
    stl_monitor_nx: int = 128
    stl_monitor_nt: int = 64
    stl_penalty: str = "softplus"  # softplus | hinge
    stl_beta: float = 10.0
    stl_margin: float = 0.0
    stl_eval_specs: dict[str, Any] | None = None  # optional: named specs to evaluate post-hoc

    results_dir: str = "results"
    run_dir: str | None = None
    tag: str = "run"
    save_ckpt: bool = True
    save_metrics: bool = True
    copy_to_run_dir: bool = True

    seed: int = 0


# Helpers


def _as_dict(x: Any) -> dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _resolve_device(choice: str | None) -> torch.device:
    """Resolve a device string.

    Accepts None/"auto" for automatic selection (cuda > mps > cpu).
    """

    if choice is None:
        normalized = "auto"
    else:
        normalized = str(choice).strip().lower()

    if normalized in {"", "none", "null", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return torch.device("mps")
        return torch.device("cpu")

    # Explicit requests: validate availability.
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available (torch.cuda.is_available() is False).")
    if normalized == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()  # type: ignore[attr-defined]
    ):
        raise RuntimeError("MPS was requested but is not available (torch.backends.mps.is_available() is False).")

    return torch.device(choice)


_DTYPE_ALIASES: dict[str, torch.dtype] = {
    "float": torch.float32,
    "fp32": torch.float32,
    "float32": torch.float32,
    "single": torch.float32,
    "float64": torch.float64,
    "double": torch.float64,
    "fp64": torch.float64,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def _resolve_dtype(spec: Any) -> torch.dtype:
    if isinstance(spec, torch.dtype):
        return spec
    if spec is None:
        return torch.float32
    key = str(spec).strip().lower().replace("torch.", "")
    if key in _DTYPE_ALIASES:
        return _DTYPE_ALIASES[key]
    raise ValueError(f"Unsupported dtype spec {spec!r}. Supported: {sorted(_DTYPE_ALIASES)}")


def _maybe_compile(module: nn.Module, do_compile: bool) -> nn.Module:  # pragma: no cover
    if not do_compile:
        return module
    compile_fn = getattr(torch, "compile", None)
    if not callable(compile_fn):
        return module
    try:
        return compile_fn(module, mode="reduce-overhead", fullgraph=False)  # type: ignore[misc]
    except Exception:
        # torch.compile is optional; silently fall back.
        return module


def _make_grad_scaler(*, device_type: str, enabled: bool):
    """Create a GradScaler without triggering deprecation warnings.

    Newer PyTorch versions prefer `torch.amp.GradScaler`; older versions used
    `torch.cuda.amp.GradScaler`. We support both.
    """

    # If scaling is disabled, we return a disabled scaler without caring about
    # device support (this avoids passing an unsupported device string).
    if not enabled:
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            return torch.amp.GradScaler(device="cuda", enabled=False)
        # Fallback for older torch
        return torch.cuda.amp.GradScaler(enabled=False)

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device=device_type, enabled=True)
    # Fallback for older torch
    return torch.cuda.amp.GradScaler(enabled=True)


def _autocast(*, device_type: str, enabled: bool):
    """Autocast context manager without triggering deprecation warnings."""

    if not enabled:
        return nullcontext()

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device_type, enabled=True)
    # Fallback for older torch
    return torch.cuda.amp.autocast(enabled=True)


def _smooth_max(x: Tensor, *, temp: float, dim: int) -> Tensor:
    """Smooth max via log-sum-exp (temperature τ).

    For τ->0, approaches hard max.
    """

    if temp <= 0:
        return x.amax(dim=dim)
    return float(temp) * torch.logsumexp(x / float(temp), dim=dim)


def _smooth_min(x: Tensor, *, temp: float, dim: int) -> Tensor:
    """Smooth min via log-sum-exp (temperature τ)."""

    if temp <= 0:
        return x.amin(dim=dim)
    return -float(temp) * torch.logsumexp(-x / float(temp), dim=dim)


def _reduce_x(u_xt: Tensor, *, mode: str, temp: float) -> Tensor:
    """Reduce a field u(x,t) over x -> signal s(t)."""

    mode_n = str(mode).strip().lower()
    if mode_n in {"mean", "avg", "average"}:
        return u_xt.mean(dim=0)
    if mode_n in {"amax", "max"}:
        return u_xt.amax(dim=0)
    if mode_n in {"softmax", "lse", "logsumexp"}:
        return _smooth_max(u_xt, temp=temp, dim=0)
    raise ValueError(f"Unknown spatial reducer {mode!r}. Expected mean|softmax|amax.")


_SIMPLE_FORMULA_RE = re.compile(
    r"^\s*(?P<op>always|eventually)\s*\[\s*(?P<t0>[^,\]]+)\s*,\s*(?P<t1>[^\]]+)\s*\]"
    r"\s*\(\s*(?P<var>[A-Za-z_]\w*)\s*(?P<cmp><=|>=|<|>)\s*(?P<bound>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)\s*$",
    flags=re.IGNORECASE,
)


def _parse_endpoint(tok: str) -> float | None:
    s = tok.strip().lower()
    if s in {"inf", "infty", "infinity", "∞"}:
        return None
    return float(s)


def _parse_simple_formula(formula: str) -> dict[str, Any] | None:
    """Parse a tiny STL subset used in configs for post-hoc evaluation.

    Supported forms (case-insensitive, whitespace tolerant):

        always[a,b](s <= c)
        eventually[a,b](s <= c)
        ... with comparators <=, <, >=, > and b in {float, inf}.

    Returns a dict with keys: op, t0, t1, cmp, bound.
    """

    m = _SIMPLE_FORMULA_RE.match(formula)
    if m is None:
        return None

    op = m.group("op").strip().lower()
    t0 = float(_parse_endpoint(m.group("t0")))
    t1 = _parse_endpoint(m.group("t1"))
    cmp_ = m.group("cmp")
    bound = float(m.group("bound"))

    return {"op": op, "t0": t0, "t1": t1, "cmp": cmp_, "bound": bound}


def _eval_formula_on_signal(
    *,
    signal_t: Tensor,
    t_vals: Tensor,
    parsed: Mapping[str, Any],
    temp: float,
) -> tuple[float, float]:
    """Evaluate a parsed formula on a scalar time-series.

    Returns (soft_robustness, hard_robustness).
    """

    # Select time window.
    t0 = float(parsed["t0"])
    t1 = parsed.get("t1", None)
    if t1 is not None:
        t1 = float(t1)

    mask = t_vals >= t0
    if t1 is not None:
        mask = mask & (t_vals <= t1)

    if not bool(mask.any()):
        # Degenerate window (e.g., t0 outside grid). Fall back to full horizon.
        mask = torch.ones_like(t_vals, dtype=torch.bool)

    s = signal_t[mask]

    # Predicate margin: positive => satisfied.
    bound = float(parsed["bound"])
    cmp_ = str(parsed["cmp"])
    if cmp_ in {"<=", "<"}:
        margins = bound - s
    elif cmp_ in {">=", ">"}:
        margins = s - bound
    else:  # pragma: no cover (regex restricts)
        raise ValueError(f"Unsupported comparator {cmp_!r}")

    op = str(parsed["op"]).lower()
    if op == "always":
        hard = float(margins.min().item())
        if _HAS_STL:
            soft = float(always(margins, temp=float(temp), time_dim=-1).item())
        else:
            soft = hard
        return soft, hard

    if op == "eventually":
        hard = float(margins.max().item())
        if _HAS_STL:
            soft = float(eventually(margins, temp=float(temp), time_dim=-1).item())
        else:
            soft = hard
        return soft, hard

    raise ValueError(f"Unsupported op {op!r}")


def _parse_config(cfg_dict: Mapping[str, Any]) -> Diffusion1DConfig:
    """Parse YAML/CLI config dicts into a concrete config.

    Important: ``scripts/run_experiment.py`` supports CLI overrides like
    ``--set training.epochs=50``. To make that work, we treat ``training`` as an
    alias of ``optim`` and let it override fields from ``optim``.
    """

    cfg = dict(cfg_dict)

    model = _as_dict(cfg.get("model"))
    grid = _as_dict(cfg.get("grid"))
    physics = _as_dict(cfg.get("physics"))
    stl = _as_dict(cfg.get("stl"))
    io = _as_dict(cfg.get("io"))

    # Merge optim + training (training wins).
    optim_cfg = _as_dict(cfg.get("optim"))
    training_cfg = _as_dict(cfg.get("training"))
    train = {**optim_cfg, **training_cfg}

    hidden_raw = model.get("hidden", (64, 64, 64))
    if isinstance(hidden_raw, int):
        hidden = (int(hidden_raw),)
    elif isinstance(hidden_raw, (list, tuple)):
        hidden = tuple(int(h) for h in hidden_raw)
    else:
        hidden = (64, 64, 64)

    activation = str(model.get("activation", model.get("act", "tanh")))
    out_activation = model.get("out_activation", model.get("out_act", None))
    out_activation = None if out_activation is None else str(out_activation)

    n_x = int(grid.get("n_x", cfg.get("n_x", 128)))
    n_t = int(grid.get("n_t", cfg.get("n_t", 64)))
    x_min = float(grid.get("x_min", cfg.get("x_min", 0.0)))
    x_max = float(grid.get("x_max", cfg.get("x_max", 1.0)))
    t_min = float(grid.get("t_min", cfg.get("t_min", 0.0)))
    t_max = float(grid.get("t_max", cfg.get("t_max", 1.0)))

    lr = float(train.get("lr", cfg.get("lr", 1e-3)))
    epochs = int(train.get("epochs", cfg.get("epochs", 200)))
    batch = int(train.get("batch", cfg.get("batch", 4096)))
    weight_decay = float(train.get("weight_decay", cfg.get("weight_decay", 0.0)))
    bcic_weight = float(train.get("bcic_weight", cfg.get("bcic_weight", 1.0)))

    alpha = float(physics.get("alpha", cfg.get("alpha", 0.1)))

    n_boundary = int(train.get("n_boundary", cfg.get("n_boundary", 256)))
    n_initial = int(train.get("n_initial", cfg.get("n_initial", 512)))
    sample_method = str(train.get("sample_method", cfg.get("sample_method", "sobol")))

    device = cfg.get("device", io.get("device", None))
    dtype = cfg.get("dtype", io.get("dtype", "float32"))
    amp = bool(cfg.get("amp", io.get("amp", False)))
    compile_ = bool(cfg.get("compile", io.get("compile", False)))
    print_every = int(cfg.get("print_every", io.get("print_every", 25)))

    stl_use = bool(stl.get("use", cfg.get("stl_use", False)))
    stl_weight = float(stl.get("weight", cfg.get("stl_weight", 0.0)))
    stl_u_max_train = float(
        stl.get(
            "u_max",
            stl.get("u_max_train", cfg.get("stl_u_max", 1.0)),
        )
    )
    u_max_eval_raw = stl.get("u_max_eval", stl.get("eval_u_max", None))
    stl_u_max_eval = None if u_max_eval_raw is None else float(u_max_eval_raw)

    stl_temp = float(stl.get("temp", cfg.get("stl_temp", 0.1)))
    stl_reduce_x = str(
        stl.get(
            "reduce_x",
            stl.get("spatial", stl.get("reduce", cfg.get("stl_spatial", "mean"))),
        )
    )
    stl_every = int(stl.get("every", cfg.get("stl_every", 1)))

    stl_monitor_nx = int(
        stl.get(
            "monitor_nx",
            stl.get(
                "monitor_n_x",
                stl.get("n_x", stl.get("monitor_nx", grid.get("n_x", 128))),
            ),
        )
    )
    stl_monitor_nt = int(
        stl.get(
            "monitor_nt",
            stl.get(
                "monitor_n_t",
                stl.get("n_t", stl.get("monitor_nt", grid.get("n_t", 64))),
            ),
        )
    )

    stl_penalty = str(stl.get("penalty", cfg.get("stl_penalty", "softplus")))
    stl_beta = float(stl.get("beta", cfg.get("stl_beta", 10.0)))
    stl_margin = float(stl.get("margin", stl.get("m0", cfg.get("stl_margin", 0.0))))

    eval_specs_raw = stl.get("eval_specs", None)
    stl_eval_specs = eval_specs_raw if isinstance(eval_specs_raw, dict) else None

    results_dir = str(io.get("results_dir", cfg.get("results_dir", "results")))
    run_dir = io.get("run_dir", cfg.get("run_dir", None))
    run_dir = None if run_dir is None else str(run_dir)

    tag = cfg.get("tag", io.get("tag", "run"))
    tag = str(tag)

    save_ckpt = bool(io.get("save_ckpt", cfg.get("save_ckpt", True)))
    save_metrics = bool(io.get("save_metrics", cfg.get("save_metrics", True)))
    copy_to_run_dir = bool(io.get("copy_to_run_dir", cfg.get("copy_to_run_dir", True)))

    seed = int(cfg.get("seed", 0))

    return Diffusion1DConfig(
        hidden=hidden,
        activation=activation,
        out_activation=out_activation,
        n_x=n_x,
        n_t=n_t,
        x_min=x_min,
        x_max=x_max,
        t_min=t_min,
        t_max=t_max,
        lr=lr,
        epochs=epochs,
        batch=batch,
        weight_decay=weight_decay,
        bcic_weight=bcic_weight,
        alpha=alpha,
        n_boundary=n_boundary,
        n_initial=n_initial,
        sample_method=sample_method,
        device=None if device is None else str(device),
        dtype=str(dtype),
        amp=amp,
        compile=compile_,
        print_every=print_every,
        stl_use=stl_use,
        stl_weight=stl_weight,
        stl_u_max_train=stl_u_max_train,
        stl_u_max_eval=stl_u_max_eval,
        stl_temp=stl_temp,
        stl_reduce_x=stl_reduce_x,
        stl_every=stl_every,
        stl_monitor_nx=stl_monitor_nx,
        stl_monitor_nt=stl_monitor_nt,
        stl_penalty=stl_penalty,
        stl_beta=stl_beta,
        stl_margin=stl_margin,
        stl_eval_specs=stl_eval_specs,
        results_dir=results_dir,
        run_dir=run_dir,
        tag=tag,
        save_ckpt=save_ckpt,
        save_metrics=save_metrics,
        copy_to_run_dir=copy_to_run_dir,
        seed=seed,
    )


def _copy_artifacts_to_run_dir(*, run_dir: str | None, paths: list[Path]) -> None:
    if run_dir is None:
        return
    dst = Path(run_dir)
    dst.mkdir(parents=True, exist_ok=True)

    for p in paths:
        if not p.exists():
            continue
        try:
            shutil.copy2(p, dst / p.name)
        except Exception:
            # Best-effort provenance copy.
            pass


# Main entry point


def run_diffusion1d(cfg_dict: Mapping[str, Any]) -> str:
    """Run the 1-D diffusion experiment.

    Returns:
        Path to the saved field tensor (``diffusion1d_{tag}_field.pt``).
    """

    cfg = _parse_config(cfg_dict)
    seed_everything(cfg.seed)

    device = _resolve_device(cfg.device)
    dtype = _resolve_dtype(cfg.dtype)

    model = MLP(
        in_dim=2,
        out_dim=1,
        hidden=cfg.hidden,
        activation=cfg.activation,
        out_activation=cfg.out_activation,
        dtype=dtype,
        device=device,
    )
    model = _maybe_compile(model, cfg.compile)

    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler: Any = _make_grad_scaler(device_type=device.type, enabled=use_amp)

    # STL: monitor is allowed with weight==0; penalty only affects training if weight>0.
    stl_enabled = bool(cfg.stl_use)
    stl_can_do_soft = bool(stl_enabled and _HAS_STL)
    stl_do_penalty = bool(stl_can_do_soft and cfg.stl_weight > 0.0)

    penalty = None
    if stl_do_penalty:
        penalty = STLPenalty(
            weight=float(cfg.stl_weight),
            kind=str(cfg.stl_penalty),
            beta=float(cfg.stl_beta),
            margin=float(cfg.stl_margin),
        )
    elif stl_enabled and not _HAS_STL:
        print(
            "[diffusion1d] STL requested but soft semantics are unavailable; "
            "training will proceed without differentiable STL."
        )

    # Full grid for final export/snapshots.
    X, T, XT = grid1d(
        n_x=cfg.n_x,
        n_t=cfg.n_t,
        x_min=cfg.x_min,
        x_max=cfg.x_max,
        t_min=cfg.t_min,
        t_max=cfg.t_max,
        device=device,
        dtype=dtype,
    )

    # Coarse grid for STL monitoring.
    Xs, Ts, XTs = grid1d(
        n_x=max(8, int(cfg.stl_monitor_nx)),
        n_t=max(4, int(cfg.stl_monitor_nt)),
        x_min=cfg.x_min,
        x_max=cfg.x_max,
        t_min=cfg.t_min,
        t_max=cfg.t_max,
        device=device,
        dtype=dtype,
    )

    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    base = results_dir / f"diffusion1d_{cfg.tag}"
    log_path = base.with_suffix(".csv")
    ckpt_path = base.with_suffix(".pt")
    field_path = results_dir / f"diffusion1d_{cfg.tag}_field.pt"
    metrics_path = results_dir / f"diffusion1d_{cfg.tag}_metrics.json"

    log = CSVLogger(
        log_path,
        header=[
            "epoch",
            "lr",
            "time_sec",
            "loss",
            "loss_total",
            "loss_pde",
            "loss_bcic",
            "loss_stl",
            "robustness",
        ],
    )

    t_train_start = time.perf_counter()
    model.train()

    for epoch in range(int(cfg.epochs)):
        t_epoch_start = time.perf_counter()

        coords = sample_interior_1d(
            int(cfg.batch),
            x_min=cfg.x_min,
            x_max=cfg.x_max,
            t_min=cfg.t_min,
            t_max=cfg.t_max,
            method=cfg.sample_method,
            device=device,
            dtype=dtype,
            seed=cfg.seed + epoch,
        )
        coords.requires_grad_(True)

        with _autocast(device_type=device.type, enabled=use_amp):
            loss_pde = residual_loss(model, coords, alpha=cfg.alpha, reduction="mean")

            loss_bcic_raw = boundary_loss(
                model,
                x_left=cfg.x_min,
                x_right=cfg.x_max,
                t_min=cfg.t_min,
                t_max=cfg.t_max,
                device=device,
                dtype=dtype,
                method=cfg.sample_method,
                seed=cfg.seed + 17 * epoch,
                n_boundary=cfg.n_boundary,
                n_initial=cfg.n_initial,
            )
            loss_bcic = float(cfg.bcic_weight) * loss_bcic_raw

            # STL monitoring/penalty.
            loss_stl = torch.zeros((), device=device, dtype=dtype)
            rob = torch.tensor(float("nan"), device=device, dtype=dtype)

            if stl_enabled:
                # Compute robustness each epoch for logging; apply gradients only
                # when the penalty is active and on the chosen cadence.
                apply_penalty_now = bool(stl_do_penalty and (epoch % max(1, int(cfg.stl_every)) == 0))

                if stl_can_do_soft and apply_penalty_now:
                    u_mon = model(XTs).reshape(Xs.shape)
                    s_t = _reduce_x(u_mon, mode=cfg.stl_reduce_x, temp=float(cfg.stl_temp))
                    margins = pred_leq(s_t, float(cfg.stl_u_max_train))
                    rob = always(margins, temp=float(cfg.stl_temp), time_dim=-1)
                    loss_stl = penalty(rob) if penalty is not None else torch.zeros((), device=device, dtype=dtype)
                else:
                    with torch.no_grad():
                        u_mon = model(XTs).reshape(Xs.shape)
                        s_t = _reduce_x(u_mon, mode=cfg.stl_reduce_x, temp=float(cfg.stl_temp))
                        # Hard robustness fallback if soft semantics missing.
                        if _HAS_STL:
                            margins = pred_leq(s_t, float(cfg.stl_u_max_train))
                            rob = always(margins, temp=float(cfg.stl_temp), time_dim=-1)
                        else:
                            rob = (float(cfg.stl_u_max_train) - s_t).amin(dim=-1)

            loss = loss_pde + loss_bcic + loss_stl

        opt.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        lr_now = float(opt.param_groups[0]["lr"])
        t_epoch = time.perf_counter() - t_epoch_start

        log.append(
            [
                epoch,
                lr_now,
                t_epoch,
                float(loss.detach().item()),
                float(loss.detach().item()),
                float(loss_pde.detach().item()),
                float(loss_bcic.detach().item()),
                float(loss_stl.detach().item()),
                float(rob.detach().item()) if torch.isfinite(rob).item() else float("nan"),
            ]
        )

        if (epoch % max(1, int(cfg.print_every)) == 0) or (epoch == cfg.epochs - 1):
            loss_val = float(loss.detach().cpu().item())
            loss_pde_val = float(loss_pde.detach().cpu().item())
            loss_bcic_val = float(loss_bcic.detach().cpu().item())
            loss_stl_val = float(loss_stl.detach().cpu().item())
            rob_val = float(rob.detach().cpu().item()) if torch.isfinite(rob).item() else float("nan")
            print(
                f"[diffusion1d] epoch={epoch:04d} lr={lr_now:.2e} time={t_epoch:.2f}s "
                f"loss={loss_val:.4e} pde={loss_pde_val:.4e} "
                f"bcic={loss_bcic_val:.4e} stl={loss_stl_val:.4e} rob={rob_val:.4e}"
            )

    train_wall_time = time.perf_counter() - t_train_start

    model.eval()
    with torch.inference_mode():
        U = model(XT).reshape(X.shape).detach().to("cpu")
        X_cpu = X.detach().to("cpu")
        T_cpu = T.detach().to("cpu")

    # Hard always robustness for the bound (exact on this discrete grid):
    #   ρ_hard = U_max - max_{x,t} u(x,t).
    u_max_pred = float(U.max().item())

    u_max_train = float(cfg.stl_u_max_train)
    u_max_eval = float(cfg.stl_u_max_eval) if cfg.stl_u_max_eval is not None else u_max_train

    rho_hard_train = u_max_train - u_max_pred
    rho_hard_eval = u_max_eval - u_max_pred

    # Relative error vs the default analytic solution (sine IC, zero Dirichlet BCs).
    u_true = sine_solution(X_cpu, T_cpu, alpha=float(cfg.alpha))
    rel_l2 = float(torch.linalg.norm(U - u_true).item() / (torch.linalg.norm(u_true).item() + 1e-12))

    # Soft robustness (full export grid) using the configured reducer, if available.
    rho_soft_train = float("nan")
    rho_soft_eval = float("nan")
    if stl_enabled:
        s_full = _reduce_x(U, mode=cfg.stl_reduce_x, temp=float(cfg.stl_temp))
        if _HAS_STL:
            rho_soft_train = float(always(pred_leq(s_full, u_max_train), temp=float(cfg.stl_temp), time_dim=-1).item())
            rho_soft_eval = float(always(pred_leq(s_full, u_max_eval), temp=float(cfg.stl_temp), time_dim=-1).item())
        else:
            rho_soft_train = float((u_max_train - s_full).amin(dim=-1).item())
            rho_soft_eval = float((u_max_eval - s_full).amin(dim=-1).item())

    metrics: dict[str, Any] = {
        "tag": cfg.tag,
        "epochs": int(cfg.epochs),
        "train_wall_time_sec": float(train_wall_time),
        "avg_epoch_time_sec": float(train_wall_time / max(1, int(cfg.epochs))),
        "device": str(device),
        "dtype": str(dtype),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "alpha": float(cfg.alpha),
        "u_max_pred": u_max_pred,
        "u_max_train": u_max_train,
        "u_max_eval": u_max_eval,
        "rho_hard_train": float(rho_hard_train),
        "rho_hard_eval": float(rho_hard_eval),
        "rho_soft_train": float(rho_soft_train),
        "rho_soft_eval": float(rho_soft_eval),
        "rel_l2": float(rel_l2),
    }

    # Optional: evaluate any named specs in cfg.stl_eval_specs.
    if cfg.stl_eval_specs:
        # Time vector (T is a meshgrid constant in x; take first row).
        t_vals = T_cpu[0, :] if T_cpu.ndim == 2 else T_cpu

        for name, spec in cfg.stl_eval_specs.items():
            if not isinstance(spec, dict):
                continue
            formula = spec.get("rtamt", spec.get("formula", spec.get("spec", None)))
            if formula is None:
                continue

            parsed = _parse_simple_formula(str(formula))
            if parsed is None:
                continue

            agg = spec.get("agg", spec.get("reduce_x", spec.get("spatial", cfg.stl_reduce_x)))
            signal = _reduce_x(U, mode=str(agg), temp=float(cfg.stl_temp))

            soft_rho, hard_rho = _eval_formula_on_signal(
                signal_t=signal,
                t_vals=t_vals,
                parsed=parsed,
                temp=float(cfg.stl_temp),
            )

            metrics[f"spec_{name}_formula"] = str(formula)
            metrics[f"spec_{name}_agg"] = str(agg)
            metrics[f"spec_{name}_rho_soft"] = float(soft_rho)
            metrics[f"spec_{name}_rho_hard"] = float(hard_rho)

    artifact_paths: list[Path] = [log_path]

    if cfg.save_ckpt:
        torch.save(
            {
                "model": model.state_dict(),
                "config": cfg.__dict__,
                "metrics": metrics,
            },
            ckpt_path,
        )
        artifact_paths.append(ckpt_path)

    torch.save(
        {
            "u": U,
            "X": X_cpu,
            "T": T_cpu,
            "alpha": float(cfg.alpha),
            "u_max_train": u_max_train,
            "u_max_eval": u_max_eval,
            "config": cfg.__dict__,
            "metrics": metrics,
        },
        field_path,
    )
    artifact_paths.append(field_path)

    if cfg.save_metrics:
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact_paths.append(metrics_path)

    if cfg.copy_to_run_dir:
        _copy_artifacts_to_run_dir(run_dir=cfg.run_dir, paths=artifact_paths)

    return str(field_path)
