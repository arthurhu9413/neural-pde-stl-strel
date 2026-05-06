"""train_heat2d_strel.py

End-to-end *example script* for a 2-D heat-equation PINN + STREL monitoring.

This script is intentionally self-contained (single file) and is designed to support
"demo-style" exploratory usage:

  1) Train a small physics-informed neural network (PINN) for the 2-D heat equation.
  2) Export a dense rollout u(x,y,t) on a regular grid.
  3) Convert that rollout to a *boolean* signal hot(x,y,t) := (u >= θ).
  4) Monitor a spatio-temporal specification written in STREL using MoonLight.

The default STREL script is: scripts/specs/contain_hotspot.mls
and the default formula is: contain_hotspot

Why the boolean encoding matters (IMPORTANT)
MoonLight's BooleanDomain interprets a numeric input v as:

  True  if v >= 0
  False if v <  0

Therefore, encoding False as 0.0 is WRONG (0.0 is treated as True).
This script encodes:

  True  -> +1.0
  False -> -1.0

This is consistent with MoonLight's BooleanDomain.toDouble(Boolean) which returns
+1.0 for True and -1.0 for False.

High-level data flow (block diagram)

        PDE + IC/BC + PINN architecture + (optional) safety surrogate
                               |
                               v
                        train PINN (PyTorch)
                               |
                               v
                   dense rollout u(x,y,t) on grid
                               |
                      threshold at θ (hot predicate)
                               |
                               v
         graph (grid adjacency + edge weight w) + hot(x,y,t) signal
                               |
                               v
                 MoonLight STREL monitor  -> satisfaction trace + summary

Notes on discretization:
  - Spatial quantification is approximated on the sampled grid (nx × ny nodes).
  - Temporal quantification is evaluated on the sampled time grid (nt points).

Outputs (run directory under --out-root):
  - config.json          : all settings used for the run
  - metrics.csv          : epoch-by-epoch losses (and optional terms)
  - field_xy_t.npy       : exported rollout of u on grid (shape: [nx, ny, nt])
  - meta.json            : environment + runtime + monitoring summaries
  - figures/*.png        : loss curves, field snapshots, max-temp trace, etc.

Example usage (CPU friendly quick run):
  python scripts/train_heat2d_strel.py --quick

Run with STREL monitoring (requires MoonLight + Java):
  python scripts/train_heat2d_strel.py --quick --audit-strel

"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Torch is required for training. We keep import-time errors explicit and friendly.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = e
else:
    _TORCH_IMPORT_ERROR = None

# MoonLight is optional; the script runs without it and will simply skip STREL audit.
try:
    from moonlight import ScriptLoader  # type: ignore
except Exception as e:  # pragma: no cover
    ScriptLoader = None  # type: ignore[assignment]
    _MOONLIGHT_IMPORT_ERROR = e
else:
    _MOONLIGHT_IMPORT_ERROR = None


# Small utilities
def _require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required to run this script.\n"
            f"Import error: {_TORCH_IMPORT_ERROR}"
        )


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def seed_all(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def get_device(device: str) -> "torch.device":
    """Resolve a device string into a torch.device.

    Accepted values: auto, cpu, cuda, mps
    """
    _require_torch()
    dev = device.strip().lower()
    if dev == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return torch.device("mps")
        return torch.device("cpu")
    if dev in {"cpu", "cuda", "mps"}:
        if dev == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested device=cuda but CUDA is not available.")
        if dev == "mps":
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is None or not mps_backend.is_available():
                raise RuntimeError("Requested device=mps but MPS is not available.")
        return torch.device(dev)
    raise ValueError(f"Unknown device: {device}")


def get_dtype(dtype: str) -> "torch.dtype":
    """Resolve dtype string into torch dtype."""
    _require_torch()
    d = dtype.strip().lower()
    if d in {"float32", "fp32", "f32"}:
        return torch.float32
    if d in {"float64", "fp64", "f64", "double"}:
        return torch.float64
    raise ValueError(f"Unknown dtype: {dtype}")


def _git_head_short() -> Optional[str]:
    """Best-effort git commit hash (short). Returns None if unavailable."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _env_summary() -> Dict[str, Any]:
    """Collect lightweight environment info for reproducibility."""
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "cwd": str(Path.cwd()),
        "git_head": _git_head_short(),
    }
    if torch is not None:
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_capability"] = torch.cuda.get_device_capability(0)
    # psutil is commonly available; if not, skip silently.
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["ram_total_gb"] = round(vm.total / (1024**3), 3)
    except Exception:
        pass
    # Java version (useful when MoonLight is enabled)
    try:
        out = subprocess.check_output(["java", "-version"], stderr=subprocess.STDOUT)
        info["java_version"] = out.decode("utf-8", errors="replace").splitlines()[0].strip()
    except Exception:
        info["java_version"] = None
    return info


# Heat 2D PINN model
class MLP(nn.Module):
    """Simple fully-connected MLP for PINN.

    Uses Xavier initialization and configurable activation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2 (including output layer)")
        act = activation.strip().lower()
        if act == "tanh":
            act_fn: nn.Module = nn.Tanh()
        elif act == "relu":
            act_fn = nn.ReLU()
        elif act == "gelu":
            act_fn = nn.GELU()
        elif act == "sine":
            act_fn = _Sine()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers: List[nn.Module] = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 2) + [output_dim]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act_fn)
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Sine(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return torch.sin(x)


def gaussian_hotspot(
    x: torch.Tensor,
    y: torch.Tensor,
    amp: float,
    cx: float,
    cy: float,
    sharpness: float,
) -> torch.Tensor:
    """Gaussian hotspot used as initial condition u(x,y,0)."""
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    return amp * torch.exp(-sharpness * r2)


def heat_residual(xyt: torch.Tensor, u: torch.Tensor, alpha: float) -> torch.Tensor:
    """Compute heat equation residual: u_t - alpha*(u_xx + u_yy)."""
    if xyt.requires_grad is False:
        raise ValueError("xyt must require grad for PDE residual computation.")
    if u.ndim != 2 or u.shape[1] != 1:
        raise ValueError("u must be shape [N,1].")

    grad_u = torch.autograd.grad(
        u,
        xyt,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    u_t = grad_u[:, 2:3]

    u_xx = torch.autograd.grad(
        u_x,
        xyt,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True,
    )[0][:, 0:1]
    u_yy = torch.autograd.grad(
        u_y,
        xyt,
        grad_outputs=torch.ones_like(u_y),
        create_graph=True,
        retain_graph=True,
    )[0][:, 1:2]

    return u_t - alpha * (u_xx + u_yy)


def smooth_max_violation(violation: torch.Tensor, beta: float) -> torch.Tensor:
    """Smooth approximation of `max(violation)` using log-mean-exp.

    We compute:
        sm = (logsumexp(beta * v) - log(n)) / beta

    This is a soft-max / log-mean-exp; it approaches max(v) as beta -> +inf, and the
    subtraction of log(n) avoids the large positive bias that plain logsumexp/beta
    introduces for big sample sizes.
    """
    if violation.ndim != 1:
        violation = violation.reshape(-1)
    if violation.numel() == 0:
        return torch.zeros((), device=violation.device, dtype=violation.dtype)
    b = float(beta)
    if b <= 0:
        raise ValueError("beta must be > 0")
    return (torch.logsumexp(b * violation, dim=0) - math.log(violation.numel())) / b


# Sampling utilities
def _rand_uniform(n: int, low: float, high: float, device: "torch.device", dtype: "torch.dtype") -> torch.Tensor:
    return low + (high - low) * torch.rand((n, 1), device=device, dtype=dtype)


def sample_interior_xyt(cfg: "TrainConfig", n: int, device: "torch.device", dtype: "torch.dtype") -> torch.Tensor:
    """Uniform random samples in the interior of space-time domain."""
    x = _rand_uniform(n, cfg.x_min, cfg.x_max, device, dtype)
    y = _rand_uniform(n, cfg.y_min, cfg.y_max, device, dtype)
    t = _rand_uniform(n, cfg.t_min, cfg.t_max, device, dtype)
    return torch.cat([x, y, t], dim=1)


def sample_initial_xyt(cfg: "TrainConfig", n: int, device: "torch.device", dtype: "torch.dtype") -> torch.Tensor:
    """Samples on the initial time slice t = t_min."""
    x = _rand_uniform(n, cfg.x_min, cfg.x_max, device, dtype)
    y = _rand_uniform(n, cfg.y_min, cfg.y_max, device, dtype)
    t = torch.full((n, 1), float(cfg.t_min), device=device, dtype=dtype)
    return torch.cat([x, y, t], dim=1)


def sample_boundary_xyt(cfg: "TrainConfig", n: int, device: "torch.device", dtype: "torch.dtype") -> torch.Tensor:
    """Samples on spatial boundary (Dirichlet) for random times."""
    # Choose boundary side uniformly: 0:x=x_min, 1:x=x_max, 2:y=y_min, 3:y=y_max
    side = torch.randint(0, 4, (n,), device=device)
    x = _rand_uniform(n, cfg.x_min, cfg.x_max, device, dtype).squeeze(1)
    y = _rand_uniform(n, cfg.y_min, cfg.y_max, device, dtype).squeeze(1)
    t = _rand_uniform(n, cfg.t_min, cfg.t_max, device, dtype).squeeze(1)

    x = torch.where(side == 0, torch.full_like(x, float(cfg.x_min)), x)
    x = torch.where(side == 1, torch.full_like(x, float(cfg.x_max)), x)
    y = torch.where(side == 2, torch.full_like(y, float(cfg.y_min)), y)
    y = torch.where(side == 3, torch.full_like(y, float(cfg.y_max)), y)

    return torch.stack([x, y, t], dim=1)


# STREL: graph + signal construction
def build_grid_edges(nx: int, ny: int, weight: float) -> List[List[float]]:
    """Build undirected 4-neighborhood edges for an (nx, ny) grid.

    Node indexing convention used by this script:
        node_id(ix, iy) = ix * ny + iy

    This matches NumPy's default C-order flatten for arrays with shape (nx, ny):
        flat = arr.reshape(nx * ny)

    Each edge record is [src, dst, w], where w is the edge weight (distance).
    """
    w = float(weight)
    edges: List[List[float]] = []
    for ix in range(nx):
        for iy in range(ny):
            i = ix * ny + iy
            if ix + 1 < nx:
                j = (ix + 1) * ny + iy
                edges.append([float(i), float(j), w])
                edges.append([float(j), float(i), w])
            if iy + 1 < ny:
                j = ix * ny + (iy + 1)
                edges.append([float(i), float(j), w])
                edges.append([float(j), float(i), w])
    return edges


def field_to_hot_signal_pm1(field_xy_t: np.ndarray, theta: float) -> List[List[List[float]]]:
    """Convert u(x,y,t) field into MoonLight boolean signal hot := (u >= theta).

    Returns a node-major list suitable for MoonLight:
        signal[node][time][feature]

    For boolean domain, encode:
        True  -> +1.0
        False -> -1.0
    """
    if field_xy_t.ndim != 3:
        raise ValueError("field_xy_t must be a 3D array [nx, ny, nt].")
    nx, ny, nt = field_xy_t.shape
    n_nodes = nx * ny

    # Flatten per time step in C-order to match node_id(ix,iy)=ix*ny+iy.
    # We build node-major by first flattening all nodes for each t, then transposing.
    hot_time_node = np.empty((nt, n_nodes), dtype=np.float64)
    for k in range(nt):
        flat = field_xy_t[:, :, k].reshape(n_nodes)
        hot = flat >= float(theta)
        hot_time_node[k, :] = np.where(hot, 1.0, -1.0)

    hot_node_time = hot_time_node.T  # [node, time]
    signal: List[List[List[float]]] = [
        [[float(hot_node_time[i, k])] for k in range(nt)] for i in range(n_nodes)
    ]
    return signal


def compute_time_axis(t_min: float, t_max: float, nt: int) -> List[float]:
    if nt <= 1:
        return [float(t_min)]
    return np.linspace(float(t_min), float(t_max), int(nt)).astype(np.float64).tolist()


def compute_max_temp_trace(field_xy_t: np.ndarray) -> np.ndarray:
    """Max over space at each time."""
    if field_xy_t.ndim != 3:
        raise ValueError("field_xy_t must be [nx, ny, nt]")
    nx, ny, nt = field_xy_t.shape
    flat = field_xy_t.reshape(nx * ny, nt)
    return flat.max(axis=0)


def compute_quench_time(time_axis: Sequence[float], max_temp: np.ndarray, theta: float) -> Optional[float]:
    """Return earliest time when max_temp < theta (strict)."""
    theta = float(theta)
    idx = np.where(max_temp < theta)[0]
    if idx.size == 0:
        return None
    k = int(idx[0])
    if 0 <= k < len(time_axis):
        return float(time_axis[k])
    return None


# MoonLight monitoring (STREL)
@dataclass(frozen=True)
class StrelSummary:
    ok: bool
    reason: str
    out_shape: Tuple[int, ...]
    per_time_min: Optional[List[float]]
    first_all_true_time: Optional[float]
    all_true_at_t0: Optional[bool]


def _summarize_moonlight_output(
    out: np.ndarray,
    time_axis: Sequence[float],
    n_nodes: int,
    nt: int,
) -> StrelSummary:
    """Summarize MoonLight output robustly.

    MoonLight's SpatialTemporalScriptComponent.toArray returns location-major:
        out[location][time][feature]

    But we still handle common permutations:
        out[time][location][feature]
        out[location][time]
        out[time][location]
    """
    arr = np.asarray(out)
    out_shape = tuple(int(x) for x in arr.shape)

    if arr.ndim < 2:
        return StrelSummary(
            ok=False,
            reason=f"Unexpected output rank: {arr.ndim}",
            out_shape=out_shape,
            per_time_min=None,
            first_all_true_time=None,
            all_true_at_t0=None,
        )

    # Determine orientation.
    # Preferred: node_time_(dim)
    assume = "node_time_dim"
    if arr.ndim >= 3 and arr.shape[0] == n_nodes and arr.shape[1] == nt:
        assume = "node_time_dim"
    elif arr.ndim >= 3 and arr.shape[0] == nt and arr.shape[1] == n_nodes:
        assume = "time_node_dim"
    elif arr.ndim == 2 and arr.shape[0] == n_nodes and arr.shape[1] == nt:
        assume = "node_time_2d"
    elif arr.ndim == 2 and arr.shape[0] == nt and arr.shape[1] == n_nodes:
        assume = "time_node_2d"
    else:
        # Fall back to node-major, which matches MoonLight toArray() for spatial signals.
        assume = "unknown_assume_node_time"

    # Compute per-time min across nodes (and features).
    if assume in {"node_time_dim", "node_time_2d", "unknown_assume_node_time"}:
        # time axis is 1
        if arr.ndim == 2:
            per_time = np.nanmin(arr, axis=0)  # [time]
        else:
            per_time = np.nanmin(arr, axis=tuple([0] + list(range(2, arr.ndim))))  # reduce nodes+features
    else:
        # time axis is 0
        if arr.ndim == 2:
            per_time = np.nanmin(arr, axis=1)  # [time]
        else:
            per_time = np.nanmin(arr, axis=tuple(range(1, arr.ndim)))  # reduce nodes+features

    if per_time.ndim != 1:
        return StrelSummary(
            ok=False,
            reason=f"Unexpected per_time shape: {per_time.shape}",
            out_shape=out_shape,
            per_time_min=None,
            first_all_true_time=None,
            all_true_at_t0=None,
        )

    if len(per_time) != len(time_axis):
        return StrelSummary(
            ok=False,
            reason=f"per_time length {len(per_time)} != time_axis length {len(time_axis)}",
            out_shape=out_shape,
            per_time_min=per_time.astype(np.float64).tolist(),
            first_all_true_time=None,
            all_true_at_t0=None,
        )

    # In boolean domain, +1 => True, -1 => False. Use >0 as True.
    per_time_list = per_time.astype(np.float64).tolist()
    satisfied = np.where(per_time > 0)[0]
    first_time: Optional[float] = None
    if satisfied.size > 0:
        k = int(satisfied[0])
        if 0 <= k < len(time_axis):
            first_time = float(time_axis[k])

    all_true_t0: Optional[bool] = None
    if len(time_axis) > 0:
        all_true_t0 = bool(per_time[0] > 0)

    return StrelSummary(
        ok=True,
        reason="ok",
        out_shape=out_shape,
        per_time_min=per_time_list,
        first_all_true_time=first_time,
        all_true_at_t0=all_true_t0,
    )


def run_strel_audit_moonlight(
    mls_path: Path,
    formula: str,
    edges: List[List[float]],
    time_axis: List[float],
    signal_node_time: List[List[List[float]]],
    tau: float,
) -> Tuple[Optional[np.ndarray], StrelSummary]:
    """Run MoonLight STREL monitoring (if available) and summarize output."""
    if ScriptLoader is None:
        return None, StrelSummary(
            ok=False,
            reason=f"MoonLight not available: {_MOONLIGHT_IMPORT_ERROR}",
            out_shape=(),
            per_time_min=None,
            first_all_true_time=None,
            all_true_at_t0=None,
        )

    if not mls_path.exists():
        return None, StrelSummary(
            ok=False,
            reason=f"MLScript not found: {mls_path}",
            out_shape=(),
            per_time_min=None,
            first_all_true_time=None,
            all_true_at_t0=None,
        )

    script = ScriptLoader.loadFromFile(str(mls_path))
    monitor = script.getMonitor(str(formula))

    # Prefer monitor_static for static graphs (avoids the verbose 'WithPrint' path).
    try:
        if hasattr(monitor, "monitor_static"):
            out = monitor.monitor_static(edges, time_axis, signal_node_time, float(tau))
        else:
            # Fallback: wrap static graph into time series with a single graph snapshot.
            graph_times = [float(time_axis[0]) if time_axis else 0.0]
            out = monitor.monitor(graph_times, [edges], time_axis, signal_node_time, float(tau))
        out_arr = np.asarray(out, dtype=np.float64)
    except Exception as e:  # pragma: no cover
        return None, StrelSummary(
            ok=False,
            reason=f"MoonLight monitor error: {e}",
            out_shape=(),
            per_time_min=None,
            first_all_true_time=None,
            all_true_at_t0=None,
        )

    # Summarize
    n_nodes = len(signal_node_time)
    nt = len(time_axis)
    summary = _summarize_moonlight_output(out_arr, time_axis, n_nodes=n_nodes, nt=nt)
    return out_arr, summary


# Training config + experiment runner
def _default_spec_path() -> Path:
    return Path(__file__).resolve().parent / "specs" / "contain_hotspot.mls"


@dataclass
class TrainConfig:
    # Domain
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    # PDE
    alpha: float = 0.25

    # IC hotspot params
    u0_amp: float = 1.0
    u0_cx: float = 0.5
    u0_cy: float = 0.5
    u0_sharpness: float = 80.0

    # PINN model
    hidden_dim: int = 128
    num_layers: int = 5
    activation: str = "tanh"

    # Training
    epochs: int = 2000
    lr: float = 1e-3
    n_collocation: int = 4096
    n_initial: int = 2048
    n_boundary: int = 2048

    # Optional soft safety surrogate (scaled by lambda_safety)
    lambda_safety: float = 0.0
    safety_threshold: float = 1.2
    safety_probes: int = 2048
    safety_beta: float = 10.0

    # Export grid (dense rollout)
    nx: int = 64
    ny: int = 64
    nt: int = 50
    eval_chunk: int = 16384

    # STREL monitoring config (hot predicate)
    mls_path: Path = dataclasses.field(default_factory=_default_spec_path)
    formula_name: str = "contain_hotspot"
    tau: float = 2.0
    hot_threshold: float = 0.6
    adj_weight: float = 1.0

    # Misc / I/O
    seed: int = 0
    device: str = "auto"
    dtype: str = "float32"
    out_root: Path = Path("runs/heat2d_strel_pinn")
    run_name: str = dataclasses.field(default_factory=_now_tag)

    # Controls
    plots: bool = True
    audit_strel: bool = False
    save_strel_raw: bool = False

    def as_jsonable(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["mls_path"] = str(self.mls_path)
        d["out_root"] = str(self.out_root)
        return d


def train_pinn(cfg: TrainConfig, run_dir: Path) -> Tuple[MLP, List[Dict[str, float]]]:
    """Train the PINN and return (model, history)."""
    _require_torch()
    device = get_device(cfg.device)
    dtype = get_dtype(cfg.dtype)

    seed_all(cfg.seed)

    model = MLP(
        input_dim=3,
        output_dim=1,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        activation=cfg.activation,
    ).to(device=device, dtype=dtype)

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))

    history: List[Dict[str, float]] = []
    metrics_path = run_dir / "metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=["epoch", "loss_total", "loss_pde", "loss_ic", "loss_bc", "loss_safety"],
        )
        writer.writeheader()

        t_start = time.perf_counter()
        for ep in range(1, int(cfg.epochs) + 1):
            # Sample points
            xyt_f = sample_interior_xyt(cfg, cfg.n_collocation, device, dtype)
            xyt_f.requires_grad_(True)
            u_f = model(xyt_f)
            res = heat_residual(xyt_f, u_f, alpha=float(cfg.alpha))
            loss_pde = torch.mean(res**2)

            xyt_i = sample_initial_xyt(cfg, cfg.n_initial, device, dtype)
            u_i = model(xyt_i)
            u0 = gaussian_hotspot(
                xyt_i[:, 0:1],
                xyt_i[:, 1:2],
                amp=float(cfg.u0_amp),
                cx=float(cfg.u0_cx),
                cy=float(cfg.u0_cy),
                sharpness=float(cfg.u0_sharpness),
            )
            loss_ic = torch.mean((u_i - u0) ** 2)

            xyt_b = sample_boundary_xyt(cfg, cfg.n_boundary, device, dtype)
            u_b = model(xyt_b)
            loss_bc = torch.mean(u_b**2)  # Dirichlet 0 boundary

            loss_safety = torch.zeros((), device=device, dtype=dtype)
            if cfg.lambda_safety > 0:
                xyt_s = sample_interior_xyt(cfg, cfg.safety_probes, device, dtype)
                u_s = model(xyt_s)
                violation = u_s.squeeze(-1) - float(cfg.safety_threshold)
                sm = smooth_max_violation(violation, beta=float(cfg.safety_beta))
                # Smooth hinge on the (smooth) maximum violation. For large beta this
                # approaches ReLU(max(v)), while remaining differentiable everywhere.
                loss_safety = F.softplus(sm, beta=float(cfg.safety_beta))

            loss_total = loss_pde + loss_ic + loss_bc + float(cfg.lambda_safety) * loss_safety

            opt.zero_grad(set_to_none=True)
            loss_total.backward()
            opt.step()

            row = {
                "epoch": float(ep),
                "loss_total": _safe_float(loss_total.item()),
                "loss_pde": _safe_float(loss_pde.item()),
                "loss_ic": _safe_float(loss_ic.item()),
                "loss_bc": _safe_float(loss_bc.item()),
                "loss_safety": _safe_float(loss_safety.item()),
            }
            history.append({k: float(v) for k, v in row.items() if k != "epoch"})
            writer.writerow({k: (int(v) if k == "epoch" else v) for k, v in row.items()})
            if ep == 1 or ep % 100 == 0 or ep == cfg.epochs:
                print(
                    f"[ep {ep:5d}/{cfg.epochs}] "
                    f"L={row['loss_total']:.3e} (pde={row['loss_pde']:.2e}, "
                    f"ic={row['loss_ic']:.2e}, bc={row['loss_bc']:.2e}, "
                    f"saf={row['loss_safety']:.2e})"
                )

        t_end = time.perf_counter()
        _json_dump({"train_seconds": t_end - t_start}, run_dir / "timing_train.json")

    return model, history


@torch.no_grad()
def export_field_xy_t(cfg: TrainConfig, model: MLP, run_dir: Path) -> np.ndarray:
    """Export u(x,y,t) on a regular grid (nx, ny, nt) to field_xy_t.npy."""
    _require_torch()
    device = get_device(cfg.device)
    dtype = get_dtype(cfg.dtype)

    nx, ny, nt = int(cfg.nx), int(cfg.ny), int(cfg.nt)
    xs = torch.linspace(float(cfg.x_min), float(cfg.x_max), nx, device=device, dtype=dtype)
    ys = torch.linspace(float(cfg.y_min), float(cfg.y_max), ny, device=device, dtype=dtype)
    ts = torch.linspace(float(cfg.t_min), float(cfg.t_max), nt, device=device, dtype=dtype)

    XX, YY = torch.meshgrid(xs, ys, indexing="ij")
    n_pts = nx * ny
    xy_flat = torch.stack([XX.reshape(-1), YY.reshape(-1)], dim=1)

    field = np.zeros((nx, ny, nt), dtype=np.float32)
    chunk = int(cfg.eval_chunk) if cfg.eval_chunk and cfg.eval_chunk > 0 else n_pts

    for k in range(nt):
        tval = ts[k].item()
        tcol = torch.full((n_pts, 1), float(tval), device=device, dtype=dtype)
        xyt = torch.cat([xy_flat, tcol], dim=1)

        # Chunked evaluation
        outs: List[torch.Tensor] = []
        for s in range(0, n_pts, chunk):
            outs.append(model(xyt[s : s + chunk]))
        u = torch.cat(outs, dim=0).reshape(nx, ny)
        field[:, :, k] = u.detach().cpu().numpy().astype(np.float32)

    out_path = run_dir / "field_xy_t.npy"
    np.save(out_path, field)
    return field


def plot_figures(cfg: TrainConfig, run_dir: Path, history: List[Dict[str, float]], field_xy_t: np.ndarray,
    strel: Optional[StrelSummary]) -> None:
    """Generate a small set of figures for the repo/report."""
    if not cfg.plots:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Loss curves
    if history:
        epochs = np.arange(1, len(history) + 1)
        loss_total = np.array([h["loss_total"] for h in history], dtype=np.float64)
        loss_pde = np.array([h["loss_pde"] for h in history], dtype=np.float64)
        loss_ic = np.array([h["loss_ic"] for h in history], dtype=np.float64)
        loss_bc = np.array([h["loss_bc"] for h in history], dtype=np.float64)
        loss_saf = np.array([h["loss_safety"] for h in history], dtype=np.float64)

        fig, ax = plt.subplots()
        ax.semilogy(epochs, loss_total, label="total")
        ax.semilogy(epochs, loss_pde, label="pde")
        ax.semilogy(epochs, loss_ic, label="ic")
        ax.semilogy(epochs, loss_bc, label="bc")
        if cfg.lambda_safety > 0:
            ax.semilogy(epochs, loss_saf, label="safety")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss (log)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "loss_curves.png", dpi=180)
        plt.close(fig)

    # Field snapshots
    nx, ny, nt = field_xy_t.shape
    ts = np.linspace(float(cfg.t_min), float(cfg.t_max), nt)

    snap_ids = [0, nt // 2, nt - 1] if nt >= 3 else list(range(nt))
    fig, axes = plt.subplots(1, len(snap_ids), figsize=(4 * len(snap_ids), 3))
    if len(snap_ids) == 1:
        axes = [axes]
    for ax, k in zip(axes, snap_ids):
        im = ax.imshow(field_xy_t[:, :, k].T, origin="lower", aspect="auto")
        ax.set_title(f"u(x,y,t) at t={ts[k]:.3f}")
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(fig_dir / "field_snapshots.png", dpi=180)
    plt.close(fig)

    # Max temperature trace + threshold
    max_temp = compute_max_temp_trace(field_xy_t)
    fig, ax = plt.subplots()
    ax.plot(ts, max_temp, label="max_x,y u")
    ax.axhline(float(cfg.hot_threshold), linestyle="--", label="hot threshold θ")
    ax.set_xlabel("time")
    ax.set_ylabel("temperature")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "max_temp_trace.png", dpi=180)
    plt.close(fig)

    # STREL satisfaction trace (min over nodes)
    if strel is not None and strel.per_time_min is not None:
        per_time = np.array(strel.per_time_min, dtype=np.float64)
        fig, ax = plt.subplots()
        ax.step(ts, per_time, where="post", label="min over nodes")
        ax.axhline(0.0, linestyle="--", label="0 boundary")
        ax.set_xlabel("time")
        ax.set_ylabel("MoonLight output (min)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "strel_per_time_min.png", dpi=180)
        plt.close(fig)


def run(cfg: TrainConfig) -> None:
    """Main entry point: train, export, (optional) audit, plot, write meta."""
    _require_torch()

    run_dir = Path(cfg.out_root) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config early
    _json_dump(cfg.as_jsonable(), run_dir / "config.json")

    env = _env_summary()

    # Train
    t0 = time.perf_counter()
    model, history = train_pinn(cfg, run_dir)
    t1 = time.perf_counter()

    # Export rollout
    field_xy_t = export_field_xy_t(cfg, model, run_dir)
    t2 = time.perf_counter()

    # CPU-only quench time for hot predicate
    time_axis = compute_time_axis(cfg.t_min, cfg.t_max, cfg.nt)
    max_temp = compute_max_temp_trace(field_xy_t)
    quench_time = compute_quench_time(time_axis, max_temp, cfg.hot_threshold)

    # Optional STREL audit
    strel_summary: Optional[StrelSummary] = None
    if cfg.audit_strel:
        edges = build_grid_edges(int(cfg.nx), int(cfg.ny), float(cfg.adj_weight))
        signal = field_to_hot_signal_pm1(field_xy_t, theta=float(cfg.hot_threshold))
        out_arr, strel_summary = run_strel_audit_moonlight(
            mls_path=Path(cfg.mls_path),
            formula=str(cfg.formula_name),
            edges=edges,
            time_axis=time_axis,
            signal_node_time=signal,
            tau=float(cfg.tau),
        )
        _json_dump(dataclasses.asdict(strel_summary), run_dir / "strel_summary.json")
        if cfg.save_strel_raw and out_arr is not None:
            np.save(run_dir / "strel_output_raw.npy", out_arr)

    # Plots
    plot_figures(cfg, run_dir, history, field_xy_t, strel_summary)

    # Meta
    spec_source: Optional[str] = None
    try:
        spec_source = Path(cfg.mls_path).read_text(encoding="utf-8")
    except Exception:
        spec_source = None

    meta: Dict[str, Any] = {
        "env": env,
        "runtime_seconds": {
            "train": t1 - t0,
            "export": t2 - t1,
            "total": t2 - t0,
        },
        "grid": {"nx": int(cfg.nx), "ny": int(cfg.ny), "nt": int(cfg.nt)},
        "hot_predicate": {"theta": float(cfg.hot_threshold), "cpu_quench_time": quench_time},
        "strel": {
            "enabled": bool(cfg.audit_strel),
            "mls_path": str(cfg.mls_path),
            "formula": str(cfg.formula_name),
            "tau": float(cfg.tau),
            "adj_weight": float(cfg.adj_weight),
            "summary": dataclasses.asdict(strel_summary) if strel_summary is not None else None,
            "spec_source": spec_source,
        },
    }
    _json_dump(meta, run_dir / "meta.json")

    print(f"Done. Outputs written to: {run_dir}")


# CLI
def _apply_quick_defaults(cfg: TrainConfig) -> TrainConfig:
    """Return a modified config for a CPU-friendly quick demo."""
    cfg.epochs = min(cfg.epochs, 250)
    cfg.hidden_dim = min(cfg.hidden_dim, 64)
    cfg.num_layers = min(cfg.num_layers, 4)

    cfg.n_collocation = min(cfg.n_collocation, 1024)
    cfg.n_initial = min(cfg.n_initial, 512)
    cfg.n_boundary = min(cfg.n_boundary, 512)
    cfg.safety_probes = min(cfg.safety_probes, 512)

    cfg.nx = min(cfg.nx, 32)
    cfg.ny = min(cfg.ny, 32)
    cfg.nt = min(cfg.nt, 40)
    cfg.eval_chunk = min(cfg.eval_chunk, 8192)
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a 2D heat PINN and optionally audit a STREL specification with MoonLight.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--quick", action="store_true", help="Apply CPU-friendly quick preset.")

    # Domain / PDE
    p.add_argument("--alpha", type=float, default=0.25, help="Heat diffusivity α.")
    p.add_argument("--t-max", type=float, default=1.0, help="Final time.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")

    # Model
    p.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer width.")
    p.add_argument("--num-layers", type=int, default=5, help="Number of layers (incl. output).")
    p.add_argument("--activation", type=str, default="tanh", help="Activation: tanh|relu|gelu|sine.")

    # Training
    p.add_argument("--epochs", type=int, default=2000, help="Training epochs.")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    p.add_argument("--n-collocation", type=int, default=4096, help="Collocation points per epoch.")
    p.add_argument("--n-initial", type=int, default=2048, help="Initial-condition points per epoch.")
    p.add_argument("--n-boundary", type=int, default=2048, help="Boundary points per epoch.")

    # Safety surrogate
    p.add_argument("--lambda-safety", type=float, default=0.0, help="Weight λ for the safety surrogate term.")
    p.add_argument("--safety-threshold", type=float, default=1.2, help="Safety threshold for u (u <= threshold).")
    p.add_argument("--safety-probes", type=int, default=2048, help="Probe points per epoch for safety surrogate.")
    p.add_argument("--safety-beta", type=float, default=10.0, help="Smooth-max sharpness (larger = closer to max).")

    # Export grid
    p.add_argument("--nx", type=int, default=64, help="Grid points in x.")
    p.add_argument("--ny", type=int, default=64, help="Grid points in y.")
    p.add_argument("--nt", type=int, default=50, help="Grid points in time.")
    p.add_argument("--eval-chunk", type=int, default=16384, help="Chunk size for rollout evaluation.")

    # STREL monitoring
    p.add_argument("--audit-strel", action="store_true", help="Run MoonLight STREL audit (requires MoonLight + Java).")
    p.add_argument("--mls", type=str, default=str(_default_spec_path()), help="Path to MoonLight .mls script.")
    p.add_argument("--formula", type=str, default="contain_hotspot", help="Formula name in the .mls script.")
    p.add_argument("--tau", type=float, default=2.0, help="STREL spatial radius parameter tau.")
    p.add_argument("--hot-threshold", type=float, default=0.6, help="θ for hot := (u >= θ).")
    p.add_argument("--adj-weight", type=float, default=1.0, help="Edge weight w for grid adjacency.")
    p.add_argument("--save-strel-raw", action="store_true", help="Save raw MoonLight output array as .npy.")

    # Runtime / output
    p.add_argument("--device", type=str, default="auto", help="Device: auto|cpu|cuda|mps.")
    p.add_argument("--dtype", type=str, default="float32", help="Dtype: float32|float64.")
    p.add_argument("--out-root", type=str, default="runs/heat2d_strel_pinn", help="Root directory for runs.")
    p.add_argument("--run-name", type=str, default=_now_tag(), help="Run subdirectory name.")
    p.add_argument("--plots", dest="plots", action="store_true", help="Enable plot generation.")
    p.add_argument("--no-plots", dest="plots", action="store_false", help="Disable plot generation.")
    p.set_defaults(plots=True)

    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    _require_torch()
    args = build_arg_parser().parse_args(argv)

    cfg = TrainConfig()
    cfg.alpha = float(args.alpha)
    cfg.t_max = float(args.t_max)
    cfg.seed = int(args.seed)

    cfg.hidden_dim = int(args.hidden_dim)
    cfg.num_layers = int(args.num_layers)
    cfg.activation = str(args.activation)

    cfg.epochs = int(args.epochs)
    cfg.lr = float(args.lr)
    cfg.n_collocation = int(args.n_collocation)
    cfg.n_initial = int(args.n_initial)
    cfg.n_boundary = int(args.n_boundary)

    cfg.lambda_safety = float(args.lambda_safety)
    cfg.safety_threshold = float(args.safety_threshold)
    cfg.safety_probes = int(args.safety_probes)
    cfg.safety_beta = float(args.safety_beta)

    cfg.nx = int(args.nx)
    cfg.ny = int(args.ny)
    cfg.nt = int(args.nt)
    cfg.eval_chunk = int(args.eval_chunk)

    cfg.audit_strel = bool(args.audit_strel)
    cfg.mls_path = Path(str(args.mls))
    cfg.formula_name = str(args.formula)
    cfg.tau = float(args.tau)
    cfg.hot_threshold = float(args.hot_threshold)
    cfg.adj_weight = float(args.adj_weight)
    cfg.save_strel_raw = bool(args.save_strel_raw)

    cfg.device = str(args.device)
    cfg.dtype = str(args.dtype)
    cfg.out_root = Path(str(args.out_root))
    cfg.run_name = str(args.run_name)
    cfg.plots = bool(args.plots)

    if args.quick:
        cfg = _apply_quick_defaults(cfg)

    run(cfg)


if __name__ == "__main__":
    main()
