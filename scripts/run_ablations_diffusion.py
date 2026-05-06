#!/usr/bin/env python3
"""
run_ablations_diffusion.py

Run STL-weight (λ) ablations for the 1D diffusion PINN example.

This script sweeps the STL penalty weight λ used by
`neural_pde_stl_strel.experiments.diffusion1d.run_diffusion1d` and records the
resulting training metrics, producing:

  1) A tiny, headerless 2-column CSV:  (lambda, robustness)
     suitable for `scripts/plot_ablations.py`.

  2) (By default) additional, publication-ready artifacts:
       * per-run CSV with loss components, runtimes, and field metrics
       * aggregated "Table 2"-style CSV (mean over final K epochs)
       * JSON metadata capturing hardware/software setup

Monitored specification

The monitored (training-time) STL safety property is:

    φ_safe := G_[0,T] ( reduce_x u(x,t) ≤ U_max )

where reduce_x is controlled by `--stl-spatial`:
  * mean   : average over the spatial grid (cheap, not worst-case)
  * amax   : hard maximum over the spatial grid (worst-case)
  * softmax: smooth max using LogSumExp (see `neural_pde_stl_strel.monitoring.stl_soft.softmax`)

Robustness reporting convention

The "robustness" value written to the main ablation CSV is, by default, the mean
robustness over the final `--tail` epochs (default: 50). This matches the
convention used in the report's Table 2 (losses/robustness averaged over the
final 50 epochs). You can instead record the last-epoch value via `--metric last`.

Design notes

* "Show what you ran with what specs":
    - The script bakes in explicit φ_safe and records the spatial reduction mode.
    - It optionally evaluates an "eventually cooling" spec on the final field
      (see `--cool-*`), useful for demonstrating F[...] style properties.

* "Include computational cost + hardware specs":
    - Per-run wall-clock training times are recorded.
    - A JSON metadata file records CPU, RAM, OS, Python, Torch, and (if present)
      GPU information.

* "Make plots/figures easier":
    - The output CSVs are structured so that `scripts/plot_ablations.py` can
      directly visualize robustness vs λ.
    - The per-run CSV contains loss components and robustness (ready for plots).

Usage examples

Default sweep (matches report): λ ∈ {0,2,4,6,8,10}
    python scripts/run_ablations_diffusion.py

Custom sweep and fewer epochs:
    python scripts/run_ablations_diffusion.py --weights 0 1 2 5 10 20 --epochs 200

Record last-epoch robustness instead of tail-mean:
    python scripts/run_ablations_diffusion.py --metric last

Disable extra outputs (only write the 2-col CSV):
    python scripts/run_ablations_diffusion.py --extras none

Notes on smooth-max ("softmax") temperature

This repository's smooth-max is LogSumExp. A standard bound is:

    max(x) ≤ LSE_tau(x) ≤ max(x) + tau * log(n)

Thus, if you choose `--stl-spatial softmax` with a relatively large tau and many
spatial samples, the smooth max may significantly overestimate the true max.
This is not "wrong" (it is the intended smooth abstraction), but it can be more
conservative than expected. Use smaller tau for a closer approximation.

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

SpatialAgg = Literal["mean", "softmax", "amax"]
MetricMode = Literal["tail_mean", "last"]
ExtrasMode = Literal["all", "none"]


@dataclass(frozen=True)
class CoolingSpec:
    """
    Optional "eventually cooling" property evaluated on the final field.

    We evaluate (soft and hard) robustness of:

        φ_cool := F_[t_start,t_end] ( u(x*, t) ≤ u_max )

    at x* = x_star (nearest grid location), using the same smooth temperature.
    """

    x_star: float
    u_max: float
    t_start: float
    t_end: float


@dataclass(frozen=True)
class Args:
    weights: list[float]
    epochs: int
    repeats: int
    seed: int
    u_max: float
    alpha: float
    stl_temp: float
    stl_spatial: SpatialAgg
    n_x: int
    n_t: int
    monitor_every: int
    monitor_n_x: int
    monitor_n_t: int
    device: str  # "cpu" | "cuda" | "mps" | "auto"
    results_dir: Path
    out: Path
    per_repeat: bool
    metric: MetricMode
    tail: int
    extras: ExtrasMode
    runs_out: Path
    table_out: Path
    meta_out: Path
    base_config: Path | None
    reuse_existing: bool
    quiet: bool
    cooling: CoolingSpec | None
    allow_proxy: bool


def _ensure_src_on_path() -> None:
    """
    Allow running from a source checkout without `pip install -e .`.

    If `neural_pde_stl_strel` is importable, do nothing.
    Else add `<repo_root>/src` to sys.path.
    """
    try:
        import neural_pde_stl_strel  # noqa: F401
        return
    except Exception:
        repo_root = Path(__file__).resolve().parents[1]
        src_dir = repo_root / "src"
        if src_dir.is_dir():
            sys.path.insert(0, str(src_dir))


def _format_float(x: float) -> str:
    # Stable, compact scientific formatting (matches existing repo conventions).
    return f"{x:.6g}"


def _derive_sibling_path(out: Path, suffix: str, ext: str) -> Path:
    return out.with_name(f"{out.stem}{suffix}{ext}")


def _seed_everything(seed: int) -> None:
    """
    Best-effort determinism. For GPU determinism, users may additionally set:
      CUBLAS_WORKSPACE_CONFIG, CUDA_LAUNCH_BLOCKING, etc.
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Determinism flags:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _parse_weights(raw: Sequence[str]) -> list[float]:
    """
    Parse weights from either:
      * explicit list: 0 2 4 6 8 10
      * range form:    a:b:n  (linspace from a to b inclusive with n points)
    """
    weights: list[float] = []
    for item in raw:
        if ":" in item:
            parts = item.split(":")
            if len(parts) != 3:
                raise ValueError(f"Bad range '{item}'. Expected a:b:n.")
            a, b = float(parts[0]), float(parts[1])
            n = int(parts[2])
            if n <= 0:
                raise ValueError(f"Bad range '{item}': n must be positive.")
            if n == 1:
                weights.append(a)
            else:
                step = (b - a) / (n - 1)
                weights.extend([a + i * step for i in range(n)])
        else:
            weights.append(float(item))
    return weights


def _read_csv_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            out: dict[str, float] = {}
            for k, v in r.items():
                if v is None:
                    continue
                try:
                    out[k] = float(v)
                except Exception:
                    continue
            rows.append(out)
    return rows


def _tail_mean(rows: list[dict[str, float]], key: str, tail: int) -> float:
    if not rows:
        return float("nan")
    tail_rows = rows[-tail:] if tail > 0 else rows
    vals = [r[key] for r in tail_rows if key in r and not math.isnan(r[key])]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _summarize_training_log(log_path: Path, tail: int) -> dict[str, float]:
    rows = _read_csv_rows(log_path)
    if not rows:
        raise RuntimeError(f"Empty log: {log_path}")
    last = rows[-1]
    keys = ("loss", "loss_pde", "loss_bcic", "loss_stl", "robustness", "lr")
    out: dict[str, float] = {}
    for k in keys:
        out[f"{k}_last"] = float(last.get(k, float("nan")))
        out[f"{k}_tail_mean"] = _tail_mean(rows, k, tail)
    return out


def _cpu_model() -> str | None:
    try:
        if platform.system().lower() == "linux":
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    try:
        m = platform.processor()
        return m.strip() if m else None
    except Exception:
        return None


def _ram_bytes() -> int | None:
    try:
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(pages) * int(page_size)
    except Exception:
        pass
    return None


def _git_commit(repo_root: Path) -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        return r.stdout.strip()
    except Exception:
        return None


def _collect_env_summary() -> dict[str, Any]:
    env: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "ram_bytes": _ram_bytes(),
    }
    try:
        import torch

        env["torch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        env["cuda_device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            env["cuda_device_name"] = torch.cuda.get_device_name(0)
            env["cuda_capability"] = torch.cuda.get_device_capability(0)
    except Exception:
        pass

    repo_root = Path(__file__).resolve().parents[1]
    env["git_commit"] = _git_commit(repo_root)
    return env


def _proxy_robustness(weight: float) -> float:
    """
    Deterministic fallback used only when real training cannot run.
    Monotone-ish in λ to keep plotting sane.
    """
    return math.sqrt(weight + 0.25) - 0.1 * weight


def _eval_field_metrics(
    field_path: Path,
    *,
    u_max: float,
    stl_spatial: SpatialAgg,
    stl_temp: float,
    alpha: float,
    cooling: CoolingSpec | None,
) -> dict[str, float]:
    """
    Compute additional evaluation metrics from the saved field:
      * u_max_xt: max_{x,t} u
      * rho_hard_max: u_max - max_{x,t} u   (hard robustness for G(u<=u_max) under max)
      * rho_soft_field: soft robustness on the full grid using stl_soft semantics
      * rel_l2_error: relative L2 error vs analytic sine solution (when applicable)
      * cooling robustness (optional): eventually property over a window
    """
    _ensure_src_on_path()

    import torch

    blob = torch.load(field_path, map_location="cpu")
    u = blob.get("u", None)
    X = blob.get("X", None)
    T = blob.get("T", None)
    if u is None:
        raise RuntimeError(f"Field file missing 'u': {field_path}")
    if X is None or T is None:
        # Not fatal for safety metrics, but needed for cooling and analytic error.
        X = torch.arange(u.shape[0], dtype=torch.float32)
        T = torch.arange(u.shape[1], dtype=torch.float32)

    u_max_xt = float(u.amax().item())
    rho_hard_max = float(u_max - u_max_xt)

    # Soft robustness (full-grid) consistent with repository's stl_soft semantics.
    from neural_pde_stl_strel.monitoring import stl_soft

    if stl_spatial == "mean":
        signal_t = u.mean(dim=0)
    elif stl_spatial == "amax":
        signal_t = u.amax(dim=0)
    elif stl_spatial == "softmax":
        # smooth max via log-sum-exp
        signal_t = stl_soft.softmax(u, temp=stl_temp, dim=0, keepdim=False)
    else:
        raise ValueError(f"Unknown stl_spatial: {stl_spatial}")

    rho_soft_field = float(
        stl_soft.always(
            stl_soft.pred_leq(signal_t, u_max),
            temp=stl_temp,
            time_dim=-1,
        ).item()
    )

    # Analytic relative L2 error for the sine IC case (diffusion benchmark).
    rel_l2_error = float("nan")
    try:
        from neural_pde_stl_strel.physics.diffusion1d import sine_solution

        # Broadcasting: X: (Nx,), T: (Nt,)
        u_exact = sine_solution(X, T, alpha=alpha)
        num = torch.linalg.vector_norm(u - u_exact).item()
        den = torch.linalg.vector_norm(u_exact).item()
        rel_l2_error = float(num / den) if den > 0 else float("nan")
    except Exception:
        pass

    out: dict[str, float] = {
        "u_max_xt": u_max_xt,
        "rho_hard_max": rho_hard_max,
        "rho_soft_field": rho_soft_field,
        "rel_l2_error": rel_l2_error,
    }

    # Optional cooling "eventually" robustness.
    if cooling is not None:
        # Find x index closest to x_star.
        x_star = float(cooling.x_star)
        t_start = float(cooling.t_start)
        t_end = float(cooling.t_end)
        cool_u_max = float(cooling.u_max)

        # Ensure sorted 1D tensors:
        X1 = X.flatten()
        T1 = T.flatten()

        ix = int(torch.argmin(torch.abs(X1 - x_star)).item())
        # map time window to indices
        t0_idx = int(torch.searchsorted(T1, torch.tensor(t_start), right=False).item())
        t1_excl = int(torch.searchsorted(T1, torch.tensor(t_end), right=True).item())
        t1_idx = max(0, t1_excl - 1)

        # Clip and order
        t0_idx = max(0, min(t0_idx, T1.numel() - 1))
        t1_idx = max(0, min(t1_idx, T1.numel() - 1))
        if t0_idx > t1_idx:
            t0_idx, t1_idx = t1_idx, t0_idx

        sig = u[ix, :]
        margin = stl_soft.pred_leq(sig, cool_u_max)
        window = margin[t0_idx : t1_idx + 1]

        rho_cool_hard = float(window.max().item()) if window.numel() > 0 else float("nan")
        rho_cool_soft = (
            float(stl_soft.eventually(window, temp=stl_temp, time_dim=-1).item())
            if window.numel() > 0
            else float("nan")
        )

        out.update(
            {
                "cool_x_index": float(ix),
                "cool_x_used": float(X1[ix].item()) if X1.numel() > 0 else float("nan"),
                "cool_t0_index": float(t0_idx),
                "cool_t1_index": float(t1_idx),
                "rho_cool_hard": rho_cool_hard,
                "rho_cool_soft": rho_cool_soft,
            }
        )

    return out


def _expected_artifact_paths(results_dir: Path, tag: str) -> tuple[Path, Path, Path]:
    """
    Diffusion1D experiment writes:
      diffusion1d_{tag}.pt
      diffusion1d_{tag}.csv
      diffusion1d_{tag}_field.pt
    """
    base = results_dir / f"diffusion1d_{tag}"
    ckpt = base.with_suffix(".pt")
    log = base.with_suffix(".csv")
    field = results_dir / f"diffusion1d_{tag}_field.pt"
    return ckpt, log, field


def _build_cfg(args: Args, *, stl_weight: float, seed: int, tag: str) -> dict[str, Any]:
    """
    Build a config dictionary for `run_diffusion1d`.

    CRITICAL DETAIL (bugfix vs older versions):
      diffusion1d reads the run identifier from TOP-LEVEL `cfg['tag']`,
      not only from `cfg['io']['tag']`.

    We therefore set BOTH:
      * cfg['tag'] = tag
      * cfg['io']['tag'] = tag   (harmless; keeps config self-descriptive)

    The results_dir is read from cfg['io']['results_dir'] if present.
    """
    cfg: dict[str, Any]

    if args.base_config is not None:
        import yaml

        cfg = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise ValueError(f"Base config is not a mapping: {args.base_config}")

        cfg.setdefault("grid", {})
        cfg.setdefault("optim", {})
        cfg.setdefault("physics", {})
        cfg.setdefault("model", {})
        cfg.setdefault("stl", {})
        cfg.setdefault("io", {})

        # Required overrides for ablation:
        cfg["seed"] = seed
        cfg["tag"] = tag  # REQUIRED by diffusion1d
        cfg["io"].update({"results_dir": str(args.results_dir), "tag": tag})

        cfg["grid"].update({"n_x": args.n_x, "n_t": args.n_t})
        cfg["optim"].update({"epochs": args.epochs})
        cfg["physics"].update({"alpha": args.alpha})

        cfg["stl"].update(
            {
                "use": True,
                "weight": stl_weight,
                "u_max": args.u_max,
                "temp": args.stl_temp,
                "spatial": args.stl_spatial,
                "every": args.monitor_every,
                "n_x": args.monitor_n_x,
                "n_t": args.monitor_n_t,
            }
        )

        # Device handling:
        if args.device != "auto":
            cfg["device"] = args.device

        return cfg

    # Built-in default config (CPU-friendly, deterministic).
    cfg = {
        "seed": seed,
        "tag": tag,  # REQUIRED by diffusion1d
        "io": {"results_dir": str(args.results_dir), "tag": tag},
        "model": {"hidden": [64, 64, 64], "act": "tanh"},
        "grid": {"n_x": args.n_x, "n_t": args.n_t},
        "optim": {"lr": 2e-3, "epochs": args.epochs, "batch": 4096},
        "physics": {"alpha": args.alpha},
        "stl": {
            "use": True,
            "weight": stl_weight,
            "u_max": args.u_max,
            "temp": args.stl_temp,
            "spatial": args.stl_spatial,
            "every": args.monitor_every,
            "n_x": args.monitor_n_x,
            "n_t": args.monitor_n_t,
        },
    }

    if args.device != "auto":
        cfg["device"] = args.device

    return cfg


@dataclass
class RunResult:
    stl_weight: float
    seed: int
    used_proxy: bool
    runtime_sec: float
    log_metrics: dict[str, float]
    field_metrics: dict[str, float]


def _run_one(args: Args, *, stl_weight: float, seed: int) -> RunResult:
    """
    Run training once for a given λ and seed, returning summarized metrics.

    If training cannot run and args.allow_proxy is True, returns deterministic
    proxy outputs (used_proxy=True). Otherwise raises.
    """
    _seed_everything(seed)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    tag = f"abl_w{stl_weight:g}_s{seed}"
    ckpt_path, log_path, field_path = _expected_artifact_paths(args.results_dir, tag)

    # Cache reuse
    if args.reuse_existing and log_path.exists() and field_path.exists():
        runtime = 0.0
    else:
        _ensure_src_on_path()
        try:
            from neural_pde_stl_strel.experiments.diffusion1d import run_diffusion1d
        except Exception as e:
            if not args.allow_proxy:
                raise
            # Proxy fallback
            rho = _proxy_robustness(stl_weight)
            log_metrics = {
                "robustness_last": rho,
                "robustness_tail_mean": rho,
                "loss_last": float("nan"),
                "loss_tail_mean": float("nan"),
                "loss_pde_last": float("nan"),
                "loss_pde_tail_mean": float("nan"),
                "loss_bcic_last": float("nan"),
                "loss_bcic_tail_mean": float("nan"),
                "loss_stl_last": float("nan"),
                "loss_stl_tail_mean": float("nan"),
                "lr_last": float("nan"),
                "lr_tail_mean": float("nan"),
            }
            field_metrics = {
                "u_max_xt": float("nan"),
                "rho_hard_max": float("nan"),
                "rho_soft_field": float("nan"),
                "rel_l2_error": float("nan"),
            }
            return RunResult(
                stl_weight=stl_weight,
                seed=seed,
                used_proxy=True,
                runtime_sec=0.0,
                log_metrics=log_metrics,
                field_metrics=field_metrics,
            )

        cfg = _build_cfg(args, stl_weight=stl_weight, seed=seed, tag=tag)
        t0 = time.perf_counter()
        try:
            run_diffusion1d(cfg)
        except Exception:
            if not args.allow_proxy:
                raise
            rho = _proxy_robustness(stl_weight)
            log_metrics = {
                "robustness_last": rho,
                "robustness_tail_mean": rho,
                "loss_last": float("nan"),
                "loss_tail_mean": float("nan"),
                "loss_pde_last": float("nan"),
                "loss_pde_tail_mean": float("nan"),
                "loss_bcic_last": float("nan"),
                "loss_bcic_tail_mean": float("nan"),
                "loss_stl_last": float("nan"),
                "loss_stl_tail_mean": float("nan"),
                "lr_last": float("nan"),
                "lr_tail_mean": float("nan"),
            }
            field_metrics = {
                "u_max_xt": float("nan"),
                "rho_hard_max": float("nan"),
                "rho_soft_field": float("nan"),
                "rel_l2_error": float("nan"),
            }
            return RunResult(
                stl_weight=stl_weight,
                seed=seed,
                used_proxy=True,
                runtime_sec=float(time.perf_counter() - t0),
                log_metrics=log_metrics,
                field_metrics=field_metrics,
            )

        runtime = float(time.perf_counter() - t0)

    # Summarize training log (tail mean + last)
    if not log_path.exists():
        if not args.allow_proxy:
            raise FileNotFoundError(f"Missing training log: {log_path}")
        rho = _proxy_robustness(stl_weight)
        log_metrics = {
            "robustness_last": rho,
            "robustness_tail_mean": rho,
            "loss_last": float("nan"),
            "loss_tail_mean": float("nan"),
            "loss_pde_last": float("nan"),
            "loss_pde_tail_mean": float("nan"),
            "loss_bcic_last": float("nan"),
            "loss_bcic_tail_mean": float("nan"),
            "loss_stl_last": float("nan"),
            "loss_stl_tail_mean": float("nan"),
            "lr_last": float("nan"),
            "lr_tail_mean": float("nan"),
        }
        field_metrics = {
            "u_max_xt": float("nan"),
            "rho_hard_max": float("nan"),
            "rho_soft_field": float("nan"),
            "rel_l2_error": float("nan"),
        }
        return RunResult(
            stl_weight=stl_weight,
            seed=seed,
            used_proxy=True,
            runtime_sec=runtime,
            log_metrics=log_metrics,
            field_metrics=field_metrics,
        )

    log_summary = _summarize_training_log(log_path, args.tail)

    # Normalize names into a stable schema
    log_metrics = {
        "robustness_last": log_summary["robustness_last"],
        "robustness_tail_mean": log_summary["robustness_tail_mean"],
        "loss_last": log_summary["loss_last"],
        "loss_tail_mean": log_summary["loss_tail_mean"],
        "loss_pde_last": log_summary["loss_pde_last"],
        "loss_pde_tail_mean": log_summary["loss_pde_tail_mean"],
        "loss_bcic_last": log_summary["loss_bcic_last"],
        "loss_bcic_tail_mean": log_summary["loss_bcic_tail_mean"],
        "loss_stl_last": log_summary["loss_stl_last"],
        "loss_stl_tail_mean": log_summary["loss_stl_tail_mean"],
        "lr_last": log_summary["lr_last"],
        "lr_tail_mean": log_summary["lr_tail_mean"],
    }

    # Field-based evaluation metrics
    field_metrics: dict[str, float] = {
        "u_max_xt": float("nan"),
        "rho_hard_max": float("nan"),
        "rho_soft_field": float("nan"),
        "rel_l2_error": float("nan"),
    }
    if field_path.exists():
        try:
            field_metrics = _eval_field_metrics(
                field_path,
                u_max=args.u_max,
                stl_spatial=args.stl_spatial,
                stl_temp=args.stl_temp,
                alpha=args.alpha,
                cooling=args.cooling,
            )
        except Exception:
            # Keep going; logs are still valid for the ablation CSV.
            pass

    return RunResult(
        stl_weight=stl_weight,
        seed=seed,
        used_proxy=False,
        runtime_sec=runtime,
        log_metrics=log_metrics,
        field_metrics=field_metrics,
    )


def _parse_args() -> Args:
    p = argparse.ArgumentParser(description="Run diffusion1d STL-weight ablations.")
    p.add_argument(
        "--weights",
        nargs="*",
        default=None,
        help="List of λ values or ranges a:b:n. Default: 0 2 4 6 8 10",
    )
    p.add_argument("--epochs", type=int, default=150, help="Training epochs per run.")
    p.add_argument("--repeats", type=int, default=1, help="Repeats per λ (different seeds).")
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    p.add_argument("--u-max", type=float, default=1.0, help="Safety bound U_max.")
    p.add_argument("--alpha", type=float, default=0.1, help="Diffusion coefficient α.")
    p.add_argument("--stl-temp", type=float, default=0.1, help="STL soft temperature τ.")
    p.add_argument(
        "--stl-spatial",
        choices=["mean", "softmax", "amax"],
        default="mean",
        help="Spatial reduction reduce_x for φ_safe.",
    )
    p.add_argument("--n-x", type=int, default=128, help="Full evaluation grid points in x.")
    p.add_argument("--n-t", type=int, default=64, help="Full evaluation grid points in t.")
    p.add_argument(
        "--monitor-every",
        type=int,
        default=1,
        help="Monitor robustness every N epochs (training-time logging).",
    )
    p.add_argument(
        "--monitor-n-x",
        type=int,
        default=64,
        help="Monitor grid points in x (training-time robustness grid).",
    )
    p.add_argument(
        "--monitor-n-t",
        type=int,
        default=64,
        help="Monitor grid points in t (training-time robustness grid).",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="cpu",
        help="Training device (default: cpu). Use 'auto' to let diffusion1d decide.",
    )
    p.add_argument("--results-dir", type=str, default="results", help="Where to write artifacts.")
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path (2 columns). Default: <results-dir>/diffusion1d_ablations.csv",
    )
    p.add_argument(
        "--per-repeat",
        action="store_true",
        help="If set, write one row per repeat (λ, ρ). Otherwise average across repeats.",
    )
    p.add_argument(
        "--metric",
        choices=["tail_mean", "last"],
        default="tail_mean",
        help="Which robustness value to write to the ablation CSV.",
    )
    p.add_argument(
        "--tail",
        type=int,
        default=50,
        help="Number of final epochs used for tail-mean metrics (default: 50).",
    )
    p.add_argument(
        "--extras",
        choices=["all", "none"],
        default="all",
        help="Write extra CSV/JSON outputs (runs/table/meta) in addition to main CSV.",
    )
    p.add_argument("--base-config", type=str, default=None, help="Optional YAML config to load.")
    p.add_argument(
        "--reuse-existing",
        action="store_true",
        help="If set, reuse existing run artifacts when present.",
    )
    p.add_argument("--quiet", action="store_true", help="Reduce console printing.")
    p.add_argument(
        "--allow-proxy",
        action="store_true",
        help="Allow deterministic proxy fallback if real training cannot run.",
    )

    # Eventually cooling spec parameters (optional)
    p.add_argument("--cool-x", type=float, default=0.5, help="x* for φ_cool (nearest grid point).")
    p.add_argument("--cool-u-max", type=float, default=0.5, help="Cooling threshold for φ_cool.")
    p.add_argument("--cool-t-start", type=float, default=0.5, help="Start time for φ_cool window.")
    p.add_argument("--cool-t-end", type=float, default=1.0, help="End time for φ_cool window.")
    p.add_argument(
        "--no-cooling",
        action="store_true",
        help="Disable evaluating the optional eventual cooling spec on the final field.",
    )

    ns = p.parse_args()

    raw_weights = ns.weights if ns.weights is not None else ["0", "2", "4", "6", "8", "10"]
    weights = _parse_weights(raw_weights)

    results_dir = Path(ns.results_dir).expanduser().resolve()
    out = Path(ns.out).expanduser().resolve() if ns.out else (results_dir / "diffusion1d_ablations.csv")

    # Derive extra outputs (even if extras=none we still fill paths for simplicity).
    runs_out = _derive_sibling_path(out, "_runs", ".csv")
    table_out = _derive_sibling_path(out, "_table", ".csv")
    meta_out = _derive_sibling_path(out, "_meta", ".json")

    base_config = Path(ns.base_config).expanduser().resolve() if ns.base_config else None
    if base_config is not None and not base_config.exists():
        raise FileNotFoundError(f"Base config not found: {base_config}")

    cooling = None
    if not ns.no_cooling:
        cooling = CoolingSpec(
            x_star=float(ns.cool_x),
            u_max=float(ns.cool_u_max),
            t_start=float(ns.cool_t_start),
            t_end=float(ns.cool_t_end),
        )

    return Args(
        weights=weights,
        epochs=int(ns.epochs),
        repeats=int(ns.repeats),
        seed=int(ns.seed),
        u_max=float(ns.u_max),
        alpha=float(ns.alpha),
        stl_temp=float(ns.stl_temp),
        stl_spatial=ns.stl_spatial,
        n_x=int(ns.n_x),
        n_t=int(ns.n_t),
        monitor_every=int(ns.monitor_every),
        monitor_n_x=int(ns.monitor_n_x),
        monitor_n_t=int(ns.monitor_n_t),
        device=str(ns.device),
        results_dir=results_dir,
        out=out,
        per_repeat=bool(ns.per_repeat),
        metric=ns.metric,
        tail=int(ns.tail),
        extras=ns.extras,
        runs_out=runs_out,
        table_out=table_out,
        meta_out=meta_out,
        base_config=base_config,
        reuse_existing=bool(ns.reuse_existing),
        quiet=bool(ns.quiet),
        cooling=cooling,
        allow_proxy=bool(ns.allow_proxy),
    )


def main() -> None:
    args = _parse_args()

    t_start = time.time()

    # Run sweep
    results: list[RunResult] = []
    for lam in args.weights:
        for r in range(args.repeats):
            seed = args.seed + r
            rr = _run_one(args, stl_weight=lam, seed=seed)
            results.append(rr)

            if not args.quiet:
                key = "robustness_tail_mean" if args.metric == "tail_mean" else "robustness_last"
                rho = rr.log_metrics[key]
                print(
                    f"λ={lam:g}, seed={seed}: "
                    f"ρ({args.metric})={rho:.6g}, wall={rr.runtime_sec:.2f}s"
                    + (" [PROXY]" if rr.used_proxy else "")
                )

    # Write main ablation CSV (headerless: λ, ρ)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.per_repeat:
        rows = []
        for rr in results:
            key = "robustness_tail_mean" if args.metric == "tail_mean" else "robustness_last"
            rho = rr.log_metrics[key]
            rows.append((rr.stl_weight, rho))
        with args.out.open("w", encoding="utf-8") as f:
            for lam, rho in rows:
                f.write(f"{_format_float(lam)},{_format_float(rho)}\n")
    else:
        by_lam: dict[float, list[float]] = {}
        for rr in results:
            key = "robustness_tail_mean" if args.metric == "tail_mean" else "robustness_last"
            rho = rr.log_metrics[key]
            by_lam.setdefault(rr.stl_weight, []).append(rho)
        with args.out.open("w", encoding="utf-8") as f:
            for lam in sorted(by_lam.keys()):
                vals = by_lam[lam]
                mean = sum(vals) / len(vals)
                f.write(f"{_format_float(lam)},{_format_float(mean)}\n")

    if args.extras == "none":
        return

    # Per-run CSV (includes losses, runtime, field metrics, cooling robustness, etc.)
    args.runs_out.parent.mkdir(parents=True, exist_ok=True)

    run_fields = [
        "lambda",
        "seed",
        "used_proxy",
        "runtime_sec",
        "robustness_last",
        "robustness_tail_mean",
        "loss_last",
        "loss_tail_mean",
        "loss_pde_last",
        "loss_pde_tail_mean",
        "loss_bcic_last",
        "loss_bcic_tail_mean",
        "loss_stl_last",
        "loss_stl_tail_mean",
        "lr_last",
        "lr_tail_mean",
        # field metrics
        "u_max_xt",
        "rho_hard_max",
        "rho_soft_field",
        "rel_l2_error",
        # cooling (optional)
        "cool_x_index",
        "cool_x_used",
        "cool_t0_index",
        "cool_t1_index",
        "rho_cool_hard",
        "rho_cool_soft",
    ]

    with args.runs_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=run_fields)
        w.writeheader()
        for rr in results:
            row: dict[str, str] = {
                "lambda": _format_float(rr.stl_weight),
                "seed": str(rr.seed),
                "used_proxy": str(rr.used_proxy),
                "runtime_sec": _format_float(rr.runtime_sec),
            }
            for k in (
                "robustness_last",
                "robustness_tail_mean",
                "loss_last",
                "loss_tail_mean",
                "loss_pde_last",
                "loss_pde_tail_mean",
                "loss_bcic_last",
                "loss_bcic_tail_mean",
                "loss_stl_last",
                "loss_stl_tail_mean",
                "lr_last",
                "lr_tail_mean",
            ):
                row[k] = _format_float(float(rr.log_metrics.get(k, float("nan"))))

            for k in (
                "u_max_xt",
                "rho_hard_max",
                "rho_soft_field",
                "rel_l2_error",
                "cool_x_index",
                "cool_x_used",
                "cool_t0_index",
                "cool_t1_index",
                "rho_cool_hard",
                "rho_cool_soft",
            ):
                row[k] = _format_float(float(rr.field_metrics.get(k, float("nan"))))

            w.writerow(row)

    # Aggregated "Table 2"-style CSV:
    # mean over repeats of tail-mean metrics (losses + rho_tail_mean).
    args.table_out.parent.mkdir(parents=True, exist_ok=True)

    # group results by lambda
    grouped: dict[float, list[RunResult]] = {}
    for rr in results:
        grouped.setdefault(rr.stl_weight, []).append(rr)

    with args.table_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lambda", "L_total", "L_pde", "L_icbc", "L_stl", "rho_soft"])
        for lam in sorted(grouped.keys()):
            group = grouped[lam]
            n = len(group)

            def mean_key(key: str) -> float:
                vals = [g.log_metrics.get(key, float("nan")) for g in group]
                vals = [v for v in vals if v is not None and not math.isnan(float(v))]
                return float(sum(vals) / len(vals)) if vals else float("nan")

            L_total = mean_key("loss_tail_mean")
            L_pde = mean_key("loss_pde_tail_mean")
            L_icbc = mean_key("loss_bcic_tail_mean")
            L_stl = mean_key("loss_stl_tail_mean")
            rho = mean_key("robustness_tail_mean")

            w.writerow([
                _format_float(lam), _format_float(L_total),
                _format_float(L_pde), _format_float(L_icbc),
                _format_float(L_stl), _format_float(rho),
            ])

    # Metadata JSON (hardware/software setup + args + outputs)
    args.meta_out.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start)),
        "duration_sec": float(time.time() - t_start),
        "environment": _collect_env_summary(),
        "args": {
            "weights": args.weights,
            "epochs": args.epochs,
            "repeats": args.repeats,
            "seed": args.seed,
            "u_max": args.u_max,
            "alpha": args.alpha,
            "stl_temp": args.stl_temp,
            "stl_spatial": args.stl_spatial,
            "n_x": args.n_x,
            "n_t": args.n_t,
            "monitor_every": args.monitor_every,
            "monitor_n_x": args.monitor_n_x,
            "monitor_n_t": args.monitor_n_t,
            "device": args.device,
            "tail": args.tail,
            "metric": args.metric,
            "per_repeat": args.per_repeat,
            "reuse_existing": args.reuse_existing,
            "base_config": str(args.base_config) if args.base_config else None,
            "cooling": None
            if args.cooling is None
            else {
                "x_star": args.cooling.x_star,
                "u_max": args.cooling.u_max,
                "t_start": args.cooling.t_start,
                "t_end": args.cooling.t_end,
            },
        },
        "outputs": {
            "main_csv": str(args.out),
            "runs_csv": str(args.runs_out),
            "table_csv": str(args.table_out),
            "meta_json": str(args.meta_out),
        },
    }

    args.meta_out.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
