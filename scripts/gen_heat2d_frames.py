#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts/gen_heat2d_frames.py

Generate a small 2D heat-equation rollout (and optional figures) for STL/STREL demos.

This script simulates the 2D heat equation

    u_t = α (u_xx + u_yy)

on a Cartesian grid for ``nt`` time steps, producing outputs that are easy to use in
the repository's MoonLight STREL monitoring pipeline (see
``scripts/eval_heat2d_moonlight.py``).

Outputs
-------
Depending on flags, the script can write:

1) Per-time frames (NumPy .npy):
   ``outdir/frame_0000.npy, frame_0001.npy, ...``

2) Packed spatio-temporal tensor (NumPy .npy), MoonLight-friendly:
   - ``field_xy_t.npy`` with shape (nx, ny, nt), or
   - ``field_t_xy.npy`` with shape (nt, nx, ny)

3) ``meta.json`` containing parameters + stability info + basic runtime/environment details.

4) Optional figures (PNG) suitable for a report/slides:
   - ``heat2d_max_temp.png``: max temperature vs normalized time (matches the report's
     style for the scalar heat sandbox).
   - ``heat2d_quench_vs_theta.png``: quench time tq vs threshold θ (tq=1 means not
     quenched within the simulated horizon).
   - ``heat2d_mosaic.png``: a small mosaic of field snapshots.

Reproducing the committed scalar heat sandbox asset
The repository includes a small scalar (non-ML) heat rollout under ``assets/heat2d_scalar``.
This command reproduces it and also writes the figures:

    python scripts/gen_heat2d_frames.py \
      --nx 32 --ny 32 --nt 50 --dt 0.05 --alpha 0.5 \
      --bc periodic --method ftcs \
      --init gaussian --sigma 0.15 --amplitude 1.0 --noise 0.01 --seed 0 \
      --also-pack --layout xy_t --no-frames --plots \
      --outdir assets/heat2d_scalar

Notes on numerics
- FTCS (explicit 5-point stencil) is used by default. For periodic/Neumann/Dirichlet
  BCs, the standard 2D stability condition is enforced:

      r_x + r_y <= 1/2
      where r_x = α dt / dx^2, r_y = α dt / dy^2.

- An FFT-based integrator is also provided for periodic BCs. It solves the *continuous*
  periodic heat equation exactly in Fourier space (it is unconditionally stable, but
  it is not the same as integrating the discrete 5-point Laplacian).

This file is intentionally self-contained and CPU-friendly.

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
from typing import Iterable, Literal

import numpy as np

BCType = Literal["periodic", "neumann", "dirichlet"]
MethodType = Literal["ftcs", "fft"]
InitType = Literal["gaussian", "ring", "checker"]
LayoutType = Literal["xy_t", "t_xy"]


# Configuration and parsing


@dataclass(frozen=True)
class Heat2DConfig:
    # Grid / time
    nx: int = 32
    ny: int = 32
    nt: int = 50
    dt: float = 0.05
    alpha: float = 0.5
    dx: float = 1.0
    dy: float = 1.0

    # Model + boundary conditions
    bc: BCType = "periodic"
    method: MethodType = "ftcs"
    dirichlet_value: float = 0.0

    # Initial condition
    init: InitType = "gaussian"
    sigma: float = 0.15
    amplitude: float = 1.0
    noise: float = 0.01
    seed: int | None = 0

    # Output
    outdir: Path = Path("assets/heat2d_scalar")
    save_every: int = 1
    dtype: str = "float32"
    save_frames: bool = True
    also_pack: bool = False
    layout: LayoutType = "xy_t"

    # Auto-dt (FTCS only)
    auto_dt: bool = False
    target_dt: float = 0.05
    safety: float = 0.98

    # Plots
    plots: bool = False
    theta_min: float = 0.2
    theta_max: float = 0.9
    theta_num: int = 40


def _positive_int(x: str) -> int:
    v = int(x)
    if v <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return v


def _positive_float(x: str) -> float:
    v = float(x)
    if not math.isfinite(v) or v <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and > 0")
    return v


def _nonneg_float(x: str) -> float:
    v = float(x)
    if not math.isfinite(v) or v < 0.0:
        raise argparse.ArgumentTypeError("must be finite and >= 0")
    return v


def _parse_dtype(name: str) -> np.dtype:
    try:
        return np.dtype(name)
    except TypeError as e:
        raise argparse.ArgumentTypeError(f"invalid dtype: {name!r}") from e


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate a small 2D heat-equation rollout (optionally with plots) "
            "for STL/STREL demos. Defaults match the committed assets/heat2d_scalar rollout."
        )
    )

    # Grid/time
    p.add_argument("--nx", type=_positive_int, default=Heat2DConfig.nx)
    p.add_argument("--ny", type=_positive_int, default=Heat2DConfig.ny)
    p.add_argument("--nt", type=_positive_int, default=Heat2DConfig.nt)
    p.add_argument("--dt", type=_positive_float, default=Heat2DConfig.dt)
    p.add_argument("--alpha", type=_positive_float, default=Heat2DConfig.alpha)
    p.add_argument("--dx", type=_positive_float, default=Heat2DConfig.dx)
    p.add_argument("--dy", type=_positive_float, default=Heat2DConfig.dy)

    # BC/method
    p.add_argument("--bc", choices=["periodic", "neumann", "dirichlet"], default=Heat2DConfig.bc)
    p.add_argument("--method", choices=["ftcs", "fft"], default=Heat2DConfig.method)
    p.add_argument("--dirichlet-value", type=float, default=Heat2DConfig.dirichlet_value)

    # Init
    p.add_argument("--init", choices=["gaussian", "ring", "checker"], default=Heat2DConfig.init)
    p.add_argument("--sigma", type=_positive_float, default=Heat2DConfig.sigma)
    p.add_argument("--amplitude", type=_positive_float, default=Heat2DConfig.amplitude)
    p.add_argument("--noise", type=_nonneg_float, default=Heat2DConfig.noise)
    p.add_argument("--seed", type=int, default=Heat2DConfig.seed)

    # Output
    p.add_argument("--outdir", type=Path, default=Heat2DConfig.outdir)
    p.add_argument("--save-every", type=_positive_int, default=Heat2DConfig.save_every)
    p.add_argument("--dtype", type=str, default=Heat2DConfig.dtype)
    frames_group = p.add_mutually_exclusive_group()
    frames_group.add_argument("--frames", dest="save_frames", action="store_true", default=True)
    frames_group.add_argument("--no-frames", dest="save_frames", action="store_false")
    p.add_argument(
        "--also-pack",
        action="store_true",
        help="Also write a single packed tensor file (field_xy_t.npy or field_t_xy.npy).",
    )
    p.add_argument("--layout", choices=["xy_t", "t_xy"], default=Heat2DConfig.layout)

    # Auto-dt
    p.add_argument(
        "--auto-dt",
        action="store_true",
        help="(FTCS only) Override dt using a stability-based limit: dt = min(target_dt, safety*dt_limit).",
    )
    p.add_argument("--target-dt", type=_positive_float, default=Heat2DConfig.target_dt)
    p.add_argument("--safety", type=_positive_float, default=Heat2DConfig.safety)

    # Plots
    p.add_argument(
        "--plots",
        action="store_true",
        help="Write summary PNG figures (max temp, quench vs theta, mosaic) to outdir.",
    )
    p.add_argument("--theta-min", type=float, default=Heat2DConfig.theta_min)
    p.add_argument("--theta-max", type=float, default=Heat2DConfig.theta_max)
    p.add_argument("--theta-num", type=_positive_int, default=Heat2DConfig.theta_num)

    return p


# Numerics


def _stability_limit_dt(alpha: float, dx: float, dy: float) -> float:
    """FTCS stability limit: r_x + r_y <= 1/2."""

    inv = (1.0 / (dx * dx)) + (1.0 / (dy * dy))
    return 0.5 / (alpha * inv)


def _init_field(cfg: Heat2DConfig) -> np.ndarray:
    """Create u(x,y,0) in float32 (compute dtype)."""

    nx, ny = cfg.nx, cfg.ny
    x = np.linspace(-1.0, 1.0, nx, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="ij")

    if cfg.init == "gaussian":
        r2 = X * X + Y * Y
        u = cfg.amplitude * np.exp(-r2 / (2.0 * (cfg.sigma**2)))
    elif cfg.init == "ring":
        r = np.sqrt(X * X + Y * Y)
        u = cfg.amplitude * np.exp(-((r - 0.45) / cfg.sigma) ** 2)
    elif cfg.init == "checker":
        period = max(1, nx // 8)
        u = (((np.arange(nx)[:, None] // period + np.arange(ny)[None, :] // period) % 2) * 2 - 1).astype(
            np.float32
        )
        u = cfg.amplitude * u
    else:
        raise ValueError(f"Unknown init: {cfg.init}")

    u = np.asarray(u, dtype=np.float32)

    if cfg.noise > 0.0:
        rng = np.random.default_rng(cfg.seed)
        u = u + (cfg.noise * rng.standard_normal(size=u.shape, dtype=np.float32))

    if cfg.bc == "dirichlet":
        u = u.copy()
        u[0, :] = cfg.dirichlet_value
        u[-1, :] = cfg.dirichlet_value
        u[:, 0] = cfg.dirichlet_value
        u[:, -1] = cfg.dirichlet_value

    return u.astype(np.float32, copy=False)


def _laplacian_periodic(u: np.ndarray, dx: float, dy: float) -> np.ndarray:
    lap = (
        (np.roll(u, 1, axis=0) - 2.0 * u + np.roll(u, -1, axis=0)) / (dx * dx)
        + (np.roll(u, 1, axis=1) - 2.0 * u + np.roll(u, -1, axis=1)) / (dy * dy)
    )
    return lap.astype(np.float32, copy=False)


def _laplacian_neumann(u: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Zero-flux (Neumann) boundaries via mirrored ghost cells."""

    lap = np.zeros_like(u, dtype=np.float32)
    lap[1:-1, 1:-1] = (
        (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / (dx * dx)
        + (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]) / (dy * dy)
    )

    lap[0, 1:-1] = (2.0 * (u[1, 1:-1] - u[0, 1:-1]) / (dx * dx)) + (
        (u[0, 2:] - 2.0 * u[0, 1:-1] + u[0, :-2]) / (dy * dy)
    )
    lap[-1, 1:-1] = (2.0 * (u[-2, 1:-1] - u[-1, 1:-1]) / (dx * dx)) + (
        (u[-1, 2:] - 2.0 * u[-1, 1:-1] + u[-1, :-2]) / (dy * dy)
    )
    lap[1:-1, 0] = ((u[2:, 0] - 2.0 * u[1:-1, 0] + u[:-2, 0]) / (dx * dx)) + (
        2.0 * (u[1:-1, 1] - u[1:-1, 0]) / (dy * dy)
    )
    lap[1:-1, -1] = ((u[2:, -1] - 2.0 * u[1:-1, -1] + u[:-2, -1]) / (dx * dx)) + (
        2.0 * (u[1:-1, -2] - u[1:-1, -1]) / (dy * dy)
    )

    lap[0, 0] = 2.0 * (u[1, 0] - u[0, 0]) / (dx * dx) + 2.0 * (u[0, 1] - u[0, 0]) / (dy * dy)
    lap[0, -1] = 2.0 * (u[1, -1] - u[0, -1]) / (dx * dx) + 2.0 * (u[0, -2] - u[0, -1]) / (dy * dy)
    lap[-1, 0] = 2.0 * (u[-2, 0] - u[-1, 0]) / (dx * dx) + 2.0 * (u[-1, 1] - u[-1, 0]) / (dy * dy)
    lap[-1, -1] = 2.0 * (u[-2, -1] - u[-1, -1]) / (dx * dx) + 2.0 * (u[-1, -2] - u[-1, -1]) / (dy * dy)

    return lap


def _step_ftcs(
    u: np.ndarray,
    alpha: float,
    dt: float,
    dx: float,
    dy: float,
    bc: BCType,
    dirichlet_value: float,
) -> np.ndarray:
    if bc == "periodic":
        lap = _laplacian_periodic(u, dx, dy)
        return (u + (alpha * dt) * lap).astype(np.float32, copy=False)

    if bc == "neumann":
        lap = _laplacian_neumann(u, dx, dy)
        return (u + (alpha * dt) * lap).astype(np.float32, copy=False)

    if bc == "dirichlet":
        lap = np.zeros_like(u, dtype=np.float32)
        lap[1:-1, 1:-1] = (
            (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / (dx * dx)
            + (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]) / (dy * dy)
        )
        u_new = (u + (alpha * dt) * lap).astype(np.float32, copy=False)
        u_new = u_new.copy()
        u_new[0, :] = dirichlet_value
        u_new[-1, :] = dirichlet_value
        u_new[:, 0] = dirichlet_value
        u_new[:, -1] = dirichlet_value
        return u_new

    raise ValueError(f"Unknown BC: {bc}")


def _make_fft_multiplier(nx: int, ny: int, dx: float, dy: float, alpha: float, dt: float) -> np.ndarray:
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k2 = (KX * KX + KY * KY).astype(np.float32, copy=False)
    return np.exp((-alpha * dt) * k2).astype(np.float32, copy=False)


def _iter_frames(cfg: Heat2DConfig, dt: float) -> Iterable[np.ndarray]:
    u = _init_field(cfg)
    yield u

    if cfg.method == "ftcs":
        for _ in range(1, cfg.nt):
            u = _step_ftcs(u, cfg.alpha, dt, cfg.dx, cfg.dy, cfg.bc, cfg.dirichlet_value)
            yield u
        return

    # FFT integrator (periodic only)
    if cfg.bc != "periodic":
        raise ValueError("FFT integrator requires periodic boundary conditions.")
    U = np.fft.fft2(u)
    E = _make_fft_multiplier(cfg.nx, cfg.ny, cfg.dx, cfg.dy, cfg.alpha, dt)
    for _ in range(1, cfg.nt):
        U *= E
        u = np.fft.ifft2(U).real.astype(np.float32, copy=False)
        yield u


# Summaries and plots


def _first_quench_index(max_temp: np.ndarray, theta: float) -> int | None:
    """Return earliest t such that max_temp[k] < theta for all k>=t, else None."""

    if max_temp.size == 0:
        return None
    tail_max = np.maximum.accumulate(max_temp[::-1])[::-1]
    hits = np.flatnonzero(tail_max < theta)
    return int(hits[0]) if hits.size else None


def _import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433

    return plt


def _write_max_temp_plot(max_temp: np.ndarray, out: Path) -> None:
    plt = _import_pyplot()
    t_norm = np.linspace(0.0, 1.0, max_temp.size, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.plot(t_norm, max_temp)
    ax.set_xlabel("t (normalized)")
    ax.set_ylabel("max temperature")
    ax.grid(True, alpha=0.3)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_quench_plot(max_temp: np.ndarray, theta_min: float, theta_max: float, theta_num: int, out: Path) -> dict:
    plt = _import_pyplot()
    thetas = np.linspace(theta_min, theta_max, theta_num, dtype=np.float32)
    tq_norm = np.empty_like(thetas, dtype=np.float32)
    quenched = np.zeros_like(thetas, dtype=bool)

    for i, th in enumerate(thetas):
        t0 = _first_quench_index(max_temp, float(th))
        if t0 is None:
            tq_norm[i] = 1.0  # beyond horizon
        else:
            quenched[i] = True
            tq_norm[i] = float(t0) / float(max(1, max_temp.size - 1))

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.plot(thetas, tq_norm)
    ax.set_xlabel("hot threshold θ")
    ax.set_ylabel("quench time t_q (normalized)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "quenched_fraction": float(quenched.mean()) if quenched.size else 0.0,
        "tq_unquenched_value": 1.0,
    }


def _write_mosaic(frames: dict[int, np.ndarray], out: Path) -> None:
    plt = _import_pyplot()
    idxs = sorted(frames.keys())
    n = len(idxs)
    if n == 0:
        return

    vmin = float(min(float(np.min(frames[i])) for i in idxs))
    vmax = float(max(float(np.max(frames[i])) for i in idxs))

    fig, axes = plt.subplots(1, n, figsize=(1.9 * n, 2.3), squeeze=False)
    axes = axes[0]
    im = None
    for ax, idx in zip(axes, idxs):
        im = ax.imshow(frames[idx].T, origin="lower", aspect="equal", vmin=vmin, vmax=vmax)
        ax.set_title(f"t[{idx}]")
        ax.set_xticks([])
        ax.set_yticks([])
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# I/O and metadata


def _np_save(path: Path, arr: np.ndarray) -> None:
    """Save .npy without pickle (safer; identical for numeric arrays)."""

    np.save(path, arr, allow_pickle=False)


def _open_memmap_for_pack(outdir: Path, nx: int, ny: int, nt: int, dtype: np.dtype, layout: LayoutType):
    if layout == "xy_t":
        shape = (nx, ny, nt)
        name = "field_xy_t.npy"
    else:
        shape = (nt, nx, ny)
        name = "field_t_xy.npy"
    path = outdir / name
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    return path, mm


def _safe_git_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        )
        return out.strip()
    except Exception:
        return None


def _write_metadata(cfg: Heat2DConfig, realized_dt: float, stable_dt_limit: float | None, runtime: dict,
    extra: dict) -> Path:
    meta = asdict(cfg)
    # ensure JSON-serializable
    for k, v in list(meta.items()):
        if isinstance(v, Path):
            meta[k] = v.as_posix()

    meta["realized_dt"] = float(realized_dt)
    meta["stable_dt_limit"] = None if stable_dt_limit is None else float(stable_dt_limit)
    meta["numpy_version"] = np.__version__
    meta["python_version"] = sys.version.split()[0]
    meta["platform"] = platform.platform()
    meta["machine"] = platform.machine()
    meta["processor"] = platform.processor()
    meta["cpu_count"] = os.cpu_count()
    meta["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    meta["git_commit"] = _safe_git_commit(Path(__file__).resolve().parent.parent)

    for k, v in runtime.items():
        meta[k] = float(v)
    meta.update(extra)

    path = cfg.outdir / "meta.json"
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return path


# Main


def main(argv: list[str] | None = None) -> int:
    p = build_arg_parser()
    args = p.parse_args(argv)

    if args.method == "fft" and args.bc != "periodic":
        p.error("--method fft requires --bc periodic.")
    if args.auto_dt and args.method != "ftcs":
        p.error("--auto-dt is only supported for --method ftcs.")
    if args.auto_dt and not (0.0 < args.safety <= 1.0):
        p.error("--safety must satisfy 0 < safety <= 1.")
    if args.theta_min >= args.theta_max:
        p.error("--theta-min must be < --theta-max.")

    dtype = _parse_dtype(args.dtype)

    cfg = Heat2DConfig(
        nx=args.nx,
        ny=args.ny,
        nt=args.nt,
        dt=args.dt,
        alpha=args.alpha,
        dx=args.dx,
        dy=args.dy,
        bc=args.bc,
        method=args.method,
        dirichlet_value=float(args.dirichlet_value),
        init=args.init,
        sigma=args.sigma,
        amplitude=args.amplitude,
        noise=args.noise,
        seed=args.seed,
        outdir=args.outdir,
        save_every=args.save_every,
        dtype=dtype.name,
        save_frames=bool(args.save_frames),
        also_pack=bool(args.also_pack),
        layout=args.layout,
        auto_dt=bool(args.auto_dt),
        target_dt=args.target_dt,
        safety=args.safety,
        plots=bool(args.plots),
        theta_min=float(args.theta_min),
        theta_max=float(args.theta_max),
        theta_num=int(args.theta_num),
    )

    cfg.outdir.mkdir(parents=True, exist_ok=True)

    stable_dt_limit = None
    realized_dt = cfg.dt
    stability_stats: dict[str, float] = {}

    if cfg.method == "ftcs":
        stable_dt_limit = _stability_limit_dt(cfg.alpha, cfg.dx, cfg.dy)
        if cfg.auto_dt:
            realized_dt = min(cfg.target_dt, cfg.safety * stable_dt_limit)
        rx = cfg.alpha * realized_dt / (cfg.dx * cfg.dx)
        ry = cfg.alpha * realized_dt / (cfg.dy * cfg.dy)
        stability_stats = {"r_x": float(rx), "r_y": float(ry), "r_x_plus_r_y": float(rx + ry)}
        if (rx + ry) > 0.5 + 1e-12:
            p.error(f"Unstable FTCS: r_x+r_y={rx+ry:.6g} > 0.5. Reduce dt or use --auto-dt.")

    t_start = time.perf_counter()

    packed_path: Path | None = None
    mm: np.memmap | None = None
    if cfg.also_pack:
        packed_path, mm = _open_memmap_for_pack(cfg.outdir, cfg.nx, cfg.ny, cfg.nt, dtype, cfg.layout)

    max_temp = np.empty((cfg.nt,), dtype=np.float32)
    mosaic_indices = {0, cfg.nt // 3, (2 * cfg.nt) // 3, cfg.nt - 1}
    mosaic_frames: dict[int, np.ndarray] = {}
    frames_written = 0

    for t, frame in enumerate(_iter_frames(cfg, realized_dt)):
        max_temp[t] = float(np.max(frame))
        if t in mosaic_indices:
            mosaic_frames[t] = frame.copy()

        frame_out = frame.astype(dtype, copy=False)

        if cfg.save_frames and (t % cfg.save_every == 0):
            _np_save(cfg.outdir / f"frame_{t:04d}.npy", frame_out)
            frames_written += 1

        if mm is not None:
            if cfg.layout == "xy_t":
                mm[..., t] = frame_out
            else:
                mm[t, ...] = frame_out

    if mm is not None:
        mm.flush()
        del mm

    t_after_sim = time.perf_counter()

    figures: list[str] = []
    plot_summary: dict[str, float] = {}
    if cfg.plots:
        max_path = cfg.outdir / "heat2d_max_temp.png"
        _write_max_temp_plot(max_temp, max_path)
        figures.append(max_path.name)

        quench_path = cfg.outdir / "heat2d_quench_vs_theta.png"
        plot_summary.update(_write_quench_plot(max_temp, cfg.theta_min, cfg.theta_max, cfg.theta_num, quench_path))
        figures.append(quench_path.name)

        mosaic_path = cfg.outdir / "heat2d_mosaic.png"
        _write_mosaic(mosaic_frames, mosaic_path)
        figures.append(mosaic_path.name)

    t_end = time.perf_counter()

    runtime = {
        "runtime_total_sec": t_end - t_start,
        "runtime_sim_io_sec": t_after_sim - t_start,
        "runtime_plots_sec": t_end - t_after_sim,
    }

    extra = {
        "frames_written": int(frames_written),
        "packed_file": str(packed_path.name) if packed_path is not None else None,
        "figures": figures,
        **stability_stats,
        **plot_summary,
    }

    meta_path = _write_metadata(cfg, realized_dt, stable_dt_limit, runtime, extra)

    msg_parts = [f"Wrote meta: {meta_path}"]
    if cfg.save_frames:
        msg_parts.append(f"frames: {frames_written} (save_every={cfg.save_every})")
    if packed_path is not None:
        msg_parts.append(f"packed: {packed_path}")
    if figures:
        msg_parts.append(f"figures: {', '.join(figures)}")
    print(" | ".join(msg_parts))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
